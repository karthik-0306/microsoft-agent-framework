# Copyright (c) Microsoft. All rights reserved.

"""Ordered failover across chat clients for each model call of the function-invocation loop."""

from __future__ import annotations

import copy
import hashlib
import logging
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, Literal, TypedDict, cast, overload

from ._clients import BaseChatClient, SupportsChatGetResponse
from ._compaction import GROUP_ANNOTATION_KEY, GROUP_HAS_REASONING_KEY, GROUP_ID_KEY
from ._feature_stage import ExperimentalFeature, experimental
from ._middleware import ChatMiddlewareLayer, MiddlewareException
from ._sessions import is_local_history_conversation_id
from ._tools import FunctionInvocationLayer, ResponseModelBoundT
from ._types import ChatResponse, ChatResponseUpdate, Content, Message, ResponseStream
from .exceptions import ChatClientInvalidRequestException, ResponseInvalidatedException

if sys.version_info >= (3, 13):
    from typing import TypeVar  # pragma: no cover
else:
    from typing_extensions import TypeVar  # pragma: no cover

if TYPE_CHECKING:
    from ._compaction import CompactionStrategy, TokenizerProtocol
    from ._middleware import ChatAndFunctionMiddlewareTypes
    from ._tools import FunctionInvocationConfiguration
    from ._types import ChatOptions

logger = logging.getLogger("agent_framework")

OptionsCoT = TypeVar(
    "OptionsCoT",
    bound=TypedDict,  # type: ignore[valid-type]
    default="ChatOptions[None]",
    covariant=True,
)

FailoverPredicate = Callable[[Exception], bool]
"""Decide whether a failed chat client attempt may be retried on the next chat client."""

_FAILOVER_METADATA_KEY = "agent_framework.failover"
"""Reasoning ``Content.additional_properties`` namespace recording which candidate produced the reasoning.

It is stored on the content rather than the message because streamed messages are rebuilt from updates, which keeps
content metadata but not message metadata.
"""

_CANDIDATE_SESSION_KEY = "_agent_framework_failover_session"
"""Request-local client kwarg that carries the caller's session past this client's chat middleware layer."""


def _client_source(client: SupportsChatGetResponse[Any]) -> str:
    """Return a stable, opaque identity for the service a client calls.

    The identity combines the client type, model, and service URL, so two deployments of the same model in
    different regions are different sources. It is hashed so endpoints are not copied into persisted history.
    """
    service_url = getattr(client, "service_url", None)
    try:
        url = service_url() if callable(service_url) else None
    except Exception:
        url = None
    identity = f"{type(client).__module__}.{type(client).__qualname__}|{getattr(client, 'model', None)}|{url}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]


def _content_source(content: Content) -> str | None:
    metadata = content.additional_properties.get(_FAILOVER_METADATA_KEY)
    if isinstance(metadata, Mapping):
        source = cast(Mapping[str, Any], metadata).get("source")
        if isinstance(source, str):
            return source
    return None


def _is_foreign_reasoning(content: Content, accepted: frozenset[str]) -> bool:
    if content.type != "text_reasoning":
        return False
    source = _content_source(content)
    return source is not None and source not in accepted


def _group_annotation(message: Message) -> Mapping[str, Any] | None:
    annotation = message.additional_properties.get(GROUP_ANNOTATION_KEY)
    return cast(Mapping[str, Any], annotation) if isinstance(annotation, Mapping) else None


def _without_reasoning(message: Message, accepted: frozenset[str]) -> Message:
    """Return a copy of ``message`` without foreign reasoning contents or a group reasoning flag."""
    projected = copy.copy(message)
    projected.contents = [content for content in message.contents if not _is_foreign_reasoning(content, accepted)]
    projected.additional_properties = dict(message.additional_properties)
    annotation = _group_annotation(message)
    if annotation is not None and annotation.get(GROUP_HAS_REASONING_KEY) is True:
        projected.additional_properties[GROUP_ANNOTATION_KEY] = {**annotation, GROUP_HAS_REASONING_KEY: False}
    return projected


@experimental(feature_id=ExperimentalFeature.CHAT_CLIENT_FAILOVER)
class FailoverChatClient(
    FunctionInvocationLayer[OptionsCoT],
    ChatMiddlewareLayer[OptionsCoT],
    BaseChatClient[OptionsCoT],
    Generic[OptionsCoT],
):
    """Send each model call to an ordered list of chat clients until one succeeds.

    ``FailoverChatClient`` owns the single function-invocation loop for a request and fails over per model call,
    not per run. When a chat client fails after earlier model calls already requested tools, the next chat client
    receives the same conversation, including the tool results produced so far, so no tool runs again. Wrapping
    complete chat clients in a plain ``try``/``except`` does not have this property: each client runs its own tool
    loop, so a failure on a later model call restarts that loop on the next client and re-executes tools.

    Each attempt goes through the selected client's chat middleware and telemetry layers but not its
    function-invocation layer. Configure tools, function middleware, and ``function_invocation_configuration`` on
    the ``FailoverChatClient`` (or the agent that uses it). Chat middleware configured on the failover client, or
    passed by an agent, runs once per model call around all attempts. Each client's own chat middleware runs for
    each of its attempts and receives the caller's session.

    Failover rules:

    - Clients are tried in order for every model call. The first client is tried first again on the next model
      call, so a recovered primary resumes serving requests.
    - Any ``Exception`` raised by an attempt triggers failover unless ``should_failover`` returns ``False``.
      Cancellation is never caught. ``ResponseInvalidatedException`` and middleware control-flow exceptions always
      propagate unchanged, because handing an invalidated provider response to another client could run tool calls
      the provider already cancelled.
    - A streaming attempt fails over only before it releases an update that carries content. Updates without
      content (for example, response-created metadata) are held back until content arrives, so an attempt that fails
      after only sending metadata is discarded. Once content has been released, a later error propagates.
    - When the last client fails, its exception propagates unchanged.

    Conversation history must be client-managed because history stored by one service cannot be read by another.
    Requests that set ``store=True`` or carry a service-managed continuation (such as ``conversation_id`` or
    ``previous_response_id``) are rejected with ``ChatClientInvalidRequestException`` before the function-invocation
    loop starts, so no approved tool runs for a request that cannot be served. Clients that store history by default
    (``STORES_BY_DEFAULT``) receive ``store=False`` for their attempt unless the request sets ``store`` explicitly.
    Agents using this client keep history locally, as they do for any client that does not store history by default.

    Reasoning is provider-specific: a provider can only replay reasoning it produced itself (for example, OpenAI
    requires its encrypted reasoning payload). This client records on each reasoning content, in streamed and
    non-streamed responses, which client produced it. That record is kept in the conversation history, so it also
    applies to later runs. Reasoning produced by a different client is omitted from the copy of the conversation sent
    to a client, together with the compaction flag that marks its group as containing reasoning. Function calls and
    results are always sent, and the caller's transcript is never modified. Reasoning without a record (for example,
    history created before this client was used) is sent unchanged.

    The chat clients are caller-owned: ``close()`` and ``async with`` do not close them.

    Examples:
        .. code-block:: python

            from agent_framework import Agent, FailoverChatClient
            from agent_framework.anthropic import AnthropicClient
            from agent_framework.openai import OpenAIChatClient

            client = FailoverChatClient([
                OpenAIChatClient(model="gpt-5.4-mini"),
                AnthropicClient(model="claude-sonnet-5-5"),
            ])
            agent = Agent(client=client, instructions="You are a support assistant.", tools=[issue_refund])
            response = await agent.run("Refund order 123.")
    """

    def __init__(
        self,
        clients: Sequence[SupportsChatGetResponse[Any]],
        *,
        should_failover: FailoverPredicate | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        function_invocation_configuration: FunctionInvocationConfiguration | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        additional_properties: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a FailoverChatClient.

        Args:
            clients: The chat clients to try, in order. The sequence is copied; the clients remain caller-owned.

        Keyword Args:
            should_failover: Optional predicate called with the exception from a failed attempt. Return ``True`` to
                try the next client or ``False`` to propagate the exception. It is not called for the last client,
                for cancellation, for ``ResponseInvalidatedException``, or for middleware control-flow exceptions.
                When omitted, every other ``Exception`` triggers failover.
            middleware: Chat and function middleware for this client. Chat middleware runs once per model call
                around all attempts; function middleware runs around tool executions.
            function_invocation_configuration: Configuration for the function-invocation loop owned by this client.
            compaction_strategy: Optional compaction strategy applied before each model call, before any attempt.
            tokenizer: Optional tokenizer used by token-aware compaction strategies.
            additional_properties: Additional properties for the client.

        Raises:
            ValueError: If ``clients`` is empty or contains the same client more than once.
            TypeError: If an entry does not provide ``get_response``.
        """
        candidates = tuple(clients)
        if not candidates:
            raise ValueError("FailoverChatClient requires at least one chat client.")
        if len({id(candidate) for candidate in candidates}) != len(candidates):
            raise ValueError("FailoverChatClient requires distinct chat clients; the same client appears twice.")
        for candidate in candidates:
            if not callable(getattr(candidate, "get_response", None)):
                raise TypeError(
                    f"FailoverChatClient clients must provide get_response(); got {type(candidate).__name__}."
                )
        self._clients = candidates
        self._should_failover = should_failover

        continuation_option_names = {"conversation_id"}
        for candidate in candidates:
            # A nested failover client must contribute its own candidates' names, not the inherited default.
            declared = (
                candidate._continuation_option_names
                if isinstance(candidate, FailoverChatClient)
                else getattr(candidate, "SERVICE_CONTINUATION_OPTION_NAMES", ())
            )
            if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes, bytearray)):
                continuation_option_names.update(
                    name for name in cast(Sequence[Any], declared) if isinstance(name, str)
                )
        self._continuation_option_names = tuple(sorted(continuation_option_names))

        # Reasoning recorded with one of a candidate's sources is replayed to that candidate. A nested failover
        # client records its own candidates' sources, so it accepts all of them.
        candidate_sources: list[str | None] = []
        accepted_sources: list[frozenset[str]] = []
        for candidate in candidates:
            if isinstance(candidate, FailoverChatClient):
                candidate_sources.append(None)
                accepted_sources.append(candidate._all_sources)
            else:
                source = _client_source(candidate)
                candidate_sources.append(source)
                accepted_sources.append(frozenset({source}))
        self._candidate_sources = tuple(candidate_sources)
        self._accepted_sources = tuple(accepted_sources)
        self._all_sources: frozenset[str] = frozenset[str]().union(*accepted_sources)
        super().__init__(
            middleware=middleware,
            function_invocation_configuration=function_invocation_configuration,
            compaction_strategy=compaction_strategy,
            tokenizer=tokenizer,
            additional_properties=additional_properties,
        )

    @property
    def clients(self) -> tuple[SupportsChatGetResponse[Any], ...]:
        """The chat clients tried for each model call, in order."""
        return self._clients

    @overload
    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False] = ...,
        options: ChatOptions[ResponseModelBoundT],
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> Awaitable[ChatResponse[ResponseModelBoundT]]: ...

    @overload
    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[False] = ...,
        options: OptionsCoT | ChatOptions[None] | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> Awaitable[ChatResponse[Any]]: ...

    @overload
    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: Literal[True],
        options: OptionsCoT | ChatOptions[Any] | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> ResponseStream[ChatResponseUpdate, ChatResponse[Any]]: ...

    def get_response(
        self,
        messages: Sequence[Message],
        *,
        stream: bool = False,
        options: OptionsCoT | ChatOptions[Any] | None = None,
        middleware: Sequence[ChatAndFunctionMiddlewareTypes] | None = None,
        compaction_strategy: CompactionStrategy | None = None,
        tokenizer: TokenizerProtocol | None = None,
        function_invocation_kwargs: Mapping[str, Any] | None = None,
        client_kwargs: Mapping[str, Any] | None = None,
    ) -> Awaitable[ChatResponse[Any]] | ResponseStream[ChatResponseUpdate, ChatResponse[Any]]:
        """Get a response, failing over to the next chat client for each model call that fails.

        Args:
            messages: The messages to send.

        Keyword Args:
            stream: Whether to stream the response.
            options: Chat options forwarded to each attempted client.
            middleware: Additional chat and function middleware for this request.
            compaction_strategy: Optional per-call compaction override.
            tokenizer: Optional per-call tokenizer override.
            function_invocation_kwargs: Keyword arguments forwarded only to tool invocations.
            client_kwargs: Additional client keyword arguments forwarded to each attempted client.

        Returns:
            When ``stream`` is ``False``, an awaitable response; otherwise a response stream.

        Raises:
            ChatClientInvalidRequestException: If ``options`` set ``store=True`` or carry a service-managed
                continuation. This is raised before any approved tool runs or any client is called.
        """
        # Reject service-managed history before the inherited loop resolves approvals, which can run tools
        # without a model call. The per-model-call check in _inner_get_response still covers middleware changes.
        self._ensure_client_managed_history(cast("Mapping[str, Any] | None", options) or {})
        forwarded_client_kwargs = dict(client_kwargs) if client_kwargs is not None else {}
        if (session := forwarded_client_kwargs.get("session")) is not None:
            # This client's chat middleware layer consumes "session"; carry it to the candidates' chat layers.
            forwarded_client_kwargs[_CANDIDATE_SESSION_KEY] = session
        get_response = cast("Callable[..., Any]", super().get_response)
        return cast(
            "Awaitable[ChatResponse[Any]] | ResponseStream[ChatResponseUpdate, ChatResponse[Any]]",
            get_response(
                messages,
                stream=stream,
                options=options,
                middleware=middleware,
                compaction_strategy=compaction_strategy,
                tokenizer=tokenizer,
                function_invocation_kwargs=function_invocation_kwargs,
                client_kwargs=forwarded_client_kwargs,
            ),
        )

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        client_kwargs = dict(kwargs)
        if (session := client_kwargs.pop(_CANDIDATE_SESSION_KEY, None)) is not None:
            client_kwargs["session"] = session
        if stream:
            return self._stream_with_failover(messages=messages, options=options, client_kwargs=client_kwargs)
        return self._get_response_with_failover(messages=messages, options=options, client_kwargs=client_kwargs)

    async def _get_response_with_failover(
        self,
        *,
        messages: Sequence[Message],
        options: Mapping[str, Any],
        client_kwargs: Mapping[str, Any],
    ) -> ChatResponse:
        self._ensure_client_managed_history(options)
        last_index = len(self._clients) - 1
        for index, client in enumerate(self._clients):
            try:
                response = await cast(
                    "Awaitable[ChatResponse]",
                    self._call_client(
                        index, messages=messages, stream=False, options=options, client_kwargs=client_kwargs
                    ),
                )
            except Exception as exception:
                if index == last_index or not self._is_failover_allowed(exception):
                    raise
                self._log_failover(index, client, exception)
                continue
            for message in response.messages:
                self._record_reasoning_source(index, message.contents)
            return response
        raise AssertionError("unreachable")  # pragma: no cover

    def _stream_with_failover(
        self,
        *,
        messages: Sequence[Message],
        options: Mapping[str, Any],
        client_kwargs: Mapping[str, Any],
    ) -> ResponseStream[ChatResponseUpdate, ChatResponse]:
        response_format = options.get("response_format")
        final_responses: list[ChatResponse] = []

        async def _updates() -> AsyncIterator[ChatResponseUpdate]:
            self._ensure_client_managed_history(options)
            last_index = len(self._clients) - 1
            for index, client in enumerate(self._clients):
                is_last = index == last_index
                try:
                    attempt = await cast(
                        "ResponseStream[ChatResponseUpdate, ChatResponse]",
                        self._call_client(
                            index, messages=messages, stream=True, options=options, client_kwargs=client_kwargs
                        ),
                    )
                except Exception as exception:
                    if is_last or not self._is_failover_allowed(exception):
                        raise
                    self._log_failover(index, client, exception)
                    continue

                held: list[ChatResponseUpdate] = []
                released = False
                failed = False
                async with attempt:
                    iterator = attempt.__aiter__()
                    while True:
                        try:
                            update = await iterator.__anext__()
                        except StopAsyncIteration:
                            break
                        except Exception as exception:
                            if released or is_last or not self._is_failover_allowed(exception):
                                raise
                            self._log_failover(index, client, exception)
                            failed = True
                            break
                        # Record on the streamed contents too: callers rebuild streamed history from updates.
                        self._record_reasoning_source(index, update.contents)
                        if not released and not update.contents:
                            held.append(update)
                            continue
                        if not released:
                            released = True
                            for held_update in held:
                                yield held_update
                            held.clear()
                        yield update
                    if failed:
                        continue
                    # The attempt finalizes when its iteration ends, so a finalization error has already been
                    # handled above; this returns the finalized response.
                    final_response = await attempt.get_final_response()
                for held_update in held:
                    yield held_update
                for message in final_response.messages:
                    self._record_reasoning_source(index, message.contents)
                final_responses.append(final_response)
                return

        def _finalize(updates: Sequence[ChatResponseUpdate]) -> ChatResponse:
            if final_responses:
                return final_responses[-1]
            # Reached only when a direct consumer of this stream stops before an attempt completes.
            return ChatResponse.from_updates(updates, output_format_type=response_format)

        return ResponseStream(_updates(), finalizer=_finalize)

    def _call_client(
        self,
        index: int,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        client_kwargs: Mapping[str, Any],
    ) -> Awaitable[ChatResponse[Any]] | ResponseStream[ChatResponseUpdate, ChatResponse[Any]]:
        """Run one model call on the client at ``index`` without its function-invocation loop."""
        client = self._clients[index]
        attempt_options = dict(options)
        if getattr(client, "STORES_BY_DEFAULT", False) is True and attempt_options.get("store") is None:
            attempt_options["store"] = False
        attempt_client_kwargs = dict(client_kwargs)
        if isinstance(client, FailoverChatClient) and (session := attempt_client_kwargs.get("session")) is not None:
            # A nested failover client is entered below its own get_response, so hand it the session the same way.
            attempt_client_kwargs[_CANDIDATE_SESSION_KEY] = session
        target: Any = client
        if isinstance(client, FunctionInvocationLayer):
            # This is the same call a client's own function-invocation loop makes for each model call: it enters the
            # layer below FunctionInvocationLayer, keeping the client's chat middleware and telemetry.
            target = super(FunctionInvocationLayer, client)
        get_response = cast(
            "Callable[..., Awaitable[ChatResponse[Any]] | ResponseStream[ChatResponseUpdate, ChatResponse[Any]]]",
            target.get_response,
        )
        return get_response(
            messages=self._messages_for_client(index, messages),
            stream=stream,
            options=attempt_options,
            client_kwargs=attempt_client_kwargs,
        )

    def _record_reasoning_source(self, index: int, contents: Sequence[Content]) -> None:
        """Record which client produced each reasoning content, keeping a nested client's record."""
        source = self._candidate_sources[index]
        if source is None:
            return
        for content in contents:
            if content.type == "text_reasoning":
                content.additional_properties.setdefault(_FAILOVER_METADATA_KEY, {"source": source})

    def _messages_for_client(self, index: int, messages: Sequence[Message]) -> Sequence[Message]:
        """Return the conversation to send to the client at ``index``.

        Reasoning recorded as produced by a different client is omitted, along with the compaction flag that marks
        its group as containing reasoning. The caller's messages are not modified; unchanged conversations are
        returned as-is.
        """
        accepted = self._accepted_sources[index]
        projected: list[Message] | None = None
        foreign_groups: set[Any] = set()
        for position, message in enumerate(messages):
            if not any(_is_foreign_reasoning(content, accepted) for content in message.contents):
                continue
            if projected is None:
                projected = list(messages)
            annotation = _group_annotation(message)
            if annotation is not None and annotation.get(GROUP_ID_KEY) is not None:
                foreign_groups.add(annotation[GROUP_ID_KEY])
            projected[position] = _without_reasoning(message, accepted)
        if projected is None:
            return messages
        if foreign_groups:
            for position, message in enumerate(projected):
                annotation = _group_annotation(message)
                if (
                    annotation is not None
                    and annotation.get(GROUP_ID_KEY) in foreign_groups
                    and annotation.get(GROUP_HAS_REASONING_KEY) is True
                ):
                    projected[position] = _without_reasoning(message, accepted)
        return [message for message in projected if message.contents]

    def _ensure_client_managed_history(self, options: Mapping[str, Any]) -> None:
        if options.get("store") is True:
            raise ChatClientInvalidRequestException(
                "FailoverChatClient requires client-managed history so another chat client can continue the "
                "conversation; store=True is not supported. Leave store unset or set store=False."
            )
        for name in self._continuation_option_names:
            value = options.get(name)
            if value is None or value == "":
                continue
            if name == "conversation_id" and isinstance(value, str) and is_local_history_conversation_id(value):
                continue
            raise ChatClientInvalidRequestException(
                f"FailoverChatClient cannot continue a service-managed conversation ('{name}' is set) because the "
                "other chat clients cannot read history stored by that service. Use client-managed history instead."
            )

    def _is_failover_allowed(self, exception: Exception) -> bool:
        if isinstance(exception, (ResponseInvalidatedException, MiddlewareException)):
            return False
        if self._should_failover is None:
            return True
        return bool(self._should_failover(exception))

    def _log_failover(self, index: int, client: SupportsChatGetResponse[Any], exception: Exception) -> None:
        logger.warning(
            "Chat client %d of %d (%s) failed with %s; trying the next chat client.",
            index + 1,
            len(self._clients),
            type(client).__name__,
            type(exception).__name__,
            exc_info=exception,
        )
