# Copyright (c) Microsoft. All rights reserved.

"""Ordered failover across chat clients for each model call of the function-invocation loop."""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, Generic, TypedDict, cast

from ._clients import BaseChatClient, SupportsChatGetResponse
from ._feature_stage import ExperimentalFeature, experimental
from ._middleware import ChatMiddlewareLayer, MiddlewareException
from ._sessions import is_local_history_conversation_id
from ._tools import FunctionInvocationLayer
from ._types import ChatResponse, ChatResponseUpdate, Message, ResponseStream
from .exceptions import ChatClientInvalidRequestException, ResponseInvalidatedException

if sys.version_info >= (3, 13):
    from typing import TypeVar  # pragma: no cover
else:
    from typing_extensions import TypeVar  # pragma: no cover

if TYPE_CHECKING:
    from ._compaction import CompactionStrategy, TokenizerProtocol
    from ._middleware import ChatAndFunctionMiddlewareTypes
    from ._tools import FunctionInvocationConfiguration
    from ._types import ChatOptions  # referenced by the OptionsCoT default

logger = logging.getLogger("agent_framework")

OptionsCoT = TypeVar(
    "OptionsCoT",
    bound=TypedDict,  # type: ignore[valid-type]
    default="ChatOptions[None]",
    covariant=True,
)

FailoverPredicate = Callable[[Exception], bool]
"""Decide whether a failed chat client attempt may be retried on the next chat client."""


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
    passed by an agent, runs once per model call around all attempts.

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
    ``previous_response_id``) are rejected with ``ChatClientInvalidRequestException`` before any client is called.
    Clients that store history by default (``STORES_BY_DEFAULT``) receive ``store=False`` for their attempt unless
    the request sets ``store`` explicitly. Agents using this client keep history locally, as they do for any
    client that does not store history by default.

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
            declared = getattr(candidate, "SERVICE_CONTINUATION_OPTION_NAMES", ())
            if isinstance(declared, Sequence) and not isinstance(declared, (str, bytes, bytearray)):
                continuation_option_names.update(
                    name for name in cast(Sequence[Any], declared) if isinstance(name, str)
                )
        self._continuation_option_names = tuple(sorted(continuation_option_names))
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

    def _inner_get_response(
        self,
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        **kwargs: Any,
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        if stream:
            return self._stream_with_failover(messages=messages, options=options, client_kwargs=kwargs)
        return self._get_response_with_failover(messages=messages, options=options, client_kwargs=kwargs)

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
                return await cast(
                    "Awaitable[ChatResponse]",
                    self._call_client(
                        client, messages=messages, stream=False, options=options, client_kwargs=client_kwargs
                    ),
                )
            except Exception as exception:
                if index == last_index or not self._is_failover_allowed(exception):
                    raise
                self._log_failover(index, client, exception)
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
                            client, messages=messages, stream=True, options=options, client_kwargs=client_kwargs
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
        client: SupportsChatGetResponse[Any],
        *,
        messages: Sequence[Message],
        stream: bool,
        options: Mapping[str, Any],
        client_kwargs: Mapping[str, Any],
    ) -> Awaitable[ChatResponse[Any]] | ResponseStream[ChatResponseUpdate, ChatResponse[Any]]:
        """Run one model call on ``client`` without its function-invocation loop."""
        attempt_options = dict(options)
        if getattr(client, "STORES_BY_DEFAULT", False) is True and attempt_options.get("store") is None:
            attempt_options["store"] = False
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
            messages=messages,
            stream=stream,
            options=attempt_options,
            client_kwargs=dict(client_kwargs),
        )

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
