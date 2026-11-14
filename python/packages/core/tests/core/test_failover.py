# Copyright (c) Microsoft. All rights reserved.

import asyncio
from collections.abc import Awaitable, Mapping, Sequence
from typing import Any, ClassVar, cast

import pytest

from agent_framework import (
    Agent,
    BaseChatClient,
    ChatContext,
    ChatMiddleware,
    ChatMiddlewareLayer,
    ChatOptions,
    ChatResponse,
    ChatResponseUpdate,
    Content,
    FailoverChatClient,
    FunctionInvocationContext,
    FunctionInvocationLayer,
    FunctionMiddleware,
    Message,
    ResponseStream,
    tool,
)
from agent_framework._middleware import MiddlewareException
from agent_framework.exceptions import (
    ChatClientException,
    ChatClientInvalidRequestException,
    ResponseInvalidatedException,
)

pytestmark = pytest.mark.filterwarnings("ignore:.*CHAT_CLIENT_FAILOVER.*:FutureWarning")

# A step is one scripted model call: a response, an exception, or (streaming) a list of updates and exceptions.
Step = ChatResponse | BaseException | Sequence[ChatResponseUpdate | BaseException]


class _RawScriptedClient(BaseChatClient[Any]):
    """Chat client without middleware or a tool loop (like ``Raw*`` provider clients) and scripted model calls."""

    def __init__(
        self,
        steps: Sequence[Step] = (),
        *,
        raise_on_call: BaseException | None = None,
        finalize_error: Exception | None = None,
    ) -> None:
        super().__init__()
        self._steps = list(steps)
        self._raise_on_call = raise_on_call
        self._finalize_error = finalize_error
        self.calls: list[dict[str, Any]] = []
        self.closed = False

    async def close(self) -> None:
        self.closed = True

    def _inner_get_response(
        self, *, messages: Sequence[Message], stream: bool, options: Mapping[str, Any], **kwargs: Any
    ) -> Awaitable[ChatResponse] | ResponseStream[ChatResponseUpdate, ChatResponse]:
        self.calls.append({"messages": list(messages), "options": dict(options)})
        if self._raise_on_call is not None:
            raise self._raise_on_call
        step: Step = self._steps.pop(0) if self._steps else _text_response("done")

        if stream:
            items = _updates_for(step) if isinstance(step, ChatResponse) else step

            async def _updates():
                if isinstance(items, BaseException):
                    raise items
                for item in items:
                    if isinstance(item, BaseException):
                        raise item
                    yield item

            if self._finalize_error is None:
                return self._build_response_stream(_updates())
            finalize_error = self._finalize_error

            def _failing_finalizer(_: Sequence[ChatResponseUpdate]) -> ChatResponse:
                raise finalize_error

            return ResponseStream(_updates(), finalizer=_failing_finalizer)

        async def _response() -> ChatResponse:
            if isinstance(step, BaseException):
                raise step
            assert isinstance(step, ChatResponse)
            return step

        return _response()


class _ScriptedClient(FunctionInvocationLayer[Any], ChatMiddlewareLayer[Any], _RawScriptedClient):
    """Chat client with the same layering as shipped clients and scripted model calls."""

    def __init__(
        self,
        steps: Sequence[Step] = (),
        *,
        raise_on_call: BaseException | None = None,
        finalize_error: Exception | None = None,
        middleware: Sequence[Any] | None = None,
    ) -> None:
        super().__init__(middleware=middleware, steps=steps, raise_on_call=raise_on_call, finalize_error=finalize_error)


class _StoringScriptedClient(_ScriptedClient):
    STORES_BY_DEFAULT: ClassVar[bool] = True


class _ResponsesLikeScriptedClient(_StoringScriptedClient):
    SERVICE_CONTINUATION_OPTION_NAMES: ClassVar[tuple[str, ...]] = ("conversation_id", "previous_response_id")


def _text_response(text: str) -> ChatResponse:
    return ChatResponse(messages=Message(role="assistant", contents=[text]))


def _tool_call_response(call_id: str = "call-1") -> ChatResponse:
    call = Content.from_function_call(call_id=call_id, name="send_email", arguments={"to": "cfo@contoso.com"})
    return ChatResponse(messages=Message(role="assistant", contents=[call]))


def _updates_for(response: ChatResponse) -> list[ChatResponseUpdate]:
    return [ChatResponseUpdate(contents=list(message.contents), role="assistant") for message in response.messages]


def _rate_limited(name: str) -> ChatClientException:
    return ChatClientException(f"{name}: 429 Too Many Requests", log_level=None)


def _function_results(messages: Sequence[Message]) -> list[Content]:
    return [content for message in messages for content in message.contents if content.type == "function_result"]


@pytest.fixture
def sent_emails() -> list[str]:
    return []


@pytest.fixture
def send_email(sent_emails: list[str]) -> Any:
    @tool(approval_mode="never_require")
    def send_email(to: str) -> str:
        """Send an email."""
        sent_emails.append(to)
        return f"sent to {to}"

    return send_email


async def test_failover_after_tool_call_does_not_run_the_tool_again(send_email: Any, sent_emails: list[str]) -> None:
    primary = _ScriptedClient([_tool_call_response(), _rate_limited("primary")])
    fallback = _ScriptedClient([_text_response("done via fallback")])
    client = FailoverChatClient([primary, fallback])

    response = await client.get_response(
        [Message(role="user", contents=["Email the CFO."])], options={"tools": [send_email]}
    )

    assert response.text == "done via fallback"
    assert sent_emails == ["cfo@contoso.com"]
    assert len(primary.calls) == 2
    assert len(fallback.calls) == 1
    # The fallback continues the same model call, so it sees the tool result produced before the failure.
    assert [result.call_id for result in _function_results(fallback.calls[0]["messages"])] == ["call-1"]


async def test_streaming_failover_after_tool_call_does_not_run_the_tool_again(
    send_email: Any, sent_emails: list[str]
) -> None:
    primary = _ScriptedClient([_updates_for(_tool_call_response()), _rate_limited("primary")])
    fallback = _ScriptedClient([_updates_for(_text_response("done via fallback"))])
    client = FailoverChatClient([primary, fallback])

    stream = client.get_response(
        [Message(role="user", contents=["Email the CFO."])], stream=True, options={"tools": [send_email]}
    )
    async for _ in stream:
        pass
    response = await stream.get_final_response()

    assert response.text == "done via fallback"
    assert sent_emails == ["cfo@contoso.com"]
    assert len(fallback.calls) == 1
    assert [result.call_id for result in _function_results(fallback.calls[0]["messages"])] == ["call-1"]


async def test_each_model_call_starts_with_the_first_client(send_email: Any, sent_emails: list[str]) -> None:
    primary = _ScriptedClient([_rate_limited("primary"), _text_response("primary recovered")])
    fallback = _ScriptedClient([_tool_call_response()])
    client = FailoverChatClient([primary, fallback])

    response = await client.get_response(
        [Message(role="user", contents=["Email the CFO."])], options={"tools": [send_email]}
    )

    assert response.text == "primary recovered"
    assert sent_emails == ["cfo@contoso.com"]
    assert len(primary.calls) == 2
    assert len(fallback.calls) == 1


async def test_successful_first_client_does_not_call_the_others() -> None:
    primary = _ScriptedClient([_text_response("primary")])
    fallback = _ScriptedClient()
    client = FailoverChatClient([primary, fallback])

    response = await client.get_response([Message(role="user", contents=["hi"])])

    assert response.text == "primary"
    assert fallback.calls == []


async def test_last_client_error_propagates_unchanged() -> None:
    last_error = _rate_limited("fallback")
    primary = _ScriptedClient([_rate_limited("primary")])
    fallback = _ScriptedClient([last_error])
    client = FailoverChatClient([primary, fallback])

    with pytest.raises(ChatClientException) as raised:
        await client.get_response([Message(role="user", contents=["hi"])])

    assert raised.value is last_error
    assert len(primary.calls) == 1
    assert len(fallback.calls) == 1


async def test_should_failover_can_stop_failover() -> None:
    first_error = ValueError("bad request")
    seen: list[Exception] = []

    def should_failover(exception: Exception) -> bool:
        seen.append(exception)
        return False

    primary = _ScriptedClient([first_error])
    fallback = _ScriptedClient()
    client = FailoverChatClient([primary, fallback], should_failover=should_failover)

    with pytest.raises(ValueError) as raised:
        await client.get_response([Message(role="user", contents=["hi"])])

    assert raised.value is first_error
    assert seen == [first_error]
    assert fallback.calls == []


@pytest.mark.parametrize(
    "error",
    [
        ResponseInvalidatedException("provider invalidated the response", log_level=None),
        MiddlewareException("middleware stopped the call", log_level=None),
    ],
    ids=["response-invalidated", "middleware-control-flow"],
)
@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_protected_errors_never_fail_over(error: Exception, stream: bool) -> None:
    primary = _ScriptedClient([error])
    fallback = _ScriptedClient()
    client = FailoverChatClient([primary, fallback], should_failover=lambda _: True)

    with pytest.raises(type(error)) as raised:
        if stream:
            async for _ in client.get_response([Message(role="user", contents=["hi"])], stream=True):
                pass
        else:
            await client.get_response([Message(role="user", contents=["hi"])])

    assert raised.value is error
    assert fallback.calls == []


async def test_cancellation_is_not_failed_over() -> None:
    primary = _ScriptedClient([asyncio.CancelledError()])
    fallback = _ScriptedClient()
    client = FailoverChatClient([primary, fallback])

    with pytest.raises(asyncio.CancelledError):
        await client.get_response([Message(role="user", contents=["hi"])])

    assert fallback.calls == []


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_error_raised_when_starting_the_call_fails_over(stream: bool) -> None:
    primary = _ScriptedClient(raise_on_call=_rate_limited("primary"))
    fallback = _ScriptedClient([_text_response("fallback")])
    client = FailoverChatClient([primary, fallback])

    if stream:
        response_stream = client.get_response([Message(role="user", contents=["hi"])], stream=True)
        async for _ in response_stream:
            pass
        response = await response_stream.get_final_response()
    else:
        response = await client.get_response([Message(role="user", contents=["hi"])])

    assert response.text == "fallback"


async def test_streaming_error_after_content_is_not_failed_over() -> None:
    error = _rate_limited("primary")
    primary = _ScriptedClient([
        [ChatResponseUpdate(contents=[Content.from_text("Your refund")], role="assistant"), error]
    ])
    fallback = _ScriptedClient()
    client = FailoverChatClient([primary, fallback])

    received: list[str] = []
    with pytest.raises(ChatClientException) as raised:
        async for update in client.get_response([Message(role="user", contents=["hi"])], stream=True):
            received.append(update.text)

    assert raised.value is error
    assert received == ["Your refund"]
    assert fallback.calls == []


async def test_streaming_metadata_from_a_failed_attempt_is_discarded() -> None:
    primary = _ScriptedClient([
        [ChatResponseUpdate(response_id="primary-response", role="assistant"), _rate_limited("p")]
    ])
    fallback = _ScriptedClient([
        [
            ChatResponseUpdate(response_id="fallback-response", role="assistant"),
            ChatResponseUpdate(contents=[Content.from_text("done")], response_id="fallback-response", role="assistant"),
        ]
    ])
    client = FailoverChatClient([primary, fallback])

    stream = client.get_response([Message(role="user", contents=["hi"])], stream=True)
    updates = [update async for update in stream]
    response = await stream.get_final_response()

    assert {update.response_id for update in updates} == {"fallback-response"}
    assert response.text == "done"
    assert response.response_id == "fallback-response"


async def test_streaming_attempt_without_content_updates_succeeds() -> None:
    primary = _ScriptedClient([[ChatResponseUpdate(response_id="primary-response", role="assistant")]])
    client = FailoverChatClient([primary, _ScriptedClient()])

    stream = client.get_response([Message(role="user", contents=["hi"])], stream=True)
    updates = [update async for update in stream]

    assert [update.response_id for update in updates] == ["primary-response"]
    assert (await stream.get_final_response()).response_id == "primary-response"


@pytest.mark.parametrize(
    "options",
    [
        {"store": True},
        {"conversation_id": "conv_123"},
        {"previous_response_id": "resp_123"},
    ],
    ids=["store-true", "service-conversation", "candidate-declared-continuation"],
)
async def test_service_managed_history_is_rejected_before_any_call(options: dict[str, Any]) -> None:
    primary = _ResponsesLikeScriptedClient()
    fallback = _ScriptedClient()
    client = FailoverChatClient([primary, fallback])

    with pytest.raises(ChatClientInvalidRequestException, match="client-managed history|service-managed"):
        await client.get_response([Message(role="user", contents=["hi"])], options=cast("ChatOptions[Any]", options))

    assert primary.calls == []
    assert fallback.calls == []


async def test_clients_that_store_by_default_receive_store_false() -> None:
    primary = _StoringScriptedClient([_rate_limited("primary")])
    fallback = _ScriptedClient([_text_response("fallback")])
    client = FailoverChatClient([primary, fallback])

    await client.get_response([Message(role="user", contents=["hi"])], options={"temperature": 0.2})

    assert primary.calls[0]["options"]["store"] is False
    assert "store" not in fallback.calls[0]["options"]
    assert fallback.calls[0]["options"]["temperature"] == 0.2


async def test_attempts_use_client_chat_middleware_but_not_client_tool_loop(
    send_email: Any, sent_emails: list[str]
) -> None:
    attempts: list[str] = []
    client_tool_calls: list[str] = []
    failover_tool_calls: list[str] = []

    class _RecordAttempt(ChatMiddleware):
        def __init__(self, name: str) -> None:
            self.name = name

        async def process(self, context: ChatContext, call_next: Any) -> None:
            attempts.append(self.name)
            await call_next()

    class _RecordToolCall(FunctionMiddleware):
        def __init__(self, calls: list[str]) -> None:
            self.calls = calls

        async def process(self, context: FunctionInvocationContext, call_next: Any) -> None:
            self.calls.append(context.function.name)
            await call_next()

    primary = _ScriptedClient(
        [_tool_call_response(), _rate_limited("primary")],
        middleware=[_RecordAttempt("primary"), _RecordToolCall(client_tool_calls)],
    )
    fallback = _ScriptedClient([_text_response("done")], middleware=[_RecordAttempt("fallback")])
    client = FailoverChatClient(
        [primary, fallback], middleware=[_RecordAttempt("failover"), _RecordToolCall(failover_tool_calls)]
    )

    await client.get_response([Message(role="user", contents=["Email the CFO."])], options={"tools": [send_email]})

    # Failover-level chat middleware runs once per model call, around every attempt of that call.
    assert attempts == ["failover", "primary", "failover", "primary", "fallback"]
    assert client_tool_calls == []
    assert failover_tool_calls == ["send_email"]
    assert sent_emails == ["cfo@contoso.com"]


async def test_agent_keeps_history_client_side_across_failover(send_email: Any, sent_emails: list[str]) -> None:
    primary = _StoringScriptedClient([_tool_call_response(), _rate_limited("primary"), _text_response("second")])
    fallback = _ScriptedClient([_text_response("first")])
    agent = Agent(client=FailoverChatClient([primary, fallback]), tools=[send_email])
    session = agent.create_session()

    first = await agent.run("Email the CFO.", session=session)
    second = await agent.run("Thanks!", session=session)

    assert first.text == "first"
    assert second.text == "second"
    assert sent_emails == ["cfo@contoso.com"]
    # The second run reaches the primary with the whole conversation, including the turn the fallback answered.
    second_run_texts = [message.text for message in primary.calls[-1]["messages"]]
    assert "Email the CFO." in second_run_texts
    assert "first" in second_run_texts
    assert "Thanks!" in second_run_texts
    assert all(call["options"]["store"] is False for call in primary.calls)


async def test_nested_failover_client_is_a_valid_candidate(send_email: Any, sent_emails: list[str]) -> None:
    regional = FailoverChatClient([
        _ScriptedClient([_tool_call_response(), _rate_limited("east")]),
        _ScriptedClient([_rate_limited("west")]),
    ])
    other_provider = _ScriptedClient([_text_response("other provider")])
    client = FailoverChatClient([regional, other_provider])

    response = await client.get_response(
        [Message(role="user", contents=["Email the CFO."])], options={"tools": [send_email]}
    )

    assert response.text == "other provider"
    assert sent_emails == ["cfo@contoso.com"]


@pytest.mark.parametrize("stream", [False, True], ids=["non-streaming", "streaming"])
async def test_clients_without_a_tool_loop_are_supported(stream: bool, send_email: Any, sent_emails: list[str]) -> None:
    primary = _RawScriptedClient([_tool_call_response(), _rate_limited("primary")])
    fallback = _RawScriptedClient([_text_response("done via fallback")])
    client = FailoverChatClient([primary, fallback])
    messages = [Message(role="user", contents=["Email the CFO."])]

    if stream:
        response_stream = client.get_response(messages, stream=True, options={"tools": [send_email]})
        async for _ in response_stream:
            pass
        response = await response_stream.get_final_response()
    else:
        response = await client.get_response(messages, options={"tools": [send_email]})

    assert response.text == "done via fallback"
    assert sent_emails == ["cfo@contoso.com"]


async def test_streaming_finalization_error_before_content_fails_over() -> None:
    primary = _ScriptedClient(
        [[ChatResponseUpdate(response_id="primary-response", role="assistant")]],
        finalize_error=_rate_limited("primary"),
    )
    fallback = _ScriptedClient([_text_response("fallback")])
    client = FailoverChatClient([primary, fallback])

    stream = client.get_response([Message(role="user", contents=["hi"])], stream=True)
    updates = [update async for update in stream]

    assert {update.response_id for update in updates} == {None}
    assert (await stream.get_final_response()).text == "fallback"


async def test_streaming_error_opening_the_last_client_propagates() -> None:
    last_error = _rate_limited("fallback")
    client = FailoverChatClient([
        _ScriptedClient(raise_on_call=_rate_limited("primary")),
        _ScriptedClient(raise_on_call=last_error),
    ])

    with pytest.raises(ChatClientException) as raised:
        async for _ in client.get_response([Message(role="user", contents=["hi"])], stream=True):
            pass

    assert raised.value is last_error


async def test_streaming_consumer_can_stop_early() -> None:
    primary = _ScriptedClient([
        [
            ChatResponseUpdate(contents=[Content.from_text("first")], role="assistant"),
            ChatResponseUpdate(contents=[Content.from_text(" second")], role="assistant"),
        ]
    ])
    client = FailoverChatClient([primary, _ScriptedClient()])

    stream = client.get_response([Message(role="user", contents=["hi"])], stream=True)
    async for _ in stream:
        break
    await stream.close()

    assert (await stream.get_final_response()).text == "first"


async def test_local_history_conversation_id_is_not_a_service_continuation() -> None:
    from agent_framework._sessions import LOCAL_HISTORY_CONVERSATION_ID

    client = FailoverChatClient([_ScriptedClient([_text_response("ok")])])

    response = await client.get_response(
        [Message(role="user", contents=["hi"])], options={"conversation_id": LOCAL_HISTORY_CONVERSATION_ID}
    )

    assert response.text == "ok"


async def test_close_leaves_clients_open() -> None:
    primary = _ScriptedClient()
    fallback = _ScriptedClient()

    async with FailoverChatClient([primary, fallback]) as client:
        assert client.clients == (primary, fallback)

    assert not primary.closed
    assert not fallback.closed


def test_constructor_validates_clients() -> None:
    primary = _ScriptedClient()

    with pytest.raises(ValueError, match="at least one"):
        FailoverChatClient([])
    with pytest.raises(ValueError, match="distinct"):
        FailoverChatClient([primary, primary])
    not_a_client: Any = object()
    with pytest.raises(TypeError, match="get_response"):
        FailoverChatClient([primary, not_a_client])


def test_failover_client_is_experimental() -> None:
    from agent_framework import ExperimentalFeature

    assert getattr(FailoverChatClient, "__feature_id__", None) == ExperimentalFeature.CHAT_CLIENT_FAILOVER.value
