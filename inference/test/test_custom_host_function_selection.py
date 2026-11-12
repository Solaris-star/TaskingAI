import asyncio

import pytest

from app.models import (
    ChatCompletionFunction,
    ChatCompletionModelConfiguration,
    ChatCompletionUserMessage,
    ProviderCredentials,
)
from providers.custom_host.chat_completion_function_call import CustomHostFunctionCallChatCompletionModel
from providers.custom_host.chat_completion_tool_calls import CustomHostToolCallsChatCompletionModel


@pytest.mark.parametrize(
    "model_class, selector_key, definitions_key",
    [
        (CustomHostFunctionCallChatCompletionModel, "function_call", "functions"),
        (CustomHostToolCallsChatCompletionModel, "tool_choice", "tools"),
    ],
    ids=["legacy", "modern"],
)
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "function_call", ["arxiv_search", "auto", "none", None], ids=["named", "auto", "none", "omitted"]
)
def test_custom_host_function_selection(model_class, selector_key, definitions_key, stream, function_call):
    messages = [ChatCompletionUserMessage(content="Find papers about agents.")]
    functions = [
        ChatCompletionFunction(name="arxiv_search", description="Search papers", parameters={"properties": {}})
    ]
    original_messages = [message.model_dump() for message in messages]
    original_functions = [function.model_dump() for function in functions]
    credentials = ProviderCredentials(
        credentials={
            "CUSTOM_HOST_ENDPOINT_URL": "https://example.invalid/v1/chat/completions",
            "CUSTOM_HOST_API_KEY": "test-key",
        }
    )
    url, headers, payload = asyncio.run(
        model_class().prepare_request(
            stream=stream,
            provider_model_id="test-model",
            messages=messages,
            credentials=credentials,
            configs=ChatCompletionModelConfiguration(),
            function_call=function_call,
            functions=functions,
        )
    )

    assert url == credentials.CUSTOM_HOST_ENDPOINT_URL
    assert headers["Authorization"] == "Bearer test-key"
    assert payload["model"] == "test-model"
    assert payload["stream"] is stream
    if function_call is None:
        assert selector_key not in payload
    elif function_call in ("auto", "none"):
        assert payload[selector_key] == function_call
    elif selector_key == "function_call":
        assert payload[selector_key] == {"name": function_call}
    else:
        assert payload[selector_key] == {"type": "function", "function": {"name": function_call}}

    if definitions_key == "functions":
        assert payload[definitions_key] == original_functions
        assert "tools" not in payload
        assert "tool_choice" not in payload
    else:
        assert payload[definitions_key] == [{"type": "function", "function": f} for f in original_functions]
        assert "functions" not in payload
        assert "function_call" not in payload
    assert [message.model_dump() for message in messages] == original_messages
    assert [function.model_dump() for function in functions] == original_functions
