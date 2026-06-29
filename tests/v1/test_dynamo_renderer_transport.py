import httpx
import pytest

from renderers.base import ParsedResponse, RenderedTokens
from verifiers.v1.clients.train import generate_dynamo_chat


class _Renderer:
    def render(self, messages, *, tools, add_generation_prompt):
        assert messages == [{"role": "user", "content": "original prompt"}]
        assert tools is None
        assert add_generation_prompt is True
        return RenderedTokens(token_ids=[101, 102, 103])

    def parse_response(self, token_ids, *, tools):
        assert token_ids == [104, 105]
        assert tools is None
        return ParsedResponse(content="done")


class _Client:
    def __init__(self):
        self.path = None
        self.body = None

    async def post(self, path, *, cast_to, body):
        self.path = path
        self.body = body
        return httpx.Response(
            200,
            json={
                "id": "request-1",
                "choices": [{"finish_reason": "length", "logprobs": {"content": []}}],
                "nvext": {"completion_token_ids": [104, 105]},
            },
            request=httpx.Request("POST", "http://test/v1/chat/completions"),
        )


@pytest.mark.asyncio
async def test_dynamo_chat_sends_rendered_token_ids_and_reads_completion_ids():
    client = _Client()

    result = await generate_dynamo_chat(
        client=client,
        renderer=_Renderer(),
        messages=[{"role": "user", "content": "original prompt"}],
        model="test-model",
        tools=None,
        sampling_params={"max_completion_tokens": 2},
        extra_headers=None,
    )

    assert client.path == "/chat/completions"
    assert client.body["nvext"]["token_data"] == [101, 102, 103]
    assert client.body["nvext"]["extra_fields"] == ["completion_token_ids"]
    assert result["prompt_ids"] == [101, 102, 103]
    assert result["completion_ids"] == [104, 105]
