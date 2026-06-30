"""Renderer client: client-side tokenization via the `renderers` package.

A drop-in alternative to the chat-completions client: instead of sending messages
as JSON text, it renders them to token ids with a HF chat template and calls a
vLLM `/inference/v1/generate` engine, so every response carries token ids +
sampling logprobs (recorded on the trace's per-turn `tokens`) for training. It
reuses the chat client's wire translation (message/tool shapes are the same), and
needs a running vLLM engine.
"""

import json
from collections.abc import Mapping
from typing import Any

import httpx
from openai import AsyncOpenAI, OpenAIError
from renderers import OverlongPromptError as RendererOverlongPromptError
from renderers import RenderedTokens, RendererConfig

from verifiers.v1.clients.client import SESSION_ID_HEADER, Client
from verifiers.v1.dialects import FINISH_REASONS, ChatDialect, Dialect, parse_tools
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.errors import OverlongPromptError, model_error
from verifiers.v1.graph import PendingTurn
from verifiers.v1.types import (
    AssistantMessage,
    FinishReason,
    Response,
    SamplingConfig,
    Tool,
    ToolCall,
    TurnTokens,
    Usage,
)


def tool_to_wire(tool: Tool) -> dict:
    """A vf tool -> the OpenAI chat wire dict (the renderer's generate request)."""
    function: dict = {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.parameters,
    }
    if tool.strict is not None:
        function["strict"] = tool.strict
    return {"type": "function", "function": function}


def serialize_completion(response: Response, model: str) -> dict:
    """A vf `Response` -> an OpenAI chat.completion dict the program's SDK expects. The renderer
    sets this on `Response.raw` (it generates, so has no provider response to relay)."""
    message: dict = {"role": "assistant", "content": response.message.content}
    if response.message.reasoning_content is not None:
        message["reasoning_content"] = response.message.reasoning_content
    if response.message.tool_calls:
        message["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": c.arguments},
            }
            for c in response.message.tool_calls
        ]
    usage: dict | None = None
    if response.usage:
        # Usage is validated earlier in the pipeline; building its wire dict directly saves time.
        usage = {
            "completion_tokens": response.usage.completion_tokens,
            "prompt_tokens": response.usage.input_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        if response.usage.reasoning_tokens is not None:
            usage["completion_tokens_details"] = {
                "reasoning_tokens": response.usage.reasoning_tokens
            }
        if response.usage.cached_input_tokens is not None:
            usage["prompt_tokens_details"] = {
                "cached_tokens": response.usage.cached_input_tokens
            }
    return {
        "id": response.id or "vf-intercept",
        "object": "chat.completion",
        "created": response.created,
        "model": response.model or model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": response.finish_reason or "stop",
            }
        ],
        "usage": usage,
    }


def response_from_generate(
    result: dict, model: str, bridged_turn: PendingTurn | None = None
) -> Response:
    """Parse a `renderers.client.generate` result dict into a typed `Response`,
    mirroring the chat client's `response_from_wire` (plus the token encoding)."""
    finish: FinishReason = (
        result["finish_reason"]
        if result.get("finish_reason") in FINISH_REASONS
        else None
    )


    tool_calls = [
        ToolCall(
            id=tc.id or f"call_{i}",
            name=tc.name,
            arguments=tc.arguments
            if isinstance(tc.arguments, str)
            else json.dumps(tc.arguments or {}),
        )
        for i, tc in enumerate(result.get("tool_calls") or [])
        if getattr(tc, "name", None)
    ] or None
    prompt_ids = result.get("prompt_ids") or []
    completion_ids = result.get("completion_ids") or []
    # Per-message token spans (the renderer's attribution) let the trace graph store each
    # message's tokens once; carried transiently on TurnTokens and consumed by turn.commit().
    attribution = result.get("prompt_attribution")
    if attribution is None:
        message_spans = None
    elif bridged_turn is not None:
        message_spans = bridged_turn.prompt_message_spans(attribution)
    else:
        message_spans = attribution.message_token_spans()
    return Response(
        id=result.get("request_id", ""),
        created=0,
        model=model,
        message=AssistantMessage(
            content=result.get("content") or None,
            reasoning_content=result.get("reasoning_content"),
            tool_calls=tool_calls,
        ),
        finish_reason=finish,
        # /inference/v1/generate returns exact token ids but no usage details, so the
        # completion's reasoning-token subset is unknown.
        usage=Usage(
            prompt_tokens=len(prompt_ids), completion_tokens=len(completion_ids)
        ),
        # generate() returns owned, typed lists. Skip revalidation here to avoid copying
        # million-token contexts synchronously on the event loop.
        tokens=TurnTokens.model_construct(
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            completion_logprobs=result.get("completion_logprobs") or [],
            message_spans=message_spans,
            is_content=attribution.is_content if attribution is not None else None,
            multi_modal_data=result.get("multi_modal_data"),
            routed_experts=result.get("routed_experts"),
        ),
    )


async def generate_dynamo_chat(
    *,
    client: AsyncOpenAI,
    renderer,
    messages: list[dict],
    model: str,
    tools: list[dict] | None,
    sampling_params: dict[str, Any],
    extra_headers: dict[str, str] | None,
) -> dict[str, Any]:
    """Render locally and use Dynamo's tokens-in/tokens-out chat transport.

    Dynamo's current OpenAI endpoint does not accept the legacy
    ``return_token_ids`` request field. It accepts the locally rendered prompt
    through ``nvext.token_data`` and exposes generated IDs through
    ``nvext.completion_token_ids`` when explicitly requested instead.
    """
    from renderers.client import _maybe_offload

    rendered = await _maybe_offload(
        renderer, lambda: renderer.render(messages, tools=tools, add_generation_prompt=True)
    )
    params = dict(sampling_params)
    extra_body = params.pop("extra_body", {})
    if not isinstance(extra_body, dict):
        extra_body = {}
    params.pop("return_token_ids", None)
    extra_body.pop("return_token_ids", None)
    nvext = dict(extra_body.pop("nvext", {}))
    extra_fields = list(nvext.get("extra_fields") or [])
    if "completion_token_ids" not in extra_fields:
        extra_fields.append("completion_token_ids")
    nvext["extra_fields"] = extra_fields
    # token_data is authoritative: Dynamo bypasses server-side chat rendering
    # and forwards this immutable token sequence to its KV-aware router.
    nvext["token_data"] = list(rendered.token_ids)

    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        **params,
        **extra_body,
        "nvext": nvext,
        "logprobs": True,
        "top_logprobs": 1,
    }
    if tools:
        body["tools"] = tools
    post_kwargs: dict[str, Any] = {"cast_to": httpx.Response, "body": body}
    if extra_headers:
        post_kwargs["options"] = {"headers": extra_headers}
    raw_response = await client.post("/chat/completions", **post_kwargs)
    raw_response.raise_for_status()
    data = raw_response.json()
    choice = (data.get("choices") or [{}])[0]
    completion_ids = (data.get("nvext") or {}).get("completion_token_ids") or []
    if not completion_ids:
        raise ValueError("Dynamo response omitted nvext.completion_token_ids")
    parsed = await _maybe_offload(renderer, lambda: renderer.parse_response(completion_ids, tools=tools))
    raw_logprobs = choice.get("logprobs") or {}
    content_logprobs = raw_logprobs.get("content") if isinstance(raw_logprobs, dict) else None
    return {
        "request_id": data.get("id") or "",
        "prompt_ids": list(rendered.token_ids),
        "completion_ids": list(completion_ids),
        "completion_logprobs": [float(item.get("logprob") or 0.0) for item in content_logprobs or []],
        "content": parsed.content,
        "reasoning_content": parsed.reasoning_content,
        "tool_calls": parsed.tool_calls,
        "finish_reason": choice.get("finish_reason"),
        "routed_experts": None,
        "multi_modal_data": rendered.multi_modal_data,
        "prompt_attribution": rendered,
    }


def _is_valid_incremental_tail(messages: list[dict[str, Any]]) -> bool:
    """Renderer bridges may extend sampled assistant turns with tool calls and/or a new user."""
    if not messages:
        return False
    roles = []
    for message in messages:
        role = message.get("role")
        roles.append(role if isinstance(role, str) else None)
    if roles[-1] == "user":
        return all(role == "tool" for role in roles[:-1])
    return all(role == "tool" for role in roles)


def _has_multimodal_content(messages) -> bool:
    for message in messages:
        content = getattr(message, "content", None)
        if not isinstance(content, list):
            continue
        if any(getattr(part, "type", None) == "image_url" for part in content):
            return True
    return False


class TrainClient(Client):
    """Renders prompts to token ids and calls a vLLM `/inference/v1/generate` engine."""

    def __init__(
        self,
        openai: AsyncOpenAI,
        pool_size: int = 1,
        config: RendererConfig | None = None,
        renderer_model_name: str | None = None,
        renderer_transport: str = "vllm_generate",
    ) -> None:
        self.openai = openai
        self.pool_size = pool_size
        self.config = config
        self.renderer_model_name = renderer_model_name
        self.renderer_transport = renderer_transport
        self._pool = None

    def _renderer_pool(self, model: str):
        if self._pool is None:
            from renderers import create_renderer_pool

            self._pool = create_renderer_pool(
                self.renderer_model_name or model, self.config, size=self.pool_size
            )
        return self._pool

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        # The renderer tokenizes the typed prompt for training (it needs per-token ids + logprobs
        # back), so it can't forward the raw request — it parses `body` via the dialect and renders
        # it with a chat template. It leaves `Response.raw` unset; the interception server serializes
        # its `Response` for the program instead of relaying provider bytes.
        if not isinstance(dialect, ChatDialect):
            # The renderer renders a chat template, so it's only validated for chat-completions
            # input; other dialects' semantics (Responses reasoning items, Anthropic thinking) may
            # not round-trip faithfully through chat-template tokenization. Refuse them explicitly.
            raise NotImplementedError(
                f"The renderer client only supports the chat-completions dialect, got "
                f"{type(dialect).__name__}. Use the proxy client for this dialect, or add "
                f"renderer support for it."
            )
        # Intercepted turns already own the typed prompt, so only their tools need parsing here.
        if turn is not None:
            prompt = turn.prompt
            tools = parse_tools(body.get("tools"))
        else:
            prompt, tools = dialect.parse_request(body)
        renderer = self._renderer_pool(model)
        from renderers.client import _maybe_offload, generate

        wire_tools = [tool_to_wire(t) for t in tools] if tools else None
        wire_messages = (
            [message_to_wire(m) for m in turn.tail] if turn is not None else []
        )
        prompt_ids: list[int] | None = None
        multi_modal_data = None
        prompt_attribution: RenderedTokens | None = None
        sampling_params = sampling_args.model_dump(exclude_none=True)
        bridged_turn: PendingTurn | None = None

        # Only build the (O(context)) previous-turn token ids once the cheap guards pass — a
        # multimodal prompt or a tail that isn't a clean `[tool*, user?]` extension can't bridge.
        can_bridge = (
            self.renderer_transport in {"vllm_generate", "dynamo_vllm_generate"}
            and
            turn is not None
            and not _has_multimodal_content(prompt)
            and _is_valid_incremental_tail(wire_messages)
        )
        previous_ids = turn.previous_token_ids() if can_bridge else None
        if previous_ids is not None:
            previous_prompt_ids, previous_completion_ids = previous_ids

            def bridge():
                return renderer.bridge_to_next_turn(
                    previous_prompt_ids,
                    previous_completion_ids,
                    wire_messages,
                    tools=wire_tools,
                )

            bridged = await _maybe_offload(renderer, bridge)
            if bridged is not None:
                prompt_ids = bridged.token_ids
                multi_modal_data = bridged.multi_modal_data
                prompt_attribution = bridged
                bridged_turn = turn
                sampling_params["routed_experts_prompt_start"] = max(
                    len(previous_prompt_ids) + len(previous_completion_ids) - 1,
                    0,
                )

        # Bridged prompt ids bypass rendering; only fallback needs the full wire prompt.
        if prompt_ids is None:
            wire_messages = [message_to_wire(m) for m in prompt]

        try:
            if self.renderer_transport == "dynamo_chat":
                result = await generate_dynamo_chat(
                    client=self.openai,
                    renderer=renderer,
                    messages=wire_messages,
                    model=model,
                    tools=wire_tools,
                    sampling_params=sampling_params,
                    extra_headers={SESSION_ID_HEADER: session_id} if session_id else None,
                )
            else:
                generate_kwargs: dict[str, Any] = {}
                if self.renderer_transport == "dynamo_vllm_generate":
                    generate_kwargs["transport"] = self.renderer_transport
                result = await generate(
                    client=self.openai,
                    renderer=renderer,
                    messages=wire_messages,
                    model=model,
                    prompt_ids=prompt_ids,
                    multi_modal_data=multi_modal_data,
                    prompt_attribution=prompt_attribution,
                    tools=wire_tools,
                    sampling_params=sampling_params,
                    extra_headers={SESSION_ID_HEADER: session_id} if session_id else None,
                    **generate_kwargs,
                )
        except RendererOverlongPromptError as e:
            raise OverlongPromptError(str(e)) from e
        except OpenAIError as e:
            raise model_error(e) from e
        response = response_from_generate(result, model, bridged_turn)
        # No provider response to relay (we generated), so serialize one for the program; the
        # interception server hands `Response.raw` back regardless of client.
        response.raw = serialize_completion(response, model)
        return response

    async def close(self) -> None:
        await self.openai.close()
