"""
Unified LLM client for Anthropic, OpenAI, Gemini, Grok, OpenRouter, and Ollama.

Set LLM_PROVIDER and optionally LLM_MODEL in the environment.
All agents use the same ToolCall / ChatResponse types regardless of provider.
"""

import json
import os
from dataclasses import dataclass, field


# ── Provider / model config ───────────────────────────────────────────────────

PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()

_DEFAULT_MODELS: dict[str, str] = {
    "anthropic":  "claude-sonnet-4-6",
    "openai":     "gpt-4o",
    "gemini":     "gemini-2.0-flash",
    "grok":       "grok-3-mini",
    "openrouter": "anthropic/claude-sonnet-4-6",
    "ollama":     "llama3.1",
}

MODEL = os.environ.get("LLM_MODEL") or _DEFAULT_MODELS.get(PROVIDER, "gpt-4o")

_BASE_URLS: dict[str, str] = {
    "openai":     "https://api.openai.com/v1",
    "gemini":     "https://generativelanguage.googleapis.com/v1beta/openai/",
    "grok":       "https://api.x.ai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "ollama":     os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
}

_API_KEY_VARS: dict[str, str] = {
    "anthropic":  "ANTHROPIC_API_KEY",
    "openai":     "OPENAI_API_KEY",
    "gemini":     "GEMINI_API_KEY",
    "grok":       "GROK_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "ollama":     "",
}


# ── Neutral types ─────────────────────────────────────────────────────────────

@dataclass
class ToolCall:
    id: str
    name: str
    input: dict  # already-parsed


@dataclass
class ChatResponse:
    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str = ""
    stop_reason: str = "end_turn"  # "tool_use" | "end_turn"
    usage_input: int = 0
    usage_output: int = 0


# ── History format (neutral) ─────────────────────────────────────────────────
#
#   {"role": "user",         "content": str}
#   {"role": "assistant",    "text": str, "tool_calls": list[ToolCall]}
#   {"role": "tool_results", "results": [{"id": str, "content": str}]}


def _anthropic_messages(history: list[dict]) -> list[dict]:
    out = []
    for turn in history:
        role = turn["role"]
        if role == "user":
            out.append({"role": "user", "content": turn["content"]})
        elif role == "assistant":
            content: list = []
            if turn.get("text"):
                content.append({"type": "text", "text": turn["text"]})
            for tc in turn.get("tool_calls", []):
                content.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input})
            out.append({"role": "assistant", "content": content})
        elif role == "tool_results":
            out.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": r["id"], "content": r["content"]}
                    for r in turn["results"]
                ],
            })
    return out


def _anthropic_tools(tools: list[dict]) -> list[dict]:
    return [
        {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
        for t in tools
    ]


def _openai_messages(history: list[dict], system: str) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for turn in history:
        role = turn["role"]
        if role == "user":
            out.append({"role": "user", "content": turn["content"]})
        elif role == "assistant":
            msg: dict = {"role": "assistant", "content": turn.get("text") or ""}
            tcs = turn.get("tool_calls", [])
            if tcs:
                msg["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.name, "arguments": json.dumps(tc.input)}}
                    for tc in tcs
                ]
            out.append(msg)
        elif role == "tool_results":
            for r in turn["results"]:
                out.append({"role": "tool", "tool_call_id": r["id"], "content": r["content"]})
    return out


def _openai_tools(tools: list[dict]) -> list[dict]:
    return [
        {"type": "function",
         "function": {"name": t["name"], "description": t["description"], "parameters": t["parameters"]}}
        for t in tools
    ]


# ── LLMClient ─────────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, provider: str, model: str) -> None:
        self._provider = provider
        self._model = model
        self._a = None   # anthropic.Anthropic
        self._o = None   # openai.OpenAI

    def _anthropic(self):
        if self._a is None:
            import anthropic
            self._a = anthropic.Anthropic()
        return self._a

    def _openai(self):
        if self._o is None:
            from openai import OpenAI
            key_var = _API_KEY_VARS.get(self._provider, "OPENAI_API_KEY")
            api_key = os.environ.get(key_var) or "nokey"
            base_url = _BASE_URLS.get(self._provider)
            headers = (
                {"HTTP-Referer": "https://github.com/agentforge", "X-Title": "AgentForge"}
                if self._provider == "openrouter" else {}
            )
            self._o = OpenAI(base_url=base_url, api_key=api_key,
                             default_headers=headers or None)
        return self._o

    def chat(
        self,
        system: str,
        history: list[dict],
        tools: list[dict] | None = None,
        max_tokens: int = 1024,
        force_tool: str | None = None,
    ) -> ChatResponse:
        if self._provider == "anthropic":
            return self._chat_anthropic(system, history, tools, max_tokens, force_tool)
        return self._chat_openai(system, history, tools, max_tokens, force_tool)

    def _chat_anthropic(self, system, history, tools, max_tokens, force_tool) -> ChatResponse:
        kw: dict = {
            "model": self._model,
            "system": system,
            "messages": _anthropic_messages(history),
            "max_tokens": max_tokens,
        }
        if tools:
            kw["tools"] = _anthropic_tools(tools)
            kw["tool_choice"] = (
                {"type": "tool", "name": force_tool} if force_tool else {"type": "auto"}
            )
        r = self._anthropic().messages.create(**kw)
        tcs = [ToolCall(id=b.id, name=b.name, input=b.input)
               for b in r.content if b.type == "tool_use"]
        text = "\n".join(b.text for b in r.content if b.type == "text")
        return ChatResponse(tool_calls=tcs, text=text,
                            stop_reason="tool_use" if tcs else "end_turn",
                            usage_input=r.usage.input_tokens,
                            usage_output=r.usage.output_tokens)

    def _chat_openai(self, system, history, tools, max_tokens, force_tool) -> ChatResponse:
        kw: dict = {
            "model": self._model,
            "messages": _openai_messages(history, system),
            "max_tokens": max_tokens,
        }
        if tools:
            kw["tools"] = _openai_tools(tools)
            kw["tool_choice"] = (
                {"type": "function", "function": {"name": force_tool}} if force_tool else "auto"
            )
        r = self._openai().chat.completions.create(**kw)
        msg = r.choices[0].message
        tcs = []
        if msg.tool_calls:
            tcs = [ToolCall(id=tc.id, name=tc.function.name,
                            input=json.loads(tc.function.arguments))
                   for tc in msg.tool_calls]
        return ChatResponse(tool_calls=tcs, text=msg.content or "",
                            stop_reason="tool_use" if tcs else "end_turn",
                            usage_input=getattr(r.usage, "prompt_tokens", 0),
                            usage_output=getattr(r.usage, "completion_tokens", 0))


# ── Singleton ─────────────────────────────────────────────────────────────────

_client: LLMClient | None = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient(provider=PROVIDER, model=MODEL)
    return _client
