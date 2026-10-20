"""A deterministic stand-in LLM for offline tests.

It plays back a fixed list of steps (say this text / call this tool), so the real
AgentSession, tool execution, handoffs and ledger writes can be tested end to end
in CI without API keys or network. Model quality is measured separately by the
eval harness against real providers.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, llm


class ScriptedLLM(llm.LLM):
    def __init__(self, steps: list[tuple[str, Any] | tuple[str, str, dict]]) -> None:
        super().__init__()
        self.steps = list(steps)
        self.requests: list[llm.ChatContext] = []

    @property
    def model(self) -> str:
        return "scripted"

    def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS, **_: Any):
        self.requests.append(chat_ctx.copy())
        step = self.steps.pop(0) if self.steps else ("say", "")
        return _Stream(self, step, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class _Stream(llm.LLMStream):
    def __init__(self, owner: ScriptedLLM, step, **kw) -> None:
        super().__init__(owner, **kw)
        self._step = step

    async def _run(self) -> None:
        kind = self._step[0]
        if kind == "say":
            delta = llm.ChoiceDelta(role="assistant", content=self._step[1])
        elif kind == "tool":
            _, name, args = self._step
            delta = llm.ChoiceDelta(role="assistant", tool_calls=[llm.FunctionToolCall(
                name=name, arguments=json.dumps(args), call_id="call_" + uuid.uuid4().hex[:8])])
        else:
            raise ValueError(f"unknown step {self._step!r}")
        self._event_ch.send_nowait(llm.ChatChunk(id=uuid.uuid4().hex, delta=delta))
