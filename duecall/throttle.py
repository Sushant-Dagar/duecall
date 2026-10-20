"""Request pacing for free-tier LLM keys.

Free tiers cap requests and tokens per minute, and one voice turn can fire several
LLM requests in a burst (reply, tool call, follow-up). ThrottledLLM wraps any
LiveKit LLM and spaces its requests at least `min_interval` seconds apart, shared
across every user of the same limiter (the agent and the simulated customer)."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, llm


class Limiter:
    def __init__(self, requests_per_minute: float) -> None:
        self.min_interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            if self._next > now:
                await asyncio.sleep(self._next - now)
            self._next = max(now, self._next) + self.min_interval


class ThrottledLLM(llm.LLM):
    def __init__(self, inner: llm.LLM, limiter: Limiter) -> None:
        super().__init__()
        self.inner, self.limiter = inner, limiter

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def provider(self) -> str:
        return self.inner.provider

    def chat(self, *, chat_ctx, tools=None, conn_options=DEFAULT_API_CONNECT_OPTIONS, **kw: Any):
        return _Stream(self, kw, chat_ctx=chat_ctx, tools=tools or [], conn_options=conn_options)


class _Stream(llm.LLMStream):
    def __init__(self, owner: ThrottledLLM, extra: dict, **kw) -> None:
        super().__init__(owner, **kw)
        self._owner, self._extra = owner, extra

    async def _run(self) -> None:
        await self._owner.limiter.wait()
        async with self._owner.inner.chat(chat_ctx=self._chat_ctx, tools=self._tools,
                                          conn_options=self._conn_options, **self._extra) as stream:
            async for chunk in stream:
                self._event_ch.send_nowait(chunk)
