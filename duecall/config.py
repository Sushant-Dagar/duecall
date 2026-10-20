"""Model providers, picked by environment variables so the same agent can be
benchmarked on different STT / LLM / TTS stacks without code changes.

    STT_PROVIDER = deepgram (default) | google
    LLM_PROVIDER = google (default, Gemini) | openai (GPT) | anthropic (Claude)
    TTS_PROVIDER = cartesia (default) | google

Plugins are imported lazily, so tests and the dashboard run without them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODELS = {
    "llm": {"google": "gemini-2.5-flash", "openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5"},
    "stt": {"deepgram": "nova-3", "google": "chirp_2"},
}


@dataclass(frozen=True)
class Settings:
    stt_provider: str = os.getenv("STT_PROVIDER", "deepgram")
    llm_provider: str = os.getenv("LLM_PROVIDER", "google")
    llm_model: str | None = os.getenv("LLM_MODEL") or None
    tts_provider: str = os.getenv("TTS_PROVIDER", "cartesia")
    tts_voice: str | None = os.getenv("TTS_VOICE") or None
    ledger_path: str = os.getenv("LEDGER_PATH", "data/ledger.db")
    sip_trunk_id: str | None = os.getenv("SIP_OUTBOUND_TRUNK_ID") or None
    prompt_version: str = os.getenv("PROMPT_VERSION", "v2")
    agent_name: str = os.getenv("AGENT_NAME", "duecall")


def make_llm(s: Settings):
    model = s.llm_model or DEFAULT_MODELS["llm"][s.llm_provider]
    if s.llm_provider == "google":
        from livekit.plugins import google
        return google.LLM(model=model, temperature=0.3)
    if s.llm_provider == "openai":
        from livekit.plugins import openai
        return openai.LLM(model=model, temperature=0.3)
    if s.llm_provider == "anthropic":
        from livekit.plugins import anthropic
        return anthropic.LLM(model=model, temperature=0.3)
    raise ValueError(f"unknown LLM_PROVIDER {s.llm_provider!r}")


def make_stt(s: Settings):
    if s.stt_provider == "deepgram":
        from livekit.plugins import deepgram
        return deepgram.STT(model="nova-3", language="en-IN")
    if s.stt_provider == "google":
        from livekit.plugins import google
        return google.STT(model="chirp_2", languages=["en-IN"])
    raise ValueError(f"unknown STT_PROVIDER {s.stt_provider!r}")


def make_tts(s: Settings):
    if s.tts_provider == "cartesia":
        from livekit.plugins import cartesia
        return cartesia.TTS(model="sonic-3", **({"voice": s.tts_voice} if s.tts_voice else {}))
    if s.tts_provider == "google":
        from livekit.plugins import google
        return google.TTS(language="en-IN", **({"voice_name": s.tts_voice} if s.tts_voice else {}))
    raise ValueError(f"unknown TTS_PROVIDER {s.tts_provider!r}")
