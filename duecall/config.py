"""Model providers, picked by environment variables so the same agent can be
benchmarked on different STT / LLM / TTS stacks without code changes.

    STT_PROVIDER = deepgram (default) | google
    LLM_PROVIDER = google (default, Gemini) | openai (GPT) | anthropic (Claude) | groq (Llama, gpt-oss)
    TTS_PROVIDER = cartesia (default) | google

The worker (main.py) imports every plugin at startup, as LiveKit requires; the
imports below then resolve instantly. Tests and the dashboard never import plugins.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # before Settings reads the environment

DEFAULT_MODELS = {
    "llm": {"google": "gemini-3.8-flash", "openai": "gpt-4o-mini", "anthropic": "claude-haiku-4-5",
            "groq": "openai/gpt-oss-120b"},
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
    # Preemptive generation starts the LLM on the interim transcript: lower latency,
    # but up to twice the requests per turn. Turn it off on free-tier rate limits.
    preemptive: bool = os.getenv("PREEMPTIVE_GENERATION", "1") == "1"
    # Gemini 3 models think before answering; on a phone call every second of
    # thinking is silence, so default to the minimum.
    gemini_thinking: str = os.getenv("GEMINI_THINKING", "minimal")
    agent_name: str = os.getenv("AGENT_NAME", "duecall")


def make_llm(s: Settings):
    model = s.llm_model or DEFAULT_MODELS["llm"][s.llm_provider]
    if s.llm_provider == "google":
        from google.genai import types
        from livekit.plugins import google
        extra = {}
        if model.startswith("gemini-3"):
            extra["thinking_config"] = types.ThinkingConfig(thinking_level=s.gemini_thinking.upper())
        return google.LLM(model=model, temperature=0.3, **extra)
    if s.llm_provider == "groq":
        from livekit.plugins import openai  # Groq serves an OpenAI-compatible API
        extra = {"reasoning_effort": "low"} if "gpt-oss" in model else {}  # reasoning time is silence on a call
        return openai.LLM(model=model, base_url="https://api.groq.com/openai/v1",
                          api_key=os.environ["GROQ_API_KEY"], temperature=0.3,
                          parallel_tool_calls=False, _strict_tool_schema=False, **extra)
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
