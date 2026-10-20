"""LiveKit worker: joins a room, dials the customer over SIP, and runs the call.

    python -m duecall.main download-files   # fetch VAD + turn-detector weights once
    python -m duecall.main console          # talk to it in your terminal (mic + speakers)
    python -m duecall.main dev              # connect to LiveKit Cloud for browser / phone tests
    python -m duecall.main start            # production

Outbound calls are dispatched by duecall.dialer with job metadata like
{"customer_id": "CUST-001", "phone": "+9198...", "prompt_version": "v2"}.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import date

from dotenv import load_dotenv
from livekit import api
from livekit.agents import AgentServer, AgentSession, AgentStateChangedEvent, JobContext, JobProcess, MetricsCollectedEvent, cli, inference, metrics
# Plugins must be registered on the main thread at import time, not inside a job.
from livekit.plugins import anthropic, cartesia, deepgram, google, openai, silero  # noqa: F401

from . import prompts
from .agents import CallState, CollectionsAgent
from .config import Settings, make_llm, make_stt, make_tts
from .ledger import Ledger

load_dotenv()
log = logging.getLogger("duecall")
settings = Settings()


def prewarm(proc: JobProcess) -> None:
    # Loaded once per worker process, not per call: VAD load is ~100s of ms.
    proc.userdata["vad"] = silero.VAD.load()


server = AgentServer(setup_fnc=prewarm, port=int(os.getenv("PORT", "8081")))


def _record_latency(state: CallState, ev: MetricsCollectedEvent) -> None:
    m = ev.metrics
    if isinstance(m, metrics.EOUMetrics):
        state.ledger.record_metric(state.call_id, "end_of_utterance_ms", m.end_of_utterance_delay * 1000)
    elif isinstance(m, metrics.LLMMetrics) and m.ttft > 0:
        state.ledger.record_metric(state.call_id, "llm_ttft_ms", m.ttft * 1000)
    elif isinstance(m, metrics.TTSMetrics) and m.ttfb > 0:
        state.ledger.record_metric(state.call_id, "tts_ttfb_ms", m.ttfb * 1000)


@server.rtc_session(agent_name=settings.agent_name)
async def entrypoint(ctx: JobContext) -> None:
    meta = json.loads(ctx.job.metadata or "{}")
    customer_id = meta.get("customer_id", "CUST-001")
    phone = meta.get("phone")
    version = meta.get("prompt_version", settings.prompt_version)

    os.makedirs(os.path.dirname(settings.ledger_path) or ".", exist_ok=True)
    ledger = Ledger(settings.ledger_path)
    if ledger.customer(customer_id) is None:
        ledger.seed_demo(date.today())
    customer = ledger.customer(customer_id)
    call_id = ledger.start_call(customer_id, f"{version}@{prompts.fingerprint(version)}",
                                channel="phone" if phone else "web")
    state = CallState(ledger=ledger, customer_id=customer_id, call_id=call_id,
                      today=date.today(), prompt_version=version)

    session = AgentSession[CallState](
        userdata=state,
        stt=make_stt(settings),
        llm=make_llm(settings),
        tts=make_tts(settings),
        vad=ctx.proc.userdata["vad"],
        turn_detection=inference.TurnDetector(),  # semantic end-of-turn: fewer cut-offs on "I'll pay on... the fifth"
        preemptive_generation=settings.preemptive,  # LLM starts on the interim transcript (lower latency)
        max_tool_steps=6,
        # Indian-English STT can finalise a beat late; wait a little longer before
        # committing the turn so "I am from ... Shree Ganesh" arrives as one message.
        min_endpointing_delay=0.6,
    )
    session.on("metrics_collected", lambda ev: _record_latency(state, ev))

    async def on_shutdown() -> None:
        if state.outcome is None:
            ledger.end_call(call_id, "no_commitment")

    ctx.add_shutdown_callback(on_shutdown)

    def on_state(ev: AgentStateChangedEvent) -> None:
        # Once the call is closed and the goodbye has finished playing, hang up:
        # deleting the room disconnects the SIP leg, so the phone line is released.
        if state.closed and ev.old_state == "speaking" and ev.new_state == "listening":
            asyncio.create_task(_hang_up())

    async def _hang_up() -> None:
        await ctx.delete_room()
        ctx.shutdown(reason="call closed")

    session.on("agent_state_changed", on_state)

    if phone:
        if not settings.sip_trunk_id:
            raise RuntimeError("SIP_OUTBOUND_TRUNK_ID is not set; cannot place a phone call.")
        try:
            # Blocks until the customer picks up, so the agent never talks to a ringing line.
            await ctx.api.sip.create_sip_participant(api.CreateSIPParticipantRequest(
                room_name=ctx.room.name,
                sip_trunk_id=settings.sip_trunk_id,
                sip_call_to=phone,
                participant_identity=f"customer-{customer_id}",
                wait_until_answered=True,
            ))
        except api.TwirpError as e:
            log.warning("call to %s not answered: %s %s", phone, e.message, e.metadata.get("sip_status_code"))
            ledger.end_call(call_id, "voicemail")
            ctx.shutdown()
            return

    await session.start(agent=CollectionsAgent(state), room=ctx.room)
    # Fixed opening line: no LLM round-trip, so the customer hears us the moment they say hello.
    await session.say(prompts.opening_line(customer.contact_name), allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(server)
