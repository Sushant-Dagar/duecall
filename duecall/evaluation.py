"""Evaluation: run simulated calls against each prompt version and score them.

Each scenario pairs a demo account with a customer persona (an LLM playing the
customer) and the outcomes a good call could end in. Each transcript is scored on:

  outcome_ok        the call ended in an acceptable outcome
  hallucinations    amounts the agent SAID that no tool disclosed (parsed from spoken words)
  leaked_early      amounts spoken before identity was verified
  tts_violations    digits / symbols / refs a TTS engine would misread
  words_per_turn    brevity; long turns are bad on a phone
  judge             optional LLM self-reflection score (1 to 5) against a rubric

The deterministic checks are the ones that gate a release; the judge is advisory.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from decimal import Decimal

from livekit.agents import AgentSession, llm
from livekit.agents.voice.run_result import ChatMessageEvent, FunctionCallOutputEvent

from .agents import CallState, CollectionsAgent
from .speech import unspeakable_tokens

# ---------------------------------------------------------------- spoken numbers
_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if i > 1}
_SCALES = {"thousand": 1_000, "lakh": 100_000, "lakhs": 100_000, "crore": 10_000_000, "crores": 10_000_000,
           "million": 1_000_000, "billion": 1_000_000_000}
_NUM_WORDS = set(_UNITS) | set(_TENS) | set(_SCALES) | {"hundred"}
_MAJOR = {"rupees", "rupee", "dollars", "dollar", "rupiah"}
_MINOR = {"paise", "cents", "sen"}


def words_to_int(words: list[str]) -> int:
    total = current = 0
    for w in words:
        if w in _UNITS:
            current += _UNITS[w]
        elif w in _TENS:
            current += _TENS[w]
        elif w == "hundred":
            current = (current or 1) * 100
        elif w in _SCALES:
            total += (current or 1) * _SCALES[w]
            current = 0
    return total + current


def spoken_amounts(text: str) -> list[Decimal]:
    """'one lakh twenty-four thousand five hundred rupees and fifty paise' -> [Decimal('124500.50')].
    Also catches amounts the model wrote in digits ('Rs 4,800')."""
    toks = re.findall(r"[a-z]+", text.lower().replace("-", " "))
    found: list[Decimal] = []
    for i, t in enumerate(toks):
        if t not in _MAJOR:
            continue
        j = i
        while j > 0 and toks[j - 1] in _NUM_WORDS:
            j -= 1
        if j == i:
            continue
        value = Decimal(words_to_int(toks[j:i]))
        # "... rupees and fifty paise"
        k = i + 1
        if k < len(toks) and toks[k] == "and":
            m = k + 1
            while m < len(toks) and toks[m] in _NUM_WORDS:
                m += 1
            if m < len(toks) and toks[m] in _MINOR and m > k + 1:
                value += Decimal(words_to_int(toks[k + 1:m])) / 100
        found.append(value.quantize(Decimal("0.01")))
    for raw in re.findall(r"(?:₹|rs\.?|inr|\$|sgd|usd)\s?([\d,]+(?:\.\d{1,2})?)", text.lower()):
        found.append(Decimal(raw.replace(",", "")).quantize(Decimal("0.01")))
    return found


# ---------------------------------------------------------------- scenarios
@dataclass(frozen=True)
class Scenario:
    id: str
    customer_id: str
    persona: str
    acceptable: frozenset[str]


@dataclass
class Transcript:
    scenario: str
    version: str
    turns: list[tuple[str, str]] = field(default_factory=list)   # (speaker, text)
    tool_calls: list[str] = field(default_factory=list)
    outcome: str | None = None
    disclosed: set[Decimal] = field(default_factory=set)
    verified_at_turn: int | None = None


@dataclass
class Score:
    scenario: str
    version: str
    outcome: str | None
    outcome_ok: bool
    hallucinations: list[str]
    leaked_early: int
    tts_violations: list[str]
    words_per_turn: float
    agent_turns: int
    judge: float | None = None


def score(t: Transcript, s: Scenario) -> Score:
    agent = [(i, text) for i, (who, text) in enumerate(t.turns) if who == "agent"]
    halluc, early, tts = [], 0, []
    for i, text in agent:
        for amt in spoken_amounts(text):
            if amt not in t.disclosed:
                halluc.append(str(amt))
            if t.verified_at_turn is None or i < t.verified_at_turn:
                early += 1
        tts += unspeakable_tokens(text)
    words = [len(text.split()) for _, text in agent] or [0]
    return Score(s.id, t.version, t.outcome, t.outcome in s.acceptable, halluc, early, tts,
                 round(statistics.mean(words), 1), len(agent))


# ---------------------------------------------------------------- running a call
Customer = Callable[[list[tuple[str, str]]], Awaitable[str | None]]


async def run_call(scenario: Scenario, version: str, agent_llm: llm.LLM, customer: Customer,
                   make_state: Callable[[str, str], CallState], *, max_turns: int = 10,
                   opening: str | None = None) -> Transcript:
    """Text-mode call: the same AgentSession and tools as a phone call, minus audio."""
    state = make_state(scenario.customer_id, version)
    t = Transcript(scenario.id, version)
    if opening:
        t.turns.append(("agent", opening))
    async with AgentSession(llm=agent_llm, userdata=state, max_tool_steps=6) as session:
        await session.start(CollectionsAgent(state))
        for _ in range(max_turns):
            said = await customer(t.turns)
            if not said:
                break
            t.turns.append(("customer", said))
            result = await session.run(user_input=said)
            for ev in result.events:
                if isinstance(ev, FunctionCallOutputEvent):
                    t.tool_calls.append(ev.item.name + ("!" if ev.item.is_error else ""))
                    if ev.item.name == "verify_identity" and state.verified and t.verified_at_turn is None:
                        t.verified_at_turn = len(t.turns)  # agent speech from here on may cite amounts
                elif isinstance(ev, ChatMessageEvent) and ev.item.role == "assistant":
                    t.turns.append(("agent", ev.item.text_content or ""))
            if "end_call" in t.tool_calls or state.closed:
                break
    t.outcome = state.outcome
    t.disclosed = set(state.disclosed_amounts)
    return t


def llm_customer(model: llm.LLM, persona: str) -> Customer:
    """An LLM playing the customer. Returns None when it decides the call is over."""
    async def speak(turns: list[tuple[str, str]]) -> str | None:
        ctx = llm.ChatContext.empty()
        ctx.add_message(role="system", content=(
            "You are role-playing a business customer who has just answered an outbound phone call. The caller "
            "works for Acme Supplies, a supplier your company buys from, and is calling about overdue invoices. "
            "You do NOT work for Acme Supplies: when asked which company you are with, give YOUR company from "
            f"the persona.\nPersona: {persona}\nReply with ONLY what you say out loud, one or two short sentences, "
            "as transcribed speech (numbers in words). Only when the caller has clearly ended the call with a "
            "goodbye, reply exactly: <hangup>"))
        for who, text in turns:
            ctx.add_message(role="user" if who == "agent" else "assistant", content=text)
        reply = (await model.chat(chat_ctx=ctx).collect()).text.strip()
        return None if not reply or "<hangup>" in reply else reply
    return speak


JUDGE_RUBRIC = """Score this collections phone call from 1 (poor) to 5 (excellent) on: verifying identity before
sharing details, accuracy, one question per turn, respectful tone, confirming amounts and dates before saving,
and reaching a clear outcome. Reply as JSON: {"score": <1-5>, "reason": "<one sentence>"}"""


async def judge(model: llm.LLM, t: Transcript) -> tuple[float | None, str]:
    """Self-reflection pass: a model critiques the transcript against a rubric."""
    ctx = llm.ChatContext.empty()
    ctx.add_message(role="system", content=JUDGE_RUBRIC)
    ctx.add_message(role="user", content="\n".join(f"{w.upper()}: {x}" for w, x in t.turns)
                    + f"\nTOOLS: {', '.join(t.tool_calls)}\nOUTCOME: {t.outcome}")
    raw = (await model.chat(chat_ctx=ctx).collect()).text
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(m.group(0)) if m else {}
        return float(data["score"]), str(data.get("reason", ""))
    except (ValueError, KeyError, TypeError):
        return None, raw[:200]


def summarize(scores: list[Score]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for v in sorted({s.version for s in scores}):
        rows = [s for s in scores if s.version == v]
        judged = [s.judge for s in rows if s.judge is not None]
        out[v] = {
            "calls": len(rows),
            "outcome_accuracy": round(sum(s.outcome_ok for s in rows) / len(rows), 3),
            "hallucinated_amounts": sum(len(s.hallucinations) for s in rows),
            "leaked_before_verify": sum(s.leaked_early for s in rows),
            "tts_violations": sum(len(s.tts_violations) for s in rows),
            "words_per_turn": round(statistics.mean(s.words_per_turn for s in rows), 1),
            "judge_avg": round(statistics.mean(judged), 2) if judged else None,
        }
    return out


def to_markdown(summary: dict[str, dict], scores: list[Score]) -> str:
    keys = ["calls", "outcome_accuracy", "hallucinated_amounts", "leaked_before_verify",
            "tts_violations", "words_per_turn", "judge_avg"]
    lines = ["| version | " + " | ".join(keys) + " |", "|" + "---|" * (len(keys) + 1)]
    lines += [f"| {v} | " + " | ".join(str(r[k]) for k in keys) + " |" for v, r in summary.items()]
    lines += ["", "| scenario | version | outcome | ok | hallucinations | tts |", "|---|---|---|---|---|---|"]
    lines += [f"| {s.scenario} | {s.version} | {s.outcome} | {'yes' if s.outcome_ok else 'no'} | "
              f"{len(s.hallucinations)} | {len(s.tts_violations)} |" for s in scores]
    return "\n".join(lines)


def dump(scores: list[Score]) -> str:
    return json.dumps([asdict(s) for s in scores], indent=2, default=str)
