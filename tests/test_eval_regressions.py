"""Regressions found by reading the first full eval transcripts (2026-10-20)."""

from livekit.agents import AgentSession
from livekit.agents.voice.run_result import FunctionCallOutputEvent

from duecall.agents import CollectionsAgent, PaymentsAgent
from duecall.evaluation import Scenario, run_call
from duecall.scripted_llm import ScriptedLLM
from duecall.speech import normalize_for_tts, unspeakable_tokens


def outputs(result):
    return [e.item for e in result.events if isinstance(e, FunctionCallOutputEvent)]


async def test_naming_our_own_company_is_confusion_not_a_failed_attempt(make_state):
    # Transcript: agent asked "which company are you calling from?", customer said "Acme Supplies".
    state = make_state("CUST-001")
    model = ScriptedLLM([
        ("tool", "verify_identity", {"company_name": "Acme Supplies"}), ("say", "And which company are you with?"),
        ("tool", "verify_identity", {"company_name": "Acme Supplies Pvt Ltd"}), ("say", "Your company, I mean?"),
        ("tool", "verify_identity", {"company_name": "Shree Ganesh Traders"}), ("say", "Thanks."),
    ])
    async with AgentSession(llm=model, userdata=state) as session:
        await session.start(CollectionsAgent(state))
        r = await session.run(user_input="Acme Supplies.")
        assert "our own company" in outputs(r)[0].output
        await session.run(user_input="Acme Supplies.")
        await session.run(user_input="Oh, Shree Ganesh Traders.")
    assert state.verify_attempts == 0 and state.verified


async def test_failed_verification_closes_the_call_in_code(make_state, ledger):
    # Transcript: after failing, the model kept calling verify_identity and spoke JSON instead of end_call.
    state = make_state("CUST-002")
    model = ScriptedLLM([
        ("tool", "verify_identity", {"company_name": "Sunrise Hardware"}), ("say", "Could you repeat that?"),
        ("tool", "verify_identity", {"company_name": "Sunrise Hardware"}), ("say", "I'll email instead. Goodbye."),
        ("tool", "verify_identity", {"company_name": "Nova Retail"}), ("say", "Goodbye."),
    ])
    async with AgentSession(llm=model, userdata=state) as session:
        await session.start(CollectionsAgent(state))
        await session.run(user_input="Sunrise Hardware")
        await session.run(user_input="Sunrise Hardware")
        assert state.closed and state.outcome == "verification_failed"
        r = await session.run(user_input="Actually Nova Retail")
    assert "call is closed" in outputs(r)[0].output and not state.verified
    row = ledger.conn.execute("SELECT outcome, ended_at FROM calls WHERE id = ?", (state.call_id,)).fetchone()
    assert row["outcome"] == "verification_failed" and row["ended_at"]


async def test_handoff_does_not_sound_like_goodbye(make_state):
    # Transcript: "I'm transferring you to our payments specialist... thank you for your call" -> customer hung up.
    state = make_state("CUST-001")
    model = ScriptedLLM([
        ("tool", "verify_identity", {"company_name": "Shree Ganesh Traders"}), ("say", "ok"),
        ("tool", "transfer_to_payments", {}), ("say", "Sure, what date works?"),
    ])
    async with AgentSession(llm=model, userdata=state) as session:
        await session.start(CollectionsAgent(state))
        await session.run(user_input="Shree Ganesh Traders")
        r = await session.run(user_input="I'd like to pay")
    assert isinstance(session.current_agent, PaymentsAgent)
    assert "do not mention a transfer" in outputs(r)[0].output


def test_json_and_unicode_hyphens_are_never_spoken():
    text = 'I am unable to verify you. {\n  "outcome": "verification_failed"\n} Invoice INV‑2041 is due.'
    out = normalize_for_tts(text)
    assert "{" not in out and "outcome" not in out
    assert "I N V, two zero four one" in out
    assert unspeakable_tokens(text)  # the eval counts the raw text as a violation


async def test_eval_stops_when_the_call_is_closed(make_state):
    sc = Scenario("wrong-company", "CUST-002", "", frozenset({"verification_failed"}))
    model = ScriptedLLM([
        ("tool", "verify_identity", {"company_name": "Sunrise Hardware"}), ("say", "Sorry, which company?"),
        ("tool", "verify_identity", {"company_name": "Sunrise Hardware"}), ("say", "I'll email you. Goodbye."),
    ])
    lines = iter(["Sunrise Hardware.", "Sunrise Hardware.", "Hello?", "Are you there?"])

    async def customer(_turns):
        return next(lines, None)

    t = await run_call(sc, "v2", model, customer, lambda c, v: make_state(c, v))
    assert t.outcome == "verification_failed"
    assert [x for w, x in t.turns if w == "customer"] == ["Sunrise Hardware.", "Sunrise Hardware."]
