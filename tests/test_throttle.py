import asyncio
import time

from livekit.agents import AgentSession

from duecall.agents import CollectionsAgent
from duecall.scripted_llm import ScriptedLLM
from duecall.throttle import Limiter, ThrottledLLM


async def test_limiter_spaces_requests():
    lim = Limiter(requests_per_minute=600)  # 0.1 s apart
    start = time.monotonic()
    await asyncio.gather(*(lim.wait() for _ in range(4)))
    assert time.monotonic() - start >= 0.28


async def test_throttled_llm_runs_a_real_session(make_state):
    state = make_state("CUST-001")
    inner = ScriptedLLM([("tool", "verify_identity", {"company_name": "Shree Ganesh Traders"}), ("say", "ok")])
    model = ThrottledLLM(inner, Limiter(requests_per_minute=600))
    async with AgentSession(llm=model, userdata=state) as session:
        await session.start(CollectionsAgent(state))
        await session.run(user_input="Shree Ganesh Traders")
    assert state.verified and len(inner.requests) == 2
