"""End-to-end test of the RL seam: robot client → server → learner and back.

The robot client used here is the ordinary ``RemotePolicyClient`` with no RL awareness
at all, which is the property that matters most: enabling RL must not require the robot
to change. What it should observe is a normal prediction plus ``rl_enabled`` /
``q_value``, while the agent behind the server sees observations, transitions and
episode boundaries.

Needs a real pi0 checkpoint and a GPU, so it skips when either is missing.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest
import simplejpeg

from policy_inference_spec.client import RemotePolicyClient
from policy_inference_spec.hardware_model import DEFAULT_HARDWARE_MODEL
from policy_inference_spec.protocol import JOINT_STATE_KEY, MODEL_ID_KEY, SUBTASK_KEY, TASK_KEY
from policy_inference_spec.rl import RandomLatentAgent, serve_agent

torch = pytest.importorskip("torch")

CHECKPOINT = Path(__file__).resolve().parent.parent / "models" / "ultra-towerstack-pi0-15000"
POLICY_TYPE = "pi0"
AGENT_PORT = 18097

pytestmark = [
    pytest.mark.skipif(not CHECKPOINT.exists(), reason=f"checkpoint not available: {CHECKPOINT}"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _predict_frame(seed: int, **extra) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    height, width = DEFAULT_HARDWARE_MODEL.image_resolution
    frame: dict[str, object] = {
        JOINT_STATE_KEY: rng.standard_normal(DEFAULT_HARDWARE_MODEL.state_dim, dtype=np.float32),
        # The wire contract requires task and subtask to be set or empty together.
        TASK_KEY: "stack the tower",
        SUBTASK_KEY: "place block",
        MODEL_ID_KEY: "",
    }
    for cam in DEFAULT_HARDWARE_MODEL.cameras:
        rgb = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        frame[f"observation/{cam}"] = simplejpeg.encode_jpeg(rgb, quality=75)
    frame.update(extra)
    return frame


@pytest.mark.asyncio
async def test_rl_round_trip():
    from server import server as srv

    srv._init_policy(CHECKPOINT, POLICY_TYPE)

    agent = RandomLatentAgent((srv.POLICY.config.chunk_size, srv.POLICY.config.max_action_dim), scale=1.0)
    agent_task = asyncio.create_task(serve_agent(agent, port=AGENT_PORT))
    await asyncio.sleep(0.5)

    try:
        await srv._init_rl(f"ws://127.0.0.1:{AGENT_PORT}", resize_image=128)
        assert srv.RL_AGENT is not None

        async with srv.run_example_server(action_horizon=50, task=None) as url:
            client = RemotePolicyClient(url)
            async with client:
                first = await client.predict(_predict_frame(0))
                assert first.rl_enabled is True
                assert first.q_value == 0.0  # RandomLatentAgent reports a constant
                assert first.actions_d.shape == (50, DEFAULT_HARDWARE_MODEL.action_dim)
                # Nothing has been executed yet, so there is no transition to report.
                assert agent.transitions == []

                second = await client.predict(_predict_frame(1), prev_skipped_action_start=17)
                assert second.rl_enabled is True
                assert len(agent.transitions) == 1
                transition = agent.transitions[0]
                assert transition.executed_steps == 17, "the client's truncation point is the discount horizon"
                assert transition.prev_step_id < transition.step_id

                await client.mark_episode_done("success")
                assert agent.episodes == ["success"]

                # A new episode must not be stitched onto the previous one.
                await client.predict(_predict_frame(2))
                assert len(agent.transitions) == 1, "first step of a new episode has no predecessor"
    finally:
        agent_task.cancel()
        await srv._init_rl(None)


@pytest.mark.asyncio
async def test_rl_disabled_by_default():
    """The default path must stay exactly as it was: no rl_enabled, no agent."""
    from server import server as srv

    srv._init_policy(CHECKPOINT, POLICY_TYPE)
    await srv._init_rl(None)
    assert srv.RL_AGENT is None

    async with srv.run_example_server(action_horizon=50, task=None) as url:
        client = RemotePolicyClient(url)
        async with client:
            pred = await client.predict(_predict_frame(0))
            assert pred.rl_enabled is False
            assert pred.actions_d.shape == (50, DEFAULT_HARDWARE_MODEL.action_dim)
