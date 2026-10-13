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
import json
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

                # RandomLatentAgent asks for chunks, so the one just denoised comes back to
                # it. This is the only path on which a recording learner can see what its
                # latent actually produced, so it is worth asserting against the actions the
                # robot was handed rather than merely asserting that something arrived.
                assert len(agent.produced_actions) == 1
                produced = agent.produced_actions[0]
                assert produced.action_horizon == 50
                assert produced.actions.shape[1] == DEFAULT_HARDWARE_MODEL.action_dim
                np.testing.assert_allclose(
                    produced.actions[:50], first.actions_d, rtol=1e-5, atol=1e-5
                )
                # The pre-unnormalization chunk, same shape as the postprocessed one:
                # denoise() drops the policy's padding columns before returning.
                assert produced.raw_actions is not None
                assert produced.raw_actions.shape == produced.actions.shape
                assert not np.allclose(produced.raw_actions, produced.actions), (
                    "raw_actions must be the normalized chunk, not a copy of the robot-space one"
                )

                second = await client.predict(_predict_frame(1), prev_skipped_action_start=17)
                assert second.rl_enabled is True
                assert len(agent.produced_actions) == 2, "one reported chunk per denoise"
                assert len(agent.transitions) == 1
                transition = agent.transitions[0]
                assert transition.executed_steps == 17, "the client's truncation point is the discount horizon"
                assert transition.prev_step_id < transition.step_id

                await client.mark_episode_done(
                    "success", final_wire_frame=_predict_frame(2), executed_steps=9
                )
                assert agent.episodes == ["success"]
                assert len(agent.episode_terminals) == 1
                terminal = agent.episode_terminals[0]
                assert terminal is not None
                terminal_step_id, terminal_observation, terminal_executed_steps = terminal
                assert terminal_step_id > transition.step_id
                assert terminal_executed_steps == 9
                # The subtask, not the task: with --prompt-source auto the server
                # conditions on the more specific of the two strings the client sends.
                assert terminal_observation.task == "place block"
                assert terminal_observation.state.shape == (DEFAULT_HARDWARE_MODEL.action_dim,)

                # A new episode must not be stitched onto the previous one.
                await client.predict(_predict_frame(3))
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


@pytest.mark.asyncio
async def test_rl_run_dumps_the_wire_jpegs_by_default(tmp_path, monkeypatch):
    """An RL run launched with no dump flag still leaves images an offline sweep can use.

    The assertion that matters is byte equality: what lands on disk must be the JPEG the
    client put on the wire, not a re-encode of it. A dump that differs from the policy's
    own input is worth nothing -- resampling latents on it measures the difference.
    """
    from server import server as srv

    monkeypatch.setattr(srv, "DEFAULT_RL_OBS_DUMP_ROOT", tmp_path)
    args = srv._parse_args([
        "--checkpoint-path", str(CHECKPOINT), "--policy-type", POLICY_TYPE,
        "--rl-agent-url", f"ws://127.0.0.1:{AGENT_PORT + 1}",
    ])
    dump_dir = Path(srv._obs_dump_dir_from_args(args))
    assert dump_dir.parent == tmp_path, "the default must not be the caller's cwd or a fixed path"

    srv._init_policy(CHECKPOINT, POLICY_TYPE)
    agent = RandomLatentAgent((srv.POLICY.config.chunk_size, srv.POLICY.config.max_action_dim), scale=1.0)
    agent_task = asyncio.create_task(serve_agent(agent, port=AGENT_PORT + 1))
    await asyncio.sleep(0.5)

    frame = _predict_frame(7)
    try:
        await srv._init_rl(args.rl_agent_url, resize_image=args.rl_resize_image,
                           obs_dump_dir=str(dump_dir))
        async with srv.run_example_server(action_horizon=50, task=None) as url:
            client = RemotePolicyClient(url)
            async with client:
                await client.predict(frame)
        srv.RL_OBS_DUMP.close()  # flush the background writer, as main() does on exit

        index = [json.loads(line) for line in (dump_dir / "steps.jsonl").read_text().splitlines()]
        assert len(index) == 1, "one indexed observation per RL query"
        assert set(index[0]["files"]) == {"head", "left_wrist", "right_wrist"}
        for cam, name in index[0]["files"].items():
            wire = next(k for k, v in srv._CAMERA_KEY_MAP.items() if v.endswith(f".{cam}"))
            assert (dump_dir / "images" / name).read_bytes() == frame[wire]
        # The join key the viewer uses: an episode's step_ids index straight into this.
        assert index[0]["step_id"] == agent.produced_actions[0].step_id
        assert json.loads((dump_dir / "manifest.json").read_text())["checkpoint"].endswith(
            CHECKPOINT.name)
    finally:
        agent_task.cancel()
        srv.RL_OBS_DUMP = None
        await srv._init_rl(None)
