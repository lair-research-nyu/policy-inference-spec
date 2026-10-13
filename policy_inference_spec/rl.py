"""Control API for latent-space real-world RL.

Two narrow protocols sit between a policy and an RL method:

* ``LatentPolicyBackend`` — what a policy must offer to be RL-drivable. Flow-matching
  policies (pi0 / pi05) qualify: their sampler starts from a noise latent that a caller
  can supply, so the latent *is* the RL action. Non-generative policies (ACT) cannot.
* ``RLAgent`` — what an RL method must offer to drive a policy. DSRL is one
  implementation; HIL-SERL, RLPD, or a torch port plug in the same way.

The inference server owns the loop and calls the agent between the policy's prefix pass
and its denoise loop:

    prefix_rep = backend.prefix(obs)          # expensive VLM pass, once
    z          = await agent.act(...)         # RL picks the latent
    actions    = backend.denoise(z)           # cheap, reuses the prefix KV cache

Agents may live in-process (``NullAgent``) or in a separate process (``RemoteAgent``),
which matters when the RL method and the policy use different array frameworks.

Wire format for the remote link is msgpack over WebSocket, reusing
:mod:`policy_inference_spec.codec`. It is deliberately *not* the robot wire protocol in
:mod:`policy_inference_spec.protocol`: the robot never sees latents or prefix reps.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt
import websockets

from policy_inference_spec.client_helpers import policy_ws_url
from policy_inference_spec.codec import deserialize_from_msgpack, serialize_to_msgpack
from policy_inference_spec.protocol import FloatArray, ImageArray, ProtocolPayload

LOGGER = logging.getLogger(__name__)

DEFAULT_AGENT_PORT = 18091

# Wire method names.
METHOD_KEY = "method"
METHOD_ACT = "act"
METHOD_OBSERVE = "observe"
METHOD_EPISODE_END = "episode_end"
METHOD_ACTIONS = "actions_taken"

# Agent handshake keys.
LATENT_SHAPE_KEY = "latent_shape"
AGENT_ID_KEY = "agent_id"
WANTS_ACTIONS_KEY = "wants_actions"

# Message payload keys.
STEP_ID_KEY = "step_id"
PREV_STEP_ID_KEY = "prev_step_id"
OBSERVATION_KEY = "observation"
IMAGES_KEY = "images"
STATE_KEY = "state"
PREFIX_REP_KEY = "prefix_rep"
TASK_KEY = "task"
LATENT_KEY = "latent"
Q_VALUE_KEY = "q_value"
EXECUTED_STEPS_KEY = "executed_steps"
REWARD_KEY = "reward"
DONE_KEY = "done"
REASON_KEY = "reason"
ACTIONS_KEY = "actions"
RAW_ACTIONS_KEY = "raw_actions"
ACTION_HORIZON_KEY = "action_horizon"


@dataclass(frozen=True)
class AgentObservation:
    """What the RL agent sees at one decision point.

    ``prefix_rep`` is the policy's own visual-language embedding of this observation —
    for pi0 the last token of the PaliGemma prefix. Feeding it to the critic is what
    lets a small RL network reason about a scene it could never encode itself.
    """

    images: dict[str, ImageArray]
    state: FloatArray
    prefix_rep: FloatArray
    task: str

    def __post_init__(self) -> None:
        assert self.images, "images must not be empty"
        for name, image in self.images.items():
            assert image.ndim == 3 and image.shape[-1] == 3, (
                f"image {name!r} must be HWC RGB, got shape {image.shape}"
            )
            assert image.dtype == np.uint8, f"image {name!r} must be uint8, got {image.dtype}"
        assert self.state.ndim == 1, f"state must be 1-D, got shape {self.state.shape}"
        assert self.prefix_rep.ndim == 1, f"prefix_rep must be 1-D, got shape {self.prefix_rep.shape}"

    def to_payload(self) -> ProtocolPayload:
        return {
            IMAGES_KEY: {name: np.ascontiguousarray(image) for name, image in self.images.items()},
            STATE_KEY: np.ascontiguousarray(self.state, dtype=np.float32),
            PREFIX_REP_KEY: np.ascontiguousarray(self.prefix_rep, dtype=np.float32),
            TASK_KEY: self.task,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> AgentObservation:
        images_raw = payload[IMAGES_KEY]
        assert isinstance(images_raw, dict), f"{IMAGES_KEY} must be a dict, got {type(images_raw)}"
        task = payload[TASK_KEY]
        assert isinstance(task, str), f"{TASK_KEY} must be str, got {type(task)}"
        return cls(
            images={str(name): np.asarray(image, dtype=np.uint8) for name, image in images_raw.items()},
            state=np.asarray(payload[STATE_KEY], dtype=np.float32),
            prefix_rep=np.asarray(payload[PREFIX_REP_KEY], dtype=np.float32),
            task=task,
        )


@dataclass(frozen=True)
class AgentAction:
    """The latent the agent wants denoised, plus optional critic introspection."""

    latent: FloatArray
    q_value: float | None = None

    def __post_init__(self) -> None:
        assert self.latent.ndim == 2, f"latent must be (chunk, dim), got shape {self.latent.shape}"

    def to_payload(self) -> ProtocolPayload:
        payload: ProtocolPayload = {LATENT_KEY: np.ascontiguousarray(self.latent, dtype=np.float32)}
        if self.q_value is not None:
            payload[Q_VALUE_KEY] = float(self.q_value)
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> AgentAction:
        q_raw = payload.get(Q_VALUE_KEY)
        assert q_raw is None or isinstance(q_raw, (int, float)), f"{Q_VALUE_KEY} must be numeric"
        return cls(
            latent=np.asarray(payload[LATENT_KEY], dtype=np.float32),
            q_value=None if q_raw is None else float(q_raw),
        )


@dataclass(frozen=True)
class ProducedActions:
    """The action chunk the policy actually produced from the agent's latent.

    The RL loop does not need this: the agent chose the latent, and the transition that
    follows tells it what the robot did with the resulting chunk. It exists so a run can
    be *reconstructed* offline. The latent -> action map is the one link in the chain
    that lives entirely inside the server — a recorded episode without it can say which
    noise the actor picked but never what motion that noise turned into, which is exactly
    the question a steerability post-mortem asks.

    ``actions`` is the full denoised chunk in robot action space, before truncation to the
    client's ``action_horizon``. ``raw_actions`` is the same chunk one step earlier: as the
    denoiser emitted it, still normalized. It is already trimmed to the robot's action dim
    rather than padded to the policy's -- ``LatentPolicyBackend.denoise`` drops the padding
    columns -- so the two differ in scale alone. Both are kept because the normalizer lives
    in a checkpoint directory a later analysis may no longer have, and re-deriving it wrongly
    is silent: every chunk simply comes out at a plausible wrong scale.

    Sent only when the agent's handshake sets ``wants_actions``; see
    :meth:`RemoteAgent.actions_taken`.
    """

    step_id: int
    actions: FloatArray
    raw_actions: FloatArray | None = None
    action_horizon: int | None = None

    def __post_init__(self) -> None:
        assert self.actions.ndim == 2, f"actions must be (chunk, dim), got shape {self.actions.shape}"
        assert self.raw_actions is None or self.raw_actions.ndim == 2, (
            f"raw_actions must be (chunk, dim), got shape {None if self.raw_actions is None else self.raw_actions.shape}"
        )
        assert self.action_horizon is None or self.action_horizon >= 1, (
            f"action_horizon must be positive, got {self.action_horizon}"
        )

    def to_payload(self) -> ProtocolPayload:
        payload: ProtocolPayload = {
            STEP_ID_KEY: int(self.step_id),
            ACTIONS_KEY: np.ascontiguousarray(self.actions, dtype=np.float32),
        }
        if self.raw_actions is not None:
            payload[RAW_ACTIONS_KEY] = np.ascontiguousarray(self.raw_actions, dtype=np.float32)
        if self.action_horizon is not None:
            payload[ACTION_HORIZON_KEY] = int(self.action_horizon)
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> ProducedActions:
        raw = payload.get(RAW_ACTIONS_KEY)
        horizon = payload.get(ACTION_HORIZON_KEY)
        return cls(
            step_id=int(payload[STEP_ID_KEY]),
            actions=np.asarray(payload[ACTIONS_KEY], dtype=np.float32),
            raw_actions=None if raw is None else np.asarray(raw, dtype=np.float32),
            action_horizon=None if horizon is None else int(horizon),
        )


@dataclass(frozen=True)
class Transition:
    """One RL step: what happened between two consecutive ``act`` calls.

    Observations are referenced by id rather than resent — every step passed through
    ``act`` already, so the agent has both endpoints cached and we keep images off the
    wire a second time.

    ``executed_steps`` is how many actions of the previous chunk the robot actually ran
    before this decision point, which is the true discount horizon for this transition.
    It varies when the client truncates a chunk, so it must be measured, not assumed.
    """

    step_id: int
    prev_step_id: int
    executed_steps: int
    reward: float | None = None
    done: bool = False

    def __post_init__(self) -> None:
        assert self.executed_steps >= 1, f"executed_steps must be positive, got {self.executed_steps}"
        assert self.step_id != self.prev_step_id, "a transition must connect two distinct steps"

    def to_payload(self) -> ProtocolPayload:
        payload: ProtocolPayload = {
            STEP_ID_KEY: int(self.step_id),
            PREV_STEP_ID_KEY: int(self.prev_step_id),
            EXECUTED_STEPS_KEY: int(self.executed_steps),
            DONE_KEY: bool(self.done),
        }
        if self.reward is not None:
            payload[REWARD_KEY] = float(self.reward)
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Transition:
        reward_raw = payload.get(REWARD_KEY)
        assert reward_raw is None or isinstance(reward_raw, (int, float)), f"{REWARD_KEY} must be numeric"
        return cls(
            step_id=int(payload[STEP_ID_KEY]),
            prev_step_id=int(payload[PREV_STEP_ID_KEY]),
            executed_steps=int(payload[EXECUTED_STEPS_KEY]),
            reward=None if reward_raw is None else float(reward_raw),
            done=bool(payload.get(DONE_KEY, False)),
        )


@dataclass(frozen=True)
class AgentHandshake:
    """First frame the agent sends, so the server can reject a shape mismatch loudly
    instead of feeding the denoiser a wrongly shaped latent.

    ``wants_actions`` is how a recording learner opts in to :class:`ProducedActions`.
    Negotiating it here rather than always sending keeps the extension free in both
    directions: an agent written before it simply omits the key and defaults to off, and
    a server written before it ignores the key and never sends, which a new agent sees as
    "no actions recorded" rather than as a hang.
    """

    latent_shape: tuple[int, int]
    agent_id: str = ""
    wants_actions: bool = False

    def __post_init__(self) -> None:
        assert len(self.latent_shape) == 2, f"latent_shape must be 2-D, got {self.latent_shape}"
        assert all(dim > 0 for dim in self.latent_shape), f"latent_shape dims must be positive, got {self.latent_shape}"

    def to_payload(self) -> ProtocolPayload:
        return {
            LATENT_SHAPE_KEY: list(self.latent_shape),
            AGENT_ID_KEY: self.agent_id,
            WANTS_ACTIONS_KEY: bool(self.wants_actions),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> AgentHandshake:
        raw = payload[LATENT_SHAPE_KEY]
        assert isinstance(raw, (list, tuple)) and len(raw) == 2, f"{LATENT_SHAPE_KEY} must be a 2-item list"
        agent_id = payload.get(AGENT_ID_KEY, "")
        assert isinstance(agent_id, str), f"{AGENT_ID_KEY} must be str"
        return cls(
            latent_shape=(int(raw[0]), int(raw[1])),
            agent_id=agent_id,
            wants_actions=bool(payload.get(WANTS_ACTIONS_KEY, False)),
        )


@runtime_checkable
class LatentPolicyBackend(Protocol):
    """A policy whose sampler can start from a caller-supplied latent."""

    @property
    def latent_shape(self) -> tuple[int, int]:
        """(chunk_size, latent_dim) of the noise the denoiser expects."""

    @property
    def prefix_width(self) -> int:
        """Width of the embedding returned by :meth:`prefix`."""

    def prefix(self, batch: dict[str, Any]) -> tuple[FloatArray, Any]:
        """Run the expensive conditioning pass once.

        Returns ``(prefix_rep, cache)`` where ``prefix_rep`` is (B, prefix_width) and
        ``cache`` is an opaque handle to be passed straight back to :meth:`denoise`.
        """

    def denoise(self, cache: Any, noise: FloatArray) -> FloatArray:
        """Integrate the flow from ``noise``, reusing ``cache``. Returns (B, T, action_dim)."""


@runtime_checkable
class RLAgent(Protocol):
    """An RL method that chooses latents and learns from what they produced.

    All methods are async because the inference server calls them from its event loop:
    a blocking call there stalls WebSocket keepalives and drops the robot connection.

    **These members are the whole requirement, and adding one is a breaking change.** This
    package runs under ``beartype_this_package()``, so this ``runtime_checkable`` Protocol
    is enforced on every call annotated with it, ``serve_agent`` included. A member added
    here does not degrade gracefully for agents that predate it: it rejects them at the
    door, and those agents live in other repositories.

    That is why the action-recording extension is *not* here. An agent opts in by defining
    two members of its own::

        @property
        def wants_actions(self) -> bool: ...          # True to receive chunks
        async def actions_taken(self, produced: ProducedActions) -> None: ...

    Both are looked up with ``getattr``, so an agent that defines neither is served exactly
    as it was before the extension existed. See :class:`ProducedActions`.
    """

    async def act(self, step_id: int, observation: AgentObservation) -> AgentAction: ...

    async def observe(self, transition: Transition) -> None: ...

    async def episode_end(
        self,
        reason: str,
        *,
        final_step_id: int | None = None,
        final_observation: AgentObservation | None = None,
        executed_steps: int | None = None,
    ) -> None:
        """Finish an episode, optionally reporting the final executed chunk.

        The optional fields are all-or-none. They close the transition from the most
        recent :meth:`act` call to the robot's observation at episode end.
        """

    async def connection_closed(self) -> None:
        """The server went away mid-episode; discard anything in flight.

        Required because step ids are only unique within a connection — a restarted
        server counts from zero again, so a stale half-episode would silently collide
        with the new one. There is also no way to label an episode that ended in a
        disconnect, so its data cannot be used.
        """

    @property
    def latent_shape(self) -> tuple[int, int]: ...


class NullAgent:
    """Standard Gaussian latents — i.e. exactly what the policy does on its own.

    This is the default so that enabling the RL code path cannot change inference
    behavior until an actual agent is attached.
    """

    def __init__(self, latent_shape: tuple[int, int], seed: int = 0) -> None:
        self._latent_shape = latent_shape
        self._rng = np.random.default_rng(seed)

    @property
    def latent_shape(self) -> tuple[int, int]:
        return self._latent_shape

    async def act(self, step_id: int, observation: AgentObservation) -> AgentAction:
        return AgentAction(latent=self._rng.standard_normal(self._latent_shape, dtype=np.float32))

    async def observe(self, transition: Transition) -> None:
        return None

    async def actions_taken(self, produced: ProducedActions) -> None:
        return None

    @property
    def wants_actions(self) -> bool:
        return False

    async def episode_end(
        self,
        reason: str,
        *,
        final_step_id: int | None = None,
        final_observation: AgentObservation | None = None,
        executed_steps: int | None = None,
    ) -> None:
        return None

    async def connection_closed(self) -> None:
        return None


class RandomLatentAgent(NullAgent):
    """Scaled Gaussian latents, and it logs everything it is told.

    For exercising the full RL code path — transitions, episode boundaries, q_values —
    without loading a learning framework.
    """

    def __init__(self, latent_shape: tuple[int, int], scale: float = 1.0, seed: int = 0) -> None:
        super().__init__(latent_shape, seed=seed)
        assert scale > 0, f"scale must be positive, got {scale}"
        self._scale = scale
        self.transitions: list[Transition] = []
        self.produced_actions: list[ProducedActions] = []
        self.episodes: list[str] = []
        self.episode_terminals: list[tuple[int, AgentObservation, int] | None] = []
        self.disconnects = 0

    async def act(self, step_id: int, observation: AgentObservation) -> AgentAction:
        action = await super().act(step_id, observation)
        LOGGER.info(
            "act step=%d cameras=%s state_dim=%d prefix_dim=%d task=%r",
            step_id, sorted(observation.images), observation.state.size, observation.prefix_rep.size, observation.task,
        )
        return AgentAction(latent=action.latent * self._scale, q_value=0.0)

    async def observe(self, transition: Transition) -> None:
        LOGGER.info(
            "observe step=%d<-%d executed=%d reward=%s done=%s",
            transition.step_id, transition.prev_step_id, transition.executed_steps,
            transition.reward, transition.done,
        )
        self.transitions.append(transition)

    async def actions_taken(self, produced: ProducedActions) -> None:
        LOGGER.info(
            "actions_taken step=%d actions=%s raw=%s horizon=%s",
            produced.step_id, produced.actions.shape,
            None if produced.raw_actions is None else produced.raw_actions.shape,
            produced.action_horizon,
        )
        self.produced_actions.append(produced)

    @property
    def wants_actions(self) -> bool:
        return True

    async def episode_end(
        self,
        reason: str,
        *,
        final_step_id: int | None = None,
        final_observation: AgentObservation | None = None,
        executed_steps: int | None = None,
    ) -> None:
        supplied = (final_step_id is not None, final_observation is not None, executed_steps is not None)
        assert all(supplied) or not any(supplied), "terminal telemetry fields must be supplied together"
        LOGGER.info("episode_end reason=%r after %d transitions", reason, len(self.transitions))
        self.episodes.append(reason)
        if all(supplied):
            assert final_step_id is not None and final_observation is not None and executed_steps is not None
            assert executed_steps >= 1, f"executed_steps must be positive, got {executed_steps}"
            self.episode_terminals.append((final_step_id, final_observation, executed_steps))
        else:
            self.episode_terminals.append(None)

    async def connection_closed(self) -> None:
        LOGGER.info("connection closed")
        self.disconnects += 1


class RemoteAgent:
    """Proxies ``RLAgent`` to a learner process over WebSocket.

    Lets the learner run in its own interpreter — different array framework, different
    GPU, its own terminal for operator input — while the server stays a torch process.
    """

    def __init__(self, url: str, *, connect_timeout_s: float = 60.0) -> None:
        self._url = policy_ws_url(url)
        self._connect_timeout_s = connect_timeout_s
        self._ws: Any = None
        self._handshake: AgentHandshake | None = None
        self._lock = asyncio.Lock()

    @property
    def url(self) -> str:
        return self._url

    @property
    def latent_shape(self) -> tuple[int, int]:
        assert self._handshake is not None, "call connect() before reading latent_shape"
        return self._handshake.latent_shape

    async def connect(self) -> AgentHandshake:
        if self._handshake is not None:
            return self._handshake
        LOGGER.info("Connecting to RL agent at %s", self._url)
        self._ws = await asyncio.wait_for(
            websockets.connect(self._url, compression=None, max_size=None),
            timeout=self._connect_timeout_s,
        )
        first = await self._ws.recv()
        assert isinstance(first, bytes), f"expected binary handshake, got {type(first)}"
        self._handshake = AgentHandshake.from_payload(deserialize_from_msgpack(first))
        LOGGER.info(
            "RL agent connected: agent_id=%r latent_shape=%s",
            self._handshake.agent_id, self._handshake.latent_shape,
        )
        return self._handshake

    async def aclose(self) -> None:
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        self._handshake = None

    async def _call(self, payload: ProtocolPayload) -> dict[str, Any]:
        await self.connect()
        async with self._lock:
            assert self._ws is not None
            await self._ws.send(serialize_to_msgpack(payload))
            response_raw = await self._ws.recv()
        if isinstance(response_raw, str):
            raise RuntimeError(f"Error in RL agent:\n{response_raw}")
        result = deserialize_from_msgpack(response_raw)
        assert isinstance(result, dict), f"unexpected agent response type {type(result)}"
        assert "error" not in result, f"RL agent error: {result['error']}"
        return result

    async def act(self, step_id: int, observation: AgentObservation) -> AgentAction:
        result = await self._call({
            METHOD_KEY: METHOD_ACT,
            STEP_ID_KEY: int(step_id),
            OBSERVATION_KEY: observation.to_payload(),
        })
        action = AgentAction.from_payload(result)
        assert action.latent.shape == self.latent_shape, (
            f"agent returned latent {action.latent.shape}, handshake promised {self.latent_shape}"
        )
        return action

    async def observe(self, transition: Transition) -> None:
        await self._call({METHOD_KEY: METHOD_OBSERVE, **transition.to_payload()})

    @property
    def wants_actions(self) -> bool:
        assert self._handshake is not None, "call connect() before reading wants_actions"
        return self._handshake.wants_actions

    async def actions_taken(self, produced: ProducedActions) -> None:
        """Send the denoised chunk, but only to a learner that asked for it.

        Gated on the handshake so this costs nothing for agents that do not record: the
        chunk is a few tens of KB per query and every send is a round trip taken on the
        same lock ``act`` uses, i.e. it lands squarely in the robot's control loop.
        """
        await self.connect()
        assert self._handshake is not None
        if not self._handshake.wants_actions:
            return
        await self._call({METHOD_KEY: METHOD_ACTIONS, **produced.to_payload()})

    async def episode_end(
        self,
        reason: str,
        *,
        final_step_id: int | None = None,
        final_observation: AgentObservation | None = None,
        executed_steps: int | None = None,
    ) -> None:
        supplied = (final_step_id is not None, final_observation is not None, executed_steps is not None)
        assert all(supplied) or not any(supplied), "terminal telemetry fields must be supplied together"
        payload: ProtocolPayload = {METHOD_KEY: METHOD_EPISODE_END, REASON_KEY: reason}
        if all(supplied):
            assert final_step_id is not None and final_observation is not None and executed_steps is not None
            assert executed_steps >= 1, f"executed_steps must be positive, got {executed_steps}"
            payload.update({
                STEP_ID_KEY: int(final_step_id),
                OBSERVATION_KEY: final_observation.to_payload(),
                EXECUTED_STEPS_KEY: int(executed_steps),
            })
        await self._call(payload)


async def handle_agent_connection(connection: Any, agent: RLAgent) -> None:
    """Server side of the agent link: dispatch wire messages onto ``agent``.

    A learner wraps its RL implementation and calls this; see
    ``dsrl_pi0/examples/train_agent_server.py``.
    """
    await connection.send(
        serialize_to_msgpack(
            AgentHandshake(
                latent_shape=agent.latent_shape,
                agent_id=type(agent).__name__,
                # getattr, not agent.wants_actions: agents written against the earlier
                # protocol define neither member and must keep working untouched.
                wants_actions=bool(getattr(agent, "wants_actions", False)),
            ).to_payload()
        )
    )
    try:
        async for message in connection:
            assert isinstance(message, bytes), f"expected binary frame, got {type(message)}"
            payload = deserialize_from_msgpack(message)
            method = payload.get(METHOD_KEY)
            if method == METHOD_ACT:
                observation = AgentObservation.from_payload(payload[OBSERVATION_KEY])
                action = await agent.act(int(payload[STEP_ID_KEY]), observation)
                await connection.send(serialize_to_msgpack(action.to_payload()))
            elif method == METHOD_OBSERVE:
                await agent.observe(Transition.from_payload(payload))
                await connection.send(serialize_to_msgpack({"status": "ok"}))
            elif method == METHOD_ACTIONS:
                handler = getattr(agent, "actions_taken", None)
                if handler is not None:
                    await handler(ProducedActions.from_payload(payload))
                await connection.send(serialize_to_msgpack({"status": "ok"}))
            elif method == METHOD_EPISODE_END:
                reason = payload.get(REASON_KEY, "")
                assert isinstance(reason, str), f"{REASON_KEY} must be str"
                terminal_fields = (
                    STEP_ID_KEY in payload,
                    OBSERVATION_KEY in payload,
                    EXECUTED_STEPS_KEY in payload,
                )
                assert all(terminal_fields) or not any(terminal_fields), (
                    "terminal telemetry fields must be supplied together"
                )
                if all(terminal_fields):
                    executed_steps = int(payload[EXECUTED_STEPS_KEY])
                    assert executed_steps >= 1, (
                        f"{EXECUTED_STEPS_KEY} must be positive, got {executed_steps}"
                    )
                    await agent.episode_end(
                        reason,
                        final_step_id=int(payload[STEP_ID_KEY]),
                        final_observation=AgentObservation.from_payload(payload[OBSERVATION_KEY]),
                        executed_steps=executed_steps,
                    )
                else:
                    await agent.episode_end(reason)
                await connection.send(serialize_to_msgpack({"status": "ok"}))
            else:
                await connection.send(serialize_to_msgpack({"error": f"unknown method {method!r}"}))
    except websockets.ConnectionClosed:
        # A server restart must not take the learner down with it — it holds all the
        # training state, which is far more expensive than the connection.
        LOGGER.warning("Inference server disconnected")
    finally:
        await agent.connection_closed()


async def serve_agent(agent: RLAgent, *, host: str = "127.0.0.1", port: int = DEFAULT_AGENT_PORT) -> None:
    """Serve ``agent`` forever on ``host:port``.

    Keepalive pings are disabled: a learner legitimately blocks for a long time while
    running gradient steps between episodes, and that must not look like a dead peer.
    """

    async def handler(connection: Any) -> None:
        await handle_agent_connection(connection, agent)

    async with websockets.serve(handler, host, port, compression=None, max_size=None, ping_interval=None):
        LOGGER.info("RL agent listening on ws://%s:%d", host, port)
        await asyncio.Future()


__all__ = [
    "AgentAction",
    "AgentHandshake",
    "AgentObservation",
    "DEFAULT_AGENT_PORT",
    "LatentPolicyBackend",
    "NullAgent",
    "ProducedActions",
    "RLAgent",
    "RandomLatentAgent",
    "RemoteAgent",
    "Transition",
    "handle_agent_connection",
    "serve_agent",
]
