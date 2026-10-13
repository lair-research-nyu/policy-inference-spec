# policy-inference-spec

Shared Python package for the current policy-inference wire format and the helper code used around it: msgpack + NumPy serialization, request/response validation, an async WebSocket client, feature-bundle utilities, an example server, and an offline `.rrd` replay tool.

## Install

Requires Python 3.12+.

```bash
pip install -e .
```

Optional test dependencies:

```bash
pip install -e '.[dev]'
```

Optional debug-tooling dependencies (needed by `scripts/debug_dataset_frame.py`; adds `matplotlib`):

```bash
pip install -e '.[debug]'
```

The base install now includes the dependencies needed by both module CLIs:

- `python -m policy_inference_spec.smoke`
- `python -m policy_inference_spec.replay_rrd`

## Entrypoints

For indexed YOR LeRobot dataset replay with local OpenPI or LeRobot inference,
see [Replay a LeRobot episode](docs/replay_lerobot.md).

### Python module CLIs

`policy_inference_spec.smoke`

- Purpose: send a few predict requests to a policy server.
- CLI: `python -m policy_inference_spec.smoke`
- Behavior:
  - if `--url` is omitted, it starts the in-process example server from `server.minimal`
  - sends randomly generated Gen2-compatible requests
  - optionally forwards an `x-api-key` header
- Flags:
  - `--url`
  - `--predicts`
  - `--api-key`
  - `-v` / `--verbose`

Examples:

```bash
uv run python -m policy_inference_spec.smoke
uv run python -m policy_inference_spec.smoke --url ws://127.0.0.1:18090/ws --predicts 3
```

`policy_inference_spec.replay_rrd`

- Purpose: replay an `.rrd` recording against a remote policy server and write a new `.rrd` containing predictions and error metrics.
- CLI: `python -m policy_inference_spec.replay_rrd`
- Default schema: `gen2-32d-state`
- Current wire mapping:
  - `head` -> `observation/images/main_image`
  - `left_wrist` -> `observation/images/left_wrist_image`
  - `right_wrist` -> `observation/images/right_wrist_image`
- Flags:
  - `--schema`
  - `--recording-path`
  - `--output-path`
  - `--predict-url` or `--recording-server-url`
  - `--policy-id`
  - `--hz`
  - `--prediction-hz`
  - `--max-samples`
  - `--action-prefix-steps`
  - `--prefix-change-start`

Examples:

```bash
uv run python -m policy_inference_spec.replay_rrd \
  --recording-path ~/local_data/recordings/example.rrd \
  --predict-url ws://127.0.0.1:18090/ws \
  --policy-id my-policy

uv run python -m policy_inference_spec.replay_rrd \
  --schema gen2-32d-state-pose-actions \
  --recording-path ~/local_data/recordings/example.rrd \
  --output-path replay-output.rrd \
  --prediction-hz 2.0 \
  --max-samples 100
```

### Importable entrypoints

Core package exports from `policy_inference_spec`:

```python
from policy_inference_spec import (
    DEFAULT_HARDWARE_MODEL,
    DEFAULT_PREDICT_URL,
    RemotePolicyClient,
    ServerFeature,
    ServerHandshake,
    deserialize_from_msgpack,
    encode_image,
    make_server_handshake,
    policy_ws_url,
    serialize_to_msgpack,
    validate_ultra_arrays_for_hardware_model,
    validate_wire_inference_request_frame,
    validate_wire_inference_response,
)
```

Replay helpers are intentionally imported from the submodule, not re-exported from the package root:

```python
from policy_inference_spec.replay_rrd import (
    ReplaySummary,
    RerunReplayer,
    log_to_rerun,
    predict_sample,
    replay_recording,
)
```

Example server helpers live in the sibling `server` package:

```python
from server.minimal import (
    example_policy_actions,
    handle_inference_connection,
    main,
    run_example_server,
    server_handshake_config,
)
```

`server.minimal` is both importable and runnable as a module:

```bash
uv run python -m server.minimal
uv run python -m server.minimal --host 127.0.0.1 --port 18090
uv run python -m server.minimal --host 127.0.0.1 --port 18090 --action-horizon 50
uv run python -m server.minimal --no-rewards
```

The CLI defaults to `--action-horizon 50` so it works with `policy_inference_spec.replay_rrd` out of the box. The importable `run_example_server()` helper still defaults to a shorter horizon of `4` unless you override `action_horizon=...`.

### openpi server

`server.server_openpi` serves checkpoints trained with openpi (JAX) through
[lair-train-pipeline](https://github.com/lair-research-nyu/lair-train-pipeline) and the
[lair-openpi](https://github.com/lair-research-nyu/lair-openpi) fork. It speaks the same
wire protocol as `server.server`, so the robot client does not change. openpi needs
numpy<2 and Python 3.11, and lerobot needs numpy 2, so this server runs in the
lair-openpi venv with this repo on `PYTHONPATH`:

```bash
git clone https://github.com/lair-research-nyu/lair-openpi ~/projects/lair-openpi
cd ~/projects/lair-openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync
uv pip install --python .venv/bin/python msgspec simplejpeg

cd ~/projects/policy-inference-spec
PYTHONPATH=. ~/projects/lair-openpi/.venv/bin/python -m server.server_openpi \
  --checkpoint-path models/openpi-ultra-cubestack-pi0-rtctrain-10000 --rtc-train \
  --host 0.0.0.0 --port 18090
```

- The checkpoint is an openpi step directory (`params/`, `assets/<id>/norm_stats.json`).
  The server reads pi0 vs pi0.5 and the action dim from the weights. It uses
  lair-train-pipeline's camera mapping: head → `base_0_rgb`, left/right wrist → wrist slots.
- `--model-action-horizon` (default 50) and `--rtc-train-min/max-delay` (default
  [0, 15)) must match the training config, because the checkpoint does not store them.
- `--rtc-train`, `--resume-gap-s`, `--task`, `--prompt-source` and the action limiter
  flags work as they do in `server.server`. It does not implement inference-time RTC
  (`--rtc`), latent RL, temporal ensembling or state pins.
- Tests: `PYTHONPATH=. ~/projects/lair-openpi/.venv/bin/python -m pytest tests/test_server_openpi.py`
  also runs the test that loads the real checkpoint.

### DynamicVLA server

`server.server_dynamicvla` serves [DynamicVLA](https://github.com/hzxie/DynamicVLA)
checkpoints (`config.json` with `"type": "dynamicvla"` plus one `model.epochNNNN.safetensors`)
over the same wire protocol. DynamicVLA is written for lerobot 0.3.3 on Python 3.10,
so it runs in its own venv, `.venv-dynamicvla`, and imports the model code from a local
DynamicVLA checkout (`--dynamicvla-root`, default `~/embodied_at_sea_ws/src/DynamicVLA`):

```bash
uv venv --python 3.10 --python-preference only-managed .venv-dynamicvla
VIRTUAL_ENV=.venv-dynamicvla uv pip install -r requirements-dynamicvla.txt --index-strategy unsafe-best-match

PYTHONPATH=. .venv-dynamicvla/bin/python -m server.server_dynamicvla \
  --checkpoint-path models/cubestack-dynamicvla --task "stack cubes" --host 0.0.0.0 --port 18090
```

- The model reads the current frame plus older ones, set by `delta_timestamps.observation`
  in dataset frames. For cubestack that is `[-5, 0]`, i.e. 0.1 s ago at 50 Hz
  (`--dataset-fps`). The server keeps the frames it receives (inference requests, and
  telemetry frames with every camera) and uses the one closest to each offset. If none is
  within `--history-tolerance-s`, it repeats the current frame and logs `history=[MISSING now]`.
- Actions are trained relative to the current state, except the dims whose
  normalization stats show they were absolute. For cubestack those are the two grippers.
  Override the inferred dims with `--absolute-action-dims`.
- There is no RTC. `--task`, `--prompt-source` and the action limiter flags work as they do in `server.server`.

### Debug inference server

`server.debug` is a standalone inference server for robot bring-up and policy debugging. It implements the full wire protocol, but instead of running a policy it produces action chunks from a pluggable `ActionSource`, optionally piping them through `ChunkTransform`s, a `Gate` (pacing), and a `ResponseObserver` (introspection). Nothing in this module loads `lerobot` or a checkpoint — it's safe to boot on a machine without a GPU.

```bash
uv run python -m server.debug --host 127.0.0.1 --port 18090 --source=zeros
```

Pipeline stages, each independently selectable via CLI flags:

1. **Source** (`--source`): produces the `(H, 25)` action chunk.
   - `zeros` — all zeros. Smoke-test source.
   - `rrd` — replay commanded positions from a `.rrd` recording. Needs `--recording-path`, optionally `--schema` (default `gen2-32d-state`) and `--hz` (default `50`). Holds the last chunk when the recording is exhausted. Pose-action schemas are rejected (their action dim is 23, wire requires 25).
   - `goto` — interpolate linearly from the current commanded pose to a target over the first chunk, then hold. Needs `--target-pose-file` (`.npy` with 25 floats). `--max-step-per-joint-rad` (default `0.5`) aborts if the target is farther than that on any joint; set `<=0` to disable.

2. **Transforms** (applied in the order of their flags):
   - `--delete-first N` — drop the first `N` actions and back-pad with `chunk[N]` so the chunk length is unchanged. Useful to skip a policy's warm-up actions.
   - `--set-constant VALUE[,DIM,DIM...]` — pin action dims to `VALUE`. Omit dims to pin every dim; e.g. `--set-constant 0,22,23,24` zeroes only the neck.

3. **Gate** (`--gate`): paces when each response is released.
   - `none` (default) — release immediately.
   - `keypress` — pause each response until Enter is pressed in the server terminal. One press = one release (FIFO across waiting connections). Requires a TTY stdin.

4. **Observer** (`--observer`): side-effect sink called after the chunk is produced.
   - `none` (default) — no-op.
   - `rerun` — spawn a live Rerun viewer (`rr.spawn()`) that logs all three cameras, right-arm state (`state[32:40]`), and the predicted action chunk (25 dims, each row projected forward at `1/hz` per step). Customize with `--rerun-app-id`.

Other flags: `--action-horizon` (default `50`, matches `server.minimal`), `--hz` (control rate shared by `rrd` source and `rerun` observer).

Examples:

```bash
# Step through an .rrd recording chunk-by-chunk while watching it in Rerun
uv run python -m server.debug \
  --source=rrd --recording-path ~/local_data/recordings/example.rrd \
  --gate=keypress --observer=rerun

# Slowly drive the robot to a captured pose, skipping the first 10 warm-up actions
uv run python -m server.debug \
  --source=goto --target-pose-file ~/poses/ready.npy \
  --max-step-per-joint-rad 0.3 --delete-first 10

# Zero-action source with the neck pinned to zero, everything else free
uv run python -m server.debug --source=zeros --set-constant 0,22,23,24
```

Importable entrypoints:

```python
from policy_inference_spec.debug import (
    DeleteAndBackpad,
    GoToPoseSource,
    ImmediateGate,
    KeypressGate,
    RerunObserver,
    RrdActionSource,
    SetConstant,
    ZerosSource,
    run_pipeline,
)
from server.debug import handle_inference_connection, run_debug_server
```

Pipeline protocols (`ActionSource`, `ChunkTransform`, `Gate`, `ResponseObserver`) live in `policy_inference_spec.debug.pipeline`; implement any of them to add a new source/transform/gate/observer.

### Repository helper commands

`policy_inference_spec/Justfile` contains:

- `just test`
- `just smoke ...`
- `just replay-rrd ...`

Run them from the repo root:

```bash
just -f libs/submodules/policy-inference-spec/policy_inference_spec/Justfile test
just -f libs/submodules/policy-inference-spec/policy_inference_spec/Justfile smoke --predicts 3
just -f libs/submodules/policy-inference-spec/policy_inference_spec/Justfile replay-rrd --recording-path ~/local_data/recordings/example.rrd
```

## Current Wire Protocol

### Transport and URL normalization

- Transport is WebSocket binary frames carrying msgpack payloads.
- `policy_ws_url()` accepts `ws://` and `wss://`.
- If the supplied URL has an empty path or `/`, `policy_ws_url()` rewrites it to end in `/ws`.

### Handshake

The server sends the first frame. `ServerHandshake.from_payload()` accepts:

- `camera_names`: required `list[str]`
- `image_resolution`: optional `[height, width]`
- `action_space`: optional `str`, default `"joint_position"`
- `needs_wrist_camera`: optional `bool`, default `True`
- `n_external_cameras`: optional `int`, default `1`
- `server_features`: optional `list[str]`, default `[]`

`ServerHandshake.supports()` currently matters for one feature enum value:

- `ServerFeature.REWARDS` / `"rewards"`

### Current hardware model

Only `HardwareModel.GEN2` is modeled in this package.

- `state_dim = 97`
- `action_dim = 25`
- `image_resolution = (360, 640)`
- cameras:
  - `images/main_image`
  - `images/left_wrist_image`
  - `images/right_wrist_image`

### Inference request validation

`validate_wire_inference_request_frame()` requires these base request keys:

- `observation/state`
- `observation/images/main_image`
- `observation/images/left_wrist_image`
- `observation/images/right_wrist_image`
- `task`
- `subtask`
- `model_id`

Current validation rules:

- `endpoint` must not be present on inference frames.
- `task` must be `str`.
- `subtask` must be `str`.
- `task` and `subtask` are either both non-empty or both empty. `task` is the robot's
  top-level selection and `subtask` the step being executed under it — but when the run
  is driven by a construction assembled in the web UI, both hold the *construction's*
  name (`simple_list`, `generic_subtask`) rather than anything a policy was trained on,
  and the instruction itself rides in `conditioning_metadata` / `start_metadata`.

  `server.server --prompt-source` (default `auto`) decides what the policy is
  conditioned on: `auto` takes the first conventionally-named field
  (`instruction`, `language_instruction`, `prompt`, `message`, `text`, `description`)
  found in those two metadata dicts — the Ultra web UI sends
  `start_metadata={'message': 'stack cubes'}` — else the subtask, else the task; `subtask` / `task` take
  that wire field verbatim; anything else is a dotted path into the frame, e.g.
  `conditioning_metadata.instruction`. `--task` bypasses the frame entirely. The server
  logs the resolved prompt with its origin, and logs each request's non-observation
  metadata whenever it changes, so the right path can be read off the log. An
  instruction found in metadata is held for the rest of the episode:
  `start_metadata` need not repeat on every request, and falling back to the wire
  fields when it is absent would swap the policy's conditioning mid-episode.
- `model_id` must be `str`.
- `observation/state` must be a 1-D `numpy.ndarray` with shape `(97,)`.
- Each image field must be either JPEG `bytes` or a `numpy.ndarray`.
- Extra keys are rejected except the optional request fields below.

Optional request fields:

- `observation/images/table_view_image`: optional station hardware camera. JPEG `bytes` or `numpy.ndarray`. May be omitted when the station has no table-view cam.
- `action_prefix`: floating 2-D `numpy.ndarray` shaped `(prefix_steps, 25)`. It carries only real prefix actions; clients must not pad it to the full policy horizon.
- `prefix_change_start`: non-negative `int`. Required when `action_prefix` is present. In replay tooling this defaults to `--action-prefix-steps` unless `--prefix-change-start` is set.

  Pausing an async rollout (teleop / DAgger interventions) breaks the prefix's meaning: it
  is a promise that those rows are executing right now, and after a pause they never ran
  and a human has moved the arms. The first request after a resume must therefore carry
  **neither** field — that yields an unconditioned chunk planned from the live observation,
  and the client executes it from step 0 rather than skipping `prefix_change_start` steps.
  Chaining resumes on the request after it.

  `server.server` enforces this without the client's cooperation: it ignores the prefix on
  any request that opens a connection, follows an `endpoint=intervention` frame, or arrives
  more than `--resume-gap-s` (default 1 s) after the previous one. It cannot do anything
  about a client that replays the action queue it kept across the pause — those actions
  never reach the server.
- `reward`, `done`, `done_reason`, `prev_skipped_action_start`
- `start_metadata`, `conditioning_metadata`
- `observation/hidden`, `observation/env`
- `dumb_reward_goal_action_chunk` plus `dumb_reward_threshold`: optional debugging/reward fields.
- `fast_mock_action_dim` plus `fast_mock_action_horizon`: optional testing fields.

`RemotePolicyClient.predict()` accepts image arrays in HWC or single-item BHWC `uint8` form and JPEG-encodes them at quality 75 before sending. It does not resize them.

`validate_ultra_arrays_for_hardware_model()` is a separate helper for batched in-memory arrays. For Gen2 it expects:

- keys exactly equal to `observation/state` plus the three camera keys
- `observation/state.shape == (1, 97)`
- image arrays in HWC or single-item BHWC layout

### Inference response validation

`validate_wire_inference_response()` requires:

- `action`: a 2-D floating `numpy.ndarray` whose second dimension is `25`

Optional response fields:

- `chunk_id`: echo the request's `chunk_id` when provided. This lets the client discard
  replies left over from cancelled predictions after pause/resume; omitting it can make
  an old reply appear to belong to the new request.
- `inference_time`: numeric
- `policy_id`: `str`
- `rl_enabled`: `bool`, whether the returned action chunk used the RL actor
- `q_value`: numeric, critic Q estimate for the returned action chunk when available

`RemotePolicyClient.predict()` returns `RemotePolicyPrediction`:

- `actions_d`
- `total_latency_ms`
- `policy_id`
- `rl_enabled`
- `q_value`

### Reward and control messages

- `RemotePolicyClient.predict(..., reward=<float>, done=True, done_reason=<str>)` includes scalar rewards and terminal markers in the inference request when the server handshake advertises `"rewards"`.
- If reward support is not advertised, the client logs a warning and drops the reward.
- `ENDPOINT_DONE` and `ENDPOINT_TELEMETRY` constants exist, and `server.minimal` replies to both with `{"status": "ok"}`.
- `ENDPOINT_DONE` marks the current episode as done on the server (it does not reset any gym environment); `RemotePolicyClient.mark_episode_done(done_reason)` sends it over the live connection.
- There is no dedicated telemetry client helper in this package.

### Msgpack / NumPy codec

- `serialize_to_msgpack()` tags ndarrays with `__ndarray__`, `data`, `dtype`, and `shape`.
- `deserialize_from_msgpack()` reconstructs those arrays.
- The codec supports `uint8` and `float32` ndarrays.
- `encode_image()` returns an `NdarrayField` with JPEG bytes plus metadata, but `RemotePolicyClient` currently places raw JPEG bytes on the wire for image observations.

## Feature-engineering Schemas

`SchemaName` currently exposes four feature bundles:

| Schema | State dim | Action dim |
|--------|-----------|------------|
| `gen2-28d-state` | `89` | `25` |
| `gen2-32d-state` | `97` | `25` |
| `gen2-28d-state-pose-actions` | `89` | `23` |
| `gen2-32d-state-pose-actions` | `97` | `23` |

All four schemas currently use the same video stream names:

- `head`
- `left_wrist`
- `right_wrist`

And the same `camera_stream_schema()` values:

- `head`: `(360, 640, 10)`
- `left_wrist`: `(300, 480, 10)`
- `right_wrist`: `(300, 480, 10)`

`preprocess_image()` is the shared image preprocessing helper used by replay and training/data-prep code. It:

- requires HWC `uint8` RGB input
- optionally crops to the left half when `crop_to_mono=True`
- downsamples by `downsample_factor`
- can also target an explicit output shape

## Included Modules

| Module | Role |
|--------|------|
| `policy_inference_spec.protocol` | Wire keys, handshake/reward dataclasses, server features, and protocol aliases |
| `policy_inference_spec.codec` | Msgpack codec, ndarray tagging, and JPEG encoding helpers |
| `policy_inference_spec.hardware_model` | Current hardware model metadata and strict request/response validators |
| `policy_inference_spec.client` | Async WebSocket client and prediction result type |
| `policy_inference_spec.client_helpers` | URL normalization and logging/payload helpers used by the client |
| `policy_inference_spec.feature_engineering` | Feature-bundle schemas, action parsing, and image preprocessing utilities |
| `policy_inference_spec.smoke` | CLI smoke test for predict requests |
| `policy_inference_spec.replay_rrd` | Importable offline replay utilities plus the `.rrd` replay CLI |
| `server.minimal` | Importable example inference server and example linear policy |
| `server.server` | LeRobot-backed inference server (loads a checkpoint and runs a real policy) |
| `server.debug` | Debug inference server with pluggable action sources, transforms, gates, and observers |
| `policy_inference_spec.debug` | `ActionSource` / `ChunkTransform` / `Gate` / `ResponseObserver` protocols and their built-in implementations |

## Quick Checks

Run tests:

```bash
uv run --extra dev pytest tests/ -q
```

Run the smoke test against the local example server:

```bash
uv run python -m policy_inference_spec.smoke
```

Replay a recording against a server:

```bash
uv run python -m policy_inference_spec.replay_rrd \
  --recording-path ~/local_data/recordings/example.rrd \
  --predict-url ws://127.0.0.1:18090/ws
```

Run the minimal example server in another terminal:

```bash
uv run python -m server.minimal --host 127.0.0.1 --port 18090
```

## License

See `LICENSE`.
