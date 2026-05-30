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
    RewardSignal,
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

`validate_wire_inference_request_frame()` requires the request keys to match exactly:

- `observation/state`
- `observation/images/main_image`
- `observation/images/left_wrist_image`
- `observation/images/right_wrist_image`
- `prompt`
- `model_id`

Current validation rules:

- `endpoint` must not be present on inference frames.
- `prompt` must be `str`.
- `model_id` must be `str`.
- `observation/state` must be a 1-D `numpy.ndarray` with shape `(97,)`.
- Each image field must be either JPEG `bytes` or a `numpy.ndarray`.
- Extra keys are rejected.

`RemotePolicyClient.predict()` accepts image arrays in HWC or single-item BHWC `uint8` form and JPEG-encodes them at quality 75 before sending. It does not resize them.

`validate_ultra_arrays_for_hardware_model()` is a separate helper for batched in-memory arrays. For Gen2 it expects:

- keys exactly equal to `observation/state` plus the three camera keys
- `observation/state.shape == (1, 97)`
- image arrays in HWC or single-item BHWC layout

### Inference response validation

`validate_wire_inference_response()` requires:

- `action`: a 2-D floating `numpy.ndarray` whose second dimension is `25`
- `context_embeddings`: a floating `numpy.ndarray` with shape `(2, 128)`

Optional response fields:

- `inference_time`: numeric
- `policy_id`: `str`

`RemotePolicyClient.predict()` returns `RemotePolicyPrediction`:

- `actions_d`
- `context_embeddings`
- `total_latency_ms`
- `policy_id`

### Reward and control messages

- `RemotePolicyClient.reward()` only sends a reward frame if the server handshake advertises `"rewards"`.
- If reward support is not advertised, the client logs a warning and drops the message.
- `RewardSignal.to_payload()` produces:
  - `{"endpoint": "reward", "reward": <float>}`
  - optionally with `{"description": <str>}`
- The example server acknowledges rewards with `{"endpoint": "reward", "status": "ok", ...}`.
- `ENDPOINT_RESET` and `ENDPOINT_TELEMETRY` constants exist, and `server.minimal` replies to both with `{"status": "ok"}`.
- There are no dedicated reset or telemetry client helpers in this package.

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
