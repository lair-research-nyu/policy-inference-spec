"""Offline YOR LeRobot v3 episode evaluation with the inference helpers in ~/icl_ws.

Run this file directly with the selected backend's Python environment. Keeping
imports independent of this package also supports the OpenPI Python 3.11 venv.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)
CAMERAS = ("zed", "fish0", "fish1")
DEFAULT_DATASET = Path("~/projects/ultra-tools/data/icl-dataset")
DEFAULT_WORKSPACE = Path("~/icl_ws")
JOINT_STATE_CONFIG = "yor_icl_pi05_easy_pnp_v2_absolute_joint_sanity15k"
ALIGNED_Q_CONFIG = "yor_icl_pi05_aligned_q_sanity15k"
DELTA_ROT6D_CONFIG = "yor_icl_pi05_aligned_deltarot6d_sanity15k"
CANONICAL_CONFIG = "yor_icl_pi05_canonical_sanity15k"
YOR_Q_TARGET_NAMES = [f"{arm}j{i}" for arm in ("l", "r") for i in range(7)]
YOR_STATE_NAMES = [*YOR_Q_TARGET_NAMES, "lift"]
OPENPI_PRESETS = {
    name: "yor_icl_pi05_expanded"
    for name in (
        "pi05_extended_quantilesfixed_full_40k",
        "pi05_extended_quantilesfixed_2xbatch",
        "yor_icl_pi05_expanded_continued_50k_full",
    )
}
YOR_ACTION_NAMES = [
    *[f"{arm}_ee.{dim}" for arm in ("left", "right") for dim in ("qw", "qx", "qy", "qz", "x", "y", "z")],
    "gripper.left_gripper", "gripper.right_gripper",
    "base_vel.vx", "base_vel.vy", "base_vel.omega", "lift_cmd.lift_cmd",
]
# The delta checkpoint's EE-pose proprioception, and its per-arm output layout.
YOR_EE_NAMES = YOR_ACTION_NAMES[:14]
YOR_DELTA_ROT6D_NAMES = [
    name for arm in ("left", "right") for name in (
        *[f"{arm}_ee.d{axis}" for axis in ("x", "y", "z")],
        *[f"{arm}_ee.rot6d{i}" for i in range(6)],
        f"gripper.{arm}_gripper",
    )
]
# The canonical checkpoint's 18D proprioception: the same absolute EE poses, with each
# arm's orientation as rot6d instead of a quaternion, and no separate position block.
YOR_CANONICAL_STATE_NAMES = [
    name for arm in ("left", "right") for name in (
        *[f"{arm}_ee.{axis}" for axis in ("x", "y", "z")],
        *[f"{arm}_ee.rot6d{i}" for i in range(6)],
    )
]
# Targets whose model input is absolute EE proprioception rather than the joint state,
# and whose output is defined against each predicted chunk's own first frame.
EE_STATE_TARGETS = ("delta_rot6d", "canonical")


def quaternion_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """(..., 4) wxyz quaternions -> (..., 6) rot6d, the first two ROWS of the rotation matrix.

    Rows, not the columns of Zhou et al.'s original 6D parameterization: reproducing this
    checkpoint's own norm stats needs rows (max deviation 0.01 on every rot6d dimension,
    against 1.0+ for columns or for reading the quaternion scalar-last). That makes the
    stored 6D the transpose of what inference_server_openpi.py's _rot6d_to_matrix assumes,
    so rot6d_quaternion below transposes on the way back.
    """
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    w, x, y, z = (quat[..., i] for i in range(4))
    row0 = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=-1)
    row1 = np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=-1)
    return np.concatenate([row0, row1], axis=-1)


def rot6d_quaternion(rot6d: np.ndarray) -> np.ndarray:
    """(..., 6) rot6d -> (..., 4) wxyz quaternions; the inverse of quaternion_rot6d.

    Gram-Schmidt as in Zhou et al., then treat the orthonormal triple as the matrix's ROWS.
    """
    from scipy.spatial.transform import Rotation

    rot6d = np.asarray(rot6d, dtype=np.float64)
    a1, a2 = rot6d[..., :3], rot6d[..., 3:]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    rows = np.stack([b1, b2, np.cross(b1, b2)], axis=-2)
    return Rotation.from_matrix(rows).as_quat(scalar_first=True)


def canonical_state(poses: np.ndarray) -> np.ndarray:
    """(N, 14) EE observation [qw,qx,qy,qz,x,y,z] x 2 -> (N, 18) canonical state.

    Per arm: absolute position, then absolute orientation as rot6d. Verified exactly
    against lair-nyu/icl-canonical-sanity15k-proprio-viz's own `state_canonical` column
    for every frame it publishes (positions bit-identical, rot6d to 3e-8), which also
    confirms that checkpoint's rot6d is quaternion_rot6d's ROW convention and not the
    column one (deviation 1.7 there). Unlike the quaternion state it replaces, this is
    hemisphere-invariant, since R(q) == R(-q).
    """
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != len(YOR_EE_NAMES):
        raise ValueError(f"Canonical state needs the {len(YOR_EE_NAMES)}D EE observation")
    state = np.zeros((len(poses), len(YOR_CANONICAL_STATE_NAMES)), dtype=np.float64)
    for pose, out in ((0, 0), (7, 9)):
        state[:, out:out + 3] = poses[:, pose + 4:pose + 7]
        state[:, out + 3:out + 9] = quaternion_rot6d(poses[:, pose:pose + 4])
    return state


def delta_rot6d_targets(actions: np.ndarray) -> np.ndarray:
    """Recorded 20D EE actions from a window's first frame -> that window's training target.

    Per arm: position delta against the window's own first commanded pose, absolute
    orientation as rot6d, and the recorded gripper. Confirmed against this checkpoint's
    norm stats: anchoring on the commanded pose reproduces the recorded per-dimension
    means, while anchoring on the FK'd observation leaves the two streams' documented
    rigid Z offset (~0.05 m in the mean) in the deltas.

    The canonical checkpoint shares this exact output space, so both targets are built
    here. Its published `action_canonical_abs` column reproduces this function's absolute
    inputs dimension for dimension (positions bit-identical, rot6d to 3e-8, grippers
    bit-identical), and its README documents the same window-start position anchoring.
    """
    actions = np.asarray(actions, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != len(YOR_ACTION_NAMES):
        raise ValueError(f"Delta targets need the recorded {len(YOR_ACTION_NAMES)}D EE action window")
    targets = np.zeros((len(actions), len(YOR_DELTA_ROT6D_NAMES)), dtype=np.float64)
    for arm, (pose, out) in enumerate(((0, 0), (7, 10))):
        targets[:, out:out + 3] = actions[:, pose + 4:pose + 7] - actions[0, pose + 4:pose + 7]
        targets[:, out + 3:out + 9] = quaternion_rot6d(actions[:, pose:pose + 4])
        targets[:, out + 9] = actions[:, 14 + arm]
    return targets


class EpisodeReader:
    """Read one episode's parquet rows and seek its shared videos on demand.

    Uses the episode/video locators from meta/episodes, like the value-estimation
    LeRobotReader, without its fixed 25D state or whole-dataset state allocation.
    """

    def __init__(self, dataset_path: Path, episode_index: int, target_feature: str = "action",
                 ee_observation_path: Path | None = None):
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.root = dataset_path.expanduser().resolve()
        self.info = json.loads((self.root / "meta/info.json").read_text())
        if self.info.get("codebase_version") != "v3.0":
            raise ValueError("This reader requires a LeRobot v3.0 dataset")
        self.fps = float(self.info["fps"])
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("Dataset fps must be finite and positive")
        features = self.info["features"]
        if target_feature not in ("action", "observation.state", "action.q_target", *EE_STATE_TARGETS):
            raise ValueError(f"Unsupported replay target: {target_feature}")
        if (target_feature in EE_STATE_TARGETS) != (ee_observation_path is not None):
            raise ValueError("EE-space replay needs an EE-observation export, and no other target uses one")
        self.target_feature = target_feature
        required = ["observation.state", "action", *[f"observation.images.{cam}" for cam in CAMERAS]]
        if target_feature == "action.q_target":
            required.append("action.q_target")
        for key in required:
            if key not in features:
                raise ValueError(f"YOR replay requires dataset feature {key!r}")
        if features["observation.state"]["shape"] != [15]:
            raise ValueError("YOR replay expects the 15D left joints / right joints / lift state")
        self.action_names = features["action"].get("names")
        if target_feature in ("action", *EE_STATE_TARGETS) and (
                self.action_names != YOR_ACTION_NAMES or features["action"]["shape"] != [20]):
            raise ValueError("Expected the YOR 20D absolute EE / gripper / base velocity / lift action layout")
        if target_feature in ("observation.state", "action.q_target") and (
                features["observation.state"].get("names") != YOR_STATE_NAMES):
            raise ValueError("Joint-space replay requires state order lj0..lj6, rj0..rj6, lift")
        if target_feature == "action.q_target" and (features["action.q_target"].get("names") != YOR_Q_TARGET_NAMES
                                                    or features["action.q_target"]["shape"] != [14]):
            raise ValueError("Joint-target replay requires the recorded 14D action.q_target lj0..lj6, rj0..rj6")
        for cam in CAMERAS:
            if features[f"observation.images.{cam}"].get("dtype") != "video":
                raise ValueError(f"Camera {cam!r} must be a video feature")

        metadata_files = sorted((self.root / "meta/episodes").glob("chunk-*/file-*.parquet"))
        if not metadata_files:
            raise FileNotFoundError(f"No episode metadata in {self.root / 'meta/episodes'}")
        rows = []
        for path in metadata_files:
            columns = [c for c in pq.read_schema(path).names if not c.startswith("stats/")]
            rows.extend(pq.read_table(path, columns=columns, filters=[("episode_index", "=", episode_index)]).to_pylist())
        if len(rows) != 1:
            raise ValueError(f"Expected one metadata row for episode {episode_index}, found {len(rows)}")
        self.meta = rows[0]
        self.episode_index = episode_index
        self.length = int(self.meta["length"])
        if self.length <= 0:
            raise ValueError(f"Episode {episode_index} is empty")

        # "observation.state" replay compares against the state column already read below.
        target_column = "action.q_target" if target_feature == "action.q_target" else "action"
        columns = ["index", "frame_index", "timestamp", "task_index", "observation.state", target_column]
        data_path = self.root / self.info["data_path"].format(
            chunk_index=int(self.meta["data/chunk_index"]), file_index=int(self.meta["data/file_index"])
        )
        table = pq.read_table(data_path, columns=columns, filters=[("episode_index", "=", episode_index)])
        # Usually an episode fits in its metadata's shard. Also handle writers
        # that split an episode across shards, without assuming contiguous files.
        if len(table) != self.length:
            tables = [table]
            for path in sorted((self.root / "data").glob("chunk-*/file-*.parquet")):
                if path == data_path:
                    continue
                tables.append(pq.read_table(path, columns=columns, filters=[("episode_index", "=", episode_index)]))
            table = pa.concat_tables(tables)
        table = table.sort_by([("frame_index", "ascending")])
        if len(table) != self.length or table["frame_index"].to_pylist() != list(range(self.length)):
            raise ValueError(f"Episode {episode_index} has missing or duplicate frame rows")
        expected_indices = list(range(int(self.meta["dataset_from_index"]), int(self.meta["dataset_to_index"])))
        if table["index"].to_pylist() != expected_indices:
            raise ValueError(f"Episode {episode_index} global indices disagree with metadata")
        self.states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        self.actions = np.asarray(table[target_column].to_pylist(), dtype=np.float32)
        self.timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
        target_dim = 14 if target_column == "action.q_target" else 20
        if self.states.shape != (self.length, 15) or self.actions.shape != (self.length, target_dim):
            raise ValueError("Episode state/action shapes disagree with dataset features")
        if not all(np.isfinite(a).all() for a in (self.states, self.actions, self.timestamps)):
            raise ValueError("Episode contains non-finite state, action, or timestamp values")
        if target_feature == "observation.state":
            self.actions = self.states.copy()
            self.action_names = YOR_STATE_NAMES.copy()
        elif target_feature == "action.q_target":
            self.action_names = YOR_Q_TARGET_NAMES.copy()
        elif target_feature in EE_STATE_TARGETS:
            # The model sees absolute EE proprioception instead of the joint state, and is
            # scored in its own window-relative output space (see target_window). The two
            # checkpoints differ only in how that proprioception spells orientation; their
            # 20D output layouts are identical.
            poses = self._read_ee_observations(ee_observation_path)
            self.states = poses if target_feature == "delta_rot6d" else canonical_state(poses)
            self.action_names = YOR_DELTA_ROT6D_NAMES.copy()
        if not np.allclose(self.timestamps, np.arange(self.length) / self.fps, atol=1e-4, rtol=0):
            raise ValueError("Expected episode-relative timestamps at the dataset frame rate")
        tasks_table = pq.read_table(self.root / "meta/tasks.parquet").to_pandas()
        # Stock LeRobot stores the task text in the parquet's pandas index;
        # this dataset stores it in an explicit 'task' column.
        task_texts = tasks_table["task"] if "task" in tasks_table else tasks_table.index
        tasks = dict(zip(tasks_table["task_index"], map(str, task_texts)))
        self.prompts = [tasks[int(index)] for index in table["task_index"].to_pylist()]
        self._videos: dict[Path, Any] = {}

    def _read_ee_observations(self, source: Path) -> np.ndarray:
        """Absolute EE proprioception for this episode, from a v2.1 per-episode export.

        The export carries no videos and its own copy of observation.state, so requiring
        that copy to match this dataset's rows exactly is what proves the two are the same
        episode, frame for frame, rather than trusting episode_index alone.
        """
        import pyarrow.parquet as pq

        source = source.expanduser().resolve()
        info = json.loads((source / "meta/info.json").read_text())
        if info.get("codebase_version") != "v2.1":
            raise ValueError("EE observations must come from a v2.1 per-episode export")
        if float(info["fps"]) != self.fps:
            raise ValueError("EE-observation export and dataset frame rates differ")
        for key in ("observation.left_ee", "observation.right_ee"):
            if info["features"][key]["shape"] != [7]:
                raise ValueError(f"{key} must be a 7D quaternion + position pose")
        path = source / info["data_path"].format(
            episode_chunk=self.episode_index // int(info["chunks_size"]), episode_index=self.episode_index)
        if not path.is_file():
            raise FileNotFoundError(f"No EE observations for episode {self.episode_index}: {path}")
        table = pq.read_table(path, columns=["frame_index", "observation.state",
                                             "observation.left_ee", "observation.right_ee"])
        table = table.sort_by([("frame_index", "ascending")])
        if table["frame_index"].to_pylist() != list(range(self.length)):
            raise ValueError(f"EE observations for episode {self.episode_index} have missing or duplicate frames")
        if not np.array_equal(np.asarray(table["observation.state"].to_pylist(), dtype=np.float32), self.states):
            raise ValueError(f"EE observations for episode {self.episode_index} are a different recording")
        poses = np.concatenate([np.asarray(table[f"observation.{arm}_ee"].to_pylist(), dtype=np.float32)
                                for arm in ("left", "right")], axis=1)
        if poses.shape != (self.length, len(YOR_EE_NAMES)) or not np.isfinite(poses).all():
            raise ValueError("EE observations have the wrong shape or non-finite values")
        norms = np.linalg.norm(poses.reshape(self.length, 2, 7)[:, :, :4], axis=2)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("EE-observation quaternions are not unit length")
        return poses

    def target_window(self, frame: int) -> np.ndarray:
        """Ground truth for a chunk predicted at `frame`, in the policy's own output space."""
        if self.target_feature not in EE_STATE_TARGETS:
            return self.actions[frame:]
        return delta_rot6d_targets(self.actions[frame:])

    @property
    def window_relative(self) -> bool:
        """True when the target is defined against each chunk's own first frame."""
        return self.target_feature in EE_STATE_TARGETS

    def read_images(self, frame_index: int) -> dict[str, np.ndarray]:
        import av

        if not 0 <= frame_index < self.length:
            raise IndexError(f"Frame {frame_index} outside episode length {self.length}")
        images = {}
        for cam in CAMERAS:
            video_key = f"observation.images.{cam}"
            prefix = f"videos/{video_key}"
            path = self.root / self.info["video_path"].format(
                video_key=video_key,
                chunk_index=int(self.meta[f"{prefix}/chunk_index"]),
                file_index=int(self.meta[f"{prefix}/file_index"]),
            )
            if path not in self._videos:
                self._videos[path] = av.open(str(path))
            container = self._videos[path]
            stream = container.streams.video[0]
            # Seek to the middle of the display interval, avoiding rounding at
            # episode/frame boundaries in concatenated LeRobot videos.
            target = float(self.meta[f"{prefix}/from_timestamp"]) + (frame_index + 0.5) / self.fps
            end = float(self.meta[f"{prefix}/to_timestamp"])
            if target >= end:
                raise ValueError(f"Frame {frame_index} exceeds {cam} episode video interval")
            container.seek(int(target / stream.time_base), stream=stream, backward=True, any_frame=False)
            selected = None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                frame_time = float(frame.pts * frame.time_base)
                if frame_time > target:
                    break
                selected = frame
            if selected is None or target - float(selected.pts * selected.time_base) > 1.5 / self.fps:
                raise ValueError(f"Could not decode {cam} frame {frame_index} at {target:.6f}s in {path}")
            images[cam] = selected.to_ndarray(format="rgb24")
        return images

    def close(self) -> None:
        for video in self._videos.values():
            video.close()
        self._videos.clear()

    def __enter__(self) -> EpisodeReader:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()


def load_inference_module(workspace: Path, backend: str) -> Any:
    """Import only the selected workspace script; its live main() never runs."""
    path = workspace / f"inference_server_{backend}.py"
    if not path.is_file():
        raise FileNotFoundError(f"Inference helper not found: {path}; set --icl-ws")
    spec = importlib.util.spec_from_file_location(f"_replay_icl_{backend}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import inference helper {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class JointStateOutputs:
    """Remove model padding after unnormalization; all 15 real outputs are states."""

    real_dims = 15

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[:, :self.real_dims]}


class AlignedQOutputs(JointStateOutputs):
    """Same padding removal, but this checkpoint's 14 real outputs are recorded joint
    targets (lj0..lj6, rj0..rj6) with no lift column."""

    real_dims = 14


# Checkpoints this script can reconstruct an inference recipe for: the proprioception
# and real output widths their normalization stats must confirm, the dataset target each
# output is scored against, and the output transform, if the workspace has none that fits.
LOCAL_CONFIGS = {
    JOINT_STATE_CONFIG: {"state_dim": 15, "output_dim": 15, "target": "observation.state",
                         "outputs": JointStateOutputs},
    ALIGNED_Q_CONFIG: {"state_dim": 15, "output_dim": 14, "target": "action.q_target",
                       "outputs": AlignedQOutputs},
    DELTA_ROT6D_CONFIG: {"state_dim": 14, "output_dim": 20, "target": "delta_rot6d",
                         "outputs": None},
    CANONICAL_CONFIG: {"state_dim": 18, "output_dim": 20, "target": "canonical",
                       "outputs": None},
}
LOCAL_TARGETS = {name: spec["target"] for name, spec in LOCAL_CONFIGS.items()}


def register_local_config(name: str, checkpoint: Path) -> None:
    """Supply a missing inference recipe in this process, using workspace OpenPI.

    Each checkpoint README specifies its target space and its stats confirm the widths.
    This is an inference reconstruction, not the original training config.
    """
    import dataclasses
    from openpi.training import config as config_module
    from openpi import transforms

    spec = LOCAL_CONFIGS[name]
    stats = json.loads((checkpoint / "assets/icl-dataset/norm_stats.json").read_text())["norm_stats"]
    for key, width in (("state", spec["state_dim"]), ("actions", spec["output_dim"])):
        if any(len(stats[key][field]) != width for field in ("mean", "std", "q01", "q99")):
            raise ValueError(f"{name} requires {spec['state_dim']}D state and "
                             f"{spec['output_dim']}D action normalization stats")
    try:
        config_module.get_config(name)
        return
    except ValueError:
        pass

    class LocalDataConfig(config_module.YorDataConfig):
        def create(self, assets_dirs, model_config):
            data = super().create(assets_dirs, model_config)
            return dataclasses.replace(data, data_transforms=transforms.Group(
                inputs=data.data_transforms.inputs, outputs=[spec["outputs"]()],
            ))

    # The workspace already has an output transform for the delta/rot6D layout; the other
    # two need the local one above, which only trims the model's padding.
    factory = config_module.YorDeltaRot6dDataConfig if spec["outputs"] is None else LocalDataConfig
    base = config_module.get_config("yor_icl_pi05_easy_pnp_v2")
    recipe = dataclasses.replace(base, name=name, data=factory(
        state_dim=spec["state_dim"], assets=config_module.AssetsConfig(asset_id="icl-dataset"),
    ))
    # The workspace load_policy helper resolves names through this registry.
    config_module._CONFIGS_DICT[name] = recipe
    LOGGER.warning("Registered local %s inference recipe; original training horizon/alignment unverified", name)


class LocalBackend:
    def __init__(self, backend: str, workspace: Path, checkpoint: Path, config: str | None,
                 action_horizon: int | None = None, device: str | None = None):
        self.backend = backend
        self.checkpoint = checkpoint
        self.module = load_inference_module(workspace, backend)
        if backend == "openpi":
            if not config:
                raise ValueError("OpenPI requires --config for checkpoints without a preset")
            if config == "yor_icl_pi05_deltarot6d_expanded":
                raise ValueError("Delta/rot6D output cannot be compared to this dataset's absolute actions")
            if config in LOCAL_CONFIGS:
                register_local_config(config, checkpoint)
            self.policy = self.module.load_policy(config, checkpoint, action_horizon_override=action_horizon)
            self.output_dim = LOCAL_CONFIGS[config]["output_dim"] if config in LOCAL_CONFIGS else 16
        else:
            self.policy, self.preprocessor, self.postprocessor, self.config = self.module.load_policy(checkpoint, device)
            self.output_dim = 20

    def predict(self, images: dict[str, np.ndarray], state: np.ndarray, prompt: str) -> tuple[np.ndarray, float]:
        args = [images[cam] for cam in CAMERAS]
        if self.backend == "openpi":
            actions, latency = self.module.run_inference(self.policy, *args, state, prompt)
        else:
            actions, latency = self.module.run_inference(
                self.policy, self.preprocessor, self.postprocessor, *args, state, prompt, self.config.device
            )
        actions = np.asarray(actions, dtype=np.float32)
        expected_dim = self.output_dim
        if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != expected_dim:
            raise ValueError(f"Expected {self.backend} absolute action chunk (H, {expected_dim}), got {actions.shape}")
        if not np.isfinite(actions).all():
            raise ValueError("Policy returned non-finite actions")
        return actions, float(latency)


def compare_actions(actions: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compare returned dimensions and real remaining frames, with no tail padding."""
    if actions.ndim != 2 or truth.ndim != 2 or actions.shape[1] == 0 or actions.shape[1] > truth.shape[1]:
        raise ValueError("Prediction and ground truth have incompatible shapes")
    steps = min(len(actions), len(truth))
    if steps == 0:
        raise ValueError("Prediction and ground truth have incompatible shapes")
    predicted = actions[:steps]
    error = predicted.astype(np.float64) - truth[:steps, :actions.shape[1]]
    return predicted, error


class ReplayLogger:
    def __init__(self, output_path: Path, reader: EpisodeReader):
        import rerun as rr
        import rerun.blueprint as rrb

        self.rr = rr
        self.recording = rr.RecordingStream("lerobot_policy_eval")
        self.recording.save(str(output_path))
        self.groups: dict[str, list[int]] = {}
        for i, name in enumerate(reader.action_names):
            group = ("left_joints" if name.startswith("lj") else "right_joints" if name.startswith("rj")
                     else name.split(".")[0])
            self.groups.setdefault(group, []).append(i)
        self.names = reader.action_names
        self.recording.send_blueprint(rrb.Blueprint(
            rrb.Vertical(
                rrb.Horizontal(*[rrb.Spatial2DView(name=cam, contents=[f"/images/{cam}"]) for cam in CAMERAS]),
                rrb.Tabs(*[rrb.TimeSeriesView(
                    name=group, contents=[f"/ground_truth/{group}/**", f"/predicted/{group}/**"],
                ) for group in self.groups], name="Actions"),
                rrb.Tabs(*[rrb.TimeSeriesView(name=metric.upper(), contents=[f"/metrics/{metric}/**"])
                           for metric in ("l1", "mse")], name="Error"),
            ), collapse_panels=True,
        ))
        # A window-relative target has no single episode-long ground-truth curve; each
        # chunk carries its own, logged next to that chunk's prediction in log() below.
        self.window_relative = reader.window_relative
        if self.window_relative:
            return
        for group, dims in self.groups.items():
            self._style(f"/ground_truth/{group}", dims, "gt", 1.0)
        for timestamp, action in zip(reader.timestamps, reader.actions):
            self.recording.set_time("ts", duration=float(timestamp))
            for group, dims in self.groups.items():
                self.recording.log(f"/ground_truth/{group}", rr.Scalars(action[dims]))

    def _style(self, path: str, dims: list[int], label: str, width: float) -> None:
        palette = [(31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
                   (148, 103, 189), (140, 86, 75), (227, 119, 194)]
        self.recording.log(path, self.rr.SeriesLines(
            names=[f"{self.names[i]} {label}" for i in dims],
            colors=[palette[i % len(palette)] for i in dims], widths=[width] * len(dims),
        ), static=True)

    def log(self, reader: EpisodeReader, frame: int, images: dict[str, np.ndarray], prompt: str,
            predicted: np.ndarray, truth: np.ndarray, error: np.ndarray, latency_ms: float) -> None:
        rr = self.rr
        rec = self.recording
        rec.set_time("ts", duration=float(reader.timestamps[frame]))
        rec.log("/prompt", rr.TextDocument(prompt))
        rec.log("/state", rr.Scalars(reader.states[frame]))
        rec.log("/metrics/inference_ms", rr.Scalars([latency_ms]))
        for cam, image in images.items():
            rec.log(f"/images/{cam}", rr.Image(image).compress(jpeg_quality=75))
        for group, all_dims in self.groups.items():
            dims = [i for i in all_dims if i < predicted.shape[1]]
            if not dims:
                continue
            rec.log(f"/metrics/l1/{group}", rr.Scalars([float(np.abs(error[:, dims]).mean())]))
            rec.log(f"/metrics/mse/{group}", rr.Scalars([float(np.square(error[:, dims]).mean())]))
            # Separate series retain overlapping chunks without overwriting each
            # other's predictions at the same future timestamp.
            self._style(f"/predicted/{group}/frame_{frame}", dims, f"pred @{frame}", 2.0)
            if self.window_relative:
                self._style(f"/ground_truth/{group}/frame_{frame}", dims, f"gt @{frame}", 1.0)
        for offset, action in enumerate(predicted):
            rec.set_time("ts", duration=float(reader.timestamps[frame + offset]))
            for group, all_dims in self.groups.items():
                dims = [i for i in all_dims if i < predicted.shape[1]]
                if not dims:
                    continue
                rec.log(f"/predicted/{group}/frame_{frame}", rr.Scalars(action[dims]))
                if self.window_relative:
                    rec.log(f"/ground_truth/{group}/frame_{frame}", rr.Scalars(truth[offset, dims]))

    def close(self) -> None:
        self.recording.flush()
        self.recording.disconnect()


def prediction_frames(reader: EpisodeReader, prediction_hz: float, start_frame: int,
                      max_samples: int | None, prompt: str | None) -> range:
    if not np.isfinite(prediction_hz) or prediction_hz <= 0:
        raise ValueError("prediction_hz must be finite and positive")
    stride = reader.fps / prediction_hz
    if stride < 1 or not np.isclose(stride, round(stride)):
        raise ValueError("Dataset fps must be an integer multiple of prediction_hz")
    if not 0 <= start_frame < reader.length:
        raise ValueError(f"start_frame must be in [0, {reader.length})")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive")
    if prompt is not None and not prompt.strip():
        raise ValueError("Prompt override cannot be empty")
    frames = range(start_frame, reader.length, int(round(stride)))
    if max_samples is not None:
        frames = frames[:max_samples]
    if any(not (prompt if prompt is not None else reader.prompts[f]).strip() for f in frames):
        raise ValueError("Episode has empty task text; provide --prompt")
    return frames


def replay_episode(reader: EpisodeReader, backend: Any, output_path: Path, *,
                   prediction_hz: float = 1.0, start_frame: int = 0,
                   max_samples: int | None = None, prompt: str | None = None) -> dict[str, Any]:
    frames = prediction_frames(reader, prediction_hz, start_frame, max_samples, prompt)
    logger = ReplayLogger(output_path, reader)
    absolute_sum = squared_sum = None
    compared_steps = 0
    latencies = []
    horizons = set()
    start = time.perf_counter()
    try:
        for frame in frames:
            images = reader.read_images(frame)
            task = prompt if prompt is not None else reader.prompts[frame]
            actions, latency = backend.predict(images, reader.states[frame].copy(), task)
            window = reader.target_window(frame)
            predicted, error = compare_actions(actions, window)
            truth = window[:len(error), :predicted.shape[1]]
            if absolute_sum is None:
                absolute_sum = np.zeros(error.shape[1], dtype=np.float64)
                squared_sum = np.zeros_like(absolute_sum)
            if len(absolute_sum) != error.shape[1]:
                raise ValueError("Policy output dimension changed during replay")
            absolute_sum += np.abs(error).sum(axis=0)
            squared_sum += np.square(error).sum(axis=0)
            compared_steps += len(error)
            horizons.add(len(actions))
            latencies.append(latency)
            logger.log(reader, frame, images, task, predicted, truth, error, latency)
            LOGGER.info("Episode %d frame %d/%d: horizon=%d compared=%d L1=%.6f MSE=%.6f infer=%.1fms",
                        reader.episode_index, frame, reader.length, len(actions), len(error),
                        np.abs(error).mean(), np.square(error).mean(), latency)
    finally:
        logger.close()
    return {
        "dataset_path": str(reader.root), "episode_index": reader.episode_index,
        "target_feature": reader.target_feature, "target_frame_offset": 0,
        "fps": reader.fps, "episode_frames": reader.length, "prediction_frames": list(frames),
        "sample_count": len(latencies), "prediction_horizons": sorted(horizons),
        "compared_steps": compared_steps, "evaluated_action_names": reader.action_names[:len(absolute_sum)],
        "l1": float(absolute_sum.mean() / compared_steps),
        "mse": float(squared_sum.mean() / compared_steps),
        "l1_per_dimension": (absolute_sum / compared_steps).tolist(),
        "mse_per_dimension": (squared_sum / compared_steps).tolist(),
        "mean_inference_ms": float(np.mean(latencies)), "wall_time_s": time.perf_counter() - start,
        "prompt_override": prompt, "output_path": str(output_path),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--episode-index", type=int, required=True)
    parser.add_argument("--backend", choices=("openpi", "lerobot"), default="openpi")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory, or a name under <icl-ws>/checkpoints")
    parser.add_argument("--config", help="OpenPI TrainConfig; inferred for documented checkpoint presets")
    parser.add_argument("--icl-ws", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--ee-observations", type=Path,
                        help="v2.1 per-episode export supplying observation.left_ee/right_ee; "
                             "required by the delta/rot6D and canonical checkpoints, rejected otherwise")
    parser.add_argument("--action-horizon", type=int, help="OpenPI horizon override; default uses the workspace config")
    parser.add_argument("--device", help="LeRobot device override (e.g. cuda or cpu)")
    parser.add_argument("--prompt", help="Override the dataset's per-frame task text")
    parser.add_argument("--prediction-hz", type=float, default=1.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-samples", type=int, help="Limit inference calls; default evaluates the entire episode")
    parser.add_argument("--output-path", type=Path, help="Output .rrd; a metrics .json is also written alongside it")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    workspace = args.icl_ws.expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser()
    if not checkpoint.is_dir():
        checkpoint = workspace / "checkpoints" / checkpoint
    if not checkpoint.is_dir():
        parser.error(f"Checkpoint directory not found: {checkpoint}")
    checkpoint = checkpoint.resolve()
    config = args.config or OPENPI_PRESETS.get(checkpoint.name)
    if config is None and checkpoint.name in LOCAL_CONFIGS:
        config = checkpoint.name
    if args.backend == "openpi" and checkpoint.name in LOCAL_CONFIGS and config != checkpoint.name:
        parser.error(f"This checkpoint requires --config {checkpoint.name}")
    if args.backend == "openpi" and not config:
        parser.error("--config is required for this OpenPI checkpoint")
    if args.backend == "openpi" and args.device is not None:
        parser.error("--device applies to LeRobot; configure JAX devices via the OpenPI environment")
    if args.backend == "lerobot" and (args.config is not None or args.action_horizon is not None):
        parser.error("--config and --action-horizon apply only to OpenPI")
    if args.action_horizon is not None and args.action_horizon <= 0:
        parser.error("--action-horizon must be positive")
    output = (args.output_path or Path(f"replay_{args.backend}_{checkpoint.name}_ep{args.episode_index}.rrd")).expanduser().resolve()
    if output.suffix != ".rrd":
        parser.error("--output-path must end in .rrd")
    if not output.parent.is_dir():
        parser.error(f"Output directory does not exist: {output.parent}")
    # Refuse accidental overwrites of previous comparisons.
    if output.exists() or output.with_suffix(".json").exists():
        parser.error(f"Output already exists: {output} or its .json sidecar; choose a new --output-path")
    target_feature = LOCAL_TARGETS.get(config, "action") if args.backend == "openpi" else "action"
    if (target_feature in EE_STATE_TARGETS) != (args.ee_observations is not None):
        parser.error("--ee-observations is required by the delta/rot6D and canonical checkpoints "
                     "and used by no other")
    with EpisodeReader(args.dataset_path, args.episode_index, target_feature=target_feature,
                       ee_observation_path=args.ee_observations) as reader:
        prediction_frames(reader, args.prediction_hz, args.start_frame, args.max_samples, args.prompt)
        LOGGER.info("Loaded episode %d: %d frames at %g Hz; first task=%r",
                    reader.episode_index, reader.length, reader.fps, reader.prompts[0])
        backend = LocalBackend(args.backend, workspace, checkpoint, config, args.action_horizon, args.device)
        summary = replay_episode(reader, backend, output, prediction_hz=args.prediction_hz,
                                 start_frame=args.start_frame, max_samples=args.max_samples, prompt=args.prompt)
    summary.update(backend=args.backend, checkpoint=str(checkpoint), config=config if args.backend == "openpi" else None)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    LOGGER.info("Wrote %s and %s", output, output.with_suffix(".json"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
