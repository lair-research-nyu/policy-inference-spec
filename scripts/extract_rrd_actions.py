#!/usr/bin/env python
"""Extract the commanded joint trajectory of an .rrd into an .npz, resampled to a fixed rate.

Run under this repo's venv (rerun 0.30). ``server.server_openpi --replay`` calls it in a
subprocess: the lair-openpi venv ships rerun 0.23, which can neither open a recording
written by 0.30 nor serve one through ``rr.server``.

Only the action streams of the schema (``/commanded_qpos/*`` for the gen2 joint schemas)
are read, so recordings without camera streams -- e.g. the client-side rollout logs --
work too, unlike ``policy_inference_spec.replay_rrd.RerunReplayer``. Resampling is a
zero-order hold onto a uniform ``1/hz`` grid, the same rule ``RerunReplayer`` uses: row
``i`` is the last command logged at or before ``t0 + i/hz``, with ``t0`` the first time
every stream has a value.

    .venv/bin/python scripts/extract_rrd_actions.py cubestack-recording.rrd --output cubestack.npz

Output keys: ``actions`` (T, 25) float32 in wire order, ``hz``, ``t0_ns``, ``source``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rerun as rr

from policy_inference_spec.feature_engineering import SchemaName, get_feature_bundle_for_schema
from policy_inference_spec.hardware_model import DEFAULT_HARDWARE_MODEL

_POSE_ACTION_SCHEMAS = {SchemaName.GEN2_28D_STATE_POSE_ACTIONS, SchemaName.GEN2_32D_STATE_POSE_ACTIONS}


def extract(recording: Path, *, hz: float, schema: SchemaName = SchemaName.GEN2_32D_STATE) -> tuple[np.ndarray, int]:
    """The recording's commanded actions on a uniform ``hz`` grid, and the grid's start (ns)."""
    assert schema not in _POSE_ACTION_SCHEMAS, f"{schema.value} has Cartesian actions; replay needs joint positions"
    bundle = get_feature_bundle_for_schema(schema)
    assert bundle.action_dim == DEFAULT_HARDWARE_MODEL.action_dim, (bundle.action_dim, DEFAULT_HARDWARE_MODEL.action_dim)
    columns = [f"{f.rrd_entity_path}:{f.rrd_component}" for f in bundle.actions]

    server = rr.server.Server(datasets={"recording": [recording.resolve()]}, host="127.0.0.1")
    dataset = server.client().get_dataset("recording")
    segments = dataset.segment_ids()
    assert len(segments) == 1, f"expected one recording segment in {recording}, got {len(segments)}"
    df = (dataset.filter_segments(segments[0])
          .filter_contents([f.rrd_entity_path for f in bundle.actions])
          .reader(index="ts")
          .select("ts", *columns)
          .to_pandas())
    missing = [c for c in columns if df[c].notna().sum() == 0]
    assert not missing, f"{recording} has no samples for {missing}"

    df = df.sort_values("ts")
    ts = df["ts"].to_numpy().astype("datetime64[ns]").astype(np.int64)
    blocks = []
    for feature, column in zip(bundle.actions, columns):
        values = df[column].ffill()  # zero-order hold across rows where only other streams logged
        block = np.full((len(df), feature.shape), np.nan, dtype=np.float32)
        have = values.notna().to_numpy()
        block[have] = np.stack(values[have].to_numpy()).astype(np.float32)
        blocks.append(block)
    rows = np.concatenate(blocks, axis=1)

    complete = ~np.isnan(rows).any(axis=1)
    first = int(np.argmax(complete))
    assert complete[first], f"{recording}: no time at which every action stream has a value"
    t0, t_end = ts[first], ts[-1]
    grid = t0 + np.arange(0, t_end - t0 + 1, int(round(1e9 / hz)), dtype=np.int64)
    idx = np.searchsorted(ts, grid, side="right") - 1  # last sample at or before each grid time
    return rows[idx], int(t0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="Output .npz path")
    parser.add_argument("--hz", type=float, default=50.0, help="Resampling rate (default: 50, the robot's)")
    parser.add_argument("--schema", type=SchemaName, default=SchemaName.GEN2_32D_STATE)
    args = parser.parse_args()
    recording = args.recording.expanduser()
    assert recording.is_file(), f"no such recording: {recording}"
    actions, t0_ns = extract(recording, hz=args.hz, schema=args.schema)
    np.savez(args.output, actions=actions, hz=np.float64(args.hz), t0_ns=np.int64(t0_ns),
             source=np.str_(str(recording.resolve())))
    print(f"wrote {args.output}: {len(actions)} rows at {args.hz:g} Hz ({len(actions) / args.hz:.1f} s)")


if __name__ == "__main__":
    main()
