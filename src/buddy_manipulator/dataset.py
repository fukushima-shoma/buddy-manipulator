"""Versioned demonstration episode recording and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from buddy_manipulator.sim_camera import RgbdCamera
from buddy_manipulator.simulation import _mujoco


SCHEMA_VERSION = 1
CONTROLLED_JOINTS = (
    "base_yaw",
    "shoulder_joint",
    "elbow_joint",
    "wrist_joint",
    "left_finger_joint",
    "right_finger_joint",
)


@dataclass
class EpisodeRecorder:
    model: Any
    camera: RgbdCamera
    sample_hz: float = 5.0
    object_body_name: str = "red_block"
    _next_sample_time: float = field(init=False, default=0.0)
    _timestamps: list[float] = field(init=False, default_factory=list)
    _rgb: list[np.ndarray] = field(init=False, default_factory=list)
    _depth: list[np.ndarray] = field(init=False, default_factory=list)
    _joint_position: list[np.ndarray] = field(init=False, default_factory=list)
    _action: list[np.ndarray] = field(init=False, default_factory=list)
    _object_position: list[np.ndarray] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.sample_hz <= 0:
            raise ValueError("sample_hz must be positive")
        mujoco = _mujoco()
        self._qpos_addresses = []
        for name in CONTROLLED_JOINTS:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise KeyError(f"unknown joint: {name}")
            self._qpos_addresses.append(int(self.model.jnt_qposadr[joint_id]))
        self._object_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, self.object_body_name
        )
        if self._object_body_id < 0:
            raise KeyError(f"unknown body: {self.object_body_name}")

    def maybe_record(self, _model: Any, data: Any) -> None:
        if data.time + 1e-9 < self._next_sample_time:
            return
        frame = self.camera.capture(data)
        self._timestamps.append(float(data.time))
        self._rgb.append(frame.rgb.astype(np.uint8, copy=False))
        self._depth.append(frame.depth.astype(np.float32, copy=False))
        self._joint_position.append(
            np.asarray(data.qpos[self._qpos_addresses], dtype=np.float32).copy()
        )
        self._action.append(np.asarray(data.ctrl, dtype=np.float32).copy())
        self._object_position.append(
            np.asarray(data.xpos[self._object_body_id], dtype=np.float32).copy()
        )
        self._next_sample_time = float(data.time) + 1.0 / self.sample_hz

    @property
    def sample_count(self) -> int:
        return len(self._timestamps)

    def arrays(self) -> dict[str, np.ndarray]:
        if not self._timestamps:
            raise ValueError("episode has no samples")
        return {
            "timestamp": np.asarray(self._timestamps, dtype=np.float64),
            "rgb": np.stack(self._rgb),
            "depth": np.stack(self._depth),
            "joint_position": np.stack(self._joint_position),
            "action": np.stack(self._action),
            "object_position": np.stack(self._object_position),
        }


def save_episode(
    output_dir: Path,
    episode_index: int,
    arrays: dict[str, np.ndarray],
    *,
    success: bool,
    block_start_position: tuple[float, float, float],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{episode_index:05d}"
    data_path = output_dir / f"{stem}.npz"
    metadata_path = output_dir / f"{stem}.json"
    np.savez_compressed(data_path, **arrays)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "episode_index": episode_index,
        "success": success,
        "sample_count": int(arrays["timestamp"].shape[0]),
        "sample_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "controlled_joints": list(CONTROLLED_JOINTS),
        "block_start_position_m": list(block_start_position),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return data_path, metadata_path


def validate_episode(data_path: Path, metadata_path: Path) -> list[str]:
    errors: list[str] = []
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    with np.load(data_path) as arrays:
        expected = {
            "timestamp",
            "rgb",
            "depth",
            "joint_position",
            "action",
            "object_position",
        }
        missing = expected.difference(arrays.files)
        if missing:
            errors.append(f"missing arrays: {sorted(missing)}")
            return errors
        sample_count = arrays["timestamp"].shape[0]
        for key in expected.difference({"timestamp"}):
            if arrays[key].shape[0] != sample_count:
                errors.append(f"sample count mismatch: {key}")
        if arrays["rgb"].dtype != np.uint8:
            errors.append("rgb must use uint8")
        if arrays["depth"].dtype != np.float32:
            errors.append("depth must use float32")
        if arrays["joint_position"].shape[1:] != (6,):
            errors.append("joint_position must have six values per sample")
        if arrays["action"].shape[1:] != (6,):
            errors.append("action must have six values per sample")
        for key in ("timestamp", "depth", "joint_position", "action"):
            if not np.all(np.isfinite(arrays[key])):
                errors.append(f"non-finite values: {key}")
        if metadata.get("sample_count") != sample_count:
            errors.append("metadata sample_count mismatch")
    return errors
