"""MuJoCo loading and position-control helpers."""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Any, Iterable

from buddy_manipulator.kinematics import JointAngles


ARM_ACTUATORS = ("base", "shoulder", "elbow", "wrist")


def _mujoco() -> Any:
    try:
        import mujoco
    except ImportError as exc:
        import_error = str(exc)
        architecture_mismatch = (
            "x86_64 build of Python on an Apple Silicon" in import_error
            or (
                "incompatible architecture" in import_error
                and "have 'arm64', need 'x86_64'" in import_error
            )
        )
        if architecture_mismatch:
            raise RuntimeError(
                "MuJoCo requires native arm64 Python on Apple Silicon, but "
                "this process is running as x86_64 under Rosetta. Run the "
                "demo with: ./scripts/run_demo.sh"
            ) from exc
        raise RuntimeError(
            "MuJoCo could not be imported. Install the project with: "
            "python -m pip install -e '.[dev]'. Original error: "
            f"{exc}"
        ) from exc
    return mujoco


def model_path() -> str:
    return str(files("buddy_manipulator").joinpath("models/arm.xml"))


def load_model() -> tuple[Any, Any]:
    mujoco = _mujoco()
    model = mujoco.MjModel.from_xml_path(model_path())
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def actuator_id(model: Any, name: str) -> int:
    mujoco = _mujoco()
    actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator < 0:
        raise KeyError(f"unknown actuator: {name}")
    return actuator


def set_arm_target(model: Any, data: Any, joints: JointAngles) -> None:
    for name, value in zip(ARM_ACTUATORS, joints.as_tuple()):
        data.ctrl[actuator_id(model, name)] = value


def set_gripper(model: Any, data: Any, opening: float) -> None:
    opening = min(0.03, max(0.0, opening))
    data.ctrl[actuator_id(model, "left_finger")] = opening
    data.ctrl[actuator_id(model, "right_finger")] = opening


@dataclass(frozen=True)
class Keyframe:
    duration: float
    joints: JointAngles
    gripper: float = 0.03


def run_keyframes(
    model: Any,
    data: Any,
    keyframes: Iterable[Keyframe],
    *,
    viewer: Any | None = None,
) -> None:
    mujoco = _mujoco()
    for keyframe in keyframes:
        set_arm_target(model, data, keyframe.joints)
        set_gripper(model, data, keyframe.gripper)
        end_time = data.time + keyframe.duration
        while data.time < end_time:
            mujoco.mj_step(model, data)
            if viewer is not None:
                if not viewer.is_running():
                    return
                viewer.sync()
