"""Detect the red block and execute a visible top-down grasp."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from buddy_manipulator.grasping import execute_grasp
from buddy_manipulator.kinematics import JointAngles
from buddy_manipulator.sim_camera import capture_rgbd
from buddy_manipulator.simulation import Keyframe, load_model, run_keyframes
from buddy_manipulator.vision import detect_red_object, locate_detection_in_world


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the perception-driven grasp demo.")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase2"))
    return parser.parse_args()


def detect_block(model, data):
    frame = capture_rgbd(model, data, width=640, height=480)
    pixel = detect_red_object(frame.rgb)
    if pixel is None:
        raise RuntimeError("red block was not detected")
    return locate_detection_in_world(
        pixel,
        frame.depth,
        vertical_fov_degrees=frame.vertical_fov_degrees,
        camera_position=frame.camera_position,
        camera_rotation=frame.camera_rotation,
    )


def save_result(result, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "grasp_result.json"
    payload = {
        "success": result.success,
        "initial_height_m": result.initial_height,
        "final_height_m": result.final_height,
        "lift_delta_m": result.lift_delta,
        "finger_contact_count": result.finger_contact_count,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def print_result(result, result_path: Path) -> None:
    print(f"grasp success: {str(result.success).lower()}")
    print(
        f"block height: {result.initial_height:.3f} -> "
        f"{result.final_height:.3f} m "
        f"(lift {result.lift_delta:.3f} m)"
    )
    print(f"finger contacts: {result.finger_contact_count}")
    print(f"result JSON: {result_path}")


def main() -> None:
    args = parse_args()
    model, data = load_model()
    stowed = JointAngles(base=-1.2, shoulder=0.6, elbow=-1.2, wrist=0.6)
    run_keyframes(model, data, [Keyframe(1.0, stowed, 0.03)])
    detection = detect_block(model, data)
    print(
        "detected world position: "
        f"({detection.x:.3f}, {detection.y:.3f}, {detection.z:.3f}) m",
        flush=True,
    )

    if args.headless:
        result = execute_grasp(model, data, detection)
        result_path = save_result(result, args.output_dir)
        print_result(result, result_path)
        if not result.success:
            raise SystemExit(1)
        return

    import mujoco.viewer

    print("MuJoCo viewer started. Executing grasp...", flush=True)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        result = execute_grasp(model, data, detection, viewer=viewer)
        result_path = save_result(result, args.output_dir)
        print_result(result, result_path)
        print("Close the MuJoCo window to exit.", flush=True)
        while viewer.is_running():
            viewer.sync()
            time.sleep(0.05)

    if not result.success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
