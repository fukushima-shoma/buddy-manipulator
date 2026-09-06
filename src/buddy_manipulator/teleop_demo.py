"""Interactive keyboard teleoperation with demonstration recording."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import time

from buddy_manipulator.collect_demos import (
    next_episode_index,
    sample_block_position,
    set_block_position,
)
from buddy_manipulator.dataset import EpisodeRecorder, save_episode
from buddy_manipulator.grasp_demo import detect_block
from buddy_manipulator.grasping import evaluate_grasp, object_height
from buddy_manipulator.kinematics import JointAngles
from buddy_manipulator.planning import make_approach_pose
from buddy_manipulator.sim_camera import RgbdCamera
from buddy_manipulator.simulation import (
    Keyframe,
    _mujoco,
    load_model,
    run_keyframes,
    set_arm_target,
    set_gripper,
)
from buddy_manipulator.teleop import CartesianTeleopController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a keyboard teleop episode.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/demonstrations"),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample-hz", type=float, default=5.0)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=120)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    return parser.parse_args()


def print_controls() -> None:
    print("Keyboard controls (click the MuJoCo window first):")
    print("  W/S: +X/-X    A/D: +Y/-Y    R/F: +Z/-Z")
    print("  O: open       C: close      Enter: save and finish")
    print("  Esc: abort and save as a failed episode", flush=True)


def main() -> None:
    args = parse_args()
    if args.sample_hz <= 0 or args.max_seconds <= 0:
        raise ValueError("sample_hz and max_seconds must be positive")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("image dimensions must be positive")

    rng = random.Random(args.seed)
    block_position = sample_block_position(rng)
    model, data = load_model()
    set_block_position(model, data, block_position)
    stowed = JointAngles(base=-1.2, shoulder=0.6, elbow=-1.2, wrist=0.6)
    run_keyframes(model, data, [Keyframe(0.7, stowed, 0.03)])
    detection = detect_block(model, data)
    start_pose = make_approach_pose(detection)
    controller = CartesianTeleopController(start_pose)
    run_keyframes(
        model,
        data,
        [Keyframe(1.0, controller.joints(), controller.gripper)],
    )
    initial_height = object_height(model, data)
    episode_index = next_episode_index(args.output_dir)

    import mujoco.viewer

    print_controls()
    print(
        f"episode {episode_index:05d} block="
        f"({block_position[0]:.3f}, {block_position[1]:.3f}, "
        f"{block_position[2]:.3f})",
        flush=True,
    )
    mujoco = _mujoco()
    termination = "window_closed"
    session_start = float(data.time)
    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=controller.handle_key,
    ) as viewer, RgbdCamera(
        model,
        width=args.width,
        height=args.height,
    ) as camera:
        viewer.cam.distance = 1.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        recorder = EpisodeRecorder(model, camera, sample_hz=args.sample_hz)
        recorder.maybe_record(model, data)
        while viewer.is_running() and not controller.finished:
            set_arm_target(model, data, controller.joints())
            set_gripper(model, data, controller.gripper)
            mujoco.mj_step(model, data)
            recorder.maybe_record(model, data)
            viewer.sync()
            time.sleep(float(model.opt.timestep))
            if data.time - session_start >= args.max_seconds:
                termination = "timeout"
                controller.aborted = True
                break
        if controller.finished:
            termination = "aborted" if controller.aborted else "completed"

    result = evaluate_grasp(model, data, initial_height)
    data_path, _ = save_episode(
        args.output_dir,
        episode_index,
        recorder.arrays(),
        success=result.success,
        block_start_position=block_position,
        source="teleop",
        termination=termination,
    )
    print(
        f"saved {data_path}: success={str(result.success).lower()} "
        f"termination={termination} samples={recorder.sample_count} "
        f"lift={result.lift_delta:.3f}m",
        flush=True,
    )


if __name__ == "__main__":
    main()
