"""Command-line demo for the Phase 1 simulated arm."""

from __future__ import annotations

import argparse
import math
import time

from buddy_manipulator.kinematics import JointAngles, Pose, inverse_kinematics
from buddy_manipulator.simulation import Keyframe, load_model, run_keyframes


def demo_motion() -> list[Keyframe]:
    home = JointAngles(0.0, 0.45, -0.9, 0.45)
    target = inverse_kinematics(Pose(0.30, 0.08, 0.12, -math.pi / 2))
    return [
        Keyframe(1.5, home, 0.03),
        Keyframe(2.0, target, 0.03),
        Keyframe(1.0, target, 0.005),
        Keyframe(2.0, home, 0.005),
        Keyframe(1.0, home, 0.03),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 1 MuJoCo arm demo.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the MuJoCo viewer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model, data = load_model()
    if args.headless:
        run_keyframes(model, data, demo_motion())
        print(f"simulation complete: {data.time:.2f}s")
        return

    import mujoco.viewer

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 1.1
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            print("MuJoCo viewer started. Running arm demo...", flush=True)
            run_keyframes(model, data, demo_motion(), viewer=viewer, realtime=True)
            print("Demo complete. Close the MuJoCo window to exit.", flush=True)
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
    except RuntimeError as exc:
        if "requires that the Python script be run under `mjpython`" in str(exc):
            raise RuntimeError(
                "MuJoCo GUI mode requires mjpython on macOS. Run the demo "
                "with: ./scripts/run_demo.sh"
            ) from exc
        raise


if __name__ == "__main__":
    main()
