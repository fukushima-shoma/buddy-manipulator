"""Render, detect, localize, and plan an approach to the red block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from buddy_manipulator.kinematics import JointAngles
from buddy_manipulator.planning import make_approach_pose, plan_approach_joints
from buddy_manipulator.sim_camera import capture_rgbd
from buddy_manipulator.simulation import Keyframe, load_model, run_keyframes
from buddy_manipulator.vision import detect_red_object, locate_detection_in_world


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 2 RGB-D demo.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase2"))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args()


def save_annotated_image(rgb, detection, output: Path) -> None:
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    draw.rectangle(detection.bbox, outline=(255, 235, 40), width=3)
    u = round(detection.center_u)
    v = round(detection.center_v)
    draw.line((u - 8, v, u + 8, v), fill=(255, 255, 255), width=2)
    draw.line((u, v - 8, u, v + 8), fill=(255, 255, 255), width=2)
    image.save(output)


def main() -> None:
    args = parse_args()
    if args.width <= 0 or args.height <= 0:
        raise ValueError("image dimensions must be positive")

    model, data = load_model()
    stowed = JointAngles(base=-1.2, shoulder=0.6, elbow=-1.2, wrist=0.6)
    run_keyframes(model, data, [Keyframe(1.0, stowed, 0.03)])

    frame = capture_rgbd(model, data, width=args.width, height=args.height)
    pixel_detection = detect_red_object(frame.rgb)
    if pixel_detection is None:
        raise RuntimeError("red block was not detected")
    world_detection = locate_detection_in_world(
        pixel_detection,
        frame.depth,
        vertical_fov_degrees=frame.vertical_fov_degrees,
        camera_position=frame.camera_position,
        camera_rotation=frame.camera_rotation,
    )
    approach_pose = make_approach_pose(world_detection)
    approach_joints = plan_approach_joints(world_detection)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.output_dir / "rgb_detection.png"
    result_path = args.output_dir / "detection.json"
    save_annotated_image(frame.rgb, pixel_detection, image_path)

    result = {
        "pixel": {
            "center": [pixel_detection.center_u, pixel_detection.center_v],
            "bbox": list(pixel_detection.bbox),
            "area": pixel_detection.area,
        },
        "world_position_m": list(world_detection.position),
        "depth_m": world_detection.depth,
        "approach_pose": {
            "x": approach_pose.x,
            "y": approach_pose.y,
            "z": approach_pose.z,
            "pitch": approach_pose.pitch,
        },
        "approach_joints_rad": list(approach_joints.as_tuple()),
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"detected pixel: ({pixel_detection.center_u:.1f}, {pixel_detection.center_v:.1f})")
    print(
        "estimated world position: "
        f"({world_detection.x:.3f}, {world_detection.y:.3f}, "
        f"{world_detection.z:.3f}) m"
    )
    print(f"annotated image: {image_path}")
    print(f"result JSON: {result_path}")


if __name__ == "__main__":
    main()
