import pytest

mujoco = pytest.importorskip("mujoco")

from buddy_manipulator.kinematics import JointAngles
from buddy_manipulator.sim_camera import capture_rgbd
from buddy_manipulator.simulation import Keyframe, load_model, run_keyframes
from buddy_manipulator.vision import detect_red_object, locate_detection_in_world


def test_camera_localizes_simulated_red_block() -> None:
    model, data = load_model()
    stowed = JointAngles(base=-1.2, shoulder=0.6, elbow=-1.2, wrist=0.6)
    run_keyframes(model, data, [Keyframe(0.5, stowed, 0.03)])

    frame = capture_rgbd(model, data, width=320, height=240)
    pixel = detect_red_object(frame.rgb)
    assert pixel is not None

    world = locate_detection_in_world(
        pixel,
        frame.depth,
        vertical_fov_degrees=frame.vertical_fov_degrees,
        camera_position=frame.camera_position,
        camera_rotation=frame.camera_rotation,
    )
    assert world.x == pytest.approx(0.30, abs=0.015)
    assert world.y == pytest.approx(0.08, abs=0.015)
    assert world.z == pytest.approx(0.05, abs=0.012)
