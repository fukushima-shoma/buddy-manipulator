import numpy as np

from buddy_manipulator.grasping import execute_grasp
from buddy_manipulator.kinematics import JointAngles
from buddy_manipulator.simulation import Keyframe, load_model, run_keyframes
from buddy_manipulator.vision import PixelDetection, WorldDetection


def test_scripted_grasp_lifts_block_with_finger_contact() -> None:
    model, data = load_model()
    stowed = JointAngles(base=-1.2, shoulder=0.6, elbow=-1.2, wrist=0.6)
    run_keyframes(model, data, [Keyframe(0.5, stowed, 0.03)])
    pixel = PixelDetection(0.0, 0.0, 0, 0, 0, 0, 1, np.ones((1, 1)))
    detection = WorldDetection(
        pixel=pixel,
        x=0.30,
        y=0.08,
        z=0.05,
        depth=0.70,
    )

    result = execute_grasp(model, data, detection)

    assert result.success
    assert result.lift_delta >= 0.05
    assert result.finger_contact_count >= 1
