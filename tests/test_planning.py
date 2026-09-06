import numpy as np
import pytest

from buddy_manipulator.kinematics import forward_kinematics
from buddy_manipulator.planning import make_approach_pose, plan_approach_joints
from buddy_manipulator.vision import PixelDetection, WorldDetection


def make_detection() -> WorldDetection:
    pixel = PixelDetection(10.0, 12.0, 8, 10, 12, 14, 20, np.ones((2, 2)))
    return WorldDetection(pixel=pixel, x=0.30, y=0.08, z=0.05, depth=0.70)


def test_make_approach_pose_adds_vertical_clearance() -> None:
    pose = make_approach_pose(make_detection(), clearance=0.12)
    assert (pose.x, pose.y, pose.z) == pytest.approx((0.30, 0.08, 0.17))


def test_planned_joints_reach_approach_pose() -> None:
    detection = make_detection()
    expected = make_approach_pose(detection)
    joints = plan_approach_joints(detection)
    actual = forward_kinematics(joints)
    assert (actual.x, actual.y, actual.z, actual.pitch) == pytest.approx(
        (expected.x, expected.y, expected.z, expected.pitch)
    )
    assert joints.elbow < 0


def test_clearance_must_be_positive() -> None:
    with pytest.raises(ValueError, match="clearance"):
        make_approach_pose(make_detection(), clearance=0.0)
