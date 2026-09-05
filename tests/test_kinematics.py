import math

import pytest

from buddy_manipulator.kinematics import (
    JointAngles,
    Pose,
    UnreachableTargetError,
    forward_kinematics,
    inverse_kinematics,
)


@pytest.mark.parametrize(
    "joints",
    [
        JointAngles(0.0, 0.0, 0.0, 0.0),
        JointAngles(0.3, 0.4, -0.8, 0.2),
        JointAngles(-0.7, 0.8, -1.2, -0.1),
    ],
)
def test_inverse_kinematics_round_trip(joints: JointAngles) -> None:
    target = forward_kinematics(joints)
    solved = inverse_kinematics(target, elbow_up=joints.elbow < 0)
    actual = forward_kinematics(solved)

    assert actual.x == pytest.approx(target.x, abs=1e-8)
    assert actual.y == pytest.approx(target.y, abs=1e-8)
    assert actual.z == pytest.approx(target.z, abs=1e-8)
    assert actual.pitch == pytest.approx(target.pitch, abs=1e-8)


def test_forward_kinematics_for_straight_arm() -> None:
    pose = forward_kinematics(JointAngles(0.0, 0.0, 0.0, 0.0))
    assert pose.x == pytest.approx(0.48)
    assert pose.y == pytest.approx(0.0)
    assert pose.z == pytest.approx(0.12)
    assert pose.pitch == pytest.approx(0.0)


def test_inverse_kinematics_rejects_unreachable_target() -> None:
    with pytest.raises(UnreachableTargetError):
        inverse_kinematics(Pose(2.0, 0.0, 0.2, -math.pi / 2))
