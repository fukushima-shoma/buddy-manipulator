import pytest

from buddy_manipulator.kinematics import JointAngles
from buddy_manipulator.simulation import Keyframe, densify_keyframes


def test_densify_keyframes_bounds_joint_steps_and_preserves_duration() -> None:
    start = JointAngles(0.0, 0.0, 0.0, 0.0)
    keyframes = [
        Keyframe(1.0, JointAngles(0.25, -0.1, 0.0, 0.0), 0.01),
        Keyframe(0.5, JointAngles(0.25, -0.1, 0.2, 0.0), 0.03),
    ]

    dense = densify_keyframes(start, keyframes, maximum_joint_step=0.1)

    assert len(dense) == 5
    assert sum(frame.duration for frame in dense) == pytest.approx(1.5)
    previous = start
    for frame in dense:
        assert max(
            abs(current - prior)
            for current, prior in zip(frame.joints.as_tuple(), previous.as_tuple())
        ) <= 0.1 + 1e-12
        previous = frame.joints
    assert dense[-1].joints == keyframes[-1].joints
    assert dense[-1].gripper == 0.03


def test_densify_keyframes_rejects_non_positive_step() -> None:
    with pytest.raises(ValueError, match="positive"):
        densify_keyframes(
            JointAngles(0.0, 0.0, 0.0, 0.0),
            [],
            maximum_joint_step=0.0,
        )
