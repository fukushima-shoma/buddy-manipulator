"""Small task-space planning helpers used before learned policies."""

from __future__ import annotations

import math

from buddy_manipulator.kinematics import JointAngles, Pose, inverse_kinematics
from buddy_manipulator.vision import WorldDetection


def make_approach_pose(
    detection: WorldDetection,
    *,
    clearance: float = 0.10,
) -> Pose:
    if clearance <= 0:
        raise ValueError("clearance must be positive")
    return Pose(
        x=detection.x,
        y=detection.y,
        z=detection.z + clearance,
        pitch=-math.pi / 2,
    )


def plan_approach_joints(
    detection: WorldDetection,
    *,
    clearance: float = 0.10,
) -> JointAngles:
    return inverse_kinematics(make_approach_pose(detection, clearance=clearance))
