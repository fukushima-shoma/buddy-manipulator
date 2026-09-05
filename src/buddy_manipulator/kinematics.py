"""Analytical kinematics for Buddy Manipulator's yaw + planar arm."""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ArmGeometry:
    base_height: float = 0.12
    upper_arm: float = 0.22
    forearm: float = 0.18
    tool: float = 0.08


@dataclass(frozen=True)
class JointAngles:
    base: float
    shoulder: float
    elbow: float
    wrist: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return self.base, self.shoulder, self.elbow, self.wrist


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    pitch: float


class UnreachableTargetError(ValueError):
    """Raised when a Cartesian target lies outside the arm workspace."""


def forward_kinematics(
    joints: JointAngles,
    geometry: ArmGeometry = ArmGeometry(),
) -> Pose:
    """Return the tool-tip pose for a joint configuration.

    Shoulder, elbow, and wrist operate in a vertical plane. ``pitch`` is the
    tool angle measured from the horizontal plane.
    """
    shoulder_elbow = joints.shoulder + joints.elbow
    pitch = shoulder_elbow + joints.wrist
    radial = (
        geometry.upper_arm * math.cos(joints.shoulder)
        + geometry.forearm * math.cos(shoulder_elbow)
        + geometry.tool * math.cos(pitch)
    )
    z = (
        geometry.base_height
        + geometry.upper_arm * math.sin(joints.shoulder)
        + geometry.forearm * math.sin(shoulder_elbow)
        + geometry.tool * math.sin(pitch)
    )
    return Pose(
        x=radial * math.cos(joints.base),
        y=radial * math.sin(joints.base),
        z=z,
        pitch=pitch,
    )


def inverse_kinematics(
    target: Pose,
    geometry: ArmGeometry = ArmGeometry(),
    *,
    elbow_up: bool = False,
) -> JointAngles:
    """Solve a tool pose analytically for the four arm joints."""
    radial = math.hypot(target.x, target.y)
    wrist_radial = radial - geometry.tool * math.cos(target.pitch)
    wrist_z = (
        target.z
        - geometry.base_height
        - geometry.tool * math.sin(target.pitch)
    )

    numerator = (
        wrist_radial * wrist_radial
        + wrist_z * wrist_z
        - geometry.upper_arm * geometry.upper_arm
        - geometry.forearm * geometry.forearm
    )
    denominator = 2.0 * geometry.upper_arm * geometry.forearm
    cosine_elbow = numerator / denominator
    tolerance = 1e-9
    if cosine_elbow < -1.0 - tolerance or cosine_elbow > 1.0 + tolerance:
        raise UnreachableTargetError(
            f"target ({target.x:.3f}, {target.y:.3f}, {target.z:.3f}) "
            "is outside the arm workspace"
        )
    cosine_elbow = min(1.0, max(-1.0, cosine_elbow))
    sine_elbow = math.sqrt(max(0.0, 1.0 - cosine_elbow * cosine_elbow))
    if elbow_up:
        sine_elbow = -sine_elbow

    elbow = math.atan2(sine_elbow, cosine_elbow)
    shoulder = math.atan2(wrist_z, wrist_radial) - math.atan2(
        geometry.forearm * sine_elbow,
        geometry.upper_arm + geometry.forearm * cosine_elbow,
    )
    wrist = target.pitch - shoulder - elbow
    return JointAngles(
        base=math.atan2(target.y, target.x),
        shoulder=shoulder,
        elbow=elbow,
        wrist=wrist,
    )
