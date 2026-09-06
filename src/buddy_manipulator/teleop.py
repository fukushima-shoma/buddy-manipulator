"""Keyboard-driven Cartesian teleoperation state and safety checks."""

from __future__ import annotations

from dataclasses import dataclass, replace

from buddy_manipulator.kinematics import (
    JointAngles,
    Pose,
    UnreachableTargetError,
    inverse_kinematics,
)


KEY_ENTER = 257
KEY_ESCAPE = 256


@dataclass
class CartesianTeleopController:
    target: Pose
    gripper: float = 0.03
    move_step: float = 0.01
    finished: bool = False
    aborted: bool = False
    last_message: str = "ready"

    def joints(self) -> JointAngles:
        return inverse_kinematics(self.target, elbow_up=True)

    def handle_key(self, key: int) -> None:
        if key == KEY_ENTER:
            self.finished = True
            self.last_message = "finish requested"
            print(self.last_message, flush=True)
            return
        if key == KEY_ESCAPE:
            self.aborted = True
            self.finished = True
            self.last_message = "abort requested"
            print(self.last_message, flush=True)
            return
        if key in (ord("O"), ord("C")):
            self.gripper = 0.03 if key == ord("O") else 0.0
            self.last_message = (
                "gripper open" if self.gripper > 0 else "gripper closed"
            )
            print(self.last_message, flush=True)
            return

        movement = {
            ord("W"): (self.move_step, 0.0, 0.0),
            ord("S"): (-self.move_step, 0.0, 0.0),
            ord("A"): (0.0, self.move_step, 0.0),
            ord("D"): (0.0, -self.move_step, 0.0),
            ord("R"): (0.0, 0.0, self.move_step),
            ord("F"): (0.0, 0.0, -self.move_step),
        }.get(key)
        if movement is None:
            return

        candidate = replace(
            self.target,
            x=self.target.x + movement[0],
            y=self.target.y + movement[1],
            z=self.target.z + movement[2],
        )
        if not (
            0.18 <= candidate.x <= 0.42
            and -0.22 <= candidate.y <= 0.22
            and 0.04 <= candidate.z <= 0.35
        ):
            self.last_message = "target rejected: workspace boundary"
            print(self.last_message, flush=True)
            return
        try:
            joints = inverse_kinematics(candidate, elbow_up=True)
        except UnreachableTargetError:
            self.last_message = "target rejected: unreachable"
            print(self.last_message, flush=True)
            return
        if not (
            -1.57 <= joints.shoulder <= 1.57
            and -2.40 <= joints.elbow <= 2.40
            and -2.80 <= joints.wrist <= 2.80
        ):
            self.last_message = "target rejected: joint limit"
            print(self.last_message, flush=True)
            return

        self.target = candidate
        self.last_message = (
            f"target x={candidate.x:.3f} y={candidate.y:.3f} "
            f"z={candidate.z:.3f}"
        )
        print(self.last_message, flush=True)
