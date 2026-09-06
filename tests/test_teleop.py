import math

import pytest

from buddy_manipulator.kinematics import Pose
from buddy_manipulator.teleop import (
    KEY_ENTER,
    KEY_ESCAPE,
    CartesianTeleopController,
)


def make_controller(**overrides) -> CartesianTeleopController:
    values = {
        "target": Pose(0.30, 0.08, 0.15, -math.pi / 2),
        "move_step": 0.01,
    }
    values.update(overrides)
    return CartesianTeleopController(**values)


def test_keyboard_moves_cartesian_target_and_gripper() -> None:
    controller = make_controller()
    controller.handle_key(ord("F"))
    controller.handle_key(ord("W"))
    controller.handle_key(ord("A"))
    controller.handle_key(ord("C"))

    assert (controller.target.x, controller.target.y, controller.target.z) == (
        pytest.approx(0.31),
        pytest.approx(0.09),
        pytest.approx(0.14),
    )
    assert controller.gripper == 0.0
    assert controller.joints().elbow < 0


def test_workspace_boundary_rejects_unsafe_target() -> None:
    controller = make_controller(
        target=Pose(0.42, 0.08, 0.15, -math.pi / 2)
    )
    controller.handle_key(ord("W"))
    assert controller.target.x == 0.42
    assert "rejected" in controller.last_message


def test_enter_finishes_and_escape_marks_abort() -> None:
    completed = make_controller()
    completed.handle_key(KEY_ENTER)
    assert completed.finished
    assert not completed.aborted

    aborted = make_controller()
    aborted.handle_key(KEY_ESCAPE)
    assert aborted.finished
    assert aborted.aborted
