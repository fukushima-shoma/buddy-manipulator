import math
import random

import pytest

mujoco = pytest.importorskip("mujoco")

from buddy_manipulator.goal_task import (
    ManipulationGoal,
    apply_object_positions,
    detect_goal_object,
    execute_pick_and_place,
    sample_object_positions,
)
from buddy_manipulator.kinematics import JointAngles
from buddy_manipulator.sim_camera import RgbdCamera
from buddy_manipulator.simulation import Keyframe, load_model, run_keyframes


def test_goal_vector_encodes_object_and_target() -> None:
    goal = ManipulationGoal("purple", "yellow")

    assert goal.vector().tolist() == [0.0, 1.0, 0.0, 1.0]
    assert goal.instruction == "place the purple block in the yellow zone"


def test_sampled_objects_are_separated() -> None:
    positions = sample_object_positions(random.Random(11))

    assert math.dist(positions["red"][:2], positions["purple"][:2]) >= 0.11
    assert all(position[2] == 0.025 for position in positions.values())
    assert all(0.23 <= position[0] <= 0.34 for position in positions.values())


@pytest.mark.parametrize(
    ("goal", "positions"),
    [
        (
            ManipulationGoal("red", "green"),
            {"red": (0.30, 0.09, 0.025), "purple": (0.22, 0.04, 0.025)},
        ),
        (
            ManipulationGoal("purple", "yellow"),
            {"red": (0.34, 0.04, 0.025), "purple": (0.232, 0.104, 0.025)},
        ),
    ],
)
def test_goal_expert_places_selected_object(goal, positions) -> None:
    model, data = load_model()
    apply_object_positions(model, data, positions)
    stowed = JointAngles(base=-1.2, shoulder=0.6, elbow=-1.2, wrist=0.6)
    run_keyframes(model, data, [Keyframe(0.7, stowed, 0.03)])

    with RgbdCamera(model, width=160, height=120) as camera:
        detection = detect_goal_object(camera.capture(data), goal)
        result = execute_pick_and_place(model, data, goal, detection)

    assert result.success
    assert result.selected_object_in_target
    assert not result.distractor_in_target
    assert result.placement_error_m <= 0.05
