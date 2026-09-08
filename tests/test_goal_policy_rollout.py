import argparse

import numpy as np
import pytest

from buddy_manipulator.goal_policy_rollout import resolve_execute_chunk_steps
from buddy_manipulator.hierarchical_goal_policy import (
    LiftTransitionGate,
    SkillGoalPolicyRunner,
)
from buddy_manipulator.rollout_goal_bc import GoalAveragingPolicyRunner, parse_goal


class FakeGoalPolicy:
    def __init__(self, value: float) -> None:
        self.value = value
        self.action_horizon = 3
        self.device = "cpu"
        self.model = type(
            "ModelConfig",
            (),
            {
                "goal_dim": 4,
                "phase_dim": 0,
                "history_horizon": 8,
                "use_object_features": False,
                "use_goal_object_features": True,
                "use_goal_target_features": True,
                "factorized_target_heads": False,
                "target_residual_heads": False,
                "target_residual_scale": 0.25,
            },
        )()
        self.reset_seed = None

    def reset(self, seed=None) -> None:
        self.reset_seed = seed

    def predict_chunk(self, _rgb, _depth, _joint, _goal, _phase=None):
        return np.full((3, 6), self.value)


def test_parse_goal_builds_structured_goal() -> None:
    goal = parse_goal("purple:yellow")

    assert goal.object_color == "purple"
    assert goal.target_color == "yellow"


def test_parse_goal_rejects_invalid_value() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_goal("red")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_goal("blue:green")


def test_goal_ensemble_averages_compatible_policies() -> None:
    first = FakeGoalPolicy(1.0)
    second = FakeGoalPolicy(3.0)
    ensemble = GoalAveragingPolicyRunner([first, second])

    ensemble.reset(17)
    chunk = ensemble.predict_chunk(None, None, None, np.ones(4))

    assert chunk == pytest.approx(np.full((3, 6), 2.0))
    assert first.reset_seed == second.reset_seed == 17
    assert resolve_execute_chunk_steps(ensemble, 0) == 1
    assert resolve_execute_chunk_steps(ensemble, 2) == 2


def test_lift_transition_requires_stable_visual_lift_and_closed_gripper() -> None:
    gate = LiftTransitionGate(required_observations=2)
    open_joints = np.asarray([0.0, 0.0, 0.0, 0.0, 0.03, 0.03])
    closed_joints = np.zeros(6)

    assert gate.update(0.12, open_joints) is False
    assert gate.update(0.05, closed_joints) is False
    assert gate.update(0.12, closed_joints) is False
    assert gate.update(0.12, closed_joints) is True
    assert gate.update(None, closed_joints) is False


def test_skill_goal_runner_masks_irrelevant_goal_factor() -> None:
    policy = FakeGoalPolicy(1.0)
    captured = []
    policy.predict_chunk = (
        lambda _r, _d, _j, goal, _p=None: captured.append(goal.copy())
        or np.ones((3, 6))
    )

    SkillGoalPolicyRunner(policy, "grasp").predict_chunk(
        None, None, None, np.ones(4)
    )
    SkillGoalPolicyRunner(policy, "place").predict_chunk(
        None, None, None, np.ones(4)
    )

    assert captured[0].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert captured[1].tolist() == [0.0, 0.0, 1.0, 1.0]


def test_handoff_place_runner_preserves_object_for_visual_grounding() -> None:
    policy = FakeGoalPolicy(1.0)
    policy.goal_skill = "handoff_place"
    captured = []
    policy.predict_chunk = (
        lambda _r, _d, _j, goal, _p=None: captured.append(goal.copy())
        or np.ones((3, 6))
    )

    SkillGoalPolicyRunner(policy, "place").predict_chunk(
        None, None, None, np.ones(4)
    )

    assert captured[0].tolist() == [1.0, 1.0, 1.0, 1.0]
