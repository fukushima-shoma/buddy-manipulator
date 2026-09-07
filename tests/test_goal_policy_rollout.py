import argparse

import pytest

from buddy_manipulator.rollout_goal_bc import parse_goal


def test_parse_goal_builds_structured_goal() -> None:
    goal = parse_goal("purple:yellow")

    assert goal.object_color == "purple"
    assert goal.target_color == "yellow"


def test_parse_goal_rejects_invalid_value() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_goal("red")
    with pytest.raises(argparse.ArgumentTypeError):
        parse_goal("blue:green")
