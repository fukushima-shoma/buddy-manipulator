from types import SimpleNamespace
import math

import numpy as np
import pytest

from buddy_manipulator.collect_demos import set_block_position
from buddy_manipulator.kinematics import JointAngles, Pose, inverse_kinematics
from buddy_manipulator.policy_rollout import (
    controlled_joint_positions,
    rate_limit_action,
    run_closed_loop_policy,
)
from buddy_manipulator.rollout_bc import AveragingPolicyRunner
from buddy_manipulator.simulation import (
    Keyframe,
    _mujoco,
    load_model,
    run_keyframes,
)


class ScriptedChunkPolicy:
    def __init__(self, actions: np.ndarray, horizon: int = 8) -> None:
        self.actions = actions
        self.horizon = horizon
        self.index = 0

    def predict_chunk(self, _rgb, _depth, _joint_position) -> np.ndarray:
        chunk = self.actions[self.index : self.index + self.horizon]
        self.index += self.horizon
        if len(chunk) < self.horizon:
            chunk = np.concatenate(
                [chunk, np.repeat(chunk[-1:], self.horizon - len(chunk), axis=0)]
            )
        return chunk


class EmptyCamera:
    def capture(self, _data):
        return SimpleNamespace(
            rgb=np.zeros((12, 16, 3), dtype=np.uint8),
            depth=np.ones((12, 16), dtype=np.float32),
        )


class ConstantChunkPolicy:
    def __init__(self, value: float, horizon: int = 3) -> None:
        self.value = value
        self.action_horizon = horizon
        self.device = "cpu"

    def predict_chunk(self, _rgb, _depth, _joint_position) -> np.ndarray:
        return np.full((self.action_horizon, 6), self.value)


def test_rate_limit_action_limits_delta_and_actuator_range() -> None:
    model = SimpleNamespace(
        actuator_ctrlrange=np.asarray(
            [
                (-1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, 1.0),
                (-1.0, 1.0),
                (0.0, 0.03),
                (0.0, 0.03),
            ]
        )
    )
    current = np.asarray((0.9, 0.0, 0.0, 0.0, 0.02, 0.02))
    requested = np.asarray((2.0, -0.8, 0.1, -0.1, -1.0, 1.0))

    limited = rate_limit_action(
        model,
        current,
        requested,
        max_arm_delta=0.2,
        max_gripper_delta=0.01,
    )

    assert limited == pytest.approx((1.0, -0.2, 0.1, -0.1, 0.01, 0.03))


def test_averaging_policy_runner_averages_matching_chunks() -> None:
    runner = AveragingPolicyRunner(
        [ConstantChunkPolicy(1.0), ConstantChunkPolicy(3.0)]
    )

    chunk = runner.predict_chunk(None, None, None)

    assert runner.action_horizon == 3
    assert chunk == pytest.approx(np.full((3, 6), 2.0))


def test_controlled_joint_positions_follow_model_order() -> None:
    model, data = load_model()
    expected = np.asarray((0.2, 0.3, -0.8, -1.0, 0.01, 0.02))
    data.qpos[:6] = expected
    _mujoco().mj_forward(model, data)

    positions = controlled_joint_positions(model, data)

    assert positions.shape == (6,)
    assert positions == pytest.approx(expected)


def test_chunk_executor_can_complete_scripted_grasp() -> None:
    model, data = load_model()
    set_block_position(model, data, (0.30, 0.08, 0.025))
    stowed = JointAngles(base=-1.2, shoulder=0.6, elbow=-1.2, wrist=0.6)
    run_keyframes(model, data, [Keyframe(0.7, stowed, 0.03)])

    def action_for(pose: Pose, gripper: float) -> np.ndarray:
        joints = inverse_kinematics(pose, elbow_up=True)
        return np.asarray((*joints.as_tuple(), gripper, gripper))

    pregrasp = action_for(Pose(0.30, 0.08, 0.15, -math.pi / 2), 0.03)
    grasp_open = action_for(Pose(0.30, 0.08, 0.05, -math.pi / 2), 0.03)
    grasp_closed = action_for(Pose(0.30, 0.08, 0.05, -math.pi / 2), 0.0)
    lift = action_for(Pose(0.30, 0.08, 0.20, -math.pi / 2), 0.0)
    actions = np.concatenate(
        [
            np.repeat(pregrasp[None], 8, axis=0),
            np.repeat(grasp_open[None], 6, axis=0),
            np.repeat(grasp_closed[None], 5, axis=0),
            np.repeat(lift[None], 8, axis=0),
        ]
    )

    result = run_closed_loop_policy(
        model,
        data,
        ScriptedChunkPolicy(actions),
        EmptyCamera(),
        execute_chunk_steps=8,
    )

    assert result.success
    assert result.lift_delta >= 0.05
    assert result.finger_contact_count > 0
