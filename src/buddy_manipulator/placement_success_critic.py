"""Outcome-trained critic for bounded release-pose selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from buddy_manipulator.goal_task import ManipulationGoal
from buddy_manipulator.kinematics import JointAngles, forward_kinematics


PLACEMENT_CONTEXT_DIM = 9
PLACEMENT_CRITIC_INPUT_DIM = 11


def placement_context_features(
    joint_position: np.ndarray,
    goal: ManipulationGoal,
    target_position_m: np.ndarray,
) -> np.ndarray:
    joints = np.asarray(joint_position, dtype=np.float32)
    if joints.shape != (6,):
        raise ValueError("joint position must have shape (6,)")
    held = forward_kinematics(
        JointAngles(*(float(value) for value in joints[:4]))
    )
    target = np.asarray(target_position_m, dtype=np.float32)
    if target.shape != (3,):
        raise ValueError("target position must have shape (3,)")
    return np.concatenate(
        [
            np.asarray([held.x, held.y, held.z], dtype=np.float32),
            target[:2],
            goal.vector(),
        ]
    ).astype(np.float32)


def placement_critic_features(
    context: np.ndarray,
    residual_xy_mm: np.ndarray,
) -> np.ndarray:
    context = np.asarray(context, dtype=np.float32)
    residual = np.asarray(residual_xy_mm, dtype=np.float32)
    if context.shape != (PLACEMENT_CONTEXT_DIM,):
        raise ValueError("placement context must have shape (9,)")
    if residual.shape != (2,):
        raise ValueError("release residual must have shape (2,)")
    return np.concatenate([context, residual]).astype(np.float32)


@dataclass(frozen=True)
class PlacementFeatureStats:
    mean: np.ndarray
    std: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, values: dict[str, list[float]]) -> "PlacementFeatureStats":
        return cls(
            mean=np.asarray(values["mean"], dtype=np.float32),
            std=np.asarray(values["std"], dtype=np.float32),
        )


def compute_placement_feature_stats(features: np.ndarray) -> PlacementFeatureStats:
    if features.ndim != 2 or features.shape[1] != PLACEMENT_CRITIC_INPUT_DIM:
        raise ValueError("placement features must have shape (samples, 11)")
    return PlacementFeatureStats(
        mean=features.mean(axis=0).astype(np.float32),
        std=np.maximum(features.std(axis=0), np.float32(1e-6)).astype(np.float32),
    )


def grid_release_residuals(
    injected_bias_xy_mm: np.ndarray,
    *,
    step_mm: float = 10.0,
    maximum_command_mm: float = 30.0,
) -> np.ndarray:
    bias = np.asarray(injected_bias_xy_mm, dtype=np.float32)
    if bias.shape != (2,):
        raise ValueError("injected bias must have shape (2,)")
    if step_mm <= 0 or maximum_command_mm < 0:
        raise ValueError("candidate step must be positive and maximum non-negative")
    values = np.arange(
        -maximum_command_mm,
        maximum_command_mm + step_mm * 0.5,
        step_mm,
        dtype=np.float32,
    )
    if not np.any(np.isclose(values, 0.0)):
        values = np.sort(np.append(values, np.float32(0.0)))
    commands = np.asarray([(x, y) for x in values for y in values], dtype=np.float32)
    return commands + bias[None]


class PlacementSuccessCritic(nn.Module):
    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(PLACEMENT_CRITIC_INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != PLACEMENT_CRITIC_INPUT_DIM:
            raise ValueError("features must have shape (batch, 11)")
        return self.network(features).squeeze(-1)


@dataclass
class PlacementSuccessCriticRunner:
    model: PlacementSuccessCritic
    feature_stats: PlacementFeatureStats
    device: torch.device

    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: Path, *, device_name: str = "auto"
    ) -> "PlacementSuccessCriticRunner":
        from buddy_manipulator.behavior_cloning import choose_device

        device = choose_device(device_name)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        if checkpoint.get("policy_type") != "placement_success_critic":
            raise ValueError("checkpoint is not a placement success critic")
        model = PlacementSuccessCritic(
            hidden_dim=int(checkpoint["model_config"].get("hidden_dim", 64))
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return cls(
            model=model,
            feature_stats=PlacementFeatureStats.from_dict(checkpoint["feature_stats"]),
            device=device,
        )

    def choose_residual(
        self,
        joint_position: np.ndarray,
        goal: ManipulationGoal,
        target_position_m: np.ndarray,
        candidates_xy_mm: np.ndarray,
        *,
        fallback_xy_mm: np.ndarray,
        minimum_score_improvement: float = 0.03,
    ) -> tuple[np.ndarray, np.ndarray]:
        candidates = np.asarray(candidates_xy_mm, dtype=np.float32)
        fallback = np.asarray(fallback_xy_mm, dtype=np.float32)
        if candidates.ndim != 2 or candidates.shape[1] != 2:
            raise ValueError("release candidates must have shape (candidates, 2)")
        if fallback.shape != (2,):
            raise ValueError("fallback must have shape (2,)")
        if minimum_score_improvement < 0:
            raise ValueError("minimum score improvement must be non-negative")
        matches = np.flatnonzero(
            np.all(np.isclose(candidates, fallback[None], atol=1e-5), axis=1)
        )
        if matches.size == 0:
            raise ValueError("fallback must be present in candidates")
        context = placement_context_features(joint_position, goal, target_position_m)
        features = np.stack(
            [placement_critic_features(context, candidate) for candidate in candidates]
        )
        normalized = (features - self.feature_stats.mean) / self.feature_stats.std
        with torch.no_grad():
            probabilities = torch.sigmoid(
                self.model(torch.from_numpy(normalized).to(self.device))
            ).cpu().numpy()
        selected_index = int(np.argmax(probabilities))
        fallback_index = int(matches[0])
        if probabilities[selected_index] < (
            probabilities[fallback_index] + minimum_score_improvement
        ):
            selected_index = fallback_index
        return candidates[selected_index].astype(np.float64), probabilities.astype(np.float64)
