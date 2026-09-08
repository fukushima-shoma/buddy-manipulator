"""Outcome-trained critic for ranking bounded grasp-pose candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from buddy_manipulator.goal_task import ManipulationGoal, goal_grasp_keyframes
from buddy_manipulator.kinematics import UnreachableTargetError
from buddy_manipulator.grasp_pose_residual import (
    GRASP_POSE_FEATURE_DIM,
    grasp_pose_features,
)
from buddy_manipulator.sim_camera import RgbdFrame
from buddy_manipulator.vision import WorldDetection


GRASP_CRITIC_INPUT_DIM = GRASP_POSE_FEATURE_DIM + 3


def grid_candidate_residuals(
    known_bias_y_mm: float,
    *,
    step_mm: float = 3.0,
    maximum_command_mm: float = 9.0,
) -> np.ndarray:
    """Build resulting XY residuals for a grid of bounded controller commands."""
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
    commands = np.asarray(
        [(x, y, 0.0) for x in values for y in values], dtype=np.float32
    )
    commands[:, 1] += np.float32(known_bias_y_mm)
    return commands


@dataclass(frozen=True)
class GraspCriticFeatureStats:
    mean: np.ndarray
    std: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, values: dict[str, list[float]]) -> "GraspCriticFeatureStats":
        return cls(
            mean=np.asarray(values["mean"], dtype=np.float32),
            std=np.asarray(values["std"], dtype=np.float32),
        )


def grasp_critic_features(
    pose_features: np.ndarray,
    residual_mm: np.ndarray,
) -> np.ndarray:
    if pose_features.shape != (GRASP_POSE_FEATURE_DIM,):
        raise ValueError("pose features must have shape (8,)")
    residual = np.asarray(residual_mm, dtype=np.float32)
    if residual.shape != (3,):
        raise ValueError("residual must have shape (3,)")
    return np.concatenate([pose_features, residual]).astype(np.float32)


def compute_critic_feature_stats(features: np.ndarray) -> GraspCriticFeatureStats:
    if features.ndim != 2 or features.shape[1] != GRASP_CRITIC_INPUT_DIM:
        raise ValueError("critic features must have shape (samples, 11)")
    return GraspCriticFeatureStats(
        mean=features.mean(axis=0).astype(np.float32),
        std=np.maximum(features.std(axis=0), np.float32(1e-6)).astype(np.float32),
    )


def apply_grasp_residual(
    detection: WorldDetection,
    residual_mm: np.ndarray,
) -> WorldDetection:
    residual = np.asarray(residual_mm, dtype=np.float64)
    if residual.shape != (3,):
        raise ValueError("residual must have shape (3,)")
    corrected = np.asarray(detection.position) + residual / 1000.0
    return WorldDetection(
        pixel=detection.pixel,
        x=float(corrected[0]),
        y=float(corrected[1]),
        z=float(corrected[2]),
        depth=detection.depth,
    )


class GraspSuccessCritic(nn.Module):
    """Predict a grasp-success logit from scene and candidate pose features."""

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError("hidden dimension must be positive")
        self.hidden_dim = hidden_dim
        self.network = nn.Sequential(
            nn.Linear(GRASP_CRITIC_INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != GRASP_CRITIC_INPUT_DIM:
            raise ValueError("features must have shape (batch, 11)")
        return self.network(features).squeeze(-1)


@dataclass
class GraspSuccessCriticRunner:
    model: GraspSuccessCritic
    feature_stats: GraspCriticFeatureStats
    device: torch.device

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path,
        *,
        device_name: str = "auto",
    ) -> "GraspSuccessCriticRunner":
        from buddy_manipulator.behavior_cloning import choose_device

        device = choose_device(device_name)
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if checkpoint.get("format_version") != 1:
            raise ValueError("unsupported checkpoint format")
        if checkpoint.get("policy_type") != "grasp_success_critic":
            raise ValueError("checkpoint is not a grasp success critic")
        model = GraspSuccessCritic(
            hidden_dim=int(checkpoint["model_config"].get("hidden_dim", 64))
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return cls(
            model=model,
            feature_stats=GraspCriticFeatureStats.from_dict(
                checkpoint["feature_stats"]
            ),
            device=device,
        )

    def choose_detection(
        self,
        frame: RgbdFrame,
        goal: ManipulationGoal,
        candidate_residuals_mm: np.ndarray,
        *,
        fallback_residual_mm: np.ndarray | None = None,
        minimum_score_improvement: float = 0.0,
    ) -> tuple[WorldDetection, np.ndarray, np.ndarray]:
        if minimum_score_improvement < 0.0:
            raise ValueError("minimum score improvement must be non-negative")
        candidates = np.asarray(candidate_residuals_mm, dtype=np.float32)
        if candidates.ndim != 2 or candidates.shape[1] != 3:
            raise ValueError("candidate residuals must have shape (candidates, 3)")
        fallback = np.zeros(3, dtype=np.float32)
        if fallback_residual_mm is not None:
            fallback = np.asarray(fallback_residual_mm, dtype=np.float32)
        if fallback.shape != (3,):
            raise ValueError("fallback residual must have shape (3,)")
        fallback_matches = np.flatnonzero(
            np.all(np.isclose(candidates, fallback[None], atol=1e-5), axis=1)
        )
        if fallback_matches.size == 0:
            raise ValueError("fallback residual must be present in candidates")
        pose_features, baseline = grasp_pose_features(frame, goal)
        features = np.stack(
            [grasp_critic_features(pose_features, residual) for residual in candidates]
        )
        normalized = (
            features - self.feature_stats.mean
        ) / self.feature_stats.std
        with torch.no_grad():
            probabilities = torch.sigmoid(
                self.model(torch.from_numpy(normalized).to(self.device))
            ).cpu().numpy()
        selected_index = int(np.argmax(probabilities))
        fallback_index = int(fallback_matches[0])
        if (
            probabilities[selected_index]
            < probabilities[fallback_index] + minimum_score_improvement
        ):
            selected_index = fallback_index
        selected_residual = candidates[selected_index]
        return (
            apply_grasp_residual(baseline, selected_residual),
            selected_residual.astype(np.float64),
            probabilities.astype(np.float64),
        )


@dataclass
class OutcomeRankedGraspPosePolicy:
    """Expose a safe critic-ranked pose through the hierarchical policy interface."""

    runner: GraspSuccessCriticRunner
    known_bias_y_mm: float = 0.0
    candidate_step_mm: float = 3.0
    maximum_command_mm: float = 9.0
    minimum_score_improvement: float = 0.03
    last_residual_mm: np.ndarray | None = None

    @property
    def transition_mode(self) -> str:
        return "outcome_ranked_grasp"

    def predict_detection(
        self,
        frame: RgbdFrame,
        goal: ManipulationGoal,
    ) -> WorldDetection:
        candidates = grid_candidate_residuals(
            self.known_bias_y_mm,
            step_mm=self.candidate_step_mm,
            maximum_command_mm=self.maximum_command_mm,
        )
        fallback = np.asarray([0.0, self.known_bias_y_mm, 0.0], dtype=np.float32)
        _, baseline = grasp_pose_features(frame, goal)
        reachable = []
        for candidate in candidates:
            try:
                goal_grasp_keyframes(apply_grasp_residual(baseline, candidate))
            except UnreachableTargetError:
                continue
            reachable.append(candidate)
        reachable_candidates = np.asarray(reachable, dtype=np.float32)
        detection, selected, _ = self.runner.choose_detection(
            frame,
            goal,
            reachable_candidates,
            fallback_residual_mm=fallback,
            minimum_score_improvement=self.minimum_score_improvement,
        )
        self.last_residual_mm = selected
        return detection
