"""Observable online calibration-bias estimation for structured placement."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class OnlinePlanarBiasEstimator:
    """Estimate persistent XY execution bias from commanded and observed outcomes."""

    smoothing: float = 0.5
    maximum_bias_mm: float = 100.0
    estimate_mm: np.ndarray = field(
        default_factory=lambda: np.zeros(2, dtype=np.float64)
    )
    observations: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.smoothing <= 1.0:
            raise ValueError("smoothing must be in (0, 1]")
        if self.maximum_bias_mm <= 0:
            raise ValueError("maximum bias must be positive")
        self.estimate_mm = np.asarray(self.estimate_mm, dtype=np.float64)
        if self.estimate_mm.shape != (2,):
            raise ValueError("initial estimate must have shape (2,)")

    def predicted_outcomes(self, command_candidates_mm: np.ndarray) -> np.ndarray:
        commands = np.asarray(command_candidates_mm, dtype=np.float32)
        if commands.ndim != 2 or commands.shape[1] != 2:
            raise ValueError("command candidates must have shape (candidates, 2)")
        return commands + self.estimate_mm.astype(np.float32)[None]

    def update(
        self,
        observed_object_xy_m: np.ndarray,
        target_xy_m: np.ndarray,
        command_xy_mm: np.ndarray,
    ) -> np.ndarray:
        observed = np.asarray(observed_object_xy_m, dtype=np.float64)
        target = np.asarray(target_xy_m, dtype=np.float64)
        command = np.asarray(command_xy_mm, dtype=np.float64)
        if observed.shape != (2,) or target.shape != (2,) or command.shape != (2,):
            raise ValueError("observed, target, and command XY values must have shape (2,)")
        measurement = (observed - target) * 1000.0 - command
        measurement = np.clip(
            measurement, -self.maximum_bias_mm, self.maximum_bias_mm
        )
        weight = 1.0 if self.observations == 0 else self.smoothing
        self.estimate_mm = (
            (1.0 - weight) * self.estimate_mm + weight * measurement
        )
        self.observations += 1
        return self.estimate_mm.copy()
