"""Semantic phase encoding for the scripted Phase 5 manipulation task."""

from __future__ import annotations

import bisect

import numpy as np


GOAL_PHASE_NAMES = (
    "approach",
    "descend",
    "close",
    "lift",
    "retract",
    "rotate",
    "extend",
    "lower",
    "open",
    "retreat",
)
GOAL_PHASE_DURATIONS_SECONDS = (
    1.2,
    1.0,
    1.0,
    1.5,
    1.2,
    2.5,
    1.5,
    1.5,
    1.0,
    1.0,
)
GOAL_PHASE_DIM = len(GOAL_PHASE_NAMES)
GOAL_TRAJECTORY_SECONDS = sum(GOAL_PHASE_DURATIONS_SECONDS)


def goal_phase_index(elapsed_seconds: float) -> int:
    """Map elapsed task time to one of the expert's semantic motion phases."""
    if elapsed_seconds < 0:
        raise ValueError("elapsed_seconds must be non-negative")
    boundaries = np.cumsum(GOAL_PHASE_DURATIONS_SECONDS).tolist()
    return min(bisect.bisect_right(boundaries, elapsed_seconds), GOAL_PHASE_DIM - 1)


def encode_goal_phase(elapsed_seconds: float) -> np.ndarray:
    """Return a one-hot phase vector compatible with training and rollout."""
    phase = np.zeros(GOAL_PHASE_DIM, dtype=np.float32)
    phase[goal_phase_index(elapsed_seconds)] = 1.0
    return phase
