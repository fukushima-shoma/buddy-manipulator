"""Seeded simulation randomization for Phase 5 robustness evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
from typing import Any

import numpy as np

from buddy_manipulator.goal_task import OBJECT_BODIES
from buddy_manipulator.simulation import _mujoco


@dataclass(frozen=True)
class DomainRandomizationConfig:
    finger_friction_scale: tuple[float, float] = (0.7, 1.3)
    object_mass_scale: tuple[float, float] = (0.6, 1.6)
    appearance_scale: tuple[float, float] = (0.7, 1.15)
    place_bias_jitter_mm: float = 15.0

    def __post_init__(self) -> None:
        for name in (
            "finger_friction_scale",
            "object_mass_scale",
            "appearance_scale",
        ):
            lower, upper = getattr(self, name)
            if lower <= 0.0 or upper < lower:
                raise ValueError(f"invalid {name} range")
        if self.place_bias_jitter_mm < 0.0:
            raise ValueError("place bias jitter must be non-negative")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class DomainRandomizationSample:
    finger_friction_scale: float
    object_mass_scale: float
    appearance_scale: float
    place_bias_jitter_xy_mm: tuple[float, float]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def sample_domain_randomization(
    rng: random.Random,
    config: DomainRandomizationConfig,
) -> DomainRandomizationSample:
    """Sample one reproducible combined stress condition."""
    return DomainRandomizationSample(
        finger_friction_scale=rng.uniform(*config.finger_friction_scale),
        object_mass_scale=rng.uniform(*config.object_mass_scale),
        appearance_scale=rng.uniform(*config.appearance_scale),
        place_bias_jitter_xy_mm=(
            rng.uniform(-config.place_bias_jitter_mm, config.place_bias_jitter_mm),
            rng.uniform(-config.place_bias_jitter_mm, config.place_bias_jitter_mm),
        ),
    )


def _body_geom_ids(model: Any, body_id: int) -> range:
    start = int(model.body_geomadr[body_id])
    return range(start, start + int(model.body_geomnum[body_id]))


def apply_domain_randomization(
    model: Any,
    data: Any,
    sample: DomainRandomizationSample,
) -> None:
    """Apply dynamics and appearance changes to a freshly loaded model."""
    mujoco = _mujoco()
    for geom_name in ("left_finger_pad", "right_finger_pad"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id < 0:
            raise KeyError(f"unknown geom: {geom_name}")
        model.geom_friction[geom_id, 0] *= sample.finger_friction_scale

    for body_name in OBJECT_BODIES.values():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise KeyError(f"unknown body: {body_name}")
        model.body_mass[body_id] *= sample.object_mass_scale
        model.body_inertia[body_id] *= sample.object_mass_scale
        for geom_id in _body_geom_ids(model, body_id):
            model.geom_rgba[geom_id, :3] = np.clip(
                model.geom_rgba[geom_id, :3] * sample.appearance_scale,
                0.0,
                1.0,
            )
    mujoco.mj_forward(model, data)
