import random

import numpy as np
import pytest

from buddy_manipulator.domain_randomization import (
    DomainRandomizationConfig,
    apply_domain_randomization,
    sample_domain_randomization,
)
from buddy_manipulator.goal_task import configure_goal_scene
from buddy_manipulator.simulation import _mujoco, load_model


def test_domain_randomization_sample_is_seeded_and_bounded() -> None:
    config = DomainRandomizationConfig()
    first = sample_domain_randomization(random.Random(17), config)
    second = sample_domain_randomization(random.Random(17), config)

    assert first == second
    assert 0.7 <= first.finger_friction_scale <= 1.3
    assert 0.6 <= first.object_mass_scale <= 1.6
    assert 0.7 <= first.appearance_scale <= 1.15
    assert all(abs(value) <= 15.0 for value in first.place_bias_jitter_xy_mm)


def test_domain_randomization_changes_model_parameters() -> None:
    model, data = load_model()
    configure_goal_scene(model, data)
    mujoco = _mujoco()
    finger_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, "left_finger_pad"
    )
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
    geom_id = int(model.body_geomadr[body_id])
    friction = float(model.geom_friction[finger_id, 0])
    mass = float(model.body_mass[body_id])
    color = model.geom_rgba[geom_id, :3].copy()

    sample = sample_domain_randomization(
        random.Random(23),
        DomainRandomizationConfig(
            finger_friction_scale=(0.8, 0.8),
            object_mass_scale=(1.4, 1.4),
            appearance_scale=(0.75, 0.75),
            place_bias_jitter_mm=0.0,
        ),
    )
    apply_domain_randomization(model, data, sample)

    assert model.geom_friction[finger_id, 0] == pytest.approx(friction * 0.8)
    assert model.body_mass[body_id] == pytest.approx(mass * 1.4)
    np.testing.assert_allclose(model.geom_rgba[geom_id, :3], color * 0.75)


def test_domain_randomization_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="finger_friction"):
        DomainRandomizationConfig(finger_friction_scale=(1.0, 0.5))
    with pytest.raises(ValueError, match="jitter"):
        DomainRandomizationConfig(place_bias_jitter_mm=-1.0)
