import pytest

mujoco = pytest.importorskip("mujoco")

from buddy_manipulator.demo import demo_motion
from buddy_manipulator.simulation import load_model, run_keyframes


def test_model_loads_and_demo_runs_headless() -> None:
    model, data = load_model()
    run_keyframes(model, data, demo_motion())
    assert model.nu == 6
    assert data.time >= 7.5
