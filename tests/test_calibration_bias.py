import numpy as np
import pytest

from buddy_manipulator.calibration_bias import OnlinePlanarBiasEstimator


def test_online_bias_estimator_subtracts_known_command() -> None:
    estimator = OnlinePlanarBiasEstimator(smoothing=0.5)

    first = estimator.update(
        np.asarray([0.28, 0.21]),
        np.asarray([0.32, 0.16]),
        np.asarray([10.0, 0.0]),
    )
    second = estimator.update(
        np.asarray([0.29, 0.20]),
        np.asarray([0.32, 0.16]),
        np.asarray([10.0, 0.0]),
    )

    assert first == pytest.approx([-50.0, 50.0])
    assert second == pytest.approx([-45.0, 45.0])
    assert estimator.observations == 2


def test_online_bias_estimator_offsets_candidate_commands() -> None:
    estimator = OnlinePlanarBiasEstimator(estimate_mm=np.asarray([-40.0, 5.0]))
    outcomes = estimator.predicted_outcomes(
        np.asarray([[0.0, 0.0], [30.0, -10.0]], dtype=np.float32)
    )

    np.testing.assert_allclose(outcomes, [[-40.0, 5.0], [-10.0, -5.0]])


def test_online_bias_estimator_validates_configuration() -> None:
    with pytest.raises(ValueError, match="smoothing"):
        OnlinePlanarBiasEstimator(smoothing=0.0)
