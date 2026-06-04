"""Contracts for the Duan smearing post-prediction correction."""

import numpy as np
import pytest

from bike_sharing.postprocess import apply_smearing, compute_smearing_factor


def test_compute_smearing_factor_is_mean_exp_residuals():
    r = np.array([-0.2, 0.0, 0.1, 0.3])
    assert compute_smearing_factor(r) == pytest.approx(float(np.mean(np.exp(r))))


def test_compute_smearing_factor_not_forced_above_one():
    # All-negative residuals -> theta < 1; the factor must not be clamped.
    r = np.array([-0.5, -0.3, -0.4])
    theta = compute_smearing_factor(r)
    assert theta < 1.0
    assert theta == pytest.approx(float(np.mean(np.exp(r))))


def test_apply_smearing_exact_formula_and_nonnegative():
    log_pred = np.log(np.array([2.0, 5.0, 0.1]))  # exp(log_pred) = [2, 5, 0.1]
    theta = 1.2
    out = apply_smearing(log_pred, theta)
    expected = np.clip(theta * np.exp(log_pred) - 1.0, 0.0, None)
    assert np.allclose(out, expected)
    # Third element: 1.2 * 0.1 - 1 = -0.88 -> clipped to 0.
    assert out[2] == 0.0
    assert (out >= 0).all()
    # It must NOT be the incorrect theta * expm1(log_pred) form.
    wrong = np.clip(theta * np.expm1(log_pred), 0.0, None)
    assert not np.allclose(out, wrong)


def test_apply_smearing_theta_one_is_expm1_noop():
    # theta=1 makes theta*exp(log_pred)-1 == expm1(log_pred), the uncorrected
    # inverse, so smearing is a no-op there.
    log_pred = np.log(np.array([2.0, 5.0, 10.0]))
    assert np.allclose(apply_smearing(log_pred, 1.0), np.expm1(log_pred))
