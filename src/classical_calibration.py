"""
Classical (non-Bayesian) IDM calibration via least-squares.

Fits the 5 IDM parameters by minimizing the sum of squared errors
between the one-step-ahead IDM-predicted acceleration and the
observed acceleration (from differentiated speed), using
scipy.optimize.least_squares with bounds.

This is the baseline the project compares against the Bayesian
approach — the headline result is that this method returns a single
point estimate that can look "confident" and correct even when the
underlying parameter (s0 in free flow) is barely identifiable from
the data.
"""

import numpy as np
from scipy.optimize import least_squares

from .idm_simulator import one_step_predictions, observed_acceleration

# (lower, upper) bounds for [v0, T, a_max, b, s0]
DEFAULT_BOUNDS = (
    [5.0, 0.3, 0.3, 0.3, 0.1],
    [40.0, 4.0, 4.0, 5.0, 8.0],
)
DEFAULT_INIT = [25.0, 1.5, 1.2, 2.0, 2.0]


def _residuals(theta, leader_pos, leader_speed, follower_pos, follower_speed, dt):
    v0, T, a_max, b, s0 = theta
    pred_accel = one_step_predictions(
        leader_pos, leader_speed, follower_pos, follower_speed,
        dt, v0, T, a_max, b, s0,
    )
    obs_accel = observed_acceleration(follower_speed, dt)
    return pred_accel - obs_accel


def calibrate_lsq(df, dt, x0=None, bounds=None):
    """
    Fit IDM parameters to a single vehicle's trajectory DataFrame
    (as produced by data_loader) using nonlinear least squares.

    Returns
    -------
    dict with fitted params, the scipy OptimizeResult, and RMSE.
    """
    x0 = DEFAULT_INIT if x0 is None else x0
    bounds = DEFAULT_BOUNDS if bounds is None else bounds

    result = least_squares(
        _residuals, x0, bounds=bounds,
        args=(df["leader_pos"].values, df["leader_speed"].values,
              df["follower_pos"].values, df["follower_speed_obs"].values, dt),
        method="trf",
    )

    v0, T, a_max, b, s0 = result.x
    rmse = np.sqrt(np.mean(result.fun ** 2))

    return {
        "v0": v0, "T": T, "a_max": a_max, "b": b, "s0": s0,
        "rmse": rmse,
        "success": result.success,
        "raw_result": result,
    }


def calibrate_lsq_multi_start(df, dt, n_starts=10, seed=0, bounds=None):
    """
    Robustness check: run LSQ from multiple random initial guesses
    within bounds and return all fits plus the best one. Useful for
    demonstrating identifiability issues — if many different starting
    points converge to wildly different s0 values with similarly low
    RMSE, that's evidence of a flat/ridge-shaped likelihood surface
    that only Bayesian posterior geometry reveals cleanly.
    """
    bounds = DEFAULT_BOUNDS if bounds is None else bounds
    rng = np.random.default_rng(seed)
    lo, hi = np.array(bounds[0]), np.array(bounds[1])

    fits = []
    for _ in range(n_starts):
        x0 = rng.uniform(lo, hi)
        fits.append(calibrate_lsq(df, dt, x0=x0, bounds=bounds))

    best = min(fits, key=lambda f: f["rmse"])
    return fits, best
