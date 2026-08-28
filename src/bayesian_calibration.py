"""
Bayesian IDM calibration using PyMC.

Two models are provided:

1. build_single_vehicle_model()
   One-step-ahead Bayesian model for a SINGLE vehicle. Priors are
   TruncatedNormal on each of the 5 IDM parameters (v0, T, a_max, b, s0),
   centered on plausible values with realistic spread. The likelihood
   compares IDM-predicted acceleration against observed (differentiated)
   acceleration with a Normal noise model.

2. build_hierarchical_model()
   Partial-pooling model across MULTIPLE vehicles. Each vehicle gets its
   own parameter draw from a population-level Normal distribution
   (hyper-priors on population mean/sd per parameter). This lets
   information be shared across vehicles ("pooled") while still
   allowing individual variation ("unpooled"), which is more
   statistically honest than either fitting each vehicle in complete
   isolation or forcing all vehicles to share one parameter set.

Both models use PyMC's NUTS sampler via pm.sample().
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def _idm_accel_pt(v, delta_v, s, v0, T, a_max, b, s0, delta=4.0):
    """PyTensor version of the IDM acceleration law (differentiable,
    used inside the PyMC model graph)."""
    s_safe = pt.maximum(s, 1e-3)
    s_star = s0 + pt.maximum(0.0, v * T + (v * delta_v) / (2 * pt.sqrt(a_max * b)))
    accel = a_max * (1 - (v / v0) ** delta - (s_star / s_safe) ** 2)
    return accel


def build_single_vehicle_model(leader_pos, leader_speed, follower_pos,
                                follower_speed, obs_accel,
                                priors=None):
    """
    Build (but do not sample) a PyMC model for single-vehicle IDM
    calibration.

    priors: optional dict overriding the default TruncatedNormal prior
            (mu, sigma, lower, upper) tuples for each of
            v0, T, a_max, b, s0.
    """
    default_priors = {
        "v0":    dict(mu=25.0, sigma=8.0, lower=5.0, upper=40.0),
        "T":     dict(mu=1.5, sigma=0.8, lower=0.3, upper=4.0),
        "a_max": dict(mu=1.2, sigma=0.6, lower=0.3, upper=4.0),
        "b":     dict(mu=2.0, sigma=1.0, lower=0.3, upper=5.0),
        "s0":    dict(mu=2.0, sigma=1.5, lower=0.1, upper=8.0),
    }
    if priors:
        default_priors.update(priors)
    p = default_priors

    with pm.Model() as model:
        v0 = pm.TruncatedNormal("v0", **p["v0"])
        T = pm.TruncatedNormal("T", **p["T"])
        a_max = pm.TruncatedNormal("a_max", **p["a_max"])
        b = pm.TruncatedNormal("b", **p["b"])
        s0 = pm.TruncatedNormal("s0", **p["s0"])

        v = pt.as_tensor_variable(follower_speed)
        delta_v = follower_speed - leader_speed
        s = leader_pos - follower_pos

        pred_accel = _idm_accel_pt(v, delta_v, s, v0, T, a_max, b, s0)

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)
        pm.Normal("accel_obs", mu=pred_accel, sigma=sigma_obs, observed=obs_accel)

    return model


def build_hierarchical_model(vehicle_data, priors=None):
    """
    Build a hierarchical (partial-pooling) PyMC model across multiple
    vehicles.

    Parameters
    ----------
    vehicle_data : list of dicts, one per vehicle, each with keys
        'leader_pos','leader_speed','follower_pos','follower_speed','obs_accel'
        (all 1D numpy arrays, can have different lengths per vehicle).

    Returns a PyMC model with population-level hyper-priors
    (mu_pop, sigma_pop) for each of the 5 parameters, and per-vehicle
    parameters drawn from Normal(mu_pop, sigma_pop) then truncated by
    construction via a softplus-like reparam (here: simple Normal, with
    bounds enforced softly through the physics — negative v0/a_max/b/s0
    are penalized naturally by the likelihood since they produce
    nonsensical accelerations). For strict positivity we sample in
    log-space for the strictly-positive parameters.

    Vehicles can have unequal-length trajectories; we build one
    likelihood term per vehicle inside a loop (PyMC handles this fine
    at model-build time, even though it's not fully vectorized).
    """
    n_veh = len(vehicle_data)

    with pm.Model() as model:
        # population-level hyper-priors (on log-scale for positive params)
        mu_log_v0 = pm.Normal("mu_log_v0", mu=np.log(25.0), sigma=0.3)
        sd_log_v0 = pm.HalfNormal("sd_log_v0", sigma=0.3)

        mu_log_T = pm.Normal("mu_log_T", mu=np.log(1.5), sigma=0.3)
        sd_log_T = pm.HalfNormal("sd_log_T", sigma=0.3)

        mu_log_amax = pm.Normal("mu_log_amax", mu=np.log(1.2), sigma=0.3)
        sd_log_amax = pm.HalfNormal("sd_log_amax", sigma=0.3)

        mu_log_b = pm.Normal("mu_log_b", mu=np.log(2.0), sigma=0.3)
        sd_log_b = pm.HalfNormal("sd_log_b", sigma=0.3)

        mu_log_s0 = pm.Normal("mu_log_s0", mu=np.log(2.0), sigma=0.5)
        sd_log_s0 = pm.HalfNormal("sd_log_s0", sigma=0.5)

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=1.0)

        # per-vehicle (non-centered parameterization for better sampling)
        z_v0 = pm.Normal("z_v0", 0, 1, shape=n_veh)
        z_T = pm.Normal("z_T", 0, 1, shape=n_veh)
        z_amax = pm.Normal("z_amax", 0, 1, shape=n_veh)
        z_b = pm.Normal("z_b", 0, 1, shape=n_veh)
        z_s0 = pm.Normal("z_s0", 0, 1, shape=n_veh)

        v0_veh = pm.Deterministic("v0_veh", pt.exp(mu_log_v0 + z_v0 * sd_log_v0))
        T_veh = pm.Deterministic("T_veh", pt.exp(mu_log_T + z_T * sd_log_T))
        amax_veh = pm.Deterministic("amax_veh", pt.exp(mu_log_amax + z_amax * sd_log_amax))
        b_veh = pm.Deterministic("b_veh", pt.exp(mu_log_b + z_b * sd_log_b))
        s0_veh = pm.Deterministic("s0_veh", pt.exp(mu_log_s0 + z_s0 * sd_log_s0))

        for i, veh in enumerate(vehicle_data):
            v = veh["follower_speed"]
            delta_v = veh["follower_speed"] - veh["leader_speed"]
            s = veh["leader_pos"] - veh["follower_pos"]

            pred_accel = _idm_accel_pt(
                v, delta_v, s,
                v0_veh[i], T_veh[i], amax_veh[i], b_veh[i], s0_veh[i],
            )
            pm.Normal(f"accel_obs_{i}", mu=pred_accel, sigma=sigma_obs,
                      observed=veh["obs_accel"])

    return model


def sample_model(model, draws=1000, tune=1000, chains=4, target_accept=0.9,
                  random_seed=42):
    """Convenience wrapper around pm.sample with sensible defaults for
    this problem (IDM posteriors can have tricky geometry, hence the
    raised target_accept).

    All logging/warning output during sampling (including divergence
    notices, OpenMP warnings, etc.) is suppressed regardless of which
    internal logger name PyMC/pytensor happens to use — logging.disable
    blocks everything at the manager level rather than relying on
    guessing specific logger names.
    """
    import logging
    import warnings

    previous_disable_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with model:
                idata = pm.sample(
                    draws=draws, tune=tune, chains=chains,
                    target_accept=target_accept, random_seed=random_seed,
                    return_inferencedata=True,
                )
    finally:
        logging.disable(previous_disable_level)
    return idata
