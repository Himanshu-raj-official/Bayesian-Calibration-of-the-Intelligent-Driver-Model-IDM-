"""
Intelligent Driver Model (IDM) — forward simulator.

The IDM describes car-following behaviour with 5 parameters:
    v0     : desired (free-flow) speed                [m/s]
    T      : desired time headway                      [s]
    a_max  : maximum acceleration                       [m/s^2]
    b      : comfortable braking deceleration           [m/s^2]
    s0     : minimum (jam) gap                          [m]

Given a leader trajectory (position & speed over time), the simulator
propagates a following vehicle forward in time using the IDM
acceleration law. This is the core physics model that both the
classical least-squares calibration and the Bayesian calibration
try to fit to observed trajectory data.

Reference: Treiber, Hennecke & Helbing (2000), "Congested traffic
states in empirical observations and microscopic simulations."
"""

import numpy as np


def idm_acceleration(v, delta_v, s, v0, T, a_max, b, s0, delta=4.0):
    """
    Compute IDM acceleration for a single vehicle-following pair.

    Parameters
    ----------
    v       : float or array, follower speed [m/s]
    delta_v : float or array, approaching rate = v_follower - v_leader [m/s]
              (positive = follower closing the gap)
    s       : float or array, bumper-to-bumper gap to leader [m]
    v0, T, a_max, b, s0 : IDM parameters (see module docstring)
    delta   : acceleration exponent (usually fixed at 4)

    Returns
    -------
    acceleration [m/s^2]
    """
    s = np.maximum(s, 1e-3)  # avoid division by zero / negative gaps
    s_star = s0 + np.maximum(0.0, v * T + (v * delta_v) / (2 * np.sqrt(a_max * b)))
    accel = a_max * (1 - (v / v0) ** delta - (s_star / s) ** 2)
    return accel


def simulate_follower(leader_pos, leader_speed, dt, v0, T, a_max, b, s0,
                       init_pos, init_speed, delta=4.0, v_min=0.0):
    """
    Simulate a following vehicle's trajectory given a leader's trajectory
    and a set of IDM parameters, using simple Euler integration.

    Parameters
    ----------
    leader_pos, leader_speed : 1D arrays, leader trajectory at each timestep
    dt        : timestep [s]
    v0,T,a_max,b,s0 : IDM parameters
    init_pos, init_speed : initial condition of the follower
    v_min     : floor on follower speed (no reversing)

    Returns
    -------
    dict with 'position', 'speed', 'accel' arrays (same length as leader arrays)
    """
    n = len(leader_pos)
    pos = np.zeros(n)
    speed = np.zeros(n)
    accel = np.zeros(n)

    pos[0] = init_pos
    speed[0] = init_speed

    for t in range(n - 1):
        s = leader_pos[t] - pos[t]
        delta_v = speed[t] - leader_speed[t]
        a = idm_acceleration(speed[t], delta_v, s, v0, T, a_max, b, s0, delta)
        accel[t] = a

        speed[t + 1] = max(v_min, speed[t] + a * dt)
        pos[t + 1] = pos[t] + speed[t] * dt + 0.5 * a * dt ** 2

    # final-step acceleration for completeness
    s = leader_pos[-1] - pos[-1]
    delta_v = speed[-1] - leader_speed[-1]
    accel[-1] = idm_acceleration(speed[-1], delta_v, s, v0, T, a_max, b, s0, delta)

    return {"position": pos, "speed": speed, "accel": accel}


def one_step_predictions(leader_pos, leader_speed, follower_pos, follower_speed,
                          dt, v0, T, a_max, b, s0, delta=4.0):
    """
    Given an OBSERVED follower trajectory (not simulated), compute the
    IDM-predicted acceleration at every timestep using the observed
    state (gap, speed, closing rate) at that timestep.

    This is the "one-step-ahead" formulation used for calibration:
    instead of propagating errors forward through a full simulation,
    we ask "given where the vehicles actually were, what acceleration
    would IDM have predicted?" This avoids compounding integration
    error and gives a clean, differentiable-in-parameters likelihood
    for Bayesian inference.

    Returns
    -------
    predicted acceleration array, same length as input trajectories
    """
    s = leader_pos - follower_pos
    delta_v = follower_speed - leader_speed
    return idm_acceleration(follower_speed, delta_v, s, v0, T, a_max, b, s0, delta)


def observed_acceleration(speed, dt):
    """
    Estimate observed/'true' acceleration from a speed time series via
    central differences (used as the calibration target).
    """
    accel = np.gradient(speed, dt)
    return accel
