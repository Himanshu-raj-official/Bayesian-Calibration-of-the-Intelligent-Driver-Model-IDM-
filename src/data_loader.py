
"""
Data loading utilities.

Two modes:
1. SYNTHETIC (default, works offline) — generates leader/follower
   trajectory pairs from known "true" IDM parameters plus measurement
   noise, so you can validate the whole pipeline (and reproduce the
   classical-vs-Bayesian identifiability finding) before touching
   real data.

2. REAL DATA — loaders for NGSIM and highD once you've downloaded
   them. See data/raw/README.md for download instructions.
"""

import numpy as np
import pandas as pd

from .idm_simulator import simulate_follower


# ---------------------------------------------------------------------
# 1. SYNTHETIC DATA
# ---------------------------------------------------------------------

TRUE_PARAMS_FREE_FLOW = dict(v0=30.0, T=1.5, a_max=1.5, b=2.0, s0=2.5)
TRUE_PARAMS_CONGESTED = dict(v0=30.0, T=1.2, a_max=1.0, b=2.5, s0=2.5)


def generate_leader_trajectory(n_steps, dt, regime="free_flow", seed=None):
    """
    Generate a synthetic leader speed profile.

    regime = "free_flow": leader cruises near desired speed with mild
             perturbations (low density, few interactions -> s0 becomes
             practically unidentifiable, which is the key finding this
             project reproduces).
    regime = "congested": leader does stop-and-go oscillations, which
             forces small headways and makes s0 identifiable.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps) * dt

    if regime == "free_flow":
        base = 28.0
        speed = base + 2.0 * np.sin(2 * np.pi * t / 40.0) + rng.normal(0, 0.15, n_steps)
        speed = np.clip(speed, 15, 32)
    elif regime == "congested":
        # stop-and-go wave
        speed = 10 + 8 * np.sin(2 * np.pi * t / 25.0) + rng.normal(0, 0.2, n_steps)
        speed = np.clip(speed, 0, 20)
    else:
        raise ValueError("regime must be 'free_flow' or 'congested'")

    pos = np.concatenate([[0.0], np.cumsum(speed[:-1] * dt)])
    return pos, speed


def generate_synthetic_pair(n_steps=600, dt=0.1, regime="free_flow",
                             noise_std=0.05, seed=None):
    """
    Generate one leader-follower trajectory pair with known true IDM
    parameters, plus Gaussian measurement noise on the follower's
    observed speed (simulating sensor/detection noise in real
    trajectory datasets like NGSIM).

    Returns
    -------
    df : pandas.DataFrame with columns
         [time, leader_pos, leader_speed, follower_pos, follower_speed_true,
          follower_speed_obs]
    true_params : dict of the IDM parameters used to generate the data
    """
    true_params = (TRUE_PARAMS_FREE_FLOW if regime == "free_flow"
                    else TRUE_PARAMS_CONGESTED)

    leader_pos, leader_speed = generate_leader_trajectory(n_steps, dt, regime, seed)

    init_gap = 25.0 if regime == "free_flow" else 12.0
    sim = simulate_follower(
        leader_pos, leader_speed, dt,
        init_pos=leader_pos[0] - init_gap,
        init_speed=leader_speed[0],
        **true_params,
    )

    rng = np.random.default_rng(None if seed is None else seed + 1)
    follower_speed_obs = sim["speed"] + rng.normal(0, noise_std, n_steps)

    df = pd.DataFrame({
        "time": np.arange(n_steps) * dt,
        "leader_pos": leader_pos,
        "leader_speed": leader_speed,
        "follower_pos": sim["position"],
        "follower_speed_true": sim["speed"],
        "follower_speed_obs": follower_speed_obs,
    })
    return df, true_params


def generate_multi_vehicle_dataset(n_vehicles=8, n_steps=600, dt=0.1,
                                    regime="free_flow", param_jitter=0.15,
                                    seed=42):
    """
    Generate a small fleet of follower vehicles, each with its OWN true
    parameters (drawn around the regime's base values with param_jitter
    relative spread). This is the dataset used for the hierarchical
    partial-pooling model, mimicking driver-to-driver heterogeneity.

    Returns
    -------
    dict[vehicle_id] -> (df, true_params)
    """
    base = TRUE_PARAMS_FREE_FLOW if regime == "free_flow" else TRUE_PARAMS_CONGESTED
    rng = np.random.default_rng(seed)
    dataset = {}

    for i in range(n_vehicles):
        params = {k: max(0.05, v * (1 + rng.normal(0, param_jitter)))
                  for k, v in base.items()}
        leader_pos, leader_speed = generate_leader_trajectory(
            n_steps, dt, regime, seed=seed + i)
        init_gap = 25.0 if regime == "free_flow" else 12.0
        sim = simulate_follower(leader_pos, leader_speed, dt,
                                 init_pos=leader_pos[0] - init_gap,
                                 init_speed=leader_speed[0], **params)
        noise_rng = np.random.default_rng(seed + 1000 + i)
        follower_speed_obs = sim["speed"] + noise_rng.normal(0, 0.05, n_steps)

        df = pd.DataFrame({
            "time": np.arange(n_steps) * dt,
            "leader_pos": leader_pos,
            "leader_speed": leader_speed,
            "follower_pos": sim["position"],
            "follower_speed_true": sim["speed"],
            "follower_speed_obs": follower_speed_obs,
        })
        dataset[f"veh_{i:02d}"] = (df, params)

    return dataset


# ---------------------------------------------------------------------
# 2. REAL DATA LOADERS (NGSIM / highD)
# ---------------------------------------------------------------------

def load_ngsim_csv(path, vehicle_id_col="Vehicle_ID", leader_id_col="Preceding",
                    time_col="Frame_ID", pos_col="Local_Y", speed_col="v_Vel",
                    fps=10, location_col=None, location_value=None,
                    min_frames=20, max_pairs=None):
    """
    Load a NGSIM trajectory CSV (e.g. I-80 or US-101 dataset) and return
    it in the same [time, leader_pos, leader_speed, follower_pos,
    follower_speed_obs] format used by the synthetic generator, for
    every follower vehicle matched to its recorded leader.

    This uses a single groupby pass (not a per-vehicle loop), so it
    stays fast even on multi-million-row files.

    Parameters
    ----------
    location_col, location_value : optional — if your CSV combines
        multiple locations (e.g. a "Location" column with values like
        "us-101", "i-80"), set both to filter to just one before
        processing. Strongly recommended on large combined exports.
    min_frames : drop pairs with fewer than this many matched frames.
    max_pairs  : optional cap on number of pairs returned (for a quick
        first pass on huge files — e.g. max_pairs=200).

    NOTE: You must download the raw NGSIM CSV yourself first — see
    data/raw/README.md. Adjust the *_col arguments if your export uses
    different column names than the classic NGSIM schema.
    """
    usecols = [vehicle_id_col, leader_id_col, time_col, pos_col, speed_col]
    if location_col:
        usecols.append(location_col)

    raw = pd.read_csv(path, usecols=usecols, low_memory=False)

    # Some NGSIM re-exports store numeric columns as text with comma
    # thousand-separators (e.g. "2,195.462") and/or store IDs as strings.
    # Coerce everything that should be numeric, stripping commas first.
    for col in [vehicle_id_col, leader_id_col, time_col, pos_col, speed_col]:
        if not pd.api.types.is_numeric_dtype(raw[col]):
            raw[col] = raw[col].astype(str).str.replace(",", "", regex=False)
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    if location_col and location_value is not None:
        if isinstance(location_value, (list, tuple, set)):
            raw = raw[raw[location_col].isin(location_value)]
        else:
            raw = raw[raw[location_col] == location_value]

    raw = raw.dropna(subset=[vehicle_id_col, leader_id_col, time_col, pos_col, speed_col])

    # Build the lookup table from ALL vehicles (including ones with no
    # leader of their own) — they may still be someone else's leader.
    raw = raw.sort_values([vehicle_id_col, time_col])
    grouped = {vid: g for vid, g in raw.groupby(vehicle_id_col)}

    pairs = []
    # Only vehicles that themselves HAVE a recorded leader are candidate followers.
    follower_ids = raw.loc[raw[leader_id_col] != 0, vehicle_id_col].unique()

    for vid in follower_ids:
        v = grouped[vid]
        lead_id = v[leader_id_col].iloc[0]
        lead = grouped.get(lead_id)
        if lead is None:
            continue

        merged = pd.merge(v, lead, on=time_col, suffixes=("_f", "_l"))
        if len(merged) < min_frames:
            continue

        df = pd.DataFrame({
            "time": (merged[time_col] - merged[time_col].iloc[0]) / fps,
            "leader_pos": merged[f"{pos_col}_l"].values / 3.28084,   # ft -> m
            "leader_speed": merged[f"{speed_col}_l"].values / 3.28084,
            "follower_pos": merged[f"{pos_col}_f"].values / 3.28084,
            "follower_speed_obs": merged[f"{speed_col}_f"].values / 3.28084,
        })
        pairs.append((vid, lead_id, df))

        if max_pairs and len(pairs) >= max_pairs:
            break

    return pairs


def load_highd_csv(tracks_path, static_path, follower_id):
    """
    Load a highD dataset trajectory for a given follower track id,
    matched against its preceding vehicle (highD already records
    'precedingId' and 'dhw' per frame). See data/raw/README.md for
    the academic-request download process (levelxdata.com).
    """
    tracks = pd.read_csv(tracks_path)
    v = tracks[tracks["id"] == follower_id].sort_values("frame")
    if v.empty:
        raise ValueError(f"No track found for follower_id={follower_id}")
    lead_id = v["precedingId"].iloc[0]
    if lead_id == 0:
        raise ValueError("Selected follower has no preceding vehicle recorded.")
    lead = tracks[tracks["id"] == lead_id].sort_values("frame")
    merged = pd.merge(v, lead, on="frame", suffixes=("_f", "_l"))

    df = pd.DataFrame({
        "time": (merged["frame"] - merged["frame"].iloc[0]) / 25.0,  # highD = 25fps
        "leader_pos": merged["x_l"].values,
        "leader_speed": merged["xVelocity_l"].abs().values,
        "follower_pos": merged["x_f"].values,
        "follower_speed_obs": merged["xVelocity_f"].abs().values,
    })
    return df
