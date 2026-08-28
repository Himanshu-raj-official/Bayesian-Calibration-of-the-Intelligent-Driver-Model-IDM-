"""Shared metrics and plotting helpers."""

import numpy as np
import matplotlib.pyplot as plt


def rmse(a, b):
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def param_table(true_params, fitted_params, param_names=("v0", "T", "a_max", "b", "s0")):
    """Print a quick comparison table of true vs fitted IDM parameters."""
    print(f"{'param':<8}{'true':>10}{'fitted':>10}{'abs err':>10}{'% err':>10}")
    for name in param_names:
        t = true_params[name]
        f = fitted_params[name]
        err = abs(f - t)
        pct = 100 * err / t if t != 0 else float("nan")
        print(f"{name:<8}{t:>10.3f}{f:>10.3f}{err:>10.3f}{pct:>9.1f}%")


def plot_trajectories(df, title=""):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["time"], df["leader_speed"], label="leader speed")
    axes[0].plot(df["time"], df["follower_speed_obs"], label="follower speed (obs)")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("speed [m/s]")
    axes[0].legend()
    axes[0].set_title("Speeds")

    gap = df["leader_pos"] - df["follower_pos"]
    axes[1].plot(df["time"], gap, color="tab:green")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("gap [m]")
    axes[1].set_title("Bumper-to-bumper gap")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_posterior_vs_true(idata, true_params, param_names=("v0", "T", "a_max", "b", "s0")):
    """Plot posterior histograms with a vertical line at the true value
    (only meaningful for synthetic data where ground truth is known)."""
    fig, axes = plt.subplots(1, len(param_names), figsize=(4 * len(param_names), 3.5))
    for ax, name in zip(axes, param_names):
        samples = idata.posterior[name].values.flatten()
        ax.hist(samples, bins=40, alpha=0.7, color="tab:blue")
        ax.axvline(true_params[name], color="red", linestyle="--", label="true")
        ax.set_title(name)
        ax.legend()
    fig.tight_layout()
    return fig
