"""Analysis utility to calculate and plot core point 1D density profiles."""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt


def compute_core_density_profile(
    core_points: List[np.ndarray],
    x_bounds: Tuple[float, float] = (-6.0, 6.0),
    sample_step: float = 0.05,
    window_half_width: float = 0.81,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes local core density along the x-axis within window_half_width (= r).
    """
    x_min, x_max = x_bounds
    x_samples = np.arange(x_min, x_max + sample_step * 0.5, sample_step)

    if not core_points:
        return x_samples, np.zeros_like(x_samples, dtype=int)

    core_x = np.array([p[0] for p in core_points])

    # Count core points where |p_x - x_sample| <= r
    diffs = np.abs(x_samples[:, np.newaxis] - core_x[np.newaxis, :])
    densities = np.sum(diffs <= window_half_width, axis=1)

    return x_samples, densities


def plot_core_density(
    core_points: List[np.ndarray],
    x_bounds: Tuple[float, float] = (-6.0, 6.0),
    sample_step: float = 0.02,
    r_neighbor: float = 0.81,
) -> None:
    """Generates and displays the line plot of x position vs local core density using r."""
    x_samples, densities = compute_core_density_profile(
        core_points=core_points,
        x_bounds=x_bounds,
        sample_step=sample_step,
        window_half_width=r_neighbor,
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        x_samples,
        densities,
        color="#28dc6e",
        linewidth=2.0,
        label=rf"Core Density ($|p_x - x| \leq {r_neighbor:.2f}$ m)",
    )

    # Mark x = 0 boundary where k_min switches
    plt.axvline(0.0, color="#ffd250", linestyle="--", linewidth=1.5, label="Switch Boundary (x = 0)")

    plt.title(f"Core Point Local Density vs. X Position (r = {r_neighbor:.2f} m)", fontsize=13, fontweight="bold")
    plt.xlabel("X Position (m)", fontsize=11)
    plt.ylabel("Number of Core Points", fontsize=11)
    plt.xlim(x_bounds[0], x_bounds[1])
    plt.ylim(bottom=0)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", framealpha=0.9)
    plt.tight_layout()
    plt.show()