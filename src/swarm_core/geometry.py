"""Geometry metrics for planar teams with equal circular footprints."""

from itertools import combinations

import numpy as np


def formation_metrics(states, desired_distance, footprint_radius):
    """Return center distances, formation errors, centroid and clearance.

    All-pairs equal-distance targets are intended for the initial
    three-agent task, not arbitrary team sizes.
    Clearance is signed: negative values indicate footprint overlap.
    """
    positions = np.asarray([(s.x, s.y) for s in states], dtype=float)
    if len(states) < 2 or not np.isfinite(positions).all():
        raise ValueError("At least two finite positions are required.")
    if not np.isfinite([desired_distance, footprint_radius]).all():
        raise ValueError("Geometry parameters must be finite.")
    if desired_distance <= 0 or footprint_radius <= 0:
        raise ValueError("Geometry parameters must be positive.")

    pairs = list(combinations(range(len(states)), 2))
    distances = np.array([
        np.linalg.norm(positions[i] - positions[j]) for i, j in pairs
    ])
    errors = distances - desired_distance

    return {
        "pairs": pairs,
        "pair_distances_m": distances,
        "formation_rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "max_pair_error_m": float(np.max(np.abs(errors))),
        "centroid_m": positions.mean(axis=0),
        "minimum_clearance_m": float(distances.min() - 2 * footprint_radius),
        "collision": bool(np.any(distances <= 2 * footprint_radius)),
    }
