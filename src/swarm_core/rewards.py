"""Team reward for the initial formation-and-hold task.

Uses true simulator states for training reward calculation.
These states must not be exposed to decentralized actors.
"""

import math

import numpy as np

from swarm_core.geometry import formation_metrics


def formation_hold_reward(states, limits, desired_distance, dt, weights):
    """Return a scalar team reward and separately logged components.

    Collision penalty is an event penalty. The environment must terminate
    on collision so it is not repeatedly charged on subsequent steps.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Time step must be positive and finite.")

    for name in ("formation_weight", "motion_weight", "collision_penalty"):
        if not math.isfinite(weights[name]) or weights[name] < 0:
            raise ValueError("Reward weights must be finite and nonnegative.")

    speed_scale = max(
        limits["max_forward_speed_mps"],
        limits["max_reverse_speed_mps"],
    )
    yaw_scale = limits["max_yaw_rate_radps"]
    if not np.isfinite([speed_scale, yaw_scale]).all():
        raise ValueError("Velocity scales must be finite.")
    if speed_scale <= 0 or yaw_scale <= 0:
        raise ValueError("Velocity scales must be positive.")

    geometry = formation_metrics(
        states, desired_distance, limits["footprint_radius_m"]
    )

    formation_error = (
        geometry["formation_rmse_m"] / desired_distance
    ) ** 2

    motion_cost = float(np.mean([
        (state.speed / speed_scale) ** 2
        + (state.yaw_rate / yaw_scale) ** 2
        for state in states
    ]))
    if not math.isfinite(motion_cost):
        raise ValueError("Motion cost must be finite.")

    formation_term = -dt * weights["formation_weight"] * formation_error
    motion_term = -dt * weights["motion_weight"] * motion_cost
    collision_term = (
        -weights["collision_penalty"] if geometry["collision"] else 0.0
    )

    terms = {
        "formation": formation_term,
        "motion": motion_term,
        "collision": collision_term,
    }
    return float(sum(terms.values())), terms