"""Planar unicycle motion with bounded commands and acceleration.

State: x [m], y [m], heading [rad], speed [m/s], yaw rate [rad/s].
Action: two normalized requests in [-1, 1].

This model does not represent wheel slip or legged locomotion.
"""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RobotState:
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0
    speed: float = 0.0
    yaw_rate: float = 0.0


def step(state, action, limits, dt, *, steering_curvature=None):
    """Apply command limits and integrate constant-twist motion over dt."""
    action = np.asarray(action, dtype=float)
    state_values = [
        state.x, state.y, state.heading, state.speed, state.yaw_rate
    ]
    if action.shape != (2,) or not np.isfinite(action).all():
        raise ValueError("Action must contain two finite values.")
    if not np.isfinite(state_values).all():
        raise ValueError("State must be finite.")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("Time step must be positive and finite.")

    names = (
        "max_forward_speed_mps",
        "max_reverse_speed_mps",
        "max_yaw_rate_radps",
        "max_linear_acceleration_mps2",
        "max_angular_acceleration_radps2",
    )
    if any(not math.isfinite(limits[k]) or limits[k] <= 0 for k in names):
        raise ValueError("Motion limits must be positive and finite.")
    if not (
        -limits["max_reverse_speed_mps"]
        <= state.speed
        <= limits["max_forward_speed_mps"]
    ) or abs(state.yaw_rate) > limits["max_yaw_rate_radps"]:
        raise ValueError("Initial velocities exceed motion limits.")

    a = np.clip(action, -1.0, 1.0)
    linear_scale = (
        limits["max_forward_speed_mps"]
        if a[0] >= 0
        else limits["max_reverse_speed_mps"]
    )
    target_v = float(a[0] * linear_scale)
    target_w = float(a[1] * limits["max_yaw_rate_radps"])

    dv = limits["max_linear_acceleration_mps2"] * dt
    dw = limits["max_angular_acceleration_radps2"] * dt
    v = state.speed + float(np.clip(target_v - state.speed, -dv, dv))
    w = state.yaw_rate + float(np.clip(target_w - state.yaw_rate, -dw, dw))

    if steering_curvature is not None:
        if not math.isfinite(steering_curvature):
            raise ValueError("Steering curvature must be finite.")
        w = float(np.clip(v * steering_curvature, -limits["max_yaw_rate_radps"],
                          limits["max_yaw_rate_radps"]))

    # Use the updated velocities throughout this discrete interval.
    # sinc provides stable straight-line and circular-arc integration.
    turn = w * dt
    distance = v * dt * float(np.sinc(turn / (2.0 * math.pi)))
    midpoint_heading = state.heading + turn / 2.0

    x = state.x + distance * math.cos(midpoint_heading)
    y = state.y + distance * math.sin(midpoint_heading)
    heading = (state.heading + turn + math.pi) % (2.0 * math.pi) - math.pi

    return RobotState(x, y, heading, v, w)
