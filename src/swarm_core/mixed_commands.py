"""Command limits shared by mixed-platform training and Gazebo."""

import math


def update_rosbot_curvature(curvature, linear_speed, yaw_rate, dt, time_constant_s):
    """Track a curvature command with first-order steering response."""
    if not math.isfinite(time_constant_s) or time_constant_s <= 0:
        raise ValueError("Steering time constant must be positive and finite.")
    target = yaw_rate / linear_speed if linear_speed else 0.
    gain = -math.expm1(-dt / time_constant_s)
    return curvature + gain * (target - curvature)

GO2_LINEAR_DEADBAND = .015
GO2_YAW_DEADBAND = .08
ROSBOT_TURN_RADIUS = 1.02

# Linear and yaw deadbands, minimum curve radius, can turn in place.
FORMATION_CAPABILITIES = (
    (GO2_LINEAR_DEADBAND, GO2_YAW_DEADBAND, 0., 1.),
    (0., 0., ROSBOT_TURN_RADIUS, 0.),
    (0., 0., 0., 1.),
)

def constrain_mixed_commands(commands, *, go2_deadband=True, rosbot_curvature=True):
    """Apply the Go2 deadband and RosBot steering limit in SI units."""
    go2_v, go2_w = commands[0]
    if go2_deadband and abs(go2_v) < GO2_LINEAR_DEADBAND and abs(go2_w) < GO2_YAW_DEADBAND:
        go2_v = go2_w = 0.
    rosbot_v, rosbot_w = commands[1]
    max_yaw_rate = abs(rosbot_v) / ROSBOT_TURN_RADIUS
    if rosbot_curvature:
        rosbot_w = max(-max_yaw_rate, min(max_yaw_rate, rosbot_w))
    return [
        [go2_v, go2_w],
        [rosbot_v, rosbot_w],
        commands[2],
    ]
