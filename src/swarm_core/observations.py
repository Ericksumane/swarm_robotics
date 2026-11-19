"""Neighbor features built exclusively from received state messages.

Feature order:
[relative_x, relative_y, sin_relative_heading, cos_relative_heading,
 neighbor_speed, neighbor_yaw_rate, normalized_age]

Relative position is expressed in the receiving robot's body frame.
Positions, speeds and yaw rates retain SI units in this initial version.
Missing or expired rows are zeroed and masked out.
"""

import math

import numpy as np


NEIGHBOR_FEATURES = 7


def formation_pose_features(heading, direction=(1., 0.)):
    """Express the mission direction in the observing robot's body frame."""
    error = heading - math.atan2(direction[1], direction[0])
    return np.array([math.cos(error), -math.sin(error)], dtype=np.float32)


def build_neighbor_observation(
    own_state, receiver, neighbor_ids, now_s, predict_to_now=False
):
    """Return neighbor IDs, feature rows and a separate validity mask.

    own_state must be the robot's locally available estimate.
    No global simulator states are accepted by this function.
    """
    ids = tuple(sorted(neighbor_ids))
    if len(set(ids)) != len(ids):
        raise ValueError("Neighbor IDs must be unique.")
    if any(type(i) is not int or i < 0 for i in ids):
        raise ValueError("Neighbor IDs must be nonnegative integers.")
    if receiver.robot_id in ids:
        raise ValueError("Neighbor list must exclude the receiving robot.")

    own_values = (
        own_state.x, own_state.y, own_state.heading,
        own_state.speed, own_state.yaw_rate,
    )
    if not np.isfinite(own_values).all():
        raise ValueError("Own state must be finite.")

    features = np.zeros((len(ids), NEIGHBOR_FEATURES), dtype=np.float32)
    mask = np.zeros(len(ids), dtype=np.float32)

    c = math.cos(own_state.heading)
    s = math.sin(own_state.heading)

    for row, sender_id in enumerate(ids):
        message, age, valid = receiver.read(sender_id, now_s)
        if not valid:
            continue

        if predict_to_now:
            from swarm_core.communication.prediction import predict_received_pose
            peer_x,peer_y,peer_heading=predict_received_pose(message,age)
        else:
            peer_x,peer_y,peer_heading=message.x,message.y,message.heading
        dx = peer_x - own_state.x
        dy = peer_y - own_state.y
        relative_heading = peer_heading - own_state.heading

        features[row] = [
            c * dx + s * dy,
            -s * dx + c * dy,
            math.sin(relative_heading),
            math.cos(relative_heading),
            message.speed,
            message.yaw_rate,
            min(1., age / receiver.max_age_s),
        ]
        mask[row] = 1.0

    return {
        "neighbor_ids": ids,
        "neighbor_features": features,
        "neighbor_mask": mask,
    }

LOCAL_FEATURE_NAMES = (
    "own_speed_mps",
    "own_yaw_rate_radps",
    "max_forward_speed_mps",
    "max_reverse_speed_mps",
    "max_yaw_rate_radps",
    "max_linear_acceleration_mps2",
    "max_angular_acceleration_radps2",
    "footprint_radius_m",
    "desired_pair_distance_m",
)


def build_local_features(own_state, limits, desired_distance):
    """Build own-state and capability features in SI units.

    No neighbor state or global simulator state is used.
    Normalization is not applied in this initial version.
    """
    capability_keys = LOCAL_FEATURE_NAMES[2:-1]
    capabilities = [limits[key] for key in capability_keys]

    if not np.isfinite(capabilities).all():
        raise ValueError("Capabilities must be finite.")
    if any(value <= 0 for value in capabilities):
        raise ValueError("Capabilities must be positive.")
    if not np.isfinite(desired_distance) or desired_distance <= 0:
        raise ValueError("Desired distance must be positive and finite.")
    if not np.isfinite([own_state.speed, own_state.yaw_rate]).all():
        raise ValueError("Own velocities must be finite.")

    features = np.asarray(
        [
            own_state.speed,
            own_state.yaw_rate,
            *capabilities,
            desired_distance,
        ],
        dtype=np.float32,
    )
    return features

def append_formation_geometry(rows, mask, desired_distance):
    """Append relative distance and signed error, both in desired-distance units."""
    distance = np.linalg.norm(rows[:, :2], axis=1)
    ratio = distance / desired_distance
    geometry = np.stack((ratio, ratio - 1.), axis=1)
    geometry *= mask[:, None]
    return np.concatenate((rows, geometry), axis=1).astype(np.float32)
