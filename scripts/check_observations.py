"""Verify relative coordinates and exclusion of unavailable information."""

import math

import numpy as np

from swarm_core.communication.message import StateMessage
from swarm_core.communication.receiver import Receiver
from swarm_core.dynamics import RobotState
from swarm_core.observations import build_neighbor_observation


def main():
    receiver = Receiver(robot_id=0, max_age_s=0.5)
    own = RobotState(x=1.0, y=1.0, heading=math.pi / 2)

    message = StateMessage(
        sender_id=1,
        sequence=0,
        generated_at_s=0.0,
        x=1.0,
        y=2.0,
        heading=math.pi / 2,
        speed=0.1,
        yaw_rate=0.0,
        payload_size_bytes=64,
    )
    receiver.accept(message, now_s=0.0)

    observation = build_neighbor_observation(
        own, receiver, [1, 2], now_s=0.0
    )
    features = observation["neighbor_features"]
    mask = observation["neighbor_mask"]

    # World +Y is forward when the receiving robot faces +Y.
    np.testing.assert_allclose(features[0, :2], [1, 0], atol=1e-6)
    np.testing.assert_array_equal(mask, [1, 0])
    np.testing.assert_array_equal(features[1], np.zeros(7))

    expired = build_neighbor_observation(
        own, receiver, [1, 2], now_s=0.6
    )
    assert not expired["neighbor_mask"].any()
    assert not expired["neighbor_features"].any()

    print("Body-frame relative position: PASSED")
    print("Missing neighbor masking: PASSED")
    print("Expired information exclusion: PASSED")


if __name__ == "__main__":
    main()