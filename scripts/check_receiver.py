"""Verify receiver ordering, information age and expiration."""

from dataclasses import replace
import math

from swarm_core.communication.message import StateMessage
from swarm_core.communication.receiver import Receiver


def main():
    receiver = Receiver(robot_id=1, max_age_s=0.5)

    assert receiver.read(0, 0.0) == (None, None, False)

    message = StateMessage(
        sender_id=0,
        sequence=1,
        generated_at_s=0.0,
        x=1.0,
        y=0.0,
        heading=0.0,
        speed=0.0,
        yaw_rate=0.0,
        payload_size_bytes=64,
    )

    assert receiver.accept(message, now_s=0.2)
    assert not receiver.accept(message, now_s=0.2)
    assert not receiver.accept(
        replace(message, sequence=0), now_s=0.2
    )

    cached, age, valid = receiver.read(0, now_s=0.4)
    assert cached == message
    assert math.isclose(age, 0.4)
    assert valid

    cached, age, valid = receiver.read(0, now_s=0.6)
    assert cached == message
    assert math.isclose(age, 0.6)
    assert not valid

    newer = replace(message, sequence=2, generated_at_s=0.6)
    assert receiver.accept(newer, now_s=0.7)
    assert receiver.read(0, now_s=0.7)[2]

    print("Missing information handling: PASSED")
    print("Duplicate and out-of-order rejection: PASSED")
    print("Information age and expiration: PASSED")
    print("Reception recovery: PASSED")


if __name__ == "__main__":
    main()