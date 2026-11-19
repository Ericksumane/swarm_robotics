"""Check delayed delivery, complete loss and directed-link independence."""

from swarm_core.communication.channel import Channel
from swarm_core.communication.message import StateMessage


def main():
    message = StateMessage(
        sender_id=0,
        sequence=0,
        generated_at_s=0.0,
        x=0.0,
        y=0.0,
        heading=0.0,
        speed=0.0,
        yaw_rate=0.0,
        payload_size_bytes=64,
    )

    delayed = Channel(0, 1, delay_s=0.2)
    assert delayed.send(message, now_s=0.0)
    assert delayed.receive(0.1) == []
    assert delayed.receive(0.2) == [(0.2, message)]
    assert delayed.receive(0.3) == []
    assert delayed.delivered == 1

    blocked = Channel(0, 1, loss_probability=1.0)
    assert not blocked.send(message, now_s=0.0)
    assert blocked.receive(1.0) == []
    assert blocked.dropped == 1

    # The same outgoing state can reach another receiver.
    other_link = Channel(0, 2)
    assert other_link.send(message, now_s=0.0)
    assert other_link.receive(0.0) == [(0.0, message)]

    print("Delayed delivery without duplicates: PASSED")
    print("Complete packet loss: PASSED")
    print("Independent directed links: PASSED")


if __name__ == "__main__":
    main()