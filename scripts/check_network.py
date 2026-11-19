"""Check three-agent delivery and expiration across six directed links."""

from swarm_core.communication.message import StateMessage
from swarm_core.communication.network import Network


def main():
    network = Network(3, delay_s=0.2, max_age_s=0.5)
    assert len(network.links) == 6

    for sender in range(3):
        message = StateMessage(
            sender_id=sender,
            sequence=0,
            generated_at_s=0.0,
            x=float(sender),
            y=0.0,
            heading=0.0,
            speed=0.0,
            yaw_rate=0.0,
            payload_size_bytes=64,
        )
        network.broadcast(message, now_s=0.0)

    assert network.deliver(0.1) == 0
    assert network.deliver(0.2) == 6

    for receiver in range(3):
        for sender in range(3):
            if sender != receiver:
                message, age, valid = network.read(receiver, sender, 0.2)
                assert message.sender_id == sender
                assert age == 0.2
                assert valid

    assert network.deliver(0.6) == 0
    for receiver in range(3):
        for sender in range(3):
            if sender != receiver:
                assert not network.read(receiver, sender, 0.6)[2]

    print("Six directed links: PASSED")
    print("All-to-all delayed state delivery: PASSED")
    print("Neighbor information expiration: PASSED")


if __name__ == "__main__":
    main()