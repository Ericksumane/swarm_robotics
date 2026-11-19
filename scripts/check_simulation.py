"""Check motion and received-state integration under nominal communication."""

import json

import numpy as np

from swarm_core.simulation import Simulation


def main():
    with open("configs/tasks/formation_v1.json") as file:
        task = json.load(file)
    with open("configs/communication/nominal.json") as file:
        communication = json.load(file)

    simulation = Simulation(task, communication)
    initial = list(simulation.states)

    # Controlled inputs exercise the simulator, not a learned policy.
    actions = np.array([[0.5, 0.0], [0.0, 0.5], [0.0, 0.0]])
    simulation.advance(actions)

    assert simulation.states[0] != initial[0]
    assert simulation.states[1].heading != initial[1].heading
    assert simulation.states[2] == initial[2]

    for receiver in range(3):
        for sender in range(3):
            if sender == receiver:
                continue
            message, age, valid = simulation.network.read(
                receiver, sender, simulation.time_s
            )
            assert valid and age == 0.0
            assert message.x == simulation.states[sender].x
            assert message.y == simulation.states[sender].y

    simulation.reset()
    assert simulation.states == initial
    assert simulation.time_s == 0.0

    print("Motion and nominal communication integration: PASSED")
    print("Reproducible episode reset: PASSED")


if __name__ == "__main__":
    main()