"""Check that packet loss prevents neighbor-state access by observations."""

from copy import deepcopy
import json

import numpy as np

from swarm_core.simulation import Simulation


def main():
    with open("configs/tasks/formation_v1.json") as file:
        task = json.load(file)
    with open("configs/communication/nominal.json") as file:
        communication = json.load(file)

    nominal = Simulation(task, communication)
    for observation in nominal.observe_all():
        assert observation["local_features"].shape == (9,)
        assert observation["neighbor_features"].shape == (2, 7)
        np.testing.assert_array_equal(
            observation["neighbor_mask"], [1, 1]
        )

    blocked_config = deepcopy(communication)
    blocked_config["loss_probability"] = 1.0
    blocked = Simulation(task, blocked_config)

    # Physical neighbor states exist, but no messages were received.
    assert len(blocked.states) == 3
    for observation in blocked.observe_all():
        assert not observation["neighbor_mask"].any()
        assert not observation["neighbor_features"].any()
        assert np.isfinite(observation["local_features"]).all()

    print("Nominal local observation integration: PASSED")
    print("No hidden neighbor state under complete packet loss: PASSED")


if __name__ == "__main__":
    main()