"""Load and validate the initial nominal communication configuration."""

import json
import math

from swarm_core.communication.network import Network


def main():
    with open("configs/communication/nominal.json") as file:
        communication = json.load(file)

    with open("configs/tasks/formation_v1.json") as file:
        task = json.load(file)

    frequency = communication["transmission_frequency_hz"]
    dt = task["simulation"]["time_step_s"]

    assert math.isfinite(frequency) and frequency > 0
    assert math.isfinite(dt) and dt > 0

    # Initial scheduling will transmit every integer number of control steps.
    steps_per_message = 1.0 / (frequency * dt)
    assert steps_per_message >= 1.0
    assert math.isclose(
        steps_per_message, round(steps_per_message), abs_tol=1e-9
    ), "Transmission period must be an integer multiple of the time step."

    size = communication["payload_size_bytes"]
    assert type(size) is int and size > 0

    network = Network(
        num_agents=task["team"]["num_agents"],
        delay_s=communication["delay_s"],
        loss_probability=communication["loss_probability"],
        max_age_s=communication["max_information_age_s"],
        seed=task["seed"],
    )

    print("Profile:", communication["profile_name"])
    print("Directed links:", len(network.links))
    print("Control steps per transmission:", round(steps_per_message))
    print("Communication configuration: PASSED")


if __name__ == "__main__":
    main()