"""Verify terminal penalties through controlled environment fixtures."""

import json

import numpy as np

from swarm_core.dynamics import RobotState
from swarm_core.formation_env import FormationEnv


def load(path):
    with open(path) as file:
        return json.load(file)


def main():
    task = load("configs/tasks/formation_v1.json")
    communication = load("configs/communication/nominal.json")
    weights = load("configs/training/formation_reward_v1.json")
    env = FormationEnv(task, communication, weights)
    actions = np.zeros((3, 2), dtype=np.float32)

    # Direct state assignment is only a test fixture, not an actor input.
    env.reset()
    env.simulation.states = [
        RobotState(x=10.0, y=0.0),
        RobotState(x=8.0, y=0.0),
        RobotState(x=8.0, y=2.0),
    ]
    _, rewards, terminated, truncated, info = env.step(actions)

    assert terminated and not truncated
    assert info["reason"] == "outside_area"
    assert info["reward_terms"]["outside_area"] == -weights["outside_area_penalty"]
    assert info["reward_terms"]["collision"] == 0.0
    np.testing.assert_allclose(
        rewards,
        sum(info["reward_terms"].values()),
        rtol=1e-6,
    )

    env.reset()
    env.simulation.states = [
        RobotState(x=0.0, y=0.0),
        RobotState(x=0.1, y=0.0),
        RobotState(x=2.0, y=0.0),
    ]
    _, rewards, terminated, truncated, info = env.step(actions)

    assert terminated and not truncated
    assert info["reason"] == "collision"
    assert info["reward_terms"]["collision"] == -weights["collision_penalty"]
    assert info["reward_terms"]["outside_area"] == 0.0
    np.testing.assert_allclose(
        rewards,
        sum(info["reward_terms"].values()),
        rtol=1e-6,
    )

    print("Outside-area terminal penalty: PASSED")
    print("Collision terminal penalty: PASSED")
    print("Shared reward component accounting: PASSED")


if __name__ == "__main__":
    main()