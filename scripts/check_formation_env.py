"""Check environment stepping, truncation, reset and shared rewards."""

from copy import deepcopy
import json

import numpy as np

from swarm_core.formation_env import FormationEnv


def load(path):
    with open(path) as file:
        return json.load(file)


def main():
    task = load("configs/tasks/formation_v1.json")
    communication = load("configs/communication/nominal.json")
    weights = load("configs/training/formation_reward_v1.json")

    # A short episode exercises truncation without running a training job.
    short_task = deepcopy(task)
    short_task["simulation"]["episode_time_limit_s"] = 0.3
    env = FormationEnv(short_task, communication, weights)
    initial, _ = env.reset(seed=42)

    actions = np.zeros((3, 2), dtype=np.float32)
    for _ in range(3):
        observations, rewards, terminated, truncated, info = env.step(actions)
        assert len(observations) == 3
        assert rewards.shape == (3,)
        assert np.isfinite(rewards).all()
        assert np.all(rewards == rewards[0])

    assert truncated and not terminated
    assert info["reason"] == "time_limit"

    try:
        env.step(actions)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Stepping after episode end was accepted.")

    restored, info = env.reset(seed=42)
    assert info["time_s"] == 0.0
    for before, after in zip(initial, restored):
        for key in ("local_features", "neighbor_features", "neighbor_mask"):
            np.testing.assert_array_equal(before[key], after[key])

    print("Environment stepping and shared rewards: PASSED")
    print("Time-limit handling and post-end guard: PASSED")
    print("Complete reproducible reset: PASSED")


if __name__ == "__main__":
    main()