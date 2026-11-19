"""Diagnostic episode returns; neither policy is a learned controller."""

import json

import numpy as np

from swarm_core.formation_env import FormationEnv


def load(path):
    with open(path) as file:
        return json.load(file)


def run(task, communication, weights, seed, mode, gamma):
    env = FormationEnv(task, communication, weights)
    env.reset(seed=seed)
    rng = np.random.default_rng(seed + 10000)

    total = 0.0
    discounted = 0.0
    discount = 1.0
    steps = 0

    while True:
        if mode == "stationary":
            actions = np.zeros((task["team"]["num_agents"], 2))
        else:
            actions = rng.uniform(
                -1.0, 1.0, size=(task["team"]["num_agents"], 2)
            )

        _, rewards, terminated, truncated, info = env.step(actions)
        reward = float(rewards[0])
        total += reward
        discounted += discount * reward
        discount *= gamma
        steps += 1

        if terminated or truncated:
            return {
                "seed": seed,
                "mode": mode,
                "steps": steps,
                "reason": info["reason"],
                "return": total,
                "discounted_return": discounted,
            }


def main():
    task = load("configs/tasks/formation_v1.json")
    communication = load("configs/communication/nominal.json")
    weights = load("configs/training/formation_reward_v1.json")

    # Diagnostic assumption only; training gamma is not finalized.
    gamma = 0.99
    records = []

    print(f"Diagnostic gamma: {gamma}")
    for seed in range(5):
        for mode in ("stationary", "random"):
            result = run(task, communication, weights, seed, mode, gamma)
            records.append(result)
            print(
                f"seed={seed} mode={mode:10s} "
                f"steps={result['steps']:4d} "
                f"reason={result['reason']:12s} "
                f"return={result['return']:.3f} "
                f"discounted={result['discounted_return']:.3f}"
            )

    with open("experiments/runs/return_audit.json", "w") as file:
        json.dump(
            {
                "gamma": gamma,
                "reward_version": weights["reward_version"],
                "records": records,
            },
            file,
            indent=2,
        )

    print("Saved: experiments/runs/return_audit.json")


if __name__ == "__main__":
    main()