"""Warm-start the shared formation actor from two Gazebo RosBot traces."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from swarm_core.external_simulation import ExternalSimulation
from swarm_core.learning.mappo_adapter import hidden_inputs, make_policy, pack


def examples(task, communication, directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    simulation = ExternalSimulation(task, communication)
    initial = manifest["initial_physical_states"]
    simulation.initialize_physics(initial, 0., 42)
    observations, actions = [], []
    limits = task["robot_defaults"]
    for tick, line in enumerate((directory / "control.jsonl").read_text().splitlines(), 1):
        sample = json.loads(line)
        observations.append(simulation.observe(1))
        speed, yaw = sample["applied_commands"][1]
        scale = limits["max_forward_speed_mps"] if speed >= 0 else limits["max_reverse_speed_mps"]
        actions.append((speed / scale, yaw / limits["max_yaw_rate_radps"]))
        simulation.offer(sample["states"], tick * simulation.dt)
        simulation.advance(np.zeros((3, 2)))
    return pack(observations), torch.tensor(actions, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(186992)
    runs = Path("experiments/runs")
    task = json.loads(Path("configs/tasks/mixed_formation_steering080.json").read_text())
    communication = json.loads(Path("configs/communication/nominal.json").read_text())
    bundle = torch.load(args.source, weights_only=True, map_location="cpu")
    component = bundle["formation"]["fresh_actor"]
    config, policy, _ = make_policy(
        local_dim=component["local_dim"], neighbor_dim=component["neighbor_dim"],
        critic_dim=component["critic_dim"], geometry_summary=component["geometry_summary"],
    )
    policy.actor.load_state_dict(component["actor"])
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=.0003)
    train = examples(task, communication, runs / "rosbot_replan_gazebo8103")
    validation = examples(task, communication, runs / "rosbot_replan_gazebo8304")

    def loss(data):
        inputs, labels = data
        hidden, masks = hidden_inputs(len(inputs), config)
        prediction, _, _ = policy.actor(inputs, hidden, masks, deterministic=True)
        return torch.nn.functional.mse_loss(prediction, labels)

    with torch.no_grad():
        initial = {"train_mse": loss(train).item(), "validation_mse": loss(validation).item()}
    best = initial["validation_mse"]
    best_epoch = 0
    weights = {key: value.clone() for key, value in policy.actor.state_dict().items()}
    for epoch in range(1, 151):
        optimizer.zero_grad()
        loss(train).backward()
        torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), .5)
        optimizer.step()
        with torch.no_grad():
            error = loss(validation).item()
            if error < best:
                best, best_epoch = error, epoch
                weights = {key: value.clone() for key, value in policy.actor.state_dict().items()}
    component["actor"] = weights
    policy.actor.load_state_dict(weights)
    args.output.mkdir(parents=True)
    candidate = args.output / "candidate_bundle.pt"
    bundle["status"] = "demonstration_warm_start_pending_evaluation"
    torch.save(bundle, candidate)
    with torch.no_grad():
        report = {
            "training_trace": 8103, "validation_trace": 8304,
            "train_samples": len(train[0]), "validation_samples": len(validation[0]),
            "initial": initial, "best_epoch": best_epoch,
            "train_mse": loss(train).item(), "validation_mse": best,
            "candidate_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        }
    (args.output / "progress.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
