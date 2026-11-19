"""Warm-start the shared pose actor from nominal Gazebo formation traces."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import torch

from cooperative_pose_reference import CooperativePoseReference, CooperativeTranslationReference
from swarm_core.external_simulation import ExternalSimulation
from swarm_core.mixed_commands import ROSBOT_TURN_RADIUS
from swarm_core.simulation import Simulation
from swarm_core.learning.mappo_adapter import hidden_inputs, make_policy, pack


def load(path):
    return json.loads(path.read_text())


def examples(task, communication, run, robot_ids=(0, 1, 2)):
    directory = Path(run["output"])
    manifest = load(directory / "manifest.json")
    simulation = ExternalSimulation(task, communication)
    simulation.initialize_physics(manifest["initial_physical_states"], 0., run["seed"])
    inputs, targets = [], []
    limits = task["robot_defaults"]
    cutoff = round(run["formation_s"] / simulation.dt)
    with (directory / "control.jsonl").open() as stream:
        for tick, line in enumerate(stream, 1):
            if tick > cutoff:
                break
            row = json.loads(line)
            observations = simulation.observe_all()
            inputs.extend(observations[i] for i in robot_ids)
            for i in robot_ids:
                speed, yaw = row["applied_commands"][i]
                scale = limits["max_forward_speed_mps"] if speed >= 0 else limits["max_reverse_speed_mps"]
                targets.append([speed / scale, yaw / limits["max_yaw_rate_radps"]])
            simulation.offer(row["states"], row["elapsed_s"])
            simulation.advance(np.zeros((3, 2)))
    if len(targets) != cutoff * len(robot_ids):
        raise ValueError(f"Incomplete formation trace: {directory}")
    return torch.from_numpy(pack(inputs)), torch.tensor(targets, dtype=torch.float32).clamp(-1., 1.)



def local_alignment_action(observation, limits):
    """Label alignment using only the actor's own state and received peers."""
    local = observation["local_features"]
    error = math.atan2(local[10], local[9])
    deadband, radius = local[12], local[13]
    if abs(error) < math.radians(2):
        return [0., 0.]
    yaw = min(.30, max(float(deadband), 2.0 * abs(error)))
    if deadband > 0:
        yaw = max(yaw, float(deadband) + .01)
    yaw = math.copysign(yaw, error)
    speed = 0.
    if radius > 0:
        peers = observation["neighbor_features"][observation["neighbor_mask"].astype(bool)]
        desired = local[8]
        offsets = peers[:, :2]
        # Compare short forward and reverse arcs using the received positions.
        distance = min(.12, max(.04, abs(error) * ROSBOT_TURN_RADIUS))
        heading_change = math.copysign(distance / ROSBOT_TURN_RADIUS, error)
        displacement = distance * np.array([math.cos(heading_change / 2), math.sin(heading_change / 2)])

        def cost(sign):
            ranges = np.linalg.norm(offsets - sign * displacement, axis=1)
            return np.sum((ranges - desired) ** 2)

        speed = .12 if cost(1) <= cost(-1) else -.12
        yaw = math.copysign(min(abs(speed) / ROSBOT_TURN_RADIUS, abs(yaw)), error)
    scale = limits["max_forward_speed_mps"] if speed >= 0 else limits["max_reverse_speed_mps"]
    return [speed / scale, yaw / limits["max_yaw_rate_radps"]]


def local_examples(task, communication, seeds, *, wheeltec_recovery=False):
    simulation = Simulation(task, communication)
    inputs, targets = [], []
    reverse_examples = 0
    for seed in seeds:
        simulation.reset(seed)
        if wheeltec_recovery:
            states = [vars(state) for state in simulation.states]
            goals = CooperativeTranslationReference(states).translation_goals(0.)
            teacher = CooperativePoseReference(
                [(*goal, 0.) for goal in goals], task["operating_area"],
                wheeltec_reversal_deceleration_mps2=.30,
            )
        for _ in range(150 if wheeltec_recovery else 90):
            observations = simulation.observe_all()
            actions = [local_alignment_action(row, task["robot_defaults"]) for row in observations]
            if wheeltec_recovery:
                commands, status = teacher.command([vars(state) for state in simulation.states])
                if status == "unresolved":
                    break
                speed, yaw = commands[1]
                limits = task["robot_defaults"]
                speed_limit = limits["max_forward_speed_mps"] if speed >= 0 else limits["max_reverse_speed_mps"]
                actions[1] = [speed / speed_limit, yaw / limits["max_yaw_rate_radps"]]
                reverse_examples += int(speed < 0.)
            inputs.extend(observations)
            targets.extend(actions)
            simulation.advance(actions)
    return torch.from_numpy(pack(inputs)), torch.tensor(targets, dtype=torch.float32), reverse_examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--local", action="store_true", help="Generate nominal labels from local alignment feedback.")
    parser.add_argument("--wheeltec-recovery", action="store_true")
    parser.add_argument("--gazebo-anchor-runs", nargs="+", type=Path,
                        help="Successful nominal traces; use the last run for validation.")
    args = parser.parse_args()
    if args.gazebo_anchor_runs and (args.local or len(args.gazebo_anchor_runs) < 2):
        parser.error("--gazebo-anchor-runs requires at least two runs and excludes --local")
    if args.wheeltec_recovery and not args.local:
        parser.error("--wheeltec-recovery requires --local")
    torch.set_num_threads(1)
    torch.manual_seed(187205)
    task = load(Path("configs/tasks/mixed_pose_curriculum_near.json"))
    communication = load(Path("configs/communication/nominal.json"))
    reverse_examples = 0
    if args.wheeltec_recovery:
        task["initialization"]["near_formation"]["position_jitter_m"] = .06
    if args.gazebo_anchor_runs:
        runs = []
        for directory in args.gazebo_anchor_runs:
            manifest = load(directory / "manifest.json")
            summary = load(directory / "summary.json")
            if (manifest["settings"]["communication"] != "nominal"
                    or summary["reason"] != "cooperative_mission_success"
                    or not summary["formation_completed_s"]):
                parser.error(f"Expected a completed nominal formation: {directory}")
            runs.append(dict(output=str(directory), seed=manifest["settings"]["seed"],
                             formation_s=summary["formation_completed_s"]))
        if len({run["seed"] for run in runs}) != len(runs):
            parser.error("Demonstration seeds must be distinct.")
        train_seeds = [run["seed"] for run in runs[:-1]]
        validation_seeds = [runs[-1]["seed"]]
        batches = [examples(task, communication, run, (0, 2)) for run in runs[:-1]]
        train_x, train_y = (torch.cat(items) for items in zip(*batches))
        validation_x, validation_y = examples(task, communication, runs[-1], (0, 2))
    elif args.local:
        train_seeds = list(range(192001, 192025)) if args.wheeltec_recovery else list(range(112001, 112033))
        validation_seeds = list(range(192101, 192107)) if args.wheeltec_recovery else list(range(112101, 112109))
        train_x, train_y, reverse_examples = local_examples(
            task, communication, train_seeds, wheeltec_recovery=args.wheeltec_recovery)
        validation_x, validation_y, _ = local_examples(
            task, communication, validation_seeds, wheeltec_recovery=args.wheeltec_recovery)
    else:
        report = load(Path("experiments/runs/cooperative_smooth_validation_summary.json"))
        runs = {run["seed"]: run for run in report["runs"]}
        train_seeds, validation_seeds = [8601, 8602, 8605], [8608]
        batches = [examples(task, communication, runs[seed]) for seed in train_seeds]
        train_x, train_y = (torch.cat(items) for items in zip(*batches))
        validation_x, validation_y = examples(task, communication, runs[validation_seeds[0]])
    bundle = torch.load(args.source, map_location="cpu", weights_only=True)
    component = bundle["formation"]["fresh_actor"]
    if component["contract"] != "formation_pose_v1":
        raise ValueError("The source bundle must contain a shared formation pose actor.")
    config, policy, _ = make_policy(
        local_dim=component["local_dim"], neighbor_dim=component["neighbor_dim"],
        critic_dim=component["critic_dim"], geometry_summary=component["geometry_summary"],
    )
    policy.actor.load_state_dict(component["actor"])
    optimizer = torch.optim.Adam(policy.actor.parameters(), lr=1e-4)

    def loss(inputs, targets):
        hidden, masks = hidden_inputs(len(inputs), config)
        actions, _, _ = policy.actor(inputs, hidden, masks, deterministic=True)
        return torch.nn.functional.mse_loss(actions, targets)

    with torch.no_grad():
        initial_error = loss(validation_x, validation_y).item()
    best = initial_error
    epoch_selected = 0
    weights = deepcopy(policy.actor.state_dict())
    for epoch in range(1, 61):
        for indices in torch.randperm(len(train_x)).split(256):
            optimizer.zero_grad()
            loss(train_x[indices], train_y[indices]).backward()
            torch.nn.utils.clip_grad_norm_(policy.actor.parameters(), .5)
            optimizer.step()
        with torch.no_grad():
            error = loss(validation_x, validation_y).item()
        if error < best:
            best, epoch_selected = error, epoch
            weights = deepcopy(policy.actor.state_dict())

    args.output.mkdir(parents=True)
    component["actor"] = weights
    bundle["status"] = "mixed_pose_demonstration_warm_start_pending_evaluation"
    torch.save(bundle, args.output / "candidate_bundle.pt")
    result = {
        "train_seeds": train_seeds, "validation_seeds": validation_seeds, "local_labels": args.local,
        "wheeltec_reverse_examples": reverse_examples,
        "gazebo_anchor_runs": [str(run) for run in args.gazebo_anchor_runs or ()],
        "train_samples": len(train_x), "validation_samples": len(validation_x),
        "initial_validation_mse": initial_error, "best_validation_mse": best,
        "selected_epoch": epoch_selected,
    }
    (args.output / "training_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
