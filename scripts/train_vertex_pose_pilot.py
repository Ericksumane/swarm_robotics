"""Train and evaluate an experimental local target-pose formation actor."""

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import torch

from swarm_core.external_simulation import ExternalSimulation
from swarm_core.geometry import formation_metrics
from swarm_core.learning.mappo_adapter import hidden_inputs, make_policy, pack
from swarm_core.mission_env import MissionEnv, MOVE


def read(path):
    return json.loads(Path(path).read_text())


def target_inputs(simulation, goals):
    observations = simulation.observe_all()
    for observation, state, goal in zip(observations, simulation.estimates, goals):
        c, s = math.cos(state.heading), math.sin(state.heading)
        dx, dy = goal[0] - state.x, goal[1] - state.y
        error = goal[2] - state.heading
        pose = (c * dx + s * dy, -s * dx + c * dy, math.cos(error), math.sin(error))
        observation["local_features"] = np.concatenate((observation["local_features"], pose))
    return pack(observations)


def normalize(commands, limits):
    forward = limits["max_forward_speed_mps"]
    reverse = limits["max_reverse_speed_mps"]
    yaw = limits["max_yaw_rate_radps"]
    return np.asarray([[v / (forward if v >= 0 else reverse), w / yaw]
                       for v, w in commands], dtype=np.float32)


def examples(task, communication, record):
    """Reconstruct local messages and use fixed goals from a planned reference."""
    simulation = ExternalSimulation(task, communication)
    rows = record["samples"]
    goals = [robot["target_world"] for robot in rows[0]["robots"]]
    limits = task["robot_defaults"]
    inputs, labels = [], []
    speed = np.zeros((3, 2))

    def states(row, measured):
        return [dict(x=r["pose"][0], y=r["pose"][1], heading=r["pose"][2],
                     speed=v, yaw_rate=w)
                for r, (v, w) in zip(row["robots"], measured)]

    simulation.initialize_physics(states(rows[0], speed), 0., record["seed"])
    for tick, row in enumerate(rows):
        if tick:
            simulation.offer(states(row, speed), tick * simulation.dt)
            simulation.advance(np.zeros((3, 2)))
        inputs.append(target_inputs(simulation, goals))
        commands = [robot["command"] for robot in row["robots"]]
        labels.append(normalize(commands, limits))
        speed = np.asarray(commands)
    return np.concatenate(inputs), np.concatenate(labels)


def evaluate(task, communication, case, actor, config, *, intervention=None, max_steps=1200):
    env = MissionEnv(task, communication, {})
    env.reset(case["seed"])
    plan = case["planned"]
    goals = [(*plan["staged_anchors"][0], 0.), (*plan["goal"], 0.),
             (*plan["staged_anchors"][1], 0.)]
    limits = task["robot_defaults"]
    h, mask = hidden_inputs(3, config)
    minimum = float("inf")
    held = 0
    formed_at = None
    start_times = [None] * 3
    radius = limits['footprint_radius_m']
    x_bounds = task['operating_area']['x_bounds_m']
    y_bounds = task['operating_area']['y_bounds_m']
    reason = 'time_limit'
    for _ in range(max_steps):
        inputs = target_inputs(env.simulation, goals)
        with torch.no_grad():
            actions, _, _ = actor(inputs, h, mask, deterministic=True)
        actions = actions.numpy()
        if intervention is not None:
            actions = intervention(env.simulation.states, goals, inputs, actions)
        env.simulation.advance(actions)
        env._update_phases()
        states = env.simulation.states
        geometry = formation_metrics(states, 1.2, limits["footprint_radius_m"])
        minimum = min(minimum, geometry["minimum_clearance_m"])
        heading = max(abs(math.degrees(s.heading)) for s in states)
        settled = (geometry["max_pair_error_m"] <= .10 and heading <= 5.
                   and all(abs(s.speed) <= .03 and abs(s.yaw_rate) <= .05 for s in states))
        held = held + 1 if settled else 0
        if held >= 20 and formed_at is None:
            formed_at = env.simulation.time_s
        for index, phase in enumerate(env.phases):
            if phase == MOVE and start_times[index] is None:
                start_times[index] = env.simulation.time_s
        if geometry["collision"]:
            reason = "collision"
            break
        if any(not (x_bounds[0] + radius <= state.x <= x_bounds[1] - radius
                    and y_bounds[0] + radius <= state.y <= y_bounds[1] - radius)
               for state in states):
            reason = "outside_area"
            break
        if all(t is not None for t in start_times):
            reason = "all_move"
            break
    return dict(seed=case["seed"], reason=reason, time_s=env.simulation.time_s, formed_at_s=formed_at,
                start_times_s=start_times, collision=minimum <= 0,
                minimum_clearance_m=minimum, final_pair_error_m=geometry["max_pair_error_m"],
                max_heading_deg=heading)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(114201)
    torch.set_num_threads(1)
    root = Path("experiments/runs")
    repair = root / "mixed_mission_repair_001"
    task = read(repair / "formation_training_task.json")
    mission = read("configs/tasks/history_full_mission.json")
    for key in ("history_startup", "formation_peer_history"):
        task[key] = mission[key]
    communication = read("configs/communication/nominal.json")
    demos = [json.loads(line) for line in
             (root / "cooperative_reference_001/demonstrations.jsonl").read_text().splitlines()]
    train = [r for r in demos if r["seed"] not in (8607, 8608)]
    validation = [r for r in demos if r["seed"] in (8607, 8608)]
    source = torch.load(repair / "pose_braking_candidate.pt", weights_only=True)
    component = source["formation"]["fresh_actor"]
    assert component["local_dim"] == 15
    config, model, _ = make_policy(local_dim=19, neighbor_dim=9,
                                   critic_dim=component["critic_dim"], geometry_summary=True)
    weights = deepcopy(component["actor"])
    first = weights["base.head.0.weight"]
    weights["base.head.0.weight"] = torch.cat(
        (first[:, :15], first.new_zeros((first.shape[0], 4)), first[:, 15:]), dim=1)
    model.actor.load_state_dict(weights)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=.001)

    def dataset(records):
        batches = [examples(task, communication, record) for record in records]
        return tuple(torch.from_numpy(np.concatenate(values)) for values in zip(*batches))

    x, y = dataset(train)
    vx, vy = dataset(validation)

    def loss(inputs, labels):
        h, mask = hidden_inputs(len(inputs), config)
        actions, _, _ = model.actor(inputs, h, mask, deterministic=True)
        return torch.nn.functional.mse_loss(actions, labels)

    with torch.no_grad():
        initial = loss(vx, vy).item()
    best = initial
    selected = 0
    for epoch in range(1, 81):
        for ids in torch.randperm(len(x)).split(256):
            optimizer.zero_grad()
            loss(x[ids], y[ids]).backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), .5)
            optimizer.step()
        with torch.no_grad():
            error = loss(vx, vy).item()
        if error < best:
            best, selected = error, epoch
            weights = deepcopy(model.actor.state_dict())

    args.output.mkdir(parents=True)
    model.actor.load_state_dict(weights)
    model.actor.eval()
    torch.save(dict(contract="formation_vertex_pose_v1", local_dim=19, neighbor_dim=9,
                    geometry_summary=True, critic_dim=component["critic_dim"],
                    actor=weights, task=task), args.output / "experimental_actor.pt")
    report = dict(train_seeds=[r["seed"] for r in train], validation_seeds=[r["seed"] for r in validation],
                  training_samples=len(x), validation_samples=len(vx), initial_validation_mse=initial,
                  best_validation_mse=best, selected_epoch=selected)
    print(json.dumps(report), flush=True)
    cases = [json.loads(line) for line in
             (repair / "wheeltec_reference_evaluation.jsonl").read_text().splitlines()]
    report["closed_loop"] = [evaluate(task, communication, case, model.actor, config) for case in cases]
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["closed_loop"]), flush=True)


if __name__ == "__main__":
    main()
