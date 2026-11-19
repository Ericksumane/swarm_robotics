"""Train and evaluate Go2 and Jackal pose actions with stationary peers."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from cooperative_pose_reference import CooperativePoseReference, stopped
from swarm_core.dynamics import RobotState, step
from swarm_core.mixed_commands import GO2_YAW_DEADBAND


DT = .1
GOAL = (math.sqrt(1.08), 0., 0.)
ANCHORS = ((0., -.6), (0., .6))
ROBOTS = {"go2": 0, "jackal": 2}


def pose_inputs(state, aligning):
    dx, dy = GOAL[0] - state.x, GOAL[1] - state.y
    c, s = math.cos(state.heading), math.sin(state.heading)
    heading_error = math.remainder(GOAL[2] - state.heading, math.tau)
    return [c * dx + s * dy, -s * dx + c * dy,
            math.sin(heading_error), math.cos(heading_error), float(aligning)]


def reference_controller(area):
    return CooperativePoseReference(
        [(*ANCHORS[0], 0.), GOAL, (*ANCHORS[1], 0.)], area)


def fit(index, area, seed):
    rng = np.random.default_rng(seed)
    teacher = reference_controller(area)
    inputs, targets = [], []
    for n in range(8192):
        radius = rng.uniform(0., .10 if n % 2 else .65)
        angle = rng.uniform(-math.pi, math.pi)
        state = RobotState(GOAL[0] + radius * math.cos(angle),
                           GOAL[1] + radius * math.sin(angle),
                           rng.uniform(-math.pi, math.pi))
        teacher.aligning[index] = bool(n % 2)
        action = teacher.point_command(index, vars(state), GOAL)
        inputs.append(pose_inputs(state, teacher.aligning[index]))
        targets.append([abs(value) for value in action])
    x = torch.tensor(inputs, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32)
    model = torch.nn.Sequential(torch.nn.Linear(5, 32), torch.nn.Tanh(),
                                torch.nn.Linear(32, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    for _ in range(70):
        for ids in torch.randperm(len(x)).split(256):
            loss = (model(x[ids]) - y[ids]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def learned_command(model, index, state, aligning):
    x, y, sin_heading, cos_heading, _ = features = pose_inputs(state, aligning)
    if aligning:
        error = math.atan2(sin_heading, cos_heading)
        direction = 0.
    else:
        error = math.atan2(y, x)
        direction = 1.
        if abs(error) > math.pi / 2:
            direction = -1.
            error = math.remainder(error + math.pi, math.tau)
    with torch.no_grad():
        speed, yaw = model(torch.tensor(features, dtype=torch.float32)).tolist()
    if not direction or abs(error) >= .25:
        speed = 0.
    else:
        speed = direction * max(.035, min(speed, .25 if direction > 0 else .15))
    yaw = (0. if abs(error) <= .035 else
           math.copysign(max(.09 if index == 0 else .04,
                             min(yaw, .35 if index == 0 else .50)), error))
    return speed, yaw


def evaluate(index, task, seed, model=None):
    rng = np.random.default_rng(seed)
    state = RobotState(GOAL[0] + rng.uniform(-.35, .35),
                       rng.uniform(-.22, .22), rng.uniform(-math.pi, math.pi))
    limits = task["robot_defaults"]
    teacher = reference_controller(task["operating_area"])
    aligning = False
    hold = 0
    minimum = min(math.dist((state.x, state.y), a) for a in ANCHORS)
    reason = "time_limit"
    for tick in range(600):
        distance = math.dist((state.x, state.y), GOAL[:2])
        if model is None:
            v, w = teacher.point_command(index, vars(state), GOAL)
        else:
            aligning = distance <= ((.065 if index == 0 else .05) if aligning else .04)
            v, w = learned_command(model, index, state, aligning)
        if index == 0 and abs(v) < .015 and abs(w) < GO2_YAW_DEADBAND:
            v = w = 0.
        scale = limits["max_forward_speed_mps" if v >= 0 else "max_reverse_speed_mps"]
        state = step(state, [v / scale, w / limits["max_yaw_rate_radps"]], limits, DT)
        ranges = [math.dist((state.x, state.y), anchor) for anchor in ANCHORS]
        minimum = min(minimum, *ranges)
        pair_error = max(abs(value - 1.2) for value in ranges)
        heading = abs(math.degrees(math.remainder(state.heading, math.tau)))
        if minimum < .65:
            reason = "clearance_limit"
            break
        ready = (math.dist((state.x, state.y), GOAL[:2]) <= .05
                 and pair_error <= .05 and heading <= 5. and stopped(vars(state)))
        hold = hold + 1 if ready else 0
        if hold >= 20:
            reason = "success"
            break
    return dict(seed=seed, reason=reason, time_s=(tick + 1) * DT,
                pair_error_m=pair_error, heading_error_deg=heading,
                minimum_center_distance_m=minimum)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    task = json.loads(Path("configs/tasks/mixed_pose_curriculum_near.json").read_text())
    torch.set_num_threads(1)
    torch.manual_seed(195400)
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    for name, index in ROBOTS.items():
        model = fit(index, task["operating_area"], 195400 + index)
        actor = {"contract": "anchor_pose_v1", "robot": name,
                 "weights": {key: value.tolist() for key, value in model.state_dict().items()}}
        (args.output / f"{name}_actor.json").write_text(json.dumps(actor) + "\n")
        results[name] = {
            label: [evaluate(index, task, seed, policy) for seed in range(195501, 195513)]
            for label, policy in (("reference", None), ("learned", model))
        }
        print(name, {label: sum(row["reason"] == "success" for row in rows)
                     for label, rows in results[name].items()}, flush=True)
    (args.output / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
