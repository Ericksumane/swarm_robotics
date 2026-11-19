"""Compare local WheelTec waypoint feedback with a learned copy in planar physics."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from check_rosbot_formation_path import advance
from swarm_core.dynamics import RobotState, step
from swarm_core.learning.mappo_adapter import hidden_inputs, make_policy
from swarm_core.mixed_commands import ROSBOT_TURN_RADIUS, update_rosbot_curvature


def path_segments(plan):
    pose = tuple(plan["start"])
    segments = []
    for distance, curvature in plan["primitives"]:
        direction = math.copysign(1., distance)
        if not segments or segments[-1][0] != direction:
            segments.append((direction, [pose]))
        for index in range(1, 10):
            segments[-1][1].append(advance(pose, distance * index / 9., curvature))
        pose = segments[-1][1][-1]
    return [(direction, np.asarray(points)) for direction, points in segments]


def local_goal(state, waypoint, direction, remaining):
    dx, dy = waypoint[:2] - (state.x, state.y)
    c, s = math.cos(state.heading), math.sin(state.heading)
    error = waypoint[2] - state.heading
    return np.asarray([c * dx + s * dy, -s * dx + c * dy,
                       math.sin(error), math.cos(error), direction, remaining,
                       state.speed, state.yaw_rate], dtype=np.float32)


def feedback(features, limits):
    x, y, _, _, direction, remaining, _, _ = features
    curvature = np.clip(2. * y / max(x*x + y*y, .0025),
                        -1. / ROSBOT_TURN_RADIUS, 1. / ROSBOT_TURN_RADIUS)
    speed = direction * min(.13, max(.025, .65 * remaining))
    scale = limits["max_forward_speed_mps" if speed >= 0 else "max_reverse_speed_mps"]
    return np.asarray([speed / scale, speed * curvature / limits["max_yaw_rate_radps"]],
                      dtype=np.float32)


def rollout(plan, limits, area, steering_time, policy, disturbance):
    state = RobotState(*plan["start"])
    anchors = plan["staged_anchors"]
    segments = path_segments(plan)
    dt = .1
    steering = 0.
    segment_index = index = 0
    minimum = float("inf")
    reverse = 0.
    forward = 0.
    reason = "time_limit"
    for tick in range(1200):
        if disturbance and tick == 25:
            # Apply a fixed cross-track and heading perturbation once.
            state = RobotState(state.x - .04 * math.sin(state.heading),
                               state.y + .04 * math.cos(state.heading),
                               state.heading + math.radians(4), state.speed, state.yaw_rate)
        direction, points = segments[segment_index]
        current = np.asarray([state.x, state.y])
        remaining = float(np.linalg.norm(points[-1, :2] - current))
        at_end = remaining < .035
        if at_end and abs(state.speed) <= .03 and abs(state.yaw_rate) <= .05:
            if segment_index == len(segments) - 1:
                reason = "path_complete"
                break
            segment_index += 1
            index = 0
            continue
        if at_end:
            action = np.zeros(2)
        else:
            candidates = points[index:index + 12, :2]
            index += int(np.argmin(np.linalg.norm(candidates - current, axis=1)))
            waypoint = points[min(index + 10, len(points) - 1)]
            action = policy(local_goal(state, waypoint, direction, remaining), limits)
        scale = limits["max_forward_speed_mps" if action[0] >= 0 else "max_reverse_speed_mps"]
        v = float(action[0]) * scale
        w = float(np.clip(action[1] * limits["max_yaw_rate_radps"],
                          -abs(v) / ROSBOT_TURN_RADIUS, abs(v) / ROSBOT_TURN_RADIUS))
        steering = update_rosbot_curvature(steering, v, w, dt, steering_time)
        state = step(state, [action[0], w / limits["max_yaw_rate_radps"]],
                     limits, dt, steering_curvature=steering)
        reverse += max(0., -state.speed) * dt
        forward += max(0., state.speed) * dt
        radius = limits["footprint_radius_m"]
        minimum = min(minimum, *(math.dist((state.x, state.y), p) - 2 * radius for p in anchors))
        if minimum <= 0:
            reason = "collision"
            break
        if not (area["x_bounds_m"][0] + radius <= state.x <= area["x_bounds_m"][1] - radius
                and area["y_bounds_m"][0] + radius <= state.y <= area["y_bounds_m"][1] - radius):
            reason = "outside_area"
            break
    position = (state.x, state.y)
    pair_error = max(abs(math.dist(position, anchor) - 1.2) for anchor in anchors)
    heading = abs(math.degrees(math.remainder(state.heading, math.tau)))
    return dict(reason=reason, time_s=tick*dt, final_goal_error_m=math.dist(position, plan["goal"]),
                max_pair_error_m=pair_error, heading_error_deg=heading,
                min_clearance_m=minimum, reverse_m=reverse, forward_m=forward,
                success=reason=="path_complete" and pair_error<=.10 and heading<=2.9)


def train(limits):
    torch.set_num_threads(1)
    torch.manual_seed(114401)
    rng = np.random.default_rng(114401)
    count = 8192
    direction = rng.choice([-1., 1.], count)
    angles = rng.uniform(-1.3, 1.3, count)
    distance = rng.uniform(.03, .33, count)
    x = direction * distance * np.cos(angles)
    y = distance * np.sin(angles)
    heading = rng.uniform(-1., 1., count)
    inputs = np.column_stack((x, y, np.sin(heading), np.cos(heading), direction,
                              rng.uniform(.03, 3., count),
                              rng.uniform(-.15, .15, count),
                              rng.uniform(-.16, .16, count))).astype(np.float32)
    # Endpoint distance cannot be less than lookahead when both point to the end.
    inputs[:, 5] = np.maximum(inputs[:, 5], distance)
    labels = np.asarray([feedback(row, limits) for row in inputs])
    config, model, _ = make_policy(num_agents=1, local_dim=8)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=.001)
    for _ in range(65):
        for ids in torch.randperm(count).split(256):
            h, masks = hidden_inputs(len(ids), config)
            prediction, _, _ = model.actor(inputs[ids], h, masks, deterministic=True)
            loss = torch.nn.functional.mse_loss(prediction, torch.from_numpy(labels[ids.numpy()]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    h, masks = hidden_inputs(1, config)

    def learned(features, limits):
        with torch.no_grad():
            action, _, _ = model.actor(features[None, :], h, masks, deterministic=True)
        return np.clip(action.numpy()[0], -1., 1.)

    return model.actor, learned


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    repair = Path("experiments/runs/mixed_mission_repair_001")
    task = json.loads((repair / "formation_training_task.json").read_text())
    limits = task["robot_defaults"]
    actor, learned = train(limits)
    results = []
    for line in (repair / "wheeltec_reference_evaluation.jsonl").read_text().splitlines():
        case = json.loads(line)
        for name, policy in (("feedback", feedback), ("learned", learned)):
            for disturbance in (False, True):
                result = rollout(case["planned"], limits, task["operating_area"],
                                 task["rosbot_steering_time_constant_s"], policy, disturbance)
                results.append(dict(seed=case["seed"], controller=name,
                                    disturbance=disturbance, **result))
                print(results[-1], flush=True)
    args.output.mkdir(parents=True)
    torch.save({"contract": "wheeltec_arc_waypoint_v1", "actor": actor.state_dict()},
               args.output / "experimental_actor.pt")
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
