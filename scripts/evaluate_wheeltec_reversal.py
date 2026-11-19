"""Evaluate WheelTec reversal recovery with fixed formation anchors."""

import argparse
import json
import math
from pathlib import Path

from cooperative_pose_reference import CooperativePoseReference
from swarm_core.dynamics import RobotState, step
from swarm_core.mixed_commands import update_rosbot_curvature


def run(case, task, perturb, strict, *, actor=None, samples=None, vertex_features=False):
    plan = case["planned"]
    goals = [(*plan["staged_anchors"][0], 0.), (*plan["goal"], 0.),
             (*plan["staged_anchors"][1], 0.)]
    limits = task["robot_defaults"]
    area = task["operating_area"]
    dt = task["simulation"]["time_step_s"]
    radius = limits["footprint_radius_m"]
    stationary = [vars(RobotState(*goals[0])), vars(RobotState(*goals[2]))]
    settings = dict(wheeltec_heading_tolerance_rad=math.radians(2.5),
                    wheeltec_reversal_deceleration_mps2=.30) if strict else {}
    ctl = CooperativePoseReference(goals, area, precision=True, **settings)
    ctl.wheeltec_actor = actor
    ctl.wheeltec_actor_uses_vertex = vertex_features
    ctl.path = list(plan["primitives"])
    ctl.path_uses_measured_geometry = True
    ctl.segment_start = tuple(plan["start"])
    state = RobotState(*plan["start"])
    curvature = reverse = 0.
    minimum = min(math.dist(plan["start"][:2], g[:2]) - 2 * radius
                  for g in (goals[0], goals[2]))
    reason = "time_limit"
    for tick in range(1200):
        if perturb and tick == 25:
            state = RobotState(state.x - .04 * math.sin(state.heading),
                               state.y + .04 * math.cos(state.heading),
                               state.heading + math.radians(4), state.speed, state.yaw_rate)
        commands, status = ctl.command([stationary[0], vars(state), stationary[1]])
        if status in ("settled", "unresolved"):
            reason = status
            break
        v, w = commands[1]
        if samples is not None and ctl.wheeltec_features is not None:
            samples.append({"seed": case["seed"], "disturbance": perturb,
                            "time_s": tick * dt, "state": vars(state),
                            "features": ctl.wheeltec_features, "command": [v, w]})
        curvature = update_rosbot_curvature(curvature, v, w, dt,
                                            task["rosbot_steering_time_constant_s"])
        scale = limits["max_forward_speed_mps" if v >= 0 else "max_reverse_speed_mps"]
        state = step(state, [v / scale, w / limits["max_yaw_rate_radps"]],
                     limits, dt, steering_curvature=curvature)
        reverse += max(0., -state.speed) * dt
        minimum = min(minimum, *(math.dist((state.x, state.y), g[:2]) - 2 * radius
                                 for g in (goals[0], goals[2])))
        if minimum <= 0:
            reason = "collision"
            break
        if not (area["x_bounds_m"][0] + radius <= state.x <= area["x_bounds_m"][1] - radius
                and area["y_bounds_m"][0] + radius <= state.y <= area["y_bounds_m"][1] - radius):
            reason = "outside_area"
            break
    error = max(abs(math.dist((state.x, state.y), g[:2]) -
                    task["team"]["desired_pair_distance_m"]) for g in (goals[0], goals[2]))
    heading = abs(math.remainder(state.heading, math.tau))
    return dict(seed=case["seed"], disturbance=perturb, strict=strict, reason=reason,
                time_s=tick * dt, reverse_m=reverse, minimum_clearance_m=minimum,
                max_pair_error_m=error, heading_error_deg=math.degrees(heading),
                reversal_stops=ctl.reversal_stops, rejected_arcs=ctl.arc_rejections,
                replans=ctl.replans,
                startup_geometry_and_heading_ready=reason=="settled" and error<=.10 and heading<=.05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = Path("experiments/runs/mixed_mission_repair_001")
    task = json.loads((root / "formation_training_task.json").read_text())
    cases = [json.loads(line) for line in (root / "wheeltec_reference_evaluation.jsonl").read_text().splitlines()]
    with args.output.open("x") as stream:
        for case in cases:
            for perturb in (False, True):
                result = run(case, task, perturb, True)
                stream.write(json.dumps(result) + "\n")
                stream.flush()
                print(result, flush=True)


if __name__ == "__main__":
    main()
