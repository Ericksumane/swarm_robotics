"""Evaluate deterministic mixed-platform formation on paired initial states."""

import argparse
from collections import Counter
from contextlib import nullcontext
import json
import math
from dataclasses import replace
from pathlib import Path
import subprocess
import sys

from swarm_core.dynamics import RobotState, step
from swarm_core.initialization import sample_initial_states
from swarm_core.mixed_commands import constrain_mixed_commands, update_rosbot_curvature


def evaluate(bundle, task, seed, condition="both", gazebo_run=None, pose_goal=False, steering_time_constant=None, full_mission=False, communication="nominal", start_aligned=False, trace=None, jackal_heading_probe=False):
    if gazebo_run is None:
        states = sample_initial_states(task, seed)
    else:
        manifest = json.loads((gazebo_run / "manifest.json").read_text())
        states = [
            RobotState(*(state[key] for key in ("x", "y", "heading", "speed", "yaw_rate")))
            for state in manifest["initial_physical_states"]
        ]
    dt = task["simulation"]["time_step_s"]
    limits = task["robot_defaults"]
    forward = limits["max_forward_speed_mps"]
    command = [
        sys.executable, "scripts/live_policy_worker.py",
        "--formation-speed", str(forward), "--communication", communication,
        "--seed", str(seed), "--time-limit", "60", "--bundle", str(bundle),
    ]
    if start_aligned:
        command.append("--start-aligned")
    if not full_mission:
        command.append("--formation-only")
    if pose_goal:
        command.append("--pose-goal")
    with subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
    ) as worker:
        def request(operation, clock):
            worker.stdin.write(json.dumps({
                "op": operation, "clock_s": clock,
                "states": [vars(state) for state in states],
            }) + "\n")
            worker.stdin.flush()
            return json.loads(worker.stdout.readline())

        ready = json.loads(worker.stdout.readline())
        limits = ready["task"]["robot_defaults"]
        if start_aligned:
            side = ready["task"]["team"]["desired_pair_distance_m"]
            height = side * math.sqrt(3.) / 2.
            states = [replace(state, x=x, y=y, heading=0., speed=0., yaw_rate=0.)
                      for state, (x, y) in zip(states, ((2*height/3, 0.), (-height/3, side/2),
                                                        (-height/3, -side/2)))]
        forward = limits["max_forward_speed_mps"]
        reverse = limits["max_reverse_speed_mps"]
        yaw = limits["max_yaw_rate_radps"]
        result = request("initialize", 0.)
        clearance = result["minimum_clearance_m"]
        rosbot_curvature = 0.
        starts = states.copy()
        peak_speeds = [0.] * len(states)
        mode_counts = [Counter() for _ in states]
        for tick in range(1, round(60 / dt) + 1):
            for counts, mode in zip(mode_counts, result["modes"]):
                counts[mode] += 1
            commands = constrain_mixed_commands(
                result["commands"],
                go2_deadband=condition in ("go2", "both"),
                rosbot_curvature=condition in ("rosbot", "both"),
            )
            if jackal_heading_probe:
                # Diagnostic intervention in the nominal +X formation task.
                error = math.remainder(-states[2].heading, 2 * math.pi)
                commands[2] = [commands[2][0], max(-yaw, min(yaw, 1.5 * error))]
            actions = [
                [v / (forward if v >= 0 else reverse), w / yaw]
                for v, w in commands
            ]
            if steering_time_constant is not None:
                rosbot_curvature = update_rosbot_curvature(
                    rosbot_curvature, *commands[1], dt, steering_time_constant
                )
            previous = states
            states = [
                step(state, action, limits, dt,
                     steering_curvature=rosbot_curvature if steering_time_constant is not None and i == 1 else None)
                for i, (state, action) in enumerate(zip(states, actions))
            ]
            if trace is not None:
                for robot_id, (before, after) in enumerate(zip(previous, states)):
                    trace.write(json.dumps({
                        "bundle": str(bundle), "seed": seed, "profile": communication,
                        "condition": condition, "robot_id": robot_id,
                        "jackal_heading_probe": jackal_heading_probe,
                        "time_s": (tick - 1) * dt, "mode": result["modes"][robot_id],
                        "state_before": vars(before), "state_after": vars(after),
                        "normalized_action": result["actions"][robot_id],
                        "worker_command": result["commands"][robot_id],
                        "applied_command": commands[robot_id],
                        "pair_error_m": result["max_pair_error_m"],
                    }) + "\n")
            peak_speeds = [max(peak, state.speed) for peak, state in zip(peak_speeds, states)]
            result = request("step", tick * dt)
            clearance = min(clearance, result["minimum_clearance_m"])
            if result["done"]:
                break
        worker.stdin.close()
    return {
        "bundle": str(bundle), "seed": seed, "condition": condition, "reason": result["reason"],
        "jackal_heading_probe": jackal_heading_probe,
        "time_s": result["time_s"], "final_error_m": result["max_pair_error_m"],
        "minimum_circular_clearance_m": clearance,
        "peak_speeds_mps": peak_speeds,
        "final_speeds_mps": [state.speed for state in states],
        "final_yaw_rates_radps": [state.yaw_rate for state in states],
        "mode_counts": [dict(counts) for counts in mode_counts],
        "forward_displacement_m": [state.x - start.x for state, start in zip(states, starts)],
        "communication": communication, "phases": result["phases"],
        "goal_errors_m": result["goal_errors_m"],
        "heading_errors_deg": [math.degrees(x) for x in result["startup_heading_errors_rad"]],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundles", type=Path, nargs="+")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--gazebo-prefix", type=Path)
    parser.add_argument("--hard-starts", action="store_true")
    parser.add_argument("--task", type=Path, default=Path("configs/tasks/mixed_formation_nominal080.json"))
    parser.add_argument("--pose-goal", action="store_true")
    parser.add_argument("--steering-time-constant", type=float)
    parser.add_argument("--full-mission", action="store_true")
    parser.add_argument("--profiles", nargs="+", choices=("nominal", "good", "moderate", "severe"),
                        default=["nominal"])
    parser.add_argument("--start-aligned", action="store_true")
    parser.add_argument("--trace", type=Path, help="Write per-step commands and states as JSON Lines.")
    parser.add_argument("--jackal-heading-probe", action="store_true",
                        help="Replace only Jackal yaw with heading feedback for diagnosis.")
    args = parser.parse_args()
    if args.start_aligned and not args.full_mission:
        parser.error("--start-aligned requires --full-mission")
    task = json.loads(args.task.read_text())
    if args.jackal_heading_probe and (
        args.full_mission or args.profiles != ["nominal"] or not args.pose_goal
        or task["mission"]["direction_world"] != [1.0, 0.0]
    ):
        parser.error("--jackal-heading-probe requires nominal +X pose-goal formation.")
    if args.hard_starts:
        task["initialization"]["hard_dispersed"] = {
            "probability": 1., "min_center_distance_m": 1.65,
        }
    with (args.trace.open("x") if args.trace else nullcontext()) as trace:
        for seed in args.seeds:
            for bundle in args.bundles:
                gazebo_run = Path(f"{args.gazebo_prefix}{seed}") if args.gazebo_prefix else None
                conditions = ("none", "go2", "rosbot", "both") if args.ablation else ("both",)
                for profile in args.profiles:
                    for condition in conditions:
                        result = evaluate(bundle, task, seed, condition, gazebo_run, args.pose_goal,
                                          args.steering_time_constant, args.full_mission, profile, args.start_aligned, trace,
                                          args.jackal_heading_probe)
                        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
