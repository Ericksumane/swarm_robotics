"""Evaluate nominal terminal pose recovery from a recorded arrival and perturbations."""
import json
import math
from pathlib import Path

from cooperative_pose_reference import CooperativePoseReference
from swarm_core.dynamics import RobotState, step
from swarm_core.mixed_commands import constrain_mixed_commands, update_rosbot_curvature


def main():
    root = Path("experiments/runs/nominal_stop_24k_001")
    record = json.loads((root / "hold_recovery_geometry.json").read_text())
    task = json.loads(Path("experiments/runs/wheeltec_arc_executor_001/terminal_refinement_task.json").read_text())
    limits = task["robot_defaults"]
    original = [RobotState(**row) for row in record["final_states"]]
    destinations = [(5 + 1.2 / math.sqrt(3), 0.), (5 - 1.2 * math.sqrt(3) / 6, .6),
                    (5 - 1.2 * math.sqrt(3) / 6, -.6)]
    results, fixtures = [], []
    for seed, sign, offset in ((114751, 1, 0.), (114752, -1, 0.),
                               (114753, 1, .02), (114754, -1, -.02)):
        states = [RobotState(s.x, s.y + (offset if i == 1 else 0.), sign * s.heading)
                  for i, s in enumerate(original)]
        goals = [(s.x, s.y, 0.) for s in states]
        fixtures.append(dict(seed=seed, destinations=[(*g, 0.) for g in destinations], samples=[dict(robots=[
            dict(pose=[s.x, s.y, s.heading], target_world=g) for s, g in zip(states, goals)])]))
        controller = CooperativePoseReference(
            goals, task["operating_area"], precision=True,
            wheeltec_heading_tolerance_rad=math.radians(3),
            wheeltec_reversal_deceleration_mps2=.30, wheeltec_path_pair_tolerance_m=.075)
        steering = 0.
        held = 0
        worst = 0.
        reason = "time_limit"
        for tick in range(600):
            commands, status = controller.command([vars(s) for s in states])
            commands = constrain_mixed_commands(commands)
            steering = update_rosbot_curvature(steering, *commands[1], .1, .64)
            states = [step(s, [v / limits["max_forward_speed_mps" if v >= 0 else "max_reverse_speed_mps"],
                              w / limits["max_yaw_rate_radps"]], limits, .1,
                           steering_curvature=steering if i == 1 else None)
                      for i, (s, (v, w)) in enumerate(zip(states, commands))]
            pairs = [math.hypot(a.x-b.x, a.y-b.y) for i, a in enumerate(states) for b in states[i+1:]]
            pair_error = max(abs(distance - 1.2) for distance in pairs)
            worst = max(worst, pair_error)
            heading = max(abs(math.degrees(s.heading)) for s in states)
            position = max(math.dist((s.x, s.y), g) for s, g in zip(states, destinations))
            settled = pair_error <= .1 and heading <= 5 and position <= .2 and all(
                abs(s.speed) <= .03 and abs(s.yaw_rate) <= .05 for s in states)
            held = held + 1 if settled else 0
            if min(pairs) <= 2 * limits["footprint_radius_m"]:
                reason = "collision"
                break
            if held >= 20:
                reason = "success"
                break
            if status == "unresolved":
                reason = status
                break
        row = dict(seed=seed, reason=reason, time_s=(tick+1)*.1, worst_pair_error_m=worst,
                   final_pair_error_m=pair_error, max_heading_deg=heading, max_destination_error_m=position)
        results.append(row)
        print(row, flush=True)
    (root / "hold_recovery_spacing_verified_cases.jsonl").write_text("".join(json.dumps(x)+"\n" for x in fixtures))
    (root / "hold_recovery_spacing_verified.json").write_text(json.dumps(results, indent=2)+"\n")


if __name__ == "__main__":
    main()
