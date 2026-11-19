"""Generate paired pose and action demonstrations for nominal mixed formation."""

import argparse
from itertools import combinations, product
import json
import math
from pathlib import Path

from check_rosbot_formation_path import advance, check
from swarm_core.initialization import sample_initial_states

DT = .1
SIDE = 1.2
RADIUS = .25


def append_segment(trace, pose, distance, curvature, speed_limit):
    ticks = max(1, math.ceil(abs(distance) / (speed_limit * DT) - 1e-9))
    speed = distance / (ticks * DT)
    for _ in range(ticks):
        trace.append((pose, (speed, speed * curvature)))
        pose = advance(pose, speed * DT, curvature)
    return pose


def differential_path(start, target):
    trace = []
    bearing = math.atan2(target[1] - start[1], target[0] - start[0])
    pose = start
    for turn, distance in ((math.remainder(bearing - start[2], math.tau), math.dist(start[:2], target)),
                           (math.remainder(-bearing, math.tau), 0.)):
        ticks = math.ceil(abs(turn) / (.35 * DT))
        if ticks:
            yaw = turn / (ticks * DT)
            for _ in range(ticks):
                trace.append((pose, (0., yaw)))
                pose = (*pose[:2], math.remainder(pose[2] + yaw * DT, math.tau))
        if distance:
            pose = append_segment(trace, pose, distance, 0., .25)
    return trace, pose


def sample_path(trace, start, end, delay, tick):
    index = tick - delay
    if index < 0:
        return start, (0., 0.)
    return trace[index] if index < len(trace) else (end, (0., 0.))


def plan_team(seed, task):
    states = sample_initial_states(task, seed)
    starts = [(s.x, s.y, s.heading) for s in states]
    reference = check(seed, starts, task)
    if not reference["found"]:
        return None
    goals = [reference["staged_anchors"][0], reference["goal"], reference["staged_anchors"][1]]
    first, first_end = differential_path(starts[0], goals[0])
    last, last_end = differential_path(starts[2], goals[2])
    rosbot, rosbot_end = [], starts[1]
    for distance, curvature in reference["primitives"]:
        rosbot_end = append_segment(rosbot, rosbot_end, distance, curvature, .15)
    paths = [first, rosbot, last]
    ends = [first_end, rosbot_end, last_end]
    best = None
    for delays in product(range(0, 61, 10), repeat=3):
        if min(delays):
            continue
        finish = max(len(trace) + delay for trace, delay in zip(paths, delays))
        if best and finish >= best[0]:
            continue
        samples = [[sample_path(trace, start, end, delay, tick)
                    for trace, start, end, delay in zip(paths, starts, ends, delays)]
                   for tick in range(finish + 21)]
        if not any(all(abs(command[0]) > 0 for _, command in team) for team in samples):
            continue
        clearance = min(math.dist(a[0][:2], b[0][:2]) - 2 * RADIUS
                        for team in samples for a, b in combinations(team, 2))
        if clearance < .13:
            continue
        best = finish, clearance, samples
    if best is None:
        return None
    finish, clearance, samples = best
    final_error = max(abs(math.dist(a[:2], b[:2]) - SIDE) for a, b in combinations(ends, 2))
    if final_error > .10 or any(abs(end[2]) > math.radians(5) for end in ends):
        return None
    rows = []
    for tick, team in enumerate(samples):
        robots = []
        for (pose, command), goal in zip(team, goals):
            x, y, heading = pose
            dx, dy = goal[0] - x, goal[1] - y
            c, s = math.cos(heading), math.sin(heading)
            robots.append(dict(pose=pose, command=command, target_world=(*goal, 0.),
                               target_body=(c*dx+s*dy, -s*dx+c*dy, c, -s)))
        rows.append(dict(time_s=tick*DT, robots=robots))
    return dict(seed=seed, finish_s=finish*DT, minimum_sampled_clearance_m=clearance,
                final_pair_error_m=final_error, samples=rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    task = json.loads(Path("configs/tasks/mixed_formation_nominal080.json").read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        for seed in args.seeds:
            result = plan_team(seed, task)
            if result is not None:
                stream.write(json.dumps(result) + "\n")
            print(seed, "accepted" if result is not None else "no qualifying schedule", flush=True)


if __name__ == "__main__":
    main()
