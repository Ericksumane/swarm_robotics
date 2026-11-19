"""Find a bounded-curvature RosBot path to a staged formation vertex."""

from heapq import heappop, heappush
from itertools import count
import json
import math
from pathlib import Path

from swarm_core.initialization import sample_initial_states
from swarm_core.mixed_commands import ROSBOT_TURN_RADIUS


def advance(pose, distance, curvature):
    x, y, heading = pose
    turn = distance * curvature
    chord = distance if not turn else 2 * math.sin(turn / 2) / curvature
    return (x + chord * math.cos(heading + turn / 2),
            y + chord * math.sin(heading + turn / 2),
            math.remainder(heading + turn, 2 * math.pi))


def find_path(start, goal, anchors, area, *, position_tolerance=.08,
              heading_tolerance=math.radians(5), minimum_center_distance=.60,
              stride=.18, goal_anchors=(), max_expansions=80000, goal_pair_tolerance=.085, path_pair_tolerance=None):
    radius = ROSBOT_TURN_RADIUS
    identifiers = count()
    queue = [(0., next(identifiers), 0., start, ())]
    visited = set()
    path_pair_limits = []
    if path_pair_tolerance is not None:
        for anchor in goal_anchors:
            initial_error = abs(math.dist(start[:2], anchor) - 1.2)
            limit = path_pair_tolerance if initial_error <= path_pair_tolerance else initial_error + .02
            path_pair_limits.append((anchor, limit))
    while queue and len(visited) < max_expansions:
        _, _, distance, pose, path = heappop(queue)
        x, y, heading = pose
        key = (round(x / (stride * .44)), round(y / (stride * .44)),
               round(heading / (stride * .67)))
        if key in visited:
            continue
        visited.add(key)
        if (math.dist(pose[:2], goal) <= position_tolerance
                and abs(heading) <= heading_tolerance
                and all(abs(math.dist(pose[:2], anchor) - 1.2) <= goal_pair_tolerance
                        for anchor in goal_anchors)):
            return path, pose, distance
        for travel in (stride, -stride):
            for curvature in (-1 / radius, 0., 1 / radius):
                samples = [advance(pose, travel * index / 6, curvature)
                           for index in range(1, 7)]
                if any(
                    not (area['x_bounds_m'][0] + .25 <= point[0] <= area['x_bounds_m'][1] - .25
                         and area['y_bounds_m'][0] + .25 <= point[1] <= area['y_bounds_m'][1] - .25)
                    or min(math.dist(point[:2], anchor) for anchor in anchors) < minimum_center_distance
                    or any(abs(math.dist(point[:2], anchor) - 1.2) > limit
                           for anchor, limit in path_pair_limits)
                    for point in samples
                ):
                    continue
                endpoint = samples[-1]
                length = distance + stride
                estimate = length + max(math.dist(endpoint[:2], goal),
                                        radius * abs(endpoint[2]))
                heappush(queue, (estimate, next(identifiers), length, endpoint,
                                 (*path, (travel, curvature))))
    return None


def check(seed, states, task, *, choose_reachable=False):
    start = states[1]
    a, b = states[0][:2], states[2][:2]
    length = math.dist(a, b)
    axis = tuple((right - left) / length for left, right in zip(a, b))
    center = tuple((left + right) / 2 for left, right in zip(a, b))
    side = task['team']['desired_pair_distance_m']
    anchors = [tuple(c + sign * u * side / 2 for c, u in zip(center, axis))
               for sign in (-1, 1)]
    height = side * math.sqrt(3) / 2
    goals = [(center[0] - sign * axis[1] * height,
              center[1] + sign * axis[0] * height) for sign in (-1, 1)]
    goal = min(goals, key=lambda vertex: math.dist(start[:2], vertex))
    if choose_reachable:
        obstacles = anchors + [a, b]
        result = None
        for goal in sorted(goals, key=lambda vertex: math.dist(start[:2], vertex)):
            result = find_path(start, goal, obstacles, task['operating_area'],
                               minimum_center_distance=.65, max_expansions=12000)
            if result is not None:
                break
    else:
        result = find_path(start, goal, anchors, task['operating_area'])
    report = dict(seed=seed, start=start, goal=goal, staged_anchors=anchors,
                  radius_m=ROSBOT_TURN_RADIUS, found=result is not None)
    if result is None:
        return report
    path, endpoint, length = result
    pose = start
    samples = [start]
    for travel, curvature in path:
        samples.extend(advance(pose, travel * index / 6, curvature)
                       for index in range(1, 7))
        pose = samples[-1]
    report.update(path_length_m=length, endpoint=endpoint,
                  max_pair_error_m=max(abs(math.dist(endpoint[:2], a) - side) for a in anchors),
                  minimum_circular_clearance_m=min(math.dist(p[:2], a) - .5
                                                  for p in samples for a in anchors),
                  reverse_distance_m=sum(-travel for travel, _ in path if travel < 0),
                  primitives=path)
    return report


def main():
    task = json.loads(Path('configs/tasks/mixed_formation_nominal080.json').read_text())
    manifest = Path('experiments/runs/mixed_formation_precision080_gazebo8103/manifest.json')
    recorded = json.loads(manifest.read_text())['initial_physical_states']
    task['initialization']['hard_dispersed'] = {'probability': 1., 'min_center_distance_m': 1.65}
    sampled = [vars(state) for state in sample_initial_states(task, 8304)]
    for seed, records in ((8103, recorded), (8304, sampled)):
        poses = [tuple(r[key] for key in ('x', 'y', 'heading')) for r in records]
        print(json.dumps(check(seed, poses, task)), flush=True)


if __name__ == '__main__':
    main()
