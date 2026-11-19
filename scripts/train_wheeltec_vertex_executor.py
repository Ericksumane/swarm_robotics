"""Train a WheelTec arc executor with fixed anchors and a local vertex target."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from check_rosbot_formation_path import find_path
from evaluate_wheeltec_reversal import run


def cases(task, seeds):
    anchors = [(0., -.6), (0., .6)]
    goal = (math.sqrt(1.08), 0.)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        start = (goal[0] + rng.uniform(-.25, .25), rng.uniform(-.10, .10),
                 rng.uniform(-.30, .30))
        path = find_path(start, goal, anchors, task["operating_area"],
                         position_tolerance=.04, heading_tolerance=math.radians(2.5),
                         minimum_center_distance=.65, stride=.06,
                         goal_anchors=anchors, goal_pair_tolerance=.055)
        yield dict(seed=seed, planned=None if path is None else dict(
            start=start, goal=goal, staged_anchors=anchors, primitives=path[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(193300)
    task = json.loads(Path("configs/tasks/mixed_pose_curriculum_near.json").read_text())
    task["rosbot_steering_time_constant_s"] = .15
    training = list(cases(task, range(193301, 193319)))
    held_out = list(cases(task, range(193401, 193409)))
    samples = []
    teacher_successes = 0
    for case in training:
        if case["planned"] is None:
            continue
        for disturbed in (False, True):
            trace = []
            report = run(case, task, disturbed, True, samples=trace, vertex_features=True)
            if report["startup_geometry_and_heading_ready"]:
                teacher_successes += 1
                samples.extend(row for row in trace if row["command"][0])
    if not samples or not any(row["command"][0] < 0. for row in samples):
        raise RuntimeError("The demonstration bank needs successful reverse examples.")
    x = torch.tensor([row["features"] for row in samples], dtype=torch.float32)
    y = torch.tensor([[v, w / v] for v, w in (row["command"] for row in samples)])
    model = torch.nn.Sequential(torch.nn.Linear(10, 32), torch.nn.Tanh(), torch.nn.Linear(32, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    for _ in range(90):
        for ids in torch.randperm(len(x)).split(256):
            loss = (model(x[ids]) - y[ids]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def predict(features):
        with torch.no_grad():
            return model(torch.tensor(features, dtype=torch.float32)).tolist()

    args.output.mkdir(parents=True, exist_ok=False)
    export = {"contract": "wheeltec_vertex_arc_v1",
              "weights": {key: value.detach().tolist() for key, value in model.state_dict().items()}}
    (args.output / "actor.json").write_text(json.dumps(export) + "\n")
    results = []
    for case in held_out:
        for disturbed in (False, True):
            for label, actor in (("reference", None), ("learned", predict)):
                row = (run(case, task, disturbed, True, actor=actor, vertex_features=True)
                       if case["planned"] else dict(seed=case["seed"], disturbance=disturbed,
                                                    reason="initial_plan_unresolved",
                                                    startup_geometry_and_heading_ready=False))
                results.append(dict(controller=label, **row))
                print(json.dumps(results[-1]), flush=True)
    summary = dict(training_seeds=[case["seed"] for case in training],
                   teacher_successes=teacher_successes, samples=len(samples),
                   reverse_samples=sum(row["command"][0] < 0. for row in samples),
                   evaluation=results)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
