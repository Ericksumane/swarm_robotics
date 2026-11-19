"""Distil longitudinal commands from nominal Gazebo translation traces."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cooperative_pose_reference import CooperativeTranslationReference


def examples(directory):
    summary = json.loads((directory / "summary.json").read_text())
    if summary["reason"] != "cooperative_mission_success":
        raise ValueError(f"Expected a completed mission in {directory}.")
    with (directory / "control.jsonl").open() as file:
        rows = [json.loads(line) for line in file]
    formation = summary["formation_completed_s"]
    terminal = summary["terminal_started_s"]
    start = min(range(len(rows)), key=lambda i: abs(rows[i]["elapsed_s"] - formation))
    controller = CooperativeTranslationReference(
        rows[start]["states"], group_speed_mps=summary["cooperative_cruise_speed_mps"],
        adaptive_speed=summary["cooperative_adaptive_speed"], smooth_entry=True,
        terminal_recovery=True,
    )
    data = []

    def record(features, reference_speed):
        data.append((features, [reference_speed]))
        return reference_speed

    controller.speed_actor = record
    for row in rows[start:]:
        if row["elapsed_s"] >= terminal - 1e-7:
            break
        controller.command(row["states"])
    if not data:
        raise ValueError(f"No translation samples in {directory}.")
    return data


def tensors(data):
    features, labels = zip(*data)
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("runs", nargs="+", type=Path)
    args = parser.parse_args()
    if len(args.runs) < 2:
        parser.error("Provide training traces and a separate final validation trace.")
    torch.set_num_threads(1)
    torch.manual_seed(195700)
    train_data = [row for path in args.runs[:-1] for row in examples(path)]
    vx, vy = tensors(examples(args.runs[-1]))
    rng = np.random.default_rng(195700)
    # Cover braking and catch-up inputs beyond the successful trajectories.
    extra = rng.uniform([0., -.5, -.12], [.69, .6, .40], (8192, 3))
    x, y = tensors(train_data + [(list(row), [float(np.clip(row[0]+.4*row[1]+row[2], .025, 1.60))])
                                 for row in extra])
    model = torch.nn.Sequential(torch.nn.Linear(3, 32), torch.nn.Tanh(),
                                torch.nn.Linear(32, 1))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    for _ in range(90):
        for ids in torch.randperm(len(x)).split(256):
            loss = (model(x[ids]) - y[ids]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        error = (model(vx) - vy).abs()
    args.output.mkdir(parents=True, exist_ok=False)
    actor = dict(contract="translation_speed_v1",
                 weights={key: value.tolist() for key, value in model.state_dict().items()})
    (args.output / "actor.json").write_text(json.dumps(actor) + "\n")
    result = dict(training_runs=[str(path) for path in args.runs[:-1]],
                  validation_run=str(args.runs[-1]), trace_samples=len(train_data),
                  synthetic_samples=len(extra), validation_samples=len(vx),
                  validation_mae_mps=error.mean().item(), validation_max_error_mps=error.max().item())
    (args.output / "training_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
