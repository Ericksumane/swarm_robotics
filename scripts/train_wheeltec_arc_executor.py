"""Train the experimental WheelTec arc executor and run a seed-held-out check."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_wheeltec_reversal import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(114501)
    torch.set_num_threads(1)
    repair = Path("experiments/runs/mixed_mission_repair_001")
    task = json.loads((repair / "formation_training_task.json").read_text())
    cases = [json.loads(line) for line in (repair / "wheeltec_reference_evaluation.jsonl").read_text().splitlines()]
    args.output.mkdir(parents=True)
    samples = []
    demonstrations = [run(cases[0], task, disturbance, True, samples=samples)
                      for disturbance in (False, True)]
    if not all(row["startup_geometry_and_heading_ready"] for row in demonstrations):
        raise RuntimeError("Corrective demonstrations did not reach the required pose.")
    (args.output / "demonstrations.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in samples))
    moving = [row for row in samples if row["command"][0] != 0.]
    x = torch.tensor([row["features"] for row in moving], dtype=torch.float32)
    y = torch.tensor([[v, w / v] for v, w in (row["command"] for row in moving)])
    model = torch.nn.Sequential(torch.nn.Linear(7, 32), torch.nn.Tanh(), torch.nn.Linear(32, 2))
    optimizer = torch.optim.Adam(model.parameters(), lr=.003)
    for _ in range(350):
        loss = torch.nn.functional.mse_loss(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    export = {key: value.detach().tolist() for key, value in model.state_dict().items()}
    (args.output / "actor.json").write_text(json.dumps(export) + "\n")

    def predict(features):
        with torch.no_grad():
            return model(torch.tensor(features, dtype=torch.float32)).tolist()

    result = {"training_seed": cases[0]["seed"], "evaluation_seed": cases[1]["seed"],
              "samples": len(samples), "moving_samples": len(moving), "training_mse": loss.item(), "evaluation": []}
    for disturbance in (False, True):
        row = run(cases[1], task, disturbance, True, actor=predict)
        result["evaluation"].append(row)
        print(row, flush=True)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
