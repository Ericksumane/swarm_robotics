"""Aggregate corrective labels on mixed-physics rollouts of the vertex actor."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from check_rosbot_formation_path import check
from cooperative_pose_reference import CooperativePoseReference
from swarm_core.initialization import sample_initial_states
from swarm_core.learning.mappo_adapter import hidden_inputs, make_policy
from train_vertex_pose_pilot import evaluate, examples, normalize, read


class CorrectiveRollout:
    def __init__(self, task, probability, rng, inputs, labels):
        self.task = task
        self.probability = probability
        self.rng = rng
        self.inputs = inputs
        self.labels = labels
        self.reference = None
        self.teacher_ticks = 0
        self.total_ticks = 0

    def __call__(self, states, goals, inputs, actions):
        if self.reference is None:
            self.reference = CooperativePoseReference(
                goals, self.task["operating_area"], precision=True
            )
        commands, status = self.reference.command([vars(state) for state in states])
        correction = normalize(commands, self.task["robot_defaults"])
        self.total_ticks += 1
        if status == "unresolved":
            return actions
        self.inputs.append(inputs.copy())
        self.labels.append(correction)
        if self.rng.random() < self.probability:
            self.teacher_ticks += 1
            return correction
        return actions


def training_cases(task):
    for seed in range(114301, 114305):
        states = sample_initial_states(task, seed)
        poses = [(state.x, state.y, state.heading) for state in states]
        plan = check(seed, poses, task)
        if plan["found"]:
            yield dict(seed=seed, planned=plan)


def fit(model, config, inputs, labels):
    x = torch.from_numpy(np.concatenate(inputs))
    y = torch.from_numpy(np.concatenate(labels))
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=.0003)
    for _ in range(24):
        for indices in torch.randperm(len(x)).split(256):
            hidden, masks = hidden_inputs(len(indices), config)
            actions, _, _ = model.actor(x[indices], hidden, masks, deterministic=True)
            loss = torch.nn.functional.mse_loss(actions, y[indices])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), .5)
            optimizer.step()
    return len(x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(114300)
    torch.set_num_threads(1)
    rng = np.random.default_rng(114300)
    root = Path("experiments/runs")
    source = root / "nominal_vertex_pose_pilot_001/experimental_actor.pt"
    bundle = torch.load(source, weights_only=True)
    task = bundle["task"]
    communication = read("configs/communication/nominal.json")
    config, model, _ = make_policy(local_dim=19, neighbor_dim=9,
                                   critic_dim=bundle["critic_dim"], geometry_summary=True)
    model.actor.load_state_dict(bundle["actor"])
    initial_weights = deepcopy(model.actor.state_dict())
    original = root / "cooperative_reference_001/demonstrations.jsonl"
    data = [examples(task, communication, record)
            for record in map(json.loads, original.read_text().splitlines())
            if record["seed"] not in (8607, 8608)]
    inputs, labels = [list(items) for items in zip(*data)]
    held_out = root / "mixed_mission_repair_001/wheeltec_reference_evaluation.jsonl"
    evaluation = [json.loads(line) for line in held_out.read_text().splitlines()]
    cases = list(training_cases(task))
    if len(cases) < 2:
        raise RuntimeError("Fewer than two development starts have a reference path.")
    args.output.mkdir(parents=True)
    report = {"source": str(source), "development_seeds": [r["seed"] for r in cases],
              "held_out_seeds": [r["seed"] for r in evaluation], "rounds": []}
    for index, probability in enumerate((.75, .45), 1):
        collection = []
        for case in cases:
            rollout = CorrectiveRollout(task, probability, rng, inputs, labels)
            result = evaluate(task, communication, case, model.actor, config,
                              intervention=rollout, max_steps=220)
            collection.append({**result, "teacher_ticks": rollout.teacher_ticks,
                               "total_ticks": rollout.total_ticks})
            print("collection", index, collection[-1], flush=True)
        count = fit(model, config, inputs, labels)
        results = [evaluate(task, communication, case, model.actor, config)
                   for case in evaluation]
        report["rounds"].append(dict(round=index, teacher_probability=probability,
                                     samples=count, collection=collection, evaluation=results))
        bundle["actor"] = deepcopy(model.actor.state_dict())
        torch.save(bundle, args.output / f"actor_round_{index}.pt")
        (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        print("evaluation", index, results, flush=True)
    assert any(not torch.equal(initial_weights[key], value)
               for key, value in model.actor.state_dict().items())


if __name__ == "__main__":
    main()
