"""Compare two mixed-fleet checkpoints on reserved planar seeds."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import torch

from train_mixed_actor_campaign import PROFILES, atomic_json, load_models, read, run_episode


def evaluate(job):
    name, directory, profile, seeds = job
    torch.set_num_threads(1)
    _, models = load_models(directory)
    task = read(Path("configs/tasks/mixed_pose_curriculum_near.json"))
    radio = read(Path(f"configs/communication/{profile}.json"))
    results = {}
    for phase in ("formation", "translation"):
        rows = []
        for seed in seeds:
            ticks, reason = run_episode(
                task, radio, seed, models, None, 550, phase == "translation",
                force_learner=True, recover_arrival=True,
            )
            rows.append(dict(seed=seed, steps=ticks, reason=reason))
        results[phase] = dict(counts=dict(Counter(row["reason"] for row in rows)), episodes=rows)
    return name, profile, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", type=int, default=16)
    args = parser.parse_args()
    if not 1 <= args.episodes <= 1000:
        parser.error("--episodes must be in [1, 1000].")
    if args.output.exists():
        parser.error("Use a new output path.")
    directories = dict(reference=args.reference, candidate=args.candidate)
    jobs = []
    for index, profile in enumerate(PROFILES):
        start = 3750000 + index * 1000
        for name, directory in directories.items():
            jobs.append((name, directory, profile, range(start, start + args.episodes)))
    results = dict(reference={}, candidate={})
    with ProcessPoolExecutor(max_workers=4) as executor:
        for name, profile, phases in executor.map(evaluate, jobs):
            results[name][profile] = phases
            print(name, profile, json.dumps({k: v["counts"] for k, v in phases.items()}), flush=True)
    atomic_json(args.output, dict(actors={k: str(v) for k, v in directories.items()}, results=results))


if __name__ == "__main__":
    main()
