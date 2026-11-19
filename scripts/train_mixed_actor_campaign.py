"""Refine the four Gazebo actors with teacher-labelled planar rollouts."""
import argparse
from collections import Counter
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cooperative_pose_reference import (
    CooperativePoseReference, CooperativeTranslationReference, hold_braking_risk,
    hold_pose_ready, stopped,
)
from cooperative_telemetry import CooperativeTelemetry
from swarm_core.dynamics import RobotState, step
from swarm_core.mixed_commands import constrain_mixed_commands, update_rosbot_curvature


DT = .1
ACTORS = (
    ("go2_actor.json", "anchor_pose_v1", 5, 2),
    ("wheeltec_actor.json", "wheeltec_vertex_arc_v1", 10, 2),
    ("jackal_actor.json", "anchor_pose_v1", 5, 2),
    ("translation_actor.json", "translation_speed_v1", 3, 1),
)
PROFILES = ("nominal", "good", "moderate", "severe")
# Number of translation episodes in each four-episode cycle.
PILOT_TRANSLATION_EPISODES = {"nominal": 2, "good": 2, "moderate": 3, "severe": 2}


def read(path):
    return json.loads(path.read_text())


def atomic_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def load_models(directory):
    artifacts, models = [], []
    for name, contract, inputs, outputs in ACTORS:
        artifact = read(directory / name)
        if artifact["contract"] != contract:
            raise ValueError(f"Incorrect actor contract in {name}.")
        net = torch.nn.Sequential(torch.nn.Linear(inputs, 32), torch.nn.Tanh(),
                                  torch.nn.Linear(32, outputs))
        net.load_state_dict({k: torch.tensor(v) for k, v in artifact["weights"].items()})
        artifacts.append(artifact)
        models.append(net)
    return artifacts, models


def infer(model, features):
    with torch.no_grad():
        return model(torch.tensor(features, dtype=torch.float32, device=next(model.parameters()).device)).tolist()


def run_episode(task, profile, seed, models, samples, budget, translation_phase,
                *, force_learner=False, recover_arrival=False):
    rng = np.random.default_rng(seed)
    goals = [(0., -.6, 0.), (math.sqrt(1.08), 0., 0.), (0., .6, 0.)]
    width = .018 if translation_phase else .15
    angle = .035 if translation_phase else .25
    states = [RobotState(x + rng.uniform(-width, width), y + rng.uniform(-width, width),
                         rng.uniform(-angle, angle)) for x, y, _ in goals]
    limits = task["robot_defaults"]
    telemetry = None if profile["profile_name"] == "nominal" else CooperativeTelemetry(profile, seed, 3)
    teacher = CooperativePoseReference(goals, task["operating_area"], wheeltec_reversal_deceleration_mps2=.30)
    teacher.wheeltec_actor_uses_vertex = True

    def record(index, features, label):
        if samples is not None:
            samples[index].append((features.copy(), label))

    teacher.demonstration_observer = record
    use_learner = force_learner or seed % 4 in (0, 1)
    if use_learner:
        teacher.anchor_pose_actors = {i: lambda features, i=i: infer(models[i], features)
                                      for i in (0, 2)}
        teacher.wheeltec_actor = lambda features: infer(models[1], features)
    translation = None
    if translation_phase:
        cruise = .32 if profile["profile_name"] == "severe" else .55
        translation = CooperativeTranslationReference([vars(s) for s in states], group_speed_mps=cruise,
                                                      adaptive_speed=True, smooth_entry=True, terminal_recovery=True)

        def speed_actor(features, reference):
            record(3, features, [reference])
            return infer(models[3], features)[0] if use_learner else reference

        translation.speed_actor = speed_actor

    terminal = None
    terminal_goals = [(*goal, 0.) for goal in translation.goals] if translation else None
    curvature = 0.
    hold = 0
    for tick in range(min(550 if translation_phase else 450, budget)):
        measured = [vars(s) for s in states]
        control, ages = (measured, None) if telemetry is None else telemetry.update(measured, tick * DT)
        terminal_reason = None
        if control is None:
            commands = [[0., 0.]] * 3
            ready = bool(terminal is not None and hold
                         and hold_pose_ready(measured, terminal_goals, pair_tolerance_m=.05)
                         and all(stopped(state) for state in measured))
        elif translation is None:
            commands, status = teacher.command(control, go2_local_state=measured[0],
                                               wheeltec_local_state=measured[1],
                                               jackal_local_state=measured[2] if profile["profile_name"] == "severe" else None)
            if status == "unresolved":
                terminal_reason = "unresolved"
            ready = status == "settled"
        else:
            commands = translation.command(control) if terminal is None else [[0., 0.]] * 3
            if profile["profile_name"] == "severe":
                commands = [[min(v, .42), w] for v, w in commands]
            metrics = translation.metrics(measured)
            ready = metrics["settled"] and metrics["max_pair_error_m"] <= .05
            if recover_arrival:
                if (terminal is None
                        and translation.reference_progress >= translation.distance_m - .08
                        and translation.metrics(control)["max_destination_error_m"] <= .40
                        and all(stopped(state) for state in control)):
                    terminal = CooperativePoseReference(
                        terminal_goals, task["operating_area"], precision=True,
                        wheeltec_heading_tolerance_rad=math.radians(3),
                        wheeltec_reversal_deceleration_mps2=.30,
                        wheeltec_path_pair_tolerance_m=.075,
                        wheeltec_goal_pair_tolerance_m=.05,
                    )
                    terminal.demonstration_observer = record
                    terminal.wheeltec_actor_uses_vertex = True
                    terminal.anchor_pose_actors = teacher.anchor_pose_actors
                    terminal.wheeltec_actor = teacher.wheeltec_actor
                ready = False
                if terminal is not None:
                    estimated_ready = hold_pose_ready(control, terminal_goals, pair_tolerance_m=.05)
                    if estimated_ready:
                        commands = [[0., 0.]] * 3
                        ready = (all(stopped(state) for state in control)
                                 and hold_pose_ready(measured, terminal_goals, pair_tolerance_m=.05)
                                 and all(stopped(state) for state in measured))
                    else:
                        commands, status = terminal.command(
                            control, go2_local_state=measured[0],
                            wheeltec_local_state=measured[1], jackal_local_state=measured[2],
                        )
                        if status == "unresolved":
                            terminal_reason = "unresolved"
                        braking_states = [control[0], measured[1], control[2]]
                        if hold_braking_risk(braking_states, commands[1], .30):
                            commands[1] = [0., 0.]
                            terminal.path.clear()
        # An outage adds no confirmation ticks. Reset if measured geometry is lost.
        if not ready:
            hold = 0
        elif control is not None:
            hold += 1
        if hold >= 20:
            terminal_reason = "success"
        if terminal_reason is not None:
            commands = [[0., 0.]] * 3
        commands = constrain_mixed_commands(commands)
        if telemetry is not None:
            commands = telemetry.constrain_commands(control, ages, commands)
        commands = CooperativeTelemetry.filter_local_clearance(measured, commands)
        updated = []
        for i, (state, (v, w)) in enumerate(zip(states, commands)):
            steer = None
            if i == 1:
                curvature = update_rosbot_curvature(curvature, v, w, DT, .15)
                steer = curvature
            scale = limits["max_forward_speed_mps" if v >= 0 else "max_reverse_speed_mps"]
            updated.append(step(state, [v / scale, w / limits["max_yaw_rate_radps"]],
                                limits, DT, steering_curvature=steer))
        states = updated
        if min(math.hypot(a.x-b.x, a.y-b.y) for i,a in enumerate(states) for b in states[i+1:]) < .65:
            return tick + 1, "clearance_limit"
        if terminal_reason is not None:
            return tick + 1, terminal_reason
    return tick + 1, "time_limit"


def update_models(models, optimizers, samples):
    losses = {}
    for index, rows in enumerate(samples):
        if not rows:
            continue
        net, optimizer = models[index], optimizers[index]
        device = next(net.parameters()).device
        x, y = (torch.tensor(values, dtype=torch.float32, device=device) for values in zip(*rows))
        for ids in torch.randperm(len(x), device=device).split(256):
            loss = (net(x[ids])-y[ids]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses[ACTORS[index][0]] = {"samples": len(rows), "mse": loss.detach().item()}
    return losses


def evaluate_checkpoint(task, profiles, models, *, recover_arrival=False):
    results = {}
    for index, (name, profile) in enumerate(profiles.items()):
        rows = []
        for translation_phase in (False, True):
            for offset in (0, 1):
                seed = 2900000 + index*10 + 2*int(translation_phase) + offset
                ticks, reason = run_episode(task, profile, seed, models, None, 550,
                                            translation_phase, force_learner=True, recover_arrival=recover_arrival)
                rows.append(dict(seed=seed, phase="translation" if translation_phase else "formation",
                                 reason=reason, steps=ticks))
        results[name] = rows
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--steps-per-profile", type=int, required=True)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--robustness-pilot", action="store_true",
                        help="Use new rollout seeds and prioritize degraded communication.")
    parser.add_argument("--recover-arrival", action="store_true")
    parser.add_argument("--formation-focus", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    if args.steps_per_profile < 2:
        parser.error("Each profile needs at least two steps.")
    if args.evaluation_interval < 1:
        parser.error("Evaluation interval must be positive.")
    torch.set_num_threads(1)
    torch.manual_seed(196000)
    artifacts, models = load_models(args.source)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable.")
    models = [net.to(args.device) for net in models]
    optimizers = [torch.optim.Adam(net.parameters(), lr=.00015) for net in models]
    task = read(Path("configs/tasks/mixed_pose_curriculum_near.json"))
    profiles = {name: read(Path(f"configs/communication/{name}.json")) for name in PROFILES}
    budgets = dict.fromkeys(PROFILES, args.steps_per_profile)
    if args.robustness_pilot:
        # Allocate in 1:1:3:5 proportions, preserving the total rollout budget.
        total = sum(budgets.values())
        budgets = {name: total * share // 10
                   for name, share in zip(PROFILES, (1, 1, 3, 5))}
        budgets["severe"] += total - sum(budgets.values())
        if min(budgets.values()) < 2:
            parser.error("Robustness pilot needs at least five steps per profile.")
    args.output.mkdir(parents=True, exist_ok=False)
    atomic_json(args.output / "manifest.json", dict(
        source=str(args.source),
        source_sha256={name: hashlib.sha256((args.source / name).read_bytes()).hexdigest()
                       for name, _, _, _ in ACTORS},
        steps_per_profile=args.steps_per_profile,
        profile_budgets=budgets,
        robustness_pilot=args.robustness_pilot,
        recover_arrival=args.recover_arrival,
        formation_focus=args.formation_focus,
        device=args.device, seed_offset=args.seed_offset,
        evaluation_interval=args.evaluation_interval,
        method="Teacher-labelled planar rollouts with periodic learner control",
        profiles=profiles))
    progress = {profile: 0 for profile in PROFILES}
    episodes = Counter()
    outcomes = {profile: Counter() for profile in PROFILES}
    checkpoint = 0
    started = time.monotonic()
    while progress != budgets:
        samples = [[] for _ in ACTORS]
        for profile_index, (name, config) in enumerate(profiles.items()):
            stop = min(progress[name]+2000, budgets[name])
            while progress[name] < stop:
                episode = episodes[name]
                seed = 1960000 + profile_index*100000 + episode + args.seed_offset
                translation_phase = episode % 2 == 1
                if args.robustness_pilot:
                    seed += 2000000
                    translation_phase = episode % 4 < PILOT_TRANSLATION_EPISODES[name]
                if args.formation_focus:
                    seed += 4000000
                    translation_phase = False
                ticks, reason = run_episode(task, config, seed, models, samples,
                                            stop-progress[name], translation_phase,
                                            recover_arrival=args.recover_arrival)
                phase = "translation" if translation_phase else "formation"
                outcomes[name][f"{phase}:{reason}"] += 1
                episodes[name] += 1
                progress[name] += ticks
        losses = update_models(models, optimizers, samples)
        checkpoint += 1
        directory = args.output / f"checkpoint_{checkpoint:04d}"
        directory.mkdir()
        for (name, _, _, _), artifact, net in zip(ACTORS, artifacts, models):
            weights = {k: v.detach().tolist() for k, v in net.state_dict().items()}
            atomic_json(directory / name, {**artifact, "weights": weights})
        result = dict(steps_by_profile=progress, total_environment_steps=sum(progress.values()),
                      elapsed_wall_s=time.monotonic()-started,
                      steps_per_second=sum(progress.values())/(time.monotonic()-started),
                      checkpoint=str(directory), outcomes=outcomes, updates=losses)
        atomic_json(args.output / "progress.json", result)
        print(json.dumps(result), flush=True)
        if checkpoint % args.evaluation_interval == 0 or progress == budgets:
            validation = evaluate_checkpoint(task, profiles, models, recover_arrival=args.recover_arrival)
            atomic_json(directory / "evaluation.json", validation)
            print(json.dumps(dict(checkpoint=checkpoint, evaluation=validation)), flush=True)


if __name__ == "__main__":
    main()
