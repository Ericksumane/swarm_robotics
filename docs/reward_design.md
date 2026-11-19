# Reward Design

## Scope

Version: formation_hold_v1

This reward applies only to the initial formation-and-hold stage.
It does not yet reward the final 5 m translation mission.

Implementation: src/swarm_core/rewards.py
Configuration: configs/training/formation_reward_v1.json

## Formation error

For each pair of robots:

d_ij = ||p_i - p_j||

With desired distance d_star and pair set E:

E_f = (1 / |E|) * sum_(i,j in E) [(d_ij - d_star) / d_star]^2

The initial three-agent task uses all three pairs.

## Motion cost

Let v_scale be the larger of the forward and reverse speed limits.
Let omega_scale be the yaw-rate limit.

E_v = (1 / N) * sum_i [
    (v_i / v_scale)^2 + (omega_i / omega_scale)^2
]

This term encourages settling after forming the triangle.
It is not a command-variation or energy model.

## Shared team reward

r_t = -dt * (w_f * E_f + w_v * E_v) - B_c * I_collision

Initial development weights:
- w_f = 1.0
- w_v = 0.05
- B_c = 10.0

Collision is detected using equal circular footprints at sampled states.
Continuous-path collision detection is not implemented.

The collision penalty is charged on the terminal collision transition.
The environment must prevent stepping after termination.

## Interpretation

A perfect stationary formation has reward zero.
Formation error and movement produce negative rewards.
All agents receive the same scalar team reward.

The reward uses true simulator states for training.
Actors receive only their permitted local observations.

## Verification completed

Analytical checks passed for:
- A perfect stationary equilateral triangle.
- A triangle with doubled pairwise distances.
- A perfect formation moving at maximum forward speed.
- Overlapping robot footprints.

## Required checks before training

The current weights are provisional.

Define episode success, failure and time-limit handling.
Check whether early failure can yield a better return than continued
operation because it removes future negative rewards.

Evaluate reward scales over complete trajectories.
Do not assume that a finite collision penalty guarantees safety.

Formation-and-hold results do not establish movement-task performance.

## Reward revision: formation_hold_v2

The environment adds an outside-area terminal penalty:

r_total = r_formation_hold - B_outside * I_outside_area

Initial B_outside = 10.0.

Implementation: src/swarm_core/formation_env.py

The outcome evaluator gives collision precedence over outside-area failure.
When both occur, only the collision event penalty is applied.

These finite penalties do not prove that early failure is unattractive.
Complete-trajectory return comparisons remain required before training.
The configuration filename remains formation_reward_v1.json;
its reward_version field identifies the active revision.
## Development experiment: formation_hold_progress_v3

Separate configuration: configs/training/formation_reward_development.json.
Adds 20*(previous_RMSE-current_RMSE), a success bonus of 20, and terminal
failure penalties of 50. This dense progress term is task-specific shaping,
not a claim of policy-invariant potential shaping.
The 50000-step development campaign failed held-out evaluation with collisions.
Do not treat this configuration or its weights as validated.
