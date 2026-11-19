# Timed formation-stabilization campaign

## Preflight corrections

- Source hashes now include actual training/evaluation implementations under learning/.
- Training has a monotonic wall-clock limit and atomic checkpoints after updates.
- Reports include elapsed time, throughput, action clipping and action std.
- Optional shaping parameters are validated before the first transition.
- Explicit curriculum begins with 95% near triangles, +/-0.05 m jitter.
- Over the campaign this changes linearly to 30% near triangles,
  +/-0.25 m jitter; remaining starts use the original broad sampler.
- Curriculum changes only at episode reset; no state teleportation during episodes.

## Scope and evaluation

Offline MAPPO formation-and-hold, not translation or hardware validation.
No classical motion controller or demonstration is introduced.
The original upstream PPO update is unchanged.
Development settings: gamma=.999, lambda=.99, initial std=.3, entropy=.001.
Task: formation_curriculum_timed.json; reward: formation_reward_proximity.json.
Measure CPU and GPU throughput on short runs, then select a step target for
approximately 1800 seconds and enforce a wall-clock cap of 1800 seconds.
The final optimization/checkpoint may finish a few seconds after the cap.

Evaluate near-formation starts separately from broad starts, and nominal
communication separately from impaired communication. Report success,
collision, time limit, formation RMSE and minimum clearance. Fresh final
holdout seeds remain reserved until the experiment design is frozen.
