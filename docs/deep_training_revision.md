# Formation training revision

## Scope and evidence boundary

This revision addresses formation acquisition and stationary hold with perfect
communication. It does not establish five-meter translation, general initial
placements, hardware support, or Gazebo closed-loop physics. All actions remain
samples or deterministic means of the shared MAPPO actor; no expert action,
imitation loss, or auxiliary motion controller is introduced.

## Reset curriculum

Implementation: src/swarm_core/curriculum.py and initialization.py.
For sampled positions, compute g_i = sum_{j != i} (d_ij - d*) (p_j-p_i)/d_ij.
Only at episode reset, sample heading_i = atan2(g_iy,g_ix) + U(-a,a).
This simplifies initial body orientations while preserving nontrivial geometry.
It is an initialization prior, not an action rule or an actor input.
Stages use a = [0.15, 0.6, 1.2, pi]. At a=pi headings are uniform.
Advance one stage after at least 70% successes in a full window of 20 episodes;
clear the history on promotion. No time-based automatic promotion or threshold
relaxation is used. The current stage is recorded in update and summary logs.
Mastery checkpoint resume is rejected until its state can be restored exactly.

## Reward v5

Configuration: configs/training/formation_reward_mastery.json.
Retain formation RMSE squared running cost, normalized speed/yaw running cost,
collision/outside penalties and proximity cost. Replace RMSE progress shaping
with 20*(e_max(t)-e_max(t+1)), where e_max=max_{i<j}|d_ij-d*|.
This is project-specific shaping and is not claimed policy invariant.
Add 0.5*dt for each qualifying formation-and-stop step; terminal success bonus
is 2 instead of 20, reducing the isolated terminal target jump. Geometric
tolerance 0.1m, speed tolerances and two-second continuous hold are unchanged.
Legacy reward configurations retain their old progress metric and zero hold credit.

## Evaluation protection

Task overrides now must match checkpoint observation contract and team size.
The previous check only compared checkpoint metadata to the original manifest,
allowing an incompatible replacement task through the guard.
Training, checkpoint selection, and final evaluation use separate seed ranges.
Evaluate random headings independently of curriculum performance; easy-reset
success cannot stand in for the requested general formation behavior.

## Validation

scripts/check_mastery_curriculum.py verifies nontrivial seeded resets, the reset
heading calculation, promotion windows, invalid configurations, dense hold credit,
the unchanged success duration, and exact worst-pair reward accounting.
Run all project regression checks before launching training.
