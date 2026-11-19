# HOLD hysteresis reserved evaluation

Frozen actors, 40 matched initial seeds 59000000 through 59000039 per profile.
This set was reserved after development selection. All episodes begin from the
original near-formed, approximately aligned distribution at zero progress;
goal translation is 5 m and the time limit is 120 s.

| Protocol | Nominal endpoint | Good endpoint | Whole-path formation | Collisions |
|---|---:|---:|---:|---:|
| Legacy entry/exit 0.2 m | 40/40 | 23/40 | 0/40 in each profile | 0 |
| Entry 0.15 m, exit 0.2 m | 40/40 | 35/40 | 0/40 in each profile | 0 |

Good endpoint completion increased from 57.5% to 87.5% on this matched set.
There were 13 rescued cases and 1 regressions relative to legacy.

Physical success was unchanged: all goal errors <=0.2 m, all pair errors
<=0.1 m, absolute linear speed <=0.03 m/s and angular speed <=0.05 rad/s,
continuously for 2 s. Hysteresis changes the local phase cue, not velocity
commands, actor weights or the scoring tolerances. See hold_hysteresis_protocol.md.

These endpoint results do not validate the complete formation-control task:
every reserved episode violated the pair-distance tolerance somewhere along
the path. Cruise geometry is therefore the next controlled training target.
The selected translation registry remains the historical reference while the
geometry experiment is evaluated. No bad/severe communication or hardware
claims follow from these results. Fixed synthetic localization and narrow
initialization still limit transfer.

Artifacts: experiments/runs/hold_hysteresis_final_001/summary.json, per-cell
evaluation.json, trajectories, manifests and the actor-only hysteresis_bundle.pt.
Reproduction: learning/evaluation/run_hold_hysteresis_final.py (requires a new
output directory before rerunning; do not overwrite the reserved evidence).
