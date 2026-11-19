# Frozen-policy HOLD entry control

Time alignment was numerically valid but did not change development endpoint
counts: nominal 16/16 and good 5/16 for both versions. Keep the default legacy
observations. Preserve that negative result rather than adopting the adapter.

Historical failed missions all entered the goal region; 21 robot trajectories
crossed its boundary at least three times. Test a single phase parameter:
HOLD entry 0.15 m versus 0.2 m, retaining exit radius 0.2 m. This creates
phase hysteresis without changing actor weights or adding velocity commands.

Physical success remains every true goal error <=0.2 m, pair error <=0.1 m,
all |v|<=0.03 m/s and |w|<=0.05 rad/s for 2 s. Scoring no longer requires an
internal phase label. For the legacy protocol this is logically equivalent:
its phase update already makes every in-region robot HOLD. Unit tests and
reproduced legacy development counts verify that equivalence. A stricter
HOLD-entry cue must not accidentally make an otherwise physically complete
mission fail solely because its phase label is still MOVE.

All actor inputs retain the legacy mission_local_goal_v3 semantics. Compare
the two frozen protocols under nominal/good on the same 16 development seeds
57M. Candidate bundles record the phase protocol separately from actor weights.
If promising, freeze it and evaluate 40 new 59M cases per profile. Continuous
formation remains separately scored; this end-phase change cannot repair
formation violations that occur earlier during cruise.
