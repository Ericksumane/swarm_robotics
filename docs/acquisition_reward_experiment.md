# Acquisition versus settling reward experiment

The newly instrumented baseline failed to form in 29/40 nominal episodes. In
every such episode the full team remained below the stopping speed/yaw
thresholds for all 30 seconds, with zero qualifying formation hold. These are
small-command, incorrect-geometry failures rather than excessive movement while
already formed. They do not prove motion cost is the sole cause.

Reward v6 changes exactly one mechanism relative to v5: the existing motion
penalty applies only when max pair error is within the formation tolerance.
Outside tolerance it is zero; progress, geometry cost, proximity, terminal
penalties and hold reward are unchanged. This uses evaluator truth for reward
calculation; no actor observations, commands or success thresholds are changed.
Hypothesis: removing settling pressure during acquisition improves geometry.

Compare to the retained-curriculum arm with identical initial actor, fresh
critic initialization, training seed, sample seed, optimizer, exploration and
60000-step budget. Select checkpoints on development seeds; use a new final
seed range for comparison. The policy must demonstrate improvement in behavior,
not merely a higher return under a changed reward.
