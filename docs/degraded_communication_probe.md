# Degraded communication probe

Only start after the frozen control actor achieves 40/40 endpoint and strict
whole-path successes, with zero collisions, under both nominal and good on
reserved 61M seeds. Freeze the selected cruise and approach actors and HOLD
protocol. Change only the communication profile, with 12 fresh matched starts
62000000 through 62000011. These are exploratory diagnostics for the next
training stage, not a final robustness claim or a checkpoint selection set.

Existing profiles are synthetic, not measured lab Wi-Fi:

| Profile | Base delay | Jitter | Independent loss | Per-directed-link bandwidth | Scheduled outages |
|---|---:|---:|---:|---:|---|
| moderate | 100 ms | 50 ms | 10% | 2000 B/s | 15 to 16 s |
| severe | 300 ms | 200 ms | 30% | 500 B/s | 10 to 12 s; 25 to 28 s |

Both also include burst-state losses, bounded queues and 0.5 s information
expiration. See the versioned JSON for all burst transition probabilities.
The repository calls the first degraded condition moderate; it has not been
silently renamed bad or claimed to represent measured network quality.

Keep the 5 m /120 s task and all physical success inequalities unchanged.
Do not retrain, replace missing packets with simulator truth, relax tolerances,
or add an external controller in this diagnostic. Whole-path geometry and
collisions remain separate from endpoint success. Localization remains ideal.

## Observed results

The frozen selected model completed 12/12 moderate missions, all with whole-path
formation compliance, and 0/12 severe missions. Neither profile had a collision.
These 12-case exploratory counts are not equivalent to the 40-case reserved
nominal/good validation and do not establish general network robustness.

Severe diagnostic counts (overlapping conditions, not causal attribution):
- final_goal_outside: 12
- final_motion_outside: 0
- path_geometry_violation: 0
- time_limits: 12
- episodes: 12

Retain the nominal/good selected model. The severe result is a failure, not
grounds to increase the timeout or relax goal/stop criteria. Next isolate the
near-goal actor's behavior when packet age, expiration and queueing remove
neighbor observations. Use existing delivered-message masks and local state;
never substitute hidden simulator truth. A subsequent controlled training
experiment should preserve the cruise actor, mix network conditions for the
approach actor, and retest nominal/good for regression before promotion.
The present diagnostic did not retrain any policy in moderate or severe.
