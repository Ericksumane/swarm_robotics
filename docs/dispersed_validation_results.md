Historical diagnostic before learned startup recovery. See [new recovery results](dispersed_recovery_results.md) for the latest evidence.

# Dispersed-start validation

This validation did not pass. Do not infer broad-position acquisition from the
earlier translation results, which began already formed and approximately aligned.

Initial centers are uniform in [-1.5,1.5] m per axis, at least 0.7 m apart,
with independent headings uniform in [-pi,pi] and zero initial velocities.
There is no near-triangle sampling mixture. All evaluated starts were outside
the formation tolerance at reset. Episode limit remains 120 s.

| Stage | Communication | Acquired formation | Completed 5 m mission | Collisions |
|---|---|---:|---:|---:|
| formation | nominal | 10/20 | not tested in this stage | 3 |
| formation | good | 10/20 | not tested in this stage | 3 |
| full_mission | nominal | 7/10 | 0/10 | 6 |
| full_mission | good | 7/10 | 0/10 | 6 |

Acquisition requires all pair errors <=0.1 m, absolute linear speed <=0.03
m/s and yaw rate <=0.05 rad/s continuously for 2 s. It is measured independently
of policy phase labels. Complete mission success additionally requires 5 m
translation to the per-robot goals and the original final hold conditions.
The complete-chain stage uses the first 10 of the same 20 starts; its counts
must not be pooled with the formation-only stage as independent samples.

## Diagnosed limitations

The frozen formation actor was trained on near-triangle starts, not this
uniform broad distribution. This is an observed generalization failure.
Some failures are collisions, so extending the time limit alone is insufficient.

The frozen cruise actor was trained with headings within +/-0.05 rad of the
mission direction. Formation acquisition constrains distances and speeds but
does not align headings. In nominal seed 63000007, all robots began MOVE at
51.3 s with headings [-2.1515, 0.3833, 1.9235] rad; collision followed at
58.3 s. This demonstrates an unsupported transition distribution. It does not
prove that heading mismatch is the sole cause of every complete-chain failure.

## Integration and next work

The evaluator composes the existing learned formation actor with the frozen
translation policy. Each robot selects FORM from its own phase. Formation
observations are projected from the same local state and received neighbor
rows/masks; no simulator truth is given to actors. This composition has passed
projection and local-routing tests, but empirical mission behavior failed.
The network-specialist translation candidate is a separate experiment and
has not been validated as a new complete-chain dispersed controller.

Do not promote this complete-chain composition. Next train acquisition on
genuinely broad positions, retain collision penalties and test startup heading
alignment or train cruise starts across the observed heading range. A learned
alignment/acquisition phase must be tested before linking it to the current
narrow cruise policy. Do not insert unreported scripted robot motions or alter
poses to manufacture a passing mission. Shared localization remains ideal.

Artifacts: experiments/runs/dispersed_validation_001; phase event and arrival
capture diagnostics: experiments/runs/stage_failure_diagnosis_001/diagnosis.json.
The initial integrated evaluator used the wrong reward schema and stopped before
full-chain rollouts. This configuration error was corrected; completed acquisition
cells were retained and missing cells resumed, with reward provenance recorded.
