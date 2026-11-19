# Formation training diagnosis

## Observations

The progress-v3 policy collided in all five inspected evaluation episodes.
Progress toward the mean distance objective did not ensure minimum clearance.
Original action standard deviation was 1.0, so substantial samples were clipped
at the action bounds. GAE lambda=0.95 with gamma=0.999 has an approximate
trace horizon of dt/(1-gamma*lambda)=1.96 seconds, much shorter than many
observed collision approach sequences. These are diagnostic hypotheses,
not established causes of failure.

## Development experiment v4

Adds pre-contact penalty -dt*w_p*mean(max(0,(margin-clearance)/margin)^2).
Initial margin=0.3 m, w_p=10. Target 1 m center spacing with 0.25 m radii
has 0.5 m clearance and no proximity penalty.
Exploration std starts at 0.3; entropy coefficient=0.001; GAE lambda=0.99.
All actions still come from MAPPO; no classical control or demonstrations.
Several factors change together, so this experiment cannot isolate causality.
The prior five inspected seeds are now development seeds, not final holdout.
Future final testing must use a fresh reserved seed set and frozen settings.

## Measured v4 result

20000 training steps; five development episodes at seeds 100100–100104.
Success rate: 0.0. All five ended in collision.
Formation error reduction did not imply safe task completion.

## Next curriculum, not yet trained

configs/tasks/formation_curriculum.json samples 80% of episodes near a
randomly translated and rotated triangle, with independent coordinate jitter
of +/-0.12 m. Remaining episodes retain the original random placement.
Episode limit is 30 s. No controller, demonstration or actor ground-truth
input is added. Sampling is rejection-based and enforces existing bounds
and separation. Evaluation must retain the original broad initial conditions
and separately report near-formation stabilization; success on easy initial
conditions must not be presented as broad formation acquisition.

## Thirty-minute result and stationary reference

Training completed 375952 environment steps in 1800.19 seconds.
The final actor succeeded in 1/5 near-start episodes and 0/5 in each broad
nominal/moderate/severe scenario. On the same five near-start seeds, the
stationary reference succeeded in 5/5. Thus near-start success is not evidence
of learned formation acquisition, and this campaign did not solve the task.

The initial +/-0.05 m curriculum admitted already successful geometry.
New profile formation_nontrivial_curriculum.json requires maximum pair error
at least 0.12 m, above the 0.10 m success tolerance, with +/-0.15 m jitter.
This correction is tested but has NOT been trained under the completed budget.
The historical campaign configurations and results are preserved unchanged.
Do not resume the old scheduled curriculum and assume this new criterion is
active: select the new profile explicitly and review the schedule separately.

## Curriculum configuration override corrected

The runner previously hardcoded its scheduled starting jitter to 0.05 m,
overriding the new profile's 0.15 m after the first episode. It now retains
the configured initial jitter and probability and interpolates from those
values. The minimum_pair_error_m criterion remains intact. This correction
does not retroactively change the completed thirty-minute run.
