# Training implementation review

## Confirmed defects corrected

1. Formation success bonus was inside the progress-shaping conditional.
   Success now receives its configured bonus even when progress weight is zero.
   Regression: scripts/check_reward_success_independence.py.
2. Loading an initial actor overwrote the explicit initial standard deviation.
   Non-resume initialization now applies the requested scale after actor loading.
   Complete checkpoint resume retains saved exploration unless scheduled.
   Regression: scripts/check_exploration_initialization.py.

Neither defect explains the last nominal/good comparison: that experiment used
nonzero progress shaping and applied its standard deviation schedule before
collecting each rollout. Its reported evaluations remain valid.

## Optimizer diagnostics

The runner now records approximate KL, PPO clip fraction, actor and critic
gradient norms, and policy loss in updates.jsonl. Gradient norms are measured
before clipping by the upstream trainer. Approximate KL uses the sampled
estimator mean((ratio - 1) - log(ratio)); metrics average across minibatches
and epochs rather than measuring only the final updated policy.

Artifact: experiments/runs/ppo_review_diagnostic_001.
10000 nominal-communication steps, seed 43, same initial actor as the paired
comparison; this is an optimizer diagnostic, not an evaluation of task success.
Mean approximate KL: 0.003548; maximum rollout average: 0.005218.
Mean PPO clip fraction: 0.03749; maximum rollout average: 0.06576.
These observations do not support excessive policy change in this early
training segment. They do not establish stability after the exploration scale
reaches 0.02, which occurs after this diagnostic ends.

## Learning limitations still open

- Low action noise permits stationary success but reduces movement diversity.
  The original stopping experiment began with an exactly correct formation;
  it did not test acquisition. It cannot establish a generally optimal scale.
- The current near-formation curriculum still randomizes all robot headings.
  The unicycle dynamics require sequences of turns and translation; independent
  small per-step noise may provide poor exploration of those sequences.
  This is a hypothesis requiring a controlled heading-curriculum comparison.
- Shared scalar team reward provides indirect credit to each action. Correct
  GAE and finite gradients alone do not establish effective credit assignment.
- A terminal success bonus is sparse. Dense progress is based on RMSE while
  success constrains the largest pair error and all velocities. These signals
  are related but not identical. Future shaping changes need controlled tests.
- Twenty reused development starts and one paired training seed cannot justify
  a reliability claim or checkpoint selection for deployment.
- The present evidence concerns formation and hold only. Five-meter translation,
  heterogeneous hardware, and closed-loop Gazebo physics remain unvalidated.

## Next controlled experiment

Keep perfect communication and the unchanged MAPPO optimizer. Compare the current
random-heading curriculum against gradually broadened headings on the same
nontrivial formation starts. Use separate training, selection, and held-out
evaluation seeds. Log the new PPO diagnostics after standard deviation reaches
its floor. Keep the two-second stop and geometric success thresholds unchanged.
Do not extend training duration or claim improvement solely from reward growth.
