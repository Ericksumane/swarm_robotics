# Focused arrival adaptation: 100,000 environment steps

## Scope and budget

Only the approach actor is adapted. Formation, fresh/history routing, alignment,
both cruise experts, deployment stop supervisors, 0.33/0.165 m/s limits and actual
mission success tolerances remain unchanged. The total budget is 100,000
three-robot environment steps: 25,000 each in nominal, good, moderate and severe
communication. Smoke and evaluation steps are excluded. This is not 100,000
steps per profile. The optimizer is the unchanged upstream MAPPO PPO update.

## Corrected training design

The old free-moving approach task did not reproduce deployment's already parked
peers. ArrivalTrainingEnv applies the SAME GoalStopLatch and PredictiveStopGate
used by deployment, using each robot's local observation. Parked agents are
masked from the actor loss; the critic still trains. Fully inactive minibatches
use safe finite probability evaluation and skip actor updates. No expert
steering actions are provided, and no new deployment controller is introduced.

Reset data comprise verified positions, phases, immutable starts/goals and parked
flags from previous live failures/successes and old approach resets, with small
state perturbations. No demonstrated actions are used. Message queues/history
are resampled, not claimed to reproduce a recorded network trajectory. All
source cases, including previous campaign evaluation cases, are DEVELOPMENT
training data for this campaign. Confirmation uses separate initializations.

## Training-only reward extension

For active agents define q(e)=clip(max(e-0.16,0)/0.2,0,1), and Q=max q(e).
Added shared reward is -dt*2*Q_next + 20*(Q_before-Q_next), plus -5 times the
fraction of active agents crossing from <=0.19 m to >0.20 m. Existing formation
reward remains, reduced to weight 3 and maximum-formation progress weight 5;
collision/outside-area penalty is 200. Existing actual success conditions are
unchanged. These are project-specific shaping choices, not upstream MAPPO or
a claim of policy-invariant shaping.

If all local stop latches are parked while the instantaneous real success
conditions are false, the noiseless training episode is unrecoverable under that
same stop supervisor. It ends as training_park_deadlock with -40 rather than
wasting its entire horizon. Correct parked formations still accumulate the
normal two-second hold and succeed. A 45-second training timeout adds -20.
These early failures/rewards apply ONLY to training, not evaluation/deployment.

Actor warm start is the currently retained baseline. Critic is freshly initialized,
with four critic-only warmup updates. Actor LR is 1e-5, gamma .99, GAE lambda .95,
PPO clip .1, fixed exploration std .02 and at most three PPO epochs. The lower
arrival horizon and explicit active-goal cost target the observed late overshoot
and near-goal stalls. Full sources/settings/checkpoints are saved in
experiments/runs/arrival_refinement_100k_001. No success-rate claim is made before
paired independent mission and live Gazebo validation.
