# Speed 0.33 m/s training campaign

## Requested standard and scope

The current live delivery registry now selects baseline_bundle.pt under
experiments/runs/speed033_campaign_001. Its task specifies forward 0.33 m/s,
reverse 0.165 m/s, yaw 0.8 rad/s, linear acceleration 0.3 m/s^2 and angular
acceleration 0.8 rad/s^2. Policy weights initially match the previous release;
the task/profile changed. Previous registry and manifest copies provide rollback.
The historical runtime installation archive remains the earlier frozen release.
New speed does not constitute physical-robot approval or improved performance.

## Exact training budget

Four independent communication environments (nominal, good, moderate, severe)
are advanced in parallel by one shared expert per stage. Each vector tick counts
as one environment step in EACH profile, not twelve steps or one sample per
robot. The requested budget is 200,000 per profile, 800,000 total environmental
steps and 2,400,000 individual robot transitions. Smoke/evaluation steps do not
count. No online deployment learning occurs.

| Expert | Environment steps per profile |
| --- | ---: |
| History-based formation | 90,000 |
| Fresh-message formation | 30,000 |
| Own-state alignment | 10,000 |
| Cruise | 25,000 |
| Degraded cruise | 25,000 |
| Approach/stop | 20,000 |
| Total | 200,000 |

This adapts six existing MAPPO experts, not one end-to-end mission policy.
Coordination, routing and stop supervisors remain explicit program logic.
Formation training mixes dispersed resets with known difficult DEVELOPMENT
initializations, including the user's severe failure. Those seeds cannot serve
as held-out evidence. Translation and arrival stages use their existing staged
resets; complete dispersed missions are evaluated separately afterward.

## Optimization and reproducibility

learning/training/train_speed033.py uses unchanged upstream R_MAPPO. The four
profile trajectories keep separate terminal/reset masks in GAE. Central critic
inputs never become actor observations. Actor learning rate is 1e-5, gamma .999,
GAE lambda .95, fixed exploration standard deviation .03, PPO clip .1 and up to
three epochs. Each new stage begins with two critic-only warmup updates. All
critic weights are freshly initialized: exported actors do not carry the old
critics/optimizer states. This is actor warm-start adaptation, not bitwise
continuation of a previous complete training checkpoint.

CPU rollout replicas synchronize actor/critic weights at every rollout boundary;
CUDA runs the PPO optimization if selected. Rollout length is 128 vector ticks,
with four minibatches per epoch and an approximate KL early stop. Updates check
finite outputs. Model/progress writes use atomic replacement. Full source hashes,
plan, device and source commit are recorded before production training.

Check progress in training/progress.json under the campaign directory. For an
interrupted campaign, rerun with the same output and --resume. It resumes model,
optimizer and completed-step counters and starts fresh episodes; partial episode
state is not restored, so the resumed trajectory is not bitwise identical.

The trained artifact is a candidate pending independent evaluation. Do not
silently replace it with a lower-step checkpoint, merge separately trained
weights by averaging, or count smoke steps toward the requested 200,000.
