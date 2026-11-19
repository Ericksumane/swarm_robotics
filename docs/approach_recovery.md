# Moving-start approach recovery

## Evidence and hypothesis

The full-distance heading-precision policy passed the destinations without
stopping (0/40 final successes). A paired development probe using the previous
short-distance actor within 0.5 m also achieved 0/8, versus 0/8 for cruise alone.
Do not register this naive composition as an improvement.

The short actor had been evaluated from rest and small initial heading errors.
Full-distance encounters include incoming velocity, lateral errors and distorted
geometry. This experiment explicitly trains those encountered states rather
than assuming that stationary-start success establishes braking competence.

## Data and episode scope

The frozen heading-precision actor generates 40 simulator encounters from
source seeds 49,000,000..49,000,039 for training and a separate 20 from
50,000,000..50,000,019 for development. Capture occurs when at least one robot
is within 0.7 m and all are within 1.2 m of their own fixed destinations.
Only states, original starts and goals are retained. No teacher actions,
expert demonstrations or old rewards are reused for PPO updates.

Resets perturb position by up to 0.01 m, heading by 0.08 rad and speed by
0.02 m/s, respecting footprints and motion limits. Goals remain unchanged.
Episode time starts at zero with a 45 s limit, even though the physical state
already represents partial mission progress. Near-goal successes must never
be reported as complete five-meter missions. Evaluation records initial
translation explicitly. Fresh receiver queues avoid old-position messages.

## Learning and evaluation

Continue the short-trained actor and centralized critic using unchanged upstream
MAPPO PPO updates: 150,000 steps, seed 449, gamma 0.99, GAE lambda 0.95,
Gaussian exploration std 0.02. Actor inputs remain local own state, mission
features and received peer messages. No programmed velocity or yaw controller.
Development sampling seeds start at 51,000,000. The development encounter
pool is used for checkpoint selection and is not the final independent test.

Rewards retain the mission terms and add a worst-pair-error progress term
20 (max_pair_error_before - max_pair_error_after). Settling weight becomes 1,
formation cost 10, translation progress 20 and goal-heading cost 1.
World-heading precision terms are disabled to allow lateral goal correction.
These are project-specific reward choices, not upstream MAPPO equations.

Full-distance validation must start from the normal initial triangle with
zero initial mission progress, using fresh cases, 5 m goals and 120 s cutoff.
Report endpoint success and continuous-formation success separately.
Any composition uses common frozen actors on every robot and a local
distance/phase selector; it is a project hierarchical extension, not a claim
that upstream MAPPO implements the selector or jointly trained the hierarchy.

The approach reward also uses wrapped absolute own-target bearing error B_i:
-0.5 dt mean_(MOVE)(B_i), and
5 mean_i[1_(MOVE before)(B_i before-B_i after)]. These target-directed terms
allow lateral correction rather than forcing world +x heading near the goal.
Their signs are checked with stationary turn-toward and turn-away fixtures.

## Phase-conditioned exploration comparison

A separate seed-450 run retains the same state bank, reward and 150,000-step
budget, but uses sigma_MOVE=0.12 and sigma_HOLD=0.01. The latter is obtained
with observable HOLD scale 1/12. Gaussian means are unchanged by this mechanism;
the actor must learn braking. Sampling and PPO evaluation use the same scaled
distribution, so stored log probabilities remain consistent. The extra phase
bit is internal to the actor graph; raw observation contract is unchanged.
Old models default to no scale and retain their original behavior. Checkpoints,
exports and inference consumers preserve the explicit hold_std_scale metadata.
This is a project exploration extension to the unchanged upstream PPO trainer.

## Frozen full-distance validation protocol

Select between the two approach experiments by the same 12-case near-goal
development score: fewer collisions, more successes, then lower maximum-goal
error averaged over episodes. Freeze this choice before full-distance tests.
Evaluate both cruise alone and the selected composition on the same 40 fresh
seeds 53,000,000..53,000,039. Use the normal full-distance reset, original 5 m
targets and 120 s limit. No warm-reset bank is used in full-distance testing.

The new composition latches APPROACH locally once the robot's observed goal
norm is below 0.5-1e-5 m or its local phase is HOLD. The latch resets between
episodes. It prevents a robot that overshoots from reverting to the cruise
policy. This is explicit mission-state routing, not a velocity controller;
both candidate velocity/yaw commands still come from frozen MAPPO actors.
This latched hierarchy differs from the rejected stateless 0/8 probe.
One training seed per exploration setting does not isolate exploration effects
from random-seed variation. Full-distance endpoint and continuous-formation
success remain separate required reported metrics.

Full-distance baseline and composition evaluations run in separate independent
processes, with distinct output folders and the same final seed range. Their
reports are joined only after both complete successfully. The local router is
shared between the evaluation code and actor-only inference service to avoid
different transition behavior at export. Bundle export is gated on a paired
endpoint improvement without additional collisions; continuous-formation limits
are retained explicitly rather than inferred from endpoint success.

## Entry-range correction and new independent validation

The first full validation (53M seeds) exposed late activation: training
encounters start at 0.7 m, whereas routing on the clipped actor vector entered
at 0.5 m. That changes incoming bearing errors and available correction space.
The first result is retained; it is not reused as an untouched final test
after this correction. A development replay may reuse inspected 53M cases.

Mission observations now expose own_goal distance separately as goal_distance_m,
computed exclusively from own pose estimate and own fixed goal before clipping.
The 36 actor inputs and all learned weights remain unchanged. Only the local
mission selector uses this full range to latch approach at 0.7 m. No peer truth,
central coordinator, velocity correction or teacher action is introduced.
The v2 inference API requires that distance metadata and checks consistency
with the clipped actor goal vector. Reset routing at every new mission.

New final validation is predeclared at seeds 55,000,000..55,000,039, paired
baseline and composition, with unchanged 5 m/120 s criteria. Preserve separate
endpoint and continuous-formation scores. Do not tune on this new final set.
