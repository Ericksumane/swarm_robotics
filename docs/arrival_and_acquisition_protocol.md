# Arrival recovery and dispersed initialization validation

Preserve the 40/40 nominal and good whole-path translation reference. The severe
12-case diagnostic failed on final goal position, not final speed or geometry.

## Arrival training

Only the near-goal actor is trained. Cruise weights, routing distance 0.7 m,
observation contract, HOLD entry/exit 0.15/0.2 m, and physical success criteria
remain fixed. Build training reset banks on fresh 64M starts from good and severe
encounters at first approach and at terminal/90 s, plus 12 legacy approach resets.
These are simulator states for initialization, not demonstrated actions. Actors
still see only their own state and delivered neighbor messages.

Continue the 100096-step approach checkpoint for 40000 steps, seed 452, actor
LR 0.00003, fixed MOVE std 0.04 and existing HOLD std multiplier 1/12, gamma
0.99 and upstream MAPPO updates. Rebuild the environment per episode with seeded
network selection: nominal 12.5%, good 25%, moderate 12.5%, severe 50%. Keep
the 45 s component-training limit; evaluation is the full 5 m /120 s mission.
Snapshots every 10000 steps. Choose on four fresh 65M starts per nominal/good/
severe profile, requiring 4/4 strict nominal AND good successes before ranking
severe strict success, endpoint success, then final goal error. Freeze before
40 fresh 66M reserved starts per nominal/good/moderate/severe profile. No final
case may be used for checkpoint selection. Do not promote a regression.

## Dispersed starts

Use positions uniform in [-1.5,1.5] m in each axis, pair separation >=0.7 m,
random headings in [-pi,pi], zero initial speed, no near-triangle mixture.
Validate the frozen formation actor on 20 fresh 63M starts per nominal/good.
Then test the full FORM/5 m/stop chain on 10 of the same starts per profile.
The formation contract is projected from mission observations by retaining the
first nine own features and the unchanged received neighbor rows/masks. Each
robot chooses its actor from its own local phase. There is no leader or global
phase broadcast. Routing and projection parity are tested.

Measure acquisition as true pair error <=0.1 m and all absolute speeds <=0.03
m/s, yaw rates <=0.05 rad/s for 2 s. These metrics are evaluator-only. Report
initially formed cases, acquisition, collisions and complete-mission success
separately. Whole-path geometry is expected to be invalid during acquisition;
the full-chain strict metric begins when all agents have locally started MOVE.
The complete chain has the existing 120 s limit, not an implicit extra budget.
These results do not validate real robot dynamics or shared-frame localization.

## Follow-up: diagnosed cruise drift

The first approach-only continuation preserved nominal/good development results
but achieved 0/4 severe on every snapshot. Captured severe approaches already
had 0.373 m average lateral error at approach entry, versus 0.095 m for good.
Therefore test a cruise-only change rather than repeatedly extending near-goal
training. Keep the original approach actor frozen, continue the selected cruise
checkpoint 20000 steps, seed 454, LR 0.0001, std 0.02, gamma 0.99, and the
unchanged geometry-control reward. Episode network mixture: 25% nominal,
25% good, 50% severe. Snapshots at 10000 and 20000. Select on four 69M starts
per nominal/good/severe, require 4/4 strict nominal/good then rank severe
success and goal error. Freeze before 40 fresh 70M cases per four profiles.
This is a separately identified follow-up, not a retroactive change to the
first protocol. No success criterion, timeout or observation meaning changes.

## Follow-up: preserve specialists using local received-information availability

Cruise-only mixed-network continuation improved severe development starts but
regressed nominal/good. Test an explicit composition before replacing weights.
Preserve the original cruise and approach experts. Add one frozen degraded-cruise
expert from each cruise training snapshot. During cruise only, select the new
expert if either received-neighbor validity mask is zero; otherwise select the
original cruise expert. The existing local distance-based approach latch and
original approach actor remain unchanged. No actual network profile label,
true neighbor state or global quality signal is available to the router.

This is architecture local_network_routed_mappo_v3 with three independently
trained MAPPO actors. The routing rule is a fixed local software rule, not a
learned gating network or an end-to-end joint policy. Actions always come from
one learned actor, without an analytical motion controller or online weight
updates. The extra inference/packaging complexity must be reported explicitly.

Use the same 69M development cases for model selection, requiring 4/4 strict
nominal/good, then ranking severe strict success, endpoint success and final
goal error. Freeze before 40 fresh reserved 71M starts in all four profiles.
Reject a candidate that regresses nominal/good. Validate legacy all-valid action
parity, per-robot isolation, mask-only selection and actor-service compatibility.

The 71M candidate had 38/40 nominal and 40/40 good completions. Because all-valid
local routing reproduces the original actor exactly, compare the original
reference on the same 71M starts before interpreting nominal failures as a
regression. This evaluation does not change checkpoint selection. Report both
paired counts and any rescued/regressed cases, not only the earlier 61M 40/40.
