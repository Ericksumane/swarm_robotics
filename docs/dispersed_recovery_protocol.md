# Learned acquisition and startup alignment

Preserve the selected translation experts and their narrow validation scope.
Previous broad starts yielded 10/20 acquisition successes with 3 collisions per
nominal/good condition, and 0/10 complete missions with 6 collisions. The cruise
actor expects headings near the mission direction, which acquisition did not ensure.

## Acquisition component

Continue the existing shared formation actor for 50000 environment steps, seed
455, LR 0.0001, gamma 0.999, GAE lambda 0.99. Start Gaussian std 0.05, anneal
to 0.02 at 25000 steps. Snapshots at 25000 and 50000. Training resets contain
75% uniformly dispersed positions and 25% previous near-triangle starts; all
headings are uniform [-pi,pi]. Validation is exclusively uniform dispersed
starts, never the easier training mixture. Keep the 120 s limit and physical
formation/speed tolerances unchanged.

Reward extends the existing formation reward: geometry weight 5, progress weight
20, proximity weight 30, collision/outside event penalties 200, success bonus 20,
ready reward 0.5/s. Settling cost applies only inside formation tolerance. These
are project-specific changes, not upstream MAPPO equations. Several settings
change together; resulting gains cannot be attributed to a single term. A finite
collision penalty is not a safety guarantee.

## Alignment component

Train a separate shared MAPPO actor with the existing mission-local observation
contract (own state, phase FORM, own mission direction, delivered neighbor data).
No goal or neighbor truth is exposed. Reset within valid formation but with
independent uniform headings. Critic uses mission-relative centralized state.
The actor reuses the translation encoder initialization with zero output means;
this is parameter initialization, not a runtime controller or expert actions.
Train 40000 steps, seed 456, LR 0.0001, gamma 0.99, GAE lambda 0.95, std 0.2
annealed to 0.02 over 20000 steps. Snapshots at 20000 and 40000.

Let e_i=wrap(theta_i-theta_command), E=mean_i |e_i| and G the pair-distance RMSE.
The component reward is:

r = 20*(E_before-E_after) - dt*[2*mean(1-cos(e_i))
    +2*mean((v_i/v_max)^2) +0.01*mean((w_i/w_max)^2) +20*G^2+0.02]
    + terminal_event.

Terminal success bonus is 20; collision/outside penalty is 200. Success requires
all |e_i|<=0.04 rad, pair error<=0.1 m and original speed bounds continuously
for 2 s. Component timeout is 45 s. This component does not count as a 5 m mission.
Progress shaping is not claimed to be policy-invariant potential shaping.

## Evidence separation

Select component snapshots on eight fresh 72M development starts per nominal/
good condition, with collision avoidance prioritized before success counts.
Freeze before 40 fresh 73M broad acquisition cases and 40 fresh 74M alignment
cases per profile. Compare the original acquisition actor on the same broad
cases. Integration must use local received-information readiness and a learned
alignment actor, not scripted motions or pose edits. Keep the original 120 s
complete-mission budget for initial integration tests; report component and
complete-chain outcomes separately. No translation registry is replaced solely
because a component passes. Hardware and noisy localization remain outside scope.

## Local startup integration

While local phase is FORM, each robot independently routes to the learned
alignment actor after its delivered-neighbor geometry is within 0.1 m and all
observed linear speeds are <=0.03 m/s for 2 s. Yaw motion does not prevent
another robot entering alignment. Alignment is retained while valid geometry
remains; missing peer information or geometry outside tolerance returns to
acquisition. Counters assume one call per 0.1 s control tick and reset per mission.
No global phase or simulator truth is used by this selection rule.

The new task additionally requires observed own/peer heading errors <=0.05 rad
in the existing distributed readiness test before a robot can start MOVE. The
default task behavior remains unchanged when this optional setting is absent.
The alignment actor emits both velocity requests itself; routing and readiness
never prescribe a yaw/linear velocity or edit a pose.

## Reserved integration timing diagnostic

Component selection prioritizes fewer collisions, then more successes, then
lower total elapsed time on development cases. Frozen component checkpoints
are evaluated before integration. Complete-chain evaluation uses 20 fresh 75M
starts per nominal/good profile, at the original 120 s limit. Also run the same
starts with a separately labelled 240 s limit to measure startup latency effects;
these are not independent extra samples and cannot replace the 120 s results.
No checkpoint is selected on these reserved outcomes. Both limits are declared
before evaluation. A 240 s success must not be reported as a 120 s success.

## Follow-up: hard close encounters and startup margin

The first selected acquisition checkpoint achieved 38/40 nominal and good
reserved acquisitions, but had one collision in each. Alignment achieved 40/40
in both. A separate follow-up mines failures from 40 fresh 76M starts, plus two
constructed close head-on initial configurations. No 73M reserved cases are
reused as training examples. Hard resets rotate/translate/perturb initial poses;
they do not prescribe actions or edit an ongoing rollout.

Train 30000 further steps, seed 457, LR 0.00005, gamma 0.999, GAE lambda 0.99,
std 0.04 annealed to 0.015 at 20000, snapshots at 15000 and 30000. Reset mixture
is 40% hard starts, 15% previous near-triangle and 45% uniform dispersed.
Use a stricter 0.05 m formation tolerance during this component training only;
benchmark physical formation criteria remain 0.1 m. Reward weights: geometry
10, proximity 50 with 0.4 m clearance margin, collision penalty 400; other first
recovery reward settings remain. Neither finite penalties nor observed zero
collisions would prove safety for arbitrary initial poses.

For integration, test a separately versioned startup geometry margin 0.05 m
for local acquisition-to-alignment and distributed MOVE readiness. Final
formation tolerance stays 0.1 m and final goal tolerance stays 0.2 m. The margin
aims to prevent a formation near the acceptance boundary from immediately
violating it during translation. Stricter startup can increase latency.
Select on eight 77M development starts per nominal/good, then freeze before
40 fresh 78M component starts per profile (paired against the first recovery
actor). Test the selected startup components on 20 fresh 79M starts per profile
with separate, labelled 120 s and 240 s complete-chain limits. Alignment and all
translation experts remain frozen. Do not claim a pass at the shorter limit
from the longer-limit results or promote an observed safety regression.

The hard-follow-up development selector measures the 0.05 m startup precision
target, including the previous selected actor as a frozen reference. It uses
fewer collisions, then more successes, then lower elapsed time. The following
78M physical benchmark returns to the original 0.1 m acceptance criterion for
both actors. Stricter development and original-criterion evaluation are labelled
separately; the observation/action contract and desired 1 m distances are unchanged.
