# Translation and stop training stage

## Scope and contract

Contract: mission_relational_v2. Start in a valid 1 m equilateral triangle,
with independent heading perturbations within 0.05 rad of world +x.
Geometry orientation and center are randomized; heading is deliberately easy.
All robots receive a preconfigured mission at reset, with no leader or central
policy input. This reset convention is not a distributed start protocol.
Each robot freezes its own estimated start and target g_i = p_i(0) + 5 d.
The existing full FORM readiness protocol is unchanged outside this stage.

Actor input: nine local state/capability features, phase one-hot (3),
body-frame target error (2), body-frame mission direction (2), plus received
nine-feature neighbor rows and masks. The relational encoder is optimized
jointly through upstream MAPPO PPO updates. No hand-coded action controller.
Critic: centralized training-only physical states, phases, goals, readiness
and hold time; it is absent from deployed actor inference.

## Reward

With dt=0.1, pairwise formation RMSE E, mean remaining own-target distance D,
normalized proximity cost C, and normalized squared non-MOVE velocities S:

r = -5 dt E_t^2 + 10(E_(t-1)-E_t)
    + 15 mean_i(D_i(t-1)-D_i(t)) - 10 dt C - 0.02 dt - 0.2 dt S.

Success adds 20; collision or operating-area exit subtracts 50 and terminates.
These are project-specific rewards, not equations introduced by upstream MAPPO.
Implementation: src/swarm_core/mission_env.py; settings:
configs/training/translation_reward.json.

Success requires every own-target error <=0.2 m, every pair error <=0.1 m,
all |v|<=0.03 m/s and |yaw rate|<=0.05 rad/s continuously for 2 seconds.
The 120 s cutoff is a truncation with value bootstrap and no cross-reset GAE.
An agent enters HOLD on reaching its target and resumes MOVE if it leaves.

## Evaluation

Train with good communication, keep formation-only candidate unchanged.
Use disjoint reset seeds for training, development and final evaluation.
Report successful 5 m missions, goal errors, displacement, final formation
error, clearance, collisions and timeouts. Starting aligned does not establish
performance with arbitrary headings or formation acquisition.
Python dynamics and ideal shared-frame localization remain simulation limits.

## Second pilot: coordination shaping

The first 120,000-step run did not learn a successful mission. Training episodes
showed curved paths and divergent progress. A second reward revision adds:

r_heading = -2 dt mean_(i in MOVE)[1-cos(theta_i-theta_direction)]
r_turning = -0.1 dt mean_(i in MOVE)[(omega_i/omega_max)^2]
r_speed_spread = -2 dt Var_(i in MOVE)(v_i/v_max).

These centralized training-only costs encourage parallel travel and similar
speeds. No velocity target, controller action, teacher or action correction is
provided. All actor commands still come from the shared learned MAPPO policy.
Optional weights default to zero for old mission experiments. Same 5 m goal,
reset distribution, success tolerances and time limit; new independent seed.

## Distance curriculum pilot

The first stage uses 0.5 m and a 30 s time limit to expose arrival and braking
more frequently. Its success is not evidence of a five-meter mission.
Reward revision v3 replaces world-direction heading cost with goal-bearing cost:
-0.5 dt mean_(i in MOVE)[1-cos(atan2(g_iy-y_i,g_ix-x_i)-theta_i)].
This permits lateral correction toward each own goal. Target, phase and heading
are local actor inputs; the cost does not prescribe an action. All old optional
weights default to zero. Success tolerances, dynamics and formation are unchanged.
Only promote this pilot to longer distances after evaluating learned stopping.

## Explicit neural initialization pilot

Both direct pilots and the first short-distance pilot failed evaluation.
The first short-distance policy advanced only about 0.17 m before its cutoff.
A separate short pilot uses a disclosed actor initialization: zero Gaussian
mean-head weights and bias [0.4, 0], giving an initial constant forward mean.
This is an action prior for initial neural weights, not a runtime controller,
expert dataset or scripted stop. Every mean-head and encoder parameter remains
trainable and deployed actions remain entirely neural policy outputs.
Stopping must still be learned from phase, goal error and rewards.
Exploration is fixed at 0.02 to make the 2 s stationary condition attainable.
The initializer and initial-actor hash are preserved for reproducibility.

## Relative mission critic

Optional critic contract mission_relative_v1 uses per-receiver body-frame peer
positions, sine/cosine headings, scaled velocities, phases and scaled own-goal
errors for each peer, followed by pair distances, hold fraction and body-frame
mission direction. Width for three robots is 39. Coordinates are invariant to
a common global translation. This replaces absolute coordinates only in new
experiments; old checkpoints retain their original critic width and contract.
The actor contract remains mission_relational_v2; central state is never fed
to actor inference. Tests verify invariance and the information boundary.

## Bounded local-goal observation contract

mission_local_goal_v3 changes only the two body-frame goal features:
e_local = R_i^T(g_i-p_i), e_actor = e_local min(1, h/max(||e_local||,1e-12)),
with horizon h=0.5 m. Direction is retained; distances within h are unchanged.
All physical destinations stay fixed at p_i(0)+5 d and success uses true full
target errors. This is an observation transform, not an action controller.
The explicit adapter records the source short-training task, checkpoint hash
and unchanged weights. It does not claim that a short-trained actor was trained
on five-meter episodes. New independent tests must validate any transfer.

## Full-distance local-goal continuation

The observation adapter alone did not solve long-distance drift. A bounded
180,000-step continuation uses the full 5 m task, same short-pilot gamma=0.99
and GAE lambda=0.95, plus world-direction heading weight 0.5 and own-goal bearing
weight 0.5. All success criteria remain unchanged. Development seeds start at
39,000,000 (8 per snapshot); final seeds at 40,000,000 (40). Record failed
experiments as well as any success; do not replace the formation candidate.

## Mastery-based distance curriculum

The final curriculum experiment uses 0.5, 1, 2, 3 and 5 m stages, with training
limits 30, 45, 60, 90 and 120 s. Promotion requires at least 6 successes among
8 frontier episodes. Twenty percent of resets rehearse earlier stages;
rehearsal never promotes an untested frontier. Only reset mission parameters
change; goals remain fixed during every episode. This curriculum never emits
actions and is absent from deployment. Training is bounded at 240,000 steps.
Checkpoint development tests (41,000,000) and final tests (42,000,000) always
use the fixed 5 m /120 s evaluation task, regardless of reached training stage.

## Continuous formation audit

Endpoint success alone does not prove formation was maintained during travel.
Evaluation additionally reports maximum pair error over the trajectory,
fraction of steps satisfying the 0.1 m pair tolerance, final speeds and yaw
rates, elapsed time and strict_formation_success. The latter requires both
endpoint success and every measured step within formation tolerance. Report
both metrics separately; do not describe endpoint-only success as continuous
formation maintenance. These diagnostics do not feed the actor or reward.

## Heading precision diagnostic and bounded reward experiment

Static heading probes (20 geometry fixtures, ideal messages) showed a weak
restoring local yaw slope, approximately -0.0046/s for the adapted short actor,
and mean absolute yaw bias about 0.011 rad/s at zero heading error. The selected
long-trained alternative retained a weak slope around -0.0048/s and bias about
0.0068 rad/s. These probes are not closed-loop stability proofs or success tests.

Quadratic heading and yaw costs are locally weak near zero. A final bounded
120,000-step experiment adds, with wrapped absolute world-heading error A_i:
r_abs = -dt mean_(MOVE)(A_i),
r_heading_progress = 10 mean_i[1_(MOVE before)(A_i before-A_i after)],
r_turn_abs = -0.2 dt mean_(MOVE)(|omega_i|/omega_max).
These are training reward terms only; no actions are corrected or supplied.
Old experiments default to zero for these optional terms. Goals, stopping
thresholds, triangle distances and 120 s final cutoff remain unchanged.
Development seeds 44,000,000; final seeds 45,000,000.
