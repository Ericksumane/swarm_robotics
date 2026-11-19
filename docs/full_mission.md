# Full leaderless mission: form, translate, stop

## Scope

Environment: src/swarm_core/mission_env.py. Task: configs/tasks/full_mission.json.
Reward: configs/training/mission_reward.json. Observation contract:
mission_local16_v1. Previous formation-only checkpoints are incompatible.
The environment now specifies all three required behaviors. A trained policy
that succeeds at the complete mission has not yet been demonstrated.

## Local phases and information

Each robot has its own FORM, MOVE or HOLD phase, selected from its local
estimate and delivered peer messages. FORM readiness requires pair errors
within 0.10 m and low speeds for 2 s. Readiness is latched and transmitted
in StateMessage. A robot starts MOVE after its own readiness and valid
received readiness from all peers. There is no global phase broadcast.

For robot i, upon its local transition at t_i:

\(g_i = \hat p_i(t_i) + 5 d\), with \(\|d\|=1\).

The goal is frozen. The direction comes from the common mission definition;
positions are assumed to share a calibrated planar coordinate frame.
Different local transition times are possible with delayed messages.
The readiness protocol does not guarantee simultaneous start or safe operation
under arbitrary outages. Restart/session and distributed abort remain work.

HOLD begins inside a 0.20 m goal tolerance and returns to MOVE after leaving
that tolerance. Success requires all robots at their goals, correct formation
and low linear/angular speeds for 2 consecutive seconds. Formation alone is
never reported as full-mission success. Failure and time limits remain explicit.

## Actor and critic

The actor receives the original nine own-state/capability features plus a
three-value local phase one-hot vector, two body-frame goal-error components
and two body-frame mission-direction components. Received neighbor rows and
masks are unchanged. The shared encoder has trainable parameters.
Global true states are used only by reward, evaluation and the training critic.
The critic additionally sees phases, goals and readiness counters.

## Reward

Shared reward combines formation error and formation progress, mean progress
toward frozen individual goals during MOVE/HOLD, a pre-contact clearance
penalty, time cost, settling cost outside MOVE and terminal success/failure.
Settling cost is deliberately not charged during MOVE, unlike the earlier
formation-only reward that preferred stopping everywhere.
Weights are explicit in mission_reward.json. This is a proposed task reward,
not a proof of safe or convergent control. It requires empirical validation.

## Verification boundary

scripts/check_full_mission.py checks readiness, missing-message behavior,
fixed five-meter goals, actual integrated straight-line translation and final
stop. It uses explicit test actions and state fixtures. It does not demonstrate
learned formation acquisition or learned mission execution.

ROS shadow runtime still implements the old formation-only interface. Do not
route mission artifacts to hardware through that node. Gazebo currently
replays recorded Python trajectories; it is not a closed-loop physics test.

## Train and evaluate

```bash
python scripts/train_formation.py --steps 30000 --task configs/tasks/full_mission.json --reward configs/training/mission_reward.json --initial-std 0.3 --gae-lambda 0.99 --output experiments/runs/full_mission_001
python scripts/evaluate_formation.py experiments/runs/full_mission_001 --episodes 10
```

Episode reports include local phases, goal errors and formation displacement.
No success claim is justified until these reports confirm all mission phases.
