# Cruise geometry reward control

Freeze the HOLD hysteresis protocol (entry 0.15 m, exit 0.2 m). Reserved
59M tests found nominal 40/40 endpoint success for both protocols, and good
23/40 legacy versus 35/40 hysteresis, with zero collisions. Both had zero
whole-path formation successes. Do not interpret endpoint success as full-task
validation. Time alignment remains an optional negative experiment.

Test one reward change with identical MAPPO continuation settings. For maximum
pair-distance error e, tolerance tau=0.1 m and time step dt:

u = max(0, (e - 0.8*tau)/(0.2*tau))
r_barrier = -dt * w * min(u^2, 1)

Control w=0; treatment w=5. The penalty begins at 0.08 m and saturates at
0.1 m. This is a soft cost, not a safety guarantee. Existing progress, geometry,
collision and stopping rewards remain. Actors receive no true-state geometry
through this reward. There is no external controller or action replacement.

Both branches start from the same 120000-step cruise checkpoint, resume its
optimizers, then set actor LR=0.0001, Gaussian std=0.02, entropy coefficient=0,
gamma=0.99, GAE lambda=0.95, seed=451, nominal communication. Budget 20000
steps each, snapshots at 10000 and 20000; keep the approach actor frozen.
Training remains the original full-distance mission, not a shortened success
task. Changes common to both branches cannot be attributed to the barrier.

Select among snapshots on 8 development seeds 60M under nominal communication,
using endpoint and whole-path success, then formation compliance. Evaluate
the selected control and treatment plus the frozen baseline on 40 fresh seeds
61M for nominal and good. All use the same hysteresis protocol. Preserve the
baseline if geometry improves at the expense of completing the mission.
One training seed does not establish reproducibility across training seeds.
Bad/severe conditions remain outside this experiment.
