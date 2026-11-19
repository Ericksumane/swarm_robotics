# Focused continuation on independently mined hard starts

## Diagnosis

The five previous nominal and good-profile failures used the same initial seeds.
They never satisfied the geometric tolerance; their best good-profile maximum
pair errors were about 0.108-0.134m against a 0.1m tolerance. They were not
failures to sustain a hold after reaching formation. Velocity commands remained
small. Those already-inspected test seeds are not recycled as training examples.

## Mining and reset mixture

Evaluate the frozen current actor on 40 new near-formation starts (seeds 20000000
onwards) and 20 new broad starts (21000000 onwards), under good communication.
Store only failed initial configurations as reset templates. During subsequent
on-policy training, 30% of resets vary those templates with +/-0.04m position
jitter, +/-0.35rad heading jitter, a random global rotation and a new center.
The remaining resets use 80% fresh near-formation and 20% fresh broad sampling:
overall expected proportions are 30% hard variations, 56% near and 14% broad.
Hard examples include both kinds and can increase effective broad coverage.
Reject overlaps, out-of-bounds placements and already-successful formations.

Only initial states are retained: no old actions, rewards or expert targets are
used for learning. Actor observations and MAPPO PPO updates remain unchanged.
Restore actor, critic and optimizers from the current research checkpoint; use
fresh rollout and environment seeds. This is not an exact RNG-state resume.

## Fixed conditions and evidence gates

Keep 120s per episode, 1m for each triangle edge, existing success thresholds,
reward v5 and fixed Gaussian action std 0.02. Train 120000 further steps with
good communication, saving checkpoints every 30000 steps. Compare to the frozen
current actor on separate selection seeds 22000000 onwards, final near seeds
23000000 onwards and broad diagnostic seeds 24000000 onwards. Training does not
include moderate/severe network faults. Do not replace the prior candidate if
matched final near performance or collision outcomes regress. Report broad
performance separately and keep the five-meter mission out of these claims.
