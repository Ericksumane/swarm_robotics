# Retained curriculum and failure diagnostics

Priority remains task competence with nominal/good communication. No moderate
or severe communication training is added.

FormationTermination now reports geometric readiness, low-speed readiness,
qualifying time, maximum continuous hold, first geometric readiness, best/final
maximum pair error, and final velocity maxima. Time-limit failures are classified
as never_formed, formed_never_stopped, or hold_not_sustained. These metrics use
simulator truth for evaluation only; actor observations and rewards are unchanged.
Success geometry, velocity thresholds and two-second duration remain unchanged.

The new curriculum uses heading half-widths [0.15,0.6,1.2,1.8,2.4,pi] instead
of jumping from 1.2 to pi. After learning a stage, 25% of new resets sample a
previous stage uniformly, while 75% train the current frontier. Promotion still
requires 70% successes over 20 frontier episodes. Rehearsal outcomes are excluded
from promotion statistics, preventing easy tasks from falsely establishing mastery.
A separately seeded curriculum RNG makes the reset mixture reproducible.
All trajectories remain on-policy; rehearsal changes initial states, not a replay
buffer of old transitions. Robot actions remain entirely MAPPO policy outputs.

The experiment compares legacy versus retained curriculum from the same saved
actor, fresh identical critics, matching training/sampling seeds, PPO budget,
reward and exploration schedule. Checkpoint selection uses development seeds;
final tests use different seeds and the same starts for both arms and baseline.
Any improvement is a single-training-seed result requiring replication.
