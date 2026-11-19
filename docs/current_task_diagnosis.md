# Current task acquisition diagnosis

The changes in this iteration passed all 32 project checks. Three 60000-step
training runs were completed: legacy curriculum, retained curriculum, and
one-factor acquisition reward. All training used nominal communication.

## Confirmed observations

- Baseline 20-episode diagnostic: 17 never formed, 3 succeeded.
- First held-out set: baseline 11/40, legacy fine-tune 6/40, retained fine-tune 9/40; zero collisions.
- Second held-out set: baseline 4/40, retained reward v5 8/40, acquisition reward v6 9/40.
  V6 had one collision; the baseline and v5 had zero. No replacement candidate was promoted.
- In the first set, all 29 failed baseline episodes stayed under the stopping velocity
  thresholds for all 30 seconds while never satisfying the geometry tolerance.
- Initial-state response audit: mean absolute baseline speed after one step was
  approximately 0.00251 m/s. This is an initial-state statistic, not an episode average.
- With the same baseline and second set, increasing evaluation time from 30 to 120 s yielded 4/40 to 20/40 nominal successes, zero collisions.
- Good communication at 120 s: 25/40 successes, 0 collisions.

## Interpretation and limits

Acquisition is often slow or incomplete. A 30-second formation-only cutoff was a
material contributor to recorded failures. Allowing four times longer does not
mean training improved, and it does not satisfy the full five-meter mission.
The retained curriculum and motion-cost gate did not demonstrate consistent
safe improvement across the held-out sets. Motion penalty alone is not an
adequate explanation for the low-command behavior. The main failure persists
even with perfect communication; severe network impairments remain deferred.

## Next experiment requirements

Specify a formation acquisition time budget consistent with the full mission,
measure convergence time and command magnitude over whole episodes, and test
learning incentives for faster acquisition without rewarding unsafe motion.
Investigate state-dependent exploration and critic credit assignment as
hypotheses with controlled comparisons. Do not extend duration alone or treat
the timeout diagnostic as a solved task. Preserve all previous actor artifacts.

See retained_curriculum_results.md, acquisition_reward_results.md and
formation_time_budget.md for separate experiment results and seed provenance.
