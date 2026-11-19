# Acquisition reward experiment results

One-factor comparison: settling motion cost is disabled only outside formation tolerance.
Matching initial actor, training/sampling seeds, fresh critic, task, optimizer and 60000-step budget.
Checkpoint selection used development seeds; final evaluation used 40 new common starts.
One training seed; near-formation positions. No five-meter mission claim.

| Evaluation | Success | Collisions | Failure types |
|---|---:|---:|---|
| baseline_nominal_heldout | 4/40 | 0 | {'never_formed': 35, 'success': 4, 'hold_not_sustained': 1} |
| retained_v5_nominal_heldout | 8/40 | 0 | {'success': 8, 'never_formed': 32} |
| acquisition_v6_nominal_heldout | 9/40 | 1 | {'never_formed': 29, 'success': 9, 'hold_not_sustained': 1, 'collision': 1} |
| selected_good_heldout | 4/40 | 0 | {'never_formed': 36, 'success': 4} |

Selected before final evaluation: ~/swarm_robotics/experiments/runs/acquisition_reward_001/baseline
No automatic deployment or replacement of the current candidate.
