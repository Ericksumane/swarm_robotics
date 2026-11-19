# Retained curriculum comparison

Two paired 60000-step fine-tunes from the same actor and identical fresh critics.
Reward, exploration schedule and optimizer are matched. Nominal communication only during training.
Selection uses 12 development starts; final tests use 40 different common starts.
Single training seed; near-formation initial geometry. Full five-meter mission is not evaluated.

| Evaluation | Success | Collisions | Failure diagnostics |
|---|---:|---:|---|
| baseline_nominal_heldout | 11/40 | 0 | {'never_formed': 29, 'success': 11} |
| legacy_nominal_heldout | 6/40 | 0 | {'never_formed': 34, 'success': 6} |
| retained_nominal_heldout | 9/40 | 0 | {'never_formed': 30, 'success': 9, 'hold_not_sustained': 1} |
| selected_good_heldout | 11/40 | 0 | {'never_formed': 29, 'success': 11} |

Selected before final evaluation: ~/swarm_robotics/experiments/runs/retained_curriculum_001/baseline
No automatic hardware deployment or replacement of the prior candidate.
