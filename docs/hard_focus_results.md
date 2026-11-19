# Focused hard-start training results

The original five inspected failures were not used as training seeds.
New mining pools supplied perturbed hard reset templates; no expert actions or old rollout data were reused.
The policy was fine-tuned for 120000 steps with good communication and unchanged task/reward criteria.
Selection and final tests use disjoint fresh seed ranges. One training seed only.

| Evaluation | Success | Collisions | Median successful time (s) |
|---|---:|---:|---:|
| baseline_nominal_heldout | 39/40 | 0 | 33.5 |
| baseline_good_heldout | 40/40 | 0 | 38.25 |
| baseline_broad_good | 10/20 | 1 | 79.95000000000002 |
| trained_nominal_heldout | 39/40 | 0 | 38.7 |
| trained_good_heldout | 40/40 | 0 | 39.550000000000004 |
| trained_broad_good | 9/20 | 0 | 67.10000000000001 |

Selected before final tests: ~/swarm_robotics/experiments/runs/hard_focus_001/checkpoint_0120000
Best trained checkpoint: ~/swarm_robotics/experiments/runs/hard_focus_001/checkpoint_0120000
The five-meter mission and physical robot deployment are not evaluated. No automatic candidate replacement.
