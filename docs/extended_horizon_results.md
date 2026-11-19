# Extended-horizon training results

Same 120-second limit, distance/stop tolerances and common final starts for both policies.
Additional training steps: 200000. Training recovered from an interrupted process at step 69120.
Actor, critic and optimizer state were restored; episode and RNG state restarted as documented in protocol.json.
Selection: 12 development starts. Final tests: 40 different common starts.
Near-triangle initial positions with random headings. No broad-start or five-meter-mission claim.

| Policy/profile | Success | Collisions | Median successful duration (s) | Mean RMSE (m) |
|---|---:|---:|---:|---:|
| baseline_nominal_heldout | 20/40 | 0 | 47.050000000000004 | 0.1107 |
| baseline_good_heldout | 23/40 | 0 | 64.0 | 0.1077 |
| trained_nominal_heldout | 35/40 | 0 | 36.5 | 0.0966 |
| trained_good_heldout | 35/40 | 0 | 37.6 | 0.0957 |

Selected before final tests: ~/swarm_robotics/experiments/runs/extended_horizon_001/continued_final
Best trained checkpoint: ~/swarm_robotics/experiments/runs/extended_horizon_001/continued_final
No automatic deployment or replacement of the registered research candidate.
