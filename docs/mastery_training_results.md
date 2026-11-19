# Revised formation training results

Selection used 12 development seeds; final tests used 40 different common starts.
One training seed only. Initial geometry remains near formation; broad placement is not tested.
No claim of five-meter translation or hardware readiness.

| Evaluation | Success | Collisions | Mean RMSE (m) |
|---|---:|---:|---:|
| revised_nominal_heldout | 4/40 | 1 | 0.1577 |
| revised_good_heldout | 4/40 | 1 | 0.1585 |
| revised_easy_diagnostic | 19/20 | 0 | 0.0973 |
| previous_nominal_heldout | 6/40 | 0 | 0.1307 |
| stationary_nominal_heldout | 0/40 | 0 | 0.1326 |

Selected checkpoint: ~/swarm_robotics/experiments/runs/mastery_revision_001/checkpoint_0060416
Training summary: {'completed_steps': 120000, 'elapsed_s': 290.02307093499985, 'episodes': 459, 'outcomes': {'time_limit': 355, 'collision': 7, 'success': 97}, 'mastery_curriculum': {'stage': 3, 'half_width_rad': 3.141592653589793, 'recent_episodes': 20, 'recent_success_rate': 0.05}, 'stop_reason': 'step_limit'}
