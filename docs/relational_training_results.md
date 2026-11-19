# Revised formation training results

Selection used 12 development seeds; final tests used 40 different common starts.
One training seed only. Initial geometry remains near formation; broad placement is not tested.
No claim of five-meter translation or hardware readiness.

| Evaluation | Success | Collisions | Mean RMSE (m) |
|---|---:|---:|---:|
| revised_nominal_heldout | 11/40 | 0 | 0.1139 |
| revised_good_heldout | 10/40 | 0 | 0.1137 |
| revised_easy_diagnostic | 16/20 | 0 | 0.0953 |
| previous_nominal_heldout | 2/40 | 0 | 0.1381 |
| stationary_nominal_heldout | 0/40 | 0 | 0.1292 |

Selected checkpoint: ~/swarm_robotics/experiments/runs/relational_revision_001/checkpoint_0020480
Training summary: {'completed_steps': 120000, 'elapsed_s': 315.5078492549983, 'episodes': 463, 'outcomes': {'time_limit': 359, 'success': 79, 'collision': 25}, 'mastery_curriculum': {'stage': 3, 'half_width_rad': 3.141592653589793, 'recent_episodes': 20, 'recent_success_rate': 0.05}, 'stop_reason': 'step_limit'}
