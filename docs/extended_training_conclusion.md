# Extended training conclusion

Both policies were tested on the same 40 initial seeds with identical 120-second limits.
The new policy completed 200000 additional training steps; checkpoint selection used separate seeds.
An interrupted process was recovered with actor/critic/optimizer state intact and fresh episode RNG.

| Communication | Previous success | Trained success | Previous collisions | Trained collisions |
|---|---:|---:|---:|---:|
| nominal | 20/40 | 35/40 | 0 | 0 |
| good | 23/40 | 35/40 | 0 | 0 |

Paired success changes: {'nominal': {'new_only': 18, 'previous_only': 3}, 'good': {'new_only': 16, 'previous_only': 4}}
Registered as updated research candidate: True
Candidate: ~/swarm_robotics/experiments/runs/extended_horizon_001/continued_final
One training seed and 40 evaluation starts do not establish a population success rate.
These tests cover positions near the target triangle and random initial headings.

Separate broad-position diagnostic (10 cases): {'seed': 13000000, 'episodes': 10, 'baseline_successes': 1, 'trained_successes': 5, 'baseline_collisions': 1, 'trained_collisions': 0, 'scope': 'Small broad-position diagnostic, separate from main near-formation evaluation.'}
The broad-position diagnostic is not a generalization guarantee.

## Geometry and scope

The retained triangle is equilateral: all three center distances are 1m.
Distances 1m,1m,2m define a collinear degenerate triangle and were not substituted.
The geometric tolerance and velocity/hold thresholds were not relaxed.
Five-meter translation and hardware execution remain unvalidated.
The registered candidate is for research and offline replay; ROS actuator deployment is not approved by these results.

## External references used

VMAS DiffDrive equations informed the independent unicycle integration check.
BenchMARL was reviewed for standardized training/evaluation practices.
ChoiRbot was reviewed for decentralized ROS 2 architecture; its README targets an older ROS release.
See repository_review.md for source links and compatibility boundaries.
