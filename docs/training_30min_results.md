# Thirty-minute formation training report

## Measured execution

- Training elapsed: 1800.19 seconds.
- Environment steps: 375952.
- Completed training episodes: 1275.
- Stop reason: wall_limit.
- Training outcomes: {'collision': 82, 'time_limit': 1193}.
- Three agents share one actor; an environment step includes all three actions.
- Offline training only; no physical robot was actuated.

## Evaluation

| Scenario | Success | Collision | Time limit | Mean formation RMSE (m) | Minimum clearance (m) |
|---|---:|---:|---:|---:|---:|
| near_nominal | 1/5 | 0 | 4 | 0.0932 | 0.0449 |
| broad_nominal | 0/5 | 2 | 3 | 0.3186 | -0.0013 |
| broad_moderate | 0/5 | 1 | 4 | 0.3141 | -0.0003 |
| broad_severe | 0/5 | 2 | 3 | 0.5789 | -0.0003 |

## Interpretation and limits

Evaluation seeds start at 200100 and are now development evidence. Five
episodes per scenario are insufficient to establish reliability. Near-start
stabilization is not evidence of formation acquisition from dispersed starts.
Formation-and-hold is not the five-meter translation mission.
Synthetic communication and geometry do not establish physical transfer.
A valid checkpoint and finite optimization losses do not establish task success.

## Intermediate checkpoints

Snapshots at approximately 5, 10 and 20 minutes are preserved under this run.
They are development comparisons, not a formally selected best policy.

## Reproduction

Configuration, source hashes and upstream commit: manifest.json.
Updates: updates.jsonl. Episode outcomes: episodes.jsonl.
Final weights: checkpoint.pt. Actor-only artifact: actor.pt.
Evaluation details and logs: adjacent _control directory.
Visual replay: replay.html. Gazebo replay accepts this run directory.
