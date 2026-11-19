# Controlled cruise geometry continuation

Reserved evaluation: 40 matched seeds 61000000 through 61000039 per cell.
Both trained branches use the same seed and settings; only barrier weight differs.
The approach actor and HOLD hysteresis remain fixed. All tests start already
formed and approximately aligned, from zero progress, with 5 m / 120 s tasks.

| Model | Communication | Endpoint success | Entire-path success | Collisions | Mean formation compliance |
|---|---|---:|---:|---:|---:|
| baseline | nominal | 40/40 | 0/40 | 0 | 58.0% |
| baseline | good | 32/40 | 0/40 | 0 | 54.9% |
| control | nominal | 40/40 | 40/40 | 0 | 100.0% |
| control | good | 40/40 | 40/40 | 0 | 100.0% |
| barrier | nominal | 11/40 | 0/40 | 0 | 19.5% |
| barrier | good | 8/40 | 0/40 | 0 | 19.3% |

Endpoint success requires every robot within 0.2 m of its own target,
all pair errors <=0.1 m, linear speed <=0.03 m/s and yaw speed <=0.05 rad/s
continuously for 2 s. Entire-path success additionally requires pair error
<=0.1 m at every simulated step. Mean compliance is a time fraction, not
a mission success rate. No tolerance or time-limit relaxation was used.

Each training branch completed 20000 additional environment steps. The selected
checkpoint was chosen on 8 nominal development cases before reserved tests.
See geometry_control_protocol.md for the cost equation and fixed settings.
The barrier is a project-specific reward extension; PPO updates use upstream
MAPPO unchanged. A single seed does not establish training-seed robustness.

Training logs, source hashes, optimizer settings, checkpoint selection and all
reserved trajectories are under experiments/runs/geometry_control_001.
No bad/severe communication, broad-position formation acquisition, real robot
dynamics or hardware transfer is established by this experiment. Gazebo remains
recorded Python-trajectory playback, not closed-loop Gazebo physics validation.

## Interpretation and selected artifact

The additional barrier cost was rejected: 11/40 nominal and 8/40 good endpoint
successes, with no whole-path successes. The zero-barrier continuation reached
40/40 endpoint AND whole-path successes in both profiles. Thus the gain cannot
be attributed to the new barrier. The control differs from the old reference
through nominal continuation training and lower actor learning rate, while both
comparison branches share those settings. Their individual causal contributions
were not separately measured. Retain the selected 10240-step checkpoint rather
than assuming the final 20000-step snapshot is best.

The selected translation registry now points to the control cruise actor plus
the unchanged near-goal actor and hysteresis protocol. The previous registry is
preserved in experiments/references/translation_before_geometry.json. The actor
bundle hash and narrow validation scope are recorded in the current registry.
Weights remain fixed at inference; local routing is mission state, not online
learning. This is a composition of two MAPPO actors, not a jointly trained
end-to-end hierarchical policy.

Verification: all 47 project checks passed after the training-code changes.
The selected actor bundle loads, produces finite local actions and has complete
recorded replay assets. To view the successful good-communication seed 61000000:

```bash
cd ~/swarm_robotics
bash scripts/view_translation.sh
```

The replay contains the actual simulated stop and a labelled display-only
final-frame pause. No desktop application was controlled to prepare it.
