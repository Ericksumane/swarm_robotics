# Paired critic comparison

Three training seeds; 20000 steps per model; identical initial actor per pair.
Same reward, actor input, sampling seed and evaluation starts. Final checkpoints only.

| Critic | Mean success rate | Total collisions | Mean RMSE (m) |
|---|---:|---:|---:|
| global | 6.7% | 0 | 0.1438 |
| relative | 15.0% | 0 | 0.1389 |

Stationary reference: 0.0% success; mean RMSE 0.1226 m.
Each model is tested on the same 20 initial states. These are not 60 independent starts.
Three training seeds are a small development comparison, not a reliability guarantee.
This experiment tests formation acquisition only, not five-meter translation or hardware.
