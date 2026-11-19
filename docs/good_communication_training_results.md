# Nominal versus good communication with settling-compatible exploration

One paired training seed, identical initial actor, 50000 steps per model.
Action standard deviation decreases from 0.1 to 0.02 over 20000 steps.
Twenty common evaluation starts per condition, separate from training.

| Train communication | Evaluate communication | Success | Collisions | Mean RMSE (m) |
|---|---|---:|---:|---:|
| nominal | nominal | 10.0% | 0 | 0.1198 |
| nominal | good | 5.0% | 0 | 0.1197 |
| good | nominal | 15.0% | 0 | 0.1209 |
| good | good | 15.0% | 0 | 0.1208 |

Stationary nominal reference: 0.0% success.
This is a single-seed development test, not a robust-performance claim.
The formation task excludes five-meter translation and hardware transfer.
