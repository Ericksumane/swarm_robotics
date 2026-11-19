# Formation clearance: 100,000-step evaluation

The experiment completed exactly 100,000 environment steps: 25,000 per communication profile. Training took 864.8 seconds. Only the history-formation expert changed; the other five experts and deployment parameters were preserved.

Decision: **not promoted**. The selected arrival-refined baseline remains active.

## Paired new starts

| Communication | Baseline successes | Candidate successes | Candidate minimum clearance (m) | No clearance regression beyond 1 cm |
| --- | --- | --- | --- | --- |
| nominal | 4/4 | 4/4 | 0.0794 | False |
| good | 4/4 | 4/4 | 0.0793 | False |
| moderate | 4/4 | 4/4 | 0.0822 | False |
| severe | 3/4 | 3/4 | 0.2044 | False |

Both models completed 15/16 new-start missions. The same severe seed failed with peer_timeout. Successful candidate missions retained 60 seconds of qualifying hold. These 16 cases are a small paired experiment, not a general success rate.

## Known Gazebo development cases

| Case | Completed | Minimum circular clearance (m) | Final maximum pair error (m) |
| --- | --- | --- | --- |
| nominal_close | False | 0.1401 | 0.104303 |
| good_close | True | 0.1225 | 0.086178 |
| arrival_regression | False | 0.3147 | 0.102644 |

The known good/nominal close-approach clearances increased from 0.0984/0.1120 m to approximately 0.1225/0.1401 m. Neither reached the predefined 0.15 m minimum gate. The nominal close case and the previous arrival-regression case stopped with pair errors just above the unchanged 0.10 m nominal tolerance; their missions timed out. Individual destination proximity does not imply team geometry success. Increasing the timeout cannot repair a parked configuration with zero commands.

This reveals coupling between formation geometry at MOVE start and final per-robot parking. Merely extending this same training is not justified by the results. A subsequent design experiment should explicitly test formation-to-arrival compatibility and parking recovery without relaxing mission criteria. This experiment does not establish that proposed change will work.

## Verification

- All 71 project checks passed.
- Independent local inference matched batched inference within 1.79e-7.
- Exactly one of six neural experts changed.
- No deployment tolerance, routing, speed or predictive veto change.
- No selected model or hardware approval change.

Clearance uses circular footprints sampled at 0.1 s; it is not a continuous physical contact measurement. Known Gazebo probes were development/training cases. No hardware readiness claim is made.

Protocol and equations: formation_clearance_100k.md. Machine-readable results: experiments/runs/formation_clearance_100k_001/summary.json. Candidate retained for research, not selected: e6895acebdc609fa12be2d37a42789c4701e867d8b509264ceabe870eae74087.
