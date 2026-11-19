# Mixed-fleet arrival campaign

The campaign completed 1,000,000 additional planar environment steps from the
160k reference. Each profile received 250,000 steps. The trainer used
teacher-labelled rollouts with terminal recovery active and saved 125
checkpoints. The final four actors and SHA-256 values are recorded in
`experiments/mixed_arrival_1m_experimental.json`.

The comparison used 16 reserved seeds per profile per phase, identical for
both checkpoints. Both ran with terminal pose recovery enabled. Formation
and translation were evaluated separately. The latter starts close to the
aligned triangle and includes the final stop.

| Profile | Formation reference | Formation 1m | Translation and stop reference | Translation and stop 1m |
| --- | ---: | ---: | ---: | ---: |
| nominal | 16/16 | 16/16 | 16/16 | 16/16 |
| good | 16/16 | 16/16 | 16/16 | 16/16 |
| moderate | 15/16 | 9/16 | 15/16 | 16/16 |
| severe | 5/16 | 5/16 | 12/16 | 13/16 |

All unsuccessful trials in this comparison ended at the time limit. The
extra arrival success in each degraded profile did not compensate for the
six lost formation successes in moderate. The 160k manifest remains the
default for Gazebo and for the lab's shadow package.

The raw per-seed output is
`experiments/runs/mixed_160k_arrival_campaign_1m_001/paired_holdout.json`.
Use `scripts/evaluate_mixed_actor_campaign.py` with both checkpoint paths
to reproduce the planar comparison. The experimental weights can be inspected
with the launcher command in [quick start](../GETTING_STARTED.md).

The mixed simulator targets RTF 0.98. The WSL system measured lower rates.
The evidence below concerns the 160k reference and seed 196802. The final
million-step weights were evaluated on the planar holdout above.

| Profile | Simulated seconds | Measured RTF | Travel (m) | Run under experiments/runs |
| --- | ---: | ---: | ---: | --- |
| nominal | 75.3 | 0.322 | 4.97 | `gazebo_live_20261118_111406_5089/summary.json` |
| good | 107.4 | 0.302 | 5.00 | `gazebo_live_20261118_110029_4319/summary.json` |
| moderate | 68.2 | 0.479 | 5.00 | `gazebo_live_20261118_112509_393/summary.json` |
| severe | 111.0 | 0.324 | 5.02 | `gazebo_live_20261118_110720_4652/summary.json` |

All four recorded runs succeeded. The shorter bridge pause was tried in
moderate and produced a formation timeout on this seed. The launcher
retains the original pause for that profile. Wall-clock performance
varied with host load while the training job and checks were running.
