# Speed 0.33 component selection

The requested campaign completed 200,000 environment steps per communication
profile (800,000 total) across all six experts. The fully adapted model was
retained but not promoted because nominal/good completion regressed. See
speed033_training_results.md for that negative result; it must not be hidden.

A subsequent DEVELOPMENT selection retains trained history-formation (90,000
steps/profile) and alignment (10,000 steps/profile), with the previous fresh
formation, cruise, degraded cruise and approach experts. No weights were
averaged. No further training was performed. The selected artifact did not
receive 200,000 steps/profile in each individual expert: that was the total
campaign allocation across six experts, as documented in speed033_training.md.

| Sample | Profile | Speed-0.33 baseline | Selected components | Selected minimum circular clearance (m) |
| --- | --- | --- | --- | --- |
| evaluation | nominal | 3/4 | 3/4 | 0.120 |
| evaluation | good | 3/4 | 3/4 | 0.120 |
| evaluation | moderate | 3/4 | 4/4 | 0.119 |
| evaluation | severe | 3/4 | 3/4 | 0.164 |
| confirmation | nominal | 2/2 | 1/2 | 0.095 |
| confirmation | good | 2/2 | 1/2 | 0.095 |
| confirmation | moderate | 2/2 | 2/2 | 0.101 |
| confirmation | severe | 2/2 | 1/2 | 0.167 |

Promoted: False. The first sample has four starts/profile (91033000..91033003); confirmation has two additional starts/profile (92033000..92033001). These samples do not establish a general success percentage. All outcomes, including failures, remain in the campaign traces.

The user-discovered severe seed 3624788268 was a TRAINING case, not held-out evidence. In live Gazebo, the selected components completed it at 248.20000000000002 s and remained within terminal tolerances through 480 s; the audit found 0 hold violations. The baseline at the same new speed had timed out at 420 s. Additional seed-42 nominal/good Gazebo results are in component_selection_summary.json.

The promotion gate required no lower per-profile completion in both samples, no observed sampled collisions, at least 0.05 m sampled circular clearance, continuous qualifying hold in successful local-core episodes, and all three Gazebo probes completing. This is not formal collision avoidance or the proposed 100 episodes/profile acceptance benchmark. Hardware command adapters remain absent.

Current artifact: experiments/runs/speed033_campaign_001/formation_adapted_bundle.pt; SHA-256 935eff1e7e485c2ab7d9cada95a0f84b876662c3404414ab7824bf5de94fa9e6. The full six-expert trained artifact and original speed-0.33 baseline remain available separately for reproducibility. No online learning or end-to-end learned phase coordination was introduced.
