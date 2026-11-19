# Speed 0.33 training results

All six MAPPO experts were adapted with 200,000 environment steps per
communication profile: nominal, good, moderate and severe. Total 800,000
three-robot environment steps (2,400,000 robot transitions). Smoke and evaluation
steps are excluded. Training used the plan and source hashes in the campaign.

The complete trained candidate was frozen before four new initializations per
profile (seed 91033000..91033003). Baseline and trained models use the same
0.33/0.165 m/s standard. Tests execute three independent local runtime cores
with synthetic impaired messages and a Python physical plant, including a
60-second observation after success. They are not Gazebo or hardware trials.

| Profile | Baseline success | Trained success | Trained collisions | Trained minimum circular clearance (m) |
| --- | --- | --- | --- | --- |
| nominal | 3/4 | 1/4 | 0 | 0.120 |
| good | 3/4 | 1/4 | 0 | 0.120 |
| moderate | 3/4 | 4/4 | 0 | 0.119 |
| severe | 3/4 | 3/4 | 0 | 0.164 |

Promoted into the current live registry: False.
Training completion is not proof of improvement. The gate requires no sampled collisions, no lower per-profile completion, at least 0.05 m sampled clearance, and either improved completion or all 16 successes. This is a small development gate, not the proposed 100 episodes per profile acceptance benchmark.

The trained candidate remains at experiments/runs/speed033_campaign_001/training/candidate_bundle.pt, SHA-256 4977485bb4abe4d696b0564a7f7df85fcb4432b938cff71bf1da59ea7ada0bdb. Previous weights and the new-speed baseline are preserved. Routing, phase coordination and stop supervisors remain programmed; the campaign did not learn an end-to-end coordinator. Hardware approval remains false.


## Final operating decision

All six experts changed neural weights (excluding merely changing exploration
standard deviations). Training took approximately 27.9 minutes. The full trained
candidate completed 9/16 paired evaluation missions versus 12/16 for the
speed-0.33 baseline. It was not promoted.

A component selection completed 13/16 development missions, but only 5/8 new
confirmation missions versus the baseline's 8/8, and its good seed-42 Gazebo
probe failed. It was also not promoted. See speed033_component_selection.md.
The known difficult severe Gazebo case improved to completion at 248.2 s under
that component selection; because it was a training case, this is a regression
repair example, not proof of generalization.

The live default therefore retains the previous neural weights with the newly
requested 0.33/0.165 m/s profile. Its good seed-42 Gazebo recheck completed at
125.3 s. This single recheck does not validate all starts or hardware transfer.
The historical runtime archive remains the earlier frozen installation package.
The full 800,000-step candidate and component selection remain available for
inspection; neither is silently presented as the selected live policy.

Final code validation: all 69 project checks passed. CPU inference of the retained default also passed. Passing code checks did not override the negative policy-performance evaluation.

Subsequent update: focused arrival-only adaptation was selected after separate validation. See arrival_refinement_100k_results.md for the current live candidate; this report remains the historical six-expert campaign result.
