# Deep code revision and training evidence

Two fresh MAPPO training runs, 120000 environment steps each, with perfect communication.
Both completed. Selection and final evaluation used disjoint seeds.
The two experiments used different final seeds; their percentages are not a paired comparison.
The previous model is evaluated on identical final seeds within each experiment.

| Second experiment evaluation | Success | Collisions | Mean RMSE (m) |
|---|---:|---:|---:|
| previous_nominal_heldout | 2/40 | 0 | 0.1381 |
| revised_nominal_heldout | 11/40 | 0 | 0.1139 |
| revised_good_heldout | 10/40 | 0 | 0.1137 |
| revised_easy_diagnostic | 16/20 | 0 | 0.0953 |

Broad-position diagnostic: 0/10 successes, 1 collisions.
Paired nominal outcomes: new-only successes=10, previous-only successes=1.
Selected research artifact: ~/swarm_robotics/experiments/runs/relational_revision_001/checkpoint_0020480
The last training checkpoint was not automatically selected; late training degraded performance.
One training seed and these development experiments do not establish reliable improvement.
All models remain research artifacts; no hardware or ROS actuator deployment was performed.
The five-meter mission is not learned or validated by these formation-only tests.

## Corrected code paths

- Mastery-based initialization curriculum with logged progression and unchanged success thresholds.
- Worst-pair progress and dense qualifying-hold reward, versioned separately from legacy rewards.
- Versioned observable relational actor summaries with masking and permutation invariance.
- Zero-neighbor actor reshape now preserves the explicit batch size.
- Evaluation overrides verify task contract and robot count.
- Task-qualified trajectory filenames prevent evidence overwriting across task variants.
- Actor-only export preserves architecture metadata; old artifacts remain compatible.

## Remaining learning risks

The jump from 1.2-radian heading perturbations to uniform headings is large; mastery on the
previous distribution did not prevent regression on the new distribution. The current
experiments do not justify extending duration alone. Intermediate heading stages and
retaining previous reset distributions are further hypotheses, not validated fixes.
See docs/deep_training_revision.md and docs/relational_actor.md for equations and contracts.
