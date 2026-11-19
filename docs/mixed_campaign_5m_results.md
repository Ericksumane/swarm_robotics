# Five-million-step campaign and moderate regression

Ten 500k refinements completed. Each block started from the retained reference.
No block satisfied the non-regression selection criterion. The frozen reference
manifest is experiments/runs/mixed_campaign_5m_001/retained_manifest.json.

| Block | Moderate formation | Moderate arrival | Severe formation | Severe arrival |
| --- | ---: | ---: | ---: | ---: |
| 1 | 9/16 | 15/16 | 16/16 | 11/16 |
| 2 | 6/16 | 15/16 | 16/16 | 12/16 |
| 3 | 9/16 | 14/16 | 16/16 | 11/16 |
| 4 | 12/16 | 13/16 | 16/16 | 13/16 |
| 5 | 14/16 | 13/16 | 16/16 | 13/16 |
| 6 | 9/16 | 12/16 | 16/16 | 12/16 |
| 7 | 16/16 | 13/16 | 16/16 | 13/16 |
| 8 | 16/16 | 14/16 | 16/16 | 12/16 |
| 9 | 8/16 | 16/16 | 16/16 | 14/16 |
| 10 | 6/16 | 14/16 | 16/16 | 12/16 |

The reference scores are moderate 15/16 formation and 15/16 arrival,
severe 16/16 formation and 12/16 arrival. Nominal and good retained 16/16
in both phases for all ten candidates.

## Actor replacement experiment

On the original 16 moderate formation seeds, the final candidate scored 6/16.
Replacing only its Jackal actor with the reference recovered 15/16.
Restoring Go2 alone or WheelTec alone retained 6/16.
On 32 new seeds, reference and restored-Jackal combinations scored 27/32
while the complete final candidate scored 19/32.

This isolates a Jackal actor regression in these planar tests. It does not
validate a new mixed artifact in complete Gazebo missions. The reference
remains selected. Per-seed data and final errors are in
experiments/runs/moderate_regression_5m_001/results.json and fresh_seeds.json.
