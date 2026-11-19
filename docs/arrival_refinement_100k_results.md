# Focused arrival training: results

Training completed exactly 100,000 three-robot environment steps: 25,000 each in
nominal/good/moderate/severe. Only the approach expert changed neural weights;
the other five experts were verified byte-for-byte unchanged. Limits remain
0.33/0.165 m/s. Deployment routing, supervisors and mission success criteria were
unchanged. See arrival_refinement_100k.md for the training-only design changes.

| Profile | Baseline | Candidate | Candidate collisions | Every baseline success preserved |
| --- | --- | --- | --- | --- |
| nominal | 4/4 | 4/4 | 0 | True |
| good | 4/4 | 4/4 | 0 | True |
| moderate | 4/4 | 4/4 | 0 | True |
| severe | 4/4 | 4/4 | 0 | True |

Promoted: True. These are four new starts/profile, seeds 93044000..93044003; a small paired sample, not a general success percentage or the proposed final benchmark. Successful candidate runs were audited for 60 seconds of continuous endpoint, speed and formation tolerance.

Live Gazebo probes (known development/training cases, not held out):
- nominal_known: completed=True, first completion=164.8 s, hold violations=0.
- good42: completed=True, first completion=124.60000000000001 s, hold violations=0.
- severe_known: completed=True, first completion=128.20000000000002 s, hold violations=0.

Known nominal baseline: completed=False, first completion=None s. New candidate artifact: experiments/runs/arrival_refinement_100k_001/training/candidate_bundle.pt; SHA-256 ec542ed24149fe46f77a0bfeaff162ccc043acf7f736a72dcd60dff7f985911b. It is retained whether or not promoted. Promotion required every baseline-success case to remain successful, no observed sampled collisions, at least 0.05 m sampled circular clearance, continuous hold, all three physical-simulation probes succeeding and measured improvement. No physical robot adapter is enabled.

Final implementation checks: all 70 project checks passed. Independent local versus batched inference passed with maximum action difference approximately 1.79e-7. Training took approximately 7.3 minutes; validation time is separate. Verification confirmed the task, routing, stop supervisors and other five experts unchanged.
