# Focused training conclusion

## Decision

Keep the existing registered research candidate. The additional 120,000-step
experiment is retained as an experimental checkpoint, not a global improvement.
The existing candidate registry is unchanged.

## Matched independent evaluation

Both models used the same new initial-state seeds and 120-second episode limit.
Near-formation evaluations used 40 episodes per communication profile;
the separate broad-position diagnostic used 20 episodes with good communication.

| Model and evaluation | Successes | Collisions | Mean trajectory formation RMSE (m) |
|---|---:|---:|---:|
| baseline_nominal_heldout | 39/40 | 0 | 0.0959 |
| trained_nominal_heldout | 39/40 | 0 | 0.0800 |
| baseline_good_heldout | 40/40 | 0 | 0.0943 |
| trained_good_heldout | 40/40 | 0 | 0.0805 |
| baseline_broad_good | 10/20 | 1 | 0.3725 |
| trained_broad_good | 9/20 | 0 | 0.4633 |

The earlier 87.5% result used a different sample. On this fresh sample the old
model also achieved 40/40 with good communication. Therefore 87.5% to 100%
must not be interpreted as a measured training improvement on matched cases.
Near-formation trajectory error improved, while broad-position success declined
from 10/20 to 9/20. Broad-position collisions declined from one to zero.
These small samples and one training seed do not establish reliable superiority.

## Historical regression audit

The new checkpoint solved all five previously inspected failed seeds.
These are historical regression cases, not an untouched final test set.
They were not reused as training seeds. Training templates came from separate
mining pools and were perturbed; no expert actions were supplied.

## Verification and artifacts

- All 34 project checks passed before the campaign.
- Training completed 120,000 additional environment steps.
- Training, selection, final near-position tests and broad-position tests used
  disjoint seed ranges; see docs/hard_focus_protocol.md.
- Full comparison: docs/hard_focus_results.md.
- Machine-readable results: experiments/runs/hard_focus_001/summary.json.
- Historical audit: experiments/runs/hard_focus_001/old_failure_regression.json.
- Experimental checkpoint: experiments/runs/hard_focus_001/checkpoint_0120000.

## Next development priority

Investigate the eleven broad-position failures before another long run.
All eleven were classified as never_formed, so compare initial geometry,
headings, pair errors over time and action saturation to identify where progress
stalls. Use new development cases for design changes and reserve fresh final
tests. Preserve near-formation performance as a regression constraint.
Do not relax success tolerances or extend time solely to inflate success rates.

## Scope

This evaluates forming and holding a 1 m equilateral triangle, with ideal
localization in a common frame. It does not validate the full five-meter mission,
physical Gazebo closed-loop dynamics, real localization or hardware deployment.
