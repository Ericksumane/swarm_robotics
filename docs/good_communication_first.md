# Priority: task competence before communication robustness

User direction: first make the requested behavior work with good communication;
only then introduce communication impairments. Do not advance robustness training
while formation acquisition is unreliable.

Use the zero-delay, zero-loss nominal profile as the diagnostic reference.
Then verify the existing good profile, which includes small synthetic delay/loss
and occasional burst loss. The good profile is not a perfect-channel baseline.
Moderate/severe communication experiments are deferred.

Required progression:
1. Reliable one-meter formation and stationary hold across varied headings and
   positions, with explicit per-stage failure diagnostics and separate test seeds.
2. Formation followed by the full five-meter translation and stop under nominal
   and good communication. Formation-only success does not satisfy this gate.
3. Only after task competence is demonstrated, evaluate increasing impairments.

Current evidence: selected relational actor 11/40 nominal versus 10/40 good;
the one-episode difference does not support blaming the network for the main
failure. Nominal/good failures were primarily time limits. The 20480-step
checkpoint exceeded later checkpoints on the selection set: extending the
unchanged training run is not supported as the sole remedy.

Next training design must address the curriculum jump from 1.2 radians to pi,
retain previously learned reset distributions, and measure geometric readiness
separately from low-speed readiness and hold duration. The latter diagnostics
are required to distinguish failure to assemble from failure to settle.
Reward/critic credit assignment and late-training forgetting remain hypotheses
requiring controlled comparisons; passing unit tests does not establish learning.
