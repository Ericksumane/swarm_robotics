# Controlled reward and gradient diagnosis

## Measured reward ranking

Three robots start in an equilateral triangle with 1.2 m sides, facing its
center. With explicit diagnostic inputs, inward movement returned +3.7183,
stationary input returned -0.1200, and outward movement returned -2.1087.
Final inward RMSE was 0.01294 m versus 0.20 m when stationary.
Thus these tests do not support the claim that the reward always prefers
immobility. They do not establish a globally optimal or safe reward.

## PPO direction

Reusing one sampled batch improved one heading group and harmed the other.
With fresh on-policy batches and a state-conditioned leave-one-out baseline,
both controlled heading groups acquired the beneficial action direction.
The temporary policy and its diagnostic state fixtures were not exported,
used as demonstrations or deployed. This is a one-step action-credit test,
not learned formation or evidence that the production critic is correct.
It narrows the next hypotheses to representation, value fitting, exploration
and temporal credit assignment; no unique root cause is established.

## Candidate representation correction

Contract formation_geometry_critic_v3 keeps the geometry actor inputs and
uses body-frame relative positions, relative heading sine/cosine, normalized
velocities, pair distances, readiness time and formation error for the critic.
It removes dependence on arbitrary global translation and orientation.
For three robots, critic width is 23. Critic state remains training-only.
The previous contract and checkpoints remain valid under their original schema.
This is a tested representation change, not a claim of improved task success.

Reproduce: scripts/diagnose_formation_learning.py and
scripts/check_relative_critic.py. Raw diagnostic data is stored under
experiments/runs/reward_gradient_diagnostic.json.
