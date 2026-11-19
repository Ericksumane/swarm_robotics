# Technical Decisions

## D01 — Official MAPPO implementation

Use marlbenchmark/on-policy as the algorithm source.
Record the exact upstream commit before integration.
Document every modification to the upstream implementation.

## D02 — Offline training and local inference

Train on the development computer.
Deploy fixed actor weights to each robot.
The training computer is not required to calculate deployment actions.

## D03 — Initial parameter sharing

Start with a shared policy conditioned on robot capabilities.
Parameter sharing does not imply centralized execution.
Its suitability for the heterogeneous team must be evaluated.

## D04 — Leaderless execution

No robot is designated as the formation leader.
Mission distribution, formation assignment, and synchronized start behavior
must be specified separately.

## D05 — Communication topology

The initial logical communication graph connects all robot pairs.
A shared Wi-Fi access point does not constitute a physical wireless mesh.
Communication failures may be directed and asymmetric.

## D06 — Shared training and deployment contracts

Use the same feature definitions, units, normalization, and action
interpretation during training and deployment.
Do not expose simulator ground truth to actors unless an equivalent
measurement is available during deployment.

## D07 — Variable team size

Design neighbor processing for a variable number of agents.
Initially validate three agents.
Additional team sizes and membership changes require explicit evaluation.

## D08 — Mathematical traceability

Define symbols, assumptions, equations, and references in
docs/mathematical_formulation.md.

Document the Dec-POMDP, return, actor and critic inputs, GAE,
PPO clipped objective, value loss, entropy term, task reward,
and communication model.

Link implemented equations to source modules.
Distinguish upstream MAPPO equations from project-specific extensions.

## D09 — Formation distance

The intended initial formation distance is 1.0 m.
The measurement reference and physical feasibility remain to be confirmed.

## D10 — Preserve previous work

Keep the previous project as a reference.
Reuse components only after reviewing their assumptions and compatibility.
