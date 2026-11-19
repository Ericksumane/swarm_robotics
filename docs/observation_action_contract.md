# Observation and Action Contract

## Status

Initial neighbor-observation implementation.
The complete actor input and normalization are not finalized.

Implementation: src/swarm_core/observations.py

## Information boundary

Each actor may use:
- Its own locally available state estimate.
- Mission information explicitly available to that robot.
- Messages accepted by its local receiver.

Actors must not access other robots' true simulator states.

The current simulation uses ideal own-state estimates.
Measurement noise and drift remain to be implemented.

## Neighbor features

Each neighbor row contains seven features, in this exact order:

| Index | Feature | Unit |
|---|---|---|
| 0 | Relative forward position | m |
| 1 | Relative lateral position, positive left | m |
| 2 | Sine of relative heading | dimensionless |
| 3 | Cosine of relative heading | dimensionless |
| 4 | Neighbor longitudinal speed | m/s |
| 5 | Neighbor yaw rate | rad/s |
| 6 | Information age divided by maximum age | dimensionless |

Neighbor speed is the sender's longitudinal speed, not relative speed.

## Relative coordinates

Let dx = received_neighbor_x - own_x.
Let dy = received_neighbor_y - own_y.
Let theta be the receiver's current estimated heading.

relative_forward = cos(theta)*dx + sin(theta)*dy

relative_left = -sin(theta)*dx + cos(theta)*dy

The received neighbor position may be old.
The current implementation does not predict it forward in time.

All exchanged positions must use a common coordinate frame.

## Missing and expired information

A separate mask identifies valid rows:
- 1: received information is available and not expired.
- 0: information is missing or expired.

Invalid rows contain zeros.
Zero-valued features must not be interpreted without the mask.

Expired cached states are not exposed in the feature table.
The current actor input therefore does not distinguish missing from expired
information. Recurrent memory may retain earlier valid observations.

## Neighbor ordering and team size

Rows are ordered by sender ID.
IDs are metadata and are not included among the seven features.

The table supports a variable number of neighbors.
A permutation-invariant neural aggregation remains to be implemented.
Variable-size tables alone do not establish variable-team-size policy support.

## Actions

The initial motion model accepts:
- Normalized longitudinal velocity request.
- Normalized yaw-rate request.

The dynamics module clips requests to [-1, 1], scales them using robot limits,
and applies acceleration constraints.

Upstream MAPPO Gaussian samples are not inherently bounded by the Box space.
The rollout buffer must retain the sampled action and its matching log
probability. The environment applies the documented command transformation.

See docs/dynamics.md for motion equations.

## Versioning

Any change to feature order, meaning, units, masking, normalization,
or action transformation requires an explicit contract version.
Deployment policies must identify their compatible contract.

## Learned neighbor aggregation

Implementation:
src/swarm_core/learning/neighbor_encoder.py

Each neighbor feature vector f_j is processed by the same neural encoder:

phi(f_j) = tanh(W2 * tanh(W1 * f_j + b1) + b2)

The embedding dimension is initially 32.

Let m_j be the binary validity mask and n_valid = sum_j m_j.

The pooled embedding is:

z = sum_j [m_j * phi(f_j)] / max(1, n_valid)

Two additional features are appended:
- Valid-neighbor fraction: n_valid / max(1, n_rows).
- Availability indicator: 1 when n_valid > 0, otherwise 0.

The resulting vector has 34 components.

Invalid feature rows are zeroed before encoding.
Their embeddings are masked before pooling.
No valid neighbors produces an all-zero output.

## Properties and limitations

The output is invariant to a joint permutation of rows and masks.
The encoder accepts a variable number of neighbor rows.

The valid-neighbor fraction assumes that every known team neighbor
has a row, including neighbors with missing messages. Additional batch
padding must not be counted as team members.

Mean pooling does not preserve every detail of the neighbor set.
This representation must be evaluated on formation-control tasks.

The current encoder does not explicitly expose the total team size.
Support for additional team sizes requires training and evaluation.

## MAPPO integration requirement

This encoder is a project-specific actor extension.
It is not yet connected to the upstream MAPPO actor.

Its parameters must belong to the actor optimizer.
Rollouts must retain raw neighbor features and masks so embeddings
are recomputed during PPO updates.

Do not store detached embeddings as a replacement for raw observations
when training the encoder jointly.

## Verification

- Permutation invariance: passed.
- Invalid-row exclusion: passed.
- Empty and fully missing neighbor sets: passed.
- Finite, nonzero encoder gradients: passed.

## Experimental history formation and stationary alignment

formation_history_v5 has eleven peer features: the previous nine relational
features, plus binary freshness and normalized position-error bound. Its mask
means an estimate is available, not that a packet is fresh. History is separate
from the original translation observations. Contract formation_history_risk_v6
adds closing speed and constant-velocity predicted clearance; it belongs to an
unfinished alternative experiment and is not in the released candidate.
For relative position p and relative velocity v derived only from own state and
received payload, closing speed is -dot(p,v)/max(norm(p),epsilon). Closest time
is clip(-dot(p,v)/max(dot(v,v),epsilon),0,H), and clearance is
norm(p+t*v)-diameter. These features do not guarantee safe trajectories.

The explicit stationary_alignment artifact flag constrains ALIGN linear command
to zero during training and inference. Yaw remains policy-controlled; physical
acceleration/deceleration limits still apply. It is a stage action constraint,
not an assertion that all motion decisions are a single end-to-end learned
controller. See history_coordination_results.md for the selected six-expert
architecture, routing and evaluation scope.
