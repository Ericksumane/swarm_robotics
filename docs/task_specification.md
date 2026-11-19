# Initial Formation-Control Task

## Status

Initial implementation specification.
Numerical tolerances and robot limits require validation before training.

## Objective

Three robots autonomously establish a triangular formation, translate
the formation by 5.0 m along a mission-defined direction, and stop.

There is no designated leader.
Each robot executes a local copy of a shared MAPPO actor.
Policy weights remain fixed during deployment.

## Formation geometry

The desired distance between every pair of robot centers is 1.0 m.
For three robots, this defines an equilateral triangle.

The initial task constrains pairwise distances, not triangle orientation.
The formation must remain physically feasible for the robot footprints.

## Mission phases

1. FORM: establish and maintain the desired pairwise distances.
2. MOVE: translate the formation by 5.0 m.
3. HOLD: stop and maintain the formation at the destination.

Phase transitions require explicit readiness and completion rules.
Deployment phase coordination must not depend on a designated leader.
The distributed transition protocol remains to be specified and tested
under delayed or missing messages.

## Displacement reference

At the start of MOVE, freeze the formation reference position.
For evaluation, use the centroid of the three robot centers.

The destination reference is 5.0 m from that position along a unit
direction supplied by the mission.

Do not move the destination when robots deviate from the intended path.
Individual path lengths may differ from the formation displacement.

The evaluator may use true simulator positions.
Actors must use only their permitted observations.
Any mission reference required by actors must have a deployment-compatible
construction and distribution mechanism.

## Initial conditions

Start with three stationary robots at distinct, collision-free positions.
Use a bounded initial-position distribution.
Initial conditions must permit formation within the operating area.

Begin with no obstacles and nominal communication.
Introduce communication impairments after basic environment validation.

## Motion interface

Each actor produces two continuous action components:
- Longitudinal velocity request.
- Yaw-rate request.

Define action normalization, scaling, and bounds before training.
Apply platform-specific speed, acceleration, and yaw-rate limits.
Use identical action interpretation during training and deployment.

The Go2 low-level locomotion controller remains responsible for walking.
The policy does not command joint torques.

## Localization interface

Each robot provides a timestamped local pose and velocity estimate.
Initial experiments assume known alignment to a common planar frame.

Simulation must distinguish:
- True physical state.
- Locally measured or estimated state.
- Neighbor information delivered through communication.

Never replace missing neighbor messages with hidden simulator state.
Noise and drift parameters must be configurable.

Hardware odometry availability, frame alignment, and accuracy remain
to be verified on each laboratory robot.

## Communication

Use directed logical links between every pair of robots.
Each robot sends its locally available state information.

Messages include sender identity, sequence number, generation timestamp,
and payload size.

The channel supports configurable delay, loss, burst loss, outages,
and bandwidth constraints.

Receivers track message age and validity.
Message ordering, expiration, and queue policies must be specified.

Shared Wi-Fi infrastructure is not a physical wireless mesh.

## Reward requirements

Use a shared cooperative reward that considers:
- Formation error.
- Mission progress.
- Unsafe proximity and collisions.
- Command variation.
- Completion time.
- Successful completion.

Normalize terms and document units, equations, and weights.
Do not penalize externally imposed packet loss directly.
Reward penalties are not safety guarantees.

## Episode outcomes

Success requires reaching the destination, satisfying formation
tolerances, and remaining below a velocity threshold for a hold interval.

Failure conditions include collision and operating-area violation.
A time limit ends an unfinished episode as a truncation.

Tolerance values, hold duration, operating area, and time limit remain
to be specified before training.

## Initial evaluation metrics

- Success rate.
- Pairwise formation error.
- Minimum physical clearance.
- Formation displacement and destination error.
- Completion time.
- Applied command variation.
- Message delivery statistics and information age.
- Recovery after communication degradation.

## Scope boundaries

The first environment uses simplified planar robot dynamics.
Initial validation covers three agents only.

Variable team sizes, automatic initial-frame alignment, obstacles,
and hardware transfer require subsequent implementation and evaluation.
