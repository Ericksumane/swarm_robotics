# Requirements

## Confirmed

| ID | Requirement |
|---|---|
| R01 | Use MAPPO based on the official marlbenchmark/on-policy repository. |
| R02 | Start with three robots. |
| R03 | Execute actors locally without a designated leader. |
| R04 | Support logical communication between every pair of robots. |
| R05 | Train offline and deploy fixed policy weights. |
| R06 | Evaluate formation control under degraded communication. |
| R07 | Use ROS 2 for simulation and robot integration. |
| R08 | Use English throughout the project. |
| R09 | Document mathematical definitions and their implementation. |
| R10 | Design interfaces for variable team sizes; validate supported sizes experimentally. |
| R11 | Use 1.0 m as the intended initial formation distance. |

## Initial design decisions

- Shared actor parameters, conditioned on robot capabilities.
- Independent local inference on each robot.
- Centralized critic during training only.
- Variable-size neighbor observations.
- Versioned policies and deployment configurations.
- No online learning during deployment.

## Pending specifications

- Whether formation distance is measured between robot centers or boundaries.
- Exact formation geometry and assignment of formation positions.
- Collective motion objective and mission-start protocol.
- Available relative sensing and localization.
- Observation and action definitions.
- Robot command interfaces and operating limits.
- Communication message contents and transmission rates.
- Communication impairment distributions and evaluation ranges.
- Reward, episode termination, and acceptance criteria.

## Intended laboratory platforms

- Unitree Go2.
- Clearpath Jackal.
- Wheeltec RosBot Pro.

Exact hardware versions and interfaces have not yet been verified.

## Evidence boundary

Simulation results do not establish hardware transfer.
Support for variable team sizes does not establish performance for arbitrary
numbers or types of robots.
