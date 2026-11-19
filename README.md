# AI-Based Cooperative and Decentralized Control for Robot Swarms Under Communication Constraints

Project developed during my internship in September
2026 at JKK (Járműipari Kutatóközpont, Vehicle Industry Research Center),
investigating AI-based cooperative and decentralized control methods for robot
swarms. The work focuses on developing and evaluating control strategies under
practical inter-robot communication constraints, including limited bandwidth,
latency, packet loss, communication range and changing network topology.
The project explores approaches such as multi-agent reinforcement learning and evaluates them in simulation. Deployment packages have been prepared for supervised trials, but the system has yet to be tested on physical robots.

The mixed-fleet mission starts three robots from dispersed planar poses, forms
an aligned triangle with 1.20 m between robot centres, translates it 5 meters in the
mission direction and stops with the same heading and spacing. Gazebo runs the
controllers in closed loop with simulated odometry and physics. The models
represent a Go2 Edu, a Jackal J100 and a provisional Ackermann WheelTec RosBot Pro.

The selected mixed-fleet simulation uses four frozen learned components
(Go2 pose, WheelTec arc, Jackal pose and translation speed). A programmed
controller plans feasible arcs, coordinates phase transitions, applies
clearance constraints and corrects the final pose. The mixed-fleet campaign
refines those components with teacher-labelled planar rollouts.

## Run the mixed-fleet simulation

Follow the [simulation setup](docs/deployment/SIMULATION_SETUP.md), then run one
profile at a time from the repository root.

```bash
bash scripts/run_mixed_realtime.sh nominal
bash scripts/run_mixed_realtime.sh good
bash scripts/run_mixed_realtime.sh moderate
bash scripts/run_mixed_realtime.sh severe
```

Each launch samples new starting poses. Add a numeric seed to reproduce them.
The launcher verifies the selected actor hashes and sets a wall-clock pacing
target of 0.98. Gazebo can run below that target when the physics, bridge or
rendering cannot keep up. Check the terminal RTF and the recorded
`target_real_time_factor` and `measured_real_time_factor` in `summary.json`.
The communication profiles apply simulated communication impairments.

See [GETTING_STARTED.md](GETTING_STARTED.md) for setup, logs, reproducible
commands and training. See [implementation status](docs/implementation_status.md)
for current evidence and [random-start results](experiments/results/final_random_acceptance.json)
for the finite set of verified runs.

## Repository map

| Path | Contents |
| --- | --- |
| `scripts/gazebo_live.py` | Live Gazebo integration and mixed mission |
| `scripts/train_mixed_actor_campaign.py` | Mixed-fleet planar training |
| `src/swarm_core/` | Geometry, dynamics, communication and runtime |
| `configs/` | Task, platform and communication settings |
| `policies/`, `experiments/` | Frozen actors, provenance and experiment records |
| `ros2_ws/` | ROS 2 packages |
| `docs/` | Requirements, results and deployment notes |


## Laboratory installation

- [Shadow test package](deliverables/swarm_mixed_distributed_shadow.tar.gz) and
  [lab checklist](docs/deployment/LAB_TRIAL.md) for odometry and network tests
  with drive outputs disconnected.
- [Supervised motion trial package](deliverables/swarm_mixed_motion_trial.tar.gz)
  and [motion setup](docs/deployment/MOTION_TRIAL.md) for drivers that accept
  ROS 2 Twist and have a verified watchdog and independent stop.

The motion package is the execution build for supervised trials. Its command
gate has software checks. End-to-end motion on physical hardware remains
unverified. The Jackal is available in simulation only for this project.
The complete group mission requires three participants.
