# Deployment entry points

For the mixed fleet onboard dry run, see [distributed shadow](MIXED_DISTRIBUTED_SHADOW.md).

The mixed-fleet JSON actors use the [mixed-fleet shadow package](MIXED_ACTOR_SHADOW.md).
It runs centrally on a Jazzy computer and reads three odometry streams in a
common mission frame. The sections below describe the separate generic MAPPO
runtime and its distributed ROS contract.

## Generic MAPPO runtime installation and operation

## Scope

This package runs the complete fixed-weight mission separately for each of three
robots and publishes SHADOW velocity commands. It contains no vendor actuator
adapter and cannot be called hardware-ready. Robot models/revisions, onboard
computers, OS/ROS, sensors, localization and stopping remain unverified.

The common installer is limited to Ubuntu 24.04 x86_64 with ROS 2 Jazzy, colcon,
ROS interface generators and uv already installed. Target ARM/Jetson or other
ROS distributions need their own verified platform instructions. The installer
does not modify the OS, install ROS or install robot SDKs. Internet access is
needed for Python packages. Prepare an architecture-specific offline wheelhouse
only after the actual onboard configuration is known.

## Fresh-directory installation

Copy the release archive to the target development machine and run:

```bash
tar -xzf swarm_runtime_release.tar.gz
cd swarm_runtime_release
sha256sum -c SHA256SUMS
bash scripts/install_runtime.sh
```

Use a freshly extracted directory: installation refuses to replace an existing
.venv. Python 3.10 and CPU PyTorch 2.5.1 are isolated from system Python. The
installer loads/checks the pinned artifact, compiles the typed ROS interfaces
with the ROS system Python and performs a shadow preflight. A package snapshot
is saved in requirements/runtime-installed.txt. It does not install a training
environment or use GPU/CUDA on the robot.

If installation fails, keep the error output. A successful import is not a
physical-trial approval. Do not bypass a failed platform or checksum check.

## Configuration and launch

Each robot needs a unique ID 0, 1 or 2. Use the same NEW mission identifier on
all three; do not reuse it after a node reboot or an aborted experiment. Clocks
must share a time basis. Each odometry stream must already describe pose in the
common `mission` frame, with twist in the configured robot body frame. There
is no automatic alignment of independent odometry origins.

On robot 0, from its installed release directory:

```bash
bash scripts/run_full_mission_shadow.sh 0 trial_001
```

On the other two installations use IDs 1 and 2 with the same `trial_001`. For a
new attempt use a different mission ID on all nodes. Configure the driver/TF
adapter to publish `/robot_ID/odom` in frame `mission` and child frame
`robot_ID/base_link`. The supported synthetic motion is planar/nonholonomic;
unexpected lateral velocity or incompatible frames produce a latched stop.

Published output is `/robot_ID/policy/cmd_vel`, not a vendor command topic.
Do not connect/remap it to real actuators before the platform adapter, measured
limits and independent physical stop path have been implemented and tested.
`/robot_ID/policy/status` provides mode, phase, local goal, timing and faults.
Peer messages use `/swarm/robot_ID/state_v1` and `swarm_interfaces/PeerState`.

The optional ROS parameter `completion_profile` chooses endpoint tolerances and the mission deadline;
it DOES NOT inject network loss or delay. Real ROS peer transport uses the real
network. Synthetic impairment profiles are still provided by the separate
Gazebo/Python evaluation harness. Do not interpret the modeled 72-byte packet
budget as DDS wire size or measured Wi-Fi performance.

## Stop and restart

Ctrl+C stops a shadow node and publishes a zero request on shutdown when its ROS
context remains available. This is not an independent emergency stop. Local
odometry/clock/inference faults latch zero commands. A peer fault is propagated
through peer state; a peer restart in the same mission also latches a stop.
Peer fault delivery is not guaranteed during a network partition, so a local
peer timeout is checked too. After a latched fault, fix the cause and restart
all nodes with a new common mission ID. Fresh data alone does not re-enable motion.

## Upgrade and rollback

Stop all nodes. Install each new release into a separate directory, verify its
archive/source checksums and policy checksum, then repeat installation/preflight
and shadow checks. Never mix policy versions in one mission. Retain the previous
release directory and its manifest for rollback; start a new mission after
switching back. No training or automatic weight adaptation occurs on the robot.

## Preflight

```bash
.venv/bin/python scripts/preflight_deployment.py --mode shadow
.venv/bin/python scripts/preflight_deployment.py --mode hardware
```

The hardware check intentionally fails in this release because robot profiles,
actuator adapters, localization, clocks and independent stopping are unverified.
Do not replace this result with a manually flipped readiness label.

See docs/deployment/PROTOCOL.md, OPERATIONS.md and HARDWARE_PENDING.md in the
release directory for interface details and pending laboratory work.
