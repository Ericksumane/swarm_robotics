# Full ROS 2 shadow mission runtime

Packages: swarm_interfaces and swarm_runtime. Build with bash scripts/build_ros.sh
on Ubuntu 24.04 / ROS 2 Jazzy. The build deliberately uses system Python 3.12;
inference is isolated in Python 3.10 with fixed CPU dependencies.

The full_mission_node entry point handles local odometry, typed versioned peers,
the isolated robot_mission_service process and shadow velocity/status output.
It executes formation, alignment, future-slot start, translation and terminal
stop. No vendor actuator driver or physical enabling path is included.

Use scripts/run_full_mission_shadow.sh ROBOT_ID UNIQUE_COMMON_MISSION_ID. All
three nodes need the same new mission identifier and genuinely aligned pose
frames/clocks. Changing mission configuration during execution is rejected.
Accepted data are per-robot only; no simulator state or central phase oracle is
required. A peer reboot or detected clock/odometry/inference fault latches stop.

The three-process ROS smoke test verifies typed transport, scheduled start and
fault propagation with stationary odometry. It is not a full-trajectory test.
Full trajectory tests of local cores and live Gazebo tests are recorded separately.

The older policy_node is retained as a legacy formation-only prototype and is
not the full mission deployment entry point. See deployment/README.md for tested
common installation and operation, and deployment/HARDWARE_PENDING.md for what
still requires real robot data. Software shadow checks do not approve hardware.
