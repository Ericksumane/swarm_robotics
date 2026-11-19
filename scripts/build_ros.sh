#!/usr/bin/env bash
set -eo pipefail
# Use the ROS distribution's system Python, never a training virtualenv.
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PATH=/usr/bin:/bin
source /opt/ros/jazzy/setup.bash
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root/ros2_ws"
exec colcon build --symlink-install --packages-select swarm_interfaces swarm_runtime \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPYTHON_EXECUTABLE=/usr/bin/python3
