#!/usr/bin/env bash
set -eo pipefail
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PATH=/usr/bin:/bin
source /opt/ros/jazzy/setup.bash
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$task_root/ros2_ws/install/setup.bash"
cd "$task_root"
exec /usr/bin/python3 scripts/test_ros_mission_shadow.py
