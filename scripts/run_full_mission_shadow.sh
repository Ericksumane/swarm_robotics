#!/usr/bin/env bash
set -eo pipefail
if [[ $# -lt 2 ]]; then echo "Usage: bash scripts/run_full_mission_shadow.sh ROBOT_ID UNIQUE_COMMON_MISSION_ID [ROS arguments...]" >&2; exit 2; fi
robot_id="$1";mission_id="$2";shift 2
case "$robot_id" in 0|1|2) ;; *) echo "ROBOT_ID must be 0, 1 or 2." >&2; exit 2;; esac
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PATH=/usr/bin:/bin
source /opt/ros/jazzy/setup.bash
source "$task_root/ros2_ws/install/setup.bash"
cd "$task_root"
exec ros2 run swarm_runtime full_mission_node --ros-args -r "__ns:=/robot_$robot_id" \
  -p "project_root:=$task_root" -p "robot_id:=$robot_id" -p "mission_id:=$mission_id" \
  -p common_frame:=mission -p "body_frame:=robot_$robot_id/base_link" "$@"
