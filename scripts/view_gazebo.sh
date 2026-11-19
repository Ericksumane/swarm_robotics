#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PATH="/usr/bin:/bin:$PATH"
source /opt/ros/jazzy/setup.bash
exec /usr/bin/python3 scripts/launch_gazebo_replay.py "$@"
