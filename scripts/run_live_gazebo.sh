#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PATH="/usr/bin:/bin:$PATH"
source /opt/ros/jazzy/setup.bash
exec /usr/bin/python3 scripts/gazebo_live.py "$@"
