#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PATH="/usr/bin:/bin:$PATH"
source /opt/ros/jazzy/setup.bash
source go2_gait_ws/install/local_setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-59}"
start_args=(--seed 8605 --cooperative-reference experiments/runs/cooperative_reference_001/demonstrations.jsonl)
for arg in "$@"; do
  if [[ "$arg" == --random-starts ]]; then
    start_args=()
    break
  fi
done
exec /usr/bin/python3 scripts/gazebo_live.py \
  --mixed --communication nominal "${start_args[@]}" \
  --cooperative-feedback --cooperative-translation --cooperative-cruise-speed .55 \
  --cooperative-adaptive-speed --go2-forward-stride-scale 1.15 \
  --go2-stride-cruise-only --wheeltec-reversal-recovery --real-time-factor .99 "$@"
