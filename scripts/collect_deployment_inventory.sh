#!/usr/bin/env bash
set -eo pipefail
if [[ -z ${ROS_DISTRO:-} ]]; then
    printf '%s\n' 'Source the installed ROS 2 setup.bash first.' >&2
    exit 2
fi
printf 'ROS distribution: %s\n' "$ROS_DISTRO"
printf 'ROS domain: %s\n' "${ROS_DOMAIN_ID:-0}"
uname -m
cat /etc/os-release
printf '\nROS nodes\n'
timeout 10 ros2 node list
printf '\nROS topic types\n'
timeout 10 ros2 topic list -t
if [[ -n ${1:-} ]]; then
    printf '\nOdometry topic information\n'
    timeout 10 ros2 topic info -v "$1"
fi
