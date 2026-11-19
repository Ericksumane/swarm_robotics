#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
/usr/bin/python3 -c 'import json; d=json.load(open("experiments/selected_startup_candidate.json")); e=d["complete_mission"]; print("Successful recorded dispersed-start demo:", d["demo_seed"], "profile:",d["demo_profile"]); print("Good strict full missions at 120 s:",e["good_120s"]["strict_successes"],"/20; at 240 s:",e["good_240s"]["strict_successes"],"/20"); print("Research partial result. Recorded Python poses, not Gazebo physics validation.")'
startup_run=$(/usr/bin/python3 -c 'import json; print(json.load(open("experiments/selected_startup_candidate.json"))["replay"])')
exec bash scripts/view_gazebo.sh "$startup_run" --speed 2 --gui-config "$startup_run/gui.config" --end-hold-s 3 "$@"
