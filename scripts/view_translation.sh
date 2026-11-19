#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
/usr/bin/python3 -c 'import json; d=json.load(open("experiments/selected_translation_candidate.json")); e=d["evaluation"]; print("Recorded successful demo seed:", d["demo_seed"]); print("Full test endpoint successes:",e["composed"]["successes"],"/",e["episodes"],"; continuous formation:",e["composed"]["strict_successes"],"/",e["episodes"]); print("Abstract models and recorded Python poses; not Gazebo physics validation.")'
translation_run=$(/usr/bin/python3 -c 'import json; print(json.load(open("experiments/selected_translation_candidate.json"))["replay"])')
exec bash scripts/view_gazebo.sh "$translation_run" --speed 2 --gui-config "$translation_run/gui.config" --end-hold-s 3 "$@"
