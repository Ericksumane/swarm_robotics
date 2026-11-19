#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PATH="/usr/bin:/bin:$PATH"
source /opt/ros/jazzy/setup.bash
source go2_gait_ws/install/local_setup.bash
bundle="$(/usr/bin/python3 - <<'PYCODE'
from pathlib import Path
import hashlib
import json

record = json.loads(Path("experiments/selected_delivery_candidate.json").read_text())
path = Path(record["actor_bundle"])
with path.open("rb") as file:
    digest = hashlib.file_digest(file, "sha256").hexdigest()
if digest != record["sha256"]:
    raise SystemExit("Selected policy checksum mismatch.")
print(path)
PYCODE
)"
exec /usr/bin/python3 scripts/gazebo_live.py --mixed --communication nominal \
  --seed 42 --time-limit 150 "$@" --startup-bundle "$bundle"
