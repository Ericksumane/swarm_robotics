#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
actor="$(/usr/bin/python3 - <<'PY'
import hashlib
import json
from pathlib import Path

record = json.loads(Path("experiments/selected_wheeltec_candidate.json").read_text())
actor = Path(record["actor"])
content = actor.read_bytes()
if hashlib.sha256(content).hexdigest() != record["sha256"]:
    raise SystemExit("Selected WheelTec actor checksum mismatch.")
if json.loads(content)["contract"] != record["contract"]:
    raise SystemExit("Selected WheelTec actor contract mismatch.")
print(actor)
PY
)"
exec bash scripts/run_refined_cooperative_gazebo.sh --random-starts \
    --cooperative-terminal-recovery "$@" --wheeltec-arc-actor "$actor"
