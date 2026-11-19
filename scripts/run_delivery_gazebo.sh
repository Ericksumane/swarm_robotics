#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
profile="${1:-good}"
if [ "$#" -gt 0 ]; then shift; fi
case "$profile" in
 nominal|good|moderate) deadline=300; duration=360 ;;
 severe) deadline=420; duration=480 ;;
 *) echo 'Usage: bash scripts/run_delivery_gazebo.sh nominal|good|moderate|severe [Gazebo options]' >&2; exit 2 ;;
esac
bundle="$(.venv/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
record=json.loads(Path('experiments/selected_delivery_candidate.json').read_text())
path=Path(record['actor_bundle'])
if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
    raise SystemExit('Delivery policy checksum mismatch.')
print(path)
PY
)"
exec bash scripts/run_live_gazebo.sh --communication "$profile" --startup-bundle "$bundle" --time-limit "$deadline" --duration "$duration" --keep-controlling "$@"
