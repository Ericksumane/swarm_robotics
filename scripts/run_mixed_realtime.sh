#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."

# The validated default uses 20-step physics batches.
# Set SWARM_PHYSICS_CHUNK_STEPS explicitly for experimental transport timing.
profile="${1:-nominal}"
shift "$(( $# > 0 ))"
case "$profile" in
    nominal|good|moderate|severe) ;;
    *) echo "Usage: $0 {nominal|good|moderate|severe} [seed] [--headless]" >&2; exit 2 ;;
esac
args=()
if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
    args+=(--seed "$1")
    shift
fi
if [[ "${1:-}" == --headless ]]; then
    args+=(--headless)
    shift
fi
if (( $# )); then
    echo "Expected an optional numeric seed followed by optional --headless." >&2
    exit 2
fi

# Select the verified WSLg NVIDIA renderer when available.
# Preserve explicit caller settings and fall back to the current renderer.
if [[ -e /dev/dxg && -z "${GALLIUM_DRIVER:-}" ]] && command -v glxinfo >/dev/null; then
    if renderer=$(GALLIUM_DRIVER=d3d12 MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA timeout 8 glxinfo -B 2>/dev/null)        && [[ "$renderer" == *"D3D12 (NVIDIA"* && "$renderer" == *"Accelerated: yes"* ]]; then
        export GALLIUM_DRIVER=d3d12
        export MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA
        echo "Gazebo graphics: D3D12 NVIDIA hardware acceleration"
    fi
fi

unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
export PATH="/usr/bin:/bin:$PATH"
source /opt/ros/jazzy/setup.bash
source go2_gait_ws/install/local_setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-76}"
manifest_path="${SWARM_ACTOR_MANIFEST:-experiments/mixed_campaign_160k_candidate.json}"
actors=$(/usr/bin/python3 - "$manifest_path" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
actors = Path(manifest["actor_directory"])
for name in ("go2_actor.json", "jackal_actor.json", "wheeltec_actor.json", "translation_actor.json"):
    if hashlib.sha256((actors / name).read_bytes()).hexdigest() != manifest["sha256"][name]:
        raise SystemExit(f"Actor checksum mismatch: {name}")
print(actors)
PY
)

exec /usr/bin/python3 scripts/gazebo_live.py --mixed --random-starts \
  --communication "$profile" --real-time-factor .98 --time-limit 240 \
  --cooperative-feedback --cooperative-translation --cooperative-cruise-speed .55 \
  --cooperative-adaptive-speed --cooperative-terminal-recovery \
  --go2-forward-stride-scale 1.15 --go2-stride-cruise-only --wheeltec-reversal-recovery \
  --wheeltec-arc-actor "$actors/wheeltec_actor.json" --anchor-pose-actors "$actors" \
  --translation-speed-actor "$actors/translation_actor.json" "${args[@]}"
