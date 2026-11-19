#!/usr/bin/env bash
set -eo pipefail
task_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_root"
if [[ "$(uname -m)" != x86_64 ]]; then echo "This installer is tested only on x86_64. Obtain the target platform profile first." >&2; exit 2; fi
source /etc/os-release
if [[ "$ID" != ubuntu || "$VERSION_ID" != 24.04 ]]; then echo "This installer is tested only on Ubuntu 24.04 / ROS 2 Jazzy." >&2; exit 2; fi
command -v uv >/dev/null || { echo "Install uv before running this installer." >&2; exit 2; }
[[ -f /opt/ros/jazzy/setup.bash ]] || { echo "ROS 2 Jazzy is a prerequisite; this script does not install the OS or ROS." >&2; exit 2; }
[[ ! -e .venv ]] || { echo "Refusing to replace an existing .venv. Use a freshly extracted release directory." >&2; exit 2; }
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV
uv venv --python 3.10 .venv
UV_HTTP_TIMEOUT=300 uv pip install --python .venv/bin/python 'torch==2.5.1+cpu' --index-url https://download.pytorch.org/whl/cpu
UV_HTTP_TIMEOUT=300 uv pip install --python .venv/bin/python -r requirements/runtime-locked.txt
uv pip install --python .venv/bin/python --no-deps --no-build-isolation -e . -e external/on-policy
uv pip check --python .venv/bin/python
.venv/bin/python scripts/smoke_runtime_install.py
bash scripts/build_ros.sh
.venv/bin/python scripts/preflight_deployment.py --mode shadow
uv pip freeze --python .venv/bin/python --exclude-editable > requirements/runtime-installed.txt
echo "Common shadow runtime installed. Hardware actuation is not enabled."
