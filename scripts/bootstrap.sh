#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_executable="${PYTHON:-python3}"
"$python_executable" -c 'import sys; assert (3,12) <= sys.version_info[:2] < (3,15), "Python 3.12–3.14 required"'
"$python_executable" -m venv "$project_root/.venv"
"$project_root/.venv/bin/python" -m pip install -r "$project_root/requirements/dev.txt"
"$project_root/.venv/bin/python" -m pip check
"$project_root/.venv/bin/python" "$project_root/scripts/check_environment.py" --output "$project_root/reports/environment.json"
