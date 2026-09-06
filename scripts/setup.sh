#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$project_dir"

if [ "$(uname -s)" = "Darwin" ] && [ "$(/usr/sbin/sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ]; then
  /usr/bin/arch -arm64 /usr/bin/python3 -m venv .venv
  /usr/bin/arch -arm64 .venv/bin/python -m pip install --upgrade pip
  /usr/bin/arch -arm64 .venv/bin/python -m pip install -e '.[dev]'
else
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -e '.[dev]'
fi

echo "Setup complete. Run ./scripts/run_demo.sh"
