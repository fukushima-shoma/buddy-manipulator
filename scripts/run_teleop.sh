#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_path="$project_dir/.venv/bin/python"
mjpython_path="$project_dir/.venv/bin/mjpython"

if [ ! -x "$python_path" ] || [ ! -x "$mjpython_path" ]; then
  echo "Virtual environment is incomplete. Run ./scripts/setup.sh first." >&2
  exit 1
fi

cd "$project_dir"
if [ "$(uname -s)" = "Darwin" ] && [ "$(/usr/sbin/sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ]; then
  exec /usr/bin/arch -arm64 "$python_path" "$mjpython_path" \
    -m buddy_manipulator.teleop_demo "$@"
fi

if [ "$(uname -s)" = "Darwin" ]; then
  exec "$mjpython_path" -m buddy_manipulator.teleop_demo "$@"
fi

exec "$python_path" -m buddy_manipulator.teleop_demo "$@"
