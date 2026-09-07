#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_path="$project_dir/.venv/bin/python"

if [ ! -x "$python_path" ]; then
  echo "Virtual environment not found. Run ./scripts/setup.sh first." >&2
  exit 1
fi

cd "$project_dir"
if [ "$(uname -s)" = "Darwin" ] && [ "$(/usr/sbin/sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ]; then
  exec /usr/bin/arch -arm64 "$python_path" -m buddy_manipulator.collect_goal_demos "$@"
fi

exec "$python_path" -m buddy_manipulator.collect_goal_demos "$@"
