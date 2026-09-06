#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
python_path="$project_dir/.venv/bin/python"
mjpython_path="$project_dir/.venv/bin/mjpython"

if [ ! -x "$python_path" ]; then
  echo "Virtual environment not found. Run ./scripts/setup.sh first." >&2
  exit 1
fi

headless=false
for argument in "$@"; do
  if [ "$argument" = "--headless" ]; then
    headless=true
  fi
done

cd "$project_dir"
if [ "$(uname -s)" = "Darwin" ] && [ "$(/usr/sbin/sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = "1" ]; then
  if [ "$headless" = false ]; then
    exec /usr/bin/arch -arm64 "$python_path" "$mjpython_path" -m buddy_manipulator.grasp_demo "$@"
  fi
  exec /usr/bin/arch -arm64 "$python_path" -m buddy_manipulator.grasp_demo "$@"
fi

if [ "$(uname -s)" = "Darwin" ] && [ "$headless" = false ]; then
  exec "$mjpython_path" -m buddy_manipulator.grasp_demo "$@"
fi

exec "$python_path" -m buddy_manipulator.grasp_demo "$@"
