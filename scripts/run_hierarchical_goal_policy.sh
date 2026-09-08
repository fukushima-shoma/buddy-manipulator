#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec "$project_dir/scripts/run_python.sh" \
  -m buddy_manipulator.rollout_hierarchical_goal_bc "$@"
