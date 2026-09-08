#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec "$project_dir/scripts/run_structured_goal_policy.sh" "$@" \
  --domain-randomization --retry-failed-grasp
