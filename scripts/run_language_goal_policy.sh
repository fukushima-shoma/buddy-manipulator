#!/bin/sh
set -eu

if [ "$#" -lt 3 ]; then
  echo "usage: $0 GRASP_CRITIC PLACE_CRITIC INSTRUCTION [ROLLOUT_ARGS...]" >&2
  exit 2
fi

grasp_checkpoint=$1
placement_checkpoint=$2
instruction=$3
shift 3

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
exec "$project_dir/scripts/run_structured_goal_policy.sh" \
  "$grasp_checkpoint" "$placement_checkpoint" \
  --instruction "$instruction" "$@"
