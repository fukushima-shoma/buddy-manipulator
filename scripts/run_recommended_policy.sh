#!/bin/sh
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
seed7="$project_dir/outputs/phase4/object_centric/seed7/bc_policy.pt"
seed17="$project_dir/outputs/phase4/object_centric/seed17/bc_policy.pt"
seed27="$project_dir/outputs/phase4/object_centric/seed27/bc_policy.pt"

for checkpoint in "$seed7" "$seed17" "$seed27"; do
  if [ ! -f "$checkpoint" ]; then
    echo "Recommended checkpoint not found: $checkpoint" >&2
    echo "See docs/phase4.md to reproduce the object-centric ensemble." >&2
    exit 1
  fi
done

exec "$project_dir/scripts/run_policy_rollout.sh" \
  "$seed7" \
  --ensemble-checkpoint "$seed17" \
  --ensemble-checkpoint "$seed27" \
  "$@"
