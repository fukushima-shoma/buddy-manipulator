# Phase 5A: Goal-conditioned pick-and-place foundation

## Goal

Phase 5 expands the single red-block grasp into an instruction-dependent task. The initial structured
instruction is `place the <object> block in the <target> zone`. Natural-language embeddings and a VLM
will be added only after the task, data, and evaluation interfaces are working end to end.

The first scene contains red and purple blocks plus green and yellow target zones, producing four
object-target combinations. Purple is used instead of blue because the robot arm is blue; a simple
blue color expert would otherwise segment the arm and contaminate demonstration labels.

## Goal representation

Each recorded frame includes a four-value goal vector:

| Goal component | Vector index |
|---|---:|
| red object | 0 |
| purple object | 1 |
| green target | 2 |
| yellow target | 3 |

For example, red-to-yellow is `[1, 0, 0, 1]`. The metadata also retains the readable instruction,
object color, target color, both initial object positions, and target position.

## Expert and operating envelope

The expert detects only the requested object in RGB-D, plans a top-down grasp with analytical IK,
retracts to a 20 cm transport radius, rotates through a low-speed corridor, extends to the requested
zone, releases, and retreats. It does not read simulator object coordinates for grasp localization.

Random scenes enforce two physical constraints discovered during development:

- object X is 0.23-0.34 m and Y is 0.04-0.13 m;
- object centers are separated by at least 0.11 m, giving the open gripper approach clearance.

The optional purple block and yellow zone are stored outside the Phase 4 camera view until a Phase 5
scene is activated. Finger-pad friction is increased only in the Phase 5 scene. This preserves Phase 4
checkpoint inputs and dynamics for reproducible historical evaluation.

## Run one task

```bash
./scripts/run_goal_demo.sh --object purple --target yellow
```

Headless mode writes a machine-readable result:

```bash
./scripts/run_goal_demo.sh \
  --object red --target green --seed 1001 --headless \
  --output outputs/phase5/goal_demo.json
```

Success requires the selected object center to finish inside the requested target with the distractor
outside it. The result includes placement error and final selected-object/target positions.

## Collect demonstrations

```bash
./scripts/collect_goal_demos.sh --episodes 20 --seed 905
./scripts/validate_dataset.sh data/goal_demonstrations
```

The collector cycles through all four goal combinations after a seed-controlled shuffle. Episode
arrays retain the existing RGB-D, joint, action, and object-position fields and add `(samples, 4)`
goal vectors. Existing Phase 3/4 schema-version-1 data remains valid because goal fields are optional.

The locked seed-905 foundation benchmark completed 20/20 tasks. Earlier diagnostic runs exposed and
recorded three design changes: slower transport alone was insufficient; retract-rotate-extend
waypoints removed long-radius throws; and the valid X/separation envelope removed unstable close-in
grasps and distractor collisions. Superseded runs are retained under
`outputs/phase5/goal_diagnostics/`; the clean training directory contains only the locked 20/20 run.
The machine-readable foundation result is
`docs/experiment_results/2026-09-07-phase5a-foundation.json`.

## Next build

The behavior-cloning pipeline now accepts optional four-value goals and can reserve one combination
with `--holdout-goal OBJECT:TARGET`. The held-out episodes are never used for parameter updates or
checkpoint selection.

```bash
./scripts/run_training.sh data/goal_demonstrations \
  --epochs 50 --action-horizon 8 \
  --holdout-goal purple:yellow --device mps \
  --output-dir outputs/phase5/goal_bc/data80_seed7

./scripts/run_goal_policy_rollout.sh \
  outputs/phase5/goal_bc/data80_seed7/bc_policy.pt \
  --goal purple:yellow --episodes 10 --seed 1409 --device mps
```

The first 20-demo model reached held-out offline MAE 0.0672 but scored 0/10 on held-out rollout and
0/5 on a seen red-to-green goal. Scaling collection to 78 successful episodes reduced held-out MAE
to 0.0518 and produced 2/10 on a fresh seen-goal rollout, but held-out rollout remained 0/10.
Therefore this model is a baseline, not a promoted policy.

The next experiment should address temporal state rather than add more of the same trajectories.
The 13.6-second expert behavior contains approach, grasp, transport, place, and retreat stages, while
the current policy receives no stage/history signal. A phase-conditioned or recurrent policy is the
next controlled hypothesis. Promotion still requires correct-object and correct-target placement;
action loss alone is insufficient.
