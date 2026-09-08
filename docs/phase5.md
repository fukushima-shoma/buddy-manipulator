# Phase 5: Goal-conditioned pick-and-place

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
`outputs/phase5/goal_diagnostics/`. Episodes 0-19 in the training directory are the locked 20/20
foundation run. Phase 5B then scaled the directory to 80 attempts and 78 successes; the two retained
failures are excluded from BC training by the default success filter.
The machine-readable foundation result is
`docs/experiment_results/2026-09-07-phase5a-foundation.json`.

## Phase 5B: goal-conditioned BC baseline

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

## Phase 5C: semantic phase conditioning

Phase 5C tests the temporal-aliasing hypothesis without changing the demonstrations or held-out
split. Each sample receives a ten-value one-hot phase derived from its episode-relative timestamp:
approach, descend, close, lift, retract, rotate, extend, lower, open, or retreat. Closed-loop rollout
generates the same signal from elapsed control time. Existing checkpoints remain compatible because
phase conditioning defaults to disabled.

```bash
./scripts/run_training.sh data/goal_demonstrations \
  --epochs 50 --action-horizon 8 \
  --holdout-goal purple:yellow --phase-conditioning --device mps \
  --output-dir outputs/phase5/goal_bc/phase_seed7
```

The phase-only model reduced seen-goal validation MAE from 0.04465 to 0.04000 and held-out MAE from
0.05180 to 0.04756. On fresh closed-loop scenes it scored 1/10 red-to-green and, importantly, the
first non-zero held-out result: 1/10 purple-to-yellow. Executing only one action from each predicted
chunk was tested on the same ten held-out scenes and regressed from 1/10 to 0/10, so open-loop chunk
execution was not the primary failure cause.

A second controlled model added a deterministic RGB-D centroid/depth/area bottleneck for whichever
red or purple object the goal selected. It reached 2/10 on a fresh seen-goal set and 1/10 on a fresh
held-out set, while held-out MAE regressed slightly to 0.04829. The extra feature is therefore kept
as an ablation option (`--use-goal-object-features`) but is not promoted.

Phase conditioning is a partial positive result, not a robust policy: failures still cluster at
grasp acquisition and compound into transport errors. The next controlled experiment should add
learned recurrent history, preserving the same held-out combination and comparing against both the
Phase 5B and phase-only baselines. Full metrics and decisions are recorded in
`docs/experiment_results/2026-09-07-phase5c-phase-conditioning.json`.

## Phase 5D: learned recurrent history

Phase 5D removes the scripted clock and encodes the most recent eight normalized joint states with a
64-unit GRU. The current RGB-D frame still passes through the CNN. At episode start, missing history
is padded with the first observation; runner state is reset between episodes. Because demonstrations
are sampled at 5 Hz, history checkpoints automatically execute one action before re-observation so
the runtime window has the same cadence as training.

```bash
./scripts/run_training.sh data/goal_demonstrations \
  --epochs 50 --action-horizon 8 --history-horizon 8 \
  --use-goal-object-features --holdout-goal purple:yellow --device mps \
  --output-dir outputs/phase5/goal_bc/history8_object_seed7
```

The GRU-only model reduced held-out MAE to 0.02716, but scored 0/10 in rollout. Audit showed that
its average validation result depended on expert-generated histories: first-frame action MAE was
0.149 and base-yaw MAE was 0.104. Adding the goal-selected RGB-D object bottleneck reduced those
bootstrap errors to 0.0926 and 0.0762. The resulting seed-7 model reached 3/10 on a fresh seen task
and 2/10 on a fresh held-out task.

Seeds 7, 17, and 27 were then trained with identical data split and sampling. Their equal-weight
ensemble reached 5/10 on one fresh held-out set, but a balanced 40-scene benchmark exposed a strong
target asymmetry:

| Goal | Success |
|---|---:|
| red to green | 0/10 |
| red to yellow | 8/10 |
| purple to green | 0/10 |
| purple to yellow (held out) | 4/10 |
| **Total** | **12/40 (30%)** |

The recurrent model is therefore not promoted. It demonstrates that learned history can greatly
improve offline action prediction and that object grounding improves bootstrap control, but green
target geometry remains unresolved. The next controlled experiment should provide an explicit
observable target representation or a factorized target-conditioned decoder. Full results are in
`docs/experiment_results/2026-09-07-phase5d-recurrent-history.json`.

## Phase 5E: target-conditioning ablations

Phase 5E tests three explanations for the green-target failures while preserving the Phase 5D data,
GRU history, object bottleneck, and held-out split.

1. `--use-goal-target-features` exposes the selected green or yellow mask centroid, depth, and area.
   Segmentation was non-empty in all 5,440 recorded frames, but partial arm occlusion makes the
   feature move during execution. Held-out MAE improved slightly to 0.02538 while balanced rollout
   regressed to 3/40, with no green-target successes.
2. `--factorized-target-heads` uses independent green and yellow action decoders. It produced the
   first green success in this phase (1/20), but held-out MAE regressed to 0.04696 and held-out
   rollout scored 0/10. Splitting the entire decoder removed the sharing needed to compose the
   purple object with the yellow behavior learned only from red-to-yellow episodes.
3. `--target-residual-heads` preserves a shared decoder and adds zero-initialized target-specific
   residual heads at 25% scale. It achieved the best seed-7 validation MSE, 0.02015, but scored 0/40
   in balanced rollout. The Phase 5D ensemble scored 11/40 on the identical seed-2924 scenes.

All three variants are rejected. Better offline action prediction did not improve closed-loop task
completion, and increasingly specialized target decoders damaged compositional sharing. The next
build should stop treating the 13.4-second behavior as one monolithic action prediction problem:
train or route a shared grasp-acquisition skill first, then invoke a target-conditioned
transport-and-place skill after observable grasp confirmation. Full results are in
`docs/experiment_results/2026-09-08-phase5e-target-conditioning.json`.

## Phase 5F: hierarchical skill composition

Phase 5F replaces the single 13.4-second prediction problem with overlapping skill windows. The
grasp dataset covers 0.0-4.7 seconds through the completed vertical lift. The place dataset starts
at 3.2 seconds, immediately after gripper closure, and continues through release and retreat. This
overlap gives the place policy support around variable handoff times without rewriting the recorded
episodes. `--goal-skill grasp|place` selects the view during training and evaluation.

The goal is explicitly factorized as well: grasp sees object identity but not target identity, while
place sees target identity but not object identity. This turns held-out purple-to-yellow into the
composition of a known purple grasp and known yellow placement instead of an unseen four-way token.
An observable gate requires two consecutive RGB-D detections above 0.09 m while both finger joints
are closed. A time-based oracle is retained only as a diagnostic.

The first learned grasp models never lifted the object on fresh rollout, including the relative-phase
variant, so the observable composition scored 0/8. The gate was not relaxed because that would route
an ungrasped object into the place policy. A deterministic RGB-D/IK grasp was then used to isolate the
learned place skill. Target-only placement without phase scored 0/8; adding skill-relative phase
reached 3/8. Under the factorized goal, separate target heads no longer remove object compositional
sharing and improved a paired benchmark from 6/12 to 8/12.

The locked fresh seed-3530 benchmark for expert grasp plus the factorized target/phase place policy
scored 24/40 (60%): red-green 7/10, red-yellow 6/10, purple-green 7/10, and held-out purple-yellow
4/10. This doubles the Phase 5D model-only balanced result of 12/40, but it is a diagnostic hybrid,
not a fully learned policy and not yet a recommended robust system. The next bottleneck is learned
acquisition precision; future work should predict object-relative grasp poses or residuals around the
RGB-D/IK acquisition controller instead of continuing direct joint-action BC.

```bash
./scripts/run_training.sh data/goal_demonstrations \
  --goal-skill grasp --phase-conditioning --history-horizon 8 \
  --action-horizon 8 --use-goal-object-features \
  --holdout-goal purple:yellow \
  --output-dir outputs/phase5/hierarchical/grasp_object_phase_seed7

./scripts/run_training.sh data/goal_demonstrations \
  --goal-skill place --phase-conditioning --factorized-target-heads \
  --history-horizon 8 --action-horizon 8 \
  --holdout-goal purple:yellow \
  --output-dir outputs/phase5/hierarchical/place_target_phase_factorized_seed7

./scripts/run_hierarchical_goal_policy.sh \
  outputs/phase5/hierarchical/grasp_object_phase_seed7/bc_policy.pt \
  outputs/phase5/hierarchical/place_target_phase_factorized_seed7/bc_policy.pt \
  --grasp-controller expert --episodes 40 --seed 3530
```

Full results are in
`docs/experiment_results/2026-09-08-phase5f-hierarchical-skills.json`.

## Phase 5G: bounded grasp-pose residual

Phase 5G tests a structured alternative to failed absolute-joint grasp BC. The analytical RGB-D pose
is retained as a prior, and a zero-initialized MLP predicts a bounded XYZ correction from pose,
image centroid, mask area, and selected object color. The target is the recorded object-top pose,
with purple-to-yellow held out. This changes what the model learns from an entire trajectory to a
single object-relative decision while keeping IK and safe keyframes deterministic.

Offline held-out mean pose error improved from 1.811 mm to 1.646 mm, but this did not improve grasp
robustness. Paired grasp-only rollout scored 40/40 for the unmodified RGB-D/IK baseline, 34/40 for a
full residual, and 38/40 at 25% blend. In two full-task paired sets, the 25% blend scored 45/80 versus
44/80, with four recoveries and three regressions. The residual is therefore not promoted.

The important finding is objective mismatch: object-center geometry is not an adequate supervision
target for a narrow, contact-sensitive grasp. The next experiment should collect controlled XY/Z
pose perturbations and learn from lift success, contact stability, and margin, while adding camera
calibration or domain variation so the already-perfect in-distribution analytical baseline has
meaningful failure cases.

```bash
./scripts/run_grasp_pose_training.sh data/goal_demonstrations \
  --output-dir outputs/phase5/grasp_pose_residual/seed7 \
  --epochs 200 --blend 0.25

./scripts/run_grasp_pose_rollout.sh \
  outputs/phase5/grasp_pose_residual/seed7/grasp_pose_residual.pt \
  --controller residual --blend 0.25 --episodes 40 --seed 3631
```

Full results are in
`docs/experiment_results/2026-09-08-phase5g-grasp-pose-residual.json`.

## Phase 5H: outcome-trained grasp success critic

Phase 5H replaces geometric-center supervision with the task outcome that actually matters. For each
of 60 randomized scenes, the collector executes the zero-offset RGB-D/IK grasp and 12 bounded random
XY perturbations. The resulting 780 attempts contain RGB-D pose features, the applied residual,
reachability, selected/distractor final positions, and a binary lift label. A grasp is successful only
when the selected object reaches 8 cm while the distractor stays below 6 cm.

The MLP critic scores candidate pose residuals rather than directly controlling joints. At runtime a
bounded 3 mm grid is ranked, but the analytical candidate remains a safety fallback: a learned
candidate must exceed its predicted success by 0.03. IK-unreachable candidates are removed. This
preserves the analytical motion primitive and limits learning to a contact-sensitive pose decision.

The dataset had 548/780 successful attempts, 54/60 successful zero-offset baselines, and 47
IK-unreachable wide perturbations. The scene-held-out split reached 94.2% classification accuracy and
12/12 ranking success, although the validation baseline was also 12/12. Fresh paired rollouts provide
the meaningful evidence:

| Condition | Analytical candidate | Success critic | Recoveries | Regressions |
|---|---:|---:|---:|---:|
| Normal, seed 3934 | 38/40 | 40/40 | 2 | 0 |
| Normal replication, seed 4237 | 37/40 | 40/40 | 3 | 0 |
| **Normal total** | **75/80** | **80/80** | **5** | **0** |
| Known +6 mm Y bias, seed 4035 | 34/40 | 40/40 | 6 | 0 |
| Full task + learned place, seed 4136 | 26/40 | 26/40 | 3 | 3 |

The known-bias result assumes the execution bias is available when constructing candidate outcomes;
it is a controlled calibration-stress benchmark, not automatic online bias estimation. GRASP-CRITIC-01
is accepted as the preferred Phase 5 grasp component because it improves acquisition with no paired
grasp regressions. It does not promote the end-to-end hierarchy: the unchanged total score shows that
post-grasp pose consistency and learned placement are now the active bottlenecks.

```bash
./scripts/collect_grasp_perturbations.sh \
  --scenes 60 --candidates 13 --maximum-xy-mm 18 \
  --seed 3833 --output data/grasp_perturbations/phase5h_seed3833.npz

./scripts/train_grasp_success_critic.sh \
  data/grasp_perturbations/phase5h_seed3833.npz \
  --output-dir outputs/phase5/grasp_success_critic/seed17 \
  --epochs 160 --seed 17 --device mps

./scripts/run_grasp_success_critic.sh \
  outputs/phase5/grasp_success_critic/seed17/grasp_success_critic.pt \
  --controller critic --episodes 40 --seed 3934
```

Full results are in
`docs/experiment_results/2026-09-08-phase5h-grasp-success-critic.json`.

## Phase 5I: canonical grasp-to-place handoff

Phase 5I tests whether post-grasp state variation causes the placement regressions seen in Phase 5H.
An optional closed-gripper keyframe moves every acquired object to the reachable Cartesian pose
`(0.20, 0.06, 0.20) m` before placement. The extra 1.2 seconds is included in the fixed task budget,
the recurrent joint history includes this motion, and rollout JSON records object positions before and
after normalization.

The intervention worked mechanically. Across seed 4338, object XY standard deviation changed from
`(5.17, 21.81) mm` before normalization to `(1.95, 0.64) mm` afterward. It did not work as an
inference-only model change: the current place policy fell from 9/20 to 7/20, recovering four episodes
but regressing six because the canonical state was outside its training distribution.

To remove that mismatch, 80 new canonical-handoff expert trajectories were collected; 75 were task
successful and used by the default successful-only trainer. A new `handoff_place` dataset view starts
after the 7.1-second grasp-plus-normalization prelude. The matched factorized model achieved validation
MAE 0.01067, but scored 0/20 against 11/20 for the current hierarchy on seed 4439. A second model kept
object identity and added explicit goal-selected object/target visual features; it reached validation
MAE 0.01407 and also scored 0/20.

HANDOFF-01 is rejected as a policy improvement but retained as instrumentation and an ablation. The
result rules out handoff variance as a sufficient explanation: absolute-action BC still compounds
small closed-loop errors even from a narrow initial state. The next experiment should retain the
analytical transport primitive and learn a bounded release-pose or trajectory residual from actual
placement outcomes, mirroring the successful Phase 5H grasp critic.

```bash
./scripts/collect_goal_demos.sh \
  --episodes 80 --seed 4439 \
  --handoff-normalization canonical \
  --output-dir data/goal_demonstrations_handoff

./scripts/run_training.sh data/goal_demonstrations_handoff \
  --goal-skill handoff_place --phase-conditioning \
  --factorized-target-heads --history-horizon 8 --action-horizon 8 \
  --holdout-goal purple:yellow --device mps
```

Full results are in
`docs/experiment_results/2026-09-08-phase5i-canonical-handoff.json`.

## Phase 5J: outcome-ranked structured placement

Phase 5J applies the successful grasp-critic pattern to placement. The model does not predict joint
trajectories. It scores bounded XY release residuals using the proprioceptive handoff tool pose,
calibrated target position, structured goal, and candidate residual. RGB-D grasp selection, IK
transport, descent, gripper timing, and retreat remain deterministic. A candidate must beat the
analytical fallback probability by 0.03, and IK-unreachable candidates are excluded.

The collector executed 540 end-to-end attempts across 60 scenes. Wide ±70 mm perturbations produced
290/540 successes and 154 IK-unreachable candidates; the zero-residual analytical placement reached
60/60. The scene-held-out critic achieved 93.5% classification accuracy and 12/12 ranking success.

On fresh seed 4641, analytical and critic placement both reached 40/40 under normal calibration. On
seed 4742 with a controlled −55 mm X release bias, the analytical fallback reached 34/40 and the
critic reached 40/40, recovering all six failures with no regressions. The stress result assumes the
bias is supplied while candidate outcomes are constructed; Phase 5K removes that assumption.

PLACE-CRITIC-01 is accepted. The Phase 5 system baseline becomes critic-ranked grasp plus
critic-ranked release, with deterministic IK skills between decisions. This is a hybrid learned and
model-based system, not an end-to-end neural controller.

Full results are in
`docs/experiment_results/2026-09-08-phase5j-placement-success-critic.json`.
