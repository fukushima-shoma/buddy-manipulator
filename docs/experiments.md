# Model experiment journal

This journal records why each model change was attempted, how it was evaluated, and whether it
was promoted. Generated metrics and checkpoints live under `outputs/phase4/`; this file keeps the
durable conclusions because generated artifacts are intentionally excluded from Git.

## Promotion rule

A candidate is promoted only when paired closed-loop MuJoCo evaluation on unseen block positions
shows a repeatable improvement over the current recommended policy. Offline loss is diagnostic and
cannot promote a model by itself. Consumed rollout seeds are never reused after their failed positions
enter training data.

## Experiment history

| ID | Hypothesis | Change | Result | Decision |
|---|---|---|---|---|
| BC-01 | Longer predictions reduce temporal averaging. | Predict eight future actions and execute the full chunk. | 54-episode model reached 13/20 versus 0/5 for single-step BC. | Promoted as baseline. |
| DATA-01 | Correct expert trajectories around failed positions improve weak workspace regions. | Add failure-targeted demonstrations. | Offline error improved, but paired success fell from 15/30 to 13/30. | Rejected. |
| DATA-02 | Limiting failure replay to 20% prevents overfitting. | Weighted source-balanced replay. | Mean delta was +0.4 points across three training seeds, with 10-point standard deviation. | Not promoted. |
| STAB-01 | Spatial splitting and fewer repeated frames reduce seed variance. | Spatial/source split plus deterministic samplers. | All variants regressed; best stabilization variant reached 32/90 versus 48/90. | Rejected; controls retained for ablation. |
| DIFF-01 | Modeling a distribution over action chunks avoids harmful conditional means. | Conditional RGB-D DDPM with EMA and deterministic DDIM rollout. | 0/30 on rollout seed 404. | Rejected. |
| DIFF-02 | The 10-step sampler amplifies high-noise prediction errors. | Compare more DDIM steps and reduced initial noise without retraining. | 25-step zero-noise sampling also produced 0/30. | Rejected. |
| DIFF-03 | A proven BC prior can prevent visual-conditioning collapse while diffusion learns corrections. | Freeze the BC policy and diffuse only its normalized action residual. | Best variant reached 45/90 versus baseline 47/90. | Rejected. |
| DATA-03 | Broader corrective coverage will improve weak workspace regions. | Add one expert trajectory at all 43 failures from rollout seeds 404/505/606, then cap replay at 20%. | 13/30 on fresh seed 707 versus baseline 16/30. | Rejected alone. |
| VISION-01 | Coarse average-pooled CNN features limit object localization. | Add image-derived red-object centroid, depth, and area features to the learned CNN representation. | Single model reached 17/30 versus baseline 18/30. | Not promoted alone. |
| ENS-01 | Averaging independently initialized object-centric policies will reduce action variance. | Train model seeds 7/17/27 with fixed split and sampler, then average action chunks. | 65/90 versus baseline 57/90 on locked validation. | Provisionally promoted. |
| DATA-04 | Expert corrections at ensemble boundary failures improve weak cells. | Add 36 successful corrections from seeds 1102-1405 and retrain from scratch. | 8/30 versus current ensemble 21/30 on seed 1506. | Rejected. |
| FT-01 | Preserving the promoted representation while learning corrections avoids forgetting. | Preserve checkpoint split and normalization, freeze image encoder, and fine-tune action heads. | Fine-tuned ensemble scored 20/30 versus current 21/30; a 50/50 old/new blend scored 65/90 versus current 64/90 on untouched validation. | Not promoted; gain was too small. |
| HYBRID-01 | A calibrated model-based fallback can cover regions where learned behavior is unreliable. | Route the outer 1 cm workspace band to an RGB-D/IK expert and retain the learned ensemble in the center. | 86/90 versus current 66/90 on locked unseen placements; 20 recoveries and 0 regressions. | Promoted as recommended system policy. |
| POSE-01 | A learned metric-pose residual can preserve IK structure while correcting RGB-D bias. | Predict a bounded XYZ correction from analytical pose features. | Held-out mean error improved by 0.165 mm, but grasp success fell from 40/40 to 34/40 at full blend and 38/40 at 25%. | Rejected; tooling retained. |
| GRASP-CRITIC-01 | Actual lift outcomes can identify contact-stable residuals better than geometric-center labels. | Train a candidate-ranking critic on 780 perturbed grasps with an analytical fallback margin. | Two fresh normal seeds improved 75/80 to 80/80; known +6 mm Y-bias improved 34/40 to 40/40; full task tied 26/40. | Accepted for grasp acquisition; end-to-end policy unchanged. |
| HANDOFF-01 | Canonicalizing the post-grasp state will make learned placement more stable. | Add a fixed closed-gripper handoff pose; test inference-only use, matched retraining, and explicit object/target grounding. | XY variation fell sharply, but inference-only scored 7/20 versus 9/20 and both matched BC variants scored 0/20 versus 11/20. | Rejected as a policy; instrumentation retained. |
| PLACE-CRITIC-01 | Placement outcomes can supervise a bounded release decision more reliably than absolute-action BC. | Rank XY release residuals around deterministic IK transport with an analytical fallback. | Normal stayed 40/40; controlled −55 mm X-bias improved 34/40 to 40/40 with six recoveries and no regressions. | Accepted as the Phase 5 placement component. |
| BIAS-EST-01 | Persistent release bias can be estimated from prior observable placement outcomes. | Subtract selected commands from RGB-D object-to-target errors and rank candidates with the online estimate. | Hidden −55 mm X-bias improved from 37/40 to 40/40 on paired fresh scenes; final estimate was (−41.6, 3.7) mm. | Accepted for persistent calibration offsets. |

## DIFF-01 design decision

- Keep the observation, action horizon, normalization, data split, source weighting, control rate,
  action limits, and success definition aligned with the BC comparison.
- Predict diffusion noise for a normalized `(8, 6)` action chunk using an RGB-D/joint condition.
- Use a cosine 50-step training schedule, EMA weights, and 10 deterministic DDIM inference steps.
- Start with one training seed as a screening run. Only candidates that beat the paired baseline
  advance to the three-training-seed robustness benchmark.

The expected learning value is useful even if DIFF-01 fails: it separates representational limits
of deterministic BC from data coverage and closed-loop distribution-shift limits.

## DIFF-02 screening decision

DIFF-01's 10-step standard-normal sampler produced 0/30 successes. Held-out action MAE was 0.131
for a 10-step random sample, compared with 0.0277 when starting from zero noise. Increasing random
sampling to 50 steps improved MAE only to 0.0563, while 25-step zero-noise sampling reached 0.0203.
This isolates sampler error before spending compute on a new training seed or architecture. DIFF-02
therefore evaluates the same checkpoint with 25 DDIM steps and zero initial noise.

DIFF-02 also produced 0/30. At the initial robot state, the true block-heading target ranged from
0.169 to 0.345 rad. BC's predicted base yaw had correlation 0.83 with that target, while diffusion's
prediction was nearly constant around 0.332 rad and had correlation 0.39. The model learned the
action distribution but underused the image condition at the critical approach stage. DIFF-03 will
therefore retain the BC action as a frozen visual prior and model only a bounded residual.

## DIFF-03 and DATA-03 decisions

Full residual correction reached 35/90 (38.9%) versus the paired baseline's 47/90 (52.2%). It
recovered 19 baseline failures but regressed 31 successes. A 25% correction blend improved the
development seed 404 from 15/30 to 17/30, but validation seeds 505 and 606 gave an aggregate 45/90,
still below baseline. DIFF-03 is not promoted.

Episodes 00091 through 00133 add one successful expert correction at each of the baseline failures
from rollout seeds 404, 505, and 606. All 43 new episodes passed dataset validation. Retraining the
standard BC policy with a 20% replay cap scored 13/30 on fresh seed 707 versus baseline 16/30. This
shows that corrective coverage alone is insufficient with the current representation.

## VISION-01 design decision

Failure analysis repeatedly points to initial base-yaw localization. The standard CNN ends in a
2x2 average-pooled map, which is intentionally small but weak for precise spatial coordinates.
VISION-01 appends four observable RGB-D features: red-mask centroid X/Y, normalized masked depth,
and mask area. It does not use simulator object state and remains compatible with a real RGB-D
camera. The CNN remains present, so the bottleneck augments rather than replaces learned vision.

VISION-01 seed 7 reached 17/30 on fresh rollout seed 1001, compared with baseline 18/30. The result
does not support promotion of a single model. ENS-01 keeps split seed 7 and sampler seed 7 fixed,
changes only model initialization seeds 7/17/27, and averages their physical action chunks. Unlike
selecting the best random seed, this uses all prespecified models and directly targets variance.

## ENS-01 promotion decision

The three-model ensemble reached 21/30 on development seed 1001, compared with 18/30 for the
original baseline. The ensemble and all settings were then locked before evaluating three untouched
rollout seeds.

| Rollout seed | Original baseline | Object-centric ensemble | Delta |
|---:|---:|---:|---:|
| 1102 | 19/30 | 24/30 | +5 |
| 1203 | 21/30 | 20/30 | -1 |
| 1304 | 17/30 | 21/30 | +4 |
| **Total** | **57/90 (63.3%)** | **65/90 (72.2%)** | **+8 (+8.9 points)** |

The paired result recovered 19 baseline failures and regressed 11 baseline successes. This is the
first candidate with a material aggregate gain on prespecified unseen placements, so it becomes the
current recommended policy. The result is still provisional: 90 trials are enough for an engineering
promotion, but not a strong statistical claim. Future changes must compare against this ensemble on
new rollout seeds rather than reuse the locked validation set.

An additional user-run confirmation on fresh seed 1405 produced 19/30 for the ensemble. A subsequent
paired baseline run on the identical placements produced 17/30: four baseline failures were recovered,
two baseline successes regressed, and the net change was +2 (+6.7 points). Combined with the locked
validation, the current evidence is 84/120 (70.0%) for the ensemble versus 74/120 (61.7%) for the old
baseline. Seed 1405 is now considered consumed evaluation data and must not be used for training.

## DATA-04 and FT-01 design decision

The 36 ensemble failures from seeds 1102, 1203, 1304, and 1405 were converted into successful
expert demonstrations, episodes 00134 through 00169. All passed dataset validation. These evaluation
seeds are retired. Retraining the three object-centric members from scratch changed the episode split
because the dataset grew; the resulting ensemble scored only 8/30 on fresh development seed 1506,
versus 21/30 for the promoted ensemble. DATA-04 is rejected in this form.

FT-01 isolates continual learning from split variance. It initializes each model from its promoted
checkpoint, preserves that checkpoint's exact train/validation assignment and normalization, adds
only previously unseen episodes to training, freezes the image encoder, and updates the action head
at a lower learning rate. This treats the promoted model as prior knowledge instead of relearning the
entire policy after every aggregation round.

All three fine-tuned members improved their held-out normalized MSE, but their ensemble reached only
20/30 on development seed 1506, compared with 21/30 for the promoted ensemble. Validation loss again
proved insufficient as a promotion metric. A prespecified equal blend of the three promoted and three
fine-tuned members reached 22/30 on seed 1506, so that lower-risk variant advanced to untouched seeds.

| Rollout seed | Current ensemble | 50/50 old/new blend | Recovered | Regressed |
|---:|---:|---:|---:|---:|
| 1607 | 21/30 | 21/30 | 2 | 2 |
| 1708 | 25/30 | 25/30 | 0 | 0 |
| 1809 | 18/30 | 19/30 | 2 | 1 |
| **Total** | **64/90 (71.1%)** | **65/90 (72.2%)** | **4** | **3** |

The blend improved by one success (+1.1 points), which is not a repeatable or material gain under the
promotion rule. FT-01 and the blend are therefore retained as research artifacts but not promoted.
At the end of FT-01, the three-model object-centric ensemble remained the recommended policy. The
next experiment therefore targeted the remaining boundary failure mechanism directly.

## HYBRID-01 promotion decision

Across 270 consumed rollout placements, the ensemble succeeded on 111/118 placements (94%) inside
the central rectangle but only 79/152 (52%) in the outer workspace band. HYBRID-01 freezes a 1 cm
edge margin from that analysis. It localizes the block from calibrated RGB-D, routes edge placements
to the existing analytical IK grasp expert, and leaves central placements with the learned ensemble.
It never reads simulator object state. Development seed 1506 improved from 21/30 to 30/30 before
the margin and implementation were locked.

| Rollout seed | Learned ensemble | Gated hybrid | Recovered | Regressed |
|---:|---:|---:|---:|---:|
| 1901 | 20/30 | 29/30 | 9 | 0 |
| 2002 | 20/30 | 29/30 | 9 | 0 |
| 2103 | 26/30 | 28/30 | 2 | 0 |
| **Total** | **66/90 (73.3%)** | **86/90 (95.6%)** | **20** | **0** |

Exactly 45 validation episodes used the learned route and 45 used the expert route. All four hybrid
failures occurred on the learned center route; the expert edge route had no failures. The +22.2-point
gain is material and repeatable across all three prespecified unseen seeds, so HYBRID-01 becomes the
recommended system policy. This does not claim that the neural model itself reached 95.6%: the result
belongs to the combined learned/model-based controller. Physical deployment will require camera
intrinsic and extrinsic calibration before the fixed simulation calibration can be replaced.

The machine-readable decision record is
`docs/experiment_results/2026-09-07-model-improvement.json`.

## Phase 5C temporal-conditioning decision

The Phase 5B goal-conditioned model reduced held-out action MAE when its dataset grew from 20 to 78
successful episodes, but held-out task success remained 0/10. Phase 5C therefore kept the same data,
split seed, held-out purple-to-yellow combination, and eight-action horizon while adding only a
ten-way semantic phase signal derived from episode-relative time.

The phase-only model reduced validation MAE from 0.04465 to 0.04000 and held-out MAE from 0.05180
to 0.04756. Fresh rollouts scored 1/10 on red-to-green and 1/10 on the held-out purple-to-yellow
task. This is the first non-zero held-out composition, but it is not robust enough for promotion.
Executing one action per replan regressed to 0/10 on the same held-out scenes, ruling out full-chunk
execution as the primary failure.

A prespecified follow-up exposed the goal-selected red or purple RGB-D centroid, depth, and area to
the action head. It scored 2/10 on a fresh seen set and 1/10 on a fresh held-out set, while held-out
MAE worsened slightly to 0.04829. The bottleneck remains available as an ablation but is rejected as
the next baseline. The next experiment will replace the externally scheduled phase with learned
recurrent history and preserve the held-out protocol.

The machine-readable decision record is
`docs/experiment_results/2026-09-07-phase5c-phase-conditioning.json`.

## Phase 5D recurrent-history decision

An eight-step GRU over proprioception reduced held-out MAE from the phase-conditioned model's
0.04756 to 0.02716 without receiving the scripted phase clock. Closed-loop rollout nevertheless
scored 0/10. The discrepancy came from teacher-forced history: at the first frame, before expert
history exists, mean action MAE was 0.149 and base-yaw MAE was 0.104.

Adding the goal-selected RGB-D object bottleneck reduced first-frame action MAE to 0.0926 and
base-yaw MAE to 0.0762. Seed 7 then reached 3/10 on a fresh seen task and 2/10 on a fresh held-out
task. Three models with fixed split/sampler seeds and initialization seeds 7, 17, and 27 were averaged
without best-seed selection. The ensemble reached 5/10 on one held-out set, but its balanced fresh
benchmark scored only 12/40: 8/10 red-to-yellow, 4/10 purple-to-yellow, and 0/10 for each green-target
task.

Phase 5D is retained as a partial research result but not promoted. History solved much of the
offline temporal ambiguity, while the balanced rollout identified target geometry as the next
bottleneck. Phase 5E should test explicit visual target grounding or a factorized target decoder.

The machine-readable decision record is
`docs/experiment_results/2026-09-07-phase5d-recurrent-history.json`.

## Phase 5E target-conditioning decision

Three controlled variants tested the Phase 5D green-target failure. Goal-selected target mask
features improved held-out MAE slightly to 0.02538 but scored only 3/40 in balanced rollout, with
zero green successes. Fully factorized green/yellow decoders produced 1/20 green successes but
regressed held-out MAE to 0.04696 and scored 0/10 on the held-out composition. The split removed
cross-target sharing needed by purple-to-yellow.

A shared decoder with zero-initialized, 25%-scale target residual heads reached the best seed-7
validation MSE, 0.02015, yet scored 0/40 closed loop. On the identical seed-2924 scenes, the existing
Phase 5D recurrent ensemble scored 11/40. This is a direct rejection: lower offline error did not
translate to interaction robustness.

No Phase 5E checkpoint is promoted. The next experiment will use hierarchical skill composition:
shared grasp acquisition with an observable transition condition, followed by target-conditioned
transport and placement. This preserves object/target compositional sharing while shortening each
learned horizon.

The machine-readable decision record is
`docs/experiment_results/2026-09-08-phase5e-target-conditioning.json`.

## Phase 5F hierarchical-skill decision

The expert trajectory was split into overlapping acquisition (0.0-4.7 s) and place (3.2-13.4 s)
views. Acquisition receives only object identity and placement receives only target identity. An
RGB-D height plus finger-proprioception gate requires two stable lifted observations before learned
handoff. This is a stronger test than a permissive timer because failed grasps cannot silently enter
the transport skill.

The learned acquisition policy failed to produce a lift in 8/8 fresh trials, even after adding a
skill-relative phase signal, so the end-to-end learned hierarchy remains rejected. With deterministic
RGB-D/IK acquisition isolating the second skill, target-only BC scored 0/8 and relative phase scored
3/8. A target-factorized decoder is appropriate only after masking object identity: on identical
seed-3429 scenes it improved from 6/12 to 8/12 without sacrificing the held-out object combination.

The resulting diagnostic hybrid reached 24/40 on locked seed 3530, split 7/10 red-green, 6/10
red-yellow, 7/10 purple-green, and 4/10 held-out purple-yellow. It is retained as the Phase 5F
diagnostic baseline, not promoted as the recommended policy. It proves that shorter, factorized
skills improve the long-horizon result while identifying learned grasp precision as the next hard
block. The next experiment should learn object-relative grasp poses or residual corrections around
the analytical acquisition controller rather than regress absolute joint trajectories.

The machine-readable decision record is
`docs/experiment_results/2026-09-08-phase5f-hierarchical-skills.json`.

## Phase 5G grasp-pose residual decision

Direct joint-action grasp BC failed to reach the Phase 5F lift gate, so POSE-01 moved learning to a
structured interface. A zero-initialized MLP receives the analytical RGB-D XYZ pose, normalized
image centroid, mask area, and object identity, then predicts a correction in millimeters. Runtime
corrections are clipped to 15 mm per axis and blended with the analytical pose. Labels use the
recorded object-top position; purple-to-yellow remains test-only.

The model reduced held-out mean pose error from 1.811 mm to 1.646 mm, but increased held-out
worst-case error from 2.836 mm to 3.544 mm. That mismatch mattered: on identical seed-3631 grasp
scenes, the RGB-D/IK
baseline scored 40/40, full residual scored 34/40, and a 25% safety blend scored 38/40. Positive Y
corrections of less than 1 mm were enough to lose two grasps, showing that geometric center error is
not the same objective as contact-stable acquisition.

In full-task paired evaluation, the 25% blend scored 25/40 versus 24/40 on seed 3530, then tied 20/40
on fresh seed 3732. Across 80 trials it recovered four baseline failures but regressed three baseline
successes, for 45/80 versus 44/80. The one-point net gain is not material or repeatable enough for
promotion. POSE-01 is rejected as a controller but its bounded residual model, grasp-only evaluator,
and hierarchical integration are retained. The next data should come from on-policy pose
perturbations labeled by grasp outcome, not geometric object-center supervision. Because the current
analytical grasp already reached 40/40 in-distribution, harder calibration/domain variation is also
needed to create honest improvement headroom.

The machine-readable decision record is
`docs/experiment_results/2026-09-08-phase5g-grasp-pose-residual.json`.

## Phase 5H outcome-trained grasp decision

POSE-01 showed that smaller Cartesian error can still reduce grasp success. GRASP-CRITIC-01 therefore
uses 780 on-policy perturbation attempts from 60 scenes, labeled by actual selected-object lift and
distractor motion. The critic ranks bounded XY residual candidates while preserving RGB-D detection,
IK, keyframes, a zero-offset fallback, and an explicit confidence margin. Wide ±18 mm collection also
records IK-unreachable proposals as failures, so runtime can exclude them safely.

The collected set had a 70.3% overall success rate and a 90.0% baseline rate. A scene-level validation
split reached 94.2% classification accuracy and 12/12 candidate-ranking success. On fresh paired seed
3934, the normal analytical baseline reached 38/40 and the critic reached 40/40. Replication seed 4237
improved 37/40 to 40/40. Across both normal seeds the critic recovered all five failures with no
regressions. Under a controlled known +6 mm Y execution bias on seed 4035, the critic
reached 40/40 versus 34/40 and recovered all six failures without regressions. The stress benchmark
assumes the bias is known when candidates are constructed; bias estimation itself remains future work.

When composed with the Phase 5F learned place skill on seed 4136, both systems scored 26/40. The critic
recovered three expert-grasp failures but changed the transported-object state enough to regress three
other placements. The critic is therefore accepted as the preferred grasp component, but the
end-to-end hierarchy is not promoted. The next experiment should normalize the post-grasp handoff pose
or condition the place policy on the measured object/gripper state at transition.

The machine-readable decision record is
`docs/experiment_results/2026-09-08-phase5h-grasp-success-critic.json`.

## Phase 5I canonical-handoff decision

HANDOFF-01 added a 1.2-second closed-gripper move to `(0.20, 0.06, 0.20) m` and recorded the selected
object position on both sides of the transition. This reduced object XY standard deviation from
`(5.17, 21.81) mm` to `(1.95, 0.64) mm`, proving that the intervention actually narrowed the handoff
distribution. The existing place model nevertheless fell from 9/20 to 7/20 on paired seed 4338, with
four recoveries and six regressions.

Matched data did not solve the control problem. Of 80 newly collected canonical trajectories, 75 were
successful. A post-handoff factorized place model reached validation action MAE 0.01067, yet scored
0/20 versus the current hierarchy's 11/20 on seed 4439. Keeping object identity and exposing explicit
goal-selected object and target geometry produced validation MAE 0.01407 and the same 0/20 result.
These are additional examples of teacher-forced offline accuracy failing to predict closed-loop task
success.

The canonical motion, telemetry, dataset mode, and CLI remain useful research controls, but neither
model is promoted. Further absolute-action BC tuning is deprioritized. The next experiment will use
analytical target transport as a prior and learn only bounded placement decisions from actual task
outcomes, analogous to GRASP-CRITIC-01.

The machine-readable decision record is
`docs/experiment_results/2026-09-08-phase5i-canonical-handoff.json`.

## Phase 5J outcome-ranked placement decision

PLACE-CRITIC-01 collected 540 full grasp-and-place trials from 60 scenes with bounded random XY
release perturbations. The dataset contained 290 successes and 154 unreachable wide candidates;
the zero-residual analytical release succeeded in all 60 scenes. Proprioceptive forward kinematics is
used at handoff because the held object can be occluded from the overhead camera.

The critic reached 93.5% held-out classification accuracy and 12/12 candidate-ranking success. Paired
normal rollout preserved 40/40. With a supplied −55 mm X execution bias, it improved 34/40 to 40/40,
recovering every baseline failure without regressing a success. This passes the component promotion
gate. The next experiment must infer execution bias from observed placement error instead of receiving
it as an evaluation parameter.

The machine-readable decision record is
`docs/experiment_results/2026-09-08-phase5j-placement-success-critic.json`.

## Phase 5K online-bias decision

BIAS-EST-01 starts with no execution-bias knowledge and updates only after observing a completed
placement through the same RGB-D interface used for object localization. The estimator removes the
known candidate command from measured object-to-target displacement, clips implausible values, and
smooths repeated measurements before the placement critic ranks the next episode's outcomes.

On paired seed 4843 with a hidden −55 mm X release offset, the no-correction baseline reached 37/40
and the online system reached 40/40. All 40 final objects were observable, including the first episode
that ran with a zero estimate. The final estimate was `(−41.6, 3.7) mm`; exact identification is not
required because the observable measurement includes placement dynamics as well as the injected
offset. No baseline success regressed.

The estimator is promoted for persistent offsets, but not for per-episode disturbances. Phase 5L will
stress the hierarchy under randomized physics, appearance, and calibration conditions and report the
remaining failure envelope rather than tune against a single nominal scene.

The machine-readable decision record is
`docs/experiment_results/2026-09-08-phase5k-online-bias-estimation.json`.
