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
