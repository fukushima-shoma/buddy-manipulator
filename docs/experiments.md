# Model experiment journal

This journal records why each model change was attempted, how it was evaluated, and whether it
was promoted. Generated metrics and checkpoints live under `outputs/phase4/`; this file keeps the
durable conclusions because generated artifacts are intentionally excluded from Git.

## Promotion rule

A candidate is promoted only when paired closed-loop MuJoCo evaluation on unseen block positions
shows a repeatable improvement over `outputs/phase4/chunked/bc_policy.pt`. Offline loss is diagnostic
and cannot promote a model by itself. The standard robustness check uses training seeds 7, 17, and
27 and rollout seeds 404, 505, and 606 with 30 positions per rollout seed.

## Experiment history

| ID | Hypothesis | Change | Result | Decision |
|---|---|---|---|---|
| BC-01 | Longer predictions reduce temporal averaging. | Predict eight future actions and execute the full chunk. | 54-episode model reached 13/20 versus 0/5 for single-step BC. | Promoted as baseline. |
| DATA-01 | Correct expert trajectories around failed positions improve weak workspace regions. | Add failure-targeted demonstrations. | Offline error improved, but paired success fell from 15/30 to 13/30. | Rejected. |
| DATA-02 | Limiting failure replay to 20% prevents overfitting. | Weighted source-balanced replay. | Mean delta was +0.4 points across three training seeds, with 10-point standard deviation. | Not promoted. |
| STAB-01 | Spatial splitting and fewer repeated frames reduce seed variance. | Spatial/source split plus deterministic samplers. | All variants regressed; best stabilization variant reached 32/90 versus 48/90. | Rejected; controls retained for ablation. |
| DIFF-01 | Modeling a distribution over action chunks avoids harmful conditional means. | Conditional RGB-D DDPM with EMA and deterministic DDIM rollout. | 0/30 on rollout seed 404. | Rejected. |
| DIFF-02 | The 10-step sampler amplifies high-noise prediction errors. | Compare more DDIM steps and reduced initial noise without retraining. | 25-step zero-noise sampling also produced 0/30. | Rejected. |
| DIFF-03 | A proven BC prior can prevent visual-conditioning collapse while diffusion learns corrections. | Freeze the BC policy and diffuse only its normalized action residual. | Pending. | Pending. |

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
