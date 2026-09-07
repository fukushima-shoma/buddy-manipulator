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
| DIFF-01 | Modeling a distribution over action chunks avoids harmful conditional means. | Conditional RGB-D DDPM with EMA and deterministic DDIM rollout. | Pending. | Pending. |

## DIFF-01 design decision

- Keep the observation, action horizon, normalization, data split, source weighting, control rate,
  action limits, and success definition aligned with the BC comparison.
- Predict diffusion noise for a normalized `(8, 6)` action chunk using an RGB-D/joint condition.
- Use a cosine 50-step training schedule, EMA weights, and 10 deterministic DDIM inference steps.
- Start with one training seed as a screening run. Only candidates that beat the paired baseline
  advance to the three-training-seed robustness benchmark.

The expected learning value is useful even if DIFF-01 fails: it separates representational limits
of deterministic BC from data coverage and closed-loop distribution-shift limits.
