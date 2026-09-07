# Phase 4: RGB-D Behavior Cloning

## 到達点

Phase 3で記録したRGB-D画像と関節状態から、次の6次元position actuator targetを予測する
behavior-cloning policyを学習する。小さなCNNをbaselineにすることで、今後のDiffusion
Policyとデータ効率、予測誤差、closed-loop成功率を比較できるようにする。

## Observationとaction

1 sampleの入力と教師信号は次のとおり。

| 要素 | Shape | 前処理 |
|---|---|---|
| RGB | `(3,H,W)` | `0..1`へscale |
| Depth | `(1,H,W)` | train dataの平均・標準偏差でnormalize |
| Joint position | `(6,)` | actuatorごとにnormalize |
| Target action | `(6,)` | actuatorごとにnormalize |

RGB-Dは3層CNNでencodeし、正規化したjoint positionと結合する。MLPの出力は、4 arm
jointsと2 finger jointsのabsolute position targetである。`object_position`は教師入力に
使わず、シミュレーション評価専用に残す。

CNN feature mapには、PyTorch標準と同じwindowを明示的に平均するadaptive poolingを使う。
これはApple SiliconのMPS backendが、割り切れないinput/output sizeの標準adaptive poolingを
未実装としているためである。追加の学習parameterはなく、CPUで学習した既存checkpointと
同じ計算をMPS上でも再現できる。

## Training

PyTorchを含む環境を初回だけ更新する。

```bash
./scripts/setup.sh
```

収集済みデータで学習する。

```bash
./scripts/run_training.sh data/demonstrations --epochs 30
```

Single-step policyがpre-graspで停止する場合は、未来8 step（5 Hzで1.6秒）のaction chunkを
予測する。

```bash
./scripts/run_training.sh data/demonstrations \
  --epochs 30 \
  --action-horizon 8 \
  --output-dir outputs/phase4/chunked
```

Action chunkは同じ観測から複数のfuture actionを予測するため、「接近を続ける」「下降する」
「gripperを閉じる」が混在するtemporal ambiguityを平均化して停止する問題を軽減する。

defaultでは`success=true`のepisodeだけを使用する。失敗軌道は「何をすべきだったか」という
corrective labelを含まないため、最初のbehavior cloningには混ぜない。研究比較のため明示的に
含める場合だけ`--include-failures`を指定する。

データ分割はframe単位ではなくepisode単位で行う。同じ軌道の連続frameがtrainとvalidationへ
分かれるdata leakageを防ぐためである。乱数seedはdefaultで7に固定されている。

成果物は`outputs/phase4/`へ保存される。

- `bc_policy.pt`: model weights、normalization、train/validation episode ID
- `training_metrics.json`: epochごとのlossとvalidation metrics

## Offline evaluation

checkpointに記録されたvalidation episodesで再評価する。

```bash
./scripts/evaluate_policy.sh \
  outputs/phase4/bc_policy.pt \
  data/demonstrations
```

`normalized MSE`は学習空間での比較用、`action MAE`は元のactuator単位での解釈用である。
offline誤差が小さくても、誤差蓄積によりclosed-loop graspが失敗する場合がある。そのため次の
buildではpolicyをMuJoCo内でrolloutし、未知のblock配置に対するgrasp成功率を測定する。

## Closed-loop rollout

学習時とは異なるseedでblock位置をrandomizeし、policy自身の予測actionをMuJoCoへ戻す。

```bash
./scripts/run_policy_rollout.sh \
  outputs/phase4/chunked/bc_policy.pt \
  --episodes 5 \
  --headless
```

GUIで1 episodeを観察する場合:

```bash
./scripts/run_policy_rollout.sh \
  outputs/phase4/chunked/bc_policy.pt \
  --episodes 1
```

Policyは5 HzでRGB-Dとjoint stateを再観測する。予測したabsolute actionはactuator rangeへ
clampし、defaultでは1 control stepあたりarmを0.20 rad、fingerを0.01 mまでに制限する。
5 cm以上のliftとfinger contactが0.4秒継続した場合に成功と判定する。

各episodeのblock位置、最終lift、最大lift、contact、step数、policyが要求したaction、実際に
rate limit後に適用したactionと全体success rateは
`outputs/phase4/rollout_results.json`へ保存される。seed、control rate、time limit、action
rate limitはCLI optionで変更できる。

`--execute-chunk-steps`で再観測までに実行するaction数を変更できる。defaultの0はcheckpointの
全horizonを使う。8-step modelで2 stepだけ実行するとpre-grasp stallが再発したため、現在は
full chunkをbaselineとする。

## Baseline experiment

Seed 202で40本のscripted expert demonstrationsを追加し、合計54成功episode（1,475 frames）
で8-step modelを学習した。Seed 101の未見20配置で次の結果を得た。

| Policy | Training episodes | Closed-loop result |
|---|---:|---:|
| Single-step BC | 14 | 0 / 5 (0%) |
| 8-step chunk BC | 14 | 2 / 5 (40%) |
| 8-step chunk BC | 54 | 13 / 20 (65%) |

54-episode modelのheld-out chunk MAEは0.0274、成功時のliftは約6.4〜9.3 cmだった。失敗7件の
うち4件がsample rangeのlow-X edge、2件がhigh-Y付近にあり、boundary coverageの強化が次の
data collection targetである。ただし20 trialsなので、この空間的傾向は仮説として扱う。

## Failure-driven dataset aggregation

Rollout reportから失敗配置だけを抽出し、その位置と周辺のbounded jitterへexpert policyを
適用する。失敗trajectory自体を教師データとして模倣するのではなく、失敗したtask condition
に対するcorrect expert demonstrationを追加するDAgger-inspiredな方法である。

```bash
./scripts/collect_failure_demos.sh \
  outputs/phase4/chunked/rollout_benchmark.json \
  --repeats 5 \
  --jitter 0.006 \
  --seed 303
```

収集episodeは`source=failure_replay`としてmetadataへ記録される。同じ評価配置でのdata
leakageを避けるため、改善比較には収集元と異なるseedを使う。

```bash
# Before aggregation: untouched evaluation set
./scripts/run_policy_rollout.sh \
  outputs/phase4/chunked/bc_policy.pt \
  --episodes 30 --seed 404 --headless \
  --output outputs/phase4/chunked/rollout_seed404_before.json

# Train after aggregation, then evaluate the identical positions
./scripts/run_training.sh data/demonstrations \
  --epochs 30 --action-horizon 8 \
  --output-dir outputs/phase4/failure_replay
./scripts/run_policy_rollout.sh \
  outputs/phase4/failure_replay/bc_policy.pt \
  --episodes 30 --seed 404 --headless \
  --output outputs/phase4/failure_replay/rollout_seed404_after.json

./scripts/compare_rollouts.sh \
  outputs/phase4/chunked/rollout_seed404_before.json \
  outputs/phase4/failure_replay/rollout_seed404_after.json
```

7 failuresから35 targeted episodesを追加した実験では、held-out MSEは0.1351から0.1091へ
改善した一方、paired closed-loop successは15/30（50%）から13/30（43%）となり、task-level
improvementは確認できなかった。Naive replayによる分布の偏りとtraining varianceを分離する
には、source-balanced sampling、複数training seeds、より多いpaired trialsが必要である。

### Source-balanced replay

`failure_replay`がdataset内で増えすぎる場合、`--failure-replay-fraction`で1 epoch中の期待sampling
比率を指定できる。samplingはframe単位・replacementありで行い、epochあたりのsample数は自然な
samplingと同じに保つ。validation setにはweightを適用しない。

```bash
./scripts/run_training.sh data/demonstrations \
  --epochs 30 --action-horizon 8 \
  --failure-replay-fraction 0.2 \
  --output-dir outputs/phase4/balanced

./scripts/run_policy_rollout.sh \
  outputs/phase4/balanced/bc_policy.pt \
  --episodes 30 --seed 404 --headless \
  --output outputs/phase4/balanced/rollout_seed404.json

./scripts/compare_rollouts.sh \
  outputs/phase4/chunked/rollout_seed404_before.json \
  outputs/phase4/balanced/rollout_seed404.json \
  --output outputs/phase4/balanced/paired_comparison.json
```

89成功episodeを使い、failure replayを20%へ抑えた8-step modelは同じseed 404の30配置で
16/30（53%）に達した。54-episode baselineの15/30（50%）に対して、5失敗を回復し4成功が
退行したため、net improvementは1 episode（+3 percentage points）である。候補checkpointは
`outputs/phase4/balanced/bc_policy.pt`だが、差が小さいため複数training seedとより多いtrialで
再現性を確認するまでは、54-episode modelも比較baselineとして保持する。

### Multi-seed robustness benchmark

Training seedによるmodel varianceと、block配置によるevaluation varianceを同時に測るrunnerを
用意した。各training seedのcheckpointを独立に作り、全modelを同じrollout seed・同じblock
配置でbaselineとpaired比較する。中断後は同じ設定で`--reuse-existing`を付けると既存成果物を
再利用できる。

```bash
./scripts/run_multiseed_benchmark.sh data/demonstrations \
  --baseline outputs/phase4/chunked/bc_policy.pt \
  --train-seeds 7,17,27 \
  --rollout-seeds 404,505,606 \
  --episodes 30 \
  --epochs 30 \
  --action-horizon 8 \
  --failure-replay-fraction 0.2 \
  --device mps \
  --output-dir outputs/phase4/multiseed
```

上記3×3 experimentでは、baselineは3 rollout seedsで47/90（52.2%）だった。

| Training seed | Balanced policy | Baselineとの差 |
|---:|---:|---:|
| 7 | 48/90（53.3%） | +1.1 points |
| 17 | 56/90（62.2%） | +10.0 points |
| 27 | 38/90（42.2%） | -10.0 points |

Balanced policyのtraining-seed平均は52.6%、sample standard deviationは10.0 points、baseline
との差の平均は+0.4 pointsだった。したがって、20% source balancingに安定した改善効果はまだ
確認できない。Seed 17を評価結果から選ぶとtest-set selectionになるためrecommended modelへは
昇格せず、引き続き54-episodeの`outputs/phase4/chunked/bc_policy.pt`をbaselineとして保持する。
Aggregateと各paired resultは`outputs/phase4/multiseed/summary.json`に保存される。

### Training seed factor ablation

従来の`--seed`はdataset split、model initialization、DataLoader/replay samplerの3要因を同時に
変更していた。原因を分離するため、training CLIへ`--split-seed`、`--model-seed`、
`--sampler-seed`を追加した。指定しなければ3つとも従来どおり`--seed`を使う。

Multi-seed runnerでは`--vary-seed`で1要因だけを変更できる。例えばmodel initializationだけを
比較する場合は次のように実行する。

```bash
./scripts/run_multiseed_benchmark.sh data/demonstrations \
  --baseline outputs/phase4/chunked/bc_policy.pt \
  --train-seeds 7,17,27 \
  --vary-seed model --fixed-seed 7 \
  --rollout-seeds 404,505,606 \
  --episodes 30 --epochs 30 --action-horizon 8 \
  --failure-replay-fraction 0.2 --device mps \
  --output-dir outputs/phase4/seed_ablation/model
```

Splitとsamplerについても`--vary-seed split`または`--vary-seed sampler`で同じ実験を行った。
Seed 7のrunとbaseline reportsは共通成果物を再利用し、各要因でseed 17・27だけを再学習した。

| Varied factor | Seed results | Mean | Sample std | Range |
|---|---|---:|---:|---:|
| Model initialization | 53.3%, 53.3%, 65.6% | 57.4% | 7.1 points | 12.2 points |
| Replay sampler | 53.3%, 34.4%, 27.8% | 38.5% | 13.3 points | 25.6 points |
| Dataset split | 53.3%, 26.7%, 57.8% | 45.9% | 16.8 points | 31.1 points |

この条件ではdataset splitが最大、replacement replay samplerが次に大きいvariance sourceだった。
Model initializationにも無視できない影響がある。これはseed 7を基準とした3点のone-factor-at-a-time
experimentであり、要因間interactionを含むformal variance decompositionではない。

次のmodel improvementでは、object-position coverageを保つspatially stratified splitと、毎epochの
source比率を保ちながらsample重複を抑えるdeterministic balanced samplerを優先する。各factorの
結果は`outputs/phase4/seed_ablation/{model,sampler,split}/summary.json`に保存される。

## Current limitations

- Dataset size is still small; the current goal is pipeline validation, not robust generalization.
- Action chunks model a short future horizon but do not encode observation history.
- Training images use one camera and one lighting configuration.
- Failure conditions guide expert recollection, but policy-visited states are not yet relabeled as in full DAgger.
- Source-balanced models vary by about 10 percentage points across the three tested training seeds.
- Random episode splits and replacement replay sampling are the largest measured variance sources.
