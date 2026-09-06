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

## Current limitations

- Dataset size is still small; the current goal is pipeline validation, not robust generalization.
- Action chunks model a short future horizon but do not encode observation history.
- Training images use one camera and one lighting configuration.
- Failed demonstrations are recorded but are not yet used for corrective learning or DAgger.
