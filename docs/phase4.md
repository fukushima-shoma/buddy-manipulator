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

## Current limitations

- Dataset size is still small; the current goal is pipeline validation, not robust generalization.
- The policy predicts one action per frame and has no temporal history.
- Training images use one camera and one lighting configuration.
- Failed demonstrations are recorded but are not yet used for corrective learning or DAgger.
