# Phase 3: 模倣学習用デモデータ収集

## 到達点

ブロック位置をランダム化し、Phase 2の認識・把持パイプラインをexpert policyとして実行する。実行中のRGB、深度、関節状態、行動、物体位置を同期して保存し、データセットを自動検査する。

## データ収集

```bash
./scripts/collect_demos.sh --episodes 3
```

同じseedと空の出力先を使えば、同じブロック配置を再現できる。既存のepisode番号は上書きせず、次の番号から追記する。

```bash
./scripts/collect_demos.sh \
  --episodes 10 \
  --seed 42 \
  --sample-hz 5 \
  --width 160 \
  --height 120
```

## データ構造

各episodeは圧縮済みNumPyファイルとJSON metadataの組で保存する。

| 配列 | Shape | 型 | 内容 |
|---|---|---|---|
| `timestamp` | `(T,)` | float64 | MuJoCo時刻 |
| `rgb` | `(T,H,W,3)` | uint8 | workspace camera画像 |
| `depth` | `(T,H,W)` | float32 | メートル単位の深度 |
| `joint_position` | `(T,6)` | float32 | 4 arm joints + 2 fingers |
| `action` | `(T,6)` | float32 | position actuatorの目標値 |
| `object_position` | `(T,3)` | float32 | 評価用ブロック位置 |

JSONにはschema version、成功ラベル、sample数、各shape、開始位置を保存する。`object_position`は学習入力ではなく、評価とデバッグに使う。

## 検証

```bash
source .venv/bin/activate
buddy-validate-dataset data/demonstrations
```

validatorは必須配列、sample数、dtype、関節・行動次元、NaN/Inf、metadata整合性を検査する。

## 設計上の意図

最初はscripted expertを使い、成功軌道を安定して集める。これによりPhase 4の行動クローニングが失敗したとき、原因を「データ収集」と「学習」に分離できる。今後、同じ`EpisodeRecorder`へkeyboard/gamepad teleoperationを接続し、人間由来の多様な軌道と失敗例を追加する。
