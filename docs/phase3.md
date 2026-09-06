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
./scripts/validate_dataset.sh data/demonstrations
```

validatorは必須配列、sample数、dtype、関節・行動次元、NaN/Inf、metadata整合性を検査する。

Apple Siliconではlauncherがnative `arm64` Pythonを強制する。Rosetta Terminalで
`.venv`をactivateして直接`buddy-validate-dataset`を実行すると、arm64版NumPyを
`x86_64` Pythonから読み込むため失敗する。

## Keyboard teleoperation

人間が操作した成功・失敗episodeを記録する。

```bash
./scripts/run_teleop.sh
```

MuJoCoウィンドウをクリックしてfocusし、次のキーを使う。1回の移動量は1 cm。

| Key | Action |
|---|---|
| `W` / `S` | X方向へ前進 / 後退 |
| `A` / `D` | Y方向へ左 / 右 |
| `R` / `F` | 上昇 / 下降 |
| `O` / `C` | gripperを開く / 閉じる |
| `Enter` | episodeを完了して保存 |
| `Esc` | 中断し、失敗episodeとして保存 |

開始時は認識したブロックの10 cm上に手先がある。基本操作は`F`を約10回、`C`、
`R`を8〜10回、`Enter`。位置がずれた場合は`W/S/A/D`で補正する。

成功条件はscripted expertと同じく、5 cm以上持ち上げ、かつ最終状態でfinger contactが
あること。`source=teleop`と`termination=completed|aborted|window_closed|timeout`が
JSON metadataへ記録されるため、失敗理由を後から分類できる。

## 設計上の意図

最初はscripted expertを使い、成功軌道を安定して集める。これによりPhase 4の行動クローニングが失敗したとき、原因を「データ収集」と「学習」に分離できる。同じ`EpisodeRecorder`をkeyboard teleoperationでも利用し、人間由来の多様な軌道と失敗例を追加する。
