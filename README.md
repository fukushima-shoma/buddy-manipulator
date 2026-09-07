# Buddy Manipulator

会話で指示できるロボットアームを、シミュレーションから実機へ段階的に発展させるプロジェクト。

## 目的

Buddyで培ったROS 2・カメラ認識・会話・安全停止の知識を、物体操作とロボット学習へ広げる。

## 開発方針

最初はハードウェアを使わず、シミュレーション上で「物体を見つける → つかむ → 指定場所へ置く」を実現する。

## フェーズとハードウェア

| Phase | 内容 | ハードウェア |
|---|---|---|
| 0 | 開発環境・データ形式・評価指標 | 不要 |
| 1 | MuJoCoでアームを動かす、順逆運動学 | 不要 |
| 2 | シミュレーションカメラと物体認識 | 不要 |
| 3 | テレオペレーションでデモデータ収集 | 不要 |
| 4 | 行動クローニング、Diffusion Policy | 不要 |
| 5 | 言語指示と視覚・操作ポリシーの統合 | 不要 |
| 6 | Sim-to-Real検証 | ここから必要 |
| 7 | 実機データで微調整・安全評価 | 必要 |

### 最初に必要な実機

学習内容を確認するだけならPhase 5まで不要。Phase 6で、まずは6軸または4〜6軸の小型アーム、グリッパー、USBカメラ、非常停止手段を用意する。高価なアームは不要で、最初は安全な低速・低荷重の機体を使う。

## 技術スタック（初期案）

- Python / PyTorch
- ROS 2 / MoveIt 2
- MuJoCo（最初のシミュレーター）
- OpenCV
- imitation learning / Diffusion Policy
- 将来的にVLM・VLA、Isaac Sim、Sim-to-Real

## 最初のマイルストーン

1. MuJoCoでアームと箱・ブロックを表示する
2. 関節目標を与えてアームを動かす
3. 物体の位置から把持姿勢を作る
4. テレオペ操作を記録する
5. 記録データから行動を予測するモデルを学習する

全体計画は [docs/roadmap.md](docs/roadmap.md)、各フェーズの演習は
[docs/phase1.md](docs/phase1.md)、[docs/phase2.md](docs/phase2.md)、
[docs/phase3.md](docs/phase3.md)、[docs/phase4.md](docs/phase4.md) を参照。

## Phase 1 クイックスタート

Python 3.9以上が必要。

```bash
./scripts/setup.sh
```

GUIで4自由度アームのデモを実行する。

```bash
./scripts/run_demo.sh
```

GUIモードでは、ターミナルに`MuJoCo viewer started`と表示され、デモ終了後も
ウィンドウが開いたままになる。MuJoCoウィンドウを閉じるとコマンドが終了する。

画面を開かずに動作確認する場合:

```bash
./scripts/run_demo.sh --headless
./scripts/run_tests.sh
```

### Apple SiliconとRosetta

Apple Silicon MacではMuJoCoをnative `arm64` Pythonで実行する必要がある。
TerminalやIDEがRosetta (`x86_64`) で動いていても、`setup.sh`と`run_demo.sh`は
Pythonプロセスを`arm64`で起動する。GUI実行時はmacOSで必要な`mjpython`へ自動的に
切り替え、`--headless`では通常のPythonを使う。直接`buddy-arm-demo`を実行すると、
親シェルによっては`x86_64`で起動されるか、GUI初期化に失敗するため、上記スクリプトを使う。

Phase 1では以下を実装済み。

- プリミティブ形状だけで構成した4自由度アームと2指グリッパー
- 関節位置制御とキーフレーム動作
- 順運動学・解析的逆運動学
- 赤いブロックと緑の目標領域を含む卓上シーン
- GUIなしで実行できるシミュレーションテスト

## Phase 2 クイックスタート

RGB-Dカメラで赤いブロックを検出し、3D位置とアプローチ姿勢を計算する。

```bash
./scripts/run_vision_demo.sh
```

検出画像とJSON結果は`outputs/phase2/`へ保存される。

認識した座標へアームを動かし、ブロックを把持して持ち上げる。

```bash
./scripts/run_grasp_demo.sh
```

GUIなしで成功判定まで確認する場合:

```bash
./scripts/run_grasp_demo.sh --headless
```

## Phase 3 クイックスタート

ランダムなブロック配置でexpert graspを実行し、模倣学習用データを収集する。

```bash
./scripts/collect_demos.sh --episodes 3
./scripts/validate_dataset.sh data/demonstrations
```

各episodeにはRGB、深度、6関節の状態、6次元action、物体位置、成功ラベルが含まれる。

Keyboard teleoperationで人間操作の成功・失敗episodeを記録する。

```bash
./scripts/run_teleop.sh
```

Controls: `W/S` = X、`A/D` = Y、`R/F` = Z、`O/C` = gripper、
`Enter` = 保存、`Esc` = 中断して失敗例として保存。詳細は
[docs/phase3.md](docs/phase3.md)を参照。

Apple Silicon上でRosetta Terminalを使っている場合、`.venv`をactivateしただけでは
Pythonの実行architectureは切り替わらない。NumPy、MuJoCo、将来のPyTorchを使う
コマンドは`./scripts/run_python.sh`または各専用launcherから実行する。

## Phase 4 クイックスタート

成功episodeをtrain/validationへepisode単位で分割し、RGB-D画像と関節状態から
6次元actionを予測するbehavior-cloning policyを学習する。

```bash
./scripts/run_training.sh data/demonstrations --epochs 30
./scripts/evaluate_policy.sh outputs/phase4/bc_policy.pt data/demonstrations

./scripts/run_training.sh data/demonstrations \
  --epochs 30 \
  --action-horizon 8 \
  --output-dir outputs/phase4/chunked
./scripts/run_policy_rollout.sh \
  outputs/phase4/chunked/bc_policy.pt \
  --episodes 20 \
  --headless
```

checkpointとtraining metricsは`outputs/phase4/`へ保存される。失敗episodeはdefaultでは
学習から除外する。closed-loop rolloutの結果も同じdirectoryへ保存される。
失敗配置からexpert dataを再収集し、paired rolloutを比較するfailure-driven loopも利用できる。
targeted dataが増えた場合は`--failure-replay-fraction 0.2`でsource-balanced samplingを行える。
`./scripts/run_multiseed_benchmark.sh`は複数training/rollout seedをpaired評価し、modelの平均成功率と
seed間のばらつきを`summary.json`へ保存する。`--vary-seed split|model|sampler`を使うと、他の
seed要因を固定したcontrolled ablationも実行できる。`--split-strategy spatial`および
`--source-sampling minimal-replacement|without-replacement`は研究比較用optionであり、現時点の
recommended settingではない。
モデル構造、正規化、評価指標の詳細は
[docs/phase4.md](docs/phase4.md)を参照。
