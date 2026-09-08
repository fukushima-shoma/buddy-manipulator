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
[docs/phase5.md](docs/phase5.md)にはgoal-conditioned pick-and-placeを記載している。

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

Behavior-cloning baselineと同じ8-step action chunkをconditional diffusionで学習する場合:

```bash
./scripts/run_diffusion_training.sh data/demonstrations \
  --epochs 100 --device mps \
  --output-dir outputs/phase4/diffusion/seed7

./scripts/run_policy_rollout.sh \
  outputs/phase4/diffusion/seed7/diffusion_policy.pt \
  --episodes 30 --seed 404 --headless \
  --output outputs/phase4/diffusion/seed7/rollout_seed404.json
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
モデル変更の仮説・評価結果・採否は`docs/experiments.md`へ記録する。Offline lossだけでは昇格させず、
同じblock配置でのpaired closed-loop successを主指標にする。

現在のrecommended policyは、object-centric 3-model ensembleとvision-guided IK expertを組み
合わせたgated hybridである。中央はlearned ensemble、外周1 cmはRGB-Dで位置推定してexpertへ
routeする。未使用だった3 rollout seedsの合計で、ensemble単体の66/90（73.3%）から
86/90（95.6%）へ改善し、20失敗を回復して既存成功の退行はなかった。

```bash
./scripts/run_recommended_policy.sh --episodes 30 --seed 2204 --headless
```

モデル構造、正規化、評価指標の詳細は
[docs/phase4.md](docs/phase4.md)を参照。

## Phase 5A クイックスタート

赤または紫のblockと、緑または黄色のtargetを指定してpick-and-placeを実行する。

```bash
./scripts/run_goal_demo.sh --object purple --target yellow
./scripts/collect_goal_demos.sh --episodes 20 --seed 905
./scripts/validate_dataset.sh data/goal_demonstrations
```

各frameにはRGB-D・joint state・actionに加えて4次元goal vectorが保存される。現段階では
structured instructionとscripted RGB-D/IK expertを使う。Goal-conditioned BC baselineを
学習・closed-loop評価する場合:

```bash
./scripts/run_training.sh data/goal_demonstrations \
  --epochs 50 --action-horizon 8 --holdout-goal purple:yellow \
  --output-dir outputs/phase5/goal_bc/data80_seed7
./scripts/run_goal_policy_rollout.sh \
  outputs/phase5/goal_bc/data80_seed7/bc_policy.pt \
  --goal purple:yellow --episodes 10 --device mps
```

長いtask内のapproach/grasp/transport/placeを明示するPhase 5C policyは、既存dataを書き換えず
timestampからsemantic phaseを生成して学習する。

```bash
./scripts/run_training.sh data/goal_demonstrations \
  --epochs 50 --action-horizon 8 --holdout-goal purple:yellow \
  --phase-conditioning --device mps \
  --output-dir outputs/phase5/goal_bc/phase_seed7
```

このablationではheld-out offline MAEが0.05180から0.04756へ改善し、held-out rolloutで
初のnon-zero success（1/10）を確認した。ただしrobust policyの基準には未達で、次は
recurrent historyを比較する。

Phase 5Dでは直近8個のjoint stateをGRUでencodeする。History policyはruntimeでも5 Hzの
履歴を保つため、`--execute-chunk-steps`未指定時に1 actionずつreplanする。

```bash
./scripts/run_training.sh data/goal_demonstrations \
  --epochs 50 --action-horizon 8 --history-horizon 8 \
  --use-goal-object-features --holdout-goal purple:yellow --device mps \
  --output-dir outputs/phase5/goal_bc/history8_object_seed7
```

3-model recurrent ensembleのbalanced benchmarkは12/40だったが、green targetは0/20だった。
よってcheckpointはresearch baselineとして保持し、recommended policyには昇格していない。

Phase 5Eではgoal-selected target pixels、target別decoder、shared decoder + target residualを比較した。
Best offline modelでもpaired rolloutは旧ensembleの11/40に対して0/40となり、全variantをrejectした。

Phase 5Fではmonolithic BCをgraspとtransport/placeへ分割した。Goalもobject-only graspと
target-only placeへfactorizeし、RGB-D liftとfinger stateによるobservable handoffを追加した。
Learned graspは0/8でgateを通過できなかったが、RGB-D/IK graspでplace skillを隔離した
locked benchmarkは24/40（60%）まで改善した。これはdiagnostic hybridであり、完全学習policy
としては未昇格。

Phase 5GではRGB-D/IK poseにzero-initialized MLPのbounded XYZ residualを加えた。Held-out
pose errorは1.811 mmから1.646 mmへ改善したが、paired graspはbaseline 40/40に対して
full residual 34/40、25% blend 38/40だった。Full-task二組では45/80対44/80と僅差で、
4 recoveriesに対して3 regressionsが出たためpromoteしていない。次はgeometric centerではなく
on-policy pose perturbationのlift/contact outcomeを教師にする。

Phase 5Hでは60 scenes × 13候補（合計780 grasp attempts）のXY pose perturbationを実行し、
実際のlift successを教師にしたsuccess criticを学習した。Fresh paired graspでは
2つの通常条件seed合計を75/80から80/80へ、既知の+6 mm Y calibration bias条件を
34/40から40/40へ改善した。
一方、既存のlearned place policyまで含むfull taskは26/40で同率（3 recoveries / 3 regressions）
だったため、criticはgrasp componentとして採用するがend-to-end recommended policyは更新しない。

```bash
./scripts/collect_grasp_perturbations.sh \
  --scenes 60 --candidates 13 --maximum-xy-mm 18
./scripts/train_grasp_success_critic.sh \
  data/grasp_perturbations/phase5h_seed3833.npz
./scripts/run_grasp_success_critic.sh \
  outputs/phase5/grasp_success_critic/seed17/grasp_success_critic.pt \
  --controller critic --episodes 40 --seed 3934
```

Phase 5Iではgrasp後のarm/object stateをcanonical poseへ揃えた。XY標準偏差は
約`(5.2, 21.8) mm`から`(1.9, 0.6) mm`へ縮小したが、既存place policyへのinference-only適用は
9/20から7/20へ悪化した。Canonical trajectory 80件を新規収集して再学習しても、通常BCと
explicit object/target groundingの両方が0/20だった。Offline MAEの改善がclosed-loop安定性へ
移らないため、handoff normalizationはablationとして保持し、次はanalytical place trajectory上で
release candidateをoutcome rankingするstructured policyへ進む。

Phase 5Jではanalytical transportを維持し、release XY residualだけをtask outcomeからrankingする
placement success criticを追加した。540 attemptsで学習し、通常条件ではbaseline/criticともに
40/40を維持した。−55 mm X release biasでは34/40から40/40へ改善し、6 failuresを回復して
regressionはなかった。Phase 5のrecommended systemはgrasp critic + placement critic + IK skillsの
structured hierarchyとなる。

Phase 5K removes the known-bias assumption. After each placement, an online estimator compares the
RGB-D observed object position with the calibrated target and subtracts the selected correction
command. On 40 fresh scenes with a hidden −55 mm X offset, the uncorrected baseline reached 37/40,
while online estimation and critic ranking reached 40/40. The estimate starts at zero and uses only
prior outcomes; no simulator object coordinates or injected-bias value are exposed to the policy.

```bash
./scripts/run_structured_goal_policy.sh \
  outputs/phase5/grasp_success_critic/seed17/grasp_success_critic.pt \
  outputs/phase5/placement_success_critic/seed23/placement_success_critic.pt \
  --episodes 40 --seed 4843 --place-bias-x-mm -55 \
  --bias-knowledge online
```

```bash
./scripts/run_hierarchical_goal_policy.sh \
  outputs/phase5/hierarchical/grasp_object_phase_seed7/bc_policy.pt \
  outputs/phase5/hierarchical/place_target_phase_factorized_seed7/bc_policy.pt \
  --grasp-controller expert --episodes 40 --seed 3530
```

詳細は[docs/phase5.md](docs/phase5.md)を参照。
