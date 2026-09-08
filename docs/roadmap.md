# 学習ロードマップ

## Phase 0: 評価できる土台

成功条件を「対象物を5回中4回、目標領域へ置ける」のように定義する。観測、ロボット状態、行動、報酬・成功判定のデータ形式を決める。

## Phase 1: シミュレーション制御

MuJoCoで関節制御、順運動学、逆運動学、手先の軌道、衝突回避を学ぶ。ここではAIを使わず、制御の基準実装を作る。

## Phase 2: 視覚認識

シミュレーションカメラから画像を取得し、色・位置・深度を使って対象物の把持目標を生成する。ノイズや照明差も加える。

## Phase 3: データ収集

キーボード、ゲームパッド、またはマウスでアームを操作し、画像と行動のデモを保存する。成功・失敗の両方を記録する。

## Phase 4: 模倣学習

まず行動クローニング、次にDiffusion Policyを実装する。未知の配置への汎化、データ量、推論速度を比較する。

## Phase 5: 言語から操作へ

VLMで「赤いブロック」「左の箱」などを解釈し、対象物と目標を操作ポリシーへ渡す。VLAは既存モデルの推論・微調整から試す。

Phase 5Aでは、赤・紫のblockと緑・黄のtargetを使うstructured goal、RGB-D/IK
pick-and-place expert、4次元goal付きdataset、task-level evaluatorまで実装済み。
Phase 5B/5Cではgoal-conditioned action chunks、held-out object-target combination、semantic
phase conditioningを評価した。Phase 5DではGRU historyとmodel-seed ensembleを評価し、
target geometryが次のbottleneckだと特定した。Phase 5Eのtarget-conditioning ablationsは
closed-loop改善に失敗した。Phase 5Fではgraspとtransport/placeを分離し、expert graspで
place skillを隔離すると24/40まで改善した。一方、learned graspからのobservable handoffは
0/8だった。Phase 5Gのbounded pose residualはoffline誤差を改善したがgrasp rolloutを
40/40から38/40へ悪化させたためrejectした。Phase 5Hではpose perturbationを収集し、実際の
lift outcomeからgrasp success criticを学習した。Fresh graspを通常条件2 seeds合計75/80から
80/80、既知+6 mm Y bias条件34/40から40/40へ改善したが、learned placeまで含むfull taskは26/40で同率
だった。Phase 5Iではcanonical handoffでobject XY variationを大きく減らしたが、inference-onlyは
9/20から7/20、matched/grounded BCは両方0/20となりrejectした。次はabsolute action trajectoryを
再学習せず、analytical placementをpriorとしてrelease pose候補を実際のtask outcomeでrankingした。
Phase 5Jは通常40/40を維持し、−55 mm X biasを34/40から40/40へ改善した。次はbiasを既知として
与えず、camera observationからonline推定した。Phase 5Kはhidden −55 mm X biasでbaseline
37/40に対して40/40を達成し、persistent calibration offsetへの適応を確認した。次はphysics、
appearance、calibrationをrandomizeしたstress benchmarkでfailure envelopeを測る。

## Phase 6: 実機移行

シミュレーションと同じ観測・行動インターフェースを実機へ接続する。低速・低荷重・非常停止を必須にし、最初は人の監視下で実験する。
