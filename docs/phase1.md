# Phase 1: MuJoCoでアームを制御する

## 到達点

Phase 1の実装では、赤いブロックへ手先を近づけ、グリッパーを閉じてホーム姿勢へ戻す一連の関節位置制御を実行する。把持の成功はまだ保証せず、Phase 2以降で視覚と把持計画を追加する。

## モデル

`src/buddy_manipulator/models/arm.xml`は、次の関節を持つ。

| 関節 | 運動 |
|---|---|
| `base_yaw` | テーブル面内で旋回 |
| `shoulder_joint` | 上腕を上下 |
| `elbow_joint` | 前腕を曲げる |
| `wrist_joint` | 手先姿勢を調整 |
| 左右のfinger joint | グリッパーを開閉 |

モデルはMuJoCo標準のプリミティブ形状だけで構成しているため、CADや外部メッシュは不要。

## 順運動学

`forward_kinematics`は4つの関節角から手先の`x, y, z, pitch`を計算する。まず既知の関節角を入力し、MuJoCo画面上の姿勢との対応を確認する。

## 逆運動学

`inverse_kinematics`は目標の`x, y, z, pitch`から関節角を求める。現在はbase yawと垂直平面内の2リンク問題を解析的に解き、wristで目標pitchを合わせる。到達範囲外は`UnreachableTargetError`になる。

## 試すこと

1. `demo.py`の目標座標を少しずつ変更する
2. `elbow_up=True`で肘の解がどう変わるか比べる
3. XMLの`kp`と`damping`を変更し、振動と追従性を観察する
4. 関節可動域の外や遠すぎる目標を与えて安全な失敗を確認する

## Phase 2への課題

- シミュレーションカメラからRGB・深度画像を取得する
- 赤いブロックの位置を画像から推定する
- 手先を対象物の真上へ移動するapproach姿勢を生成する
- 接触判定を使って把持成功を評価する
