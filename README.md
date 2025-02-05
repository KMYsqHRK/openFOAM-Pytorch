# OpenFOAM Surrogate Model Generator

このプロジェクトは、OpenFOAMのシミュレーションデータを使用してサロゲートモデルを生成・学習するためのPythonプログラム群です。キャビティ流れの計算を自動化し、その結果から機械学習モデルを構築します。
現状、一列のテンソルにして学習を行っていますが、これは適切ではありません。空間畳み込みなどを行うことがこれから推奨されます。

## プログラムの構成

### 1. ExcuteFoam.py
OpenFOAMのシミュレーションを実行するためのメインスクリプトです。主な機能：
- テンプレートケースから新しいケースを生成
- 壁面速度の自動設定
- OpenFOAMの実行（blockMeshとrhoPimpleFoam）
- 複数のケースを連続して生成・実行

### 2. OrganizeData.py
OpenFOAMの出力データを機械学習用のテンソルデータに変換します。主な機能：
- 速度場と圧力場のデータ抽出
- 壁面速度の取得
- PyTorchテンソル形式への変換
- データの保存（cavity_data.pt）

### 3. TrainSurrogateModel.py
サロゲートモデルの学習を行います。主な特徴：
- PyTorchベースのニューラルネットワーク実装
- カスタムデータセットとデータローダーの実装
- 訓練プロセスの可視化
- モデルの保存と評価

### 4. ComparePredictAndResult.py
学習したモデルを使用して予測を行い、結果を可視化します。

### 5. VisualizeData.py
生成されたデータの可視化と分析を行います。機能：
- テンソルデータの読み込みと可視化
- 基本統計量の計算と表示
- 複数のデータ成分の比較プロット

## 使用方法

1. OpenFOAM環境の準備
```bash
# OpenFOAM環境を有効化
source /path/to/openfoam/etc/bashrc
```

2. ケースの生成と実行
```python
python ExcuteFoam.py
```

3. データの整理と変換
```python
python OrganizeData.py
```

4. サロゲートモデルの学習
```python
python TrainSurrogateModel.py
```

5. 予測と結果の可視化
```python
python ComparePredictAndResult.py
python VisualizeData.py
```

## 技術的な詳細

### サロゲートモデルのアーキテクチャ
- 入力層: 1ノード（壁面速度）
- 隠れ層: 64ノード → 128ノード
- 出力層: 400ノード（圧力分布）
- 活性化関数: ReLU

### データの前処理
- 圧力データは100,000 Paのオフセットを除去
- 壁面速度は0.5から10.0 m/sの範囲で生成
- データは自動的にGPU/CPU環境に対応

### 学習設定
- 最適化手法: Adam
- 学習率: 0.001
- エポック数: 1000
- バッチサイズ: 4
- 損失関数: MSE

## 注意事項

- このプログラム群を実行するには、OpenFOAMがインストールされている必要があります
- GPUが利用可能な場合は自動的にGPUを使用します
- 大量のケースを生成する場合はディスク容量に注意してください
- テンプレートケースのパスは環境に応じて適切に設定してください
