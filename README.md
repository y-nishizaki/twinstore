# TwinStore - 類似店舗売上予測パッケージ

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen.svg)](tests/)

新規出店後の限られた売上データから年間売上を高精度で予測するPythonパッケージです。類似店舗（Twin Store）マッチング技術とDTW（動的時間伸縮法）を活用し、従来手法と比較して20%以上の精度向上を実現します。

## 📑 目次

- [🚀 クイックスタート](#-クイックスタート)
- [📦 インストール](#-インストール)
- [🎯 主な機能](#-主な機能)
- [📊 基本的な使い方](#-基本的な使い方)
- [📈 高度な機能](#-高度な機能)
- [📋 データ形式](#-データ形式)
- [🔄 パイプライン処理](#-パイプライン処理)
- [📊 可視化機能](#-可視化機能)
- [🛠 API リファレンス](#-api-リファレンス)
- [🔧 トラブルシューティング](#-トラブルシューティング)
- [📚 詳細ドキュメント](#-詳細ドキュメント)

## 🚀 クイックスタート

### 最小限のコードで予測を実行

```python
from twinstore import SalesPredictor

# 1. 過去データ（店舗名: 売上配列）
historical_data = {
    'store_001': [100000, 105000, 98000, 110000, 120000],  # 5日分
    'store_002': [95000, 100000, 93000, 105000, 115000],
}

# 2. 新規店舗の売上（最低3日分）
new_store_sales = [98000, 102000, 96000]  # 3日分

# 3. 予測器を初期化して予測
predictor = SalesPredictor()
predictor.fit(historical_data)

# 4. 予測実行
result = predictor.predict(new_store_sales)

# 5. 結果確認
print(f"予測年間売上: {result.prediction:,.0f}円")
print(f"信頼区間: {result.lower_bound:,.0f} - {result.upper_bound:,.0f}円")
```

## 📦 インストール

```bash
pip install twinstore
```

### 依存関係

- Python 3.8以上
- 主要ライブラリ: numpy, pandas, scikit-learn, dtaidistance, matplotlib, plotly

詳細な依存関係は[requirements.txt](requirements.txt)を参照してください。

## 🎯 主な機能

### コア機能
- **DTW（動的時間伸縮法）による時系列マッチング** - 成長速度の違いを吸収
- **類似店舗の自動検索** - 上位k個の類似店舗を高速抽出
- **信頼区間付き予測** - 予測の不確実性を定量化
- **多様な正規化手法** - Z-score、Min-Max、初日比率正規化

### データ管理
- **自動データ品質チェック** - 欠損値、異常値、データ完全性の評価
- **リアルタイム異常検知** - 異常パターンの分類と推奨アクション
- **柔軟なデータ形式対応** - DataFrame、NumPy配列、辞書、リスト

### 分析・レポート
- **予測根拠の説明生成** - 自然言語での説明（日本語/英語）
- **What-ifシミュレーション** - シナリオ分析と感度分析
- **自動レポート生成** - PDF/Excel/PowerPoint形式

### 高度な機能
- **マッチング期間の自動最適化** - 7日～90日で最適期間を探索
- **インタラクティブ可視化** - 売上推移の比較ダッシュボード
- **バッチ処理** - 複数店舗の一括予測

## 📊 基本的な使い方

### pandas DataFrameを使った例

```python
import pandas as pd
from twinstore import SalesPredictor

# データ準備
historical_df = pd.DataFrame({
    'store_cd': ['A001'] * 30 + ['A002'] * 30,
    'date': pd.date_range('2024-01-01', periods=30).tolist() * 2,
    'sales': [100000 + i*1000 for i in range(30)] * 2
})

new_store_df = pd.DataFrame({
    'date': pd.date_range('2024-02-01', periods=7),
    'sales': [95000, 98000, 102000, 96000, 101000, 105000, 99000]
})

# 予測実行
predictor = SalesPredictor(preset='retail')
predictor.fit(historical_df)
result = predictor.predict(new_store_df['sales'].values)
```

### 店舗属性を考慮した予測

```python
# 店舗属性データ
store_attributes = {
    'store_001': {'type': 'urban', 'area': 150, 'parking': True},
    'store_002': {'type': 'suburban', 'area': 200, 'parking': True},
}

# 新規店舗の属性
new_store_attr = {'type': 'urban', 'area': 180, 'parking': False}

# 属性を考慮した予測
result = predictor.predict(
    new_store_sales,
    store_attributes=new_store_attr,
    filters={'type': 'urban'}  # 都市型店舗のみで予測
)
```

## 📈 高度な機能

### パイプライン処理

```python
from twinstore import PredictionPipeline, PipelineConfig

# パイプライン設定
config = PipelineConfig(
    validate_data=True,
    handle_missing=True,
    detect_anomalies=True,
    optimize_period=True,
    generate_report=True
)

# パイプライン実行
pipeline = PredictionPipeline(config)
pipeline_result = pipeline.run(
    historical_data=historical_data,
    new_store_sales=new_store_sales,
    store_name="新宿西口店"
)
```

### What-ifシミュレーション

```python
# シミュレーター作成
simulator = predictor.create_simulator()

# シナリオ分析
scenarios = simulator.analyze_scenarios(
    base_result=result,
    scenarios={
        'optimistic': {'month2_growth': 1.2},
        'standard': {'month2_growth': 1.0},
        'pessimistic': {'month2_growth': 0.8}
    }
)

# 結果の可視化
simulator.plot_scenarios(scenarios)
```

### 期間最適化

```python
# 最適なマッチング期間を探索
optimizer = predictor.create_period_optimizer()
optimal_period = optimizer.find_optimal_period(
    new_store_sales=new_store_sales,
    cv_folds=5,
    period_range=(7, 30)
)

print(f"推奨マッチング期間: {optimal_period.recommended_days}日")
print(f"期待精度（MAPE）: {optimal_period.expected_mape:.1f}%")
```

## 📋 データ形式

### 対応データ形式

TwinStoreは以下のデータ形式に対応しています：

1. **pandas DataFrame**
```python
df = pd.DataFrame({
    'store_cd': ['A001', 'A001', 'A002', 'A002'],
    'date': pd.date_range('2024-01-01', periods=2).tolist() * 2,
    'sales': [100000, 105000, 95000, 98000]
})
```

2. **NumPy配列**
```python
sales_array = np.array([100000, 105000, 98000, 110000])
```

3. **Python辞書**
```python
sales_dict = {
    'store_001': [100000, 105000, 98000],
    'store_002': [95000, 100000, 93000]
}
```

4. **リスト/タプル**
```python
sales_list = [100000, 105000, 98000, 110000]
```

### データ要件

- **最小データ量**: 予測対象は3日分以上
- **推奨データ量**: 7日以上で精度向上
- **データ型**: 数値型（整数または浮動小数点）
- **順序**: 時系列順（古い→新しい）

## 🔄 パイプライン処理

### 処理フロー

1. **データ検証** - 形式チェック、最小要件確認
2. **前処理** - 欠損値補完、外れ値処理
3. **品質チェック** - データ品質スコアリング
4. **異常検知** - リアルタイム異常パターン検出
5. **期間最適化** - 最適マッチング期間の探索
6. **予測実行** - 類似店舗ベース予測
7. **説明生成** - 予測根拠の自然言語説明
8. **結果保存** - 複数形式での出力

### バッチ処理

```python
# 複数店舗の一括予測
stores_to_predict = {
    'store_A': [100000, 105000, 98000],
    'store_B': [95000, 100000, 93000],
    'store_C': [110000, 115000, 108000]
}

batch_results = pipeline.run_batch(
    historical_data=historical_data,
    new_stores_data=stores_to_predict,
    parallel=True
)
```

## 📊 可視化機能

### 売上推移の比較

```python
from twinstore.visualization import SalesAlignmentVisualizer

# 可視化ツール作成
visualizer = SalesAlignmentVisualizer()

# 開店日基準で売上を比較
fig = visualizer.plot_aligned_sales(
    historical_data=historical_data,
    new_store_sales=new_store_sales,
    normalize=True,
    title="新規店舗 vs 類似店舗（開店日基準）"
)
```

### インタラクティブダッシュボード

```python
# 複合ダッシュボード作成
dashboard = visualizer.create_interactive_dashboard(
    store_groups={
        '都市型': ['store_001', 'store_003'],
        '郊外型': ['store_002', 'store_004'],
        '新規店舗': ['new_store']
    },
    include_growth_analysis=True,
    include_similarity_matrix=True
)

dashboard.show()
```

## 🛠 API リファレンス

### SalesPredictor

主要なクラスとメソッド：

```python
predictor = SalesPredictor(
    similarity_method='dtw',     # 類似性計算手法
    normalization='zscore',      # 正規化手法
    preset='retail'              # 業態別プリセット
)

# 学習
predictor.fit(historical_data, store_attributes=None)

# 予測
result = predictor.predict(
    new_store_sales,
    n_similar=5,                 # 使用する類似店舗数
    confidence_level=0.95,       # 信頼区間レベル
    filters=None                 # フィルタ条件
)

# 精度評価
accuracy = predictor.evaluate(test_data, metrics=['mape', 'rmse'])
```

### PredictionResult

予測結果オブジェクト：

```python
result.prediction          # 予測値
result.lower_bound        # 信頼区間下限
result.upper_bound        # 信頼区間上限
result.confidence_score   # 予測信頼度（0-1）
result.similar_stores     # 類似店舗リスト
result.explanation        # 予測根拠の説明
```

## 🔧 トラブルシューティング

### よくある問題と解決方法

1. **データ不足エラー**
```python
# エラー: "Insufficient data: minimum 3 days required"
# 解決: 最低3日分のデータを用意
new_store_sales = [100000, 105000, 98000]  # OK
```

2. **欠損値の処理**
```python
# 自動補完を有効化
config = PipelineConfig(handle_missing=True)
pipeline = PredictionPipeline(config)
```

3. **メモリ不足**
```python
# バッチサイズを調整
predictor.predict_batch(data, batch_size=100)
```

### デバッグモード

```python
# 詳細ログを有効化
import logging
logging.basicConfig(level=logging.DEBUG)

predictor = SalesPredictor(debug=True)
```

## 📚 詳細ドキュメント

より詳しい情報については、以下のドキュメントを参照してください：

- [パイプライン機能の詳細](docs/pipeline.md)
- [データ形式ガイド](docs/data_format.md)
- [可視化機能ガイド](docs/visualization.md)
- [API詳細リファレンス](docs/api_reference.md)
- [ベストプラクティス](docs/best_practices.md)

## 🤝 貢献

プルリクエストを歓迎します。大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## 📄 ライセンス

このプロジェクトは[MITライセンス](LICENSE)の下で公開されています。

## 📞 サポート

- Issues: [GitHub Issues](https://github.com/yourname/twinstore/issues)
- Email: support@twinstore.example.com
- Documentation: [https://twinstore.readthedocs.io](https://twinstore.readthedocs.io)