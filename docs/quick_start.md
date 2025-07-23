# TwinStore クイックスタートガイド

## 5分で始める売上予測

### 1. 最小限のコード例

```python
import numpy as np
from twinstore import PredictionPipeline

# Step 1: 過去の店舗データを準備（辞書形式）
historical_data = {
    'store_001': np.array([100000, 105000, 98000, ...]),  # 店舗1の日次売上
    'store_002': np.array([120000, 118000, 125000, ...]),  # 店舗2の日次売上
    'store_003': np.array([95000, 97000, 93000, ...]),     # 店舗3の日次売上
}

# Step 2: パイプラインを作成して学習
pipeline = PredictionPipeline()
pipeline.fit(historical_data)

# Step 3: 新規店舗の売上データ（例：30日分）
new_store_sales = [98000, 102000, 95000, 101000, 103000, ...]  # 30日分

# Step 4: 予測実行
result = pipeline.predict(new_store_sales)

# Step 5: 結果確認
print(f"予測年間売上: {result.prediction.prediction:,.0f}円")
print(f"信頼度: {result.prediction.confidence_score:.0%}")
```

### 2. pandas DataFrameを使った例

```python
import pandas as pd
from twinstore import PredictionPipeline

# CSVファイルから読み込む場合
historical_data = pd.read_csv('sales_history.csv', index_col='date', parse_dates=True)
# 形式: date | store_001 | store_002 | store_003
#      日付  |  売上1    |   売上2   |   売上3

# パイプラインで予測
pipeline = PredictionPipeline()
pipeline.fit(historical_data)

# 新規店舗データ（Series形式も可）
new_store_sales = pd.Series([98000, 102000, 95000, ...])
result = pipeline.predict(new_store_sales)
```

### 3. 店舗属性を考慮した予測

```python
# 店舗属性データ
store_attributes = pd.DataFrame({
    'store_cd': ['store_001', 'store_002', 'store_003'],
    'store_type': ['roadside', 'mall', 'urban'],
    'area': [150, 120, 100]
}).set_index('store_cd')

# 属性付きで学習
pipeline.fit(historical_data, store_attributes)

# ロードサイド店舗のみを使って予測
result = pipeline.predict(
    new_store_sales,
    filters={'store_type': 'roadside'}
)
```

## よくある使用パターン

### パターン1: Excelファイルからの読み込み

```python
# Excelから直接読み込み
historical_data = pd.read_excel('sales_data.xlsx', sheet_name='daily_sales')
store_attributes = pd.read_excel('sales_data.xlsx', sheet_name='store_info')

pipeline = PredictionPipeline()
pipeline.fit(historical_data, store_attributes)
```

### パターン2: 前処理付き予測

```python
from twinstore import PipelineConfig, PredictionPipeline

# 前処理を有効にした設定
config = PipelineConfig(
    preprocess_data=True,     # 前処理を実行
    handle_missing=True,      # 欠損値を補完
    handle_outliers=True,     # 異常値を処理
    check_quality=True        # 品質チェック実行
)

pipeline = PredictionPipeline(config)
```

### パターン3: バッチ予測

```python
# 複数店舗を一括予測
new_stores = {
    'shibuya': [95000, 98000, 102000, ...],
    'shinjuku': [105000, 108000, 111000, ...],
    'ikebukuro': [88000, 92000, 95000, ...]
}

results = pipeline.batch_predict(new_stores)

for store, result in results.items():
    print(f"{store}: {result.prediction.prediction:,.0f}円")
```

## データ要件

### 最小要件
- **過去データ**: 3店舗以上、各90日以上推奨
- **新規店舗データ**: 最低7日、推奨30日以上
- **データ型**: 数値（売上金額）

### データ形式
- **NumPy配列**: `np.array([100000, 105000, ...])`
- **リスト**: `[100000, 105000, ...]`
- **pandas Series/DataFrame**: 日付インデックス推奨
- **辞書**: `{'store_cd': [売上データ]}`

## トラブルシューティング

### Q: 「データが不足しています」エラー
```python
# 最低7日分のデータが必要
if len(new_store_sales) < 7:
    print("エラー: 最低7日分のデータが必要です")
```

### Q: 予測精度を上げたい
```python
# 1. データ期間を増やす（30日以上推奨）
# 2. 類似店舗数を調整
config = PipelineConfig(n_similar_stores=5)  # デフォルト: 5

# 3. 店舗属性でフィルタリング
result = pipeline.predict(data, filters={'area': 150})
```

### Q: 欠損値があるデータ
```python
# 自動的に処理される（preprocess_data=True）
# または手動で処理
new_store_sales = pd.Series(new_store_sales).fillna(method='ffill')
```

## 次のステップ

1. **詳細な設定**: [PIPELINE_README.md](PIPELINE_README.md) を参照
2. **データ形式の詳細**: [DATA_FORMAT_GUIDE.md](DATA_FORMAT_GUIDE.md) を参照
3. **サンプルコード**: `examples/` ディレクトリを確認

## サポート

問題が発生した場合は、以下を確認してください：

1. データが数値型であること
2. 負の売上値が含まれていないこと
3. 極端な異常値が含まれていないこと
4. 十分なデータ量があること（7日以上）