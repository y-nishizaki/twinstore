# TwinStore データ形式ガイド

## 概要

TwinStoreは柔軟なデータ形式に対応しており、pandas DataFrame、numpy配列、Python辞書、リストなど様々な形式でデータを入力できます。

## 1. 過去の店舗売上データ（学習用）

### 形式1: pandas DataFrame（推奨）

```python
import pandas as pd

# 日付をインデックス、店舗コードを列名とするDataFrame
dates = pd.date_range('2023-01-01', periods=365, freq='D')
historical_data = pd.DataFrame({
    'store_001': [100000, 105000, 98000, ...],  # 365日分
    'store_002': [120000, 118000, 125000, ...],
    'store_003': [95000, 97000, 93000, ...],
}, index=dates)

# 使用例
pipeline.fit(historical_data)
```

### 形式2: 辞書形式

```python
# 店舗コードをキー、売上配列を値とする辞書
historical_data = {
    'store_001': np.array([100000, 105000, 98000, ...]),  # numpy配列
    'store_002': [120000, 118000, 125000, ...],           # リスト
    'store_003': pd.Series([95000, 97000, 93000, ...]),   # Series
}

# 使用例
pipeline.fit(historical_data)
```

### 形式3: CSVファイルから読み込み

```python
# CSVファイルの形式：
# date,store_001,store_002,store_003
# 2023-01-01,100000,120000,95000
# 2023-01-02,105000,118000,97000

historical_data = pd.read_csv('historical_sales.csv', 
                            index_col='date', 
                            parse_dates=True)

pipeline.fit(historical_data)
```

## 2. 新規店舗の売上データ（予測対象）

### 形式1: numpy配列

```python
import numpy as np

# 30日分の売上データ
new_store_sales = np.array([
    95000, 98000, 102000, 99000, 103000,
    105000, 108000, 96000, 101000, 104000,
    # ... 30日分
])

result = pipeline.predict(new_store_sales)
```

### 形式2: pandas Series

```python
# 日付インデックス付きSeries
dates = pd.date_range('2024-01-01', periods=30, freq='D')
new_store_sales = pd.Series(
    [95000, 98000, 102000, ...],
    index=dates,
    name='new_store'
)

result = pipeline.predict(new_store_sales)
```

### 形式3: Pythonリスト

```python
# 単純なリスト
new_store_sales = [
    95000, 98000, 102000, 99000, 103000,
    # ... 30日分
]

result = pipeline.predict(new_store_sales)
```

## 3. 店舗属性データ（オプション）

### DataFrame形式（推奨）

```python
# 店舗の属性情報
store_attributes = pd.DataFrame({
    'store_cd': ['store_001', 'store_002', 'store_003'],
    'store_type': ['roadside', 'mall', 'urban'],
    'area': [150.5, 120.0, 180.3],  # 店舗面積（㎡）
    'location': ['suburban', 'urban', 'downtown'],
    'opening_date': ['2020-04-01', '2019-10-15', '2021-03-20'],
    'parking_spaces': [50, 0, 20],
})

# store_cdをインデックスに設定
store_attributes = store_attributes.set_index('store_cd')

# 使用例
pipeline.fit(historical_data, store_attributes)
```

## 4. バッチ予測用データ

### 複数店舗の辞書形式

```python
# 複数の新規店舗データ
batch_data = {
    'new_store_shibuya': np.array([95000, 98000, 102000, ...]),
    'new_store_shinjuku': [105000, 108000, 96000, ...],
    'new_store_ikebukuro': pd.Series([88000, 92000, 95000, ...]),
}

# バッチ予測
results = pipeline.batch_predict(batch_data)
```

## 5. 実践的なデータ準備例

### 例1: Excelファイルからの読み込み

```python
import pandas as pd
from twinstore import PredictionPipeline

# Excelファイルから読み込み
# シート1: 日次売上データ
# シート2: 店舗属性
excel_file = 'sales_data.xlsx'

# 売上データの読み込み
historical_data = pd.read_excel(
    excel_file, 
    sheet_name='daily_sales',
    index_col='date',
    parse_dates=True
)

# 店舗属性の読み込み
store_attributes = pd.read_excel(
    excel_file,
    sheet_name='store_info',
    index_col='store_cd'
)

# パイプラインで使用
pipeline = PredictionPipeline()
pipeline.fit(historical_data, store_attributes)
```

### 例2: データベースからの読み込み

```python
import pandas as pd
from sqlalchemy import create_engine

# データベース接続
engine = create_engine('postgresql://user:password@host:port/database')

# SQLクエリで売上データを取得
query = """
SELECT date, store_cd, sales_amount
FROM daily_sales
WHERE date >= '2023-01-01'
"""

# データの読み込みとピボット
df = pd.read_sql(query, engine)
historical_data = df.pivot(
    index='date',
    columns='store_cd',
    values='sales_amount'
)

# 使用
pipeline.fit(historical_data)
```

### 例3: APIからのデータ取得

```python
import requests
import pandas as pd

# APIからデータ取得
response = requests.get('https://api.example.com/sales/historical')
data = response.json()

# JSONをDataFrameに変換
historical_data = pd.DataFrame(data['sales'])
historical_data['date'] = pd.to_datetime(historical_data['date'])
historical_data = historical_data.set_index('date')

# 使用
pipeline.fit(historical_data)
```

## 6. データ形式の検証

### データ検証機能の使用

```python
from twinstore import DataValidator

# バリデータの作成
validator = DataValidator()

# データの検証
result = validator.validate_sales_data(historical_data)
print(result.get_report())

# 予測入力データの検証
result = validator.validate_prediction_input(new_store_sales)
if not result.is_valid:
    print("エラー:", result.errors)
```

## 7. データ準備のベストプラクティス

### 必須要件

1. **数値データ**: 売上は数値（整数または浮動小数点）
2. **時系列順**: データは時系列順に並んでいること
3. **最小データ量**: 
   - 学習用: 各店舗90日以上推奨
   - 予測用: 最低7日、推奨30日以上

### 推奨事項

1. **日付インデックス**: pandas DataFrameでは日付をインデックスに
2. **欠損値の確認**: 事前に欠損値をチェック
3. **異常値の確認**: 極端な値がないか確認
4. **一貫性**: 全店舗で同じ期間のデータを用意

### データ品質チェックリスト

```python
# データ品質の事前チェック
def check_data_quality(data):
    checks = {
        'has_data': len(data) > 0,
        'is_numeric': pd.api.types.is_numeric_dtype(data),
        'no_negative': (data >= 0).all(),
        'no_missing': not data.isna().any(),
        'sufficient_data': len(data) >= 7,
    }
    
    for check, passed in checks.items():
        print(f"{check}: {'✓' if passed else '✗'}")
    
    return all(checks.values())

# 使用例
is_valid = check_data_quality(new_store_sales)
```

## 8. トラブルシューティング

### よくあるエラーと対処法

#### 1. 「ValueError: Input series cannot be empty」
```python
# 原因: 空のデータ
# 対処: データが存在することを確認
if len(new_store_sales) == 0:
    print("エラー: データが空です")
```

#### 2. 「ValueError: Data contains NaN values」
```python
# 原因: 欠損値
# 対処: 欠損値を補完
new_store_sales = new_store_sales.fillna(method='ffill')
```

#### 3. 「ValueError: Inconsistent data lengths」
```python
# 原因: 店舗間でデータ長が異なる
# 対処: 同じ期間に揃える
min_length = min(len(data) for data in historical_data.values())
historical_data = {k: v[:min_length] for k, v in historical_data.items()}
```

## 9. サンプルデータ生成関数

```python
def generate_sample_data(n_stores=5, n_days=365):
    """テスト用のサンプルデータを生成"""
    import numpy as np
    import pandas as pd
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    
    # 売上データ
    data = {}
    for i in range(n_stores):
        base = 100000 + i * 20000
        trend = np.linspace(0, 20000, n_days)
        seasonal = 10000 * np.sin(np.arange(n_days) * 2 * np.pi / 7)
        noise = np.random.normal(0, 5000, n_days)
        
        sales = base + trend + seasonal + noise
        data[f'store_{i:03d}'] = np.maximum(sales, 0)
    
    historical_data = pd.DataFrame(data, index=dates)
    
    # 店舗属性
    store_attributes = pd.DataFrame({
        'store_cd': list(data.keys()),
        'store_type': np.random.choice(['roadside', 'mall', 'urban'], n_stores),
        'area': np.random.uniform(80, 200, n_stores),
    }).set_index('store_cd')
    
    # 新規店舗データ（30日分）
    new_store_sales = 110000 + np.random.normal(0, 8000, 30)
    new_store_sales = np.maximum(new_store_sales, 0)
    
    return historical_data, store_attributes, new_store_sales

# 使用例
hist_data, store_attrs, new_sales = generate_sample_data()
```