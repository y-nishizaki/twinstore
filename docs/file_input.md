# ファイル入力機能

TwinStoreでは、CSV、Excel、JSONファイルを直接読み込んでパイプラインを実行することができます。

## サポートしているファイル形式

- **CSV** (`.csv`)
- **Excel** (`.xlsx`, `.xls`)
- **JSON** (`.json`)

## 基本的な使用方法

### CSVファイルからの学習

```python
from twinstore import PredictionPipeline

# パイプライン初期化
pipeline = PredictionPipeline()

# CSVファイルから学習
pipeline.fit('historical_sales.csv')

# 店舗属性も同時に読み込み
pipeline.fit('historical_sales.csv', 'store_attributes.csv')
```

### Excelファイルからの学習

```python
# Excelファイルから学習
pipeline.fit('sales_data.xlsx')

# 特定のシートを指定
pipeline.fit('sales_data.xlsx', sheet_name='Sales')
```

### JSONファイルからの学習

```python
# JSONファイルから学習
pipeline.fit('sales_data.json')
```

## バッチ予測でのファイル使用

新規店舗データのファイルを直接使ってバッチ予測も可能です：

```python
# 学習済みパイプラインでバッチ予測
results = pipeline.batch_predict('new_stores.csv')

# 結果の確認
for store_cd, result in results.items():
    print(f"{store_cd}: ¥{result.prediction.prediction:,.0f}")
```

## データファイルの形式

### 過去売上データの形式

CSVまたはExcelファイルは以下の列を含む必要があります：

| 列名 | 説明 | 必須 | データ型 |
|------|------|------|-----------|
| store_cd | 店舗コード | ○ | 文字列 |
| date | 日付 | ○ | 日付 (YYYY-MM-DD) |
| sales | 売上金額 | ○ | 数値 |

**例:**
```csv
store_cd,date,sales
A001,2024-01-01,100000
A001,2024-01-02,105000
A002,2024-01-01,95000
A002,2024-01-02,98000
```

### 店舗属性データの形式

| 列名 | 説明 | 必須 | データ型 |
|------|------|------|-----------|
| store_cd | 店舗コード | ○ | 文字列 |
| type | 店舗タイプ | × | 文字列 |
| area | 店舗面積 | × | 数値 |
| location | 立地 | × | 文字列 |

**例:**
```csv
store_cd,type,area,location
A001,urban,150,Tokyo
A002,suburban,200,Osaka
```

### 新規店舗データの形式

バッチ予測用のファイルは以下の形式：

```csv
store_cd,date,sales
N001,2024-02-01,95000
N001,2024-02-02,98000
N002,2024-02-01,92000
N002,2024-02-02,94000
```

## 列名のカスタマイズ

デフォルトの列名と異なる場合は、設定でカスタマイズできます：

```python
from twinstore import PipelineConfig, PredictionPipeline

# カスタム列名を設定
config = PipelineConfig(
    store_cd_column='shop_id',      # デフォルト: 'store_cd'
    date_column='business_date',    # デフォルト: 'date'
    sales_column='daily_revenue'    # デフォルト: 'sales'
)

pipeline = PredictionPipeline(config)
pipeline.fit('custom_format.csv')
```

## ファイル読み込みオプション

### CSVオプション

```python
# カスタム区切り文字
pipeline.fit('data.csv', sep=';')

# カスタムエンコーディング
pipeline.fit('data.csv', encoding='shift-jis')

# 特定の列のみ読み込み
pipeline.fit('data.csv', usecols=['store_cd', 'date', 'sales'])
```

### Excelオプション

```python
# 特定のシート
pipeline.fit('data.xlsx', sheet_name='Sales')

# 複数シートから選択
pipeline.fit('data.xlsx', sheet_name=1)  # 2番目のシート
```

## JSON形式の詳細

### リスト形式

```json
[
    {"store_cd": "A001", "date": "2024-01-01", "sales": 100000},
    {"store_cd": "A001", "date": "2024-01-02", "sales": 105000},
    {"store_cd": "A002", "date": "2024-01-01", "sales": 95000}
]
```

### 辞書形式（店舗ごと）

```json
{
    "A001": [100000, 105000, 98000],
    "A002": [95000, 97000, 99000]
}
```

## 列名の自動検出

TwinStoreは以下の列名を自動で検出します：

- **店舗コード列**: `store_cd`, `store_id`, `store_code`, `shop_id`, `shop_code`
- **売上列**: `sales`, `amount`, `revenue`, `total`, `value`
- **日付列**: `date`, `datetime`, `timestamp`, `day`, `time`

自動検出された場合は警告メッセージが表示されます。

## エラーハンドリング

### よくあるエラーと対処法

1. **ファイルが見つからない**
```python
FileNotFoundError: File not found: data.csv
```
→ ファイルパスを確認してください

2. **必須列が見つからない**
```python
ValueError: Required columns not found: ['store_cd']
```
→ 列名を確認するか、カスタム列名を設定してください

3. **サポートされていないファイル形式**
```python
ValueError: Unsupported file format: .txt
```
→ CSV、Excel、JSONファイルを使用してください

## サンプルファイルの生成

DataLoaderクラスを使ってサンプルファイルを生成できます：

```python
from twinstore.data import DataLoader

# サンプルファイルを生成
loader = DataLoader()
loader.create_sample_files("sample_data")

# 生成されるファイル:
# - sample_data/historical_sales.csv
# - sample_data/historical_sales.xlsx
# - sample_data/store_attributes.csv
# - sample_data/new_stores.csv
```

## パフォーマンスの考慮事項

- **大きなファイル**: メモリ使用量に注意し、必要に応じてチャンク読み込みを検討
- **Excel形式**: CSVより読み込みが遅い場合があります
- **JSON形式**: 複雑な構造は避け、シンプルな形式を推奨

## 実践的な例

完全な例については、`examples/file_input_example.py`を参照してください。このファイルには以下の例が含まれています：

- CSV/Excel/JSONファイルからの学習
- カスタム列名の使用
- ファイル読み込みオプションの活用
- バッチ予測でのファイル使用
- エラーハンドリングの実装