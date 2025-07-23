# SRP適用アーキテクチャ

TwinStoreのデータローダー機能において、Single Responsibility Principle (SRP) を適用した設計について説明します。

## SRP適用前の問題点

元の`DataLoader`クラスは以下の複数の責任を持っていました：

1. **ファイル読み込み**: CSV/Excel/JSONファイルからDataFrameを読み込む
2. **列名検証・修正**: 列名の存在確認と自動検出による修正
3. **データ変換**: DataFrameを辞書やその他の形式に変換
4. **サンプル生成**: テスト用のサンプルファイルを生成

これは単一責任原則に違反し、クラスが複雑になり、テストが困難で、変更の影響範囲が広くなる問題がありました。

## SRP適用後の設計

### 1. FileReader（ファイル読み込み専用）

**責任**: 各種ファイル形式からpd.DataFrameを読み込む

```python
class FileReader:
    def read_file(self, file_path, **kwargs) -> pd.DataFrame
    def _read_csv(self, file_path, **kwargs) -> pd.DataFrame
    def _read_excel(self, file_path, **kwargs) -> pd.DataFrame
    def _read_json(self, file_path, **kwargs) -> pd.DataFrame
```

**特徴**:
- ファイル読み込みのみに集中
- 形式判定とファイル読み込みエラーのハンドリング
- pandas/jsonライブラリの薄いラッパー

### 2. ColumnValidator（列名検証・修正専用）

**責任**: DataFrameの列名を検証し、必要に応じて修正する

```python
class ColumnValidator:
    def validate_and_fix_columns(self, df) -> pd.DataFrame
    def validate_required_columns(self, df, required_columns) -> None
    def _find_column_candidate(self, columns, candidates) -> Optional[str]
    def _fix_date_column(self, df) -> pd.DataFrame
```

**特徴**:
- 列名の自動検出とマッピング
- 必須列の存在検証
- 日付列の型変換
- 設定による柔軟なカスタマイズ

### 3. DataTransformer（データ変換専用）

**責任**: DataFrameを他の形式に変換する

```python
class DataTransformer:
    def to_dict_format(self, df) -> Dict[str, np.ndarray]
    def to_timeseries_format(self, df) -> pd.DataFrame
    def to_batch_format(self, df) -> Dict[str, Union[np.ndarray, pd.Series]]
    def to_attributes_format(self, df) -> pd.DataFrame
```

**特徴**:
- 明確な変換責任の分離
- 各出力形式に特化したメソッド
- 日付ソートなどのデータ整理機能

### 4. SampleGenerator（サンプル生成専用）

**責任**: テスト・デモ用のサンプルデータを生成する

```python
class SampleGenerator:
    def generate_historical_data(self, **params) -> pd.DataFrame
    def generate_store_attributes(self, **params) -> pd.DataFrame
    def generate_new_store_data(self, **params) -> pd.DataFrame
    def create_sample_files(self, output_dir) -> None
```

**特徴**:
- サンプルデータ生成のみに集中
- 設定可能なパラメータ
- 複数のファイル形式での出力

### 5. DataLoader（調整・統合専用）

**責任**: 各専用クラスを調整し、データ読み込みプロセス全体を管理する

```python
class DataLoader:
    def __init__(self):
        self.file_reader = FileReader(...)
        self.column_validator = ColumnValidator(...)
        self.data_transformer = DataTransformer(...)
        self.sample_generator = SampleGenerator(...)
    
    def load_historical_data(self, file_path, **kwargs):
        # 1. ファイル読み込み
        df = self.file_reader.read_file(file_path, **kwargs)
        # 2. 列名検証・修正
        df = self.column_validator.validate_and_fix_columns(df)
        # 3. 形式変換
        return self.data_transformer.to_dict_format(df)
```

**特徴**:
- 各専用クラスの協調
- 統一されたインターフェース
- プロセス全体の流れの管理

## SRP適用の利点

### 1. 理解しやすさ
- 各クラスの役割が明確
- コードの可読性向上
- 新しい開発者でも理解しやすい

### 2. テストしやすさ
- 各クラスを独立してテスト可能
- モックが容易
- テストケースが簡潔

### 3. 変更しやすさ
- 機能変更の影響範囲が限定的
- 新しいファイル形式の追加が容易
- 異なる変換形式の追加が簡単

### 4. 再利用性
- 各クラスを他の用途でも利用可能
- 組み合わせの柔軟性
- 拡張性の向上

## 使用例

### 個別クラスの直接利用

```python
from twinstore.data import FileReader, ColumnValidator, DataTransformer

# ファイル読み込みのみ
reader = FileReader()
df = reader.read_file("sales.csv")

# 列名検証のみ
validator = ColumnValidator()
df = validator.validate_and_fix_columns(df)

# データ変換のみ
transformer = DataTransformer()
dict_data = transformer.to_dict_format(df)
```

### 統合されたDataLoaderの利用

```python
from twinstore.data import DataLoader

# 従来通りの利用方法
loader = DataLoader()
data = loader.load_historical_data("sales.csv")
```

## テスト戦略

各クラスごとに独立したテストを作成：

- `test_file_reader.py`: ファイル読み込み機能のテスト
- `test_column_validator.py`: 列名検証・修正機能のテスト  
- `test_data_transformer.py`: データ変換機能のテスト
- `test_sample_generator.py`: サンプル生成機能のテスト
- `test_loader.py`: 統合機能のテスト

## 拡張性

### 新しいファイル形式の追加

`FileReader`クラスのみを変更：

```python
class FileReader:
    def _read_parquet(self, file_path, **kwargs) -> pd.DataFrame:
        return pd.read_parquet(file_path, **kwargs)
```

### 新しい出力形式の追加

`DataTransformer`クラスのみを変更：

```python
class DataTransformer:
    def to_tensor_format(self, df) -> torch.Tensor:
        # PyTorch Tensor形式への変換
        pass
```

### 新しい検証ルールの追加

`ColumnValidator`クラスのみを変更：

```python
class ColumnValidator:
    def validate_business_rules(self, df) -> pd.DataFrame:
        # ビジネスルール検証
        pass
```

## まとめ

SRPの適用により、TwinStoreのデータローダー機能は：

- **保守性が向上**: 各機能の変更が他に影響しない
- **テスト性が向上**: 各責任を独立してテスト可能
- **拡張性が向上**: 新機能の追加が容易
- **再利用性が向上**: 各クラスを他の用途でも利用可能

この設計により、コードの品質と開発効率が大幅に改善されました。