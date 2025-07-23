# データ処理機能 設計仕様書

**作成日**: 2025-07-23  
**関連要件**: [requirements.md](./requirements.md)  

## 設計概要

データ処理機能は**Single Responsibility Principle (SRP)**に基づいて設計された独立したコンポーネント群で構成されています。各コンポーネントは特定の責務を持ち、組み合わせることで複雑なデータ処理パイプラインを構築できます。

## アーキテクチャ設計

### クラス構成

```python
# データローダー
class DataLoader:
    """統合データローダー"""
    def load(self, data_source) -> np.ndarray
    
class FileReader:
    """ファイル読み込み専用クラス"""
    def read_csv(self, filepath) -> pd.DataFrame
    def read_excel(self, filepath) -> pd.DataFrame
    def read_json(self, filepath) -> pd.DataFrame
    
# データ検証
class DataValidator:
    """データ検証クラス"""
    def validate(self, data) -> ValidationResult
    
class ColumnValidator:
    """列名検証・修正クラス"""
    def validate_and_fix(self, df) -> pd.DataFrame
    
# データ前処理
class DataPreprocessor:
    """データ前処理クラス"""
    def process(self, data) -> np.ndarray
    
class DataTransformer:
    """データ変換クラス"""
    def transform(self, df) -> np.ndarray
    
# 品質評価
class QualityChecker:
    """データ品質評価クラス"""
    def check(self, data) -> QualityReport
    
# 異常検知
class AnomalyDetector:
    """異常値検知クラス"""
    def detect(self, data) -> List[bool]
```

### データフロー

```
[多様な入力] → FileReader → DataFrame
    ↓
ColumnValidator → 列名正規化
    ↓
DataValidator → 検証結果
    ↓
DataPreprocessor → 前処理済みデータ
    ↓
QualityChecker → 品質評価
    ↓
AnomalyDetector → 異常値フラグ
    ↓
DataTransformer → 最終出力 (np.ndarray)
```

## 詳細設計

### 1. DataLoader クラス

```python
class DataLoader:
    """統合データローダークラス"""
    
    def __init__(self):
        self.file_reader = FileReader()
        self.column_validator = ColumnValidator()
        self.data_transformer = DataTransformer()
        self.sample_generator = SampleGenerator()
        
    def load(self, data_source: Union[str, Dict, List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """統一的なデータ読み込み"""
        
        # データ型判定
        data_type = self._detect_data_type(data_source)
        
        # 型別読み込み処理
        if data_type == 'file_path':
            df = self.file_reader.read(data_source)
        elif data_type == 'dataframe':
            df = data_source.copy()
        elif data_type == 'dictionary':
            df = pd.DataFrame(data_source)
        elif data_type == 'list':
            df = pd.DataFrame({'sales': data_source})
        elif data_type == 'numpy_array':
            df = pd.DataFrame({'sales': data_source})
        elif data_type == 'json_string':
            import json
            data_dict = json.loads(data_source)
            df = pd.DataFrame(data_dict)
        else:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # 列名検証・修正
        df = self.column_validator.validate_and_fix(df)
        
        # 最終変換
        return self.data_transformer.transform(df)
    
    def _detect_data_type(self, data_source) -> str:
        """データ型の自動判定"""
        if isinstance(data_source, str):
            if os.path.exists(data_source):
                return 'file_path'
            else:
                try:
                    json.loads(data_source)
                    return 'json_string'
                except:
                    raise ValueError("Invalid string data source")
        elif isinstance(data_source, pd.DataFrame):
            return 'dataframe'
        elif isinstance(data_source, dict):
            return 'dictionary'
        elif isinstance(data_source, list):
            return 'list'
        elif isinstance(data_source, np.ndarray):
            return 'numpy_array'
        else:
            raise ValueError(f"Unsupported data source type: {type(data_source)}")
```

### 2. DataValidator クラス

```python
class DataValidator:
    """データ検証クラス"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def validate(self, data: Union[np.ndarray, pd.DataFrame]) -> ValidationResult:
        """包括的なデータ検証"""
        
        errors = []
        warnings = []
        
        # numpy配列に変換
        if isinstance(data, pd.DataFrame):
            if 'sales' in data.columns:
                array_data = data['sales'].values
            else:
                array_data = data.iloc[:, 0].values
        else:
            array_data = data
        
        # 基本検証
        errors.extend(self._validate_basic_requirements(array_data))
        warnings.extend(self._validate_data_quality(array_data))
        
        # 品質スコア計算
        quality_score = self._calculate_quality_score(array_data, errors, warnings)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score
        )
    
    def _validate_basic_requirements(self, data: np.ndarray) -> List[str]:
        """基本要件の検証"""
        errors = []
        
        # 最小データ要件
        if len(data) < self.config.min_data_points:
            errors.append(f"Insufficient data points: {len(data)} < {self.config.min_data_points}")
        
        # 数値範囲チェック
        if not self.config.allow_negative and np.any(data < 0):
            errors.append("Negative values not allowed")
        
        # データ型チェック
        if not np.issubdtype(data.dtype, np.number):
            errors.append("Data must be numeric")
        
        return errors
    
    def _validate_data_quality(self, data: np.ndarray) -> List[str]:
        """データ品質の検証"""
        warnings = []
        
        # 欠損値チェック
        nan_count = np.sum(np.isnan(data))
        if nan_count > 0:
            nan_ratio = nan_count / len(data)
            if nan_ratio > self.config.max_missing_ratio:
                warnings.append(f"High missing value ratio: {nan_ratio:.2%}")
            else:
                warnings.append(f"Missing values detected: {nan_count} points")
        
        # ゼロ値の連続チェック
        zero_runs = self._find_zero_runs(data)
        if len(zero_runs) > 0:
            max_run = max(len(run) for run in zero_runs)
            if max_run > 3:
                warnings.append(f"Long zero value sequence detected: {max_run} consecutive zeros")
        
        # 異常な増減チェック
        if len(data) > 1:
            daily_changes = np.diff(data)
            extreme_changes = np.abs(daily_changes) > np.std(data) * 3
            if np.any(extreme_changes):
                warnings.append("Extreme daily changes detected")
        
        return warnings
```

### 3. DataPreprocessor クラス

```python
class DataPreprocessor:
    """データ前処理クラス"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        
    def process(self, data: np.ndarray) -> np.ndarray:
        """データ前処理の実行"""
        
        processed_data = data.copy()
        
        # 欠損値処理
        if self.config.handle_missing:
            processed_data = self._handle_missing_values(processed_data)
        
        # 外れ値処理
        if self.config.handle_outliers:
            processed_data = self._handle_outliers(processed_data)
        
        # データ平滑化
        if self.config.smooth_data:
            processed_data = self._smooth_data(processed_data)
        
        return processed_data
    
    def _handle_missing_values(self, data: np.ndarray) -> np.ndarray:
        """欠損値の処理"""
        
        if not np.any(np.isnan(data)):
            return data
        
        method = self.config.missing_strategy
        
        if method == 'forward_fill':
            return self._forward_fill(data)
        elif method == 'backward_fill':
            return self._backward_fill(data)
        elif method == 'interpolate':
            return self._interpolate(data)
        elif method == 'mean':
            mean_value = np.nanmean(data)
            return np.where(np.isnan(data), mean_value, data)
        elif method == 'median':
            median_value = np.nanmedian(data)
            return np.where(np.isnan(data), median_value, data)
        else:
            raise ValueError(f"Unknown missing strategy: {method}")
    
    def _handle_outliers(self, data: np.ndarray) -> np.ndarray:
        """外れ値の処理"""
        
        method = self.config.outlier_method
        
        if method == 'iqr':
            return self._handle_outliers_iqr(data)
        elif method == 'zscore':
            return self._handle_outliers_zscore(data)
        elif method == 'winsorize':
            return self._winsorize(data)
        else:
            return data
    
    def _handle_outliers_iqr(self, data: np.ndarray) -> np.ndarray:
        """IQR方式による外れ値処理"""
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # 外れ値を境界値に置換
        return np.clip(data, lower_bound, upper_bound)
```

### 4. QualityChecker クラス

```python
class QualityChecker:
    """データ品質評価クラス"""
    
    def __init__(self, config: QualityConfig):
        self.config = config
        
    def check(self, data: np.ndarray) -> QualityReport:
        """データ品質の総合評価"""
        
        # 各観点での評価
        completeness_score = self._evaluate_completeness(data)
        consistency_score = self._evaluate_consistency(data)
        accuracy_score = self._evaluate_accuracy(data)
        timeliness_score = self._evaluate_timeliness(data)
        
        # 重み付け総合スコア
        weights = self.config.quality_weights
        overall_score = (
            completeness_score * weights['completeness'] +
            consistency_score * weights['consistency'] +
            accuracy_score * weights['accuracy'] +
            timeliness_score * weights['timeliness']
        )
        
        # 改善提案の生成
        recommendations = self._generate_recommendations(
            completeness_score, consistency_score, 
            accuracy_score, timeliness_score
        )
        
        return QualityReport(
            overall_score=overall_score,
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            accuracy_score=accuracy_score,
            timeliness_score=timeliness_score,
            recommendations=recommendations
        )
    
    def _evaluate_completeness(self, data: np.ndarray) -> float:
        """完全性の評価"""
        missing_ratio = np.sum(np.isnan(data)) / len(data)
        return max(0, 100 - missing_ratio * 100)
    
    def _evaluate_consistency(self, data: np.ndarray) -> float:
        """一貫性の評価"""
        # 変動係数による一貫性評価
        if np.mean(data) == 0:
            return 50.0  # 中性スコア
        
        cv = np.std(data) / np.mean(data)
        # CVが小さいほど一貫性が高い
        consistency_score = max(0, 100 - cv * 50)
        return min(100, consistency_score)
```

### 5. AnomalyDetector クラス

```python
class AnomalyDetector:
    """異常値検知クラス"""
    
    def __init__(self, config: AnomalyConfig):
        self.config = config
        
    def detect(self, data: np.ndarray) -> List[bool]:
        """異常値の検知"""
        
        method = self.config.detection_method
        
        if method == 'iqr':
            return self._detect_iqr(data)
        elif method == 'zscore':
            return self._detect_zscore(data)
        elif method == 'isolation_forest':
            return self._detect_isolation_forest(data)
        else:
            return [False] * len(data)
    
    def _detect_iqr(self, data: np.ndarray) -> List[bool]:
        """IQR方式による異常値検知"""
        
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        return ((data < lower_bound) | (data > upper_bound)).tolist()
    
    def _detect_zscore(self, data: np.ndarray) -> List[bool]:
        """Z-score方式による異常値検知"""
        
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val == 0:
            return [False] * len(data)
        
        z_scores = np.abs((data - mean_val) / std_val)
        threshold = self.config.zscore_threshold
        
        return (z_scores > threshold).tolist()
    
    def _detect_isolation_forest(self, data: np.ndarray) -> List[bool]:
        """Isolation Forest による異常値検知"""
        
        try:
            from sklearn.ensemble import IsolationForest
            
            # データを2次元に変形
            X = data.reshape(-1, 1)
            
            # Isolation Forest モデル
            iso_forest = IsolationForest(
                contamination=self.config.contamination_ratio,
                random_state=42
            )
            
            # 異常値検知 (-1: 異常, 1: 正常)
            predictions = iso_forest.fit_predict(X)
            
            return (predictions == -1).tolist()
            
        except ImportError:
            # scikit-learn が利用できない場合はZ-scoreにフォールバック
            return self._detect_zscore(data)
```

## 設定管理

### 設定クラス

```python
@dataclass
class ValidationConfig:
    min_data_points: int = 3
    max_missing_ratio: float = 0.1
    allow_negative: bool = False
    strict_mode: bool = False

@dataclass  
class PreprocessingConfig:
    handle_missing: bool = True
    missing_strategy: str = 'interpolate'  # forward_fill, backward_fill, interpolate, mean, median
    handle_outliers: bool = True
    outlier_method: str = 'iqr'  # iqr, zscore, winsorize
    smooth_data: bool = False

@dataclass
class QualityConfig:
    quality_weights: Dict[str, float] = field(default_factory=lambda: {
        'completeness': 0.3,
        'consistency': 0.3, 
        'accuracy': 0.3,
        'timeliness': 0.1
    })
    
@dataclass
class AnomalyConfig:
    detection_method: str = 'iqr'  # iqr, zscore, isolation_forest
    zscore_threshold: float = 3.0
    contamination_ratio: float = 0.1
```

## テスト設計

### 単体テスト

```python
class TestDataLoader:
    def test_load_csv_file(self):
        """CSVファイル読み込みテスト"""
        
    def test_load_excel_file(self):
        """Excelファイル読み込みテスト"""
        
    def test_load_dataframe(self):
        """DataFrameからの読み込みテスト"""

class TestDataValidator:
    def test_basic_validation(self):
        """基本検証テスト"""
        
    def test_missing_value_detection(self):
        """欠損値検出テスト"""

class TestDataPreprocessor:
    def test_missing_value_handling(self):
        """欠損値処理テスト"""
        
    def test_outlier_handling(self):
        """外れ値処理テスト"""
```

## 関連ドキュメント

- [要件仕様書](./requirements.md)
- [実装タスク](./tasks.md)
- [売上予測設計](../sales-prediction/design.md)