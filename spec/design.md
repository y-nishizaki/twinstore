# TwinStore 全体設計仕様書

**作成日**: 2025-07-23  
**関連要件**: [requirements.md](./requirements.md)  
**関連アーキテクチャ**: [architecture.md](./architecture.md)  

## 設計概要

TwinStoreは**パイプライン・アーキテクチャ**を採用し、データの流れを明確に定義された段階に分割することで、高い保守性と拡張性を実現します。各コンポーネントは単一責任原則（SRP）に基づいて設計され、疎結合な関係を維持しています。

### 設計思想

1. **段階的処理**: データの変換を明確な段階に分離
2. **責任分離**: 各コンポーネントが単一の責務を持つ
3. **戦略的設計**: アルゴリズムの切り替えを可能にする
4. **設定外部化**: ハードコーディングを避け、設定で制御
5. **型安全性**: 明確な型定義による安全な開発

## 主要コンポーネント設計

### 1. PredictionPipeline（パイプライン管理）

```python
class PredictionPipeline:
    """売上予測パイプラインの統合管理クラス"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.validator = DataValidator(config.validation_config)
        self.preprocessor = DataPreprocessor(config.preprocessing_config)
        self.quality_checker = QualityChecker(config.quality_config)
        self.anomaly_detector = AnomalyDetector(config.anomaly_config)
        self.similarity_engine = SimilarityEngine(config.similarity_config)
        self.predictor = SalesPredictor(config.prediction_config)
        self.explainer = PredictionExplainer(config.explanation_config)
    
    def predict(self, 
                new_store_data: Union[Dict, DataFrame, np.ndarray],
                historical_data: Optional[Dict] = None) -> PredictionResult:
        """パイプライン全体の実行"""
        
        # 1. データ検証
        validation_result = self.validator.validate(new_store_data)
        if not validation_result.is_valid and self.config.strict_mode:
            raise ValidationError(validation_result.errors)
        
        # 2. 前処理
        processed_data = self.preprocessor.process(new_store_data)
        
        # 3. 品質チェック
        quality_score = self.quality_checker.check(processed_data)
        if quality_score < self.config.quality_threshold:
            warnings.warn(f"Data quality score: {quality_score}%")
        
        # 4. 異常値検出
        anomaly_flags = self.anomaly_detector.detect(processed_data)
        
        # 5. 類似店舗検索
        similar_stores = self.similarity_engine.find_similar(
            processed_data, historical_data)
        
        # 6. 売上予測
        prediction = self.predictor.predict(
            processed_data, similar_stores)
        
        # 7. 説明生成
        explanation = self.explainer.explain(
            prediction, similar_stores, quality_score)
        
        return PredictionResult(
            prediction=prediction.value,
            confidence_interval=prediction.confidence_interval,
            confidence_score=prediction.confidence,
            similar_stores=similar_stores,
            quality_score=quality_score,
            anomaly_flags=anomaly_flags,
            explanation=explanation,
            metadata=self._generate_metadata()
        )
```

### 2. SalesPredictor（売上予測エンジン）

```python
class SalesPredictor:
    """売上予測の中核クラス"""
    
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.normalizer = Normalizer(config.normalization_method)
        
    def predict(self, 
                new_store_data: np.ndarray,
                similar_stores: List[Dict]) -> PredictionData:
        """統計的手法による売上予測"""
        
        # 正規化
        normalized_data = self.normalizer.normalize(new_store_data)
        
        # 類似店舗の重み付け平均による予測
        predictions = []
        weights = []
        
        for store in similar_stores:
            # DTW距離から重みを計算
            weight = 1.0 / (1.0 + store['distance'])
            weights.append(weight)
            
            # 成長率ベースの予測
            growth_rate = self._calculate_growth_rate(
                store['sales_pattern'], len(normalized_data))
            predicted_annual = self._extrapolate_annual_sales(
                normalized_data, growth_rate)
            predictions.append(predicted_annual)
        
        # 重み付け平均
        weights = np.array(weights)
        predictions = np.array(predictions)
        
        final_prediction = np.average(predictions, weights=weights)
        
        # 信頼区間の計算
        confidence_interval = self._calculate_confidence_interval(
            predictions, weights)
        
        # 信頼度スコアの計算
        confidence_score = self._calculate_confidence_score(
            predictions, weights, len(similar_stores))
        
        return PredictionData(
            value=final_prediction,
            confidence_interval=confidence_interval,
            confidence=confidence_score,
            method='weighted_average',
            parameters={
                'normalization': self.config.normalization_method,
                'n_similar_stores': len(similar_stores),
                'growth_model': 'exponential'
            }
        )
```

### 3. SimilarityEngine（類似性計算エンジン）

```python
class SimilarityEngine:
    """類似店舗検索エンジン"""
    
    def __init__(self, config: SimilarityConfig):
        self.config = config
        self.calculator = self._create_calculator(config.method)
        
    def find_similar(self, 
                     target_data: np.ndarray,
                     historical_data: Dict[str, np.ndarray],
                     top_k: int = None) -> List[Dict]:
        """類似店舗の検索"""
        
        if not historical_data:
            raise ValueError("Historical data is required for similarity matching")
        
        similarities = []
        
        for store_name, store_data in historical_data.items():
            # 期間の調整
            aligned_data = self._align_periods(target_data, store_data)
            
            # 類似度計算
            distance = self.calculator.calculate(target_data, aligned_data)
            
            similarities.append({
                'store_name': store_name,
                'distance': distance,
                'similarity_score': 1.0 / (1.0 + distance),
                'sales_pattern': aligned_data,
                'alignment_period': len(target_data)
            })
        
        # 類似度でソート
        similarities.sort(key=lambda x: x['distance'])
        
        # 上位K件を返す
        top_k = top_k or self.config.max_similar_stores
        return similarities[:top_k]
    
    def _create_calculator(self, method: str) -> SimilarityCalculator:
        """類似度計算器の生成"""
        calculators = {
            'dtw': DTWSimilarityCalculator,
            'cosine': CosineSimilarityCalculator,
            'correlation': CorrelationSimilarityCalculator
        }
        
        if method not in calculators:
            raise ValueError(f"Unsupported similarity method: {method}")
        
        return calculators[method](self.config)
```

### 4. DataValidator（データ検証）

```python
class DataValidator:
    """入力データの検証クラス"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        
    def validate(self, data: Any) -> ValidationResult:
        """包括的なデータ検証"""
        
        errors = []
        warnings = []
        
        # データ型チェック
        if not self._is_valid_type(data):
            errors.append("Unsupported data type")
            
        # データ変換
        try:
            converted_data = self._convert_to_array(data)
        except Exception as e:
            errors.append(f"Data conversion failed: {str(e)}")
            return ValidationResult(False, errors, warnings, 0.0)
        
        # 最小データ要件
        if len(converted_data) < self.config.min_data_points:
            errors.append(f"Insufficient data points: {len(converted_data)} < {self.config.min_data_points}")
        
        # 数値範囲チェック
        if np.any(converted_data < 0):
            if self.config.allow_negative:
                warnings.append("Negative values detected")
            else:
                errors.append("Negative values not allowed")
        
        # 欠損値チェック
        nan_count = np.sum(np.isnan(converted_data))
        if nan_count > 0:
            nan_ratio = nan_count / len(converted_data)
            if nan_ratio > self.config.max_missing_ratio:
                errors.append(f"Too many missing values: {nan_ratio:.2%}")
            else:
                warnings.append(f"Missing values detected: {nan_count} points")
        
        # 品質スコア計算
        quality_score = self._calculate_quality_score(converted_data)
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            quality_score=quality_score
        )
```

## データモデル設計

### 主要データ構造

```python
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import numpy as np

@dataclass
class PredictionResult:
    """予測結果の統合データクラス"""
    prediction: float                    # 予測年間売上
    confidence_interval: Tuple[float, float]  # 95%信頼区間
    confidence_score: float              # 信頼度 (0-1)
    similar_stores: List[Dict]           # 類似店舗リスト
    quality_score: float                 # データ品質スコア (0-100)
    anomaly_flags: List[bool]            # 異常値フラグ
    explanation: str                     # 予測説明文
    metadata: Dict                       # 予測メタデータ

@dataclass 
class ValidationResult:
    """データ検証結果"""
    is_valid: bool                       # 検証成功フラグ
    errors: List[str]                    # エラーメッセージ
    warnings: List[str]                  # 警告メッセージ
    quality_score: float                 # 品質スコア (0-100)

@dataclass
class PipelineConfig:
    """パイプライン設定"""
    validation_config: Dict              # 検証設定
    preprocessing_config: Dict           # 前処理設定
    quality_config: Dict                 # 品質チェック設定
    anomaly_config: Dict                 # 異常検知設定
    similarity_config: Dict              # 類似性設定
    prediction_config: Dict              # 予測設定
    explanation_config: Dict             # 説明生成設定
    strict_mode: bool = False            # 厳格モード
    quality_threshold: float = 70.0      # 品質閾値
```

### 設定管理設計

```python
# config/defaults.py
DEFAULT_CONFIG = {
    'validation': {
        'min_data_points': 3,
        'max_missing_ratio': 0.1,
        'allow_negative': False,
        'strict_mode': False
    },
    'preprocessing': {
        'handle_missing': True,
        'missing_strategy': 'interpolate',
        'outlier_detection': True,
        'outlier_method': 'iqr'
    },
    'similarity': {
        'method': 'dtw',
        'max_similar_stores': 10,
        'dtw_window': 0.1,
        'distance_threshold': 2.0
    },
    'prediction': {
        'normalization_method': 'first_day_ratio',
        'confidence_level': 0.95,
        'extrapolation_method': 'exponential'
    }
}

# config/constants.py
class Constants:
    MIN_DATA_POINTS = 3
    MAX_SIMILAR_STORES = 50
    DEFAULT_QUALITY_THRESHOLD = 70.0
    CONFIDENCE_LEVEL = 0.95
    DTW_WINDOW_RATIO = 0.1
```

## インターフェース設計

### 公開API

```python
# メインクラスの公開インターフェース
class TwinStore:
    """TwinStore パッケージのメインクラス"""
    
    def __init__(self, preset: str = 'retail'):
        """
        初期化
        
        Args:
            preset: 業態プリセット ('retail', 'restaurant', 'service')
        """
        
    def fit(self, historical_data: Dict[str, Union[List, np.ndarray]]):
        """
        過去データでの学習
        
        Args:
            historical_data: {店舗名: 売上データ} の辞書
        """
        
    def predict(self, 
                new_store_sales: Union[List, np.ndarray, Dict],
                return_explanation: bool = True) -> Union[float, PredictionResult]:
        """
        新規店舗の売上予測
        
        Args:
            new_store_sales: 新規店舗の売上データ
            return_explanation: 詳細結果を返すかどうか
            
        Returns:
            予測値または詳細結果
        """

# Builder パターンによる設定
class PipelineBuilder:
    """パイプライン構築用ビルダー"""
    
    def with_validation(self, **kwargs) -> 'PipelineBuilder':
        """検証設定"""
        
    def with_preprocessing(self, **kwargs) -> 'PipelineBuilder':
        """前処理設定"""
        
    def with_similarity(self, method: str, **kwargs) -> 'PipelineBuilder':
        """類似性計算設定"""
        
    def with_prediction(self, **kwargs) -> 'PipelineBuilder':
        """予測設定"""
        
    def build(self) -> PredictionPipeline:
        """パイプライン構築"""
```

### データ入力インターフェース

```python
# 多様なデータ形式への対応
class DataLoader:
    """統一的なデータ読み込みクラス"""
    
    @staticmethod
    def load(data: Union[str, Dict, List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        様々な形式のデータを統一形式に変換
        
        Supported formats:
        - CSV/Excel file path (str)
        - Dictionary: {'day1': 100, 'day2': 120, ...}  
        - List: [100, 120, 130, ...]
        - NumPy array: np.array([100, 120, 130, ...])
        - pandas DataFrame: DataFrame with sales column
        - JSON string: '{"sales": [100, 120, 130, ...]}'
        """
```

## エラーハンドリング設計

### 例外階層

```python
class TwinStoreError(Exception):
    """TwinStore パッケージのベース例外"""
    pass

class ValidationError(TwinStoreError):
    """データ検証エラー"""
    def __init__(self, errors: List[str]):
        self.errors = errors
        super().__init__(f"Validation failed: {'; '.join(errors)}")

class PredictionError(TwinStoreError):
    """予測実行エラー"""
    pass

class ConfigurationError(TwinStoreError):
    """設定エラー"""
    pass

class InsufficientDataError(TwinStoreError):
    """データ不足エラー"""
    pass
```

### エラー処理パターン

```python
def predict_with_error_handling(self, data):
    """エラーハンドリング付き予測"""
    try:
        # 予測実行
        result = self.predict(data)
        return result
        
    except ValidationError as e:
        # データ検証エラー
        logger.error(f"Data validation failed: {e}")
        if self.config.strict_mode:
            raise
        else:
            # 警告として処理継続
            warnings.warn(str(e))
            
    except InsufficientDataError as e:
        # データ不足エラー
        logger.error(f"Insufficient data: {e}")
        raise
        
    except Exception as e:
        # 予期しないエラー
        logger.exception(f"Unexpected error during prediction: {e}")
        raise PredictionError(f"Prediction failed: {str(e)}")
```

## パフォーマンス設計

### 最適化方針

1. **計算効率**: DTWのウィンドウ制約による高速化
2. **メモリ効率**: 大容量データの段階的処理
3. **キャッシング**: 類似度計算結果のキャッシュ
4. **並列処理**: バッチ予測での並列実行

### パフォーマンス監視

```python
import time
from functools import wraps

def performance_monitor(func):
    """パフォーマンス監視デコレータ"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        logger.info(f"{func.__name__} executed in {execution_time:.4f}s")
        return result
    return wrapper
```

## 実装方針

### 開発優先順位

1. **Phase 1**: コア機能実装
   - SalesPredictor
   - SimilarityEngine  
   - DataValidator

2. **Phase 2**: パイプライン統合
   - PredictionPipeline
   - PipelineBuilder
   - エラーハンドリング

3. **Phase 3**: 品質・パフォーマンス
   - QualityChecker
   - AnomalyDetector
   - 最適化

### 技術選択

- **数値計算**: NumPy, SciPy
- **データ処理**: pandas
- **DTW実装**: dtaidistance
- **データ検証**: Pydantic
- **設定管理**: YAML/JSON + dataclasses
- **ログ**: Python standard logging
- **テスト**: pytest + pytest-cov

## 関連ドキュメント

- [システムアーキテクチャ](./architecture.md)
- [全体要件仕様書](./requirements.md)
- [機能別設計書](./sales-prediction/design.md)
- [実装タスク](./tasks.md)