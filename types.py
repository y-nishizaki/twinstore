"""
型定義モジュール

TwinStore全体で使用される型エイリアスを定義
"""

from typing import Union, Dict, List, Optional, Literal, TypedDict, Any, Tuple
from pathlib import Path
import numpy as np
import pandas as pd

# Python 3.10+ の型エイリアス記法を使用可能にする
try:
    from typing import TypeAlias
except ImportError:
    # Python 3.9以下の場合
    TypeAlias = type

# 基本的な配列型
ArrayLike: TypeAlias = Union[np.ndarray, pd.Series, List[float], List[int]]

# 売上データ型
SalesData: TypeAlias = Union[pd.DataFrame, Dict[str, np.ndarray]]
SingleStoreSales: TypeAlias = Union[np.ndarray, pd.Series, List[float]]
MultipleStoreSales: TypeAlias = Dict[str, SingleStoreSales]

# 店舗属性型
StoreAttributes: TypeAlias = Union[pd.DataFrame, Dict[str, Dict[str, Any]]]

# ファイルパス型
FilePath: TypeAlias = Union[str, Path]

# 正規化方法
NormalizationMethod: TypeAlias = Literal["z-score", "min-max", "first-day-ratio", "mean-scaling"]

# 類似度計算方法
SimilarityMetric: TypeAlias = Literal["dtw", "cosine", "correlation", "euclidean"]

# 補間方法
InterpolationMethod: TypeAlias = Literal["linear", "forward", "backward", "nearest", "spline"]

# 平滑化方法
SmoothingMethod: TypeAlias = Literal["moving_average", "exponential", "savgol", "lowess"]

# 異常値検出方法
OutlierMethod: TypeAlias = Literal["iqr", "zscore", "isolation_forest", "mad", "percentile"]

# 品質スコア辞書
class QualityScores(TypedDict):
    completeness: float
    consistency: float
    accuracy: float
    timeliness: float

# 予測結果辞書
class PredictionDict(TypedDict):
    prediction: float
    lower_bound: float
    upper_bound: float
    confidence_score: float
    similar_stores: List[Tuple[str, float]]
    method: str

# 検証結果辞書
class ValidationDict(TypedDict):
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    summary: Dict[str, Any]

# パイプライン設定辞書
class PipelineConfigDict(TypedDict, total=False):
    validate_input: bool
    strict_validation: bool
    min_days: int
    preprocess_data: bool
    handle_missing: bool
    handle_outliers: bool
    smooth_data: bool
    check_quality: bool
    quality_threshold: float
    similarity_metric: str
    normalization_method: str
    n_similar_stores: int
    confidence_level: float
    auto_optimize_period: bool
    generate_explanation: bool
    explanation_language: str
    save_results: bool
    output_dir: Optional[str]
    output_format: str

# 業態プリセット辞書
class IndustryPresetDict(TypedDict):
    similarity_method: str
    normalization: str
    n_similar: int
    window_constraint: Optional[int]
    confidence_level: float
    min_matching_days: int
    max_matching_days: int
    confidence_threshold: float