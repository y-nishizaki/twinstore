"""
TwinStore: 類似店舗売上予測パッケージ

新規出店後の限られた売上データから年間売上を高精度で予測するPythonパッケージ。
類似店舗（Twin Store）マッチング技術とDTW（動的時間伸縮法）を活用。
"""

__version__ = "1.0.0"
__author__ = "TwinStore Development Team"
__license__ = "MIT"

from typing import List, Optional, Dict, Any

# コア機能のインポート
from .core.predictor import SalesPredictor
from .core.similarity import SimilarityEngine
from .core.normalizer import DataNormalizer
from .core.explainer import PredictionExplainer

# データ管理機能のインポート
from .data.validator import DataValidator
from .data.preprocessor import DataPreprocessor
from .data.quality_checker import QualityChecker
from .data.anomaly_detector import AnomalyDetector

# パイプライン機能のインポート
from .pipeline import PredictionPipeline, PipelineBuilder, PipelineConfig

# 可視化機能のインポート
from .visualization.sales_alignment_visualizer import (
    SalesAlignmentVisualizer,
    AlignmentConfig
)

__all__ = [
    # コア機能
    "SalesPredictor",
    "SimilarityEngine", 
    "DataNormalizer",
    "PredictionExplainer",
    # データ管理
    "DataValidator",
    "DataPreprocessor",
    "QualityChecker",
    "AnomalyDetector",
    # パイプライン
    "PredictionPipeline",
    "PipelineBuilder",
    "PipelineConfig",
    # 可視化
    "SalesAlignmentVisualizer",
    "AlignmentConfig",
]