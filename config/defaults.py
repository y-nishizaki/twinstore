"""
デフォルト設定値の定義
"""

from typing import Dict, Any

# SalesPredictorのデフォルト設定
PREDICTOR_DEFAULTS = {
    'similarity_method': 'dtw',
    'normalization': 'zscore',
    'distance_metric': 'euclidean',
    'window_size': None,
    'n_similar': 5,
    'confidence_level': 0.95,
}

# DataPreprocessorのデフォルト設定
PREPROCESSOR_DEFAULTS = {
    'missing_threshold': None,  # constants.QUALITY_CONSTANTS['HIGH_MISSING_THRESHOLD']を使用
    'outlier_method': 'iqr',
    'interpolation_method': 'linear',
    'smooth_method': 'moving_average',
    'smooth_window': 7,
}

# QualityCheckerのデフォルト設定
QUALITY_CHECKER_DEFAULTS = {
    'missing_threshold': None,  # constants.QUALITY_CONSTANTS['MISSING_THRESHOLD']を使用
    'outlier_threshold': None,  # constants.QUALITY_CONSTANTS['OUTLIER_THRESHOLD']を使用
    'consistency_threshold': None,  # constants.QUALITY_CONSTANTS['CONSISTENCY_THRESHOLD']を使用
    'check_items': ['completeness', 'consistency', 'accuracy', 'timeliness'],
}

# AnomalyDetectorのデフォルト設定
ANOMALY_DETECTOR_DEFAULTS = {
    'method': 'statistical',
    'sensitivity': 0.95,
    'min_history': 7,
    'alert_levels': {
        'info': {'threshold': 2.0},
        'warning': {'threshold': 3.0},
        'critical': {'threshold': 4.0},
    }
}

# PipelineConfigのデフォルト設定
PIPELINE_DEFAULTS = {
    'validate_data': True,
    'preprocess_data': True,
    'handle_missing': True,
    'detect_anomalies': True,
    'check_quality': True,
    'optimize_period': False,
    'generate_explanation': True,
    'generate_report': False,
    'save_results': True,
    'output_format': 'json',
    'min_quality_score': None,  # constants.QUALITY_CONSTANTS['MIN_QUALITY_SCORE']を使用
    'anomaly_threshold': None,  # constants.QUALITY_CONSTANTS['OUTLIER_THRESHOLD']を使用
    'min_days': None,  # constants.TIME_SERIES_CONSTANTS['MIN_DAYS']を使用
    'parallel_processing': True,
}

# PredictionExplainerのデフォルト設定
EXPLAINER_DEFAULTS = {
    'language': 'ja',
    'format': 'short',
    'templates_path': None,  # Noneの場合は内蔵テンプレートを使用
}

# 業態別プリセット設定
INDUSTRY_PRESETS = {
    'retail': {
        'similarity_method': 'dtw',
        'normalization': 'zscore',
        'n_similar': 5,
        'window_constraint': 7,
        'confidence_level': 0.95,
        'min_matching_days': 7,
        'max_matching_days': 30,
        'confidence_threshold': 0.8,
    },
    'restaurant': {
        'similarity_method': 'dtw',
        'normalization': 'first_value',
        'n_similar': 7,
        'window_constraint': 14,
        'confidence_level': 0.90,
        'min_matching_days': 14,
        'max_matching_days': 60,
        'confidence_threshold': 0.7,
    },
    'service': {
        'similarity_method': 'correlation',
        'normalization': 'minmax',
        'n_similar': 10,
        'window_constraint': None,
        'confidence_level': 0.95,
        'min_matching_days': 30,
        'max_matching_days': 90,
        'confidence_threshold': 0.6,
    },
}

# データ形式関連のデフォルト
DATA_FORMAT_DEFAULTS = {
    'date_column': 'date',
    'sales_column': 'sales',
    'store_cd_column': 'store_cd',
    'date_format': '%Y-%m-%d',
    'encoding': 'utf-8',
}

# ファイルローダーのデフォルト設定
LOADER_DEFAULTS = {
    'supported_formats': ['.csv', '.xlsx', '.xls', '.json'],
    'auto_detect_columns': True,
    'column_mapping': {
        'store_columns': ['store_cd', 'store_id', 'store_code', 'shop_id', 'shop_code'],
        'sales_columns': ['sales', 'amount', 'revenue', 'total', 'value'],
        'date_columns': ['date', 'datetime', 'timestamp', 'day', 'time']
    },
    'csv_options': {
        'sep': ',',
        'header': 0,
        'encoding': 'utf-8',
        'parse_dates': True,
    },
    'excel_options': {
        'sheet_name': 0,
        'header': 0,
    },
    'json_options': {
        'encoding': 'utf-8',
    },
    'validation': {
        'check_required_columns': True,
        'allow_missing_dates': True,
        'min_rows': 1,
    }
}

# レポート生成のデフォルト設定
REPORT_DEFAULTS = {
    'formats': ['pdf', 'excel', 'json'],
    'language': 'ja',
    'include_plots': True,
    'include_raw_data': False,
    'template': 'default',
}

# API関連のデフォルト設定
API_DEFAULTS = {
    'host': '0.0.0.0',
    'port': 8000,
    'workers': 4,
    'timeout': 30,
    'max_request_size': 10 * 1024 * 1024,  # 10MB
}

# ログ設定のデフォルト
LOGGING_DEFAULTS = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'date_format': '%Y-%m-%d %H:%M:%S',
    'file': None,  # Noneの場合はコンソール出力のみ
}

def get_default_config(component: str) -> Dict[str, Any]:
    """
    コンポーネント名から対応するデフォルト設定を取得
    
    Parameters
    ----------
    component : str
        コンポーネント名（'predictor', 'preprocessor', 等）
    
    Returns
    -------
    Dict[str, Any]
        デフォルト設定の辞書
    """
    configs = {
        'predictor': PREDICTOR_DEFAULTS,
        'preprocessor': PREPROCESSOR_DEFAULTS,
        'quality_checker': QUALITY_CHECKER_DEFAULTS,
        'anomaly_detector': ANOMALY_DETECTOR_DEFAULTS,
        'pipeline': PIPELINE_DEFAULTS,
        'explainer': EXPLAINER_DEFAULTS,
        'data_format': DATA_FORMAT_DEFAULTS,
        'report': REPORT_DEFAULTS,
        'api': API_DEFAULTS,
        'logging': LOGGING_DEFAULTS,
    }
    
    return configs.get(component, {})