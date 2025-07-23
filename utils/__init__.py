"""
ユーティリティモジュール

共通処理を提供する
"""

from .statistics import (
    detect_outliers_zscore,
    detect_outliers_iqr,
    calculate_basic_statistics,
    find_zero_runs,
    calculate_confidence_interval
)
from .data_conversion import (
    to_numpy_array,
    dataframe_to_dict,
    dict_to_dataframe,
    ensure_2d_array,
    flatten_dict_values
)
from .validation import (
    check_missing_ratio,
    check_data_length,
    validate_numeric_data,
    validate_date_continuity,
    validate_store_code
)

__all__ = [
    # statistics
    'detect_outliers_zscore',
    'detect_outliers_iqr',
    'calculate_basic_statistics',
    'find_zero_runs',
    'calculate_confidence_interval',
    # data_conversion
    'to_numpy_array',
    'dataframe_to_dict',
    'dict_to_dataframe',
    'ensure_2d_array',
    'flatten_dict_values',
    # validation
    'check_missing_ratio',
    'check_data_length',
    'validate_numeric_data',
    'validate_date_continuity',
    'validate_store_code',
]