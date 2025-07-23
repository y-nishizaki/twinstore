"""
データ検証関連の定数と設定
"""

from typing import Dict, Any, List
from . import TIME_SERIES_CONSTANTS

# データ型の検証ルール
DATA_TYPE_RULES = {
    'sales': {
        'types': ['int', 'float', 'numpy.int64', 'numpy.float64'],
        'min_value': 0,
        'max_value': None,
        'allow_negative': False,
    },
    'date': {
        'types': ['datetime', 'pandas.Timestamp', 'numpy.datetime64'],
        'format': '%Y-%m-%d',
    },
    'store_cd': {
        'types': ['str', 'int'],
        'pattern': r'^[A-Za-z0-9_-]+$',
        'max_length': 50,
    }
}

# データ量の検証ルール
DATA_VOLUME_RULES = {
    'min_days': None,  # constants.TIME_SERIES_CONSTANTS['MIN_DAYS']を使用
    'recommended_days': None,  # constants.TIME_SERIES_CONSTANTS['RECOMMENDED_DAYS']を使用
    'optimal_days': 14,
    'max_days': 730,  # 2年
    'min_stores': 1,
    'recommended_stores': 10,
    'min_days_per_store': None,  # constants.TIME_SERIES_CONSTANTS['MIN_DAYS_PER_STORE']を使用
}

# 欠損値の検証ルール
MISSING_VALUE_RULES = {
    'max_missing_ratio': 0.3,          # 最大欠損率
    'max_consecutive_missing': 7,       # 最大連続欠損日数
    'critical_missing_ratio': 0.5,      # 重大な欠損率
    'interpolation_methods': ['linear', 'forward', 'backward', 'nearest', 'spline'],
}

# 異常値の検証ルール
ANOMALY_RULES = {
    'methods': ['zscore', 'iqr', 'isolation_forest', 'mad', 'percentile'],
    'zscore_thresholds': {
        'low': 3.0,
        'medium': 2.5,
        'high': 2.0,
    },
    'iqr_multiplier': 1.5,
    'percentile_bounds': (5, 95),
}

# 時系列データの検証ルール
TIME_SERIES_RULES = {
    'date_continuity': True,            # 日付の連続性チェック
    'allow_duplicates': False,          # 重複日付の許可
    'sort_required': True,              # ソート必須
    'frequency': 'D',                   # 日次データ
    'max_gap_days': 7,                  # 最大ギャップ日数
}

# 店舗属性の検証ルール
STORE_ATTRIBUTE_RULES = {
    'required_fields': [],              # 必須フィールドなし
    'optional_fields': ['type', 'area', 'location', 'parking'],
    'field_types': {
        'type': str,
        'area': (int, float),
        'location': str,
        'parking': bool,
    },
    'field_constraints': {
        'area': {'min': 0, 'max': 10000},
        'type': ['urban', 'suburban', 'rural', 'roadside'],
    }
}

# 予測結果の検証ルール
PREDICTION_VALIDATION_RULES = {
    'prediction_range': {
        'min_multiplier': 0.1,          # 最小倍率（入力データ平均の10%）
        'max_multiplier': 100.0,        # 最大倍率（入力データ平均の100倍）
    },
    'confidence_score_range': (0.0, 1.0),
    'required_fields': ['prediction', 'lower_bound', 'upper_bound', 'confidence_score'],
}

# バリデーションメッセージ
VALIDATION_MESSAGES = {
    'insufficient_data': "データが不足しています。最低{min_days}日分のデータが必要です。",
    'too_many_missing': "欠損値が多すぎます（{ratio:.1%}）。最大{max_ratio:.1%}まで許容されます。",
    'invalid_data_type': "無効なデータ型です: {data_type}。期待される型: {expected_types}",
    'negative_sales': "売上に負の値が含まれています。",
    'date_not_sorted': "日付が時系列順にソートされていません。",
    'duplicate_dates': "重複する日付が存在します: {dates}",
    'large_date_gap': "日付に大きなギャップがあります: {gap}日",
    'invalid_store_cd': "無効な店舗コード形式です: {store_cd}",
    'no_valid_data': "有効なデータが見つかりません。",
}

# すべての検証ルールをまとめた辞書（後方互換性のため）
VALIDATION_RULES = {
    'DATA_TYPE_RULES': DATA_TYPE_RULES,
    'DATA_VOLUME_RULES': DATA_VOLUME_RULES,
    'MISSING_VALUE_RULES': MISSING_VALUE_RULES,
    'ANOMALY_RULES': ANOMALY_RULES,
    'TIME_SERIES_RULES': TIME_SERIES_RULES,
    'STORE_ATTRIBUTE_RULES': STORE_ATTRIBUTE_RULES,
    'PREDICTION_VALIDATION_RULES': PREDICTION_VALIDATION_RULES,
}

def get_validation_rule(rule_type: str) -> Dict[str, Any]:
    """
    検証ルールを取得
    
    Parameters
    ----------
    rule_type : str
        ルールタイプ（'data_type', 'data_volume', 等）
    
    Returns
    -------
    Dict[str, Any]
        検証ルール
    """
    # 動的に値を設定する
    if rule_type == 'data_volume':
        return {
            'min_days': TIME_SERIES_CONSTANTS['MIN_DAYS'],
            'recommended_days': TIME_SERIES_CONSTANTS['RECOMMENDED_DAYS'],
            'optimal_days': 14,
            'max_days': 730,  # 2年
            'min_stores': 1,
            'recommended_stores': 10,
            'min_days_per_store': TIME_SERIES_CONSTANTS['MIN_DAYS_PER_STORE'],
        }
    
    rules = {
        'data_type': DATA_TYPE_RULES,
        'missing_value': MISSING_VALUE_RULES,
        'anomaly': ANOMALY_RULES,
        'time_series': TIME_SERIES_RULES,
        'store_attribute': STORE_ATTRIBUTE_RULES,
        'prediction': PREDICTION_VALIDATION_RULES,
    }
    
    return rules.get(rule_type, {})

def get_validation_message(message_key: str, **kwargs) -> str:
    """
    検証メッセージを取得してフォーマット
    
    Parameters
    ----------
    message_key : str
        メッセージキー
    **kwargs
        フォーマット用のパラメータ
    
    Returns
    -------
    str
        フォーマット済みメッセージ
    """
    template = VALIDATION_MESSAGES.get(message_key, "Validation error: {message_key}")
    return template.format(**kwargs)