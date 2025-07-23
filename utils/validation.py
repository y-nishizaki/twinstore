"""
検証ユーティリティ

共通の検証処理を提供する
"""

from typing import Union, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from ..config import QUALITY_CONSTANTS, TIME_SERIES_CONSTANTS


def check_missing_ratio(
    data: Union[np.ndarray, pd.Series, pd.DataFrame]
) -> Tuple[float, bool]:
    """
    欠損値の割合をチェック
    
    Parameters
    ----------
    data : array-like
        チェックするデータ
        
    Returns
    -------
    Tuple[float, bool]
        (欠損率, 警告が必要か)
    """
    if isinstance(data, pd.DataFrame):
        total_values = data.size
        missing_values = data.isna().sum().sum()
    else:
        data_array = np.asarray(data)
        total_values = data_array.size
        missing_values = np.sum(np.isnan(data_array))
    
    if total_values == 0:
        return 0.0, False
    
    missing_ratio = missing_values / total_values
    needs_warning = missing_ratio > QUALITY_CONSTANTS['MISSING_THRESHOLD']
    
    return missing_ratio, needs_warning


def check_data_length(
    data: Union[np.ndarray, pd.Series, List],
    min_length: Optional[int] = None
) -> Tuple[int, bool]:
    """
    データ長をチェック
    
    Parameters
    ----------
    data : array-like
        チェックするデータ
    min_length : int, optional
        最小必要長（デフォルト: TIME_SERIES_CONSTANTS['MIN_DAYS']）
        
    Returns
    -------
    Tuple[int, bool]
        (データ長, 有効か)
    """
    if min_length is None:
        min_length = TIME_SERIES_CONSTANTS['MIN_DAYS']
    
    data_array = np.asarray(data)
    length = len(data_array)
    is_valid = length >= min_length
    
    return length, is_valid


def validate_numeric_data(
    data: Union[np.ndarray, pd.Series]
) -> Tuple[bool, List[str]]:
    """
    数値データの妥当性を検証
    
    Parameters
    ----------
    data : array-like
        検証するデータ
        
    Returns
    -------
    Tuple[bool, List[str]]
        (有効か, エラーメッセージのリスト)
    """
    errors = []
    data_array = np.asarray(data)
    
    # 数値型チェック
    if not np.issubdtype(data_array.dtype, np.number):
        errors.append("Data contains non-numeric values")
        return False, errors
    
    # NaNチェック
    n_nan = np.sum(np.isnan(data_array))
    if n_nan > 0:
        errors.append(f"Data contains {n_nan} NaN values")
    
    # Infチェック
    n_inf = np.sum(np.isinf(data_array))
    if n_inf > 0:
        errors.append(f"Data contains {n_inf} Inf values")
    
    # 負の値チェック（売上データの場合）
    n_negative = np.sum(data_array < 0)
    if n_negative > 0:
        errors.append(f"Data contains {n_negative} negative values")
    
    is_valid = len(errors) == 0 or (n_nan == 0 and n_inf == 0)
    return is_valid, errors


def validate_date_continuity(
    dates: Union[pd.DatetimeIndex, pd.Series]
) -> Tuple[bool, List[str]]:
    """
    日付の連続性を検証
    
    Parameters
    ----------
    dates : pd.DatetimeIndex or pd.Series
        検証する日付データ
        
    Returns
    -------
    Tuple[bool, List[str]]
        (連続か, 警告メッセージのリスト)
    """
    warnings = []
    
    if not isinstance(dates, pd.DatetimeIndex):
        try:
            dates = pd.DatetimeIndex(dates)
        except Exception:
            return False, ["Cannot convert to DatetimeIndex"]
    
    # ソートされているかチェック
    if not dates.is_monotonic_increasing:
        warnings.append("Dates are not sorted in ascending order")
    
    # 重複チェック
    if dates.has_duplicates:
        warnings.append(f"Duplicate dates found: {dates[dates.duplicated()].tolist()}")
    
    # ギャップチェック
    if len(dates) > 1:
        date_diff = dates.to_series().diff()
        max_gap = date_diff.max()
        
        if max_gap > pd.Timedelta(days=TIME_SERIES_CONSTANTS['CONSECUTIVE_MISSING_THRESHOLD']):
            warnings.append(f"Large date gap found: {max_gap.days} days")
    
    is_continuous = len(warnings) == 0
    return is_continuous, warnings


def validate_store_code(
    store_cd: str,
    pattern: Optional[str] = None
) -> bool:
    """
    店舗コードの妥当性を検証
    
    Parameters
    ----------
    store_cd : str
        店舗コード
    pattern : str, optional
        検証パターン（正規表現）
        
    Returns
    -------
    bool
        有効か
    """
    import re
    
    if not store_cd:
        return False
    
    # デフォルトパターン: 英数字、ハイフン、アンダースコア
    if pattern is None:
        pattern = r'^[A-Za-z0-9_-]+$'
    
    return bool(re.match(pattern, str(store_cd)))