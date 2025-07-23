"""
統計処理ユーティリティ

共通の統計処理を提供する
"""

from typing import Union, Tuple, Optional, List
import numpy as np
import pandas as pd
from scipy import stats

from ..config import STATISTICS_CONSTANTS, QUALITY_CONSTANTS


def detect_outliers_zscore(
    data: Union[np.ndarray, pd.Series],
    threshold: Optional[float] = None
) -> np.ndarray:
    """
    Z-score法による異常値検出
    
    Parameters
    ----------
    data : array-like
        検査するデータ
    threshold : float, optional
        異常値判定の閾値（デフォルト: QUALITY_CONSTANTS['OUTLIER_THRESHOLD']）
        
    Returns
    -------
    np.ndarray
        異常値フラグ（True: 異常値）
    """
    if threshold is None:
        threshold = QUALITY_CONSTANTS['OUTLIER_THRESHOLD']
    
    data_array = np.asarray(data)
    mean = np.nanmean(data_array)
    std = np.nanstd(data_array)
    
    if std == 0:
        return np.zeros(len(data_array), dtype=bool)
    
    z_scores = np.abs((data_array - mean) / std)
    return z_scores > threshold


def detect_outliers_iqr(
    data: Union[np.ndarray, pd.Series],
    multiplier: Optional[float] = None
) -> np.ndarray:
    """
    IQR法による異常値検出
    
    Parameters
    ----------
    data : array-like
        検査するデータ
    multiplier : float, optional
        IQR倍率（デフォルト: STATISTICS_CONSTANTS['IQR_MULTIPLIER']）
        
    Returns
    -------
    np.ndarray
        異常値フラグ（True: 異常値）
    """
    if multiplier is None:
        multiplier = STATISTICS_CONSTANTS['IQR_MULTIPLIER']
    
    data_array = np.asarray(data)
    q1 = np.percentile(data_array[~np.isnan(data_array)], STATISTICS_CONSTANTS['QUARTILE_FIRST'])
    q3 = np.percentile(data_array[~np.isnan(data_array)], STATISTICS_CONSTANTS['QUARTILE_THIRD'])
    iqr = q3 - q1
    
    lower_bound = q1 - multiplier * iqr
    upper_bound = q3 + multiplier * iqr
    
    return (data_array < lower_bound) | (data_array > upper_bound)


def calculate_basic_statistics(
    data: Union[np.ndarray, pd.Series]
) -> dict:
    """
    基本統計量を計算
    
    Parameters
    ----------
    data : array-like
        計算対象のデータ
        
    Returns
    -------
    dict
        基本統計量の辞書
    """
    data_array = np.asarray(data)
    valid_data = data_array[~np.isnan(data_array)]
    
    if len(valid_data) == 0:
        return {
            'mean': np.nan,
            'std': np.nan,
            'min': np.nan,
            'max': np.nan,
            'median': np.nan,
            'q1': np.nan,
            'q3': np.nan,
            'count': 0,
            'missing': len(data_array)
        }
    
    return {
        'mean': np.mean(valid_data),
        'std': np.std(valid_data),
        'min': np.min(valid_data),
        'max': np.max(valid_data),
        'median': np.median(valid_data),
        'q1': np.percentile(valid_data, STATISTICS_CONSTANTS['QUARTILE_FIRST']),
        'q3': np.percentile(valid_data, STATISTICS_CONSTANTS['QUARTILE_THIRD']),
        'count': len(valid_data),
        'missing': len(data_array) - len(valid_data)
    }


def find_zero_runs(data: Union[np.ndarray, pd.Series]) -> List[int]:
    """
    ゼロの連続を検出
    
    Parameters
    ----------
    data : array-like
        検査するデータ
        
    Returns
    -------
    List[int]
        各ゼロ連続の長さ
    """
    data_array = np.asarray(data)
    is_zero = (data_array == 0).astype(int)
    runs = []
    
    if len(is_zero) == 0:
        return runs
    
    # 連続の開始と終了を検出
    diff = np.diff(np.concatenate([[0], is_zero, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    
    for start, end in zip(starts, ends):
        runs.append(end - start)
    
    return runs


def calculate_confidence_interval(
    data: Union[np.ndarray, pd.Series],
    confidence_level: float = 0.95
) -> Tuple[float, float]:
    """
    信頼区間を計算
    
    Parameters
    ----------
    data : array-like
        計算対象のデータ
    confidence_level : float, default=0.95
        信頼水準
        
    Returns
    -------
    Tuple[float, float]
        (下限, 上限)
    """
    data_array = np.asarray(data)
    valid_data = data_array[~np.isnan(data_array)]
    
    if len(valid_data) == 0:
        return (np.nan, np.nan)
    
    mean = np.mean(valid_data)
    sem = stats.sem(valid_data)
    
    if sem == 0:
        return (mean, mean)
    
    # t分布を使用
    alpha = 1 - confidence_level
    df = len(valid_data) - 1
    t_critical = stats.t.ppf(1 - alpha/2, df)
    
    margin_of_error = t_critical * sem
    
    return (mean - margin_of_error, mean + margin_of_error)