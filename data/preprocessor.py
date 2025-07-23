"""
データ前処理モジュール

売上データの前処理機能を提供する。欠損値の補完、異常値の処理、
データの平滑化などを行う。
"""

from typing import Union, Dict, List, Optional, Literal, Any, Tuple
import numpy as np
import pandas as pd
from scipy import signal
from scipy.interpolate import interp1d
import warnings

from ..config import QUALITY_CONSTANTS, STATISTICS_CONSTANTS
from ..config.defaults import PREPROCESSOR_DEFAULTS
from ..utils import detect_outliers_iqr, detect_outliers_zscore, find_zero_runs


InterpolationMethod = Literal["linear", "forward", "backward", "nearest", "spline"]
SmoothingMethod = Literal["moving_average", "exponential", "savgol", "lowess"]


class DataPreprocessor:
    """
    時系列データの前処理を行うクラス
    
    欠損値補完、異常値処理、平滑化などの前処理機能を提供。
    """
    
    def __init__(
        self,
        missing_threshold: Optional[float] = None,
        outlier_method: Optional[str] = None,
        interpolation_method: Optional[InterpolationMethod] = None,
    ):
        """
        Parameters
        ----------
        missing_threshold : float, optional
            欠損値の許容割合（これを超えると警告）。Noneの場合はデフォルト値を使用
        outlier_method : str, optional
            異常値検出手法 ('iqr', 'zscore', 'isolation_forest')。Noneの場合はデフォルト値を使用
        interpolation_method : str, optional
            欠損値補間手法。Noneの場合はデフォルト値を使用
        """
        self.missing_threshold = missing_threshold if missing_threshold is not None else QUALITY_CONSTANTS['HIGH_MISSING_THRESHOLD']
        self.outlier_method = outlier_method or PREPROCESSOR_DEFAULTS['outlier_method']
        self.interpolation_method = interpolation_method or PREPROCESSOR_DEFAULTS['interpolation_method']
        self._preprocessing_log = []
    
    def preprocess(
        self,
        data: Union[pd.Series, pd.DataFrame, np.ndarray],
        handle_missing: bool = True,
        handle_outliers: bool = True,
        smooth_data: bool = False,
        smooth_params: Optional[Dict[str, Any]] = None,
    ) -> Union[pd.Series, pd.DataFrame, np.ndarray]:
        """
        データの前処理を実行
        
        Parameters
        ----------
        data : array-like
            前処理するデータ
        handle_missing : bool, default=True
            欠損値を処理するか
        handle_outliers : bool, default=True
            異常値を処理するか
        smooth_data : bool, default=False
            データを平滑化するか
        smooth_params : dict, optional
            平滑化のパラメータ
            
        Returns
        -------
        array-like
            前処理済みデータ（入力と同じ型）
        """
        self._preprocessing_log = []
        original_type = type(data)
        
        # pandas形式に変換
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                df = pd.Series(data)
            else:
                df = pd.DataFrame(data)
        elif isinstance(data, (list, tuple)):
            df = pd.Series(data)
        else:
            df = data.copy()
        
        # 処理前の統計情報を記録
        self._log_stats("Original", df)
        
        # 1. 欠損値処理
        if handle_missing:
            df = self._handle_missing_values(df)
            self._log_stats("After missing value handling", df)
        
        # 2. 異常値処理
        if handle_outliers:
            df = self._handle_outliers(df)
            self._log_stats("After outlier handling", df)
        
        # 3. 平滑化
        if smooth_data:
            df = self._smooth_data(df, smooth_params)
            self._log_stats("After smoothing", df)
        
        # 元の型に戻す
        if original_type == np.ndarray:
            return df.values
        elif original_type == list:
            return list(df.values) if isinstance(df, pd.Series) else list(df.values.flatten())
        elif original_type == tuple:
            return tuple(df.values) if isinstance(df, pd.Series) else tuple(df.values.flatten())
        return df
    
    def fill_missing_values(
        self,
        data: Union[pd.Series, pd.DataFrame],
        method: InterpolationMethod = None,
        limit: Optional[int] = None,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        欠損値を補完
        
        Parameters
        ----------
        data : pd.Series or pd.DataFrame
            補完するデータ
        method : str, optional
            補間手法（指定しない場合はインスタンスの設定を使用）
        limit : int, optional
            連続補間の最大数
            
        Returns
        -------
        pd.Series or pd.DataFrame
            補完済みデータ
        """
        if method is None:
            method = self.interpolation_method
        
        data = data.copy()
        
        if isinstance(data, pd.Series):
            return self._interpolate_series(data, method, limit)
        else:
            for col in data.columns:
                data[col] = self._interpolate_series(data[col], method, limit)
            return data
    
    def detect_outliers(
        self,
        data: Union[pd.Series, np.ndarray],
        method: Optional[str] = None,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        異常値を検出
        
        Parameters
        ----------
        data : array-like
            検査するデータ
        method : str, optional
            検出手法（指定しない場合はインスタンスの設定を使用）
        threshold : float, default=1.5
            異常値判定の閾値
            
        Returns
        -------
        np.ndarray
            異常値フラグ（True: 異常値）
        """
        if method is None:
            method = self.outlier_method
        
        if method == "iqr":
            return detect_outliers_iqr(data, threshold)
        elif method == "zscore":
            return detect_outliers_zscore(data, threshold)
        elif method == "isolation_forest":
            # Isolation Forestの簡易実装（IQRで代用）
            return detect_outliers_iqr(data, threshold or QUALITY_CONSTANTS['OUTLIER_THRESHOLD'])
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def smooth_timeseries(
        self,
        data: Union[pd.Series, np.ndarray],
        method: SmoothingMethod = "moving_average",
        **kwargs
    ) -> Union[pd.Series, np.ndarray]:
        """
        時系列データを平滑化
        
        Parameters
        ----------
        data : array-like
            平滑化するデータ
        method : str, default='moving_average'
            平滑化手法
        **kwargs
            手法固有のパラメータ
            
        Returns
        -------
        array-like
            平滑化されたデータ
        """
        original_type = type(data)
        data_array = np.asarray(data)
        
        if method == "moving_average":
            window = kwargs.get("window", PREPROCESSOR_DEFAULTS['smooth_window'])
            smoothed = self._moving_average(data_array, window)
        elif method == "exponential":
            alpha = kwargs.get("alpha", QUALITY_CONSTANTS['HIGH_MISSING_THRESHOLD'])
            smoothed = self._exponential_smoothing(data_array, alpha)
        elif method == "savgol":
            window = kwargs.get("window", PREPROCESSOR_DEFAULTS['smooth_window'])
            polyorder = kwargs.get("polyorder", 3)
            smoothed = self._savgol_filter(data_array, window, polyorder)
        elif method == "lowess":
            frac = kwargs.get("frac", QUALITY_CONSTANTS['MISSING_THRESHOLD'])
            smoothed = self._lowess_smoothing(data_array, frac)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        
        # 元の型に戻す
        if isinstance(original_type, pd.Series):
            return pd.Series(smoothed, index=data.index)
        return smoothed
    
    def preprocess_batch(
        self,
        data_dict: Dict[str, Union[pd.Series, np.ndarray]],
        **kwargs
    ) -> Dict[str, Union[pd.Series, np.ndarray]]:
        """
        複数の時系列データを一括前処理
        
        Parameters
        ----------
        data_dict : dict
            店舗IDと時系列データの辞書
        **kwargs
            preprocess メソッドに渡すパラメータ
            
        Returns
        -------
        dict
            前処理済みデータの辞書
        """
        processed_dict = {}
        
        for store_cd, data in data_dict.items():
            try:
                processed = self.preprocess(data, **kwargs)
                processed_dict[store_cd] = processed
            except Exception as e:
                warnings.warn(f"Failed to preprocess data for {store_cd}: {e}")
                # 処理に失敗した場合は元のデータを保持
                processed_dict[store_cd] = data
        
        return processed_dict
    
    def get_preprocessing_report(self) -> str:
        """前処理のレポートを取得"""
        if not self._preprocessing_log:
            return "No preprocessing performed yet."
        
        report = ["Preprocessing Report", "=" * 50]
        for entry in self._preprocessing_log:
            report.append(f"\n{entry['stage']}:")
            for key, value in entry['stats'].items():
                report.append(f"  {key}: {value}")
        
        return "\n".join(report)
    
    def _handle_missing_values(
        self,
        data: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """欠損値の処理"""
        # 欠損値の割合を確認
        if isinstance(data, pd.Series):
            missing_ratio = data.isna().sum() / len(data)
            if missing_ratio > self.missing_threshold:
                warnings.warn(
                    f"High missing value ratio: {missing_ratio:.1%}"
                )
        else:
            for col in data.columns:
                missing_ratio = data[col].isna().sum() / len(data)
                if missing_ratio > self.missing_threshold:
                    warnings.warn(
                        f"Column '{col}' has high missing value ratio: {missing_ratio:.1%}"
                    )
        
        # 補間
        return self.fill_missing_values(data)
    
    def _handle_outliers(
        self,
        data: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """異常値の処理"""
        if isinstance(data, pd.Series):
            outliers = self.detect_outliers(data)
            if outliers.any():
                # 異常値を前後の値の平均で置換
                data = self._replace_outliers(data, outliers)
        else:
            for col in data.columns:
                outliers = self.detect_outliers(data[col])
                if outliers.any():
                    data[col] = self._replace_outliers(data[col], outliers)
        
        return data
    
    def _smooth_data(
        self,
        data: Union[pd.Series, pd.DataFrame],
        params: Optional[Dict[str, Any]]
    ) -> Union[pd.Series, pd.DataFrame]:
        """データの平滑化"""
        if params is None:
            params = {"method": "moving_average", "window": PREPROCESSOR_DEFAULTS['smooth_window']}
        
        method = params.get("method", "moving_average")
        
        if isinstance(data, pd.Series):
            return self.smooth_timeseries(data, method, **params)
        else:
            for col in data.columns:
                data[col] = self.smooth_timeseries(data[col], method, **params)
            return data
    
    def _interpolate_series(
        self,
        series: pd.Series,
        method: str,
        limit: Optional[int]
    ) -> pd.Series:
        """Seriesの補間"""
        if method in ["linear", "nearest", "spline"]:
            # scipy.interpolateを使用
            valid_idx = ~series.isna()
            if valid_idx.sum() < 2:
                # 有効なデータが少なすぎる場合は前方補完
                return series.fillna(method='ffill').fillna(method='bfill')
            
            x = np.arange(len(series))
            if method == "spline":
                f = interp1d(x[valid_idx], series[valid_idx], kind='cubic',
                           bounds_error=False, fill_value='extrapolate')
            else:
                f = interp1d(x[valid_idx], series[valid_idx], kind=method,
                           bounds_error=False, fill_value='extrapolate')
            
            interpolated = f(x)
            result = pd.Series(interpolated, index=series.index)
            
            # 元のデータで有効だった部分は保持
            result[valid_idx] = series[valid_idx]
            
        elif method == "forward":
            result = series.fillna(method='ffill', limit=limit)
        elif method == "backward":
            result = series.fillna(method='bfill', limit=limit)
        else:
            # pandas標準の補間
            result = series.interpolate(method=method, limit=limit)
        
        return result
    
    
    def _replace_outliers(self, series: pd.Series, outliers: np.ndarray) -> pd.Series:
        """異常値を置換"""
        result = series.copy()
        
        for i in np.where(outliers)[0]:
            # 前後の正常値の平均で置換
            window_start = max(0, i - 2)
            window_end = min(len(series), i + 3)
            window_data = series.iloc[window_start:window_end]
            
            # 異常値を除いた平均
            normal_values = window_data[~outliers[window_start:window_end]]
            if len(normal_values) > 0:
                result.iloc[i] = normal_values.mean()
            else:
                # 周囲に正常値がない場合は全体の中央値
                result.iloc[i] = series[~outliers].median()
        
        return result
    
    def _moving_average(self, data: np.ndarray, window: int) -> np.ndarray:
        """移動平均"""
        if window > len(data):
            window = len(data)
        
        # pandasの移動平均を使用
        series = pd.Series(data)
        smoothed = series.rolling(window=window, center=True, min_periods=1).mean()
        
        return smoothed.values
    
    def _exponential_smoothing(self, data: np.ndarray, alpha: float) -> np.ndarray:
        """指数平滑化"""
        smoothed = np.zeros_like(data)
        smoothed[0] = data[0]
        
        for i in range(1, len(data)):
            smoothed[i] = alpha * data[i] + (1 - alpha) * smoothed[i-1]
        
        return smoothed
    
    def _savgol_filter(self, data: np.ndarray, window: int, polyorder: int) -> np.ndarray:
        """Savitzky-Golayフィルタ"""
        if window > len(data):
            window = len(data)
        if window % 2 == 0:
            window -= 1  # 奇数にする
        
        if polyorder >= window:
            polyorder = window - 1
        
        return signal.savgol_filter(data, window, polyorder)
    
    def _lowess_smoothing(self, data: np.ndarray, frac: float) -> np.ndarray:
        """LOWESS平滑化（簡易版）"""
        # 実際の実装では statsmodels の lowess を使用
        # ここでは移動平均で代用
        window = int(len(data) * frac)
        if window < 3:
            window = 3
        
        return self._moving_average(data, window)
    
    def _log_stats(self, stage: str, data: Union[pd.Series, pd.DataFrame]):
        """統計情報をログに記録"""
        stats = {}
        
        if isinstance(data, pd.Series):
            stats["mean"] = data.mean()
            stats["std"] = data.std()
            stats["missing"] = data.isna().sum()
            stats["min"] = data.min()
            stats["max"] = data.max()
        elif isinstance(data, pd.DataFrame):
            stats["shape"] = data.shape
            stats["missing_total"] = data.isna().sum().sum()
            stats["mean_values"] = data.mean().mean()
        else:
            # listなど他の形式の場合
            try:
                data_series = pd.Series(data)
                stats["length"] = len(data_series)
                if not data_series.isna().all():
                    stats["mean"] = data_series.mean()
            except Exception:
                stats["type"] = str(type(data))
        
        self._preprocessing_log.append({
            "stage": stage,
            "stats": stats
        })