"""
データ正規化処理モジュール

売上データの正規化処理を提供する。異なる規模の店舗間での
比較を可能にするため、複数の正規化手法をサポート。
"""

from typing import Union, List, Optional, Literal
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import warnings


NormalizationMethod = Literal["z-score", "min-max", "robust", "first-day-ratio", "mean-ratio"]


class DataNormalizer:
    """
    時系列データの正規化を行うクラス
    
    複数の正規化手法をサポートし、売上データの特性に応じた
    適切な正規化を提供する。
    """
    
    def __init__(self, method: NormalizationMethod = "z-score"):
        """
        Parameters
        ----------
        method : str, default='z-score'
            正規化手法
            - 'z-score': 標準化（平均0、標準偏差1）
            - 'min-max': 最小最大正規化（0-1の範囲）
            - 'robust': ロバスト正規化（外れ値に強い）
            - 'first-day-ratio': 初日売上に対する比率
            - 'mean-ratio': 平均売上に対する比率
        """
        self.method = method
        self._validate_method()
        self._scaler = self._get_scaler()
        self._reference_values = {}
    
    def _validate_method(self):
        """正規化手法の検証"""
        valid_methods = ["z-score", "min-max", "robust", "first-day-ratio", "mean-ratio"]
        if self.method not in valid_methods:
            raise ValueError(
                f"Invalid normalization method: {self.method}. "
                f"Must be one of {valid_methods}"
            )
    
    def _get_scaler(self):
        """scikit-learnのスケーラーを取得"""
        if self.method == "z-score":
            return StandardScaler()
        elif self.method == "min-max":
            return MinMaxScaler()
        elif self.method == "robust":
            return RobustScaler()
        else:
            return None  # カスタム正規化の場合
    
    def fit(self, data: Union[np.ndarray, pd.Series, pd.DataFrame]) -> "DataNormalizer":
        """
        正規化パラメータを学習
        
        Parameters
        ----------
        data : array-like
            学習用データ
            
        Returns
        -------
        self : DataNormalizer
            学習済みのインスタンス
        """
        data_array = self._to_numpy(data)
        
        if self._scaler is not None:
            # scikit-learnのスケーラーを使用
            if data_array.ndim == 1:
                data_array = data_array.reshape(-1, 1)
            self._scaler.fit(data_array)
        else:
            # カスタム正規化の場合は参照値を保存
            if self.method == "first-day-ratio":
                if data_array.ndim == 1:
                    self._reference_values["first_value"] = data_array[0]
                else:
                    self._reference_values["first_values"] = data_array[0, :]
            elif self.method == "mean-ratio":
                if data_array.ndim == 1:
                    self._reference_values["mean_value"] = np.mean(data_array)
                else:
                    self._reference_values["mean_values"] = np.mean(data_array, axis=0)
        
        return self
    
    def transform(self, data: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
        """
        データを正規化
        
        Parameters
        ----------
        data : array-like
            正規化するデータ
            
        Returns
        -------
        np.ndarray
            正規化されたデータ
        """
        data_array = self._to_numpy(data)
        original_shape = data_array.shape
        
        if self._scaler is not None:
            # scikit-learnのスケーラーを使用
            if data_array.ndim == 1:
                data_array = data_array.reshape(-1, 1)
            normalized = self._scaler.transform(data_array)
            if len(original_shape) == 1:
                normalized = normalized.flatten()
        else:
            # カスタム正規化
            normalized = self._custom_transform(data_array)
        
        return normalized
    
    def fit_transform(self, data: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
        """
        学習と変換を同時に実行
        
        Parameters
        ----------
        data : array-like
            正規化するデータ
            
        Returns
        -------
        np.ndarray
            正規化されたデータ
        """
        return self.fit(data).transform(data)
    
    def _custom_transform(self, data: np.ndarray) -> np.ndarray:
        """カスタム正規化の実装"""
        if self.method == "first-day-ratio":
            if data.ndim == 1:
                first_val = self._reference_values.get("first_value", data[0])
                if first_val == 0:
                    warnings.warn("First day value is 0, returning original data")
                    return data
                return data / first_val
            else:
                first_vals = self._reference_values.get("first_values", data[0, :])
                # ゼロ除算を避ける
                first_vals = np.where(first_vals == 0, 1, first_vals)
                return data / first_vals
                
        elif self.method == "mean-ratio":
            if data.ndim == 1:
                mean_val = self._reference_values.get("mean_value", np.mean(data))
                if mean_val == 0:
                    warnings.warn("Mean value is 0, returning original data")
                    return data
                return data / mean_val
            else:
                mean_vals = self._reference_values.get("mean_values", np.mean(data, axis=0))
                # ゼロ除算を避ける
                mean_vals = np.where(mean_vals == 0, 1, mean_vals)
                return data / mean_vals
        
        return data
    
    def inverse_transform(self, data: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
        """
        正規化を元に戻す
        
        Parameters
        ----------
        data : array-like
            正規化されたデータ
            
        Returns
        -------
        np.ndarray
            元のスケールのデータ
        """
        data_array = self._to_numpy(data)
        original_shape = data_array.shape
        
        if self._scaler is not None:
            # scikit-learnのスケーラーを使用
            if data_array.ndim == 1:
                data_array = data_array.reshape(-1, 1)
            denormalized = self._scaler.inverse_transform(data_array)
            if len(original_shape) == 1:
                denormalized = denormalized.flatten()
        else:
            # カスタム正規化の逆変換
            denormalized = self._custom_inverse_transform(data_array)
        
        return denormalized
    
    def _custom_inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """カスタム正規化の逆変換"""
        if self.method == "first-day-ratio":
            if data.ndim == 1:
                first_val = self._reference_values.get("first_value", 1)
                return data * first_val
            else:
                first_vals = self._reference_values.get("first_values", np.ones(data.shape[1]))
                return data * first_vals
                
        elif self.method == "mean-ratio":
            if data.ndim == 1:
                mean_val = self._reference_values.get("mean_value", 1)
                return data * mean_val
            else:
                mean_vals = self._reference_values.get("mean_values", np.ones(data.shape[1]))
                return data * mean_vals
        
        return data
    
    def _to_numpy(self, data: Union[np.ndarray, pd.Series, pd.DataFrame]) -> np.ndarray:
        """データをnumpy配列に変換"""
        if isinstance(data, (pd.Series, pd.DataFrame)):
            return data.values
        return np.asarray(data)
    
    def normalize_multiple_series(
        self,
        series_dict: Dict[str, Union[np.ndarray, pd.Series, List[float]]],
        fit_on: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        複数の時系列データを一括で正規化
        
        Parameters
        ----------
        series_dict : dict
            店舗IDと時系列データの辞書
        fit_on : str, optional
            学習に使用する特定の店舗ID（指定しない場合は各系列で個別に学習）
            
        Returns
        -------
        dict
            正規化された時系列データの辞書
        """
        normalized_dict = {}
        
        if fit_on is not None:
            # 特定の店舗で学習
            if fit_on not in series_dict:
                raise ValueError(f"Store code '{fit_on}' not found in series_dict")
            self.fit(series_dict[fit_on])
            
            # 全店舗に適用
            for store_cd, series in series_dict.items():
                normalized_dict[store_cd] = self.transform(series)
        else:
            # 各店舗で個別に正規化
            for store_cd, series in series_dict.items():
                normalized_dict[store_cd] = self.fit_transform(series)
        
        return normalized_dict
    
    @staticmethod
    def compare_normalization_methods(
        data: Union[np.ndarray, pd.Series],
        methods: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        複数の正規化手法を比較（デバッグ・分析用）
        
        Parameters
        ----------
        data : array-like
            比較用データ
        methods : list, optional
            比較する正規化手法のリスト
            
        Returns
        -------
        pd.DataFrame
            各手法で正規化されたデータの比較表
        """
        if methods is None:
            methods = ["z-score", "min-max", "robust", "first-day-ratio", "mean-ratio"]
        
        results = {"original": data}
        
        for method in methods:
            try:
                normalizer = DataNormalizer(method=method)
                normalized = normalizer.fit_transform(data)
                results[method] = normalized
            except Exception as e:
                warnings.warn(f"Failed to apply {method}: {e}")
                results[method] = np.nan
        
        # DataFrameに変換
        max_len = max(len(v) if hasattr(v, '__len__') else 1 for v in results.values())
        df_data = {}
        for key, value in results.items():
            if hasattr(value, '__len__'):
                df_data[key] = value
            else:
                df_data[key] = [value] * max_len
        
        return pd.DataFrame(df_data)