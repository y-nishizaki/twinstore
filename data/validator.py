"""
データ検証モジュール

入力データの形式や内容を検証し、TwinStoreパッケージで
使用可能なデータであることを確認する。
"""

from typing import Union, List, Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator, Field
from datetime import datetime
import warnings

from ..config import VALIDATION_RULES, ERROR_MESSAGES
from ..config.validation import get_validation_rule, get_validation_message
from ..utils import find_zero_runs


class ValidationResult:
    """検証結果を格納するクラス"""
    
    def __init__(self):
        self.is_valid = True
        self.errors = []
        self.warnings = []
        self.summary = {}
    
    def add_error(self, message: str):
        """エラーを追加"""
        self.errors.append(message)
        self.is_valid = False
    
    def add_warning(self, message: str):
        """警告を追加"""
        self.warnings.append(message)
    
    def set_summary(self, key: str, value: Any):
        """サマリー情報を設定"""
        self.summary[key] = value
    
    def __str__(self):
        """文字列表現"""
        status = "Valid" if self.is_valid else "Invalid"
        return (
            f"ValidationResult(status={status}, "
            f"errors={len(self.errors)}, warnings={len(self.warnings)})"
        )
    
    def get_report(self) -> str:
        """詳細レポートを取得"""
        report = []
        report.append(f"Validation {'Passed' if self.is_valid else 'Failed'}")
        report.append("=" * 50)
        
        if self.errors:
            report.append("\nErrors:")
            for i, error in enumerate(self.errors, 1):
                report.append(f"  {i}. {error}")
        
        if self.warnings:
            report.append("\nWarnings:")
            for i, warning in enumerate(self.warnings, 1):
                report.append(f"  {i}. {warning}")
        
        if self.summary:
            report.append("\nSummary:")
            for key, value in self.summary.items():
                report.append(f"  {key}: {value}")
        
        return "\n".join(report)


class SalesDataSchema(BaseModel):
    """売上データのスキーマ定義"""
    store_cd: str = Field(..., description="店舗コード")
    date: datetime = Field(..., description="日付")
    sales: float = Field(..., ge=0, description="売上金額")
    
    @field_validator('sales')
    @classmethod
    def validate_sales(cls, v):
        if np.isnan(v) or np.isinf(v):
            raise ValueError("Sales value cannot be NaN or Inf")
        return v


class StoreAttributesSchema(BaseModel):
    """店舗属性のスキーマ定義"""
    store_cd: str = Field(..., description="店舗コード")
    store_type: Optional[str] = Field(None, description="店舗タイプ")
    location: Optional[str] = Field(None, description="立地")
    area: Optional[float] = Field(None, ge=0, description="店舗面積")
    opening_date: Optional[datetime] = Field(None, description="開店日")


class DataValidator:
    """
    データ検証を行うクラス
    
    TwinStoreパッケージで使用する各種データの形式と内容を検証する。
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Parameters
        ----------
        strict_mode : bool, default=False
            厳格モード（警告もエラーとして扱う）
        """
        self.strict_mode = strict_mode
    
    def validate_sales_data(
        self,
        data: Union[pd.DataFrame, Dict[str, np.ndarray], np.ndarray],
        data_format: str = "auto"
    ) -> ValidationResult:
        """
        売上データの検証
        
        Parameters
        ----------
        data : various
            検証する売上データ
        data_format : str, default='auto'
            データ形式 ('dataframe', 'dict', 'array', 'auto')
            
        Returns
        -------
        ValidationResult
            検証結果
        """
        result = ValidationResult()
        
        # データ形式の自動判定
        if data_format == "auto":
            data_format = self._detect_format(data)
            result.set_summary("detected_format", data_format)
        
        # 形式別の検証
        if data_format == "dataframe":
            self._validate_dataframe(data, result)
        elif data_format == "dict":
            self._validate_dict(data, result)
        elif data_format == "array":
            self._validate_array(data, result)
        else:
            result.add_error(f"Unknown data format: {data_format}")
        
        # 厳格モードの処理
        if self.strict_mode and result.warnings:
            for warning in result.warnings:
                result.add_error(f"[Strict Mode] {warning}")
            result.warnings = []
        
        return result
    
    def validate_store_attributes(
        self,
        attributes: pd.DataFrame
    ) -> ValidationResult:
        """
        店舗属性データの検証
        
        Parameters
        ----------
        attributes : pd.DataFrame
            店舗属性データ
            
        Returns
        -------
        ValidationResult
            検証結果
        """
        result = ValidationResult()
        
        # 必須カラムの確認
        required_cols = ["store_cd"]
        missing_cols = [col for col in required_cols if col not in attributes.columns]
        if missing_cols:
            result.add_error(f"Missing required columns: {missing_cols}")
            return result
        
        # 店舗コードの重複チェック
        duplicates = attributes["store_cd"].duplicated()
        if duplicates.any():
            dup_cds = attributes.loc[duplicates, "store_cd"].tolist()
            result.add_error(f"Duplicate store codes found: {dup_cds[:5]}...")
        
        # データ型の検証
        if "area" in attributes.columns:
            try:
                pd.to_numeric(attributes["area"])
            except (ValueError, TypeError):
                result.add_warning("'area' column contains non-numeric values")
        
        if "opening_date" in attributes.columns:
            try:
                pd.to_datetime(attributes["opening_date"])
            except (ValueError, TypeError):
                result.add_warning("'opening_date' column contains invalid dates")
        
        result.set_summary("n_stores", len(attributes))
        result.set_summary("columns", list(attributes.columns))
        
        return result
    
    def validate_prediction_input(
        self,
        new_store_sales: Union[np.ndarray, pd.Series, List[float]],
        min_days: Optional[int] = None
    ) -> ValidationResult:
        """
        予測入力データの検証
        
        Parameters
        ----------
        new_store_sales : array-like
            新規店舗の売上データ
        min_days : int, default=7
            最小必要日数
            
        Returns
        -------
        ValidationResult
            検証結果
        """
        result = ValidationResult()
        
        # デフォルト値を設定から取得
        if min_days is None:
            volume_rules = get_validation_rule('data_volume')
            min_days = volume_rules.get('min_days', 3)
        
        # numpy配列に変換
        try:
            sales_array = np.asarray(new_store_sales).flatten()
        except Exception as e:
            result.add_error(f"Failed to convert to array: {e}")
            return result
        
        # データ長の確認
        n_days = len(sales_array)
        result.set_summary("n_days", n_days)
        
        if n_days < min_days:
            error_msg = get_validation_message(
                'insufficient_data',
                min_days=min_days
            )
            result.add_error(error_msg)
        
        # 数値検証
        if not np.issubdtype(sales_array.dtype, np.number):
            result.add_error("Data contains non-numeric values")
            return result
        
        # NaN/Inf チェック
        n_nan = np.sum(np.isnan(sales_array))
        n_inf = np.sum(np.isinf(sales_array))
        
        if n_nan > 0:
            result.add_error(f"Data contains {n_nan} NaN values")
        if n_inf > 0:
            result.add_error(f"Data contains {n_inf} Inf values")
        
        # 負の値チェック
        n_negative = np.sum(sales_array < 0)
        if n_negative > 0:
            result.add_warning(f"Data contains {n_negative} negative values")
        
        # ゼロの連続チェック
        zero_runs = find_zero_runs(sales_array)
        if zero_runs:
            max_run = max(zero_runs)
            # 設定から閾値を取得
            volume_rules = get_validation_rule('data_volume')
            consecutive_threshold = volume_rules.get('max_gap_days', 3)
            
            if max_run >= consecutive_threshold:
                result.add_warning(
                    f"Data contains {max_run} consecutive days with zero sales"
                )
        
        # 統計サマリー
        if len(sales_array) > 0 and not result.errors:
            result.set_summary("mean_sales", np.mean(sales_array))
            result.set_summary("std_sales", np.std(sales_array))
            result.set_summary("min_sales", np.min(sales_array))
            result.set_summary("max_sales", np.max(sales_array))
        
        # 厳格モードの処理
        if self.strict_mode and result.warnings:
            for warning in result.warnings:
                result.add_error(f"[Strict Mode] {warning}")
            result.warnings = []
        
        return result
    
    def _detect_format(self, data: Any) -> str:
        """データ形式を自動判定"""
        if isinstance(data, pd.DataFrame):
            return "dataframe"
        elif isinstance(data, dict):
            return "dict"
        elif isinstance(data, (np.ndarray, list)):
            return "array"
        else:
            return "unknown"
    
    def _validate_dataframe(self, df: pd.DataFrame, result: ValidationResult):
        """DataFrameの検証"""
        # 形状の確認
        n_rows, n_cols = df.shape
        result.set_summary("shape", (n_rows, n_cols))
        
        if n_rows == 0:
            result.add_error("DataFrame is empty")
            return
        
        if n_cols == 0:
            result.add_error("DataFrame has no columns")
            return
        
        # インデックスが日付型かチェック
        if not isinstance(df.index, pd.DatetimeIndex):
            result.add_warning("Index is not DatetimeIndex")
        else:
            # 日付の連続性チェック
            date_diff = df.index.to_series().diff()
            if date_diff.dropna().nunique() > 1:
                result.add_warning("Dates are not consecutive")
        
        # 各列の検証
        for col in df.columns:
            col_data = df[col]
            
            # 数値型チェック
            if not pd.api.types.is_numeric_dtype(col_data):
                result.add_error(f"Column '{col}' is not numeric")
                continue
            
            # NaN チェック
            n_nan = col_data.isna().sum()
            if n_nan > 0:
                nan_ratio = n_nan / len(col_data)
                if nan_ratio > 0.1:  # 10%以上
                    result.add_warning(
                        f"Column '{col}' has {n_nan} ({nan_ratio:.1%}) missing values"
                    )
    
    def _validate_dict(self, data: dict, result: ValidationResult):
        """辞書形式データの検証"""
        if not data:
            result.add_error("Dictionary is empty")
            return
        
        result.set_summary("n_stores", len(data))
        
        # 各店舗のデータを検証
        lengths = []
        for store_cd, sales_data in data.items():
            # 配列に変換
            try:
                sales_array = np.asarray(sales_data)
            except (ValueError, TypeError) as e:
                result.add_error(f"Store '{store_cd}': Failed to convert to array: {e}")
                continue
            
            lengths.append(len(sales_array))
            
            # 基本的な検証
            if len(sales_array) == 0:
                result.add_error(f"Store '{store_cd}': Empty data")
            
            # NaN/Inf チェック
            if np.any(np.isnan(sales_array)):
                result.add_warning(f"Store '{store_cd}': Contains NaN values")
            if np.any(np.isinf(sales_array)):
                result.add_warning(f"Store '{store_cd}': Contains Inf values")
        
        # データ長の一貫性チェック
        if lengths:
            if len(set(lengths)) > 1:
                result.add_warning(
                    f"Inconsistent data lengths: min={min(lengths)}, max={max(lengths)}"
                )
            result.set_summary("data_length_range", (min(lengths), max(lengths)))
    
    def _validate_array(self, data: np.ndarray, result: ValidationResult):
        """配列データの検証"""
        if data.size == 0:
            result.add_error("Array is empty")
            return
        
        result.set_summary("shape", data.shape)
        
        # 1次元または2次元のみ許可
        if data.ndim > 2:
            result.add_error(f"Array dimension must be 1 or 2, got {data.ndim}")
            return
        
        # 数値型チェック
        if not np.issubdtype(data.dtype, np.number):
            result.add_error("Array contains non-numeric data")
            return
        
        # NaN/Inf チェック
        n_nan = np.sum(np.isnan(data))
        n_inf = np.sum(np.isinf(data))
        
        if n_nan > 0:
            result.add_warning(f"Array contains {n_nan} NaN values")
        if n_inf > 0:
            result.add_warning(f"Array contains {n_inf} Inf values")
    
    
    @staticmethod
    def validate_input_data(data: Any) -> bool:
        """
        簡易検証メソッド（互換性のため）
        
        Parameters
        ----------
        data : Any
            検証するデータ
            
        Returns
        -------
        bool
            検証結果（True: 有効、False: 無効）
        """
        validator = DataValidator()
        result = validator.validate_sales_data(data)
        return result.is_valid
    
    @staticmethod
    def validate_store_attributes(attributes: Optional[pd.DataFrame]) -> bool:
        """
        簡易検証メソッド（互換性のため）
        
        Parameters
        ----------
        attributes : pd.DataFrame
            店舗属性データ
            
        Returns
        -------
        bool
            検証結果（True: 有効、False: 無効）
        """
        # 基本的なチェック
        if attributes is None or attributes.empty:
            return False
        
        # インデックスにstore_cdがあるか、または列にstore_cdがあるかをチェック
        has_store_cd = (
            attributes.index.name == 'store_cd' or
            'store_cd' in attributes.columns or
            'store_cd' in str(attributes.index.name).lower()
        )
        
        return has_store_cd