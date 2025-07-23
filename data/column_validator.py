"""
列名検証・修正モジュール

DataFrameの列名を検証し、必要に応じて修正する責任のみを持つ
"""

from typing import List, Optional
import pandas as pd
import warnings

from ..config.defaults import DATA_FORMAT_DEFAULTS, LOADER_DEFAULTS


class ColumnValidator:
    """
    列名検証・修正専用クラス
    
    単一責任: DataFrameの列名を検証・修正する
    """
    
    def __init__(
        self,
        date_column: Optional[str] = None,
        sales_column: Optional[str] = None,
        store_cd_column: Optional[str] = None,
        auto_detect: bool = True
    ):
        """
        Parameters
        ----------
        date_column : str, optional
            期待する日付列名
        sales_column : str, optional
            期待する売上列名
        store_cd_column : str, optional
            期待する店舗コード列名
        auto_detect : bool, default=True
            列名の自動検出を行うか
        """
        self.expected_date_column = date_column or DATA_FORMAT_DEFAULTS['date_column']
        self.expected_sales_column = sales_column or DATA_FORMAT_DEFAULTS['sales_column']
        self.expected_store_cd_column = store_cd_column or DATA_FORMAT_DEFAULTS['store_cd_column']
        self.auto_detect = auto_detect
    
    def validate_and_fix_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        列名を検証し、必要に応じて修正する
        
        Parameters
        ----------
        df : pd.DataFrame
            検証・修正するDataFrame
            
        Returns
        -------
        pd.DataFrame
            列名が修正されたDataFrame
        """
        df = df.copy()
        missing_columns = []
        
        # 店舗コード列の検証・修正
        if self.expected_store_cd_column not in df.columns:
            if self.auto_detect:
                candidate = self._find_column_candidate(
                    df.columns, 
                    LOADER_DEFAULTS['column_mapping']['store_columns']
                )
                if candidate:
                    warnings.warn(
                        f"Using '{candidate}' as store code column instead of '{self.expected_store_cd_column}'"
                    )
                    df = df.rename(columns={candidate: self.expected_store_cd_column})
                else:
                    missing_columns.append(self.expected_store_cd_column)
            else:
                missing_columns.append(self.expected_store_cd_column)
        
        # 売上列の検証・修正
        if self.expected_sales_column not in df.columns:
            if self.auto_detect:
                candidate = self._find_column_candidate(
                    df.columns, 
                    LOADER_DEFAULTS['column_mapping']['sales_columns']
                )
                if candidate:
                    warnings.warn(
                        f"Using '{candidate}' as sales column instead of '{self.expected_sales_column}'"
                    )
                    df = df.rename(columns={candidate: self.expected_sales_column})
                else:
                    missing_columns.append(self.expected_sales_column)
            else:
                missing_columns.append(self.expected_sales_column)
        
        # 日付列の検証・修正（オプション）
        if self.expected_date_column in df.columns:
            df = self._fix_date_column(df)
        elif self.auto_detect:
            candidate = self._find_column_candidate(
                df.columns,
                LOADER_DEFAULTS['column_mapping']['date_columns']
            )
            if candidate:
                warnings.warn(
                    f"Using '{candidate}' as date column instead of '{self.expected_date_column}'"
                )
                df = df.rename(columns={candidate: self.expected_date_column})
                df = self._fix_date_column(df)
        
        if missing_columns:
            raise ValueError(f"Required columns not found: {missing_columns}")
        
        return df
    
    def validate_required_columns(self, df: pd.DataFrame, required_columns: List[str]) -> None:
        """
        必須列の存在を検証
        
        Parameters
        ----------
        df : pd.DataFrame
            検証するDataFrame
        required_columns : List[str]
            必須列名のリスト
        """
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Required columns not found: {missing_columns}")
    
    def _find_column_candidate(self, columns: pd.Index, candidates: List[str]) -> Optional[str]:
        """
        候補列名から実際の列名を検索
        
        Parameters
        ----------
        columns : pd.Index
            DataFrameの列名
        candidates : List[str]
            候補となる列名のリスト
            
        Returns
        -------
        Optional[str]
            見つかった列名、見つからない場合はNone
        """
        for candidate in candidates:
            if candidate in columns:
                return candidate
        
        # 部分一致での検索
        for candidate in candidates:
            for col in columns:
                if candidate.lower() in col.lower():
                    return col
        
        return None
    
    def _fix_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        日付列のデータ型を修正
        
        Parameters
        ----------
        df : pd.DataFrame
            修正するDataFrame
            
        Returns
        -------
        pd.DataFrame
            日付列が修正されたDataFrame
        """
        try:
            df[self.expected_date_column] = pd.to_datetime(
                df[self.expected_date_column], 
                errors='coerce'
            )
        except Exception:
            warnings.warn(f"Failed to parse date column '{self.expected_date_column}'")
        
        return df