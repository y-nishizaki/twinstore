"""
データ変換モジュール

DataFrameを他の形式（dict、特定構造）に変換する責任のみを持つ
"""

from typing import Dict, Union
import numpy as np
import pandas as pd

from ..config.defaults import DATA_FORMAT_DEFAULTS


class DataTransformer:
    """
    データ変換専用クラス
    
    単一責任: DataFrameを他の形式に変換する
    """
    
    def __init__(
        self,
        date_column: str = None,
        sales_column: str = None,
        store_cd_column: str = None
    ):
        """
        Parameters
        ----------
        date_column : str
            日付列名
        sales_column : str
            売上列名
        store_cd_column : str
            店舗コード列名
        """
        self.date_column = date_column or DATA_FORMAT_DEFAULTS['date_column']
        self.sales_column = sales_column or DATA_FORMAT_DEFAULTS['sales_column']
        self.store_cd_column = store_cd_column or DATA_FORMAT_DEFAULTS['store_cd_column']
    
    def to_dict_format(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """
        DataFrameを辞書形式に変換
        
        Parameters
        ----------
        df : pd.DataFrame
            変換するDataFrame
            
        Returns
        -------
        Dict[str, np.ndarray]
            店舗コードをキーとする売上データの辞書
        """
        result = {}
        
        for store_cd in df[self.store_cd_column].unique():
            store_data = df[df[self.store_cd_column] == store_cd].copy()
            
            # 日付でソート（日付列がある場合）
            if self.date_column in store_data.columns:
                store_data = store_data.sort_values(self.date_column)
            
            # 売上データを抽出
            sales_data = store_data[self.sales_column].values
            result[store_cd] = sales_data
        
        return result
    
    def to_timeseries_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrameを時系列形式に変換（店舗を列とする）
        
        Parameters
        ----------
        df : pd.DataFrame
            変換するDataFrame
            
        Returns
        -------
        pd.DataFrame
            時系列形式のDataFrame
        """
        if self.date_column in df.columns:
            # 日付をインデックスとするピボット
            pivot_df = df.pivot_table(
                index=self.date_column,
                columns=self.store_cd_column,
                values=self.sales_column,
                aggfunc='first'
            )
            return pivot_df
        else:
            # 日付列がない場合は店舗別に並べる
            pivot_df = df.set_index(self.store_cd_column)[self.sales_column].unstack()
            return pivot_df.T
    
    def to_batch_format(self, df: pd.DataFrame) -> Dict[str, Union[np.ndarray, pd.Series]]:
        """
        DataFrameをバッチ処理用の辞書形式に変換
        
        Parameters
        ----------
        df : pd.DataFrame
            変換するDataFrame
            
        Returns
        -------
        Dict[str, Union[np.ndarray, pd.Series]]
            店舗コードをキーとするデータの辞書
        """
        result = {}
        
        for store_cd in df[self.store_cd_column].unique():
            store_data = df[df[self.store_cd_column] == store_cd]
            
            # 日付でソート（日付列がある場合）
            if self.date_column in store_data.columns:
                store_data = store_data.sort_values(self.date_column)
            
            # 売上データを抽出
            sales_data = store_data[self.sales_column].values
            result[store_cd] = sales_data
        
        return result
    
    def to_attributes_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        DataFrameを店舗属性形式に変換（店舗コードをインデックスに）
        
        Parameters
        ----------
        df : pd.DataFrame
            変換するDataFrame
            
        Returns
        -------
        pd.DataFrame
            店舗コードをインデックスとするDataFrame
        """
        if self.store_cd_column in df.columns:
            return df.set_index(self.store_cd_column)
        else:
            raise ValueError(f"Store code column '{self.store_cd_column}' not found")