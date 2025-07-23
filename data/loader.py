"""
データローダーモジュール（SRP適用版）

各専用クラスを組み合わせてデータ読み込み機能を提供する
"""

from typing import Union, Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from .file_reader import FileReader
from .column_validator import ColumnValidator
from .data_transformer import DataTransformer
from .sample_generator import SampleGenerator


class DataLoader:
    """
    データローダークラス（SRP適用版）
    
    単一責任: 各専用クラスを調整し、外部ファイルからTwinStore形式への
    データ読み込みプロセス全体を管理する
    """
    
    def __init__(
        self,
        date_column: Optional[str] = None,
        sales_column: Optional[str] = None,
        store_cd_column: Optional[str] = None,
        date_format: Optional[str] = None,
        encoding: str = 'utf-8',
        auto_detect_columns: bool = True
    ):
        """
        Parameters
        ----------
        date_column : str, optional
            日付列名（指定しない場合はデフォルト値を使用）
        sales_column : str, optional
            売上列名（指定しない場合はデフォルト値を使用）
        store_cd_column : str, optional
            店舗コード列名（指定しない場合はデフォルト値を使用）
        date_format : str, optional
            日付フォーマット（現在未使用）
        encoding : str, default='utf-8'
            ファイルのエンコーディング
        auto_detect_columns : bool, default=True
            列名の自動検出を行うか
        """
        # 各専用クラスのインスタンスを作成
        self.file_reader = FileReader(encoding=encoding)
        self.column_validator = ColumnValidator(
            date_column=date_column,
            sales_column=sales_column,
            store_cd_column=store_cd_column,
            auto_detect=auto_detect_columns
        )
        self.data_transformer = DataTransformer(
            date_column=date_column or self.column_validator.expected_date_column,
            sales_column=sales_column or self.column_validator.expected_sales_column,
            store_cd_column=store_cd_column or self.column_validator.expected_store_cd_column
        )
        self.sample_generator = SampleGenerator(
            date_column=date_column or self.column_validator.expected_date_column,
            sales_column=sales_column or self.column_validator.expected_sales_column,
            store_cd_column=store_cd_column or self.column_validator.expected_store_cd_column,
            encoding=encoding
        )
    
    def load_historical_data(
        self,
        file_path: Union[str, Path],
        output_format: str = "dict",
        **kwargs
    ) -> Union[pd.DataFrame, Dict[str, np.ndarray]]:
        """
        過去の売上データを読み込み
        
        Parameters
        ----------
        file_path : str or Path
            データファイルのパス
        output_format : str, default='dict'
            出力形式 ('dict', 'dataframe')
        **kwargs
            ファイル読み込み時の追加オプション
            
        Returns
        -------
        pd.DataFrame or Dict[str, np.ndarray]
            過去売上データ
        """
        # 1. ファイル読み込み
        df = self.file_reader.read_file(file_path, **kwargs)
        
        # 2. 列名検証・修正
        df = self.column_validator.validate_and_fix_columns(df)
        
        # 3. 形式変換
        if output_format == "dict":
            return self.data_transformer.to_dict_format(df)
        elif output_format == "dataframe":
            return self.data_transformer.to_timeseries_format(df)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def load_store_attributes(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> pd.DataFrame:
        """
        店舗属性データを読み込み
        
        Parameters
        ----------
        file_path : str or Path
            店舗属性ファイルのパス
        **kwargs
            ファイル読み込み時の追加オプション
            
        Returns
        -------
        pd.DataFrame
            店舗属性データ（インデックス: 店舗コード）
        """
        # 1. ファイル読み込み
        df = self.file_reader.read_file(file_path, **kwargs)
        
        # 2. 店舗コード列の検証
        self.column_validator.validate_required_columns(
            df, 
            [self.column_validator.expected_store_cd_column]
        )
        
        # 3. 属性形式に変換
        return self.data_transformer.to_attributes_format(df)
    
    def load_new_store_data(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        新規店舗データを読み込み（バッチ予測用）
        
        Parameters
        ----------
        file_path : str or Path
            新規店舗データファイルのパス
        **kwargs
            ファイル読み込み時の追加オプション
            
        Returns
        -------
        Dict[str, np.ndarray]
            店舗コードをキーとする新規店舗データ
        """
        # 1. ファイル読み込み
        df = self.file_reader.read_file(file_path, **kwargs)
        
        # 2. 必須列の検証
        required_columns = [
            self.column_validator.expected_store_cd_column,
            self.column_validator.expected_sales_column
        ]
        self.column_validator.validate_required_columns(df, required_columns)
        
        # 3. バッチ形式に変換
        return self.data_transformer.to_batch_format(df)
    
    def create_sample_files(self, output_dir: Union[str, Path] = "sample_data") -> None:
        """
        サンプルファイルを生成
        
        Parameters
        ----------
        output_dir : str or Path, default='sample_data'
            出力ディレクトリ
        """
        self.sample_generator.create_sample_files(output_dir)