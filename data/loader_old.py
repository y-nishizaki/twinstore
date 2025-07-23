"""
データローダーモジュール

CSV、Excel、JSONファイルからデータを読み込み、
TwinStoreパッケージで使用可能な形式に変換する。
"""

from typing import Union, Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import json

from ..config.defaults import DATA_FORMAT_DEFAULTS


class DataLoader:
    """
    外部ファイルからデータを読み込むクラス
    
    CSV、Excel、JSON形式のファイルをサポートし、
    TwinStoreパッケージで使用可能な形式に自動変換する。
    """
    
    def __init__(
        self,
        date_column: Optional[str] = None,
        sales_column: Optional[str] = None,
        store_cd_column: Optional[str] = None,
        date_format: Optional[str] = None,
        encoding: str = 'utf-8',
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
            日付フォーマット（指定しない場合はデフォルト値を使用）  
        encoding : str, default='utf-8'
            ファイルのエンコーディング
        """
        self.date_column = date_column or DATA_FORMAT_DEFAULTS['date_column']
        self.sales_column = sales_column or DATA_FORMAT_DEFAULTS['sales_column']
        self.store_cd_column = store_cd_column or DATA_FORMAT_DEFAULTS['store_cd_column']
        self.date_format = date_format or DATA_FORMAT_DEFAULTS['date_format']
        self.encoding = encoding
    
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
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # ファイル形式を判定して読み込み
        if file_path.suffix.lower() == '.csv':
            df = self._load_csv(file_path, **kwargs)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = self._load_excel(file_path, **kwargs)
        elif file_path.suffix.lower() == '.json':
            df = self._load_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # 列名の検証と修正
        df = self._validate_and_fix_columns(df)
        
        # データ形式の変換
        if output_format == "dict":
            return self._convert_to_dict(df)
        elif output_format == "dataframe":
            return self._convert_to_dataframe(df)
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
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # ファイル形式を判定して読み込み
        if file_path.suffix.lower() == '.csv':
            df = self._load_csv(file_path, **kwargs)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = self._load_excel(file_path, **kwargs)
        elif file_path.suffix.lower() == '.json':
            df = self._load_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # 店舗コード列の確認
        if self.store_cd_column not in df.columns:
            raise ValueError(f"Store code column '{self.store_cd_column}' not found in file")
        
        # 店舗コードをインデックスに設定
        df = df.set_index(self.store_cd_column)
        
        return df
    
    def load_new_store_data(
        self,
        file_path: Union[str, Path],
        **kwargs
    ) -> Dict[str, Union[np.ndarray, pd.Series]]:
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
        Dict[str, Union[np.ndarray, pd.Series]]
            店舗コードをキーとする新規店舗データ
        """
        # 一旦DataFrameとして読み込み
        df = pd.DataFrame()
        file_path = Path(file_path)
        
        if file_path.suffix.lower() == '.csv':
            df = self._load_csv(file_path, **kwargs)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            df = self._load_excel(file_path, **kwargs)
        elif file_path.suffix.lower() == '.json':
            df = self._load_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # 店舗コード列と売上列の確認
        if self.store_cd_column not in df.columns:
            raise ValueError(f"Store code column '{self.store_cd_column}' not found")
        if self.sales_column not in df.columns:
            raise ValueError(f"Sales column '{self.sales_column}' not found")
        
        # 店舗ごとにデータを分割
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
    
    def create_sample_files(self, output_dir: Union[str, Path] = "sample_data"):
        """
        サンプルファイルを生成
        
        Parameters
        ----------
        output_dir : str or Path, default='sample_data'
            出力ディレクトリ
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # サンプル過去データの生成
        dates = pd.date_range('2024-01-01', periods=180, freq='D')
        historical_data = []
        
        np.random.seed(42)
        for store_cd in ['A001', 'A002', 'A003']:
            base_sales = 100000 + hash(store_cd) % 50000
            for i, date in enumerate(dates):
                trend = i * 100
                noise = np.random.normal(0, 5000)
                sales = max(0, base_sales + trend + noise)
                
                historical_data.append({
                    self.store_cd_column: store_cd,
                    self.date_column: date,
                    self.sales_column: int(sales)
                })
        
        historical_df = pd.DataFrame(historical_data)
        
        # CSV形式で保存
        csv_path = output_dir / "historical_sales.csv"
        historical_df.to_csv(csv_path, index=False, encoding=self.encoding)
        
        # Excel形式で保存
        excel_path = output_dir / "historical_sales.xlsx"
        historical_df.to_excel(excel_path, index=False, sheet_name="Sales")
        
        # 店舗属性サンプル
        attributes_data = [
            {'store_cd': 'A001', 'type': 'urban', 'area': 150, 'location': 'Tokyo'},
            {'store_cd': 'A002', 'type': 'suburban', 'area': 200, 'location': 'Osaka'},
            {'store_cd': 'A003', 'type': 'roadside', 'area': 300, 'location': 'Nagoya'},
        ]
        
        attributes_df = pd.DataFrame(attributes_data)
        
        # 店舗属性をCSVで保存
        attr_csv_path = output_dir / "store_attributes.csv"
        attributes_df.to_csv(attr_csv_path, index=False, encoding=self.encoding)
        
        # 新規店舗データサンプル
        new_store_data = []
        new_dates = pd.date_range('2024-07-01', periods=30, freq='D')
        
        for store_cd in ['N001', 'N002']:
            base_sales = 95000 + hash(store_cd) % 10000
            for i, date in enumerate(new_dates):
                trend = i * 50
                noise = np.random.normal(0, 3000)
                sales = max(0, base_sales + trend + noise)
                
                new_store_data.append({
                    self.store_cd_column: store_cd,
                    self.date_column: date,
                    self.sales_column: int(sales)
                })
        
        new_store_df = pd.DataFrame(new_store_data)
        new_store_csv_path = output_dir / "new_stores.csv"
        new_store_df.to_csv(new_store_csv_path, index=False, encoding=self.encoding)
        
        print(f"Sample files created in: {output_dir}")
        print(f"- Historical sales: {csv_path}, {excel_path}")
        print(f"- Store attributes: {attr_csv_path}")
        print(f"- New stores: {new_store_csv_path}")
    
    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """CSV ファイルを読み込み"""
        try:
            # デフォルト設定をマージ
            csv_kwargs = {
                'encoding': self.encoding,
                'parse_dates': [self.date_column] if self.date_column in kwargs.get('usecols', []) or 'usecols' not in kwargs else False
            }
            csv_kwargs.update(kwargs)
            
            df = pd.read_csv(file_path, **csv_kwargs)
            return df
        except Exception as e:
            raise ValueError(f"Failed to load CSV file {file_path}: {e}")
    
    def _load_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Excel ファイルを読み込み"""
        try:
            # デフォルト設定
            excel_kwargs = {
                'sheet_name': kwargs.get('sheet_name', 0),  # 最初のシートをデフォルト
            }
            excel_kwargs.update({k: v for k, v in kwargs.items() if k != 'encoding'})  # encodingは除外
            
            df = pd.read_excel(file_path, **excel_kwargs)
            
            # 日付列の変換
            if self.date_column in df.columns:
                df[self.date_column] = pd.to_datetime(df[self.date_column])
            
            return df
        except Exception as e:
            raise ValueError(f"Failed to load Excel file {file_path}: {e}")
    
    def _load_json(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """JSON ファイルを読み込み"""
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                data = json.load(f)
            
            # JSONの構造に応じて処理
            if isinstance(data, list):
                df = pd.DataFrame(data)
            elif isinstance(data, dict):
                # 辞書の場合はキーを店舗コードとして扱う
                records = []
                for store_cd, store_data in data.items():
                    if isinstance(store_data, list):
                        for i, sales in enumerate(store_data):
                            records.append({
                                self.store_cd_column: store_cd,
                                self.sales_column: sales,
                                'day': i
                            })
                    elif isinstance(store_data, dict):
                        store_data[self.store_cd_column] = store_cd
                        records.append(store_data)
                df = pd.DataFrame(records)
            else:
                raise ValueError("Unsupported JSON structure")
            
            # 日付列の変換
            if self.date_column in df.columns:
                df[self.date_column] = pd.to_datetime(df[self.date_column])
            
            return df
        except Exception as e:
            raise ValueError(f"Failed to load JSON file {file_path}: {e}")
    
    def _validate_and_fix_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """列名の検証と自動修正"""
        # 必要な列が存在するかチェック
        missing_columns = []
        
        if self.store_cd_column not in df.columns:
            # 類似の列名を探す
            candidates = [col for col in df.columns if 'store' in col.lower() or 'shop' in col.lower()]
            if candidates:
                warnings.warn(f"Using '{candidates[0]}' as store code column instead of '{self.store_cd_column}'")
                df = df.rename(columns={candidates[0]: self.store_cd_column})
            else:
                missing_columns.append(self.store_cd_column)
        
        if self.sales_column not in df.columns:
            # 類似の列名を探す
            candidates = [col for col in df.columns if 'sales' in col.lower() or 'amount' in col.lower() or 'revenue' in col.lower()]
            if candidates:
                warnings.warn(f"Using '{candidates[0]}' as sales column instead of '{self.sales_column}'")
                df = df.rename(columns={candidates[0]: self.sales_column})
            else:
                missing_columns.append(self.sales_column)
        
        if self.date_column in df.columns:
            # 日付列の変換
            try:
                df[self.date_column] = pd.to_datetime(df[self.date_column], format=self.date_format, errors='coerce')
            except:
                try:
                    df[self.date_column] = pd.to_datetime(df[self.date_column], errors='coerce')
                except:
                    warnings.warn(f"Failed to parse date column '{self.date_column}'")
        
        if missing_columns:
            raise ValueError(f"Required columns not found: {missing_columns}")
        
        return df
    
    def _convert_to_dict(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """DataFrameを辞書形式に変換"""
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
    
    def _convert_to_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """DataFrameを時系列形式に変換"""
        # 店舗ごとにピボット
        if self.date_column in df.columns:
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