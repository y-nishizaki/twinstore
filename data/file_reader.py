"""
ファイル読み込み専用モジュール

ファイルからDataFrameを読み込む責任のみを持つ
"""

from typing import Union, Dict, Any
import pandas as pd
from pathlib import Path
import json


class FileReader:
    """
    ファイル読み込み専用クラス
    
    単一責任: 各種ファイル形式からpd.DataFrameを読み込む
    """
    
    def __init__(self, encoding: str = 'utf-8'):
        """
        Parameters
        ----------
        encoding : str, default='utf-8'
            ファイルのエンコーディング
        """
        self.encoding = encoding
    
    def read_file(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        ファイルを読み込んでDataFrameを返す
        
        Parameters
        ----------
        file_path : str or Path
            読み込むファイルのパス
        **kwargs
            ファイル読み込み時のオプション
            
        Returns
        -------
        pd.DataFrame
            読み込まれたデータ
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix == '.csv':
            return self._read_csv(file_path, **kwargs)
        elif suffix in ['.xlsx', '.xls']:
            return self._read_excel(file_path, **kwargs)
        elif suffix == '.json':
            return self._read_json(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {suffix}")
    
    def _read_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """CSV ファイルを読み込み"""
        try:
            csv_kwargs = {
                'encoding': self.encoding,
            }
            csv_kwargs.update(kwargs)
            return pd.read_csv(file_path, **csv_kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load CSV file {file_path}: {e}")
    
    def _read_excel(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Excel ファイルを読み込み"""
        try:
            excel_kwargs = {
                'sheet_name': kwargs.get('sheet_name', 0),
            }
            excel_kwargs.update({k: v for k, v in kwargs.items() if k != 'encoding'})
            return pd.read_excel(file_path, **excel_kwargs)
        except Exception as e:
            raise ValueError(f"Failed to load Excel file {file_path}: {e}")
    
    def _read_json(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """JSON ファイルを読み込み"""
        try:
            with open(file_path, 'r', encoding=self.encoding) as f:
                data = json.load(f)
            
            if isinstance(data, list):
                return pd.DataFrame(data)
            elif isinstance(data, dict):
                # 辞書の場合は構造を推測してDataFrameに変換
                return self._dict_to_dataframe(data)
            else:
                raise ValueError("Unsupported JSON structure")
        except Exception as e:
            raise ValueError(f"Failed to load JSON file {file_path}: {e}")
    
    def _dict_to_dataframe(self, data: dict) -> pd.DataFrame:
        """辞書をDataFrameに変換"""
        records = []
        for key, value in data.items():
            if isinstance(value, list):
                # キーを店舗コードとして扱う
                for i, item in enumerate(value):
                    record = {'key': key, 'index': i, 'value': item}
                    records.append(record)
            elif isinstance(value, dict):
                value['key'] = key
                records.append(value)
        return pd.DataFrame(records)