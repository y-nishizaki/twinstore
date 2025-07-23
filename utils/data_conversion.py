"""
データ変換ユーティリティ

共通のデータ変換処理を提供する
"""

from typing import Union, Dict, List, Any, Optional
import numpy as np
import pandas as pd


def to_numpy_array(
    data: Union[np.ndarray, pd.Series, pd.DataFrame, List, tuple]
) -> np.ndarray:
    """
    各種データ型をnumpy配列に変換
    
    Parameters
    ----------
    data : various
        変換するデータ
        
    Returns
    -------
    np.ndarray
        numpy配列
    """
    if isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (pd.Series, pd.DataFrame)):
        return data.values
    elif isinstance(data, (list, tuple)):
        return np.asarray(data)
    else:
        raise TypeError(f"Unsupported data type: {type(data)}")


def dataframe_to_dict(
    df: pd.DataFrame,
    store_cd_column: str,
    sales_column: str,
    date_column: Optional[str] = None
) -> Dict[str, np.ndarray]:
    """
    DataFrameを辞書形式に変換（効率的な実装）
    
    Parameters
    ----------
    df : pd.DataFrame
        変換するDataFrame
    store_cd_column : str
        店舗コード列名
    sales_column : str
        売上列名
    date_column : str, optional
        日付列名（指定時はソートする）
        
    Returns
    -------
    Dict[str, np.ndarray]
        店舗コードと売上配列の辞書
    """
    # 日付でソート（日付列がある場合）
    if date_column and date_column in df.columns:
        df = df.sort_values([store_cd_column, date_column])
    
    # groupbyを使用した効率的な変換
    result = {}
    for store_cd, group in df.groupby(store_cd_column):
        result[str(store_cd)] = group[sales_column].values
    
    return result


def dict_to_dataframe(
    data_dict: Dict[str, Union[np.ndarray, List]],
    store_cd_column: str = 'store_cd',
    sales_column: str = 'sales',
    add_date_index: bool = True
) -> pd.DataFrame:
    """
    辞書形式のデータをDataFrameに変換
    
    Parameters
    ----------
    data_dict : dict
        店舗コードと売上データの辞書
    store_cd_column : str, default='store_cd'
        店舗コード列名
    sales_column : str, default='sales'
        売上列名
    add_date_index : bool, default=True
        日付インデックスを追加するか
        
    Returns
    -------
    pd.DataFrame
        変換されたDataFrame
    """
    records = []
    
    for store_cd, sales_data in data_dict.items():
        sales_array = to_numpy_array(sales_data)
        
        for i, value in enumerate(sales_array):
            record = {
                store_cd_column: store_cd,
                sales_column: value,
                'day_index': i
            }
            records.append(record)
    
    df = pd.DataFrame(records)
    
    if add_date_index:
        # 日付インデックスを追加（仮の日付）
        df['date'] = pd.date_range(
            start='2023-01-01',
            periods=df['day_index'].max() + 1
        )[df['day_index']]
        df = df.drop('day_index', axis=1)
    
    return df


def ensure_2d_array(
    data: Union[np.ndarray, pd.DataFrame, pd.Series]
) -> np.ndarray:
    """
    データを2次元配列として確保
    
    Parameters
    ----------
    data : array-like
        変換するデータ
        
    Returns
    -------
    np.ndarray
        2次元配列
    """
    array = to_numpy_array(data)
    
    if array.ndim == 1:
        return array.reshape(-1, 1)
    elif array.ndim == 2:
        return array
    else:
        raise ValueError(f"Cannot convert {array.ndim}D array to 2D")


def flatten_dict_values(
    data_dict: Dict[str, Any]
) -> np.ndarray:
    """
    辞書の値を1次元配列に展開
    
    Parameters
    ----------
    data_dict : dict
        展開する辞書
        
    Returns
    -------
    np.ndarray
        展開された1次元配列
    """
    all_values = []
    
    for value in data_dict.values():
        array = to_numpy_array(value)
        all_values.extend(array.flatten())
    
    return np.array(all_values)