"""
共通のテストフィクスチャとユーティリティ
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def sample_sales_data():
    """サンプル売上データを生成"""
    np.random.seed(42)
    n_stores = 5
    n_days = 100
    
    data = {}
    for i in range(n_stores):
        base_sales = 100000 + i * 20000
        trend = np.linspace(0, 10000, n_days)
        noise = np.random.normal(0, 5000, n_days)
        sales = base_sales + trend + noise
        sales = np.maximum(sales, 0)
        data[f'store_{i:03d}'] = sales
    
    return data


@pytest.fixture
def sample_sales_dataframe():
    """DataFrame形式のサンプル売上データ"""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    
    data = {}
    for i in range(3):
        base_sales = 100000 + i * 15000
        trend = np.linspace(0, 8000, 100)
        noise = np.random.normal(0, 3000, 100)
        sales = base_sales + trend + noise
        data[f'store_{i:03d}'] = np.maximum(sales, 0)
    
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sample_new_store_sales():
    """新規店舗の売上データ（30日分）"""
    np.random.seed(42)
    base = 110000
    trend = np.linspace(0, 5000, 30)
    noise = np.random.normal(0, 4000, 30)
    sales = base + trend + noise
    return np.maximum(sales, 0)


@pytest.fixture
def sample_store_attributes():
    """店舗属性データ"""
    data = {
        'store_cd': ['store_000', 'store_001', 'store_002', 'store_003', 'store_004'],
        'store_type': ['roadside', 'mall', 'urban', 'roadside', 'mall'],
        'area': [150, 120, 100, 180, 130],
        'location': ['suburban', 'urban', 'downtown', 'suburban', 'urban'],
        'opening_date': [
            datetime(2022, 1, 15),
            datetime(2022, 3, 1),
            datetime(2022, 6, 10),
            datetime(2022, 9, 1),
            datetime(2022, 12, 1)
        ]
    }
    return pd.DataFrame(data).set_index('store_cd')


@pytest.fixture
def sample_opening_dates():
    """開店日の辞書"""
    return {
        'store_000': datetime(2023, 1, 1),
        'store_001': datetime(2023, 1, 15),
        'store_002': datetime(2023, 2, 1),
        'store_003': datetime(2023, 3, 1),
        'store_004': datetime(2023, 4, 1),
    }


@pytest.fixture
def problematic_sales_data():
    """問題のあるデータ（欠損値、異常値、負の値を含む）"""
    return np.array([
        100000, 105000, np.nan, 110000, 115000,  # 欠損値
        120000, 125000, 1000000, 130000, 135000,  # 異常値
        140000, 145000, 0, 0, 0,  # ゼロの連続
        150000, 155000, -5000, 160000, 165000,  # 負の値
    ])


@pytest.fixture
def empty_sales_data():
    """空のデータ"""
    return np.array([])


def generate_seasonal_sales(n_days=365, base=100000, amplitude=20000):
    """季節変動を持つ売上データを生成"""
    days = np.arange(n_days)
    seasonal = base + amplitude * np.sin(days * 2 * np.pi / 365)
    noise = np.random.normal(0, base * 0.05, n_days)
    return np.maximum(seasonal + noise, 0)


def assert_valid_prediction_result(result):
    """予測結果の妥当性を検証"""
    assert hasattr(result, 'prediction')
    assert hasattr(result, 'lower_bound')
    assert hasattr(result, 'upper_bound')
    assert hasattr(result, 'confidence_score')
    assert hasattr(result, 'similar_stores')
    
    assert result.prediction > 0
    assert result.lower_bound <= result.prediction <= result.upper_bound
    assert 0 <= result.confidence_score <= 1
    assert len(result.similar_stores) > 0