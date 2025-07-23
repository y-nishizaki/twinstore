"""
DataTransformerクラスのテスト
"""

import pytest
import pandas as pd
import numpy as np

from twinstore.data.data_transformer import DataTransformer


class TestDataTransformer:
    """DataTransformerクラスのテスト"""
    
    def setup_method(self):
        """各テスト前のセットアップ"""
        self.transformer = DataTransformer()
        
        # テスト用データの準備
        self.sample_df = pd.DataFrame([
            {'store_cd': 'A001', 'date': '2024-01-01', 'sales': 100000},
            {'store_cd': 'A001', 'date': '2024-01-02', 'sales': 105000},
            {'store_cd': 'A001', 'date': '2024-01-03', 'sales': 98000},
            {'store_cd': 'A002', 'date': '2024-01-01', 'sales': 95000},
            {'store_cd': 'A002', 'date': '2024-01-02', 'sales': 97000},
            {'store_cd': 'A002', 'date': '2024-01-03', 'sales': 99000},
        ])
        self.sample_df['date'] = pd.to_datetime(self.sample_df['date'])
    
    def test_to_dict_format(self):
        """辞書形式変換テスト"""
        result = self.transformer.to_dict_format(self.sample_df)
        
        assert isinstance(result, dict)
        assert 'A001' in result
        assert 'A002' in result
        assert isinstance(result['A001'], np.ndarray)
        assert len(result['A001']) == 3
        assert len(result['A002']) == 3
        assert result['A001'][0] == 100000
        assert result['A002'][0] == 95000
    
    def test_to_timeseries_format(self):
        """時系列形式変換テスト"""
        result = self.transformer.to_timeseries_format(self.sample_df)
        
        assert isinstance(result, pd.DataFrame)
        assert 'A001' in result.columns
        assert 'A002' in result.columns
        assert len(result) == 3  # 3日分
        assert pd.api.types.is_datetime64_any_dtype(result.index)
    
    def test_to_batch_format(self):
        """バッチ形式変換テスト"""
        result = self.transformer.to_batch_format(self.sample_df)
        
        assert isinstance(result, dict)
        assert 'A001' in result
        assert 'A002' in result
        assert len(result['A001']) == 3
        assert len(result['A002']) == 3
    
    def test_to_attributes_format(self):
        """属性形式変換テスト"""
        attributes_df = pd.DataFrame([
            {'store_cd': 'A001', 'type': 'urban', 'area': 150},
            {'store_cd': 'A002', 'type': 'suburban', 'area': 200},
        ])
        
        result = self.transformer.to_attributes_format(attributes_df)
        
        assert isinstance(result, pd.DataFrame)
        assert result.index.name == 'store_cd'
        assert 'A001' in result.index
        assert 'A002' in result.index
        assert 'type' in result.columns
        assert 'area' in result.columns
    
    def test_to_attributes_format_missing_column(self):
        """属性形式変換で店舗コード列が欠けている場合のテスト"""
        df = pd.DataFrame([
            {'type': 'urban', 'area': 150},
            {'type': 'suburban', 'area': 200},
        ])
        
        with pytest.raises(ValueError, match="Store code column.*not found"):
            self.transformer.to_attributes_format(df)
    
    def test_custom_column_names(self):
        """カスタム列名でのテスト"""
        transformer = DataTransformer(
            store_cd_column='shop_id',
            date_column='timestamp',
            sales_column='amount'
        )
        
        df = pd.DataFrame([
            {'shop_id': 'S001', 'timestamp': '2024-01-01', 'amount': 100000},
            {'shop_id': 'S001', 'timestamp': '2024-01-02', 'amount': 105000},
            {'shop_id': 'S002', 'timestamp': '2024-01-01', 'amount': 95000},
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        result = transformer.to_dict_format(df)
        
        assert 'S001' in result
        assert 'S002' in result
        assert len(result['S001']) == 2
    
    def test_date_sorting(self):
        """日付順ソートのテスト"""
        # 日付を逆順で作成
        df = pd.DataFrame([
            {'store_cd': 'A001', 'date': '2024-01-03', 'sales': 98000},
            {'store_cd': 'A001', 'date': '2024-01-01', 'sales': 100000},
            {'store_cd': 'A001', 'date': '2024-01-02', 'sales': 105000},
        ])
        df['date'] = pd.to_datetime(df['date'])
        
        result = self.transformer.to_dict_format(df)
        
        # 日付順にソートされているか確認
        assert result['A001'][0] == 100000  # 2024-01-01
        assert result['A001'][1] == 105000  # 2024-01-02
        assert result['A001'][2] == 98000   # 2024-01-03