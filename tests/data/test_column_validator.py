"""
ColumnValidatorクラスのテスト
"""

import pytest
import pandas as pd
import warnings

from twinstore.data.column_validator import ColumnValidator


class TestColumnValidator:
    """ColumnValidatorクラスのテスト"""
    
    def setup_method(self):
        """各テスト前のセットアップ"""
        self.validator = ColumnValidator()
    
    def test_valid_columns(self):
        """正常な列名のテスト"""
        df = pd.DataFrame({
            'store_cd': ['A001', 'A002'],
            'date': ['2024-01-01', '2024-01-02'],
            'sales': [100000, 105000]
        })
        
        result = self.validator.validate_and_fix_columns(df)
        
        assert 'store_cd' in result.columns
        assert 'sales' in result.columns
        assert len(result) == 2
    
    def test_auto_detect_store_column(self):
        """店舗列の自動検出テスト"""
        df = pd.DataFrame({
            'shop_id': ['A001', 'A002'],
            'date': ['2024-01-01', '2024-01-02'],
            'sales': [100000, 105000]
        })
        
        with warnings.catch_warnings(record=True) as w:
            result = self.validator.validate_and_fix_columns(df)
            assert len(w) == 1
            assert "store code column" in str(w[0].message)
        
        assert 'store_cd' in result.columns
        assert result['store_cd'].tolist() == ['A001', 'A002']
    
    def test_auto_detect_sales_column(self):
        """売上列の自動検出テスト"""
        df = pd.DataFrame({
            'store_cd': ['A001', 'A002'],
            'date': ['2024-01-01', '2024-01-02'],
            'amount': [100000, 105000]
        })
        
        with warnings.catch_warnings(record=True) as w:
            result = self.validator.validate_and_fix_columns(df)
            assert len(w) == 1
            assert "sales column" in str(w[0].message)
        
        assert 'sales' in result.columns
        assert result['sales'].tolist() == [100000, 105000]
    
    def test_missing_required_column(self):
        """必須列が欠けている場合のテスト"""
        df = pd.DataFrame({
            'date': ['2024-01-01', '2024-01-02'],
            'sales': [100000, 105000]
        })
        
        with pytest.raises(ValueError, match="Required columns not found"):
            self.validator.validate_and_fix_columns(df)
    
    def test_date_column_parsing(self):
        """日付列の解析テスト"""
        df = pd.DataFrame({
            'store_cd': ['A001', 'A002'],
            'date': ['2024-01-01', '2024-01-02'],
            'sales': [100000, 105000]
        })
        
        result = self.validator.validate_and_fix_columns(df)
        
        assert pd.api.types.is_datetime64_any_dtype(result['date'])
    
    def test_validate_required_columns(self):
        """必須列検証メソッドのテスト"""
        df = pd.DataFrame({
            'store_cd': ['A001'],
            'sales': [100000]
        })
        
        # 正常ケース
        self.validator.validate_required_columns(df, ['store_cd', 'sales'])
        
        # エラーケース
        with pytest.raises(ValueError, match="Required columns not found"):
            self.validator.validate_required_columns(df, ['store_cd', 'missing_column'])
    
    def test_auto_detect_disabled(self):
        """自動検出無効時のテスト"""
        validator = ColumnValidator(auto_detect=False)
        df = pd.DataFrame({
            'shop_id': ['A001', 'A002'],
            'date': ['2024-01-01', '2024-01-02'],
            'sales': [100000, 105000]
        })
        
        with pytest.raises(ValueError, match="Required columns not found"):
            validator.validate_and_fix_columns(df)