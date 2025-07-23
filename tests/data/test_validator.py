"""
DataValidator のテスト
"""

import pytest
import numpy as np
import pandas as pd
from twinstore.data.validator import DataValidator, ValidationResult


class TestDataValidator:
    """DataValidatorのテストクラス"""
    
    def test_initialization(self):
        """初期化のテスト"""
        # デフォルト設定
        validator = DataValidator()
        assert validator.strict_mode == False
        
        # 厳格モード
        validator = DataValidator(strict_mode=True)
        assert validator.strict_mode == True
    
    def test_validation_result(self):
        """ValidationResultクラスのテスト"""
        result = ValidationResult()
        
        # 初期状態
        assert result.is_valid == True
        assert len(result.errors) == 0
        assert len(result.warnings) == 0
        
        # エラー追加
        result.add_error("Test error")
        assert result.is_valid == False
        assert len(result.errors) == 1
        
        # 警告追加
        result.add_warning("Test warning")
        assert len(result.warnings) == 1
        
        # サマリー設定
        result.set_summary("test_key", "test_value")
        assert result.summary["test_key"] == "test_value"
    
    def test_validate_sales_data_array(self):
        """配列形式の売上データ検証テスト"""
        validator = DataValidator()
        
        # 正常なデータ
        good_data = np.array([100000, 105000, 98000, 102000, 99000])
        result = validator.validate_sales_data(good_data)
        assert result.is_valid == True
        
        # 空のデータ
        empty_data = np.array([])
        result = validator.validate_sales_data(empty_data)
        assert result.is_valid == False
        assert any("empty" in error.lower() for error in result.errors)
    
    def test_validate_sales_data_dict(self):
        """辞書形式の売上データ検証テスト"""
        validator = DataValidator()
        
        # 正常なデータ
        good_data = {
            'store_1': np.array([100000, 105000, 98000]),
            'store_2': [120000, 118000, 125000],
        }
        result = validator.validate_sales_data(good_data)
        assert result.is_valid == True
        assert result.summary['n_stores'] == 2
        
        # 空の辞書
        empty_dict = {}
        result = validator.validate_sales_data(empty_dict)
        assert result.is_valid == False
    
    def test_validate_sales_data_dataframe(self):
        """DataFrame形式の売上データ検証テスト"""
        validator = DataValidator()
        
        # 正常なデータ
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'store_1': np.random.normal(100000, 10000, 30),
            'store_2': np.random.normal(120000, 12000, 30),
        }, index=dates)
        
        result = validator.validate_sales_data(df)
        assert result.is_valid == True
        assert result.summary['shape'] == (30, 2)
        
        # 非数値データを含むDataFrame
        df_bad = pd.DataFrame({
            'store_1': ['a', 'b', 'c'],
            'store_2': [1, 2, 3],
        })
        result = validator.validate_sales_data(df_bad)
        assert result.is_valid == False
        assert any("not numeric" in error for error in result.errors)
    
    def test_validate_store_attributes(self):
        """店舗属性データの検証テスト"""
        validator = DataValidator()
        
        # 正常なデータ
        good_attrs = pd.DataFrame({
            'store_cd': ['store_1', 'store_2', 'store_3'],
            'area': [100, 120, 150],
            'opening_date': ['2023-01-01', '2023-02-01', '2023-03-01']
        }).set_index('store_cd')
        
        result = validator.validate_store_attributes(good_attrs)
        assert result == True  # staticmethodはboolを返す
        
        # store_cdカラムがない
        bad_attrs = pd.DataFrame({
            'area': [100, 120, 150],
            'type': ['A', 'B', 'C']
        })
        
        result = validator.validate_store_attributes(bad_attrs)
        assert result == False  # staticmethodはboolを返す
    
    def test_validate_prediction_input(self):
        """予測入力データの検証テスト"""
        validator = DataValidator()
        
        # 正常なデータ（30日分）
        good_data = np.random.normal(100000, 10000, 30)
        result = validator.validate_prediction_input(good_data)
        assert result.is_valid == True
        assert result.summary['n_days'] == 30
        
        # データ不足（5日分）
        short_data = np.array([100000, 105000, 98000, 102000, 99000])
        result = validator.validate_prediction_input(short_data, min_days=7)
        assert result.is_valid == False
        assert any("不足" in error or "Insufficient" in error.lower() for error in result.errors)
    
    def test_validate_with_nan(self):
        """NaN値を含むデータの検証テスト"""
        validator = DataValidator()
        
        # NaNを含むデータ
        data_with_nan = np.array([100000, np.nan, 98000, 102000, np.nan])
        result = validator.validate_prediction_input(data_with_nan)
        assert result.is_valid == False
        assert any("NaN" in error for error in result.errors)
        assert result.summary.get('n_days') == 5
    
    def test_validate_with_inf(self):
        """無限大を含むデータの検証テスト"""
        validator = DataValidator()
        
        # Infを含むデータ
        data_with_inf = np.array([100000, np.inf, 98000, 102000, -np.inf])
        result = validator.validate_prediction_input(data_with_inf)
        assert result.is_valid == False
        assert any("Inf" in error for error in result.errors)
    
    def test_validate_with_negative(self):
        """負の値を含むデータの検証テスト"""
        validator = DataValidator()
        
        # 負の値を含むデータ
        data_with_negative = np.array([100000, -5000, 98000, 102000, 99000])
        result = validator.validate_prediction_input(data_with_negative, min_days=5)  # min_daysを5に設定
        assert result.is_valid == True  # 警告のみ
        assert any("negative" in warning for warning in result.warnings)
    
    def test_validate_with_zero_runs(self):
        """ゼロの連続を含むデータの検証テスト"""
        validator = DataValidator()
        
        # ゼロの連続を含むデータ
        data_with_zeros = np.array([100000, 105000, 0, 0, 0, 0, 98000, 102000])
        result = validator.validate_prediction_input(data_with_zeros)
        assert result.is_valid == True  # 警告のみ
        assert any("consecutive" in warning and "zero" in warning for warning in result.warnings)
    
    def test_strict_mode(self):
        """厳格モードのテスト"""
        # 通常モード
        validator_normal = DataValidator(strict_mode=False)
        data_with_warning = np.array([100000, -5000, 98000, 102000, 99000])
        result_normal = validator_normal.validate_prediction_input(data_with_warning, min_days=5)
        assert result_normal.is_valid == True
        assert len(result_normal.warnings) > 0
        
        # 厳格モード
        validator_strict = DataValidator(strict_mode=True)
        result_strict = validator_strict.validate_prediction_input(data_with_warning, min_days=5)
        assert result_strict.is_valid == False  # 警告がエラーになる
        assert len(result_strict.errors) > 0
        assert len(result_strict.warnings) == 0
    
    def test_dataframe_missing_values(self):
        """DataFrameの欠損値検証テスト"""
        validator = DataValidator()
        
        # 欠損値を含むDataFrame
        df = pd.DataFrame({
            'store_1': [100000, np.nan, 98000, np.nan, 99000],
            'store_2': [120000, 118000, np.nan, np.nan, np.nan],
        })
        
        result = validator.validate_sales_data(df)
        # 警告は出るが有効
        assert len(result.warnings) > 0
        assert any("missing values" in warning for warning in result.warnings)
    
    def test_dataframe_date_continuity(self):
        """DataFrameの日付連続性検証テスト"""
        validator = DataValidator()
        
        # 連続しない日付
        dates = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-04', '2023-01-07'])
        df = pd.DataFrame({
            'store_1': [100000, 105000, 98000, 102000],
        }, index=dates)
        
        result = validator.validate_sales_data(df)
        assert any("not consecutive" in warning for warning in result.warnings)
    
    def test_static_validate_methods(self):
        """静的メソッドのテスト"""
        # validate_input_data
        good_data = np.array([100000, 105000, 98000])
        assert DataValidator.validate_input_data(good_data) == True
        
        bad_data = np.array([])
        assert DataValidator.validate_input_data(bad_data) == False
        
        # validate_store_attributes
        good_attrs = pd.DataFrame({
            'store_cd': ['s1', 's2'],
            'area': [100, 120]
        }).set_index('store_cd')
        assert DataValidator.validate_store_attributes(good_attrs) == True
    
    def test_get_report(self):
        """レポート生成のテスト"""
        result = ValidationResult()
        result.add_error("Error 1")
        result.add_error("Error 2")
        result.add_warning("Warning 1")
        result.set_summary("test_stat", 123)
        
        report = result.get_report()
        
        assert "Validation Failed" in report
        assert "Errors:" in report
        assert "Error 1" in report
        assert "Error 2" in report
        assert "Warnings:" in report
        assert "Warning 1" in report
        assert "Summary:" in report
        assert "test_stat: 123" in report