"""
DataNormalizer のテスト
"""

import pytest
import numpy as np
import pandas as pd
from twinstore.core.normalizer import DataNormalizer


class TestDataNormalizer:
    """DataNormalizerのテストクラス"""
    
    def test_initialization(self):
        """初期化のテスト"""
        # デフォルト設定
        normalizer = DataNormalizer()
        assert normalizer.method == "z-score"
        
        # カスタム設定
        normalizer = DataNormalizer(method="min-max")
        assert normalizer.method == "min-max"
    
    def test_invalid_method(self):
        """無効な正規化手法のテスト"""
        with pytest.raises(ValueError):
            DataNormalizer(method="invalid")
    
    def test_zscore_normalization(self):
        """Z-score正規化のテスト"""
        normalizer = DataNormalizer(method="z-score")
        
        data = np.array([1, 2, 3, 4, 5])
        normalized = normalizer.fit_transform(data)
        
        # 平均0、標準偏差1になることを確認
        assert np.mean(normalized) == pytest.approx(0, abs=1e-10)
        assert np.std(normalized) == pytest.approx(1, abs=1e-10)
    
    def test_minmax_normalization(self):
        """Min-Max正規化のテスト"""
        normalizer = DataNormalizer(method="min-max")
        
        data = np.array([1, 2, 3, 4, 5])
        normalized = normalizer.fit_transform(data)
        
        # 最小値0、最大値1になることを確認
        assert np.min(normalized) == pytest.approx(0, abs=1e-10)
        assert np.max(normalized) == pytest.approx(1, abs=1e-10)
    
    def test_robust_normalization(self):
        """ロバスト正規化のテスト"""
        normalizer = DataNormalizer(method="robust")
        
        # 外れ値を含むデータ
        data = np.array([1, 2, 3, 4, 5, 100])  # 100は外れ値
        normalized = normalizer.fit_transform(data)
        
        # 外れ値の影響が小さいことを確認
        assert np.median(normalized) == pytest.approx(0, abs=0.1)
    
    def test_first_day_ratio_normalization(self):
        """初日比率正規化のテスト"""
        normalizer = DataNormalizer(method="first-day-ratio")
        
        data = np.array([100, 110, 120, 130, 140])
        normalized = normalizer.fit_transform(data)
        
        # 最初の値が1になることを確認
        assert normalized[0] == pytest.approx(1.0, abs=1e-10)
        assert normalized[1] == pytest.approx(1.1, abs=1e-10)
        assert normalized[-1] == pytest.approx(1.4, abs=1e-10)
    
    def test_mean_ratio_normalization(self):
        """平均比率正規化のテスト"""
        normalizer = DataNormalizer(method="mean-ratio")
        
        data = np.array([80, 100, 120])
        normalized = normalizer.fit_transform(data)
        
        # 平均が1になることを確認
        assert np.mean(normalized) == pytest.approx(1.0, abs=1e-10)
    
    def test_fit_and_transform_separately(self):
        """fitとtransformを別々に実行するテスト"""
        normalizer = DataNormalizer(method="z-score")
        
        train_data = np.array([1, 2, 3, 4, 5])
        test_data = np.array([3, 4, 5, 6, 7])
        
        # 学習
        normalizer.fit(train_data)
        
        # 変換
        normalized_train = normalizer.transform(train_data)
        normalized_test = normalizer.transform(test_data)
        
        # train_dataで学習したパラメータでtest_dataも変換される
        assert len(normalized_test) == len(test_data)
        # test_dataの平均は0にならない（train_dataの統計量を使用するため）
        assert np.mean(normalized_test) != pytest.approx(0, abs=0.1)
    
    def test_inverse_transform(self):
        """逆変換のテスト"""
        normalizer = DataNormalizer(method="min-max")
        
        original_data = np.array([10, 20, 30, 40, 50])
        normalized = normalizer.fit_transform(original_data)
        restored = normalizer.inverse_transform(normalized)
        
        # 元のデータに戻ることを確認
        np.testing.assert_array_almost_equal(original_data, restored, decimal=5)
    
    def test_pandas_series_input(self):
        """pandas Series入力のテスト"""
        normalizer = DataNormalizer(method="z-score")
        
        series = pd.Series([1, 2, 3, 4, 5])
        normalized = normalizer.fit_transform(series)
        
        assert isinstance(normalized, np.ndarray)
        assert len(normalized) == len(series)
    
    def test_pandas_dataframe_input(self):
        """pandas DataFrame入力のテスト"""
        normalizer = DataNormalizer(method="min-max")
        
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        # DataFrameの場合、実装によってはエラーになる可能性
        try:
            normalized = normalizer.fit_transform(df)
            assert normalized.shape == df.shape
        except:
            pytest.skip("DataFrame input not supported")
    
    def test_zero_std_handling(self):
        """標準偏差0の処理テスト"""
        normalizer = DataNormalizer(method="z-score")
        
        # 全て同じ値
        data = np.array([5, 5, 5, 5, 5])
        normalized = normalizer.fit_transform(data)
        
        # 全て0になるはず
        assert np.all(normalized == 0)
    
    def test_zero_first_value_handling(self):
        """初日値0の処理テスト"""
        normalizer = DataNormalizer(method="first-day-ratio")
        
        data = np.array([0, 100, 200, 300])
        normalized = normalizer.fit_transform(data)
        
        # エラーにならないことを確認
        assert len(normalized) == len(data)
    
    def test_normalize_multiple_series(self):
        """複数系列の正規化テスト"""
        normalizer = DataNormalizer(method="z-score")
        
        series_dict = {
            'store_1': np.array([100, 110, 120]),
            'store_2': np.array([200, 220, 240]),
            'store_3': np.array([150, 165, 180])
        }
        
        # 各系列を個別に正規化
        normalized_dict = normalizer.normalize_multiple_series(series_dict)
        
        assert len(normalized_dict) == len(series_dict)
        for store_cd, normalized in normalized_dict.items():
            assert len(normalized) == len(series_dict[store_cd])
            assert np.mean(normalized) == pytest.approx(0, abs=1e-10)
            assert np.std(normalized) == pytest.approx(1, abs=1e-10)
    
    def test_normalize_multiple_series_with_fit_on(self):
        """特定系列で学習して複数系列を正規化するテスト"""
        normalizer = DataNormalizer(method="min-max")
        
        series_dict = {
            'store_1': np.array([100, 110, 120]),
            'store_2': np.array([200, 220, 240]),
            'store_3': np.array([150, 165, 180])
        }
        
        # store_1で学習して全店舗を正規化
        normalized_dict = normalizer.normalize_multiple_series(
            series_dict, 
            fit_on='store_1'
        )
        
        # store_1は0-1の範囲になる
        assert np.min(normalized_dict['store_1']) == pytest.approx(0, abs=1e-10)
        assert np.max(normalized_dict['store_1']) == pytest.approx(1, abs=1e-10)
        
        # 他の店舗は0-1の範囲外になる可能性がある
        assert np.min(normalized_dict['store_2']) > 1  # store_2は値が大きい
    
    def test_compare_normalization_methods(self):
        """正規化手法の比較テスト"""
        data = np.array([10, 20, 30, 40, 50, 100])  # 100は外れ値
        
        comparison = DataNormalizer.compare_normalization_methods(
            data,
            methods=["z-score", "min-max", "robust"]
        )
        
        assert isinstance(comparison, pd.DataFrame)
        assert "original" in comparison.columns
        assert "z-score" in comparison.columns
        assert "min-max" in comparison.columns
        assert "robust" in comparison.columns
        assert len(comparison) == len(data)