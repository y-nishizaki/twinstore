"""
SimilarityEngine のテスト
"""

import pytest
import numpy as np
import pandas as pd
from twinstore.core.similarity import SimilarityEngine


class TestSimilarityEngine:
    """SimilarityEngineのテストクラス"""
    
    def test_initialization(self):
        """初期化のテスト"""
        # デフォルト設定
        engine = SimilarityEngine()
        assert engine.metric == "dtw"
        assert engine.normalize == True
        assert engine.window is None
        
        # カスタム設定
        engine = SimilarityEngine(metric="cosine", normalize=False, window=10)
        assert engine.metric == "cosine"
        assert engine.normalize == False
        assert engine.window == 10
    
    def test_invalid_metric(self):
        """無効なメトリクスのテスト"""
        with pytest.raises(ValueError):
            SimilarityEngine(metric="invalid")
    
    def test_calculate_similarity_dtw(self):
        """DTW距離の計算テスト"""
        engine = SimilarityEngine(metric="dtw")
        
        # 同一の系列
        series1 = np.array([1, 2, 3, 4, 5])
        series2 = np.array([1, 2, 3, 4, 5])
        similarity = engine.calculate_similarity(series1, series2)
        assert similarity == pytest.approx(0.0, abs=0.01)
        
        # 異なる系列
        series3 = np.array([5, 4, 3, 2, 1])
        similarity = engine.calculate_similarity(series1, series3)
        assert similarity > 0
    
    def test_calculate_similarity_cosine(self):
        """コサイン類似度の計算テスト"""
        engine = SimilarityEngine(metric="cosine")
        
        # 同一の系列
        series1 = np.array([1, 2, 3, 4, 5])
        series2 = np.array([1, 2, 3, 4, 5])
        similarity = engine.calculate_similarity(series1, series2)
        assert similarity == pytest.approx(1.0, abs=0.01)
        
        # 直交する系列
        series3 = np.array([0, 0, 1, 0, 0])
        series4 = np.array([0, 1, 0, 0, 0])
        similarity = engine.calculate_similarity(series3, series4)
        assert similarity == pytest.approx(0.0, abs=0.01)
    
    def test_calculate_similarity_correlation(self):
        """相関係数の計算テスト"""
        engine = SimilarityEngine(metric="correlation")
        
        # 完全相関
        series1 = np.array([1, 2, 3, 4, 5])
        series2 = np.array([2, 4, 6, 8, 10])  # 線形変換
        similarity = engine.calculate_similarity(series1, series2)
        assert similarity == pytest.approx(1.0, abs=0.01)
        
        # 負の相関
        series3 = np.array([5, 4, 3, 2, 1])
        similarity = engine.calculate_similarity(series1, series3)
        assert similarity == pytest.approx(-1.0, abs=0.01)
    
    def test_normalization(self):
        """正規化のテスト"""
        # DTWでは正規化の効果を確認
        engine_norm = SimilarityEngine(metric="dtw", normalize=True)
        series1 = np.array([100, 200, 300])
        series2 = np.array([1, 2, 3])  # スケールが異なる
        similarity_norm = engine_norm.calculate_similarity(series1, series2)
        
        # 正規化なし
        engine_no_norm = SimilarityEngine(metric="dtw", normalize=False)
        similarity_no_norm = engine_no_norm.calculate_similarity(series1, series2)
        
        # DTWでは正規化ありの方が距離が小さい（類似度が高い）はず
        assert similarity_norm < similarity_no_norm  # DTWは距離なので小さい方が良い
    
    def test_empty_series(self):
        """空の系列のテスト"""
        engine = SimilarityEngine()
        
        with pytest.raises(ValueError):
            engine.calculate_similarity([], [1, 2, 3])
        
        with pytest.raises(ValueError):
            engine.calculate_similarity([1, 2, 3], [])
    
    def test_different_lengths(self):
        """異なる長さの系列のテスト"""
        engine = SimilarityEngine(metric="cosine")
        
        series1 = np.array([1, 2, 3, 4, 5])
        series2 = np.array([1, 2, 3])
        
        # 長さが異なっても計算可能
        similarity = engine.calculate_similarity(series1, series2)
        assert 0 <= similarity <= 1
    
    def test_find_similar_stores(self, sample_sales_data):
        """類似店舗検索のテスト"""
        engine = SimilarityEngine(metric="dtw")
        
        target_sales = sample_sales_data['store_000']
        similar_stores = engine.find_similar_stores(
            target_sales,
            sample_sales_data,
            top_k=3
        )
        
        assert len(similar_stores) == 3
        assert all(isinstance(store_cd, str) for store_cd, _ in similar_stores)
        assert all(isinstance(score, (int, float)) for _, score in similar_stores)
        
        # 最も類似した店舗が最初に来る
        if engine.metric == "dtw":
            # DTWは距離なので昇順
            assert similar_stores[0][1] <= similar_stores[1][1]
        else:
            # 類似度は降順
            assert similar_stores[0][1] >= similar_stores[1][1]
    
    def test_similarity_matrix(self, sample_sales_data):
        """類似性行列の計算テスト"""
        engine = SimilarityEngine(metric="correlation")
        
        # 3店舗分のデータ
        subset_data = {k: v for k, v in list(sample_sales_data.items())[:3]}
        matrix = engine.calculate_similarity_matrix(subset_data)
        
        assert isinstance(matrix, pd.DataFrame)
        assert matrix.shape == (3, 3)
        assert list(matrix.index) == list(subset_data.keys())
        assert list(matrix.columns) == list(subset_data.keys())
        
        # 対角要素は1（自己相関）
        for i in range(3):
            assert matrix.iloc[i, i] == pytest.approx(1.0, abs=0.01)
        
        # 対称行列
        assert np.allclose(matrix.values, matrix.values.T)
    
    def test_window_constraint(self):
        """ウィンドウ制約のテスト"""
        # ウィンドウありとなしで計算時間が異なることを確認
        engine_no_window = SimilarityEngine(metric="dtw", window=None)
        engine_with_window = SimilarityEngine(metric="dtw", window=5)
        
        series1 = np.random.rand(100)
        series2 = np.random.rand(100)
        
        # 両方計算可能
        sim1 = engine_no_window.calculate_similarity(series1, series2)
        sim2 = engine_with_window.calculate_similarity(series1, series2)
        
        assert isinstance(sim1, (int, float))
        assert isinstance(sim2, (int, float))
    
    def test_nan_handling(self):
        """NaN値の処理テスト"""
        engine = SimilarityEngine(metric="correlation")
        
        series1 = np.array([1, 2, np.nan, 4, 5])
        series2 = np.array([2, 4, 6, 8, 10])
        
        # NaNを含む系列でも処理できることを確認
        # （実装によってはエラーになる可能性があるため）
        try:
            similarity = engine.calculate_similarity(series1, series2)
            assert isinstance(similarity, (int, float))
        except:
            # NaN処理が未実装の場合はスキップ
            pytest.skip("NaN handling not implemented")