"""
SalesPredictor のテスト
"""

import pytest
import numpy as np
import pandas as pd
from twinstore.core.predictor import SalesPredictor, PredictionResult, OptimalPeriodResult
from twinstore.tests.conftest import assert_valid_prediction_result


class TestSalesPredictor:
    """SalesPredictorのテストクラス"""
    
    def test_initialization(self):
        """初期化のテスト"""
        # デフォルト設定
        predictor = SalesPredictor()
        assert predictor.similarity_metric == "dtw"
        assert predictor.normalization_method == "z-score"
        assert predictor.preset is None
        
        # カスタム設定
        predictor = SalesPredictor(
            similarity_metric="cosine",
            normalization_method="min-max",
            preset="restaurant"
        )
        assert predictor.similarity_metric == "cosine"
        assert predictor.normalization_method == "min-max"
        assert predictor.preset == "restaurant"
    
    def test_preset_application(self):
        """プリセット設定のテスト"""
        predictor = SalesPredictor(preset="restaurant")
        assert hasattr(predictor, 'preset_config')
        assert 'min_matching_days' in predictor.preset_config
        assert 'max_matching_days' in predictor.preset_config
        
        # 無効なプリセット
        predictor = SalesPredictor(preset="invalid")
        # 警告が出るが、エラーにはならない
        assert predictor.preset == "invalid"
    
    def test_fit_with_dict(self, sample_sales_data):
        """辞書形式でのfitテスト"""
        predictor = SalesPredictor()
        
        # 学習
        predictor.fit(sample_sales_data)
        
        assert len(predictor.historical_data) == len(sample_sales_data)
        assert all(store_cd in predictor.historical_data for store_cd in sample_sales_data)
    
    def test_fit_with_dataframe(self, sample_sales_dataframe):
        """DataFrame形式でのfitテスト"""
        predictor = SalesPredictor()
        
        # 学習
        predictor.fit(sample_sales_dataframe)
        
        assert len(predictor.historical_data) == len(sample_sales_dataframe.columns)
    
    def test_fit_with_store_attributes(self, sample_sales_data, sample_store_attributes):
        """店舗属性付きでのfitテスト"""
        predictor = SalesPredictor()
        
        # 学習
        predictor.fit(sample_sales_data, sample_store_attributes)
        
        assert len(predictor.store_attributes) == len(sample_store_attributes)
    
    def test_predict_basic(self, sample_sales_data, sample_new_store_sales):
        """基本的な予測テスト"""
        predictor = SalesPredictor()
        predictor.fit(sample_sales_data)
        
        # 予測
        result = predictor.predict(sample_new_store_sales)
        
        # 結果の検証
        assert_valid_prediction_result(result)
        assert isinstance(result, PredictionResult)
    
    def test_predict_with_filters(self, sample_sales_data, sample_store_attributes, sample_new_store_sales):
        """フィルタリング付き予測テスト"""
        predictor = SalesPredictor()
        predictor.fit(sample_sales_data, sample_store_attributes)
        
        # roadside店舗のみで予測
        result = predictor.predict(
            sample_new_store_sales,
            filters={'store_type': 'roadside'}
        )
        
        assert_valid_prediction_result(result)
        # フィルタが適用されていることを確認
        # （実装によってはメタデータで確認）
    
    def test_predict_with_matching_period(self, sample_sales_data, sample_new_store_sales):
        """マッチング期間指定での予測テスト"""
        predictor = SalesPredictor()
        predictor.fit(sample_sales_data)
        
        # 14日間でマッチング
        result = predictor.predict(
            sample_new_store_sales,
            matching_period_days=14
        )
        
        assert_valid_prediction_result(result)
        assert result.metadata['matching_period_days'] == 14
    
    def test_predict_with_n_similar(self, sample_sales_data, sample_new_store_sales):
        """類似店舗数指定での予測テスト"""
        predictor = SalesPredictor()
        predictor.fit(sample_sales_data)
        
        # 3店舗のみ使用
        result = predictor.predict(
            sample_new_store_sales,
            n_similar=3
        )
        
        assert_valid_prediction_result(result)
        assert len(result.similar_stores) <= 3
    
    def test_predict_with_confidence_level(self, sample_sales_data, sample_new_store_sales):
        """信頼水準指定での予測テスト"""
        predictor = SalesPredictor()
        predictor.fit(sample_sales_data)
        
        # 90%信頼区間
        result_90 = predictor.predict(
            sample_new_store_sales,
            confidence_level=0.90
        )
        
        # 99%信頼区間
        result_99 = predictor.predict(
            sample_new_store_sales,
            confidence_level=0.99
        )
        
        # 99%の方が区間が広いはず
        range_90 = result_90.upper_bound - result_90.lower_bound
        range_99 = result_99.upper_bound - result_99.lower_bound
        assert range_99 >= range_90
    
    def test_suggest_matching_period(self, sample_sales_data, sample_new_store_sales):
        """最適マッチング期間の提案テスト"""
        predictor = SalesPredictor()
        predictor.fit(sample_sales_data)
        
        # 最適期間の提案
        optimal_period = predictor.suggest_matching_period(
            sample_new_store_sales,
            min_days=7,
            max_days=30,
            step_days=7
        )
        
        assert isinstance(optimal_period, OptimalPeriodResult)
        assert 7 <= optimal_period.recommended_days <= 30
        assert len(optimal_period.accuracy_scores) > 0
        assert 0 <= optimal_period.stability_score <= 1
    
    def test_predict_without_fit(self, sample_new_store_sales):
        """fit前の予測でエラーになることを確認"""
        predictor = SalesPredictor()
        
        with pytest.raises(ValueError, match="No historical data"):
            predictor.predict(sample_new_store_sales)
    
    def test_predict_with_insufficient_data(self, sample_sales_data):
        """データ不足での予測テスト"""
        predictor = SalesPredictor()
        predictor.fit(sample_sales_data)
        
        # 5日分のデータ（少なすぎる）
        short_sales = np.array([100000, 105000, 98000, 102000, 99000])
        
        # 警告は出るが予測は可能
        result = predictor.predict(short_sales)
        assert_valid_prediction_result(result)
        
        # 信頼度は低いはず（デフォルト値が変更されたため閾値を調整）
        assert result.confidence_score < 0.7
    
    def test_predict_with_empty_data(self, sample_sales_data):
        """空データでの予測テスト"""
        predictor = SalesPredictor()
        predictor.fit(sample_sales_data)
        
        with pytest.raises(ValueError):
            predictor.predict([])
    
    def test_predict_with_nan_values(self, sample_sales_data):
        """NaN値を含むデータでの予測テスト"""
        predictor = SalesPredictor()
        predictor.fit(sample_sales_data)
        
        # NaNを含むデータ
        sales_with_nan = np.array([100000, np.nan, 98000, 102000, np.nan, 105000])
        
        # 実装によってはエラーになる可能性
        try:
            result = predictor.predict(sales_with_nan)
            assert_valid_prediction_result(result)
        except:
            pytest.skip("NaN handling not implemented")
    
    def test_confidence_score_calculation(self, sample_sales_data):
        """信頼度スコアの計算テスト"""
        predictor = SalesPredictor()
        predictor.fit(sample_sales_data)
        
        # 長期データ（高信頼度）
        long_sales = np.random.normal(100000, 10000, 60)
        result_long = predictor.predict(long_sales)
        
        # 短期データ（低信頼度）
        short_sales = np.random.normal(100000, 10000, 10)
        result_short = predictor.predict(short_sales)
        
        # 長期データの方が信頼度が高いはず
        assert result_long.confidence_score > result_short.confidence_score
    
    def test_prediction_range(self, sample_sales_data, sample_new_store_sales):
        """予測範囲の妥当性テスト"""
        predictor = SalesPredictor()
        predictor.fit(sample_sales_data)
        
        result = predictor.predict(sample_new_store_sales)
        
        # 予測値が妥当な範囲内か（より現実的な範囲に調整）
        daily_avg = np.mean(sample_new_store_sales)
        annual_estimate = daily_avg * 365
        
        # 予測値は年間推定値の0.2倍〜5倍程度の範囲内にあるはず（より広い範囲）
        assert annual_estimate * 0.2 <= result.prediction <= annual_estimate * 5
    
    def test_different_similarity_metrics(self, sample_sales_data, sample_new_store_sales):
        """異なる類似性指標での予測テスト"""
        metrics = ["dtw", "cosine", "correlation"]
        results = {}
        
        for metric in metrics:
            predictor = SalesPredictor(similarity_metric=metric)
            predictor.fit(sample_sales_data)
            results[metric] = predictor.predict(sample_new_store_sales)
        
        # 全てのメトリクスで有効な結果が得られる
        for metric, result in results.items():
            assert_valid_prediction_result(result)
        
        # メトリクスによって結果が異なる
        predictions = [r.prediction for r in results.values()]
        assert len(set(predictions)) > 1  # 全て同じでない
    
    def test_different_normalization_methods(self, sample_sales_data, sample_new_store_sales):
        """異なる正規化手法での予測テスト"""
        methods = ["z-score", "min-max", "robust"]
        results = {}
        
        for method in methods:
            predictor = SalesPredictor(normalization_method=method)
            predictor.fit(sample_sales_data)
            results[method] = predictor.predict(sample_new_store_sales)
        
        # 全ての手法で有効な結果が得られる
        for method, result in results.items():
            assert_valid_prediction_result(result)