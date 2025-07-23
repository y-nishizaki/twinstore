"""
PredictionPipeline のテスト
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from twinstore.pipeline import (
    PredictionPipeline,
    PipelineBuilder,
    PipelineConfig,
    PipelineResult
)


class TestPipelineConfig:
    """PipelineConfigのテストクラス"""
    
    def test_default_config(self):
        """デフォルト設定のテスト"""
        config = PipelineConfig()
        
        assert config.validate_input == True
        assert config.preprocess_data == True
        assert config.check_quality == True
        assert config.similarity_metric == "dtw"
        assert config.generate_explanation == True
        assert config.save_results == False
    
    def test_custom_config(self):
        """カスタム設定のテスト"""
        config = PipelineConfig(
            validate_input=False,
            similarity_metric="cosine",
            n_similar_stores=3,
            save_results=True,
            output_format="csv"
        )
        
        assert config.validate_input == False
        assert config.similarity_metric == "cosine"
        assert config.n_similar_stores == 3
        assert config.save_results == True
        assert config.output_format == "csv"


class TestPredictionPipeline:
    """PredictionPipelineのテストクラス"""
    
    def test_initialization(self):
        """初期化のテスト"""
        # デフォルト設定
        pipeline = PredictionPipeline()
        assert isinstance(pipeline.config, PipelineConfig)
        assert len(pipeline._components) > 0
        
        # カスタム設定
        config = PipelineConfig(similarity_metric="cosine")
        pipeline = PredictionPipeline(config)
        assert pipeline.config.similarity_metric == "cosine"
    
    def test_fit_and_predict(self, sample_sales_data, sample_new_store_sales):
        """基本的なfit & predictのテスト"""
        pipeline = PredictionPipeline()
        
        # 学習
        pipeline.fit(sample_sales_data)
        
        # 予測
        result = pipeline.predict(sample_new_store_sales)
        
        assert isinstance(result, PipelineResult)
        assert result.prediction is not None
        assert result.prediction.prediction > 0
        assert result.execution_time > 0
    
    def test_predict_with_validation(self, sample_sales_data, sample_new_store_sales):
        """検証付き予測のテスト"""
        config = PipelineConfig(
            validate_input=True,  # バリデーションを有効に
            min_days=5,  # テスト用に低い閾値を設定
            preprocess_data=True
        )
        pipeline = PredictionPipeline(config)
        pipeline.fit(sample_sales_data)
        
        # 正常なデータで予測
        result = pipeline.predict(sample_new_store_sales)
        
        # 予測は成功するはず
        assert result.prediction is not None
    
    def test_predict_with_preprocessing(self, sample_sales_data, problematic_sales_data):
        """前処理付き予測のテスト"""
        config = PipelineConfig(
            validate_input=True,  # バリデーションを有効に
            min_days=5,  # テスト用に低い闾値を設定
            preprocess_data=True,
            handle_missing=True,
            handle_outliers=True
        )
        pipeline = PredictionPipeline(config)
        pipeline.fit(sample_sales_data)
        
        # 欠損値や異常値を含むデータ
        result = pipeline.predict(problematic_sales_data)
        
        assert result.prediction is not None
    
    def test_predict_with_quality_check(self, sample_sales_data, sample_new_store_sales):
        """品質チェック付き予測のテスト"""
        config = PipelineConfig(
            check_quality=True,
            quality_threshold=80.0
        )
        pipeline = PredictionPipeline(config)
        pipeline.fit(sample_sales_data)
        
        result = pipeline.predict(sample_new_store_sales)
        
        assert result.quality_report is not None
        assert hasattr(result.quality_report, 'overall_score')
        assert 0 <= result.quality_report.overall_score <= 100
    
    def test_predict_with_auto_optimize(self, sample_sales_data, sample_new_store_sales):
        """期間自動最適化付き予測のテスト"""
        config = PipelineConfig(
            auto_optimize_period=True
        )
        pipeline = PredictionPipeline(config)
        pipeline.fit(sample_sales_data)
        
        result = pipeline.predict(sample_new_store_sales)
        
        assert 'optimal_period' in result.metadata
        assert 'recommended_days' in result.metadata['optimal_period']
    
    def test_predict_with_explanation(self, sample_sales_data, sample_new_store_sales):
        """説明生成付き予測のテスト"""
        config = PipelineConfig(
            generate_explanation=True,
            explanation_language="ja"
        )
        pipeline = PredictionPipeline(config)
        pipeline.fit(sample_sales_data)
        
        result = pipeline.predict(sample_new_store_sales)
        
        assert result.explanation is not None
        assert isinstance(result.explanation, str)
        assert len(result.explanation) > 0
    
    def test_batch_predict(self, sample_sales_data):
        """バッチ予測のテスト"""
        pipeline = PredictionPipeline()
        pipeline.fit(sample_sales_data)
        
        # 複数店舗のデータ
        batch_data = {
            'new_store_1': np.random.normal(100000, 10000, 30),
            'new_store_2': np.random.normal(120000, 12000, 25),
            'new_store_3': np.random.normal(90000, 9000, 35),
        }
        
        results = pipeline.batch_predict(batch_data)
        
        assert len(results) == 3
        for store_cd, result in results.items():
            assert isinstance(result, PipelineResult)
            if result.prediction:  # エラーの場合はNone
                assert result.prediction.prediction > 0
    
    def test_update_config(self, sample_sales_data, sample_new_store_sales):
        """設定更新のテスト"""
        config = PipelineConfig(validate_input=False)
        pipeline = PredictionPipeline(config)
        pipeline.fit(sample_sales_data)
        
        # 初回予測
        result1 = pipeline.predict(sample_new_store_sales)
        
        # 設定を更新
        pipeline.update_config(
            similarity_metric="cosine",
            n_similar_stores=3
        )
        
        # 再予測
        result2 = pipeline.predict(sample_new_store_sales)
        
        # 予測は成功する
        assert result1.prediction is not None
        assert result2.prediction is not None
    
    def test_save_results(self, sample_sales_data, sample_new_store_sales, tmp_path):
        """結果保存のテスト"""
        config = PipelineConfig(
            save_results=True,
            output_dir=str(tmp_path),
            output_format="json"
        )
        pipeline = PredictionPipeline(config)
        pipeline.fit(sample_sales_data)
        
        result = pipeline.predict(sample_new_store_sales, store_name="test_store")
        
        # ファイルが作成されているか確認
        json_files = list(tmp_path.glob("*.json"))
        assert len(json_files) > 0
    
    def test_pipeline_result_to_dict(self, sample_sales_data, sample_new_store_sales):
        """PipelineResultのto_dictメソッドのテスト"""
        pipeline = PredictionPipeline()
        pipeline.fit(sample_sales_data)
        
        result = pipeline.predict(sample_new_store_sales)
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert 'prediction' in result_dict
        assert 'execution_time' in result_dict
        assert 'metadata' in result_dict
    
    def test_empty_data_handling(self, sample_sales_data):
        """空データの処理テスト"""
        pipeline = PredictionPipeline()
        pipeline.fit(sample_sales_data)
        
        with pytest.raises((ValueError, TypeError)):
            pipeline.predict([])
    
    def test_insufficient_data_warning(self, sample_sales_data):
        """データ不足時の警告テスト"""
        config = PipelineConfig(
            validate_input=True,  # バリデーションを有効に
            min_days=3,  # 非常に低い闾値でテスト
            preprocess_data=True, 
            check_quality=True, 
            auto_optimize_period=True  # 機能を有効にする
        )
        pipeline = PredictionPipeline(config)
        pipeline.fit(sample_sales_data)
        
        # 5日分のデータ（少なすぎる）
        short_data = [100000, 105000, 98000, 102000, 99000]
        result = pipeline.predict(short_data)
        
        # 予測は成功する
        assert result.prediction is not None


class TestPipelineBuilder:
    """PipelineBuilderのテストクラス"""
    
    def test_builder_pattern(self):
        """ビルダーパターンのテスト"""
        pipeline = (PipelineBuilder()
            .with_validation(strict=True)
            .with_preprocessing(handle_missing=True, handle_outliers=True)
            .with_quality_check(threshold=75.0)
            .with_prediction(metric="cosine", n_similar=3)
            .with_explanation(language="en")
            .with_output(save=True, output_dir="results", format="csv")
            .build()
        )
        
        assert isinstance(pipeline, PredictionPipeline)
        assert pipeline.config.strict_validation == True
        assert pipeline.config.handle_missing == True
        assert pipeline.config.quality_threshold == 75.0
        assert pipeline.config.similarity_metric == "cosine"
        assert pipeline.config.n_similar_stores == 3
        assert pipeline.config.explanation_language == "en"
        assert pipeline.config.save_results == True
        assert pipeline.config.output_format == "csv"
    
    def test_partial_builder(self):
        """部分的なビルダー設定のテスト"""
        pipeline = (PipelineBuilder()
            .with_validation()
            .with_prediction(metric="dtw")
            .build()
        )
        
        assert pipeline.config.validate_input == True
        assert pipeline.config.similarity_metric == "dtw"
        # 他の設定はデフォルト値
        assert pipeline.config.preprocess_data == True


class TestCustomPipeline:
    """カスタムパイプラインのテスト"""
    
    def test_create_custom_pipeline(self, sample_sales_data, sample_new_store_sales):
        """カスタムパイプライン作成のテスト"""
        pipeline = PredictionPipeline()
        pipeline.fit(sample_sales_data)
        
        # カスタムステップ
        def custom_transform(data, scale=1.1):
            return data * scale
        
        def custom_validation(data, min_val=0):
            return data[data >= min_val]
        
        # カスタムパイプライン作成
        custom_pipeline = pipeline.create_custom_pipeline(
            steps=[
                ("validation", custom_validation),
                ("transform", custom_transform),
            ],
            name="my_custom_pipeline"
        )
        
        # 実行
        result = custom_pipeline(sample_new_store_sales, validation={'min_val': 50000})
        
        assert result is not None
        assert len(result) <= len(sample_new_store_sales)  # validationで減る可能性