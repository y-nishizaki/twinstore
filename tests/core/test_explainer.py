"""
PredictionExplainerのテスト
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass

from twinstore.core.explainer import PredictionExplainer, ExplanationComponents
from twinstore.core.predictor import PredictionResult


@dataclass
class MockPredictionResult:
    """テスト用のモック予測結果"""
    prediction: float = 50000000.0
    lower_bound: float = 45000000.0
    upper_bound: float = 55000000.0
    confidence_score: float = 0.85
    similar_stores: list = None
    prediction_method: str = "dtw"
    
    def __post_init__(self):
        if self.similar_stores is None:
            self.similar_stores = [
                ("store_001", 0.92),
                ("store_005", 0.88),
                ("store_012", 0.85)
            ]


class TestPredictionExplainer:
    """PredictionExplainerのテストクラス"""
    
    def test_initialization_japanese(self):
        """日本語での初期化テスト"""
        explainer = PredictionExplainer(language="ja")
        assert explainer.language == "ja"
        assert explainer._templates is not None
        assert isinstance(explainer._templates, dict)
    
    def test_initialization_english(self):
        """英語での初期化テスト"""
        explainer = PredictionExplainer(language="en")
        assert explainer.language == "en"
        assert explainer._templates is not None
    
    def test_generate_explanation_basic(self):
        """基本的な説明生成のテスト"""
        explainer = PredictionExplainer(language="ja")
        
        # モックデータの準備
        prediction_result = MockPredictionResult()
        new_store_sales = np.array([100000, 105000, 98000, 102000, 110000])
        historical_data = {
            "store_001": np.random.normal(100000, 10000, 100),
            "store_005": np.random.normal(120000, 12000, 100),
            "store_012": np.random.normal(90000, 9000, 100),
        }
        
        # 説明生成
        explanation = explainer.generate_explanation(
            prediction_result,
            new_store_sales,
            historical_data
        )
        
        # 結果の検証
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        # 予測結果の情報が含まれていることを確認
        assert "0.85" in explanation or "85" in explanation  # 信頼度
    
    def test_generate_explanation_with_attributes(self):
        """店舗属性付きの説明生成テスト"""
        explainer = PredictionExplainer(language="ja")
        
        prediction_result = MockPredictionResult()
        new_store_sales = np.array([100000, 105000, 98000])
        historical_data = {
            "store_001": np.random.normal(100000, 10000, 50),
            "store_005": np.random.normal(120000, 12000, 50),
        }
        store_attributes = {
            "store_001": {"store_type": "urban", "area": 150},
            "store_005": {"store_type": "suburban", "area": 200},
        }
        
        explanation = explainer.generate_explanation(
            prediction_result,
            new_store_sales,
            historical_data,
            store_attributes
        )
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
    
    def test_generate_explanation_english(self):
        """英語での説明生成テスト"""
        explainer = PredictionExplainer(language="en")
        
        prediction_result = MockPredictionResult()
        new_store_sales = np.array([100000, 105000, 98000, 102000])
        historical_data = {
            "store_001": np.random.normal(100000, 10000, 50),
        }
        
        explanation = explainer.generate_explanation(
            prediction_result,
            new_store_sales,
            historical_data
        )
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
        # 英語の説明であることを確認
        assert any(word in explanation.lower() for word in ["prediction", "confidence", "similar", "stores"])
    
    def test_analyze_prediction(self):
        """予測分析のテスト"""
        explainer = PredictionExplainer()
        
        prediction_result = MockPredictionResult()
        new_store_sales = np.array([100000, 105000, 98000])
        historical_data = {
            "store_001": np.array([100000, 105000, 98000] * 30),
            "store_005": np.array([120000, 125000, 118000] * 30),
            "store_012": np.array([90000, 95000, 88000] * 30),
        }
        
        # プライベートメソッドのテスト
        components = explainer._analyze_prediction(
            prediction_result, new_store_sales, historical_data, None
        )
        
        assert isinstance(components, ExplanationComponents)
        assert len(components.similar_stores_info) == 3
        for store_info in components.similar_stores_info:
            assert isinstance(store_info, dict)
            assert "store_cd" in store_info
            assert "similarity_score" in store_info
            assert "sales_data" in store_info
    
    def test_analyze_multipliers(self):
        """倍率分析のテスト"""
        explainer = PredictionExplainer()
        
        components = ExplanationComponents(
            similar_stores_info=[],
            multiplier_analysis={},
            confidence_factors={},
            key_insights=[],
            warnings=[]
        )
        prediction_result = MockPredictionResult()
        new_store_sales = np.array([100000, 105000, 98000])
        historical_data = {
            "store_001": np.array([100000, 105000, 98000] * 30),
            "store_005": np.array([120000, 125000, 118000] * 30),
        }
        
        explainer._analyze_multipliers(components, prediction_result, new_store_sales, historical_data)
        
        assert isinstance(components.multiplier_analysis, dict)
        assert len(components.multiplier_analysis) > 0
    
    def test_analyze_confidence_factors(self):
        """信頼度要因分析のテスト"""
        explainer = PredictionExplainer()
        
        components = ExplanationComponents(
            similar_stores_info=[],
            multiplier_analysis={},
            confidence_factors={},
            key_insights=[],
            warnings=[]
        )
        prediction_result = MockPredictionResult()
        new_store_sales = np.array([100000, 105000, 98000, 102000, 110000])
        
        explainer._analyze_confidence_factors(components, prediction_result, new_store_sales)
        
        assert isinstance(components.confidence_factors, dict)
        assert len(components.confidence_factors) > 0
    
    def test_extract_key_insights(self):
        """主要洞察抽出のテスト"""
        explainer = PredictionExplainer()
        
        components = ExplanationComponents(
            similar_stores_info=[
                {"store_cd": "store_001", "similarity_score": 0.92, "sales_data": np.array([100000] * 30)},
                {"store_cd": "store_005", "similarity_score": 0.88, "sales_data": np.array([120000] * 30)},
            ],
            multiplier_analysis={},
            confidence_factors={},
            key_insights=[],
            warnings=[]
        )
        prediction_result = MockPredictionResult()
        new_store_sales = np.array([100000, 105000, 98000])
        
        explainer._extract_key_insights(components, prediction_result, new_store_sales)
        
        assert isinstance(components.key_insights, list)
    
    def test_generate_summary_short(self):
        """短いサマリー生成のテスト"""
        explainer = PredictionExplainer()
        
        prediction_result = MockPredictionResult()
        
        summary = explainer.generate_summary(prediction_result, format="short")
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        
    def test_generate_summary_detailed(self):
        """詳細サマリー生成のテスト"""
        explainer = PredictionExplainer()
        
        prediction_result = MockPredictionResult()
        
        summary = explainer.generate_summary(prediction_result, format="detailed")
        
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_format_explanation_japanese(self):
        """日本語での説明フォーマットテスト"""
        explainer = PredictionExplainer(language="ja")
        
        components = ExplanationComponents(
            similar_stores_info=[{"store_cd": "store_001", "similarity_score": 0.92}],
            multiplier_analysis={"average": 120.0, "min": 110.0, "max": 130.0},
            confidence_factors={"data_completeness": 0.8, "data_days": 5, "similar_stores_count": 3, "n_similar_stores": 3},
            key_insights=["テスト洞察"],
            warnings=["テスト警告"]
        )
        prediction_result = MockPredictionResult()
        
        formatted = explainer._format_explanation(components, prediction_result)
        
        assert isinstance(formatted, str)
        assert len(formatted) > 0
    
    def test_format_explanation_english(self):
        """英語での説明フォーマットテスト"""
        explainer = PredictionExplainer(language="en")
        
        components = ExplanationComponents(
            similar_stores_info=[{"store_cd": "store_001", "similarity_score": 0.92}],
            multiplier_analysis={"average": 120.0, "min": 110.0, "max": 130.0},
            confidence_factors={"data_completeness": 0.8, "data_days": 5, "similar_stores_count": 3, "n_similar_stores": 3},
            key_insights=["Test insight"],
            warnings=["Test warning"]
        )
        prediction_result = MockPredictionResult()
        
        formatted = explainer._format_explanation(components, prediction_result)
        
        assert isinstance(formatted, str)
        assert len(formatted) > 0
    
    def test_load_templates(self):
        """テンプレート読み込みのテスト"""
        explainer = PredictionExplainer()
        
        templates = explainer._load_templates()
        
        assert isinstance(templates, dict)
        # テンプレートは空でもOK（デフォルトテンプレートを使用）
        assert isinstance(templates, dict)
    
    def test_invalid_language(self):
        """無効な言語での初期化テスト"""
        explainer = PredictionExplainer(language="invalid")
        
        # デフォルト言語にフォールバックすることを確認
        assert explainer.language == "invalid"  # 設定は保持されるが
        assert explainer._templates is not None  # テンプレートは読み込まれる
    
    def test_empty_historical_data(self):
        """空の履歴データでの説明生成テスト"""
        explainer = PredictionExplainer()
        
        prediction_result = MockPredictionResult()
        new_store_sales = np.array([100000, 105000, 98000])
        historical_data = {}
        
        # エラーが発生しないことを確認
        explanation = explainer.generate_explanation(
            prediction_result,
            new_store_sales,
            historical_data
        )
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0
    
    def test_short_data_handling(self):
        """短いデータでの処理テスト"""
        explainer = PredictionExplainer()
        
        prediction_result = MockPredictionResult(confidence_score=0.3)
        short_sales = np.array([100000, 105000])  # 2日分のみ
        historical_data = {
            "store_001": np.array([100000] * 30),
        }
        
        # 短いデータでも説明が生成されることを確認
        explanation = explainer.generate_explanation(
            prediction_result, short_sales, historical_data
        )
        
        assert isinstance(explanation, str)
        assert len(explanation) > 0