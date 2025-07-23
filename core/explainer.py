"""
予測説明生成モジュール

予測結果の根拠を自然言語で説明する機能を提供する。
"""

from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass
import warnings


@dataclass
class ExplanationComponents:
    """説明の構成要素"""
    similar_stores_info: List[Dict[str, Any]]
    multiplier_analysis: Dict[str, float]
    confidence_factors: Dict[str, float]
    key_insights: List[str]
    warnings: List[str]


class PredictionExplainer:
    """
    予測結果の説明を生成するクラス
    
    予測の根拠、信頼度の要因、注意点などを
    わかりやすい形で説明する。
    """
    
    def __init__(self, language: str = "ja"):
        """
        Parameters
        ----------
        language : str, default='ja'
            説明の言語（'ja': 日本語, 'en': 英語）
        """
        self.language = language
        self._templates = self._load_templates()
    
    def generate_explanation(
        self,
        prediction_result: Any,  # PredictionResultオブジェクト
        new_store_sales: np.ndarray,
        historical_data: Dict[str, np.ndarray],
        store_attributes: Optional[Dict[str, Dict]] = None,
    ) -> str:
        """
        予測結果の説明を生成
        
        Parameters
        ----------
        prediction_result : PredictionResult
            予測結果オブジェクト
        new_store_sales : np.ndarray
            新規店舗の売上データ
        historical_data : dict
            過去の店舗売上データ
        store_attributes : dict, optional
            店舗属性情報
            
        Returns
        -------
        str
            生成された説明文
        """
        # 説明の構成要素を収集
        components = self._analyze_prediction(
            prediction_result,
            new_store_sales,
            historical_data,
            store_attributes
        )
        
        # 説明文を生成
        explanation = self._format_explanation(components, prediction_result)
        
        return explanation
    
    def generate_summary(
        self,
        prediction_result: Any,
        format: str = "short"
    ) -> str:
        """
        予測結果のサマリーを生成
        
        Parameters
        ----------
        prediction_result : PredictionResult
            予測結果オブジェクト
        format : str, default='short'
            サマリーの形式（'short', 'detailed'）
            
        Returns
        -------
        str
            サマリー文
        """
        if format == "short":
            return self._generate_short_summary(prediction_result)
        else:
            return self._generate_detailed_summary(prediction_result)
    
    def _analyze_prediction(
        self,
        prediction_result: Any,
        new_store_sales: np.ndarray,
        historical_data: Dict[str, np.ndarray],
        store_attributes: Optional[Dict[str, Dict]]
    ) -> ExplanationComponents:
        """予測の分析"""
        components = ExplanationComponents(
            similar_stores_info=[],
            multiplier_analysis={},
            confidence_factors={},
            key_insights=[],
            warnings=[]
        )
        
        # 類似店舗の情報を収集
        for store_cd, similarity_score in prediction_result.similar_stores:
            store_info = {
                "store_cd": store_cd,
                "similarity_score": similarity_score,
                "sales_data": historical_data.get(store_cd, [])
            }
            
            if store_attributes and store_cd in store_attributes:
                store_info["attributes"] = store_attributes[store_cd]
            
            components.similar_stores_info.append(store_info)
        
        # 倍率の分析
        self._analyze_multipliers(components, prediction_result, new_store_sales, historical_data)
        
        # 信頼度の要因分析
        self._analyze_confidence_factors(components, prediction_result, new_store_sales)
        
        # 主要な洞察の抽出
        self._extract_key_insights(components, prediction_result, new_store_sales)
        
        return components
    
    def _analyze_multipliers(
        self,
        components: ExplanationComponents,
        prediction_result: Any,
        new_store_sales: np.ndarray,
        historical_data: Dict[str, np.ndarray]
    ):
        """倍率の分析"""
        multipliers = []
        
        for store_cd, _ in prediction_result.similar_stores[:3]:  # 上位3店舗
            if store_cd in historical_data:
                hist_sales = historical_data[store_cd]
                period_sales = np.sum(hist_sales[:len(new_store_sales)])
                annual_sales = np.sum(hist_sales[:365]) if len(hist_sales) >= 365 else np.sum(hist_sales)
                
                if period_sales > 0:
                    multiplier = annual_sales / period_sales
                    multipliers.append(multiplier)
        
        if multipliers:
            components.multiplier_analysis = {
                "average": np.mean(multipliers),
                "min": np.min(multipliers),
                "max": np.max(multipliers),
                "std": np.std(multipliers)
            }
    
    def _analyze_confidence_factors(
        self,
        components: ExplanationComponents,
        prediction_result: Any,
        new_store_sales: np.ndarray
    ):
        """信頼度の要因分析"""
        # データ量による信頼度
        data_days = len(new_store_sales)
        if data_days >= 30:
            data_confidence = "high"
        elif data_days >= 14:
            data_confidence = "medium"
        else:
            data_confidence = "low"
        
        # 類似店舗数による信頼度
        n_similar = len(prediction_result.similar_stores)
        if n_similar >= 5:
            store_confidence = "high"
        elif n_similar >= 3:
            store_confidence = "medium"
        else:
            store_confidence = "low"
        
        components.confidence_factors = {
            "data_days": data_days,
            "data_confidence": data_confidence,
            "n_similar_stores": n_similar,
            "store_confidence": store_confidence,
            "overall_score": prediction_result.confidence_score
        }
    
    def _extract_key_insights(
        self,
        components: ExplanationComponents,
        prediction_result: Any,
        new_store_sales: np.ndarray
    ):
        """主要な洞察の抽出"""
        # 売上の成長トレンド
        if len(new_store_sales) >= 7:
            week1_avg = np.mean(new_store_sales[:7])
            last_week_avg = np.mean(new_store_sales[-7:])
            growth_rate = (last_week_avg - week1_avg) / week1_avg if week1_avg > 0 else 0
            
            if growth_rate > 0.1:
                components.key_insights.append("steady_growth")
            elif growth_rate < -0.1:
                components.key_insights.append("declining_trend")
            else:
                components.key_insights.append("stable_performance")
        
        # 予測の信頼度
        if prediction_result.confidence_score > 0.8:
            components.key_insights.append("high_confidence")
        elif prediction_result.confidence_score < 0.5:
            components.warnings.append("low_confidence")
        
        # 予測範囲の幅
        prediction_range = prediction_result.upper_bound - prediction_result.lower_bound
        range_ratio = prediction_range / prediction_result.prediction if prediction_result.prediction > 0 else 0
        
        if range_ratio > 0.5:
            components.warnings.append("wide_prediction_range")
    
    def _format_explanation(
        self,
        components: ExplanationComponents,
        prediction_result: Any
    ) -> str:
        """説明文のフォーマット"""
        if self.language == "ja":
            return self._format_japanese_explanation(components, prediction_result)
        else:
            return self._format_english_explanation(components, prediction_result)
    
    def _format_japanese_explanation(
        self,
        components: ExplanationComponents,
        prediction_result: Any
    ) -> str:
        """日本語の説明文を生成"""
        lines = []
        
        # ヘッダー
        lines.append("【予測根拠】")
        lines.append("")
        
        # 類似店舗情報
        lines.append("1. 最も類似した店舗:")
        for i, store_info in enumerate(components.similar_stores_info[:3], 1):
            similarity = store_info['similarity_score']
            if hasattr(prediction_result, 'similarity_metric') and prediction_result.similarity_metric == 'dtw':
                lines.append(f"   {i}. {store_info['store_cd']} (DTW距離: {similarity:.3f})")
            else:
                lines.append(f"   {i}. {store_info['store_cd']} (類似度: {similarity:.3f})")
        
        # 倍率分析
        if components.multiplier_analysis:
            lines.append("")
            lines.append("2. 売上倍率の分析:")
            avg_multiplier = components.multiplier_analysis['average']
            lines.append(f"   - 平均倍率: {avg_multiplier:.1f}倍")
            lines.append(f"   - 範囲: {components.multiplier_analysis['min']:.1f}倍 〜 {components.multiplier_analysis['max']:.1f}倍")
        
        # 信頼度要因
        lines.append("")
        lines.append("3. 予測の信頼度:")
        lines.append(f"   - 総合スコア: {prediction_result.confidence_score:.2f}")
        lines.append(f"   - データ期間: {components.confidence_factors['data_days']}日間")
        lines.append(f"   - 類似店舗数: {components.confidence_factors['n_similar_stores']}店舗")
        
        # 調整要因
        if components.key_insights:
            lines.append("")
            lines.append("4. 特記事項:")
            insight_messages = {
                "steady_growth": "   - 安定した成長傾向を示しています",
                "declining_trend": "   - 売上の減少傾向が見られます",
                "stable_performance": "   - 売上は安定しています",
                "high_confidence": "   - 予測の信頼性は高いです"
            }
            for insight in components.key_insights:
                if insight in insight_messages:
                    lines.append(insight_messages[insight])
        
        # 警告
        if components.warnings:
            lines.append("")
            lines.append("5. 注意事項:")
            warning_messages = {
                "low_confidence": "   - 予測の信頼性が低いため、参考程度にご利用ください",
                "wide_prediction_range": "   - 予測範囲が広いため、不確実性が高いです"
            }
            for warning in components.warnings:
                if warning in warning_messages:
                    lines.append(warning_messages[warning])
        
        return "\n".join(lines)
    
    def _format_english_explanation(
        self,
        components: ExplanationComponents,
        prediction_result: Any
    ) -> str:
        """英語の説明文を生成"""
        lines = []
        
        lines.append("【Prediction Rationale】")
        lines.append("")
        
        lines.append("1. Most Similar Stores:")
        for i, store_info in enumerate(components.similar_stores_info[:3], 1):
            similarity = store_info['similarity_score']
            lines.append(f"   {i}. {store_info['store_cd']} (similarity: {similarity:.3f})")
        
        if components.multiplier_analysis:
            lines.append("")
            lines.append("2. Sales Multiplier Analysis:")
            avg_multiplier = components.multiplier_analysis['average']
            lines.append(f"   - Average multiplier: {avg_multiplier:.1f}x")
            lines.append(f"   - Range: {components.multiplier_analysis['min']:.1f}x - {components.multiplier_analysis['max']:.1f}x")
        
        lines.append("")
        lines.append("3. Prediction Confidence:")
        lines.append(f"   - Overall score: {prediction_result.confidence_score:.2f}")
        lines.append(f"   - Data period: {components.confidence_factors['data_days']} days")
        lines.append(f"   - Similar stores: {components.confidence_factors['n_similar_stores']} stores")
        
        return "\n".join(lines)
    
    def _generate_short_summary(self, prediction_result: Any) -> str:
        """短いサマリーを生成"""
        if self.language == "ja":
            return (
                f"予測年間売上: {prediction_result.prediction:,.0f}円 "
                f"(信頼度: {prediction_result.confidence_score:.0%})"
            )
        else:
            return (
                f"Predicted annual sales: {prediction_result.prediction:,.0f} "
                f"(confidence: {prediction_result.confidence_score:.0%})"
            )
    
    def _generate_detailed_summary(self, prediction_result: Any) -> str:
        """詳細なサマリーを生成"""
        if self.language == "ja":
            lines = [
                f"予測年間売上: {prediction_result.prediction:,.0f}円",
                f"信頼区間: {prediction_result.lower_bound:,.0f} - {prediction_result.upper_bound:,.0f}円",
                f"予測信頼度: {prediction_result.confidence_score:.0%}",
                f"使用した類似店舗数: {len(prediction_result.similar_stores)}店舗",
                f"予測手法: {prediction_result.prediction_method}"
            ]
        else:
            lines = [
                f"Predicted annual sales: {prediction_result.prediction:,.0f}",
                f"Confidence interval: {prediction_result.lower_bound:,.0f} - {prediction_result.upper_bound:,.0f}",
                f"Prediction confidence: {prediction_result.confidence_score:.0%}",
                f"Similar stores used: {len(prediction_result.similar_stores)} stores",
                f"Prediction method: {prediction_result.prediction_method}"
            ]
        
        return "\n".join(lines)
    
    def _load_templates(self) -> Dict[str, str]:
        """説明テンプレートのロード"""
        # 将来的に外部ファイルから読み込む
        return {}