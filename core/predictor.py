"""
売上予測エンジン

類似店舗の実績データを基に、新規店舗の年間売上を予測する
メインのエンジンクラス。
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Literal
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings

from .similarity import SimilarityEngine
from .normalizer import DataNormalizer
from ..config import PREDICTION_CONSTANTS, TIME_SERIES_CONSTANTS, STATISTICS_CONSTANTS
from ..config.defaults import PREDICTOR_DEFAULTS, INDUSTRY_PRESETS
from ..types import SalesData, SingleStoreSales, StoreAttributes


@dataclass
class PredictionResult:
    """予測結果を格納するデータクラス"""
    prediction: float  # 予測年間売上
    lower_bound: float  # 信頼区間下限
    upper_bound: float  # 信頼区間上限
    confidence_score: float  # 予測の信頼度スコア（0-1）
    similar_stores: List[Tuple[str, float]]  # 類似店舗リスト
    prediction_method: str  # 使用した予測手法
    metadata: Dict[str, Any]  # その他のメタデータ


@dataclass
class OptimalPeriodResult:
    """最適マッチング期間の探索結果"""
    recommended_days: int  # 推奨マッチング期間（日数）
    accuracy_scores: Dict[int, float]  # 各期間での精度スコア
    stability_score: float  # 予測の安定性スコア


class SalesPredictor:
    """
    売上予測を行うメインクラス
    
    類似店舗マッチングと統計的手法を組み合わせて、
    新規店舗の年間売上を予測する。
    """
    
    def __init__(
        self,
        similarity_metric: str = "dtw",
        normalization_method: str = "z-score",
        preset: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        similarity_metric : str, default='dtw'
            類似性計算に使用するメトリクス
        normalization_method : str, default='z-score'
            データ正規化手法
        preset : str, optional
            業態別のプリセット設定（'restaurant', 'retail', 'service'）
        """
        self.similarity_metric = similarity_metric
        self.normalization_method = normalization_method
        self.preset = preset
        
        # エンジンの初期化
        self.similarity_engine = SimilarityEngine(metric=similarity_metric)
        self.normalizer = DataNormalizer(method=normalization_method)
        
        # 学習データの保存
        self.historical_data = {}
        self.store_attributes = {}
        
        # プリセット設定の適用
        if preset:
            self._apply_preset(preset)
    
    def _apply_preset(self, preset: str) -> None:
        """業態別のプリセット設定を適用"""
        if preset not in INDUSTRY_PRESETS:
            warnings.warn(f"Unknown preset: {preset}. Using default settings.")
            return
        
        self.preset_config: Dict[str, Any] = INDUSTRY_PRESETS[preset]
    
    def fit(
        self,
        historical_sales_data: SalesData,
        store_attributes: Optional[StoreAttributes] = None,
    ) -> "SalesPredictor":
        """
        過去の店舗データで学習
        
        Parameters
        ----------
        historical_sales_data : pd.DataFrame or dict
            過去の店舗売上データ
            - DataFrame: 行が日付、列が店舗IDの形式
            - dict: {店舗ID: 売上配列}の形式
        store_attributes : pd.DataFrame, optional
            店舗属性データ（立地、面積、タイプなど）
            
        Returns
        -------
        self : SalesPredictor
            学習済みのインスタンス
        """
        # データの形式を統一
        if isinstance(historical_sales_data, pd.DataFrame):
            self.historical_data: Dict[str, np.ndarray] = {
                col: historical_sales_data[col].values
                for col in historical_sales_data.columns
            }
        else:
            self.historical_data = historical_sales_data
        
        # 店舗属性の保存
        self.store_attributes: Optional[Dict[str, Dict[str, Any]]] = None
        if store_attributes is not None:
            self.store_attributes = store_attributes.to_dict('index')
        
        # データの検証
        self._validate_historical_data()
        
        return self
    
    def predict(
        self,
        new_store_sales: SingleStoreSales,
        matching_period_days: Optional[int] = None,
        n_similar: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        confidence_level: Optional[float] = None,
    ) -> PredictionResult:
        """
        新規店舗の年間売上を予測
        
        Parameters
        ----------
        new_store_sales : array-like
            新規店舗の売上データ（日次）
        matching_period_days : int, optional
            マッチングに使用する期間（日数）
        n_similar : int, optional
            使用する類似店舗数。Noneの場合はデフォルト値を使用
        filters : dict, optional
            店舗属性によるフィルタリング条件
        confidence_level : float, optional
            信頼区間の水準。Noneの場合はデフォルト値を使用
            
        Returns
        -------
        PredictionResult
            予測結果
        """
        # 履歴データの存在確認
        if not hasattr(self, 'historical_data') or not self.historical_data:
            raise ValueError("No historical data available. Please call fit() first.")
        
        # デフォルト値の設定
        if n_similar is None:
            n_similar = getattr(self, 'preset_config', {}).get('n_similar', PREDICTION_CONSTANTS['DEFAULT_N_SIMILAR'])
        if confidence_level is None:
            confidence_level = getattr(self, 'preset_config', {}).get('confidence_level', PREDICTOR_DEFAULTS['confidence_level'])
        
        # 入力データの変換
        new_sales = np.asarray(new_store_sales).flatten()
        
        # 空データのチェック
        if len(new_sales) == 0:
            raise ValueError("Input sales data is empty")
        
        # マッチング期間の決定
        if matching_period_days is None:
            matching_period_days = len(new_sales)
        else:
            matching_period_days = min(matching_period_days, len(new_sales))
        
        # 使用するデータの切り出し
        new_sales_subset = new_sales[:matching_period_days]
        
        # フィルタリングされた候補店舗の取得
        candidate_stores = self._filter_stores(filters)
        
        # 類似店舗の検索
        similar_stores = self._find_similar_stores(
            new_sales_subset,
            candidate_stores,
            n_similar
        )
        
        # 予測の実行
        prediction, lower, upper = self._calculate_prediction(
            new_sales_subset,
            similar_stores,
            confidence_level
        )
        
        # 信頼度スコアの計算
        confidence_score = self._calculate_confidence_score(
            similar_stores,
            new_sales_subset
        )
        
        # 結果の構築
        result = PredictionResult(
            prediction=prediction,
            lower_bound=lower,
            upper_bound=upper,
            confidence_score=confidence_score,
            similar_stores=similar_stores,
            prediction_method="weighted_average",
            metadata={
                "matching_period_days": matching_period_days,
                "n_similar_used": len(similar_stores),
                "filters_applied": filters,
            }
        )
        
        return result
    
    def suggest_matching_period(
        self,
        new_store_sales: Union[np.ndarray, pd.Series, List[float]],
        min_days: Optional[int] = None,
        max_days: Optional[int] = None,
        step_days: Optional[int] = None,
    ) -> OptimalPeriodResult:
        """
        最適なマッチング期間を提案
        
        Parameters
        ----------
        new_store_sales : array-like
            新規店舗の売上データ
        min_days : int, optional
            最小期間（日数）
        max_days : int, optional
            最大期間（日数）
        step_days : int, optional
            探索のステップ幅。Noneの場合はデフォルト値を使用
            
        Returns
        -------
        OptimalPeriodResult
            最適期間の探索結果
        """
        new_sales = np.asarray(new_store_sales).flatten()
        
        # デフォルト値の設定
        if step_days is None:
            step_days = TIME_SERIES_CONSTANTS['MIN_DAYS']
        
        # 期間の範囲を設定
        if min_days is None:
            min_days = getattr(self, 'preset_config', {}).get('min_matching_days', TIME_SERIES_CONSTANTS['MIN_DAYS'])
        if max_days is None:
            max_days = getattr(self, 'preset_config', {}).get('max_matching_days', PREDICTION_CONSTANTS['PERIOD_OPTIMIZATION_RANGE'][1])
        
        max_days = min(max_days, len(new_sales))
        
        # 各期間での精度を評価（安全な実装）
        accuracy_scores = {}
        
        # 範囲が無効な場合の安全処理
        if max_days < min_days:
            max_days = min_days
        
        for days in range(min_days, max_days + 1, step_days):
            if days > len(new_sales) or days <= 0:
                break
            
            try:
                # クロスバリデーション的な評価
                score = self._evaluate_period_accuracy(new_sales, days)
                # スコアの妥当性チェック
                if isinstance(score, (int, float)) and not np.isnan(score) and not np.isinf(score):
                    accuracy_scores[days] = score
            except Exception:
                # エラーが発生した期間はスキップ
                continue
        
        # 結果が空の場合の安全処理
        if not accuracy_scores:
            accuracy_scores[min_days] = 0.0
        
        # 最適期間の選択
        best_period = max(accuracy_scores, key=accuracy_scores.get)
        
        # 安定性スコアの計算
        stability_score = self._calculate_stability_score(accuracy_scores)
        
        return OptimalPeriodResult(
            recommended_days=best_period,
            accuracy_scores=accuracy_scores,
            stability_score=stability_score
        )
    
    def _validate_historical_data(self):
        """過去データの検証"""
        if not self.historical_data:
            raise ValueError("No historical data found. Call fit() first.")
        
        # データ長の確認
        lengths = [len(data) for data in self.historical_data.values()]
        if len(set(lengths)) > 1:
            warnings.warn("Historical data has different lengths across stores.")
    
    def _filter_stores(self, filters: Optional[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """店舗属性によるフィルタリング"""
        if filters is None or not self.store_attributes:
            return self.historical_data
        
        filtered = {}
        for store_cd, sales_data in self.historical_data.items():
            if store_cd not in self.store_attributes:
                continue
            
            attrs = self.store_attributes[store_cd]
            match = True
            
            for key, value in filters.items():
                if key not in attrs:
                    match = False
                    break
                if attrs[key] != value:
                    match = False
                    break
            
            if match:
                filtered[store_cd] = sales_data
        
        return filtered
    
    def _find_similar_stores(
        self,
        target_sales: np.ndarray,
        candidate_stores: Dict[str, np.ndarray],
        n_similar: int
    ) -> List[Tuple[str, float]]:
        """類似店舗の検索"""
        # 正規化
        normalized_target = self.normalizer.fit_transform(target_sales)
        
        normalized_candidates = {}
        for store_cd, sales in candidate_stores.items():
            # 期間を揃える
            sales_subset = sales[:len(target_sales)]
            if len(sales_subset) < len(target_sales):
                continue
            normalized_candidates[store_cd] = self.normalizer.fit_transform(sales_subset)
        
        # 類似店舗の検索
        similar_stores = self.similarity_engine.find_similar_stores(
            normalized_target,
            normalized_candidates,
            top_k=n_similar
        )
        
        return similar_stores
    
    def _calculate_prediction(
        self,
        new_sales: np.ndarray,
        similar_stores: List[Tuple[str, float]],
        confidence_level: float
    ) -> Tuple[float, float, float]:
        """予測値と信頼区間を計算"""
        if not similar_stores:
            raise ValueError("No similar stores found for prediction")
        
        # 類似店舗の年間売上倍率を計算
        multipliers = []
        weights = []
        
        for store_cd, similarity_score in similar_stores:
            historical_sales = self.historical_data[store_cd]
            
            # 同期間の売上
            period_sales = np.sum(historical_sales[:len(new_sales)])
            # 年間売上
            annual_sales = np.sum(historical_sales[:365]) if len(historical_sales) >= 365 else np.sum(historical_sales)
            
            if period_sales > 0:
                multiplier = annual_sales / period_sales
                multipliers.append(multiplier)
                
                # 類似度を重みとして使用
                if self.similarity_metric == "dtw":
                    # DTWは距離なので逆数を重みに
                    weight = 1.0 / (1.0 + similarity_score)
                else:
                    weight = similarity_score
                weights.append(weight)
        
        if not multipliers:
            raise ValueError("Could not calculate multipliers from similar stores")
        
        # 重み付き平均と標準偏差
        multipliers = np.array(multipliers)
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # 正規化
        
        mean_multiplier = np.average(multipliers, weights=weights)
        std_multiplier = np.sqrt(np.average((multipliers - mean_multiplier)**2, weights=weights))
        
        # 予測値の計算
        period_sum = np.sum(new_sales)
        prediction = period_sum * mean_multiplier
        
        # 信頼区間の計算
        z_score = self._get_z_score(confidence_level)
        margin = z_score * std_multiplier * period_sum
        
        lower_bound = prediction - margin
        upper_bound = prediction + margin
        
        return prediction, lower_bound, upper_bound
    
    def _calculate_confidence_score(
        self,
        similar_stores: List[Tuple[str, float]],
        new_sales: np.ndarray
    ) -> float:
        """予測の信頼度スコアを計算"""
        if not similar_stores:
            return 0.0
        
        # 要素1: 類似度スコアの平均
        similarity_scores = [score for _, score in similar_stores]
        if self.similarity_metric == "dtw":
            # DTWは距離なので変換
            avg_similarity = 1.0 / (1.0 + np.mean(similarity_scores))
        else:
            avg_similarity = np.mean(similarity_scores)
        
        # 要素2: データの完全性（より厳しく）
        if len(new_sales) <= PREDICTION_CONSTANTS['MIN_SIMILAR_STORES']:
            data_completeness = 0.05  # 極めて低い信頼度
        elif len(new_sales) < TIME_SERIES_CONSTANTS['MIN_DAYS']:
            data_completeness = 0.1  # 非常に低い信頼度
        elif len(new_sales) < TIME_SERIES_CONSTANTS['RECOMMENDED_DAYS'] * 2:
            data_completeness = 0.3
        else:
            data_completeness = min(len(new_sales) / TIME_SERIES_CONSTANTS['RECENT_DAYS_WINDOW'], 1.0)
        
        # 要素3: 類似店舗数
        store_count_score = min(len(similar_stores) / PREDICTION_CONSTANTS['DEFAULT_N_SIMILAR'], 1.0)  # デフォルト店舗数を基準
        
        # 総合スコア（データが少ない場合はdata_completenessの影響を大きく）
        if len(new_sales) <= PREDICTION_CONSTANTS['MIN_SIMILAR_STORES']:
            confidence_score = (avg_similarity * 0.2 + 
                              data_completeness * 0.7 + 
                              store_count_score * 0.1)
        else:
            confidence_score = (avg_similarity * 0.5 + 
                              data_completeness * 0.3 + 
                              store_count_score * 0.2)
        
        return np.clip(confidence_score, 0.0, 1.0)
    
    def _evaluate_period_accuracy(self, sales_data: np.ndarray, period_days: int) -> float:
        """特定期間でのマッチング精度を評価"""
        # 簡易的な評価（実際にはクロスバリデーション等を実装）
        # ここでは類似店舗の一貫性を評価
        subset = sales_data[:period_days]
        
        # 正規化後の分散を評価（小さいほど良い）
        normalized = self.normalizer.fit_transform(subset)
        variance_score = 1.0 / (1.0 + np.var(normalized))
        
        # データ量による補正
        data_score = min(period_days / TIME_SERIES_CONSTANTS['RECENT_DAYS_WINDOW'], 1.0)
        
        return variance_score * data_score
    
    def _calculate_stability_score(self, accuracy_scores: Dict[int, float]) -> float:
        """予測の安定性スコアを計算"""
        if len(accuracy_scores) < 2:
            return 0.0
        
        scores = list(accuracy_scores.values())
        # 変動係数の逆数をスコアとする
        cv = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 1.0
        stability = 1.0 / (1.0 + cv)
        
        return stability
    
    def _get_z_score(self, confidence_level: float) -> float:
        """信頼水準に対応するz値を取得"""
        z_scores = {
            0.90: 1.645,
            0.95: STATISTICS_CONSTANTS['CONFIDENCE_MULTIPLIER'],  # 1.96
            0.99: 2.576
        }
        
        # 最も近い値を使用
        closest_level = min(z_scores.keys(), key=lambda x: abs(x - confidence_level))
        return z_scores[closest_level]
    
    def create_visualizer(self):
        """可視化オブジェクトを作成（将来の実装用）"""
        # visualization モジュールが実装されたら import する
        warnings.warn("Visualizer not yet implemented")
        return None
    
    def generate_report(
        self,
        store_name: str,
        results: PredictionResult,
        format: str = "pdf",
        language: str = "ja"
    ):
        """レポートを生成（将来の実装用）"""
        # reporting モジュールが実装されたら import する
        warnings.warn("Report generation not yet implemented")
        return None