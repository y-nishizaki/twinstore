"""
異常値検知モジュール

時系列売上データの異常値をリアルタイムで検知し、
アラートを生成する機能を提供する。
"""

from typing import Union, List, Dict, Any, Optional, Tuple, Literal
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
import warnings

from ..config import TIME_SERIES_CONSTANTS, STATISTICS_CONSTANTS, QUALITY_CONSTANTS
from ..config.defaults import ANOMALY_DETECTOR_DEFAULTS


AlertLevel = Literal["info", "warning", "critical"]
DetectionMethod = Literal["statistical", "isolation_forest", "mad", "percentile"]


@dataclass
class AnomalyAlert:
    """異常検知アラート"""
    timestamp: datetime
    level: AlertLevel
    message: str
    value: float
    expected_range: Tuple[float, float]
    detection_method: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnomalyReport:
    """異常検知レポート"""
    total_points: int
    anomaly_count: int
    anomaly_ratio: float
    alerts: List[AnomalyAlert]
    statistics: Dict[str, Any]
    recommendations: List[str]


class AnomalyDetector:
    """
    異常値検知クラス
    
    複数の検知手法を組み合わせて、売上データの異常を
    高精度で検出する。
    """
    
    def __init__(
        self,
        method: Optional[DetectionMethod] = None,
        sensitivity: Optional[float] = None,
        min_history: Optional[int] = None,
    ):
        """
        Parameters
        ----------
        method : str, optional
            検知手法。Noneの場合はデフォルト値を使用
        sensitivity : float, optional
            検知感度（0-1）。Noneの場合はデフォルト値を使用
        min_history : int, optional
            最小履歴日数。Noneの場合はデフォルト値を使用
        """
        self.method = method or ANOMALY_DETECTOR_DEFAULTS['method']
        self.sensitivity = sensitivity or ANOMALY_DETECTOR_DEFAULTS['sensitivity']
        self.min_history = min_history or ANOMALY_DETECTOR_DEFAULTS['min_history']
        self._history = []
        self._model_params = {}
    
    def detect_anomalies(
        self,
        data: Union[pd.Series, np.ndarray, List[float]],
        timestamps: Optional[Union[pd.DatetimeIndex, List[datetime]]] = None,
        return_scores: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        異常値を検出
        
        Parameters
        ----------
        data : array-like
            検査するデータ
        timestamps : array-like, optional
            データのタイムスタンプ
        return_scores : bool, default=False
            異常スコアも返すか
            
        Returns
        -------
        anomalies : np.ndarray
            異常フラグ（True: 異常）
        scores : np.ndarray, optional
            異常スコア（return_scores=True時）
        """
        data_array = np.asarray(data).flatten()
        
        if self.method == "statistical":
            anomalies, scores = self._detect_statistical(data_array)
        elif self.method == "isolation_forest":
            anomalies, scores = self._detect_isolation_forest(data_array)
        elif self.method == "mad":
            anomalies, scores = self._detect_mad(data_array)
        elif self.method == "percentile":
            anomalies, scores = self._detect_percentile(data_array)
        else:
            raise ValueError(f"Unknown detection method: {self.method}")
        
        if return_scores:
            return anomalies, scores
        return anomalies
    
    def detect_realtime(
        self,
        new_value: float,
        timestamp: Optional[datetime] = None,
    ) -> Optional[AnomalyAlert]:
        """
        リアルタイム異常検知
        
        Parameters
        ----------
        new_value : float
            新しいデータポイント
        timestamp : datetime, optional
            タイムスタンプ
            
        Returns
        -------
        AnomalyAlert or None
            異常が検出された場合はアラート、それ以外はNone
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # 履歴に追加
        self._history.append((timestamp, new_value))
        
        # 履歴が十分でない場合はスキップ
        if len(self._history) < self.min_history:
            return None
        
        # 直近のデータで異常検知
        recent_values = [v for _, v in self._history[-TIME_SERIES_CONSTANTS['RECENT_DAYS_WINDOW']:]]  # 直近データ
        anomalies = self.detect_anomalies(recent_values + [new_value])
        
        if anomalies[-1]:  # 最新値が異常
            # 期待範囲を計算
            expected_range = self._calculate_expected_range(recent_values)
            
            # アラートレベルを決定
            level = self._determine_alert_level(new_value, recent_values)
            
            # アラートを生成
            alert = AnomalyAlert(
                timestamp=timestamp,
                level=level,
                message=self._generate_alert_message(new_value, expected_range, level),
                value=new_value,
                expected_range=expected_range,
                detection_method=self.method,
                metadata={
                    "history_size": len(self._history),
                    "recent_mean": np.mean(recent_values),
                    "recent_std": np.std(recent_values)
                }
            )
            
            return alert
        
        return None
    
    def analyze_period(
        self,
        data: Union[pd.Series, np.ndarray],
        timestamps: Optional[Union[pd.DatetimeIndex, List[datetime]]] = None,
        generate_report: bool = True,
    ) -> AnomalyReport:
        """
        期間全体の異常分析
        
        Parameters
        ----------
        data : array-like
            分析するデータ
        timestamps : array-like, optional
            タイムスタンプ
        generate_report : bool, default=True
            詳細レポートを生成するか
            
        Returns
        -------
        AnomalyReport
            異常分析レポート
        """
        data_array = np.asarray(data).flatten()
        
        # 異常検出
        anomalies, scores = self.detect_anomalies(data, return_scores=True)
        
        # アラートの生成
        alerts = []
        if timestamps is not None:
            for i, (is_anomaly, score) in enumerate(zip(anomalies, scores)):
                if is_anomaly:
                    # 前後のデータから期待範囲を計算
                    window_start = max(0, i - TIME_SERIES_CONSTANTS['MIN_DAYS'])
                    window_end = min(len(data_array), i + TIME_SERIES_CONSTANTS['MIN_DAYS'])
                    window_data = data_array[window_start:window_end]
                    window_data = window_data[window_data != data_array[i]]  # 当該値を除く
                    
                    if len(window_data) > 0:
                        expected_range = (
                            np.percentile(window_data, STATISTICS_CONSTANTS['PERCENTILE_LOW']),
                            np.percentile(window_data, STATISTICS_CONSTANTS['PERCENTILE_HIGH'])
                        )
                    else:
                        expected_range = (0, 0)
                    
                    alert = AnomalyAlert(
                        timestamp=timestamps[i] if i < len(timestamps) else datetime.now(),
                        level=self._determine_alert_level(data_array[i], window_data),
                        message=f"Anomaly detected: value={data_array[i]:.2f}",
                        value=data_array[i],
                        expected_range=expected_range,
                        detection_method=self.method,
                        metadata={"anomaly_score": float(score)}
                    )
                    alerts.append(alert)
        
        # 統計情報
        statistics = {
            "mean": float(np.mean(data_array)),
            "std": float(np.std(data_array)),
            "min": float(np.min(data_array)),
            "max": float(np.max(data_array)),
            "anomaly_scores": {
                "mean": float(np.mean(scores)),
                "max": float(np.max(scores)),
                "threshold": float(self._get_threshold())
            }
        }
        
        # レポート作成
        report = AnomalyReport(
            total_points=len(data_array),
            anomaly_count=int(np.sum(anomalies)),
            anomaly_ratio=float(np.mean(anomalies)),
            alerts=alerts,
            statistics=statistics,
            recommendations=self._generate_recommendations(anomalies, data_array)
        )
        
        return report
    
    def update_model(self, historical_data: Union[pd.Series, np.ndarray]):
        """
        モデルパラメータを更新
        
        Parameters
        ----------
        historical_data : array-like
            学習用の履歴データ
        """
        data_array = np.asarray(historical_data).flatten()
        
        # 手法に応じたパラメータ更新
        if self.method == "statistical":
            self._model_params = {
                "mean": np.mean(data_array),
                "std": np.std(data_array),
                "threshold": self._calculate_threshold(data_array)
            }
        elif self.method == "mad":
            median = np.median(data_array)
            mad = np.median(np.abs(data_array - median))
            self._model_params = {
                "median": median,
                "mad": mad,
                "threshold": self._calculate_threshold(data_array)
            }
    
    def _detect_statistical(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """統計的手法による異常検出"""
        # Z-scoreベース
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return np.zeros(len(data), dtype=bool), np.zeros(len(data))
        
        z_scores = np.abs((data - mean) / std)
        threshold = self._get_threshold()
        
        anomalies = z_scores > threshold
        
        return anomalies, z_scores
    
    def _detect_isolation_forest(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Isolation Forest による異常検出（簡易版）"""
        # 実際の実装では scikit-learn を使用
        # ここでは簡易的に外れ値を検出
        q1 = np.percentile(data, STATISTICS_CONSTANTS['QUARTILE_FIRST'])
        q3 = np.percentile(data, STATISTICS_CONSTANTS['QUARTILE_THIRD'])
        iqr = q3 - q1
        
        lower = q1 - STATISTICS_CONSTANTS['IQR_MULTIPLIER'] * iqr * (2 - self.sensitivity)
        upper = q3 + STATISTICS_CONSTANTS['IQR_MULTIPLIER'] * iqr * (2 - self.sensitivity)
        
        scores = np.zeros(len(data))
        scores[data < lower] = (lower - data[data < lower]) / iqr
        scores[data > upper] = (data[data > upper] - upper) / iqr
        
        anomalies = (data < lower) | (data > upper)
        
        return anomalies, scores
    
    def _detect_mad(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Median Absolute Deviation による異常検出"""
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        
        if mad == 0:
            # MADが0の場合はIQRを使用
            return self._detect_isolation_forest(data)
        
        # Modified Z-score
        modified_z_scores = 0.6745 * (data - median) / mad
        threshold = self._get_threshold()
        
        anomalies = np.abs(modified_z_scores) > threshold
        
        return anomalies, np.abs(modified_z_scores)
    
    def _detect_percentile(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """パーセンタイルベースの異常検出"""
        lower_percentile = (1 - self.sensitivity) * 50
        upper_percentile = 100 - lower_percentile
        
        lower_bound = np.percentile(data, lower_percentile)
        upper_bound = np.percentile(data, upper_percentile)
        
        # スコア計算
        scores = np.zeros(len(data))
        range_size = upper_bound - lower_bound
        
        if range_size > 0:
            scores[data < lower_bound] = (lower_bound - data[data < lower_bound]) / range_size
            scores[data > upper_bound] = (data[data > upper_bound] - upper_bound) / range_size
        
        anomalies = (data < lower_bound) | (data > upper_bound)
        
        return anomalies, scores
    
    def _get_threshold(self) -> float:
        """感度に基づく閾値を取得"""
        # 感度を閾値に変換
        if self.method == "statistical" or self.method == "mad":
            # 正規分布を仮定
            if self.sensitivity >= 0.99:
                return QUALITY_CONSTANTS['OUTLIER_THRESHOLD']
            elif self.sensitivity >= ANOMALY_DETECTOR_DEFAULTS['sensitivity']:
                return STATISTICS_CONSTANTS['ZSCORE_MEDIUM_SENSITIVITY']
            elif self.sensitivity >= 0.90:
                return STATISTICS_CONSTANTS['ZSCORE_HIGH_SENSITIVITY']
            else:
                return STATISTICS_CONSTANTS['IQR_MULTIPLIER']
        else:
            return self.sensitivity
    
    def _calculate_threshold(self, data: np.ndarray) -> float:
        """データから適応的に閾値を計算"""
        # ブートストラップ法で閾値を推定（簡易版）
        n_samples = min(1000, len(data))
        thresholds = []
        
        for _ in range(100):
            sample = np.random.choice(data, n_samples, replace=True)
            std = np.std(sample)
            if std > 0:
                thresholds.append(STATISTICS_CONSTANTS['ZSCORE_MEDIUM_SENSITIVITY'] * std)
        
        if thresholds:
            return np.percentile(thresholds, self.sensitivity * 100)
        else:
            return self._get_threshold()
    
    def _calculate_expected_range(self, recent_values: List[float]) -> Tuple[float, float]:
        """期待範囲を計算"""
        if not recent_values:
            return (0, 0)
        
        mean = np.mean(recent_values)
        std = np.std(recent_values)
        
        # 信頼区間
        confidence_multiplier = STATISTICS_CONSTANTS['CONFIDENCE_MULTIPLIER']  # 約95%信頼区間
        lower = mean - confidence_multiplier * std
        upper = mean + confidence_multiplier * std
        
        return (lower, upper)
    
    def _determine_alert_level(self, value: float, reference_values: np.ndarray) -> AlertLevel:
        """アラートレベルを決定"""
        if len(reference_values) == 0:
            return "info"
        
        mean = np.mean(reference_values)
        std = np.std(reference_values)
        
        if std == 0:
            return "info"
        
        z_score = abs((value - mean) / std)
        
        if z_score > STATISTICS_CONSTANTS['ZSCORE_CRITICAL']:
            return "critical"
        elif z_score > STATISTICS_CONSTANTS['ZSCORE_WARNING']:
            return "warning"
        else:
            return "info"
    
    def _generate_alert_message(
        self,
        value: float,
        expected_range: Tuple[float, float],
        level: AlertLevel
    ) -> str:
        """アラートメッセージを生成"""
        if value < expected_range[0]:
            direction = "below"
            diff = expected_range[0] - value
        else:
            direction = "above"
            diff = value - expected_range[1]
        
        messages = {
            "critical": f"Critical anomaly: Value {value:.2f} is {diff:.2f} {direction} expected range",
            "warning": f"Warning: Value {value:.2f} is {direction} expected range [{expected_range[0]:.2f}, {expected_range[1]:.2f}]",
            "info": f"Anomaly detected: Value {value:.2f} is outside normal range"
        }
        
        return messages.get(level, "Anomaly detected")
    
    def _generate_recommendations(self, anomalies: np.ndarray, data: np.ndarray) -> List[str]:
        """推奨事項を生成"""
        recommendations = []
        
        anomaly_ratio = np.mean(anomalies)
        
        if anomaly_ratio > 0.2:
            recommendations.append("High anomaly rate detected. Review data collection process.")
        elif anomaly_ratio > 0.1:
            recommendations.append("Consider investigating the anomalous periods for external factors.")
        
        # 連続異常のチェック
        consecutive_anomalies = self._find_consecutive_anomalies(anomalies)
        if any(run >= 3 for run in consecutive_anomalies):
            recommendations.append("Consecutive anomalies detected. This may indicate a systematic issue.")
        
        # トレンド異常
        if len(data) >= 14:
            recent_anomaly_ratio = np.mean(anomalies[-TIME_SERIES_CONSTANTS['MIN_DAYS']:])
            overall_ratio = np.mean(anomalies)
            
            if recent_anomaly_ratio > overall_ratio * 2:
                recommendations.append("Increasing anomaly trend in recent data. Monitor closely.")
        
        return recommendations
    
    def _find_consecutive_anomalies(self, anomalies: np.ndarray) -> List[int]:
        """連続異常を検出"""
        runs = []
        current_run = 0
        
        for is_anomaly in anomalies:
            if is_anomaly:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        
        if current_run > 0:
            runs.append(current_run)
        
        return runs