"""
データ品質チェックモジュール

売上データの品質を総合的に評価し、問題点を検出・報告する。
"""

from typing import Union, Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats

from ..config import QUALITY_CONSTANTS, SCORE_WEIGHTS, STATISTICS_CONSTANTS
from ..config.defaults import QUALITY_CHECKER_DEFAULTS


@dataclass
class QualityReport:
    """データ品質レポートを格納するクラス"""
    overall_score: float = 0.0  # 総合品質スコア（0-100）
    completeness_score: float = 0.0  # 完全性スコア
    consistency_score: float = 0.0  # 一貫性スコア
    accuracy_score: float = 0.0  # 正確性スコア
    timeliness_score: float = 0.0  # 適時性スコア
    
    issues: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def add_issue(self, issue_type: str, severity: str, description: str, details: Optional[Dict] = None):
        """問題を追加"""
        issue = {
            "type": issue_type,
            "severity": severity,  # 'critical', 'high', 'medium', 'low'
            "description": description,
            "details": details or {}
        }
        self.issues.append(issue)
    
    def get_summary(self) -> str:
        """サマリーを取得"""
        summary = []
        summary.append(f"Data Quality Report")
        summary.append("=" * 50)
        summary.append(f"Overall Score: {self.overall_score:.1f}/100")
        summary.append(f"- Completeness: {self.completeness_score:.1f}")
        summary.append(f"- Consistency: {self.consistency_score:.1f}")
        summary.append(f"- Accuracy: {self.accuracy_score:.1f}")
        summary.append(f"- Timeliness: {self.timeliness_score:.1f}")
        
        if self.issues:
            summary.append(f"\nIssues Found: {len(self.issues)}")
            severity_counts = {}
            for issue in self.issues:
                severity = issue['severity']
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            for severity in ['critical', 'high', 'medium', 'low']:
                if severity in severity_counts:
                    summary.append(f"  - {severity.capitalize()}: {severity_counts[severity]}")
        
        if self.recommendations:
            summary.append(f"\nRecommendations: {len(self.recommendations)}")
        
        return "\n".join(summary)


class QualityChecker:
    """
    データ品質をチェックするクラス
    
    売上データの完全性、一貫性、正確性、適時性を評価し、
    問題点の検出と改善提案を行う。
    """
    
    def __init__(
        self,
        missing_threshold: Optional[float] = None,
        outlier_threshold: Optional[float] = None,
        consistency_threshold: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        missing_threshold : float, default=0.1
            欠損値の許容割合
        outlier_threshold : float, default=3.0
            異常値判定のZ-score閾値
        consistency_threshold : float, default=0.2
            一貫性チェックの変動係数閾値
        """
        # デフォルト値を設定から取得
        self.missing_threshold = missing_threshold if missing_threshold is not None else QUALITY_CONSTANTS['MISSING_THRESHOLD']
        self.outlier_threshold = outlier_threshold if outlier_threshold is not None else QUALITY_CONSTANTS['OUTLIER_THRESHOLD']
        self.consistency_threshold = consistency_threshold if consistency_threshold is not None else QUALITY_CONSTANTS['CONSISTENCY_THRESHOLD']
    
    def check_data_quality(
        self,
        data: Union[pd.DataFrame, Dict[str, np.ndarray], pd.Series],
        reference_data: Optional[Union[pd.DataFrame, Dict]] = None,
        check_items: Optional[List[str]] = None,
    ) -> QualityReport:
        """
        データ品質を総合的にチェック
        
        Parameters
        ----------
        data : various
            チェック対象のデータ
        reference_data : various, optional
            比較用の参照データ（過去データなど）
        check_items : list, optional
            チェック項目のリスト（指定しない場合は全項目）
            
        Returns
        -------
        QualityReport
            品質チェック結果
        """
        report = QualityReport()
        
        # データ形式の統一
        df = self._convert_to_dataframe(data)
        
        # チェック項目の決定
        if check_items is None:
            check_items = ["completeness", "consistency", "accuracy", "timeliness"]
        
        # 各品質項目のチェック
        if "completeness" in check_items:
            self._check_completeness(df, report)
        
        if "consistency" in check_items:
            self._check_consistency(df, report)
        
        if "accuracy" in check_items:
            self._check_accuracy(df, report, reference_data)
        
        if "timeliness" in check_items:
            self._check_timeliness(df, report)
        
        # 統計情報の収集
        self._collect_statistics(df, report)
        
        # 総合スコアの計算
        self._calculate_overall_score(report)
        
        # 推奨事項の生成
        self._generate_recommendations(report)
        
        return report
    
    def check_anomalies(
        self,
        data: Union[pd.Series, np.ndarray],
        method: str = "zscore",
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        異常値を検出
        
        Parameters
        ----------
        data : array-like
            チェック対象のデータ
        method : str, default='zscore'
            異常値検出手法
            
        Returns
        -------
        anomalies : np.ndarray
            異常値フラグ（True: 異常）
        details : dict
            検出の詳細情報
        """
        data_array = np.asarray(data).flatten()
        
        if method == "zscore":
            anomalies, details = self._detect_anomalies_zscore(data_array)
        elif method == "iqr":
            anomalies, details = self._detect_anomalies_iqr(data_array)
        elif method == "seasonal":
            anomalies, details = self._detect_anomalies_seasonal(data_array)
        else:
            raise ValueError(f"Unknown anomaly detection method: {method}")
        
        return anomalies, details
    
    def check_patterns(
        self,
        data: Union[pd.Series, np.ndarray],
        patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        データパターンをチェック
        
        Parameters
        ----------
        data : array-like
            チェック対象のデータ
        patterns : list, optional
            チェックするパターンのリスト
            
        Returns
        -------
        dict
            パターンチェック結果
        """
        if patterns is None:
            patterns = ["trend", "seasonality", "weekly_pattern", "sudden_changes"]
        
        data_array = np.asarray(data).flatten()
        results = {}
        
        for pattern in patterns:
            if pattern == "trend":
                results["trend"] = self._check_trend(data_array)
            elif pattern == "seasonality":
                results["seasonality"] = self._check_seasonality(data_array)
            elif pattern == "weekly_pattern":
                results["weekly_pattern"] = self._check_weekly_pattern(data_array)
            elif pattern == "sudden_changes":
                results["sudden_changes"] = self._check_sudden_changes(data_array)
        
        return results
    
    def _convert_to_dataframe(self, data: Any) -> pd.DataFrame:
        """データをDataFrameに変換"""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, pd.Series):
            return pd.DataFrame({data.name or "sales": data})
        elif isinstance(data, dict):
            return pd.DataFrame(data)
        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                return pd.DataFrame({"sales": data})
            else:
                return pd.DataFrame(data)
        elif isinstance(data, (list, tuple)):
            return pd.DataFrame({"sales": data})
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _check_completeness(self, df: pd.DataFrame, report: QualityReport):
        """完全性のチェック"""
        total_cells = df.size
        missing_cells = df.isna().sum().sum()
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
        
        # スコア計算（欠損率が低いほど高スコア）
        completeness_score = max(0, 100 * (1 - missing_ratio / self.missing_threshold))
        report.completeness_score = completeness_score
        
        # 問題の検出
        if missing_ratio > self.missing_threshold:
            report.add_issue(
                "missing_data",
                "high" if missing_ratio > QUALITY_CONSTANTS['HIGH_MISSING_THRESHOLD'] else "medium",
                f"High missing data ratio: {missing_ratio:.1%}",
                {"missing_cells": missing_cells, "total_cells": total_cells}
            )
        
        # 列ごとの欠損チェック
        for col in df.columns:
            col_missing = df[col].isna().sum()
            col_ratio = col_missing / len(df)
            if col_ratio > self.missing_threshold:
                report.add_issue(
                    "column_missing",
                    "medium",
                    f"Column '{col}' has {col_ratio:.1%} missing values",
                    {"column": col, "missing_count": col_missing}
                )
        
        # 連続欠損のチェック
        for col in df.columns:
            max_consecutive = self._find_max_consecutive_missing(df[col])
            if max_consecutive > QUALITY_CONSTANTS['MAX_CONSECUTIVE_MISSING_DAYS']:
                report.add_issue(
                    "consecutive_missing",
                    "medium",
                    f"Column '{col}' has {max_consecutive} consecutive missing values",
                    {"column": col, "consecutive_missing": max_consecutive}
                )
    
    def _check_consistency(self, df: pd.DataFrame, report: QualityReport):
        """一貫性のチェック"""
        consistency_scores = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # 変動係数による一貫性チェック
                cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                
                if cv > self.consistency_threshold * 3:
                    report.add_issue(
                        "high_variance",
                        "medium",
                        f"Column '{col}' shows high variance (CV={cv:.2f})",
                        {"column": col, "cv": cv}
                    )
                
                # 一貫性スコア（変動が小さいほど高スコア）
                col_score = max(0, 100 * (1 - cv / (self.consistency_threshold * 3)))
                consistency_scores.append(col_score)
                
                # 負の値チェック
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    report.add_issue(
                        "negative_values",
                        "high",
                        f"Column '{col}' contains {negative_count} negative values",
                        {"column": col, "negative_count": negative_count}
                    )
        
        # 平均一貫性スコア
        if consistency_scores:
            report.consistency_score = np.mean(consistency_scores)
        else:
            report.consistency_score = 100.0
    
    def _check_accuracy(self, df: pd.DataFrame, report: QualityReport, reference_data: Any):
        """正確性のチェック"""
        accuracy_scores = []
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # 異常値検出
                anomalies, details = self.check_anomalies(df[col])
                anomaly_ratio = anomalies.sum() / len(df[col])
                
                if anomaly_ratio > QUALITY_CONSTANTS['ANOMALY_RATIO_THRESHOLD']:
                    report.add_issue(
                        "anomalies",
                        "high" if anomaly_ratio > QUALITY_CONSTANTS['HIGH_ANOMALY_RATIO_THRESHOLD'] else "medium",
                        f"Column '{col}' contains {anomaly_ratio:.1%} anomalies",
                        {"column": col, "anomaly_count": anomalies.sum(), **details}
                    )
                
                # 正確性スコア（異常値が少ないほど高スコア）
                col_score = max(0, 100 * (1 - anomaly_ratio / QUALITY_CONSTANTS['HIGH_ANOMALY_RATIO_THRESHOLD']))
                accuracy_scores.append(col_score)
                
                # 参照データとの比較（あれば）
                if reference_data is not None:
                    ref_df = self._convert_to_dataframe(reference_data)
                    if col in ref_df.columns:
                        # 分布の比較
                        ks_stat, p_value = stats.ks_2samp(df[col].dropna(), ref_df[col].dropna())
                        if p_value < QUALITY_CONSTANTS['SIGNIFICANCE_LEVEL']:
                            report.add_issue(
                                "distribution_change",
                                "medium",
                                f"Column '{col}' shows significant distribution change",
                                {"column": col, "ks_statistic": ks_stat, "p_value": p_value}
                            )
        
        # 平均正確性スコア
        if accuracy_scores:
            report.accuracy_score = np.mean(accuracy_scores)
        else:
            report.accuracy_score = 100.0
    
    def _check_timeliness(self, df: pd.DataFrame, report: QualityReport):
        """適時性のチェック"""
        timeliness_score = 100.0
        
        # インデックスが日付型の場合
        if isinstance(df.index, pd.DatetimeIndex):
            # 最新データの確認
            latest_date = df.index.max()
            current_date = pd.Timestamp.now()
            days_delay = (current_date - latest_date).days
            
            if days_delay > QUALITY_CONSTANTS['DATA_DELAY_WARNING_DAYS']:
                report.add_issue(
                    "data_delay",
                    "high" if days_delay > QUALITY_CONSTANTS['DATA_DELAY_CRITICAL_DAYS'] else "medium",
                    f"Data is {days_delay} days old",
                    {"latest_date": latest_date, "days_delay": days_delay}
                )
                timeliness_score = max(0, 100 - days_delay)
            
            # 日付の連続性チェック
            expected_dates = pd.date_range(df.index.min(), df.index.max(), freq='D')
            missing_dates = expected_dates.difference(df.index)
            
            if len(missing_dates) > 0:
                report.add_issue(
                    "missing_dates",
                    "medium",
                    f"{len(missing_dates)} dates are missing",
                    {"missing_count": len(missing_dates), 
                     "sample_missing": list(missing_dates[:5])}
                )
        
        report.timeliness_score = timeliness_score
    
    def _collect_statistics(self, df: pd.DataFrame, report: QualityReport):
        """統計情報の収集"""
        stats_dict = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "memory_usage": df.memory_usage(deep=True).sum() / 1024 / 1024,  # MB
        }
        
        # 数値列の統計
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_dict["numeric_summary"] = {
                "mean": df[numeric_cols].mean().to_dict(),
                "std": df[numeric_cols].std().to_dict(),
                "min": df[numeric_cols].min().to_dict(),
                "max": df[numeric_cols].max().to_dict(),
            }
        
        report.statistics = stats_dict
    
    def _calculate_overall_score(self, report: QualityReport):
        """総合スコアの計算"""
        weights = SCORE_WEIGHTS['quality_scores']
        
        overall = (
            weights["completeness"] * report.completeness_score +
            weights["consistency"] * report.consistency_score +
            weights["accuracy"] * report.accuracy_score +
            weights["timeliness"] * report.timeliness_score
        )
        
        report.overall_score = overall
    
    def _generate_recommendations(self, report: QualityReport):
        """推奨事項の生成"""
        # スコアに基づく推奨
        if report.completeness_score < QUALITY_CONSTANTS['MIN_ACCEPTABLE_SCORE']:
            report.recommendations.append(
                "Consider implementing data imputation strategies for missing values"
            )
        
        if report.consistency_score < QUALITY_CONSTANTS['MIN_ACCEPTABLE_SCORE']:
            report.recommendations.append(
                "Review data collection processes to improve consistency"
            )
        
        if report.accuracy_score < QUALITY_CONSTANTS['MIN_ACCEPTABLE_SCORE']:
            report.recommendations.append(
                "Implement anomaly detection and data validation at collection time"
            )
        
        # 問題に基づく推奨
        for issue in report.issues:
            if issue["type"] == "missing_data" and issue["severity"] in ["high", "critical"]:
                report.recommendations.append(
                    "Urgent: Address high missing data ratio before analysis"
                )
                break
        
        # 異常値が多い場合
        anomaly_issues = [i for i in report.issues if i["type"] == "anomalies"]
        if len(anomaly_issues) > 2:
            report.recommendations.append(
                "Multiple columns show anomalies - consider data preprocessing"
            )
    
    def _find_max_consecutive_missing(self, series: pd.Series) -> int:
        """最大連続欠損数を検出"""
        is_missing = series.isna().astype(int)
        if len(is_missing) == 0:
            return 0
        
        # 連続カウント
        groups = (is_missing != is_missing.shift()).cumsum()
        consecutive = is_missing.groupby(groups).sum()
        
        return consecutive.max() if len(consecutive) > 0 else 0
    
    def _detect_anomalies_zscore(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Z-scoreによる異常値検出"""
        mean = np.nanmean(data)
        std = np.nanstd(data)
        
        if std == 0:
            return np.zeros(len(data), dtype=bool), {"method": "zscore", "mean": mean, "std": 0}
        
        z_scores = np.abs((data - mean) / std)
        anomalies = z_scores > self.outlier_threshold
        
        details = {
            "method": "zscore",
            "threshold": self.outlier_threshold,
            "mean": mean,
            "std": std,
            "max_zscore": np.nanmax(z_scores)
        }
        
        return anomalies, details
    
    def _detect_anomalies_iqr(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """IQRによる異常値検出"""
        q1 = np.nanpercentile(data, 25)
        q3 = np.nanpercentile(data, 75)
        iqr = q3 - q1
        
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        anomalies = (data < lower_bound) | (data > upper_bound)
        
        details = {
            "method": "iqr",
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound
        }
        
        return anomalies, details
    
    def _detect_anomalies_seasonal(self, data: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """季節性を考慮した異常値検出（簡易版）"""
        # 週次パターンを仮定（7日周期）
        if len(data) < 14:
            return self._detect_anomalies_zscore(data)
        
        anomalies = np.zeros(len(data), dtype=bool)
        weekly_means = []
        weekly_stds = []
        
        # 曜日別の統計
        for day in range(7):
            day_data = data[day::7]
            day_data = day_data[~np.isnan(day_data)]
            if len(day_data) > 0:
                weekly_means.append(np.mean(day_data))
                weekly_stds.append(np.std(day_data))
            else:
                weekly_means.append(np.nan)
                weekly_stds.append(np.nan)
        
        # 各データポイントを曜日別に評価
        for i, value in enumerate(data):
            if not np.isnan(value):
                day_of_week = i % 7
                mean = weekly_means[day_of_week]
                std = weekly_stds[day_of_week]
                
                if not np.isnan(mean) and not np.isnan(std) and std > 0:
                    z_score = abs((value - mean) / std)
                    if z_score > self.outlier_threshold:
                        anomalies[i] = True
        
        details = {
            "method": "seasonal",
            "pattern": "weekly",
            "weekly_means": weekly_means,
            "weekly_stds": weekly_stds
        }
        
        return anomalies, details
    
    def _check_trend(self, data: np.ndarray) -> Dict[str, Any]:
        """トレンドのチェック"""
        if len(data) < 3:
            return {"has_trend": False, "direction": "none"}
        
        # 単純な線形回帰
        x = np.arange(len(data))
        valid_mask = ~np.isnan(data)
        
        if valid_mask.sum() < 3:
            return {"has_trend": False, "direction": "none"}
        
        slope, intercept = np.polyfit(x[valid_mask], data[valid_mask], 1)
        
        # トレンドの有意性（簡易判定）
        data_range = np.nanmax(data) - np.nanmin(data)
        trend_range = abs(slope) * len(data)
        
        has_trend = trend_range > STATISTICS_CONSTANTS['TREND_THRESHOLD'] * data_range
        direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
        
        return {
            "has_trend": has_trend,
            "direction": direction,
            "slope": slope,
            "strength": trend_range / data_range if data_range > 0 else 0
        }
    
    def _check_seasonality(self, data: np.ndarray) -> Dict[str, Any]:
        """季節性のチェック（簡易版）"""
        if len(data) < 14:  # 最低2週間必要
            return {"has_seasonality": False, "period": None}
        
        # 週次パターンのチェック
        weekly_pattern = []
        for day in range(7):
            day_data = data[day::7]
            day_data = day_data[~np.isnan(day_data)]
            if len(day_data) > 0:
                weekly_pattern.append(np.mean(day_data))
            else:
                weekly_pattern.append(np.nan)
        
        # パターンの変動を評価
        pattern_std = np.nanstd(weekly_pattern)
        overall_mean = np.nanmean(data)
        
        has_pattern = pattern_std > STATISTICS_CONSTANTS['PATTERN_THRESHOLD'] * overall_mean if overall_mean > 0 else False
        
        return {
            "has_seasonality": has_pattern,
            "period": "weekly" if has_pattern else None,
            "pattern": weekly_pattern,
            "pattern_strength": pattern_std / overall_mean if overall_mean > 0 else 0
        }
    
    def _check_weekly_pattern(self, data: np.ndarray) -> Dict[str, Any]:
        """週次パターンの詳細チェック"""
        return self._check_seasonality(data)
    
    def _check_sudden_changes(self, data: np.ndarray) -> Dict[str, Any]:
        """急激な変化の検出"""
        if len(data) < 2:
            return {"has_sudden_changes": False, "change_points": []}
        
        # 前日比の計算
        diff = np.diff(data)
        valid_diff = diff[~np.isnan(diff)]
        
        if len(valid_diff) == 0:
            return {"has_sudden_changes": False, "change_points": []}
        
        # 変化の標準偏差
        diff_std = np.std(valid_diff)
        diff_mean = np.mean(np.abs(valid_diff))
        
        # 大きな変化の検出（3σ基準）
        change_points = []
        for i, change in enumerate(diff):
            if not np.isnan(change) and abs(change) > 3 * diff_std:
                change_points.append({
                    "index": i + 1,
                    "change": change,
                    "magnitude": abs(change) / diff_mean if diff_mean > 0 else 0
                })
        
        return {
            "has_sudden_changes": len(change_points) > 0,
            "change_points": change_points,
            "total_changes": len(change_points)
        }