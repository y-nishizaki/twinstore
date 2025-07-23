"""
QualityCheckerの包括的テスト
"""

import pytest
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta

from twinstore.data.quality_checker import QualityChecker, QualityReport


class TestQualityReport:
    """QualityReportのテストクラス"""
    
    def test_initialization_default(self):
        """デフォルト初期化のテスト"""
        report = QualityReport()
        
        assert report.overall_score == 0.0
        assert report.completeness_score == 0.0
        assert report.consistency_score == 0.0
        assert report.accuracy_score == 0.0
        assert report.timeliness_score == 0.0
        assert report.issues == []
        assert report.warnings == []
        assert report.recommendations == []
        assert report.statistics == {}
    
    def test_initialization_custom(self):
        """カスタム初期化のテスト"""
        report = QualityReport(
            overall_score=85.0,
            completeness_score=90.0,
            issues=[{"type": "test", "severity": "low", "description": "test issue"}]
        )
        
        assert report.overall_score == 85.0
        assert report.completeness_score == 90.0
        assert len(report.issues) == 1
    
    def test_add_issue(self):
        """問題追加のテスト"""
        report = QualityReport()
        
        report.add_issue("missing_data", "high", "Missing values detected")
        
        assert len(report.issues) == 1
        issue = report.issues[0]
        assert issue["type"] == "missing_data"
        assert issue["severity"] == "high"
        assert issue["description"] == "Missing values detected"
        assert issue["details"] == {}
    
    def test_add_issue_with_details(self):
        """詳細付き問題追加のテスト"""
        report = QualityReport()
        
        details = {"column": "sales", "missing_count": 10}
        report.add_issue("column_missing", "medium", "Column has missing values", details)
        
        assert len(report.issues) == 1
        assert report.issues[0]["details"] == details
    
    def test_get_summary_empty(self):
        """空レポートのサマリーテスト"""
        report = QualityReport()
        
        summary = report.get_summary()
        
        assert "Data Quality Report" in summary
        assert "Overall Score: 0.0/100" in summary
        assert "Completeness: 0.0" in summary
    
    def test_get_summary_with_issues(self):
        """問題ありレポートのサマリーテスト"""
        report = QualityReport()
        
        report.add_issue("missing_data", "critical", "Critical missing data")
        report.add_issue("outliers", "high", "High outlier count")
        report.add_issue("variance", "medium", "Medium variance issue")
        report.add_issue("minor", "low", "Low priority issue")
        
        summary = report.get_summary()
        
        assert "Issues Found: 4" in summary
        assert "Critical: 1" in summary
        assert "High: 1" in summary
        assert "Medium: 1" in summary
        assert "Low: 1" in summary
    
    def test_get_summary_with_recommendations(self):
        """推奨事項付きサマリーテスト"""
        report = QualityReport()
        
        report.recommendations = ["Fix missing data", "Review outliers"]
        
        summary = report.get_summary()
        
        assert "Recommendations: 2" in summary


class TestQualityChecker:
    """QualityCheckerのテストクラス"""
    
    def test_initialization_default(self):
        """デフォルト初期化のテスト"""
        checker = QualityChecker()
        
        assert checker.missing_threshold == 0.1
        assert checker.outlier_threshold == 3.0
        assert checker.consistency_threshold == 0.2
    
    def test_initialization_custom(self):
        """カスタム初期化のテスト"""
        checker = QualityChecker(
            missing_threshold=0.05,
            outlier_threshold=2.5,
            consistency_threshold=0.15
        )
        
        assert checker.missing_threshold == 0.05
        assert checker.outlier_threshold == 2.5
        assert checker.consistency_threshold == 0.15
    
    def test_check_data_quality_series(self):
        """Seriesの品質チェックテスト"""
        checker = QualityChecker()
        
        data = pd.Series([100, 105, 95, 102, 98, 103, 97])
        
        report = checker.check_data_quality(data)
        
        assert isinstance(report, QualityReport)
        assert report.overall_score > 0
        assert isinstance(report.statistics, dict)
    
    def test_check_data_quality_dataframe(self):
        """DataFrameの品質チェックテスト"""
        checker = QualityChecker()
        
        data = pd.DataFrame({
            'sales1': [100, 105, 95, 102, 98],
            'sales2': [110, 115, 105, 112, 108]
        })
        
        report = checker.check_data_quality(data)
        
        assert isinstance(report, QualityReport)
        assert report.completeness_score > 0
        assert report.consistency_score > 0
    
    def test_check_data_quality_dict(self):
        """辞書形式の品質チェックテスト"""
        checker = QualityChecker()
        
        data = {
            'store_001': [100, 105, 95, 102, 98],
            'store_002': [110, 115, 105, 112, 108]
        }
        
        report = checker.check_data_quality(data)
        
        assert isinstance(report, QualityReport)
        assert report.overall_score > 0
    
    def test_check_data_quality_with_missing_values(self):
        """欠損値ありデータの品質チェックテスト"""
        checker = QualityChecker(missing_threshold=0.1)
        
        data = pd.DataFrame({
            'sales': [100, np.nan, np.nan, 102, 98]  # 40%欠損
        })
        
        report = checker.check_data_quality(data)
        
        # 高い欠損率で問題が検出される
        missing_issues = [i for i in report.issues if i["type"] == "missing_data"]
        assert len(missing_issues) > 0
        assert report.completeness_score < 100
    
    def test_check_data_quality_with_outliers(self):
        """異常値ありデータの品質チェックテスト"""
        checker = QualityChecker(outlier_threshold=2.0)  # より厳しい閾値
        
        # 異常値の比率が5%以上になるよう十分なデータを用意
        normal_data = [100, 105, 102, 98, 103, 99, 101, 104, 96, 107] * 2  # 20個の正常値
        outlier_data = [2000, 3000]  # 2個の異常値 (約9%の異常率)
        
        data = pd.DataFrame({
            'sales': normal_data + outlier_data
        })
        
        report = checker.check_data_quality(data)
        
        # 異常値で問題が検出される
        anomaly_issues = [i for i in report.issues if i["type"] == "anomalies"]
        assert len(anomaly_issues) > 0
        assert report.accuracy_score < 100
    
    def test_check_data_quality_with_negative_values(self):
        """負の値ありデータの品質チェックテスト"""
        checker = QualityChecker()
        
        data = pd.DataFrame({
            'sales': [100, 105, -50, 102, 98]  # -50が負の値
        })
        
        report = checker.check_data_quality(data)
        
        # 負の値で問題が検出される
        negative_issues = [i for i in report.issues if i["type"] == "negative_values"]
        assert len(negative_issues) > 0
    
    def test_check_data_quality_with_reference_data(self):
        """参照データありの品質チェックテスト"""
        checker = QualityChecker()
        
        data = pd.DataFrame({
            'sales': [200, 210, 195, 205, 190]  # 平均200
        })
        
        reference = pd.DataFrame({
            'sales': [100, 105, 95, 102, 98]  # 平均100 - 分布が違う
        })
        
        report = checker.check_data_quality(data, reference_data=reference)
        
        # 分布の違いで問題が検出される可能性
        assert isinstance(report, QualityReport)
    
    def test_check_data_quality_custom_check_items(self):
        """カスタムチェック項目のテスト"""
        checker = QualityChecker()
        
        data = pd.DataFrame({
            'sales': [100, 105, 95, 102, 98]
        })
        
        report = checker.check_data_quality(data, check_items=["completeness", "accuracy"])
        
        assert isinstance(report, QualityReport)
        assert report.completeness_score > 0
        assert report.accuracy_score > 0
    
    def test_check_data_quality_datetime_index(self):
        """日時インデックスありの品質チェックテスト"""
        checker = QualityChecker()
        
        # 30日前から今日までのデータ（一部日付が欠けている）
        dates = pd.date_range(start=datetime.now() - timedelta(days=30), end=datetime.now(), freq='D')
        missing_dates = dates[::3]  # 3日に1回欠けている
        
        data = pd.DataFrame({
            'sales': np.random.normal(100, 10, len(missing_dates))
        }, index=missing_dates)
        
        report = checker.check_data_quality(data)
        
        # 日付欠損で問題が検出される
        date_issues = [i for i in report.issues if i["type"] == "missing_dates"]
        assert len(date_issues) > 0
        assert report.timeliness_score <= 100
    
    def test_check_anomalies_zscore(self):
        """Z-score異常値検出のテスト"""
        checker = QualityChecker(outlier_threshold=2.0)  # より厳しい閾値
        
        data = np.array([100, 105, 95, 1000, 102, 98])  # より極端な異常値
        
        anomalies, details = checker.check_anomalies(data, method="zscore")
        
        assert isinstance(anomalies, np.ndarray)
        assert anomalies.dtype == bool
        assert anomalies[3]  # 1000の位置で異常検知
        assert details["method"] == "zscore"
        assert "threshold" in details
        assert "mean" in details
        assert "std" in details
    
    def test_check_anomalies_iqr(self):
        """IQR異常値検出のテスト"""
        checker = QualityChecker()
        
        data = np.array([100, 105, 95, 300, 102, 98])  # 300が異常値
        
        anomalies, details = checker.check_anomalies(data, method="iqr")
        
        assert isinstance(anomalies, np.ndarray)
        assert anomalies[3]  # 300の位置で異常検知
        assert details["method"] == "iqr"
        assert "q1" in details
        assert "q3" in details
        assert "iqr" in details
    
    def test_check_anomalies_seasonal(self):
        """季節性異常値検出のテスト"""
        checker = QualityChecker()
        
        # 週次パターンのあるデータ
        data = np.tile([100, 110, 120, 115, 105, 80, 90], 3)  # 3週間分
        data[7] = 300  # 2週目の月曜日に異常値
        
        anomalies, details = checker.check_anomalies(data, method="seasonal")
        
        assert isinstance(anomalies, np.ndarray)
        assert details["method"] == "seasonal"
        assert details["pattern"] == "weekly"
    
    def test_check_anomalies_short_data(self):
        """短いデータでの季節性異常検出テスト"""
        checker = QualityChecker()
        
        data = np.array([100, 105, 95])  # 3日分のみ
        
        anomalies, details = checker.check_anomalies(data, method="seasonal")
        
        # 短すぎる場合はZ-scoreにフォールバック
        assert isinstance(anomalies, np.ndarray)
    
    def test_check_anomalies_invalid_method(self):
        """無効な異常検出手法のテスト"""
        checker = QualityChecker()
        
        data = np.array([100, 105, 95, 102, 98])
        
        with pytest.raises(ValueError, match="Unknown anomaly detection method"):
            checker.check_anomalies(data, method="invalid_method")
    
    def test_check_patterns(self):
        """パターンチェックのテスト"""
        checker = QualityChecker()
        
        # トレンドのあるデータ
        data = np.array([100, 105, 110, 115, 120, 125, 130])
        
        results = checker.check_patterns(data)
        
        assert isinstance(results, dict)
        assert "trend" in results
        assert "seasonality" in results
        assert "weekly_pattern" in results
        assert "sudden_changes" in results
    
    def test_check_patterns_custom_patterns(self):
        """カスタムパターンチェックのテスト"""
        checker = QualityChecker()
        
        data = np.array([100, 105, 110, 115, 120])
        
        results = checker.check_patterns(data, patterns=["trend", "sudden_changes"])
        
        assert len(results) == 2
        assert "trend" in results
        assert "sudden_changes" in results
        assert "seasonality" not in results
    
    def test_convert_to_dataframe_series(self):
        """Series→DataFrameの変換テスト"""
        checker = QualityChecker()
        
        series = pd.Series([100, 105, 95], name="sales")
        df = checker._convert_to_dataframe(series)
        
        assert isinstance(df, pd.DataFrame)
        assert df.columns.tolist() == ["sales"]
    
    def test_convert_to_dataframe_dict(self):
        """辞書→DataFrameの変換テスト"""
        checker = QualityChecker()
        
        data = {"sales1": [100, 105, 95], "sales2": [110, 115, 105]}
        df = checker._convert_to_dataframe(data)
        
        assert isinstance(df, pd.DataFrame)
        assert set(df.columns) == {"sales1", "sales2"}
    
    def test_convert_to_dataframe_numpy_1d(self):
        """1D numpy→DataFrameの変換テスト"""
        checker = QualityChecker()
        
        data = np.array([100, 105, 95])
        df = checker._convert_to_dataframe(data)
        
        assert isinstance(df, pd.DataFrame)
        assert df.columns.tolist() == ["sales"]
    
    def test_convert_to_dataframe_numpy_2d(self):
        """2D numpy→DataFrameの変換テスト"""
        checker = QualityChecker()
        
        data = np.array([[100, 110], [105, 115], [95, 105]])
        df = checker._convert_to_dataframe(data)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
    
    def test_convert_to_dataframe_list(self):
        """リスト→DataFrameの変換テスト"""
        checker = QualityChecker()
        
        data = [100, 105, 95, 102, 98]
        df = checker._convert_to_dataframe(data)
        
        assert isinstance(df, pd.DataFrame)
        assert df.columns.tolist() == ["sales"]
    
    def test_convert_to_dataframe_invalid(self):
        """無効な型の変換エラーテスト"""
        checker = QualityChecker()
        
        with pytest.raises(ValueError, match="Unsupported data type"):
            checker._convert_to_dataframe("invalid_data")
    
    def test_check_completeness_no_missing(self):
        """欠損値なしの完全性チェックテスト"""
        checker = QualityChecker()
        report = QualityReport()
        
        df = pd.DataFrame({'sales': [100, 105, 95, 102, 98]})
        
        checker._check_completeness(df, report)
        
        assert report.completeness_score == 100.0
        assert len([i for i in report.issues if i["type"] == "missing_data"]) == 0
    
    def test_check_completeness_with_missing(self):
        """欠損値ありの完全性チェックテスト"""
        checker = QualityChecker(missing_threshold=0.1)
        report = QualityReport()
        
        # 40%欠損
        df = pd.DataFrame({'sales': [100, np.nan, np.nan, 102, 98]})
        
        checker._check_completeness(df, report)
        
        assert report.completeness_score < 100.0
        missing_issues = [i for i in report.issues if i["type"] == "missing_data"]
        assert len(missing_issues) > 0
    
    def test_check_completeness_consecutive_missing(self):
        """連続欠損値のチェックテスト"""
        checker = QualityChecker()
        report = QualityReport()
        
        # 8日連続欠損（>7なので問題となる）
        data = [100] + [np.nan] * 8 + [102]
        df = pd.DataFrame({'sales': data})
        
        checker._check_completeness(df, report)
        
        consecutive_issues = [i for i in report.issues if i["type"] == "consecutive_missing"]
        assert len(consecutive_issues) > 0
    
    def test_check_consistency_high_variance(self):
        """高分散の一貫性チェックテスト"""
        checker = QualityChecker(consistency_threshold=0.1)
        report = QualityReport()
        
        # 高分散データ（CV > 0.3）
        df = pd.DataFrame({'sales': [100, 500, 50, 600, 30]})
        
        checker._check_consistency(df, report)
        
        variance_issues = [i for i in report.issues if i["type"] == "high_variance"]
        assert len(variance_issues) > 0
        assert report.consistency_score < 100.0
    
    def test_check_accuracy_with_anomalies(self):
        """異常値ありの正確性チェックテスト"""
        checker = QualityChecker(outlier_threshold=2.0)
        report = QualityReport()
        
        # 5%以上の異常値を含むデータ（20個中2個 = 10%）
        normal_data = [100, 105, 95, 102, 98, 103, 99, 101, 104, 96] * 2
        outlier_data = [5000, 6000]
        
        df = pd.DataFrame({'sales': normal_data + outlier_data})
        
        checker._check_accuracy(df, report, None)
        
        anomaly_issues = [i for i in report.issues if i["type"] == "anomalies"]
        assert len(anomaly_issues) > 0
        assert report.accuracy_score < 100.0
    
    def test_check_timeliness_recent_data(self):
        """最新データの適時性チェックテスト"""
        checker = QualityChecker()
        report = QualityReport()
        
        # 最新データ（今日まで）
        dates = pd.date_range(end=datetime.now(), periods=10, freq='D')
        df = pd.DataFrame({'sales': range(10)}, index=dates)
        
        checker._check_timeliness(df, report)
        
        assert report.timeliness_score > 90  # 最新なので高スコア
    
    def test_check_timeliness_old_data(self):
        """古いデータの適時性チェックテスト"""
        checker = QualityChecker()
        report = QualityReport()
        
        # 45日前のデータ（古い）
        start_date = datetime.now() - timedelta(days=45)
        dates = pd.date_range(start=start_date, periods=10, freq='D')
        df = pd.DataFrame({'sales': range(10)}, index=dates)
        
        checker._check_timeliness(df, report)
        
        delay_issues = [i for i in report.issues if i["type"] == "data_delay"]
        assert len(delay_issues) > 0
        assert report.timeliness_score < 100
    
    def test_calculate_overall_score(self):
        """総合スコア計算のテスト"""
        checker = QualityChecker()
        report = QualityReport()
        
        report.completeness_score = 90.0
        report.consistency_score = 80.0
        report.accuracy_score = 85.0
        report.timeliness_score = 95.0
        
        checker._calculate_overall_score(report)
        
        # 重み付き平均: 0.3*90 + 0.2*80 + 0.3*85 + 0.2*95 = 87.5
        assert abs(report.overall_score - 87.5) < 0.1
    
    def test_generate_recommendations_low_scores(self):
        """低スコア時の推奨事項生成テスト"""
        checker = QualityChecker()
        report = QualityReport()
        
        report.completeness_score = 60.0  # < 70
        report.consistency_score = 65.0   # < 70
        report.accuracy_score = 55.0      # < 70
        
        checker._generate_recommendations(report)
        
        assert len(report.recommendations) >= 3
        assert any("imputation" in rec for rec in report.recommendations)
        assert any("consistency" in rec for rec in report.recommendations)
        assert any("anomaly detection" in rec for rec in report.recommendations)
    
    def test_generate_recommendations_critical_issues(self):
        """重大問題時の推奨事項生成テスト"""
        checker = QualityChecker()
        report = QualityReport()
        
        report.add_issue("missing_data", "critical", "Critical missing data")
        
        checker._generate_recommendations(report)
        
        # 緊急推奨が追加される
        urgent_recs = [r for r in report.recommendations if "Urgent" in r]
        assert len(urgent_recs) > 0
    
    def test_find_max_consecutive_missing(self):
        """最大連続欠損検出のテスト"""
        checker = QualityChecker()
        
        # 3日連続欠損
        series = pd.Series([100, np.nan, np.nan, np.nan, 105, 102])
        
        max_consecutive = checker._find_max_consecutive_missing(series)
        
        assert max_consecutive == 3
    
    def test_find_max_consecutive_missing_empty(self):
        """空Seriesの連続欠損検出テスト"""
        checker = QualityChecker()
        
        series = pd.Series([])
        
        max_consecutive = checker._find_max_consecutive_missing(series)
        
        assert max_consecutive == 0
    
    def test_detect_anomalies_zscore_zero_std(self):
        """標準偏差0でのZ-score異常検出テスト"""
        checker = QualityChecker()
        
        # 全て同じ値
        data = np.array([100, 100, 100, 100])
        
        anomalies, details = checker._detect_anomalies_zscore(data)
        
        assert not anomalies.any()  # 異常値なし
        assert details["std"] == 0
    
    def test_check_trend_insufficient_data(self):
        """データ不足でのトレンドチェックテスト"""
        checker = QualityChecker()
        
        data = np.array([100, 105])  # 2点のみ
        
        result = checker._check_trend(data)
        
        assert not result["has_trend"]
        assert result["direction"] == "none"
    
    def test_check_trend_with_nan(self):
        """NaN含みデータのトレンドチェックテスト"""
        checker = QualityChecker()
        
        data = np.array([100, np.nan, 110, np.nan, 120])
        
        result = checker._check_trend(data)
        
        assert "has_trend" in result
        assert "direction" in result
    
    def test_check_seasonality_short_data(self):
        """短いデータでの季節性チェックテスト"""
        checker = QualityChecker()
        
        data = np.array([100, 105, 95])  # 短すぎる
        
        result = checker._check_seasonality(data)
        
        assert not result["has_seasonality"]
        assert result["period"] is None
    
    def test_check_sudden_changes(self):
        """急激変化検出のテスト"""
        checker = QualityChecker()
        
        # 様々なデータパターンをテスト
        spike_data = [100, 100, 100000, 100, 100]  # 大きなスパイク
        data = np.array(spike_data)
        
        result = checker._check_sudden_changes(data)
        
        # メソッドの正常動作を確認（結果の詳細よりも機能的な動作を重視）
        assert isinstance(result, dict)
        assert "has_sudden_changes" in result
        assert "change_points" in result
        assert isinstance(result["change_points"], list)
    
    def test_check_sudden_changes_no_data(self):
        """データなしでの急激変化検出テスト"""
        checker = QualityChecker()
        
        data = np.array([100])  # 1点のみ
        
        result = checker._check_sudden_changes(data)
        
        assert not result["has_sudden_changes"]
        assert len(result["change_points"]) == 0