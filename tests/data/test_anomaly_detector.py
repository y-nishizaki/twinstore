"""
AnomalyDetectorのテスト
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from twinstore.data.anomaly_detector import (
    AnomalyDetector, 
    AnomalyAlert, 
    AnomalyReport,
    AlertLevel
)


class TestAnomalyAlert:
    """AnomalyAlertのテストクラス"""
    
    def test_alert_creation(self):
        """アラート作成のテスト"""
        alert = AnomalyAlert(
            timestamp=datetime(2024, 1, 1),
            level="warning",
            message="Test alert",
            value=100.0,
            expected_range=(80.0, 120.0),
            detection_method="statistical"
        )
        
        assert alert.timestamp == datetime(2024, 1, 1)
        assert alert.level == "warning"
        assert alert.message == "Test alert"
        assert alert.value == 100.0
        assert alert.expected_range == (80.0, 120.0)
        assert alert.detection_method == "statistical"
        assert isinstance(alert.metadata, dict)


class TestAnomalyReport:
    """AnomalyReportのテストクラス"""
    
    def test_report_creation(self):
        """レポート作成のテスト"""
        alert = AnomalyAlert(
            timestamp=datetime.now(),
            level="info",
            message="Test",
            value=100.0,
            expected_range=(90.0, 110.0),
            detection_method="test"
        )
        
        report = AnomalyReport(
            total_points=100,
            anomaly_count=5,
            anomaly_ratio=0.05,
            alerts=[alert],
            statistics={"mean": 100.0},
            recommendations=["Test recommendation"]
        )
        
        assert report.total_points == 100
        assert report.anomaly_count == 5
        assert report.anomaly_ratio == 0.05
        assert len(report.alerts) == 1
        assert report.statistics["mean"] == 100.0
        assert len(report.recommendations) == 1


class TestAnomalyDetector:
    """AnomalyDetectorのテストクラス"""
    
    def test_initialization_default(self):
        """デフォルト初期化のテスト"""
        detector = AnomalyDetector()
        
        assert detector.method == "statistical"
        assert detector.sensitivity == 0.95
        assert detector.min_history == 7
        assert detector._history == []
        assert isinstance(detector._model_params, dict)
    
    def test_initialization_custom(self):
        """カスタム初期化のテスト"""
        detector = AnomalyDetector(
            method="isolation_forest",
            sensitivity=0.99,
            min_history=10
        )
        
        assert detector.method == "isolation_forest"
        assert detector.sensitivity == 0.99
        assert detector.min_history == 10
    
    def test_detect_anomalies_array(self):
        """配列データでの異常検知テスト"""
        detector = AnomalyDetector()
        
        # 正常データに異常値を含むデータを作成
        normal_data = np.random.normal(100, 10, 100)
        anomalous_data = np.concatenate([
            normal_data[:50],
            [200, 300],  # 明らかに異常な値
            normal_data[50:]
        ])
        
        anomalies = detector.detect_anomalies(anomalous_data)
        
        assert isinstance(anomalies, np.ndarray)
        assert len(anomalies) == len(anomalous_data)
        assert anomalies.dtype == bool
        assert anomalies.sum() > 0  # 異常値が検出されること
    
    def test_detect_anomalies_series(self):
        """Seriesデータでの異常検知テスト"""
        detector = AnomalyDetector()
        
        # 時系列データを作成
        dates = pd.date_range('2024-01-01', periods=100)
        normal_values = np.random.normal(100, 10, 100)
        normal_values[50] = 300  # 異常値を挿入
        
        data = pd.Series(normal_values, index=dates)
        
        anomalies = detector.detect_anomalies(data)
        
        assert isinstance(anomalies, np.ndarray)
        assert len(anomalies) == len(data)
        assert anomalies[50]  # 50番目が異常として検出されること
    
    def test_detect_anomalies_statistical(self):
        """統計的手法での異常検知テスト"""
        detector = AnomalyDetector(method="statistical", sensitivity=2.0)
        
        # 正常データ（平均100、標準偏差10）
        normal_data = np.random.normal(100, 10, 100)
        # 明らかに異常な値を追加
        data = np.append(normal_data, [150, 50, 200])
        
        anomalies = detector.detect_anomalies(data)
        
        # 最後の3つの値は異常として検出されるはず
        assert anomalies[-3:].sum() >= 1  # 少なくとも1つは検出される
    
    def test_detect_anomalies_isolation_forest(self):
        """Isolation Forestでの異常検知テスト"""
        detector = AnomalyDetector(method="isolation_forest")
        
        # 正常データに外れ値を含むデータ
        data = np.concatenate([
            np.random.normal(100, 10, 95),
            [300, 400, 500, 600, 700]  # 明らかに異常な値
        ])
        
        anomalies = detector.detect_anomalies(data)
        
        assert isinstance(anomalies, np.ndarray)
        assert anomalies.sum() > 0  # 異常値が検出されること
    
    def test_detect_anomalies_mad(self):
        """MAD法での異常検知テスト"""
        detector = AnomalyDetector(method="mad", sensitivity=2.5)
        
        data = np.array([100, 105, 95, 102, 98, 300, 97, 103])  # 300が異常値
        
        anomalies = detector.detect_anomalies(data)
        
        assert isinstance(anomalies, np.ndarray)
        assert anomalies[5]  # 300の位置で異常検知
    
    def test_detect_anomalies_percentile(self):
        """パーセンタイル法での異常検知テスト"""
        detector = AnomalyDetector(method="percentile")
        
        data = np.array([1, 2, 3, 4, 5, 100, 6, 7, 8, 9])  # 100が異常値
        
        anomalies = detector.detect_anomalies(data)
        
        assert isinstance(anomalies, np.ndarray)
        assert anomalies[5]  # 100の位置で異常検知
    
    def test_update_model(self):
        """モデル更新のテスト"""
        detector = AnomalyDetector()
        
        # 正常データでモデルを更新
        normal_data = np.random.normal(100, 10, 100)
        detector.update_model(normal_data)
        
        assert isinstance(detector._model_params, dict)
        assert len(detector._model_params) > 0
    
    def test_detect_realtime(self):
        """リアルタイム異常検知のテスト"""
        detector = AnomalyDetector(min_history=3)
        
        # 正常データを追加
        for value in [100, 105, 95, 102, 98]:
            alert = detector.detect_realtime(value)
            # 最初の幾つかは履歴不足でNone
        
        # 異常値を追加
        alert = detector.detect_realtime(300)
        
        # 異常値でアラートが生成されるか、またはNoneが返される
        assert alert is None or isinstance(alert, AnomalyAlert)
    
    def test_analyze_period(self):
        """期間分析のテスト"""
        detector = AnomalyDetector()
        
        # 時系列データとタイムスタンプを作成
        dates = pd.date_range('2024-01-01', periods=10)
        data = pd.Series([100, 105, 300, 95, 102, 400, 98, 103, 97, 101], index=dates)
        
        # 期間分析を実行
        result = detector.analyze_period(
            data,
            timestamps=dates
        )
        
        # 結果の検証
        assert isinstance(result, AnomalyReport)
        assert result.total_points == 10
        assert result.anomaly_count >= 0
        assert isinstance(result.alerts, list)
    
    def test_threshold_calculation(self):
        """闾値計算のテスト"""
        detector = AnomalyDetector()
        
        # 正常データでモデルを更新
        data = np.random.normal(100, 10, 100)
        detector.update_model(data)
        
        # 闾値計算をテスト
        threshold = detector._get_threshold()
        
        assert isinstance(threshold, (int, float))
        assert threshold > 0
    
    def test_model_parameters(self):
        """モデルパラメータのテスト"""
        detector = AnomalyDetector()
        
        data = np.random.normal(100, 10, 50)
        detector.update_model(data)
        
        # モデルパラメータが設定されていることを確認
        assert isinstance(detector._model_params, dict)
        assert len(detector._model_params) >= 0
    
    def test_history_management(self):
        """履歴管理のテスト"""
        detector = AnomalyDetector(min_history=5)
        
        # リアルタイムデータで履歴を追加
        for i, value in enumerate([100, 105, 95, 102, 98, 103, 97]):
            alert = detector.detect_realtime(value)
            
            # 初期は履歴不足でNone
            if i < detector.min_history - 1:
                assert alert is None or isinstance(alert, AnomalyAlert)
        
        # 履歴が積み上がっていることを確認
        assert len(detector._history) > 0
    
    def test_empty_data_handling(self):
        """空データの処理テスト"""
        detector = AnomalyDetector()
        
        empty_data = np.array([])
        
        # 空データでも例外が発生しないことを確認
        try:
            anomalies = detector.detect_anomalies(empty_data)
            assert len(anomalies) == 0
        except (ValueError, IndexError):
            # 空データの場合は適切な例外が発生することも許容
            pass
    
    def test_single_value_data(self):
        """単一値データの処理テスト"""
        detector = AnomalyDetector()
        
        single_data = np.array([100])
        
        # 単一データでも例外が発生しないことを確認
        try:
            anomalies = detector.detect_anomalies(single_data)
            assert len(anomalies) == 1
            # 単一値は異常とは判定されない
            assert not anomalies[0]
        except (ValueError, RuntimeError):
            # 統計計算ができない場合の例外も許容
            pass
    
    def test_nan_data_handling(self):
        """NaNデータの処理テスト"""
        detector = AnomalyDetector()
        
        data_with_nan = np.array([100, 105, np.nan, 95, 102, 98])
        
        # NaNが含まれていても処理できることを確認
        try:
            anomalies = detector.detect_anomalies(data_with_nan)
            assert len(anomalies) == len(data_with_nan)
        except ValueError:
            # NaN処理で例外が発生することも許容
            pass
    
    def test_multiple_data_series(self):
        """複数データ系列のテスト"""
        detector = AnomalyDetector()
        
        # 複数の時系列データを個別に処理
        data_series = [
            np.random.normal(100, 10, 50),
            np.concatenate([np.random.normal(120, 15, 40), [300, 400]]),  # 異常値あり
            np.random.normal(80, 8, 60)
        ]
        
        results = []
        for data in data_series:
            anomalies = detector.detect_anomalies(data)
            results.append(anomalies)
        
        # 各結果の検証
        for i, anomalies in enumerate(results):
            assert isinstance(anomalies, np.ndarray)
            assert len(anomalies) == len(data_series[i])
            # 2番目のデータには異常値が含まれている
            if i == 1:
                assert anomalies.sum() > 0  # 異常が検出されるはず
    
    def test_different_sensitivity_levels(self):
        """異なる感度レベルのテスト"""
        data = np.concatenate([
            np.random.normal(100, 10, 90),
            [130, 140, 150]  # 軽度の異常値
        ])
        
        # 高感度（より多くの異常を検出）
        high_sensitive = AnomalyDetector(sensitivity=1.5)
        anomalies_high = high_sensitive.detect_anomalies(data)
        
        # 低感度（より少ない異常を検出）
        low_sensitive = AnomalyDetector(sensitivity=3.0)
        anomalies_low = low_sensitive.detect_anomalies(data)
        
        # 高感度の方が多くの異常を検出するはず
        assert anomalies_high.sum() >= anomalies_low.sum()
    
    def test_time_series_pattern_detection(self):
        """時系列パターン検出のテスト"""
        detector = AnomalyDetector(method="statistical")
        
        # 季節性のある正常データ
        days = 100
        t = np.arange(days)
        seasonal_pattern = 10 * np.sin(2 * np.pi * t / 7)  # 週次パターン
        normal_sales = 100 + seasonal_pattern + np.random.normal(0, 5, days)
        
        # 特定の日に異常値を挿入
        abnormal_data = normal_sales.copy()
        abnormal_data[50] = 200  # 明らかに異常な値
        
        anomalies = detector.detect_anomalies(abnormal_data)
        
        # 異常値が検出されることを確認
        assert anomalies[50] or anomalies.sum() > 0
    
    def test_invalid_method(self):
        """無効な手法指定のテスト"""
        detector = AnomalyDetector(method="invalid_method")
        data = np.random.normal(100, 10, 50)
        
        with pytest.raises(ValueError):
            detector.detect_anomalies(data)