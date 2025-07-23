"""
数値定数とマジックナンバーの定義
"""

# データ品質関連の定数
QUALITY_CONSTANTS = {
    'MISSING_THRESHOLD': 0.1,          # 欠損値の許容割合（DataValidator, QualityChecker）
    'HIGH_MISSING_THRESHOLD': 0.3,     # 高欠損率の閾値（DataPreprocessor）
    'OUTLIER_THRESHOLD': 3.0,          # 異常値判定のZ-score閾値（QualityChecker）
    'CONSISTENCY_THRESHOLD': 0.2,      # 一貫性チェックの変動係数閾値（QualityChecker）
    'ANOMALY_RATIO_THRESHOLD': 0.05,   # 異常値比率の閾値（5%）
    'HIGH_ANOMALY_RATIO_THRESHOLD': 0.1,  # 高異常値比率の閾値（10%）
    'MIN_QUALITY_SCORE': 60.0,         # 最小品質スコア（PipelineConfig）
    'MIN_ACCEPTABLE_SCORE': 70.0,      # 最小許容スコア（QualityChecker）
    'SIGNIFICANCE_LEVEL': 0.05,        # 統計的有意水準
    'DATA_DELAY_WARNING_DAYS': 7,      # データ遅延警告日数
    'DATA_DELAY_CRITICAL_DAYS': 30,    # データ遅延重大日数
    'MAX_CONSECUTIVE_MISSING_DAYS': 7, # 最大連続欠損日数
}

# 時系列データ関連の定数
TIME_SERIES_CONSTANTS = {
    'MIN_DAYS': 3,                     # 最小必要日数（DataValidator）
    'MIN_DAYS_PER_STORE': 30,          # 店舗あたりの最小日数（DataValidator）
    'RECOMMENDED_DAYS': 7,             # 推奨日数
    'MIN_HISTORY': 7,                  # リアルタイム検知の最小履歴（AnomalyDetector）
    'RECENT_DAYS_WINDOW': 30,          # 直近データのウィンドウサイズ
    'QUARTER_DAYS': 90,                # 四半期日数
    'HALF_YEAR_DAYS': 180,             # 半年日数
    'DAYS_IN_YEAR': 365,               # 年間日数
    'CONSECUTIVE_MISSING_THRESHOLD': 7, # 連続欠損の閾値日数
}

# 統計関連の定数
STATISTICS_CONSTANTS = {
    'ZSCORE_CRITICAL': 4.0,            # 重大異常のZ-score（AnomalyDetector）
    'ZSCORE_WARNING': 3.0,             # 警告レベルのZ-score
    'ZSCORE_HIGH_SENSITIVITY': 2.0,   # 高感度のZ-score
    'ZSCORE_MEDIUM_SENSITIVITY': 2.5, # 中感度のZ-score
    'IQR_MULTIPLIER': 1.5,             # IQR法の係数
    'CONFIDENCE_MULTIPLIER': 1.96,     # 信頼区間の係数（95%）
    'PERCENTILE_LOW': 5,               # 下位パーセンタイル
    'PERCENTILE_HIGH': 95,             # 上位パーセンタイル
    'QUARTILE_FIRST': 25,              # 第1四分位数
    'QUARTILE_THIRD': 75,              # 第3四分位数
    'TREND_THRESHOLD': 0.1,            # トレンド判定の閾値（10%）
    'PATTERN_THRESHOLD': 0.1,          # パターン判定の閾値（10%）
}

# 予測モデル関連の定数
PREDICTION_CONSTANTS = {
    'DEFAULT_N_SIMILAR': 5,            # デフォルトの類似店舗数（SalesPredictor）
    'DEFAULT_CONFIDENCE_LEVEL': 0.95,  # デフォルトの信頼水準
    'MIN_SIMILAR_STORES': 2,           # 最小類似店舗数
    'PERIOD_OPTIMIZATION_RANGE': (7, 90),  # 期間最適化の範囲
    'PERIOD_STEP_RATIO': 0.1,          # 期間ステップの比率
    'STABILITY_THRESHOLD': 0.05,       # 安定性判定の閾値
}

# スコア計算の重み
SCORE_WEIGHTS = {
    'quality_scores': {
        'completeness': 0.3,               # 完全性の重み（QualityChecker）
        'consistency': 0.2,                # 一貫性の重み
        'accuracy': 0.3,                   # 正確性の重み
        'timeliness': 0.2,                # 適時性の重み
    },
    'COMPLETENESS': 0.3,               # 完全性の重み（後方互換性のため維持）
    'CONSISTENCY': 0.2,                # 一貫性の重み
    'ACCURACY': 0.3,                   # 正確性の重み
    'TIMELINESS': 0.2,                # 適時性の重み
}

# パフォーマンス関連の定数
PERFORMANCE_CONSTANTS = {
    'MAX_STORES_BATCH': 1000,          # バッチ処理の最大店舗数
    'DEFAULT_BATCH_SIZE': 100,         # デフォルトのバッチサイズ
    'CACHE_EXPIRY_SECONDS': 3600,      # キャッシュ有効期限（1時間）
    'MAX_MEMORY_MB': 1024,             # 最大メモリ使用量（MB）
}

# 可視化関連の定数（visualization.pyへ移動予定）
PLOT_CONSTANTS = {
    'MAX_POINTS_DISPLAY': 1000,        # 表示する最大データポイント数
    'DEFAULT_LINE_WIDTH': 2,           # デフォルトの線幅
    'FIGURE_WIDTH': 12,                # 図の幅
    'FIGURE_HEIGHT': 6,                # 図の高さ
}

# エラーメッセージテンプレート
ERROR_MESSAGES = {
    'INSUFFICIENT_DATA': "Insufficient data: minimum {min_days} days required",
    'INVALID_DATA_TYPE': "Invalid data type: {data_type}. Expected {expected_type}",
    'HIGH_MISSING_RATIO': "High missing value ratio: {ratio:.1%}",
    'NO_VALID_STORES': "No valid stores found after filtering",
    'OPTIMIZATION_FAILED': "Period optimization failed: {reason}",
}