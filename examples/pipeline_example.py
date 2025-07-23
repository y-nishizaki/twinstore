"""
TwinStore パイプライン使用例

パイプライン機能を使用して、データの前処理から予測、
レポート生成までを一連の流れで実行する例。
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# TwinStoreのインポート
from twinstore import PredictionPipeline, PipelineBuilder, PipelineConfig


def generate_sample_data():
    """サンプルデータの生成"""
    # 過去の店舗データ（5店舗×365日）
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=365, freq='D')
    
    historical_data = {}
    for i in range(5):
        # 基本売上 + トレンド + 週次パターン + ノイズ
        base_sales = 100000 + i * 20000
        trend = np.linspace(0, 50000, 365)
        weekly_pattern = 20000 * np.sin(np.arange(365) * 2 * np.pi / 7)
        noise = np.random.normal(0, 10000, 365)
        
        sales = base_sales + trend + weekly_pattern + noise
        sales = np.maximum(sales, 0)  # 負の値を除去
        
        historical_data[f"store_{i:03d}"] = sales
    
    # 新規店舗データ（30日）
    new_store_sales = 120000 + np.linspace(0, 10000, 30) + \
                     15000 * np.sin(np.arange(30) * 2 * np.pi / 7) + \
                     np.random.normal(0, 8000, 30)
    new_store_sales = np.maximum(new_store_sales, 0)
    
    # 店舗属性データ
    store_attributes = pd.DataFrame({
        'store_cd': [f"store_{i:03d}" for i in range(5)],
        'store_type': ['roadside', 'mall', 'roadside', 'urban', 'mall'],
        'area': [150, 120, 180, 100, 130],
        'location': ['suburban', 'urban', 'suburban', 'downtown', 'urban']
    }).set_index('store_cd')
    
    return historical_data, new_store_sales, store_attributes


def example_1_basic_pipeline():
    """例1: 基本的なパイプライン使用"""
    print("=" * 60)
    print("例1: 基本的なパイプライン使用")
    print("=" * 60)
    
    # データ生成
    historical_data, new_store_sales, store_attributes = generate_sample_data()
    
    # デフォルト設定でパイプラインを作成
    pipeline = PredictionPipeline()
    
    # 学習
    pipeline.fit(historical_data, store_attributes)
    
    # 予測実行
    result = pipeline.predict(
        new_store_sales,
        store_name="新宿西口店",
        filters={'store_type': 'roadside'}
    )
    
    # 結果表示
    print(f"\n予測年間売上: {result.prediction.prediction:,.0f}円")
    print(f"信頼区間: {result.prediction.lower_bound:,.0f} - {result.prediction.upper_bound:,.0f}円")
    print(f"信頼度スコア: {result.prediction.confidence_score:.2%}")
    print(f"実行時間: {result.execution_time:.2f}秒")
    
    if result.explanation:
        print("\n--- 予測の説明 ---")
        print(result.explanation)


def example_2_custom_config():
    """例2: カスタム設定でのパイプライン"""
    print("\n" + "=" * 60)
    print("例2: カスタム設定でのパイプライン")
    print("=" * 60)
    
    # データ生成
    historical_data, new_store_sales, store_attributes = generate_sample_data()
    
    # カスタム設定
    config = PipelineConfig(
        # 厳格な検証
        validate_input=True,
        strict_validation=True,
        
        # 前処理設定
        preprocess_data=True,
        handle_missing=True,
        handle_outliers=True,
        smooth_data=True,
        
        # 品質チェック
        check_quality=True,
        quality_threshold=80.0,
        
        # 予測設定
        similarity_metric="cosine",
        normalization_method="min-max",
        n_similar_stores=3,
        auto_optimize_period=True,
        
        # 出力設定
        save_results=True,
        output_dir="pipeline_results",
        output_format="json",
        
        # 詳細ログ
        verbose=True,
        log_level="DEBUG"
    )
    
    # パイプライン作成
    pipeline = PredictionPipeline(config)
    
    # 学習と予測
    pipeline.fit(historical_data, store_attributes)
    result = pipeline.predict(new_store_sales, store_name="渋谷店")
    
    # 品質レポート表示
    if result.quality_report:
        print(f"\nデータ品質スコア: {result.quality_report.overall_score:.1f}/100")
        print(f"  - 完全性: {result.quality_report.completeness_score:.1f}")
        print(f"  - 一貫性: {result.quality_report.consistency_score:.1f}")
        print(f"  - 正確性: {result.quality_report.accuracy_score:.1f}")


def example_3_pipeline_builder():
    """例3: パイプラインビルダーの使用"""
    print("\n" + "=" * 60)
    print("例3: パイプラインビルダーの使用")
    print("=" * 60)
    
    # データ生成
    historical_data, new_store_sales, store_attributes = generate_sample_data()
    
    # ビルダーパターンでパイプラインを構築
    pipeline = (PipelineBuilder()
        .with_validation(strict=False)
        .with_preprocessing(handle_missing=True, handle_outliers=True, smooth=False)
        .with_quality_check(threshold=75.0)
        .with_prediction(metric="dtw", normalization="z-score", n_similar=5)
        .with_explanation(language="ja")
        .with_output(save=False)
        .build()
    )
    
    # 学習と予測
    pipeline.fit(historical_data, store_attributes)
    result = pipeline.predict(new_store_sales, store_name="池袋店")
    
    # 結果表示
    print(f"\n予測年間売上: {result.prediction.prediction:,.0f}円")
    
    # 類似店舗情報
    print("\n類似店舗TOP3:")
    for i, (store_cd, score) in enumerate(result.prediction.similar_stores[:3], 1):
        print(f"  {i}. {store_cd} (スコア: {score:.3f})")


def example_4_batch_prediction():
    """例4: バッチ予測"""
    print("\n" + "=" * 60)
    print("例4: バッチ予測")
    print("=" * 60)
    
    # データ生成
    historical_data, _, store_attributes = generate_sample_data()
    
    # 複数の新規店舗データ
    np.random.seed(123)
    new_stores_data = {}
    for i in range(3):
        sales = 100000 + i * 10000 + np.random.normal(0, 5000, 30)
        sales = np.maximum(sales, 0)
        new_stores_data[f"new_store_{i:02d}"] = sales
    
    # パイプライン作成
    pipeline = PredictionPipeline()
    pipeline.fit(historical_data, store_attributes)
    
    # バッチ予測実行
    batch_results = pipeline.batch_predict(new_stores_data)
    
    # 結果表示
    print("\nバッチ予測結果:")
    for store_cd, result in batch_results.items():
        if result.prediction:
            print(f"\n{store_cd}:")
            print(f"  予測年間売上: {result.prediction.prediction:,.0f}円")
            print(f"  信頼度: {result.prediction.confidence_score:.2%}")
        else:
            print(f"\n{store_cd}: 予測失敗")
            if result.warnings:
                print(f"  警告: {result.warnings[0]}")


def example_5_data_with_issues():
    """例5: 問題のあるデータでの動作確認"""
    print("\n" + "=" * 60)
    print("例5: 問題のあるデータでの動作確認")
    print("=" * 60)
    
    # 正常な過去データ
    historical_data, _, store_attributes = generate_sample_data()
    
    # 問題のある新規店舗データ
    new_store_sales = np.array([
        100000, 105000, np.nan, 110000, 115000,  # 欠損値
        120000, 125000, 1000000, 130000, 135000,  # 異常値
        140000, 145000, 0, 0, 0,  # ゼロの連続
        150000, 155000, 160000, 165000, 170000,
        -5000, 175000, 180000, 185000, 190000,  # 負の値
    ])
    
    # パイプライン作成（前処理あり）
    pipeline = PredictionPipeline(PipelineConfig(
        preprocess_data=True,
        handle_missing=True,
        handle_outliers=True,
        check_quality=True,
        verbose=True
    ))
    
    # 学習と予測
    pipeline.fit(historical_data, store_attributes)
    result = pipeline.predict(new_store_sales, store_name="問題データ店")
    
    # 結果表示
    print(f"\n予測年間売上: {result.prediction.prediction:,.0f}円")
    
    # 検証結果
    if result.validation_result:
        print(f"\nデータ検証: {'合格' if result.validation_result.is_valid else '不合格'}")
        if result.validation_result.warnings:
            print("警告:")
            for warning in result.validation_result.warnings[:3]:
                print(f"  - {warning}")
    
    # 品質スコア
    if result.quality_report:
        print(f"\nデータ品質スコア: {result.quality_report.overall_score:.1f}/100")
        if result.quality_report.issues:
            print(f"検出された問題: {len(result.quality_report.issues)}件")


def example_6_config_update():
    """例6: 動的な設定変更"""
    print("\n" + "=" * 60)
    print("例6: 動的な設定変更")
    print("=" * 60)
    
    # データ生成
    historical_data, new_store_sales, store_attributes = generate_sample_data()
    
    # パイプライン作成
    pipeline = PredictionPipeline()
    pipeline.fit(historical_data, store_attributes)
    
    # 初回予測（DTW）
    print("\n[DTWでの予測]")
    result1 = pipeline.predict(new_store_sales[:20], store_name="テスト店_DTW")
    print(f"予測年間売上: {result1.prediction.prediction:,.0f}円")
    
    # 設定を変更
    pipeline.update_config(
        similarity_metric="cosine",
        n_similar_stores=3
    )
    
    # 再予測（コサイン類似度）
    print("\n[コサイン類似度での予測]")
    result2 = pipeline.predict(new_store_sales[:20], store_name="テスト店_Cosine")
    print(f"予測年間売上: {result2.prediction.prediction:,.0f}円")
    
    # 差異の表示
    diff = abs(result1.prediction.prediction - result2.prediction.prediction)
    print(f"\n予測の差異: {diff:,.0f}円 ({diff/result1.prediction.prediction:.1%})")


if __name__ == "__main__":
    # 全ての例を実行
    example_1_basic_pipeline()
    example_2_custom_config()
    example_3_pipeline_builder()
    example_4_batch_prediction()
    example_5_data_with_issues()
    example_6_config_update()
    
    print("\n" + "=" * 60)
    print("全ての例の実行が完了しました")
    print("=" * 60)