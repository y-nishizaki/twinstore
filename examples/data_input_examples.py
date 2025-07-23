"""
TwinStore データ入力の実例集

様々な形式でのデータ入力方法を示すサンプルコード
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
from pathlib import Path

# TwinStoreのインポート
from twinstore import PredictionPipeline, DataValidator


def example_1_basic_numpy():
    """例1: NumPy配列での基本的な入力"""
    print("=" * 60)
    print("例1: NumPy配列での基本的な入力")
    print("=" * 60)
    
    # 過去データ：5店舗×180日
    n_stores = 5
    n_days = 180
    
    # ランダムシードで再現性を確保
    np.random.seed(42)
    
    # 辞書形式で店舗データを作成
    historical_data = {}
    for i in range(n_stores):
        # 基本売上 + トレンド + ノイズ
        base_sales = 100000 + i * 10000
        trend = np.linspace(0, 10000, n_days)
        noise = np.random.normal(0, 5000, n_days)
        
        sales = base_sales + trend + noise
        sales = np.maximum(sales, 0)  # 負の値を0に
        
        historical_data[f'store_{i:03d}'] = sales
    
    # 新規店舗データ：30日分
    new_store_sales = np.array([
        95000, 98000, 102000, 99000, 103000, 105000, 108000,
        96000, 101000, 104000, 107000, 103000, 106000, 109000,
        98000, 102000, 105000, 108000, 104000, 107000, 110000,
        99000, 103000, 106000, 109000, 105000, 108000, 111000,
        100000, 104000
    ])
    
    # パイプラインで予測
    pipeline = PredictionPipeline()
    pipeline.fit(historical_data)
    result = pipeline.predict(new_store_sales)
    
    print(f"\n予測年間売上: {result.prediction.prediction:,.0f}円")
    print(f"データ形式: NumPy配列")
    print(f"学習データ: {n_stores}店舗 × {n_days}日")
    print(f"予測データ: {len(new_store_sales)}日")


def example_2_pandas_dataframe():
    """例2: pandas DataFrameでの入力"""
    print("\n" + "=" * 60)
    print("例2: pandas DataFrameでの入力")
    print("=" * 60)
    
    # 日付インデックスを作成
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start_date, periods=365, freq='D')
    
    # 店舗データを生成（曜日効果を含む）
    store_data = {}
    for i in range(3):
        base = 120000 + i * 15000
        
        # 曜日効果（週末は売上増）
        weekday_effect = np.array([
            1.0 if d.weekday() < 5 else 1.3 
            for d in dates
        ])
        
        # 月次トレンド
        monthly_trend = np.sin(np.arange(365) * 2 * np.pi / 30) * 10000
        
        # ランダムノイズ
        noise = np.random.normal(0, 8000, 365)
        
        # 売上計算
        sales = base * weekday_effect + monthly_trend + noise
        store_data[f'店舗{i+1:02d}'] = np.maximum(sales, 0)
    
    # DataFrameに変換
    historical_data = pd.DataFrame(store_data, index=dates)
    
    # 新規店舗データ（Seriesとして）
    new_dates = pd.date_range(datetime(2024, 1, 1), periods=21, freq='D')
    new_store_sales = pd.Series(
        data=np.random.normal(130000, 10000, 21),
        index=new_dates,
        name='新規店舗'
    )
    new_store_sales = new_store_sales.clip(lower=0)
    
    # データ情報を表示
    print("\n学習データ情報:")
    print(historical_data.info())
    print(f"\n売上統計:\n{historical_data.describe()}")
    
    # 予測実行
    pipeline = PredictionPipeline()
    pipeline.fit(historical_data)
    result = pipeline.predict(new_store_sales)
    
    print(f"\n予測年間売上: {result.prediction.prediction:,.0f}円")


def example_3_csv_input():
    """例3: CSVファイルからの入力"""
    print("\n" + "=" * 60)
    print("例3: CSVファイルからの入力")
    print("=" * 60)
    
    # サンプルCSVファイルを作成
    csv_path = Path("sample_sales_data.csv")
    
    # データ生成
    dates = pd.date_range('2023-01-01', periods=90, freq='D')
    data = {
        'date': dates,
        'store_A': np.random.normal(100000, 10000, 90),
        'store_B': np.random.normal(120000, 12000, 90),
        'store_C': np.random.normal(90000, 8000, 90),
    }
    
    df = pd.DataFrame(data)
    df[['store_A', 'store_B', 'store_C']] = df[['store_A', 'store_B', 'store_C']].clip(lower=0)
    
    # CSVに保存
    df.to_csv(csv_path, index=False)
    print(f"CSVファイルを作成: {csv_path}")
    
    # CSVから読み込み
    historical_data = pd.read_csv(
        csv_path,
        index_col='date',
        parse_dates=True
    )
    
    print(f"\n読み込んだデータの形状: {historical_data.shape}")
    print(f"期間: {historical_data.index[0]} 〜 {historical_data.index[-1]}")
    
    # 新規店舗データ
    new_store_sales = [95000, 98000, 93000, 97000, 100000, 
                      102000, 98000, 96000, 99000, 103000,
                      105000, 101000, 99000, 102000, 106000]
    
    # 予測
    pipeline = PredictionPipeline()
    pipeline.fit(historical_data)
    result = pipeline.predict(new_store_sales)
    
    print(f"\n予測年間売上: {result.prediction.prediction:,.0f}円")
    
    # クリーンアップ
    csv_path.unlink()


def example_4_with_store_attributes():
    """例4: 店舗属性付きデータ"""
    print("\n" + "=" * 60)
    print("例4: 店舗属性付きデータ")
    print("=" * 60)
    
    # 売上データ
    n_days = 120
    store_cds = ['STR001', 'STR002', 'STR003', 'STR004', 'STR005']
    
    historical_data = {}
    for i, store_cd in enumerate(store_cds):
        base = 80000 + i * 20000
        sales = base + np.random.normal(0, base * 0.1, n_days)
        historical_data[store_cd] = np.maximum(sales, 0)
    
    # 店舗属性データ
    store_attributes = pd.DataFrame({
        'store_cd': store_cds,
        'store_type': ['roadside', 'mall', 'roadside', 'urban', 'mall'],
        'area': [150, 120, 180, 100, 130],  # 売場面積（㎡）
        'location': ['suburban', 'urban', 'suburban', 'downtown', 'urban'],
        'parking_spaces': [50, 0, 60, 10, 0],
        'opening_year': [2020, 2019, 2021, 2018, 2022]
    }).set_index('store_cd')
    
    print("店舗属性データ:")
    print(store_attributes)
    
    # 新規店舗（郊外ロードサイド想定）
    new_store_sales = 95000 + np.random.normal(0, 8000, 25)
    new_store_sales = np.maximum(new_store_sales, 0)
    
    # フィルタリング付き予測
    pipeline = PredictionPipeline()
    pipeline.fit(historical_data, store_attributes)
    
    # ロードサイド店舗のみで予測
    result = pipeline.predict(
        new_store_sales,
        filters={'store_type': 'roadside'}
    )
    
    print(f"\n予測年間売上（ロードサイド店舗ベース）: {result.prediction.prediction:,.0f}円")
    print(f"使用した類似店舗: {[s[0] for s in result.prediction.similar_stores]}")


def example_5_json_input():
    """例5: JSONファイルからの入力"""
    print("\n" + "=" * 60)
    print("例5: JSONファイルからの入力")
    print("=" * 60)
    
    # サンプルJSONデータ
    json_data = {
        "metadata": {
            "period": "2023-01-01 to 2023-06-30",
            "unit": "JPY",
            "data_type": "daily_sales"
        },
        "stores": {
            "tokyo_001": {
                "sales": [100000, 105000, 98000, 102000, 99000] * 30,  # 150日分
                "attributes": {"type": "urban", "area": 120}
            },
            "osaka_001": {
                "sales": [85000, 88000, 83000, 87000, 86000] * 30,
                "attributes": {"type": "suburban", "area": 150}
            },
            "nagoya_001": {
                "sales": [92000, 95000, 90000, 93000, 91000] * 30,
                "attributes": {"type": "urban", "area": 100}
            }
        }
    }
    
    # JSONファイルに保存
    json_path = Path("sample_sales.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    # JSONから読み込み
    with open(json_path, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
    
    # データを変換
    historical_data = {}
    store_attrs_list = []
    
    for store_cd, store_info in loaded_data['stores'].items():
        historical_data[store_cd] = np.array(store_info['sales'])
        
        attrs = store_info['attributes'].copy()
        attrs['store_cd'] = store_cd
        store_attrs_list.append(attrs)
    
    store_attributes = pd.DataFrame(store_attrs_list).set_index('store_cd')
    
    print(f"読み込んだ店舗数: {len(historical_data)}")
    print(f"データ期間: {loaded_data['metadata']['period']}")
    
    # 新規店舗データ
    new_store_sales = [98000, 101000, 96000, 100000, 103000,
                      99000, 102000, 105000, 101000, 104000,
                      107000, 103000, 106000, 102000, 105000]
    
    # 予測
    pipeline = PredictionPipeline()
    pipeline.fit(historical_data, store_attributes)
    result = pipeline.predict(new_store_sales)
    
    print(f"\n予測年間売上: {result.prediction.prediction:,.0f}円")
    
    # クリーンアップ
    json_path.unlink()


def example_6_data_validation():
    """例6: データ検証の実例"""
    print("\n" + "=" * 60)
    print("例6: データ検証の実例")
    print("=" * 60)
    
    # バリデータの作成
    validator = DataValidator()
    
    # 1. 正常なデータ
    print("\n--- 正常なデータの検証 ---")
    good_data = np.random.normal(100000, 10000, 30)
    good_data = np.maximum(good_data, 0)
    
    result = validator.validate_prediction_input(good_data)
    print(f"検証結果: {'合格' if result.is_valid else '不合格'}")
    print(f"データ統計: 平均={result.summary.get('mean_sales', 0):.0f}, "
          f"標準偏差={result.summary.get('std_sales', 0):.0f}")
    
    # 2. 問題のあるデータ
    print("\n--- 問題のあるデータの検証 ---")
    bad_data = np.array([
        100000, 105000, np.nan, 110000,  # 欠損値
        -5000, 120000, 125000,            # 負の値
        0, 0, 0, 0,                       # ゼロの連続
        130000, 135000, 1000000,          # 異常値
    ])
    
    result = validator.validate_prediction_input(bad_data, min_days=7)
    print(f"検証結果: {'合格' if result.is_valid else '不合格'}")
    
    if result.errors:
        print("エラー:")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.warnings:
        print("警告:")
        for warning in result.warnings:
            print(f"  - {warning}")


def example_7_real_world_scenario():
    """例7: 実際のビジネスシナリオを想定"""
    print("\n" + "=" * 60)
    print("例7: 実際のビジネスシナリオを想定")
    print("=" * 60)
    
    # 既存チェーン店の2年分のデータ
    dates = pd.date_range('2022-01-01', '2023-12-31', freq='D')
    
    # 10店舗のデータを生成（実際のパターンを模倣）
    stores = {}
    store_info = []
    
    for i in range(10):
        store_cd = f'FC{i+1:03d}'
        
        # 店舗特性
        is_urban = i % 3 == 0
        base_sales = 150000 if is_urban else 100000
        
        # 年間の季節変動（夏に売上増）
        seasonal = 20000 * np.sin((np.arange(len(dates)) - 90) * 2 * np.pi / 365)
        
        # 曜日効果
        weekday_mult = np.array([
            0.9 if d.weekday() == 0 else  # 月曜は少ない
            1.3 if d.weekday() in [5, 6] else  # 週末は多い
            1.0 for d in dates
        ])
        
        # 成長トレンド
        growth = np.linspace(0, 20000, len(dates))
        
        # イベント効果（月初は売上増）
        event_effect = np.array([
            1.1 if d.day <= 3 else 1.0 for d in dates
        ])
        
        # 最終的な売上
        sales = (base_sales + seasonal + growth) * weekday_mult * event_effect
        sales += np.random.normal(0, 5000, len(dates))
        sales = np.maximum(sales, 0)
        
        stores[store_cd] = sales
        
        # 店舗情報
        store_info.append({
            'store_cd': store_cd,
            'type': 'urban' if is_urban else 'suburban',
            'area': np.random.choice([100, 120, 150, 180]),
            'established': 2022 - i // 2
        })
    
    # DataFrameに変換
    historical_data = pd.DataFrame(stores, index=dates)
    store_attributes = pd.DataFrame(store_info).set_index('store_cd')
    
    # 新店舗の1ヶ月のデータ（1月の都市型店舗）
    new_dates = pd.date_range('2024-01-01', periods=31, freq='D')
    new_store_sales = []
    
    for d in new_dates:
        base = 140000  # 都市型想定
        weekday_effect = 0.9 if d.weekday() == 0 else 1.3 if d.weekday() in [5, 6] else 1.0
        day_sales = base * weekday_effect + np.random.normal(0, 8000)
        new_store_sales.append(max(day_sales, 0))
    
    # 高度な設定でパイプライン実行
    from twinstore import PipelineConfig
    
    config = PipelineConfig(
        preprocess_data=True,
        handle_outliers=True,
        check_quality=True,
        similarity_metric="dtw",
        n_similar_stores=5,
        auto_optimize_period=True,
        generate_explanation=True,
        explanation_language="ja"
    )
    
    pipeline = PredictionPipeline(config)
    pipeline.fit(historical_data, store_attributes)
    
    # 都市型店舗のみを使って予測
    result = pipeline.predict(
        new_store_sales,
        store_name="恵比寿店（新規）",
        filters={'type': 'urban'}
    )
    
    print(f"\n新規都市型店舗の予測:")
    print(f"年間売上予測: {result.prediction.prediction:,.0f}円")
    print(f"月間売上予測: {result.prediction.prediction/12:,.0f}円")
    print(f"信頼区間: {result.prediction.lower_bound:,.0f} 〜 {result.prediction.upper_bound:,.0f}円")
    
    if result.explanation:
        print("\n" + result.explanation)


if __name__ == "__main__":
    # 全ての例を実行
    examples = [
        example_1_basic_numpy,
        example_2_pandas_dataframe,
        example_3_csv_input,
        example_4_with_store_attributes,
        example_5_json_input,
        example_6_data_validation,
        example_7_real_world_scenario
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("全ての例の実行が完了しました")
    print("=" * 60)