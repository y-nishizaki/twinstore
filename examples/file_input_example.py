"""
TwinStore ファイル入力サンプル

CSV、Excel、JSONファイルを直接読み込んでパイプラインを実行する例
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile

from twinstore import PredictionPipeline, PipelineConfig
from twinstore.data import DataLoader


def create_sample_data():
    """サンプルデータファイルを作成"""
    print("サンプルデータファイルを作成中...")
    
    # 一時ディレクトリを作成
    temp_dir = Path(tempfile.mkdtemp())
    print(f"サンプルファイル出力先: {temp_dir}")
    
    # DataLoaderを使ってサンプルファイルを生成
    loader = DataLoader()
    loader.create_sample_files(temp_dir)
    
    return temp_dir


def example_csv_input():
    """例1: CSVファイルからの読み込み"""
    print("\n" + "="*60)
    print("例1: CSVファイルからパイプライン実行")
    print("="*60)
    
    # サンプルファイルを作成
    sample_dir = create_sample_data()
    
    # パイプライン設定
    config = PipelineConfig(
        validate_input=True,
        preprocess_data=True,
        generate_explanation=True,
        save_results=False,
        verbose=True
    )
    
    # パイプライン初期化
    pipeline = PredictionPipeline(config)
    
    # CSVファイルから学習
    historical_csv = sample_dir / "historical_sales.csv"
    attributes_csv = sample_dir / "store_attributes.csv"
    
    print(f"過去データを読み込み: {historical_csv}")
    print(f"店舗属性を読み込み: {attributes_csv}")
    
    pipeline.fit(historical_csv, attributes_csv)
    print("✓ パイプライン学習完了")
    
    # 新規店舗データで予測
    new_stores_csv = sample_dir / "new_stores.csv"
    print(f"新規店舗データを読み込んでバッチ予測: {new_stores_csv}")
    
    batch_results = pipeline.batch_predict(new_stores_csv)
    
    print(f"✓ {len(batch_results)}店舗の予測完了")
    for store_cd, result in batch_results.items():
        if result.prediction:
            print(f"  {store_cd}: 年間売上予測 = ¥{result.prediction.prediction:,.0f}")
    
    # クリーンアップ
    import shutil
    shutil.rmtree(sample_dir)


def example_excel_input():
    """例2: Excelファイルからの読み込み"""
    print("\n" + "="*60)
    print("例2: Excelファイルからパイプライン実行")
    print("="*60)
    
    # サンプルファイルを作成
    sample_dir = create_sample_data()
    
    # パイプライン初期化
    pipeline = PredictionPipeline()
    
    # Excelファイルから学習
    historical_excel = sample_dir / "historical_sales.xlsx"
    
    print(f"Excelファイルから学習: {historical_excel}")
    pipeline.fit(historical_excel, sheet_name="Sales")
    print("✓ Excel読み込みと学習完了")
    
    # 単体予測のため、サンプルデータを生成
    np.random.seed(42)
    new_store_sales = [95000 + np.random.normal(0, 3000) for _ in range(14)]
    
    result = pipeline.predict(new_store_sales, store_name="新規店舗")
    
    if result.prediction:
        print(f"新規店舗の年間売上予測: ¥{result.prediction.prediction:,.0f}")
        print(f"信頼区間: ¥{result.prediction.lower_bound:,.0f} - ¥{result.prediction.upper_bound:,.0f}")
    
    # クリーンアップ
    import shutil
    shutil.rmtree(sample_dir)


def example_custom_columns():
    """例3: カスタム列名を使った読み込み"""
    print("\n" + "="*60)
    print("例3: カスタム列名でのファイル読み込み")
    print("="*60)
    
    # カスタム列名のデータを作成
    temp_dir = Path(tempfile.mkdtemp())
    
    # カスタム列名のCSVデータを作成
    np.random.seed(42)
    custom_data = []
    
    for shop_id in ['SHOP_001', 'SHOP_002', 'SHOP_003']:
        for i in range(60):
            date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)
            revenue = 120000 + i * 500 + np.random.normal(0, 8000)
            custom_data.append({
                'shop_code': shop_id,
                'business_date': date.strftime('%Y-%m-%d'),
                'daily_revenue': max(0, int(revenue))
            })
    
    df = pd.DataFrame(custom_data)
    custom_csv = temp_dir / "custom_format.csv"
    df.to_csv(custom_csv, index=False)
    
    print(f"カスタム列名のファイル: {custom_csv}")
    print("列名: shop_code, business_date, daily_revenue")
    
    # カスタム設定でパイプライン作成
    config = PipelineConfig(
        store_cd_column='shop_code',
        date_column='business_date',
        sales_column='daily_revenue',
        verbose=True
    )
    
    pipeline = PredictionPipeline(config)
    
    # カスタム列名のファイルから学習
    pipeline.fit(custom_csv)
    print("✓ カスタム列名での学習完了")
    
    # 予測実行
    new_shop_sales = [115000 + np.random.normal(0, 5000) for _ in range(10)]
    result = pipeline.predict(new_shop_sales, store_name="新規ショップ")
    
    if result.prediction:
        print(f"新規ショップの年間売上予測: ¥{result.prediction.prediction:,.0f}")
    
    # クリーンアップ
    import shutil
    shutil.rmtree(temp_dir)


def example_json_input():
    """例4: JSON形式ファイルからの読み込み"""
    print("\n" + "="*60)
    print("例4: JSONファイルからパイプライン実行")
    print("="*60)
    
    # JSON形式のデータを作成
    temp_dir = Path(tempfile.mkdtemp())
    
    # JSON形式（リスト形式）のデータ
    np.random.seed(42)
    json_data = []
    
    for store_cd in ['S001', 'S002']:
        for i in range(45):
            date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)
            sales = 110000 + i * 200 + np.random.normal(0, 6000)
            json_data.append({
                'store_cd': store_cd,
                'date': date.strftime('%Y-%m-%d'),
                'sales': max(0, int(sales))
            })
    
    import json
    json_path = temp_dir / "sales_data.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"JSONファイル: {json_path}")
    
    # パイプライン実行
    pipeline = PredictionPipeline()
    pipeline.fit(json_path)
    print("✓ JSON読み込みと学習完了")
    
    # 予測
    new_store_sales = [108000 + np.random.normal(0, 4000) for _ in range(12)]
    result = pipeline.predict(new_store_sales, store_name="JSON学習店舗")
    
    if result.prediction:
        print(f"年間売上予測: ¥{result.prediction.prediction:,.0f}")
    
    # クリーンアップ
    import shutil
    shutil.rmtree(temp_dir)


def example_file_options():
    """例5: ファイル読み込みオプションの使用"""
    print("\n" + "="*60)
    print("例5: ファイル読み込みオプションの活用")
    print("="*60)
    
    # 特殊な形式のCSVファイルを作成
    temp_dir = Path(tempfile.mkdtemp())
    
    # セミコロン区切りのCSVデータ
    data = []
    np.random.seed(42)
    
    for store_cd in ['ST01', 'ST02']:
        for i in range(30):
            date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)
            sales = 105000 + np.random.normal(0, 7000)
            data.append({
                'store_cd': store_cd,
                'date': date.strftime('%Y-%m-%d'),
                'sales': max(0, int(sales))
            })
    
    df = pd.DataFrame(data)
    
    # セミコロン区切りで保存
    semicolon_csv = temp_dir / "semicolon_data.csv"
    df.to_csv(semicolon_csv, index=False, sep=';')
    
    print(f"セミコロン区切りCSV: {semicolon_csv}")
    print("区切り文字: ; (セミコロン)")
    
    # カスタム区切り文字を指定してパイプライン実行
    pipeline = PredictionPipeline()
    
    # sep引数でカスタム区切り文字を指定
    pipeline.fit(semicolon_csv, sep=';')
    print("✓ セミコロン区切りCSVからの学習完了")
    
    # Excelファイルでシート名指定の例
    excel_path = temp_dir / "multi_sheet.xlsx"
    with pd.ExcelWriter(excel_path) as writer:
        df.to_excel(writer, sheet_name="ActualSales", index=False)
        pd.DataFrame({'dummy': [1, 2, 3]}).to_excel(writer, sheet_name="Other", index=False)
    
    print(f"複数シートExcel: {excel_path}")
    
    # 特定のシートを指定して学習
    pipeline2 = PredictionPipeline()
    pipeline2.fit(excel_path, sheet_name="ActualSales")
    print("✓ 指定シートからの学習完了")
    
    # クリーンアップ
    import shutil
    shutil.rmtree(temp_dir)


def main():
    """メイン実行関数"""
    print("TwinStore ファイル入力サンプル")
    print("CSV、Excel、JSONファイルの直接読み込み機能のデモンストレーション")
    
    try:
        # 各例を実行
        example_csv_input()
        example_excel_input()
        example_custom_columns()
        example_json_input()
        example_file_options()
        
        print("\n" + "="*60)
        print("すべてのファイル入力例の実行が完了しました！")
        print("="*60)
        
        print("\n利用可能なファイル形式:")
        print("- CSV (.csv)")
        print("- Excel (.xlsx, .xls)")
        print("- JSON (.json)")
        
        print("\nカスタマイズ可能な設定:")
        print("- 列名の指定 (store_cd_column, date_column, sales_column)")
        print("- ファイル読み込みオプション (区切り文字、シート名など)")
        print("- エンコーディング設定")
        
        print("\n使用方法:")
        print("pipeline.fit('historical_data.csv')")
        print("pipeline.fit('data.xlsx', sheet_name='Sales')")
        print("pipeline.batch_predict('new_stores.csv')")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()