"""
PipelineのCSV/Excel対応機能のテスト
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from twinstore import PredictionPipeline, PipelineConfig


class TestPipelineFileSupport:
    """Pipelineのファイルサポートテスト"""
    
    def setup_method(self):
        """各テスト前のセットアップ"""
        self.config = PipelineConfig(
            validate_input=True,
            preprocess_data=False,  # テスト簡略化のため無効化
            check_quality=False,    # テスト簡略化のため無効化
            generate_explanation=False,  # テスト簡略化のため無効化
            verbose=False
        )
        self.pipeline = PredictionPipeline(self.config)
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # テスト用データの準備
        np.random.seed(42)
        self.historical_data = []
        for store_cd in ['A001', 'A002', 'A003']:
            for i in range(30):
                date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=i)
                sales = 100000 + np.random.normal(0, 10000)
                self.historical_data.append({
                    'store_cd': store_cd,
                    'date': date.strftime('%Y-%m-%d'),
                    'sales': max(0, int(sales))
                })
        
        self.store_attributes = [
            {'store_cd': 'A001', 'type': 'urban', 'area': 150},
            {'store_cd': 'A002', 'type': 'suburban', 'area': 200},
            {'store_cd': 'A003', 'type': 'roadside', 'area': 300},
        ]
        
        self.new_store_data = []
        for store_cd in ['N001', 'N002']:
            for i in range(14):
                date = pd.Timestamp('2024-02-01') + pd.Timedelta(days=i)
                sales = 95000 + np.random.normal(0, 5000)
                self.new_store_data.append({
                    'store_cd': store_cd,
                    'date': date.strftime('%Y-%m-%d'),
                    'sales': max(0, int(sales))
                })
    
    def teardown_method(self):
        """各テスト後のクリーンアップ"""
        # 一時ファイルの削除
        for file in self.temp_dir.glob("*"):
            file.unlink()
        self.temp_dir.rmdir()
    
    def test_fit_with_csv_file(self):
        """CSVファイルを使ったパイプライン学習テスト"""
        # CSVファイルを作成
        df = pd.DataFrame(self.historical_data)
        csv_path = self.temp_dir / "historical.csv"
        df.to_csv(csv_path, index=False)
        
        # パイプライン学習
        trained_pipeline = self.pipeline.fit(csv_path)
        
        assert trained_pipeline is not None
        assert isinstance(trained_pipeline, PredictionPipeline)
    
    def test_fit_with_excel_file(self):
        """Excelファイルを使ったパイプライン学習テスト"""
        # Excelファイルを作成
        df = pd.DataFrame(self.historical_data)
        excel_path = self.temp_dir / "historical.xlsx"
        df.to_excel(excel_path, index=False, sheet_name="Sales")
        
        # パイプライン学習
        trained_pipeline = self.pipeline.fit(excel_path)
        
        assert trained_pipeline is not None
        assert isinstance(trained_pipeline, PredictionPipeline)
    
    def test_fit_with_store_attributes_file(self):
        """店舗属性ファイルを使ったパイプライン学習テスト"""
        # 過去データCSVを作成
        df = pd.DataFrame(self.historical_data)
        historical_csv = self.temp_dir / "historical.csv"
        df.to_csv(historical_csv, index=False)
        
        # 店舗属性CSVを作成
        attr_df = pd.DataFrame(self.store_attributes)
        attr_csv = self.temp_dir / "attributes.csv"
        attr_df.to_csv(attr_csv, index=False)
        
        # パイプライン学習
        trained_pipeline = self.pipeline.fit(historical_csv, attr_csv)
        
        assert trained_pipeline is not None
        assert isinstance(trained_pipeline, PredictionPipeline)
    
    def test_batch_predict_with_csv_file(self):
        """CSVファイルを使ったバッチ予測テスト"""
        # まずパイプラインを学習
        df = pd.DataFrame(self.historical_data)
        historical_csv = self.temp_dir / "historical.csv"
        df.to_csv(historical_csv, index=False)
        self.pipeline.fit(historical_csv)
        
        # 新規店舗データCSVを作成
        new_df = pd.DataFrame(self.new_store_data)
        new_csv = self.temp_dir / "new_stores.csv"
        new_df.to_csv(new_csv, index=False)
        
        # バッチ予測実行
        results = self.pipeline.batch_predict(new_csv)
        
        assert isinstance(results, dict)
        assert 'N001' in results
        assert 'N002' in results
        assert hasattr(results['N001'], 'prediction')
        assert hasattr(results['N002'], 'prediction')
    
    def test_json_format_support(self):
        """JSON形式ファイルサポートテスト"""
        # JSONファイルを作成
        json_path = self.temp_dir / "historical.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.historical_data, f, ensure_ascii=False)
        
        # パイプライン学習
        trained_pipeline = self.pipeline.fit(json_path)
        
        assert trained_pipeline is not None
        assert isinstance(trained_pipeline, PredictionPipeline)
    
    def test_custom_column_names(self):
        """カスタム列名を使ったテスト"""
        # カスタム列名のデータを作成
        custom_data = []
        for row in self.historical_data:
            custom_data.append({
                'shop_id': row['store_cd'],
                'timestamp': row['date'],
                'amount': row['sales']
            })
        
        df = pd.DataFrame(custom_data)
        csv_path = self.temp_dir / "custom_columns.csv"
        df.to_csv(csv_path, index=False)
        
        # カスタム設定でパイプライン作成
        custom_config = PipelineConfig(
            store_cd_column='shop_id',
            date_column='timestamp',
            sales_column='amount',
            validate_input=True,
            preprocess_data=False,
            check_quality=False,
            generate_explanation=False,
            verbose=False
        )
        custom_pipeline = PredictionPipeline(custom_config)
        
        # パイプライン学習
        trained_pipeline = custom_pipeline.fit(csv_path)
        
        assert trained_pipeline is not None
        assert isinstance(trained_pipeline, PredictionPipeline)
    
    def test_excel_sheet_name_option(self):
        """Excelのシート名指定テスト"""
        # 複数シートのExcelファイルを作成
        df = pd.DataFrame(self.historical_data)
        excel_path = self.temp_dir / "multi_sheet.xlsx"
        
        with pd.ExcelWriter(excel_path) as writer:
            df.to_excel(writer, sheet_name="SalesData", index=False)
            pd.DataFrame({'dummy': [1, 2, 3]}).to_excel(writer, sheet_name="Sheet1", index=False)
        
        # 特定のシートを指定して学習
        trained_pipeline = self.pipeline.fit(excel_path, sheet_name="SalesData")
        
        assert trained_pipeline is not None
        assert isinstance(trained_pipeline, PredictionPipeline)
    
    def test_file_not_found_error(self):
        """存在しないファイル指定時のエラーテスト"""
        non_existent_file = self.temp_dir / "non_existent.csv"
        
        with pytest.raises(FileNotFoundError):
            self.pipeline.fit(non_existent_file)
    
    def test_invalid_file_format_error(self):
        """サポートされていないファイル形式のエラーテスト"""
        # テキストファイルを作成
        txt_path = self.temp_dir / "test.txt"
        with open(txt_path, 'w') as f:
            f.write("invalid format")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            self.pipeline.fit(txt_path)
    
    def test_mixed_input_types(self):
        """異なる入力タイプの混在テスト"""
        # CSVファイルから学習
        df = pd.DataFrame(self.historical_data)
        csv_path = self.temp_dir / "historical.csv"
        df.to_csv(csv_path, index=False)
        
        # DataFrameで店舗属性を渡して学習
        attr_df = pd.DataFrame(self.store_attributes).set_index('store_cd')
        
        trained_pipeline = self.pipeline.fit(csv_path, attr_df)
        
        assert trained_pipeline is not None
        assert isinstance(trained_pipeline, PredictionPipeline)
    
    def test_loader_kwargs_passing(self):
        """DataLoaderへのkwargs引数渡しテスト"""
        # カスタム区切り文字のCSVを作成
        df = pd.DataFrame(self.historical_data)
        csv_path = self.temp_dir / "semicolon_separated.csv"
        df.to_csv(csv_path, index=False, sep=';')
        
        # カスタム区切り文字を指定して学習
        trained_pipeline = self.pipeline.fit(csv_path, sep=';')
        
        assert trained_pipeline is not None
        assert isinstance(trained_pipeline, PredictionPipeline)