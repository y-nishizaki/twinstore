"""
DataLoaderクラスのテスト
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json
import os

from twinstore.data.loader import DataLoader


class TestDataLoader:
    """DataLoaderクラスのテスト"""
    
    def setup_method(self):
        """各テスト前のセットアップ"""
        self.loader = DataLoader()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # テスト用データの準備
        self.sample_data = [
            {'store_cd': 'A001', 'date': '2024-01-01', 'sales': 100000},
            {'store_cd': 'A001', 'date': '2024-01-02', 'sales': 105000},
            {'store_cd': 'A001', 'date': '2024-01-03', 'sales': 98000},
            {'store_cd': 'A002', 'date': '2024-01-01', 'sales': 95000},
            {'store_cd': 'A002', 'date': '2024-01-02', 'sales': 97000},
            {'store_cd': 'A002', 'date': '2024-01-03', 'sales': 99000},
        ]
        
        self.sample_attributes = [
            {'store_cd': 'A001', 'type': 'urban', 'area': 150},
            {'store_cd': 'A002', 'type': 'suburban', 'area': 200},
        ]
    
    def teardown_method(self):
        """各テスト後のクリーンアップ"""
        # 一時ファイルの削除
        for file in self.temp_dir.glob("*"):
            file.unlink()
        self.temp_dir.rmdir()
    
    def test_init_default(self):
        """デフォルト設定での初期化テスト"""
        loader = DataLoader()
        assert loader.date_column == 'date'
        assert loader.sales_column == 'sales'
        assert loader.store_cd_column == 'store_cd'
        assert loader.encoding == 'utf-8'
    
    def test_init_custom(self):
        """カスタム設定での初期化テスト"""
        loader = DataLoader(
            date_column='timestamp',
            sales_column='amount',
            store_cd_column='shop_id',
            encoding='shift-jis'
        )
        assert loader.date_column == 'timestamp'
        assert loader.sales_column == 'amount'
        assert loader.store_cd_column == 'shop_id'
        assert loader.encoding == 'shift-jis'
    
    def test_load_csv_historical_data(self):
        """CSV形式の過去データ読み込みテスト"""
        # CSVファイルを作成
        df = pd.DataFrame(self.sample_data)
        csv_path = self.temp_dir / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        # 辞書形式で読み込み
        result = self.loader.load_historical_data(csv_path, output_format="dict")
        
        assert isinstance(result, dict)
        assert 'A001' in result
        assert 'A002' in result
        assert len(result['A001']) == 3
        assert len(result['A002']) == 3
        assert result['A001'][0] == 100000
        assert result['A002'][0] == 95000
    
    def test_load_csv_dataframe_format(self):
        """CSV形式でDataFrame出力テスト"""
        # CSVファイルを作成
        df = pd.DataFrame(self.sample_data)
        csv_path = self.temp_dir / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        # DataFrame形式で読み込み
        result = self.loader.load_historical_data(csv_path, output_format="dataframe")
        
        assert isinstance(result, pd.DataFrame)
        assert 'A001' in result.columns
        assert 'A002' in result.columns
        assert len(result) == 3
    
    def test_load_excel_historical_data(self):
        """Excel形式の過去データ読み込みテスト"""
        # Excelファイルを作成
        df = pd.DataFrame(self.sample_data)
        excel_path = self.temp_dir / "test_data.xlsx"
        df.to_excel(excel_path, index=False, sheet_name="Sales")
        
        # 読み込み
        result = self.loader.load_historical_data(excel_path, output_format="dict")
        
        assert isinstance(result, dict)
        assert 'A001' in result
        assert 'A002' in result
        assert len(result['A001']) == 3
        assert len(result['A002']) == 3
    
    def test_load_json_historical_data(self):
        """JSON形式の過去データ読み込みテスト"""
        # JSONファイルを作成
        json_path = self.temp_dir / "test_data.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.sample_data, f)
        
        # 読み込み
        result = self.loader.load_historical_data(json_path, output_format="dict")
        
        assert isinstance(result, dict)
        assert 'A001' in result
        assert 'A002' in result
    
    def test_load_store_attributes(self):
        """店舗属性データ読み込みテスト"""
        # CSVファイルを作成
        df = pd.DataFrame(self.sample_attributes)
        csv_path = self.temp_dir / "attributes.csv"
        df.to_csv(csv_path, index=False)
        
        # 読み込み
        result = self.loader.load_store_attributes(csv_path)
        
        assert isinstance(result, pd.DataFrame)
        assert result.index.name == 'store_cd'
        assert 'A001' in result.index
        assert 'A002' in result.index
        assert 'type' in result.columns
        assert 'area' in result.columns
    
    def test_load_new_store_data(self):
        """新規店舗データ読み込みテスト"""
        # 新規店舗データを作成
        new_store_data = [
            {'store_cd': 'N001', 'date': '2024-02-01', 'sales': 90000},
            {'store_cd': 'N001', 'date': '2024-02-02', 'sales': 92000},
            {'store_cd': 'N002', 'date': '2024-02-01', 'sales': 88000},
            {'store_cd': 'N002', 'date': '2024-02-02', 'sales': 91000},
        ]
        
        df = pd.DataFrame(new_store_data)
        csv_path = self.temp_dir / "new_stores.csv"
        df.to_csv(csv_path, index=False)
        
        # 読み込み
        result = self.loader.load_new_store_data(csv_path)
        
        assert isinstance(result, dict)
        assert 'N001' in result
        assert 'N002' in result
        assert len(result['N001']) == 2
        assert len(result['N002']) == 2
        assert result['N001'][0] == 90000
        assert result['N002'][0] == 88000
    
    def test_file_not_found(self):
        """存在しないファイル読み込みエラーテスト"""
        non_existent_path = self.temp_dir / "non_existent.csv"
        
        with pytest.raises(FileNotFoundError):
            self.loader.load_historical_data(non_existent_path)
    
    def test_unsupported_format(self):
        """サポートされていないファイル形式エラーテスト"""
        # テキストファイルを作成
        txt_path = self.temp_dir / "test.txt"
        with open(txt_path, 'w') as f:
            f.write("test data")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            self.loader.load_historical_data(txt_path)
    
    def test_missing_required_columns(self):
        """必須列が欠けている場合のエラーテスト"""
        # 店舗コード列がないデータ
        invalid_data = [
            {'date': '2024-01-01', 'sales': 100000},
            {'date': '2024-01-02', 'sales': 105000},
        ]
        
        df = pd.DataFrame(invalid_data)
        csv_path = self.temp_dir / "invalid.csv"
        df.to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError, match="Required columns not found"):
            self.loader.load_historical_data(csv_path)
    
    def test_column_auto_detection(self):
        """列名の自動検出テスト"""
        # 異なる列名のデータ
        data_with_different_columns = [
            {'shop_id': 'A001', 'timestamp': '2024-01-01', 'amount': 100000},
            {'shop_id': 'A001', 'timestamp': '2024-01-02', 'amount': 105000},
            {'shop_id': 'A002', 'timestamp': '2024-01-01', 'amount': 95000},
        ]
        
        df = pd.DataFrame(data_with_different_columns)
        csv_path = self.temp_dir / "different_columns.csv"
        df.to_csv(csv_path, index=False)
        
        # 警告が出るが読み込みは成功するはず
        with pytest.warns(UserWarning):
            result = self.loader.load_historical_data(csv_path, output_format="dict")
        
        assert isinstance(result, dict)
        assert 'A001' in result
        assert 'A002' in result
    
    def test_create_sample_files(self):
        """サンプルファイル生成テスト"""
        sample_dir = self.temp_dir / "samples"
        self.loader.create_sample_files(sample_dir)
        
        # 生成されたファイルの確認
        assert (sample_dir / "historical_sales.csv").exists()
        assert (sample_dir / "historical_sales.xlsx").exists()
        assert (sample_dir / "store_attributes.csv").exists()
        assert (sample_dir / "new_stores.csv").exists()
        
        # ファイル内容の確認
        historical_df = pd.read_csv(sample_dir / "historical_sales.csv")
        assert 'store_cd' in historical_df.columns
        assert 'date' in historical_df.columns
        assert 'sales' in historical_df.columns
        assert len(historical_df) > 0
        
        attributes_df = pd.read_csv(sample_dir / "store_attributes.csv")
        assert 'store_cd' in attributes_df.columns
        assert len(attributes_df) > 0
    
    def test_json_dict_format(self):
        """JSONの辞書形式データ読み込みテスト"""
        # 辞書形式のJSONデータ
        json_data = {
            'A001': [100000, 105000, 98000],
            'A002': [95000, 97000, 99000]
        }
        
        json_path = self.temp_dir / "dict_format.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f)
        
        # 読み込み
        result = self.loader.load_historical_data(json_path, output_format="dict")
        
        assert isinstance(result, dict)
        assert 'A001' in result
        assert 'A002' in result
        assert len(result['A001']) == 3
        assert result['A001'][0] == 100000
    
    def test_invalid_output_format(self):
        """無効な出力形式エラーテスト"""
        df = pd.DataFrame(self.sample_data)
        csv_path = self.temp_dir / "test_data.csv"
        df.to_csv(csv_path, index=False)
        
        with pytest.raises(ValueError, match="Unsupported output format"):
            self.loader.load_historical_data(csv_path, output_format="invalid_format")