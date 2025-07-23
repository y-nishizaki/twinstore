"""
FileReaderクラスのテスト
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from twinstore.data.file_reader import FileReader


class TestFileReader:
    """FileReaderクラスのテスト"""
    
    def setup_method(self):
        """各テスト前のセットアップ"""
        self.reader = FileReader()
        self.temp_dir = Path(tempfile.mkdtemp())
        
        # テスト用データの準備
        self.sample_data = [
            {'store_cd': 'A001', 'date': '2024-01-01', 'sales': 100000},
            {'store_cd': 'A001', 'date': '2024-01-02', 'sales': 105000},
            {'store_cd': 'A002', 'date': '2024-01-01', 'sales': 95000},
        ]
    
    def teardown_method(self):
        """各テスト後のクリーンアップ"""
        for file in self.temp_dir.glob("*"):
            file.unlink()
        self.temp_dir.rmdir()
    
    def test_read_csv(self):
        """CSV読み込みテスト"""
        df = pd.DataFrame(self.sample_data)
        csv_path = self.temp_dir / "test.csv"
        df.to_csv(csv_path, index=False)
        
        result = self.reader.read_file(csv_path)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
        assert 'store_cd' in result.columns
        assert 'sales' in result.columns
    
    def test_read_excel(self):
        """Excel読み込みテスト"""
        df = pd.DataFrame(self.sample_data)
        excel_path = self.temp_dir / "test.xlsx"
        df.to_excel(excel_path, index=False)
        
        result = self.reader.read_file(excel_path)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
    
    def test_read_json_list(self):
        """JSONリスト形式読み込みテスト"""
        json_path = self.temp_dir / "test.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.sample_data, f)
        
        result = self.reader.read_file(json_path)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3
    
    def test_read_json_dict(self):
        """JSON辞書形式読み込みテスト"""
        json_data = {
            'A001': [100000, 105000],
            'A002': [95000, 97000]
        }
        
        json_path = self.temp_dir / "test_dict.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f)
        
        result = self.reader.read_file(json_path)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4  # 2 stores * 2 values each
    
    def test_file_not_found(self):
        """存在しないファイルのテスト"""
        non_existent = self.temp_dir / "non_existent.csv"
        
        with pytest.raises(FileNotFoundError):
            self.reader.read_file(non_existent)
    
    def test_unsupported_format(self):
        """サポートされていない形式のテスト"""
        txt_path = self.temp_dir / "test.txt"
        with open(txt_path, 'w') as f:
            f.write("test")
        
        with pytest.raises(ValueError, match="Unsupported file format"):
            self.reader.read_file(txt_path)
    
    def test_csv_with_custom_separator(self):
        """CSV カスタム区切り文字テスト"""
        df = pd.DataFrame(self.sample_data)
        csv_path = self.temp_dir / "semicolon.csv"
        df.to_csv(csv_path, index=False, sep=';')
        
        result = self.reader.read_file(csv_path, sep=';')
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3