"""
サンプルデータ生成モジュール

テスト・デモ用のサンプルデータとファイルを生成する責任のみを持つ
"""

from typing import Union
import numpy as np
import pandas as pd
from pathlib import Path

from ..config.defaults import DATA_FORMAT_DEFAULTS


class SampleGenerator:
    """
    サンプルデータ生成専用クラス
    
    単一責任: テスト・デモ用のサンプルデータを生成する
    """
    
    def __init__(
        self,
        date_column: str = None,
        sales_column: str = None,
        store_cd_column: str = None,
        encoding: str = 'utf-8'
    ):
        """
        Parameters
        ----------
        date_column : str
            生成する日付列名
        sales_column : str
            生成する売上列名
        store_cd_column : str
            生成する店舗コード列名
        encoding : str, default='utf-8'
            ファイル出力時のエンコーディング
        """
        self.date_column = date_column or DATA_FORMAT_DEFAULTS['date_column']
        self.sales_column = sales_column or DATA_FORMAT_DEFAULTS['sales_column']
        self.store_cd_column = store_cd_column or DATA_FORMAT_DEFAULTS['store_cd_column']
        self.encoding = encoding
    
    def generate_historical_data(
        self,
        store_codes: list = None,
        days: int = 180,
        base_sales: int = 100000,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        過去売上データを生成
        
        Parameters
        ----------
        store_codes : list, optional
            店舗コードのリスト（デフォルト: ['A001', 'A002', 'A003']）
        days : int, default=180
            生成する日数
        base_sales : int, default=100000
            ベースとなる売上金額
        seed : int, default=42
            乱数シード
            
        Returns
        -------
        pd.DataFrame
            生成された過去売上データ
        """
        if store_codes is None:
            store_codes = ['A001', 'A002', 'A003']
        
        np.random.seed(seed)
        data = []
        
        dates = pd.date_range('2024-01-01', periods=days, freq='D')
        
        for store_cd in store_codes:
            store_base = base_sales + hash(store_cd) % 50000
            
            for i, date in enumerate(dates):
                trend = i * 100
                noise = np.random.normal(0, 5000)
                sales = max(0, store_base + trend + noise)
                
                data.append({
                    self.store_cd_column: store_cd,
                    self.date_column: date,
                    self.sales_column: int(sales)
                })
        
        return pd.DataFrame(data)
    
    def generate_store_attributes(
        self,
        store_codes: list = None
    ) -> pd.DataFrame:
        """
        店舗属性データを生成
        
        Parameters
        ----------
        store_codes : list, optional
            店舗コードのリスト（デフォルト: ['A001', 'A002', 'A003']）
            
        Returns
        -------
        pd.DataFrame
            生成された店舗属性データ
        """
        if store_codes is None:
            store_codes = ['A001', 'A002', 'A003']
        
        type_options = ['urban', 'suburban', 'roadside']
        location_options = ['Tokyo', 'Osaka', 'Nagoya', 'Fukuoka', 'Sendai']
        
        data = []
        for i, store_cd in enumerate(store_codes):
            data.append({
                self.store_cd_column: store_cd,
                'type': type_options[i % len(type_options)],
                'area': 150 + (i * 50),
                'location': location_options[i % len(location_options)]
            })
        
        return pd.DataFrame(data)
    
    def generate_new_store_data(
        self,
        store_codes: list = None,
        days: int = 30,
        base_sales: int = 95000,
        seed: int = 42
    ) -> pd.DataFrame:
        """
        新規店舗データを生成
        
        Parameters
        ----------
        store_codes : list, optional
            店舗コードのリスト（デフォルト: ['N001', 'N002']）
        days : int, default=30
            生成する日数
        base_sales : int, default=95000
            ベースとなる売上金額
        seed : int, default=42
            乱数シード
            
        Returns
        -------
        pd.DataFrame
            生成された新規店舗データ
        """
        if store_codes is None:
            store_codes = ['N001', 'N002']
        
        np.random.seed(seed)
        data = []
        
        dates = pd.date_range('2024-07-01', periods=days, freq='D')
        
        for store_cd in store_codes:
            store_base = base_sales + hash(store_cd) % 10000
            
            for i, date in enumerate(dates):
                trend = i * 50
                noise = np.random.normal(0, 3000)
                sales = max(0, store_base + trend + noise)
                
                data.append({
                    self.store_cd_column: store_cd,
                    self.date_column: date,
                    self.sales_column: int(sales)
                })
        
        return pd.DataFrame(data)
    
    def create_sample_files(self, output_dir: Union[str, Path] = "sample_data") -> None:
        """
        サンプルファイルを生成
        
        Parameters
        ----------
        output_dir : str or Path, default='sample_data'
            出力ディレクトリ
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 過去売上データ
        historical_df = self.generate_historical_data()
        
        # CSV形式で保存
        csv_path = output_dir / "historical_sales.csv"
        historical_df.to_csv(csv_path, index=False, encoding=self.encoding)
        
        # Excel形式で保存
        excel_path = output_dir / "historical_sales.xlsx"
        historical_df.to_excel(excel_path, index=False, sheet_name="Sales")
        
        # 店舗属性データ
        attributes_df = self.generate_store_attributes()
        attr_csv_path = output_dir / "store_attributes.csv"
        attributes_df.to_csv(attr_csv_path, index=False, encoding=self.encoding)
        
        # 新規店舗データ
        new_store_df = self.generate_new_store_data()
        new_store_csv_path = output_dir / "new_stores.csv"
        new_store_df.to_csv(new_store_csv_path, index=False, encoding=self.encoding)
        
        print(f"Sample files created in: {output_dir}")
        print(f"- Historical sales: {csv_path}, {excel_path}")
        print(f"- Store attributes: {attr_csv_path}")
        print(f"- New stores: {new_store_csv_path}")
    
    def generate_custom_format_sample(
        self,
        output_dir: Union[str, Path],
        store_column: str = 'shop_id',
        date_column: str = 'business_date',
        sales_column: str = 'daily_revenue'
    ) -> None:
        """
        カスタム列名のサンプルファイルを生成
        
        Parameters
        ----------
        output_dir : str or Path
            出力ディレクトリ
        store_column : str, default='shop_id'
            店舗コード列名
        date_column : str, default='business_date'
            日付列名
        sales_column : str, default='daily_revenue'
            売上列名
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 一時的に列名を変更
        original_store_column = self.store_cd_column
        original_date_column = self.date_column
        original_sales_column = self.sales_column
        
        self.store_cd_column = store_column
        self.date_column = date_column
        self.sales_column = sales_column
        
        try:
            # カスタム列名でデータ生成
            df = self.generate_historical_data(
                store_codes=['SHOP_001', 'SHOP_002'],
                days=60
            )
            
            custom_csv = output_dir / "custom_columns.csv"
            df.to_csv(custom_csv, index=False, encoding=self.encoding)
            
            print(f"Custom format sample created: {custom_csv}")
            print(f"Columns: {store_column}, {date_column}, {sales_column}")
            
        finally:
            # 元の列名に戻す
            self.store_cd_column = original_store_column
            self.date_column = original_date_column
            self.sales_column = original_sales_column