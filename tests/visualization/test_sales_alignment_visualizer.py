"""
SalesAlignmentVisualizer のテスト
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from twinstore.visualization.sales_alignment_visualizer import (
    SalesAlignmentVisualizer,
    AlignmentConfig
)


class TestSalesAlignmentVisualizer:
    """SalesAlignmentVisualizerのテストクラス"""
    
    def test_initialization(self):
        """初期化のテスト"""
        # デフォルト設定
        visualizer = SalesAlignmentVisualizer()
        assert visualizer.config.reference_point == "opening_date"
        assert visualizer.config.normalize_sales == False
        assert visualizer.config.show_confidence_band == True
        
        # カスタム設定
        config = AlignmentConfig(
            normalize_sales=True,
            normalization_method="opening_day",
            show_annotations=False
        )
        visualizer = SalesAlignmentVisualizer(config)
        assert visualizer.config.normalize_sales == True
        assert visualizer.config.normalization_method == "opening_day"
    
    def test_align_sales_data_dict(self, sample_sales_data, sample_opening_dates):
        """辞書形式でのアラインメントテスト"""
        visualizer = SalesAlignmentVisualizer()
        
        aligned_data = visualizer.align_sales_data(
            sample_sales_data,
            sample_opening_dates
        )
        
        assert isinstance(aligned_data, pd.DataFrame)
        assert 'store_cd' in aligned_data.columns
        assert 'days_from_opening' in aligned_data.columns
        assert 'sales' in aligned_data.columns
        assert len(aligned_data) > 0
    
    def test_align_sales_data_dataframe(self, sample_sales_dataframe, sample_opening_dates):
        """DataFrame形式でのアラインメントテスト"""
        visualizer = SalesAlignmentVisualizer()
        
        # 開店日を調整（DataFrameの日付範囲に合わせる）
        opening_dates = {
            col: sample_sales_dataframe.index[0] + timedelta(days=i*10)
            for i, col in enumerate(sample_sales_dataframe.columns)
        }
        
        aligned_data = visualizer.align_sales_data(
            sample_sales_dataframe,
            opening_dates
        )
        
        assert isinstance(aligned_data, pd.DataFrame)
        assert len(aligned_data) > 0
        
        # 各店舗のデータが含まれている
        for store_cd in sample_sales_dataframe.columns:
            assert store_cd in aligned_data['store_cd'].values
    
    def test_align_without_opening_dates(self, sample_sales_data):
        """開店日なしでのアラインメントテスト"""
        visualizer = SalesAlignmentVisualizer()
        
        # 開店日を指定しない場合も動作する
        aligned_data = visualizer.align_sales_data(sample_sales_data)
        
        assert isinstance(aligned_data, pd.DataFrame)
        assert len(aligned_data) > 0
    
    def test_normalization_max(self, sample_sales_data, sample_opening_dates):
        """最大値正規化のテスト"""
        config = AlignmentConfig(
            normalize_sales=True,
            normalization_method="max"
        )
        visualizer = SalesAlignmentVisualizer(config)
        
        aligned_data = visualizer.align_sales_data(
            sample_sales_data,
            sample_opening_dates
        )
        
        # 各店舗の最大値が1.0になっている
        for store_cd in sample_sales_data.keys():
            store_data = aligned_data[aligned_data['store_cd'] == store_cd]
            if len(store_data) > 0:
                assert store_data['sales'].max() == pytest.approx(1.0, abs=0.01)
    
    def test_normalization_opening_day(self, sample_sales_data, sample_opening_dates):
        """開店日正規化のテスト"""
        config = AlignmentConfig(
            normalize_sales=True,
            normalization_method="opening_day"
        )
        visualizer = SalesAlignmentVisualizer(config)
        
        aligned_data = visualizer.align_sales_data(
            sample_sales_data,
            sample_opening_dates
        )
        
        # 各店舗の開店日（day 0）の売上が1.0になっている
        for store_cd in sample_sales_data.keys():
            store_data = aligned_data[
                (aligned_data['store_cd'] == store_cd) & 
                (aligned_data['days_from_opening'] == 0)
            ]
            if len(store_data) > 0:
                assert store_data['sales'].iloc[0] == pytest.approx(1.0, abs=0.01)
    
    def test_normalization_mean(self, sample_sales_data, sample_opening_dates):
        """平均値正規化のテスト"""
        config = AlignmentConfig(
            normalize_sales=True,
            normalization_method="mean"
        )
        visualizer = SalesAlignmentVisualizer(config)
        
        aligned_data = visualizer.align_sales_data(
            sample_sales_data,
            sample_opening_dates
        )
        
        # 各店舗の平均が1.0になっている
        for store_cd in sample_sales_data.keys():
            store_data = aligned_data[aligned_data['store_cd'] == store_cd]
            if len(store_data) > 0:
                assert store_data['sales'].mean() == pytest.approx(1.0, abs=0.01)
    
    def test_plot_aligned_sales(self, sample_sales_data, sample_opening_dates):
        """売上推移プロットのテスト"""
        visualizer = SalesAlignmentVisualizer()
        visualizer.align_sales_data(sample_sales_data, sample_opening_dates)
        
        fig = visualizer.plot_aligned_sales(
            title="Test Plot",
            show_average=True,
            show_individual=True
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0  # トレースが含まれている
        assert fig.layout.title.text == "Test Plot"
    
    def test_plot_with_filter_days(self, sample_sales_data, sample_opening_dates):
        """期間フィルタリング付きプロットのテスト"""
        visualizer = SalesAlignmentVisualizer()
        visualizer.align_sales_data(sample_sales_data, sample_opening_dates)
        
        fig = visualizer.plot_aligned_sales(
            filter_days=(0, 30)  # 最初の30日間
        )
        
        # プロットされたデータが30日以内
        for trace in fig.data:
            if hasattr(trace, 'x') and trace.x is not None:
                assert max(trace.x) <= 30
    
    def test_plot_with_highlight(self, sample_sales_data, sample_opening_dates):
        """強調表示付きプロットのテスト"""
        config = AlignmentConfig(
            highlight_stores=['store_000', 'store_002'],
            reference_stores=['store_001']
        )
        visualizer = SalesAlignmentVisualizer(config)
        visualizer.align_sales_data(sample_sales_data, sample_opening_dates)
        
        fig = visualizer.plot_aligned_sales()
        
        # 強調表示とリファレンス店舗が異なるスタイルで描画される
        assert len(fig.data) > 0
    
    def test_plot_comparison_matrix(self, sample_sales_data, sample_opening_dates):
        """類似性マトリックスプロットのテスト"""
        visualizer = SalesAlignmentVisualizer()
        visualizer.align_sales_data(sample_sales_data, sample_opening_dates)
        
        fig = visualizer.plot_comparison_matrix(
            metric="correlation",
            period_days=60
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        # ヒートマップが含まれている
        assert any(isinstance(trace, go.Heatmap) for trace in fig.data)
    
    def test_plot_growth_patterns(self, sample_sales_data, sample_opening_dates):
        """成長パターンプロットのテスト"""
        visualizer = SalesAlignmentVisualizer()
        visualizer.align_sales_data(sample_sales_data, sample_opening_dates)
        
        fig = visualizer.plot_growth_patterns(
            period_days=90,
            n_clusters=3
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_create_interactive_dashboard(self, sample_sales_data, sample_opening_dates):
        """インタラクティブダッシュボードのテスト"""
        visualizer = SalesAlignmentVisualizer()
        visualizer.align_sales_data(sample_sales_data, sample_opening_dates)
        
        fig = visualizer.create_interactive_dashboard(
            include_plots=['timeline', 'distribution']
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
    
    def test_export_aligned_data_csv(self, sample_sales_data, sample_opening_dates, tmp_path):
        """CSV形式でのエクスポートテスト"""
        visualizer = SalesAlignmentVisualizer()
        visualizer.align_sales_data(sample_sales_data, sample_opening_dates)
        
        csv_path = tmp_path / "test_export.csv"
        visualizer.export_aligned_data(
            filepath=csv_path,
            format="csv",
            include_metadata=True
        )
        
        assert csv_path.exists()
        
        # メタデータファイルも作成される
        metadata_path = tmp_path / "test_export_metadata.csv"
        assert metadata_path.exists()
    
    def test_export_aligned_data_json(self, sample_sales_data, sample_opening_dates, tmp_path):
        """JSON形式でのエクスポートテスト"""
        visualizer = SalesAlignmentVisualizer()
        visualizer.align_sales_data(sample_sales_data, sample_opening_dates)
        
        json_path = tmp_path / "test_export.json"
        visualizer.export_aligned_data(
            filepath=json_path,
            format="json",
            include_metadata=True
        )
        
        assert json_path.exists()
        
        # JSONの内容を確認
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        assert 'aligned_sales' in data
        assert 'config' in data
        assert 'metadata' in data
    
    def test_plot_without_alignment(self):
        """アラインメント前のプロットでエラーになることを確認"""
        visualizer = SalesAlignmentVisualizer()
        
        with pytest.raises(ValueError, match="No aligned data"):
            visualizer.plot_aligned_sales()
    
    def test_empty_data_handling(self):
        """空データの処理テスト"""
        visualizer = SalesAlignmentVisualizer()
        
        empty_data = {}
        aligned_data = visualizer.align_sales_data(empty_data)
        
        assert isinstance(aligned_data, pd.DataFrame)
        assert len(aligned_data) == 0
    
    def test_single_store_handling(self):
        """単一店舗データの処理テスト"""
        visualizer = SalesAlignmentVisualizer()
        
        single_store_data = {
            'store_001': np.array([100000, 105000, 98000, 102000])
        }
        opening_dates = {
            'store_001': datetime(2023, 1, 1)
        }
        
        aligned_data = visualizer.align_sales_data(
            single_store_data,
            opening_dates
        )
        
        assert len(aligned_data) == 4
        assert aligned_data['store_cd'].unique()[0] == 'store_001'
    
    def test_custom_color_palette(self):
        """カスタムカラーパレットのテスト"""
        config = AlignmentConfig(
            color_palette=['#FF0000', '#00FF00', '#0000FF']
        )
        visualizer = SalesAlignmentVisualizer(config)
        
        assert visualizer.config.color_palette == ['#FF0000', '#00FF00', '#0000FF']
    
    def test_invalid_export_format(self, sample_sales_data, sample_opening_dates, tmp_path):
        """無効なエクスポート形式のテスト"""
        visualizer = SalesAlignmentVisualizer()
        visualizer.align_sales_data(sample_sales_data, sample_opening_dates)
        
        invalid_path = tmp_path / "test_export.invalid"
        
        with pytest.raises(ValueError, match="Unsupported format"):
            visualizer.export_aligned_data(
                filepath=invalid_path,
                format="invalid"
            )