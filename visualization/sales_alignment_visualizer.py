"""
開店日基準の売上推移可視化モジュール

店舗の開店日を基準に揃えて売上推移を比較可能にする
インタラクティブな可視化機能を提供する。
"""

from typing import Dict, List, Optional, Union, Any, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
from dataclasses import dataclass
from pathlib import Path

from ..config import TIME_SERIES_CONSTANTS, STATISTICS_CONSTANTS, PLOT_CONSTANTS
from ..config.visualization import PLOTLY_DEFAULTS, VISUALIZATION_LIMITS, PLOT_DEFAULTS
from ..config.defaults import DATA_FORMAT_DEFAULTS


@dataclass
class AlignmentConfig:
    """アラインメント設定"""
    reference_point: str = "opening_date"  # 基準点（"opening_date", "first_sale", "custom"）
    max_days_before: int = 0  # 基準日より前の表示日数
    max_days_after: int = TIME_SERIES_CONSTANTS['DAYS_IN_YEAR']  # 基準日より後の表示日数
    aggregate_method: str = "mean"  # 集計方法（"mean", "median", "sum"）
    show_confidence_band: bool = True  # 信頼区間を表示
    confidence_level: float = 0.95  # 信頼水準
    
    # 表示設定
    theme: str = PLOTLY_DEFAULTS['template']  # Plotlyテーマ
    color_palette: Optional[List[str]] = None  # カラーパレット
    line_width: float = PLOT_DEFAULTS['line_width']
    marker_size: int = PLOT_DEFAULTS['marker_size']
    
    # インタラクティブ機能
    enable_zoom: bool = True
    enable_hover: bool = True
    show_annotations: bool = True
    
    # 比較設定
    highlight_stores: Optional[List[str]] = None  # 強調表示する店舗
    reference_stores: Optional[List[str]] = None  # 基準線として表示する店舗
    
    # 正規化設定
    normalize_sales: bool = False  # 売上を正規化
    normalization_method: str = "max"  # "max", "opening_day", "mean"


class SalesAlignmentVisualizer:
    """
    開店日基準の売上推移可視化クラス
    
    複数店舗の売上データを開店日で揃えて比較可能にし、
    インタラクティブな可視化を提供する。
    """
    
    def __init__(self, config: Optional[AlignmentConfig] = None):
        """
        Parameters
        ----------
        config : AlignmentConfig, optional
            可視化設定
        """
        self.config = config or AlignmentConfig()
        self._aligned_data = None
        self._store_metadata = {}
        
        # デフォルトカラーパレット
        if self.config.color_palette is None:
            self.config.color_palette = px.colors.qualitative.Set3
    
    def align_sales_data(
        self,
        sales_data: Union[pd.DataFrame, Dict[str, np.ndarray]],
        opening_dates: Optional[Dict[str, datetime]] = None,
        store_attributes: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        売上データを基準日で揃える
        
        Parameters
        ----------
        sales_data : pd.DataFrame or dict
            店舗別売上データ
        opening_dates : dict, optional
            店舗CDと開店日の辞書
        store_attributes : pd.DataFrame, optional
            店舗属性データ（開店日を含む場合）
            
        Returns
        -------
        pd.DataFrame
            アラインメント済みデータ（days_from_opening × stores）
        """
        # データ形式の統一
        if isinstance(sales_data, dict):
            # 空の辞書の場合は空のDataFrameを返す
            if not sales_data:
                return pd.DataFrame()
            
            # 最大長に合わせてDataFrame作成
            max_len = max(len(data) for data in sales_data.values())
            df_data = {}
            for store_cd, data in sales_data.items():
                # 短いデータは末尾をNaNで埋める
                if len(data) < max_len:
                    padded = np.pad(data, (0, max_len - len(data)), 
                                  mode='constant', constant_values=np.nan)
                    df_data[store_cd] = padded
                else:
                    df_data[store_cd] = data[:max_len]
            
            sales_df = pd.DataFrame(df_data)
        else:
            sales_df = sales_data.copy()
        
        # 開店日の取得
        if opening_dates is None:
            opening_dates = self._extract_opening_dates(sales_df, store_attributes)
        
        # アラインメント実行
        aligned_data = []
        store_info = []
        
        for store_cd in sales_df.columns:
            if store_cd not in opening_dates:
                # 開店日が不明な場合は最初の売上日を使用
                first_sale_idx = sales_df[store_cd].first_valid_index()
                if first_sale_idx is None:
                    warnings.warn(f"Store {store_cd} has no valid sales data")
                    continue
                opening_date = sales_df.index[0] if hasattr(sales_df, 'index') else 0
            else:
                opening_date = opening_dates[store_cd]
            
            # 店舗データの抽出とアラインメント
            store_sales = sales_df[store_cd].values
            
            # 開店日からの日数を計算
            days_from_opening = np.arange(len(store_sales))
            
            # データフレームに追加
            for day, sales in enumerate(store_sales):
                if not np.isnan(sales):
                    aligned_data.append({
                        'store_cd': store_cd,
                        'days_from_opening': day - self.config.max_days_before,
                        'sales': sales,
                        'opening_date': opening_date
                    })
            
            # 店舗情報を保存
            store_info.append({
                'store_cd': store_cd,
                'opening_date': opening_date,
                'total_days': len(store_sales),
                'first_sale': np.nanmin(store_sales),
                'max_sale': np.nanmax(store_sales),
                'mean_sale': np.nanmean(store_sales)
            })
        
        # DataFrameに変換
        self._aligned_data = pd.DataFrame(aligned_data)
        self._store_metadata = pd.DataFrame(store_info)
        
        # 正規化処理
        if self.config.normalize_sales:
            self._normalize_aligned_data()
        
        return self._aligned_data
    
    def plot_aligned_sales(
        self,
        title: str = "店舗売上推移比較（開店日基準）",
        show_average: bool = True,
        show_individual: bool = True,
        filter_days: Optional[Tuple[int, int]] = None,
    ) -> go.Figure:
        """
        アラインメントされた売上推移をプロット
        
        Parameters
        ----------
        title : str
            グラフタイトル
        show_average : bool, default=True
            平均線を表示
        show_individual : bool, default=True
            個別店舗の線を表示
        filter_days : tuple, optional
            表示する日数範囲（開店日からの日数）
            
        Returns
        -------
        go.Figure
            Plotlyフィギュア
        """
        if self._aligned_data is None:
            raise ValueError("No aligned data. Call align_sales_data() first.")
        
        fig = go.Figure()
        
        # 日数範囲のフィルタリング
        plot_data = self._aligned_data.copy()
        if filter_days:
            plot_data = plot_data[
                (plot_data['days_from_opening'] >= filter_days[0]) &
                (plot_data['days_from_opening'] <= filter_days[1])
            ]
        
        # 個別店舗のプロット
        if show_individual:
            stores = plot_data['store_cd'].unique()
            colors = self.config.color_palette
            
            for i, store_cd in enumerate(stores):
                store_data = plot_data[plot_data['store_cd'] == store_cd]
                
                # 強調表示の判定
                is_highlighted = (
                    self.config.highlight_stores and 
                    store_cd in self.config.highlight_stores
                )
                
                # 基準線の判定
                is_reference = (
                    self.config.reference_stores and
                    store_cd in self.config.reference_stores
                )
                
                # スタイル設定
                line_width = self.config.line_width * 1.5 if is_highlighted else self.config.line_width
                opacity = 1.0 if is_highlighted else PLOT_DEFAULTS['alpha']
                dash = 'dash' if is_reference else 'solid'
                
                fig.add_trace(go.Scatter(
                    x=store_data['days_from_opening'],
                    y=store_data['sales'],
                    name=store_cd,
                    mode='lines+markers' if is_highlighted else 'lines',
                    line=dict(
                        color=colors[i % len(colors)],
                        width=line_width,
                        dash=dash
                    ),
                    opacity=opacity,
                    marker=dict(size=self.config.marker_size if is_highlighted else 0),
                    hovertemplate=(
                        f"<b>{store_cd}</b><br>" +
                        "開店後日数: %{x}日<br>" +
                        "売上: ¥%{y:,.0f}<br>" +
                        "<extra></extra>"
                    )
                ))
        
        # 平均線の追加
        if show_average:
            avg_data = plot_data.groupby('days_from_opening')['sales'].agg(['mean', 'std', 'count'])
            
            # 信頼区間の計算
            if self.config.show_confidence_band and len(avg_data) > 0:
                z_score = STATISTICS_CONSTANTS['CONFIDENCE_MULTIPLIER'] if self.config.confidence_level == 0.95 else 2.58
                avg_data['ci_lower'] = avg_data['mean'] - z_score * avg_data['std'] / np.sqrt(avg_data['count'])
                avg_data['ci_upper'] = avg_data['mean'] + z_score * avg_data['std'] / np.sqrt(avg_data['count'])
                
                # 信頼区間の塗りつぶし
                fig.add_trace(go.Scatter(
                    x=avg_data.index,
                    y=avg_data['ci_upper'],
                    fill=None,
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    showlegend=False,
                    hoverinfo='skip'
                ))
                
                fig.add_trace(go.Scatter(
                    x=avg_data.index,
                    y=avg_data['ci_lower'],
                    fill='tonexty',
                    mode='lines',
                    line_color='rgba(0,0,0,0)',
                    fillcolor='rgba(100,100,100,0.2)',
                    name=f'{int(self.config.confidence_level*100)}%信頼区間',
                    hoverinfo='skip'
                ))
            
            # 平均線
            fig.add_trace(go.Scatter(
                x=avg_data.index,
                y=avg_data['mean'],
                name='平均',
                mode='lines',
                line=dict(color='black', width=3, dash='solid'),
                hovertemplate=(
                    "<b>平均</b><br>" +
                    "開店後日数: %{x}日<br>" +
                    "平均売上: ¥%{y:,.0f}<br>" +
                    f"店舗数: %{{customdata}}<br>" +
                    "<extra></extra>"
                ),
                customdata=avg_data['count']
            ))
        
        # レイアウト設定
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=PLOT_DEFAULTS['font_size']['title'])
            ),
            xaxis=dict(
                title="開店からの日数",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                zeroline=True,
                zerolinewidth=2,
                zerolinecolor='rgba(128,128,128,0.5)'
            ),
            yaxis=dict(
                title="売上（円）" if not self.config.normalize_sales else "売上（正規化）",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                tickformat=',.0f'
            ),
            hovermode='x unified',
            template=self.config.theme,
            height=PLOTLY_DEFAULTS['height'],
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.01
            )
        )
        
        # アノテーションの追加
        if self.config.show_annotations:
            self._add_annotations(fig, plot_data)
        
        return fig
    
    def plot_comparison_matrix(
        self,
        metric: str = "correlation",
        period_days: int = 90,
    ) -> go.Figure:
        """
        店舗間の類似性マトリックスを表示
        
        Parameters
        ----------
        metric : str, default='correlation'
            比較指標（'correlation', 'dtw', 'rmse'）
        period_days : int, default=90
            比較期間（開店後日数）
            
        Returns
        -------
        go.Figure
            ヒートマップ
        """
        if self._aligned_data is None:
            raise ValueError("No aligned data. Call align_sales_data() first.")
        
        # 期間でフィルタリング
        period_data = self._aligned_data[
            self._aligned_data['days_from_opening'] <= period_days
        ]
        
        # 店舗ごとにピボット
        pivot_data = period_data.pivot_table(
            index='days_from_opening',
            columns='store_cd',
            values='sales',
            aggfunc='mean'
        )
        
        # 類似性マトリックスの計算
        stores = pivot_data.columns
        n_stores = len(stores)
        similarity_matrix = np.zeros((n_stores, n_stores))
        
        for i, store1 in enumerate(stores):
            for j, store2 in enumerate(stores):
                if metric == "correlation":
                    # 相関係数
                    corr = pivot_data[store1].corr(pivot_data[store2])
                    similarity_matrix[i, j] = corr if not np.isnan(corr) else 0
                elif metric == "rmse":
                    # RMSE（小さいほど類似）
                    diff = pivot_data[store1] - pivot_data[store2]
                    rmse = np.sqrt(np.nanmean(diff**2))
                    similarity_matrix[i, j] = -rmse  # 負の値で表示
                else:
                    # DTW（実装簡略化のため相関係数を使用）
                    similarity_matrix[i, j] = pivot_data[store1].corr(pivot_data[store2])
        
        # ヒートマップ作成
        fig = go.Figure(data=go.Heatmap(
            z=similarity_matrix,
            x=stores,
            y=stores,
            colorscale='RdBu',
            zmid=0,
            text=np.round(similarity_matrix, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hovertemplate=(
                "店舗1: %{y}<br>" +
                "店舗2: %{x}<br>" +
                f"{metric}: %{{z:.3f}}<br>" +
                "<extra></extra>"
            )
        ))
        
        fig.update_layout(
            title=f"店舗間類似性マトリックス（{metric}、開店後{period_days}日間）",
            xaxis=dict(title="店舗CD", side="bottom"),
            yaxis=dict(title="店舗CD", autorange="reversed"),
            template=self.config.theme,
            height=PLOTLY_DEFAULTS['height'],
            width=700
        )
        
        return fig
    
    def plot_growth_patterns(
        self,
        period_days: int = 90,
        n_clusters: int = 3,
    ) -> go.Figure:
        """
        成長パターンの分析と可視化
        
        Parameters
        ----------
        period_days : int, default=90
            分析期間
        n_clusters : int, default=3
            クラスタ数
            
        Returns
        -------
        go.Figure
            成長パターンのプロット
        """
        if self._aligned_data is None:
            raise ValueError("No aligned data. Call align_sales_data() first.")
        
        # 期間データの準備
        period_data = self._aligned_data[
            self._aligned_data['days_from_opening'] <= period_days
        ]
        
        # 成長率の計算
        growth_rates = []
        for store_cd in period_data['store_cd'].unique():
            store_data = period_data[period_data['store_cd'] == store_cd].sort_values('days_from_opening')
            
            if len(store_data) >= 7:
                # 週次成長率
                week1_avg = store_data[store_data['days_from_opening'] <= 7]['sales'].mean()
                last_week_avg = store_data[store_data['days_from_opening'] >= period_days-7]['sales'].mean()
                
                if week1_avg > 0:
                    growth_rate = (last_week_avg - week1_avg) / week1_avg
                    growth_rates.append({
                        'store_cd': store_cd,
                        'growth_rate': growth_rate,
                        'initial_sales': week1_avg,
                        'final_sales': last_week_avg
                    })
        
        growth_df = pd.DataFrame(growth_rates)
        
        # 簡易クラスタリング（成長率でグループ化）
        if len(growth_df) >= n_clusters:
            growth_df['cluster'] = pd.qcut(growth_df['growth_rate'], n_clusters, labels=['低成長', '中成長', '高成長'][:n_clusters])
        else:
            growth_df['cluster'] = '全店舗'
        
        # サブプロット作成
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('成長率分布', '成長パターン別売上推移'),
            specs=[[{"type": "scatter"}, {"type": "scatter"}]]
        )
        
        # 1. 散布図（初期売上 vs 成長率）
        colors = px.colors.qualitative.Set2
        for i, cluster in enumerate(growth_df['cluster'].unique()):
            cluster_data = growth_df[growth_df['cluster'] == cluster]
            
            fig.add_trace(
                go.Scatter(
                    x=cluster_data['initial_sales'],
                    y=cluster_data['growth_rate'] * 100,  # パーセント表示
                    mode='markers+text',
                    name=str(cluster),
                    text=cluster_data['store_cd'],
                    textposition="top center",
                    marker=dict(size=10, color=colors[i % len(colors)]),
                    hovertemplate=(
                        "<b>%{text}</b><br>" +
                        "初期売上: ¥%{x:,.0f}<br>" +
                        "成長率: %{y:.1f}%<br>" +
                        "<extra></extra>"
                    )
                ),
                row=1, col=1
            )
        
        # 2. クラスタ別平均推移
        for i, cluster in enumerate(growth_df['cluster'].unique()):
            cluster_stores = growth_df[growth_df['cluster'] == cluster]['store_cd'].values
            cluster_sales = period_data[period_data['store_cd'].isin(cluster_stores)]
            
            avg_sales = cluster_sales.groupby('days_from_opening')['sales'].mean()
            
            fig.add_trace(
                go.Scatter(
                    x=avg_sales.index,
                    y=avg_sales.values,
                    mode='lines',
                    name=f"{cluster}平均",
                    line=dict(width=3, color=colors[i % len(colors)]),
                ),
                row=1, col=2
            )
        
        # レイアウト更新
        fig.update_xaxes(title_text="初期売上（週平均）", row=1, col=1)
        fig.update_yaxes(title_text="成長率（%）", row=1, col=1)
        fig.update_xaxes(title_text="開店からの日数", row=1, col=2)
        fig.update_yaxes(title_text="売上（円）", row=1, col=2)
        
        fig.update_layout(
            title=f"店舗成長パターン分析（開店後{period_days}日間）",
            template=self.config.theme,
            height=500,
            showlegend=True
        )
        
        return fig
    
    def create_interactive_dashboard(
        self,
        include_plots: List[str] = None,
    ) -> go.Figure:
        """
        インタラクティブダッシュボードを作成
        
        Parameters
        ----------
        include_plots : list, optional
            含めるプロットのリスト
            
        Returns
        -------
        go.Figure
            ダッシュボード
        """
        if self._aligned_data is None:
            raise ValueError("No aligned data. Call align_sales_data() first.")
        
        if include_plots is None:
            include_plots = ['timeline', 'distribution', 'heatmap']
        
        n_plots = len(include_plots)
        
        # サブプロットの設定
        if n_plots == 1:
            specs = [[{"type": "scatter"}]]
            rows, cols = 1, 1
        elif n_plots == 2:
            specs = [[{"type": "scatter"}, {"type": "scatter"}]]
            rows, cols = 1, 2
        elif n_plots == 3:
            specs = [[{"type": "scatter", "colspan": 2}, None],
                    [{"type": "violin"}, {"type": "heatmap"}]]
            rows, cols = 2, 2
        else:
            specs = [[{"type": "scatter"}, {"type": "scatter"}],
                    [{"type": "violin"}, {"type": "heatmap"}]]
            rows, cols = 2, 2
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[p.title() for p in include_plots],
            specs=specs,
            row_heights=[0.6, 0.4] if rows > 1 else None
        )
        
        # 各プロットの追加
        plot_idx = 0
        
        if 'timeline' in include_plots:
            # メインの時系列プロット
            self._add_timeline_to_subplot(fig, row=1, col=1)
            plot_idx += 1
        
        if 'distribution' in include_plots:
            # 売上分布
            row = 1 if plot_idx < 2 else 2
            col = 2 if plot_idx == 1 else 1
            self._add_distribution_to_subplot(fig, row=row, col=col)
            plot_idx += 1
        
        if 'heatmap' in include_plots:
            # 週次ヒートマップ
            row = 2
            col = 2 if plot_idx == 2 else 1
            self._add_heatmap_to_subplot(fig, row=row, col=col)
        
        # レイアウト設定
        fig.update_layout(
            title="売上分析ダッシュボード",
            template=self.config.theme,
            height=int(PLOTLY_DEFAULTS['height'] * 1.33),
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig
    
    def export_aligned_data(
        self,
        filepath: Union[str, Path],
        format: str = "csv",
        include_metadata: bool = True,
    ):
        """
        アラインメントされたデータをエクスポート
        
        Parameters
        ----------
        filepath : str or Path
            出力ファイルパス
        format : str, default='csv'
            出力形式（'csv', 'excel', 'json'）
        include_metadata : bool, default=True
            メタデータを含めるか
        """
        if self._aligned_data is None:
            raise ValueError("No aligned data to export.")
        
        filepath = Path(filepath)
        
        if format == "csv":
            self._aligned_data.to_csv(filepath, index=False)
            if include_metadata and self._store_metadata is not None:
                metadata_path = filepath.with_name(f"{filepath.stem}_metadata.csv")
                self._store_metadata.to_csv(metadata_path, index=False)
                
        elif format == "excel":
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                self._aligned_data.to_excel(writer, sheet_name='AlignedSales', index=False)
                if include_metadata and self._store_metadata is not None:
                    self._store_metadata.to_excel(writer, sheet_name='StoreMetadata', index=False)
                    
        elif format == "json":
            export_data = {
                'aligned_sales': self._aligned_data.to_dict('records'),
                'config': {
                    'reference_point': self.config.reference_point,
                    'normalize_sales': self.config.normalize_sales,
                    'normalization_method': self.config.normalization_method
                }
            }
            if include_metadata and self._store_metadata is not None:
                export_data['metadata'] = self._store_metadata.to_dict('records')
            
            import json
            with open(filepath, 'w', encoding=DATA_FORMAT_DEFAULTS['encoding']) as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _extract_opening_dates(
        self,
        sales_df: pd.DataFrame,
        store_attributes: Optional[pd.DataFrame]
    ) -> Dict[str, datetime]:
        """開店日を抽出"""
        opening_dates = {}
        
        # 店舗属性から取得
        if store_attributes is not None and 'opening_date' in store_attributes.columns:
            for store_cd in store_attributes.index:
                if store_cd in sales_df.columns:
                    opening_dates[store_cd] = pd.to_datetime(store_attributes.loc[store_cd, 'opening_date'])
        
        # 売上データから推定（最初の売上日）
        for store_cd in sales_df.columns:
            if store_cd not in opening_dates:
                first_sale_idx = sales_df[store_cd].first_valid_index()
                if first_sale_idx is not None:
                    opening_dates[store_cd] = first_sale_idx if isinstance(first_sale_idx, datetime) else datetime.now()
        
        return opening_dates
    
    def _normalize_aligned_data(self):
        """アラインメントされたデータを正規化"""
        if self.config.normalization_method == "max":
            # 各店舗の最大値で正規化
            for store_cd in self._aligned_data['store_cd'].unique():
                mask = self._aligned_data['store_cd'] == store_cd
                max_val = self._aligned_data.loc[mask, 'sales'].max()
                if max_val > 0:
                    self._aligned_data.loc[mask, 'sales'] /= max_val
                    
        elif self.config.normalization_method == "opening_day":
            # 開店日の売上で正規化
            for store_cd in self._aligned_data['store_cd'].unique():
                mask = self._aligned_data['store_cd'] == store_cd
                opening_sales = self._aligned_data.loc[
                    mask & (self._aligned_data['days_from_opening'] == 0), 'sales'
                ].values
                if len(opening_sales) > 0 and opening_sales[0] > 0:
                    self._aligned_data.loc[mask, 'sales'] /= opening_sales[0]
                    
        elif self.config.normalization_method == "mean":
            # 平均値で正規化
            for store_cd in self._aligned_data['store_cd'].unique():
                mask = self._aligned_data['store_cd'] == store_cd
                mean_val = self._aligned_data.loc[mask, 'sales'].mean()
                if mean_val > 0:
                    self._aligned_data.loc[mask, 'sales'] /= mean_val
    
    def _add_annotations(self, fig: go.Figure, plot_data: pd.DataFrame):
        """アノテーションを追加"""
        # 開店日マーカー
        fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_annotation(
            x=0, y=1, yref="paper",
            text="開店日",
            showarrow=False,
            yshift=10
        )
        
        # マイルストーンマーカー
        milestones = [
            TIME_SERIES_CONSTANTS['RECENT_DAYS_WINDOW'], 
            TIME_SERIES_CONSTANTS['QUARTER_DAYS'], 
            TIME_SERIES_CONSTANTS['HALF_YEAR_DAYS'], 
            TIME_SERIES_CONSTANTS['DAYS_IN_YEAR']
        ]
        for milestone in milestones:
            if plot_data['days_from_opening'].max() >= milestone:
                fig.add_vline(x=milestone, line_dash="dot", line_color="lightgray", opacity=0.3)
                fig.add_annotation(
                    x=milestone, y=0, yref="paper",
                    text=f"{milestone}日",
                    showarrow=False,
                    yshift=-20
                )
    
    def _add_timeline_to_subplot(self, fig: go.Figure, row: int, col: int):
        """時系列プロットをサブプロットに追加"""
        stores = self._aligned_data['store_cd'].unique()[:10]  # 最大10店舗
        colors = self.config.color_palette
        
        for i, store_cd in enumerate(stores):
            store_data = self._aligned_data[self._aligned_data['store_cd'] == store_cd]
            
            fig.add_trace(
                go.Scatter(
                    x=store_data['days_from_opening'],
                    y=store_data['sales'],
                    name=store_cd,
                    mode='lines',
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=True
                ),
                row=row, col=col
            )
    
    def _add_distribution_to_subplot(self, fig: go.Figure, row: int, col: int):
        """売上分布をサブプロットに追加"""
        # 30日目の売上分布
        day30_data = self._aligned_data[
            self._aligned_data['days_from_opening'].between(25, 35)
        ]
        
        fig.add_trace(
            go.Violin(
                y=day30_data['sales'],
                name='30日目売上分布',
                box_visible=True,
                meanline_visible=True,
                showlegend=False
            ),
            row=row, col=col
        )
    
    def _add_heatmap_to_subplot(self, fig: go.Figure, row: int, col: int):
        """週次ヒートマップをサブプロットに追加"""
        # 週次集計
        self._aligned_data['week'] = self._aligned_data['days_from_opening'] // 7
        weekly_pivot = self._aligned_data.pivot_table(
            index='store_cd',
            columns='week',
            values='sales',
            aggfunc='mean'
        ).iloc[:VISUALIZATION_LIMITS['max_legend_items'], :12]  # 最大表示店舗数、12週間
        
        fig.add_trace(
            go.Heatmap(
                z=weekly_pivot.values,
                x=[f"第{w+1}週" for w in weekly_pivot.columns],
                y=weekly_pivot.index,
                colorscale='Viridis',
                showscale=True
            ),
            row=row, col=col
        )