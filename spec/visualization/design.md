# 可視化機能 設計仕様書

**作成日**: 2025-07-23  
**関連要件**: [requirements.md](./requirements.md)  

## 設計概要

可視化機能は**Adapter Pattern**と**Factory Pattern**を組み合わせて設計されています。異なる可視化ライブラリ（Matplotlib、Plotly）を統一的なインターフェースで扱い、用途に応じて最適な可視化を提供します。

## アーキテクチャ設計

### クラス構成

```python
# 可視化基盤
class Visualizer(ABC):
    """可視化の抽象基底クラス"""
    @abstractmethod
    def plot(self, data: Any) -> VisualizationResult
    @abstractmethod
    def export(self, format: str) -> bytes
    
class SalesVisualizer:
    """売上データ可視化のメインクラス"""
    def __init__(self, config: VisualizationConfig)
    def plot_prediction(self, result: PredictionResult) -> VisualizationResult
    def plot_comparison(self, stores: List[Dict]) -> VisualizationResult
    def plot_quality_metrics(self, metrics: Dict) -> VisualizationResult
    
class VisualizerFactory:
    """可視化器のファクトリークラス"""
    def create_visualizer(self, type: str, backend: str) -> Visualizer
    
# グラフアダプター
class MatplotlibAdapter(Visualizer):
    """Matplotlib用アダプター"""
    def plot(self, data: Any) -> VisualizationResult
    def export(self, format: str) -> bytes
    
class PlotlyAdapter(Visualizer):
    """Plotly用アダプター"""
    def plot(self, data: Any) -> VisualizationResult
    def export(self, format: str) -> bytes

# ダッシュボード
class Dashboard:
    """ダッシュボードクラス"""
    def __init__(self, config: DashboardConfig)
    def add_widget(self, widget: DashboardWidget)
    def render(self) -> Union[str, bytes]
    def update(self, data: Dict)
```

### コンポーネント構成

```
Visualizer (Abstract)
    ├── MatplotlibAdapter
    │   ├── LineChart
    │   ├── BarChart
    │   └── HeatMap
    ├── PlotlyAdapter
    │   ├── InteractiveLineChart
    │   ├── 3DScatterPlot
    │   └── SankeyDiagram
    └── ReportGenerator
        ├── PDFReport
        ├── ExcelReport
        └── HTMLReport
```

## 詳細設計

### 1. SalesVisualizer クラス

```python
class SalesVisualizer:
    """売上データ可視化のメインクラス"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.factory = VisualizerFactory()
        self.theme_manager = ThemeManager(config.theme)
        
    def plot_prediction(self, 
                       result: PredictionResult,
                       show_confidence: bool = True,
                       show_anomalies: bool = True) -> VisualizationResult:
        """予測結果の可視化"""
        
        # バックエンド選択
        backend = self._select_backend(self.config.interactive)
        visualizer = self.factory.create_visualizer('line', backend)
        
        # データ準備
        plot_data = self._prepare_prediction_data(result)
        
        # プロット設定
        plot_config = {
            'title': 'Sales Prediction Analysis',
            'xlabel': 'Date',
            'ylabel': 'Sales',
            'show_confidence': show_confidence,
            'show_anomalies': show_anomalies,
            'theme': self.theme_manager.get_theme()
        }
        
        # 可視化実行
        viz_result = visualizer.plot(plot_data, **plot_config)
        
        # 注釈追加
        if show_confidence:
            self._add_confidence_band(viz_result, result.confidence_interval)
        
        if show_anomalies and hasattr(result, 'anomaly_flags'):
            self._add_anomaly_markers(viz_result, result.anomaly_flags)
        
        return viz_result
    
    def plot_comparison(self, 
                       target_store: Dict,
                       similar_stores: List[Dict],
                       alignment_info: Dict = None) -> VisualizationResult:
        """類似店舗との比較可視化"""
        
        backend = self._select_backend(True)  # 比較は常にインタラクティブ
        visualizer = self.factory.create_visualizer('multi_line', backend)
        
        # データ準備
        comparison_data = self._prepare_comparison_data(
            target_store, similar_stores, alignment_info
        )
        
        # カラーパレット設定
        colors = self.theme_manager.get_color_palette(len(similar_stores) + 1)
        
        # プロット設定
        plot_config = {
            'title': 'Store Sales Comparison',
            'xlabel': 'Days from Opening',
            'ylabel': 'Sales',
            'colors': colors,
            'legend_position': 'best',
            'line_styles': self._get_line_styles(len(similar_stores) + 1)
        }
        
        # 可視化実行
        viz_result = visualizer.plot(comparison_data, **plot_config)
        
        # インタラクティブ要素追加
        if backend == 'plotly':
            self._add_hover_info(viz_result, comparison_data)
            self._add_zoom_controls(viz_result)
        
        return viz_result
    
    def plot_quality_metrics(self, 
                           quality_report: QualityReport) -> VisualizationResult:
        """品質メトリクスの可視化"""
        
        visualizer = self.factory.create_visualizer('radar', 'matplotlib')
        
        # レーダーチャート用データ準備
        metrics_data = {
            'Completeness': quality_report.completeness_score,
            'Consistency': quality_report.consistency_score,
            'Accuracy': quality_report.accuracy_score,
            'Timeliness': quality_report.timeliness_score
        }
        
        # プロット設定
        plot_config = {
            'title': 'Data Quality Metrics',
            'fill': True,
            'alpha': 0.3,
            'color': self.theme_manager.get_primary_color()
        }
        
        # 可視化実行
        viz_result = visualizer.plot(metrics_data, **plot_config)
        
        # スコアカード追加
        self._add_score_card(viz_result, quality_report.overall_score)
        
        return viz_result
```

### 2. MatplotlibAdapter クラス

```python
class MatplotlibAdapter(Visualizer):
    """Matplotlib用アダプター"""
    
    def __init__(self):
        self.figure = None
        self.axes = None
        
    def plot(self, data: Any, **kwargs) -> VisualizationResult:
        """Matplotlibでのプロット実行"""
        
        # Figure作成
        figsize = kwargs.get('figsize', (10, 6))
        self.figure, self.axes = plt.subplots(figsize=figsize)
        
        # テーマ適用
        if 'theme' in kwargs:
            self._apply_theme(kwargs['theme'])
        
        # プロットタイプ別処理
        plot_type = kwargs.get('plot_type', 'line')
        
        if plot_type == 'line':
            self._plot_line(data, **kwargs)
        elif plot_type == 'bar':
            self._plot_bar(data, **kwargs)
        elif plot_type == 'heatmap':
            self._plot_heatmap(data, **kwargs)
        elif plot_type == 'radar':
            self._plot_radar(data, **kwargs)
        
        # タイトル・ラベル設定
        self._set_labels(**kwargs)
        
        # レイアウト調整
        self.figure.tight_layout()
        
        return VisualizationResult(
            figure=self.figure,
            data=data,
            config=kwargs,
            export_formats=['png', 'jpg', 'svg', 'pdf']
        )
    
    def _plot_line(self, data: Dict, **kwargs):
        """折れ線グラフの描画"""
        
        for label, values in data.items():
            style = kwargs.get('line_styles', {}).get(label, '-')
            color = kwargs.get('colors', {}).get(label, None)
            
            self.axes.plot(values, label=label, linestyle=style, color=color)
        
        # 信頼区間表示
        if kwargs.get('show_confidence', False):
            self._add_confidence_interval(data, **kwargs)
        
        # 凡例
        if len(data) > 1:
            self.axes.legend(loc=kwargs.get('legend_position', 'best'))
    
    def export(self, format: str, **kwargs) -> bytes:
        """グラフのエクスポート"""
        
        if self.figure is None:
            raise ValueError("No figure to export")
        
        buffer = io.BytesIO()
        
        if format in ['png', 'jpg', 'jpeg']:
            dpi = kwargs.get('dpi', 300)
            self.figure.savefig(buffer, format=format, dpi=dpi, 
                              bbox_inches='tight')
        elif format == 'svg':
            self.figure.savefig(buffer, format='svg', bbox_inches='tight')
        elif format == 'pdf':
            self.figure.savefig(buffer, format='pdf', bbox_inches='tight')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        buffer.seek(0)
        return buffer.getvalue()
```

### 3. PlotlyAdapter クラス

```python
class PlotlyAdapter(Visualizer):
    """Plotly用アダプター"""
    
    def __init__(self):
        self.figure = None
        
    def plot(self, data: Any, **kwargs) -> VisualizationResult:
        """Plotlyでのプロット実行"""
        
        plot_type = kwargs.get('plot_type', 'line')
        
        if plot_type == 'line':
            self.figure = self._create_line_plot(data, **kwargs)
        elif plot_type == 'multi_line':
            self.figure = self._create_multi_line_plot(data, **kwargs)
        elif plot_type == '3d_scatter':
            self.figure = self._create_3d_scatter(data, **kwargs)
        elif plot_type == 'sankey':
            self.figure = self._create_sankey(data, **kwargs)
        
        # レイアウト設定
        self._update_layout(**kwargs)
        
        # インタラクティブ機能追加
        self._add_interactivity(**kwargs)
        
        return VisualizationResult(
            figure=self.figure,
            data=data,
            config=kwargs,
            export_formats=['png', 'html', 'json']
        )
    
    def _create_line_plot(self, data: Dict, **kwargs):
        """インタラクティブ折れ線グラフ"""
        
        traces = []
        
        for label, values in data.items():
            trace = go.Scatter(
                x=list(range(len(values))),
                y=values,
                mode='lines+markers',
                name=label,
                line=dict(
                    color=kwargs.get('colors', {}).get(label),
                    width=2
                ),
                hovertemplate='Day %{x}<br>Sales: %{y:,.0f}<extra></extra>'
            )
            traces.append(trace)
        
        figure = go.Figure(data=traces)
        
        # 信頼区間追加
        if kwargs.get('show_confidence', False):
            self._add_confidence_band(figure, data, **kwargs)
        
        return figure
    
    def _add_interactivity(self, **kwargs):
        """インタラクティブ機能の追加"""
        
        # ズーム・パン機能
        self.figure.update_xaxes(
            rangeslider_visible=kwargs.get('rangeslider', False),
            rangeselector=self._get_range_selector() if kwargs.get('range_selector', False) else None
        )
        
        # ボタン追加
        if kwargs.get('add_buttons', False):
            self.figure.update_layout(
                updatemenus=[self._create_update_menu()]
            )
        
        # ホバーモード
        self.figure.update_layout(
            hovermode=kwargs.get('hovermode', 'x unified')
        )
    
    def export(self, format: str, **kwargs) -> bytes:
        """グラフのエクスポート"""
        
        if format == 'png':
            return self.figure.to_image(format='png', width=kwargs.get('width', 1200), 
                                      height=kwargs.get('height', 800))
        elif format == 'html':
            return self.figure.to_html(include_plotlyjs='cdn')
        elif format == 'json':
            return self.figure.to_json()
        else:
            raise ValueError(f"Unsupported format: {format}")
```

### 4. Dashboard クラス

```python
class Dashboard:
    """ダッシュボードクラス"""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.widgets = []
        self.layout = config.layout
        self.update_interval = config.update_interval
        self.data_sources = {}
        
    def add_widget(self, widget: DashboardWidget):
        """ウィジェット追加"""
        
        # 位置検証
        if not self._validate_position(widget.position):
            raise ValueError(f"Invalid widget position: {widget.position}")
        
        self.widgets.append(widget)
        
        # データソース登録
        if widget.data_source:
            self.data_sources[widget.id] = widget.data_source
    
    def render(self) -> Union[str, bytes]:
        """ダッシュボードのレンダリング"""
        
        if self.config.backend == 'plotly':
            return self._render_plotly_dashboard()
        elif self.config.backend == 'matplotlib':
            return self._render_matplotlib_dashboard()
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
    
    def _render_plotly_dashboard(self) -> str:
        """Plotlyダッシュボードのレンダリング"""
        
        import plotly.subplots as sp
        
        # サブプロット作成
        fig = sp.make_subplots(
            rows=self.layout.rows,
            cols=self.layout.cols,
            subplot_titles=[w.title for w in self.widgets],
            specs=self._get_subplot_specs()
        )
        
        # 各ウィジェットの描画
        for widget in self.widgets:
            row, col = widget.position
            
            # データ取得
            data = self._fetch_widget_data(widget)
            
            # ウィジェットタイプ別描画
            if widget.type == 'kpi_card':
                self._add_kpi_card(fig, data, row, col)
            elif widget.type == 'line_chart':
                self._add_line_chart(fig, data, row, col)
            elif widget.type == 'heatmap':
                self._add_heatmap(fig, data, row, col)
            elif widget.type == 'ranking':
                self._add_ranking_table(fig, data, row, col)
        
        # レイアウト更新
        fig.update_layout(
            title=self.config.title,
            showlegend=True,
            height=self.layout.height,
            width=self.layout.width,
            template=self.config.theme
        )
        
        # HTML生成
        return fig.to_html(include_plotlyjs='cdn', div_id='dashboard')
    
    def update(self, data: Dict):
        """ダッシュボードデータの更新"""
        
        for widget_id, new_data in data.items():
            if widget_id in self.data_sources:
                self.data_sources[widget_id] = new_data
        
        # 自動更新が有効な場合
        if self.config.auto_refresh:
            self._schedule_refresh()
```

### 5. ReportGenerator クラス

```python
class ReportGenerator:
    """レポート生成クラス"""
    
    def __init__(self, config: ReportConfig):
        self.config = config
        self.template_engine = TemplateEngine(config.template_path)
        
    def generate_pdf_report(self, 
                          prediction_results: List[PredictionResult],
                          visualizations: List[VisualizationResult]) -> bytes:
        """PDFレポート生成"""
        
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table
        
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        
        # ストーリー構築
        story = []
        
        # タイトルページ
        story.extend(self._create_title_page())
        
        # エグゼクティブサマリー
        story.extend(self._create_executive_summary(prediction_results))
        
        # 詳細結果セクション
        for i, (result, viz) in enumerate(zip(prediction_results, visualizations)):
            story.extend(self._create_result_section(i + 1, result, viz))
        
        # 付録
        story.extend(self._create_appendix())
        
        # PDF生成
        doc.build(story)
        buffer.seek(0)
        
        return buffer.getvalue()
    
    def generate_excel_report(self, 
                            prediction_results: List[PredictionResult],
                            raw_data: Dict) -> bytes:
        """Excelレポート生成"""
        
        buffer = io.BytesIO()
        
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # サマリーシート
            summary_df = self._create_summary_dataframe(prediction_results)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # 詳細結果シート
            for i, result in enumerate(prediction_results):
                sheet_name = f'Store_{i+1}'
                detail_df = self._create_detail_dataframe(result)
                detail_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # グラフ追加
                worksheet = writer.sheets[sheet_name]
                chart = self._create_excel_chart(workbook, detail_df)
                worksheet.insert_chart('H2', chart)
            
            # 生データシート
            if raw_data:
                raw_df = pd.DataFrame(raw_data)
                raw_df.to_excel(writer, sheet_name='Raw_Data', index=False)
            
            # フォーマット適用
            self._apply_excel_formatting(writer)
        
        buffer.seek(0)
        return buffer.getvalue()
```

### 6. テーマ管理

```python
class ThemeManager:
    """テーマ管理クラス"""
    
    def __init__(self, theme_name: str = 'default'):
        self.theme_name = theme_name
        self.themes = self._load_themes()
        self.current_theme = self.themes.get(theme_name, self.themes['default'])
        
    def get_theme(self) -> Dict:
        """現在のテーマ取得"""
        return self.current_theme
    
    def get_color_palette(self, n_colors: int) -> List[str]:
        """カラーパレット取得"""
        
        base_colors = self.current_theme['colors']['palette']
        
        if n_colors <= len(base_colors):
            return base_colors[:n_colors]
        else:
            # 色の補間
            import matplotlib.cm as cm
            cmap = cm.get_cmap(self.current_theme['colors']['colormap'])
            return [cmap(i / n_colors) for i in range(n_colors)]
    
    def _load_themes(self) -> Dict:
        """テーマ定義の読み込み"""
        
        return {
            'default': {
                'colors': {
                    'primary': '#1f77b4',
                    'secondary': '#ff7f0e',
                    'success': '#2ca02c',
                    'danger': '#d62728',
                    'warning': '#ff9800',
                    'info': '#17a2b8',
                    'palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
                    'colormap': 'tab10'
                },
                'fonts': {
                    'family': 'sans-serif',
                    'size': {
                        'title': 16,
                        'label': 12,
                        'tick': 10
                    }
                },
                'grid': {
                    'show': True,
                    'alpha': 0.3,
                    'linestyle': '--'
                }
            },
            'dark': {
                'colors': {
                    'primary': '#00d9ff',
                    'secondary': '#ff6ec7',
                    'background': '#1a1a1a',
                    'text': '#ffffff',
                    'palette': ['#00d9ff', '#ff6ec7', '#00ff88', '#ffaa00'],
                    'colormap': 'plasma'
                },
                'fonts': {
                    'family': 'monospace',
                    'size': {
                        'title': 18,
                        'label': 14,
                        'tick': 11
                    }
                }
            }
        }
```

## テスト設計

### 単体テスト

```python
class TestSalesVisualizer:
    def test_plot_prediction(self):
        """予測結果プロットのテスト"""
        
    def test_plot_comparison(self):
        """比較プロットのテスト"""
        
    def test_export_formats(self):
        """エクスポート形式のテスト"""

class TestDashboard:
    def test_widget_management(self):
        """ウィジェット管理のテスト"""
        
    def test_dashboard_rendering(self):
        """ダッシュボードレンダリングのテスト"""
```

## 関連ドキュメント

- [要件仕様書](./requirements.md)
- [実装タスク](./tasks.md)
- [全体アーキテクチャ](../architecture.md)