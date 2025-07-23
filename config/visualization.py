"""
可視化関連の設定
"""

# プロット全般の設定
PLOT_DEFAULTS = {
    'figure_size': (12, 6),
    'dpi': 100,
    'font_size': {
        'title': 16,
        'axis_label': 12,
        'tick_label': 10,
        'legend': 10,
    },
    'line_width': 2,
    'marker_size': 6,
    'alpha': 0.8,
    'grid': True,
    'grid_alpha': 0.3,
}

# カラーパレット
COLOR_PALETTES = {
    'default': [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ],
    'pastel': [
        '#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5',
        '#C49C94', '#F7B6D2', '#C7C7C7', '#DBDB8D', '#9EDAE5'
    ],
    'dark': [
        '#1b4f72', '#935116', '#186a3b', '#922b21', '#5b2c6f',
        '#6e2c00', '#a93226', '#424949', '#7d6608', '#0e6655'
    ],
    'business': [
        '#003f5c', '#2f4b7c', '#665191', '#a05195', '#d45087',
        '#f95d6a', '#ff7c43', '#ffa600', '#7f7f7f', '#c7c7c7'
    ],
}

# プロットタイプ別の設定
PLOT_TYPE_CONFIGS = {
    'line': {
        'style': '-',
        'marker': 'o',
        'markersize': 6,
        'linewidth': 2,
        'alpha': 0.8,
    },
    'scatter': {
        'marker': 'o',
        'size': 50,
        'alpha': 0.6,
        'edgecolor': 'white',
        'linewidth': 0.5,
    },
    'bar': {
        'width': 0.8,
        'alpha': 0.8,
        'edgecolor': 'black',
        'linewidth': 1,
    },
    'heatmap': {
        'cmap': 'RdBu_r',
        'center': 0,
        'vmin': -1,
        'vmax': 1,
        'square': True,
        'linewidths': 0.5,
        'cbar_kws': {'shrink': 0.8},
    },
}

# インタラクティブプロット（Plotly）の設定
PLOTLY_DEFAULTS = {
    'template': 'plotly_white',
    'height': 600,
    'width': 1000,
    'margin': {'l': 80, 'r': 80, 't': 100, 'b': 80},
    'hovermode': 'x unified',
    'showlegend': True,
    'legend': {
        'orientation': 'v',
        'yanchor': 'top',
        'y': 0.99,
        'xanchor': 'left',
        'x': 1.01,
    },
    'xaxis': {
        'showgrid': True,
        'gridcolor': 'lightgray',
        'gridwidth': 0.5,
    },
    'yaxis': {
        'showgrid': True,
        'gridcolor': 'lightgray',
        'gridwidth': 0.5,
    },
}

# ダッシュボードの設定
DASHBOARD_CONFIG = {
    'layout': {
        'rows': 2,
        'cols': 2,
        'subplot_titles': True,
        'vertical_spacing': 0.1,
        'horizontal_spacing': 0.1,
    },
    'components': {
        'sales_comparison': {'row': 1, 'col': 1},
        'growth_analysis': {'row': 1, 'col': 2},
        'similarity_matrix': {'row': 2, 'col': 1},
        'statistics_table': {'row': 2, 'col': 2},
    },
}

# アニメーション設定
ANIMATION_CONFIG = {
    'duration': 500,  # ミリ秒
    'easing': 'cubic-in-out',
    'frame_duration': 100,
    'transition_duration': 300,
}

# エクスポート設定
EXPORT_CONFIG = {
    'formats': ['png', 'pdf', 'svg', 'html'],
    'png': {
        'width': 1200,
        'height': 600,
        'scale': 2,
    },
    'pdf': {
        'width': 8.5,
        'height': 11,
        'orientation': 'landscape',
    },
}

# 日本語フォント設定
JAPANESE_FONT_CONFIG = {
    'family': 'Meiryo UI',
    'fallback': ['Hiragino Sans', 'Yu Gothic', 'MS Gothic', 'sans-serif'],
    'size': 12,
}

# テーマ設定
THEMES = {
    'light': {
        'background': 'white',
        'grid': 'lightgray',
        'text': 'black',
        'accent': '#1f77b4',
    },
    'dark': {
        'background': '#1e1e1e',
        'grid': '#333333',
        'text': 'white',
        'accent': '#00bfff',
    },
    'corporate': {
        'background': '#f8f9fa',
        'grid': '#dee2e6',
        'text': '#212529',
        'accent': '#0066cc',
    },
}

# 制限値
VISUALIZATION_LIMITS = {
    'max_points_display': 1000,      # 表示する最大データポイント数
    'max_stores_heatmap': 50,        # ヒートマップの最大店舗数
    'max_legend_items': 10,          # 凡例の最大項目数
    'downsample_threshold': 5000,    # ダウンサンプリングの閾値
}

def get_color_palette(name: str = 'default') -> list:
    """カラーパレットを取得"""
    return COLOR_PALETTES.get(name, COLOR_PALETTES['default'])

def get_plot_config(plot_type: str = 'line') -> dict:
    """プロットタイプ別の設定を取得"""
    base_config = PLOT_DEFAULTS.copy()
    type_config = PLOT_TYPE_CONFIGS.get(plot_type, {})
    base_config.update(type_config)
    return base_config

def get_theme_config(theme: str = 'light') -> dict:
    """テーマ設定を取得"""
    return THEMES.get(theme, THEMES['light'])