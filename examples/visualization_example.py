"""
開店日基準の売上推移可視化の使用例

SalesAlignmentVisualizerを使用して、複数店舗の売上を
開店日で揃えて比較する例を示す。
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# TwinStoreのインポート
from twinstore.visualization.sales_alignment_visualizer import (
    SalesAlignmentVisualizer, 
    AlignmentConfig
)


def generate_store_data_with_patterns():
    """異なる成長パターンを持つ店舗データを生成"""
    np.random.seed(42)
    
    # 店舗の設定
    stores = {
        'store_001': {
            'opening_date': datetime(2023, 1, 15),
            'pattern': 'rapid_growth',  # 急成長型
            'base_sales': 80000,
            'growth_rate': 0.015,  # 日次1.5%成長
        },
        'store_002': {
            'opening_date': datetime(2023, 2, 1),
            'pattern': 'steady',  # 安定型
            'base_sales': 100000,
            'growth_rate': 0.002,
        },
        'store_003': {
            'opening_date': datetime(2023, 3, 10),
            'pattern': 'seasonal',  # 季節変動型
            'base_sales': 90000,
            'growth_rate': 0.003,
        },
        'store_004': {
            'opening_date': datetime(2023, 1, 1),
            'pattern': 'declining',  # 減少型
            'base_sales': 120000,
            'growth_rate': -0.001,
        },
        'store_005': {
            'opening_date': datetime(2023, 4, 1),
            'pattern': 'volatile',  # 変動型
            'base_sales': 95000,
            'growth_rate': 0.005,
        }
    }
    
    # 売上データの生成
    sales_data = {}
    opening_dates = {}
    
    for store_cd, config in stores.items():
        opening_date = config['opening_date']
        opening_dates[store_cd] = opening_date
        
        # 365日分のデータを生成
        n_days = 365
        days = np.arange(n_days)
        
        # 基本売上
        base = config['base_sales']
        
        # 成長トレンド
        growth = base * (1 + config['growth_rate']) ** days
        
        # パターン別の変動を追加
        if config['pattern'] == 'rapid_growth':
            # S字カーブ成長
            growth = base / (1 + np.exp(-0.05 * (days - 60))) * 2
            noise = np.random.normal(0, base * 0.05, n_days)
            
        elif config['pattern'] == 'seasonal':
            # 季節変動（30日周期）
            seasonal = base * 0.2 * np.sin(days * 2 * np.pi / 30)
            growth = growth + seasonal
            noise = np.random.normal(0, base * 0.03, n_days)
            
        elif config['pattern'] == 'volatile':
            # 高変動
            noise = np.random.normal(0, base * 0.15, n_days)
            # 週末効果
            weekend_effect = np.array([
                1.3 if i % 7 in [5, 6] else 1.0 for i in days
            ])
            growth = growth * weekend_effect
            
        else:
            # 通常のノイズ
            noise = np.random.normal(0, base * 0.05, n_days)
        
        # 最終的な売上
        sales = growth + noise
        sales = np.maximum(sales, 0)  # 負の値を除去
        
        sales_data[store_cd] = sales
    
    # 店舗属性
    store_attributes = pd.DataFrame([
        {'store_cd': sid, 
         'opening_date': info['opening_date'],
         'pattern': info['pattern'],
         'initial_sales': info['base_sales']}
        for sid, info in stores.items()
    ]).set_index('store_cd')
    
    return sales_data, opening_dates, store_attributes


def example_1_basic_alignment():
    """例1: 基本的なアラインメントと可視化"""
    print("=" * 60)
    print("例1: 基本的なアラインメントと可視化")
    print("=" * 60)
    
    # データ生成
    sales_data, opening_dates, store_attributes = generate_store_data_with_patterns()
    
    # ビジュアライザーの作成
    visualizer = SalesAlignmentVisualizer()
    
    # データのアラインメント
    aligned_data = visualizer.align_sales_data(
        sales_data,
        opening_dates,
        store_attributes
    )
    
    print(f"\nアラインメント完了:")
    print(f"- 店舗数: {aligned_data['store_cd'].nunique()}")
    print(f"- データポイント数: {len(aligned_data)}")
    print(f"- 期間: {aligned_data['days_from_opening'].min()} ~ {aligned_data['days_from_opening'].max()}日")
    
    # 基本的な可視化
    fig = visualizer.plot_aligned_sales(
        title="店舗売上推移比較（開店日基準）",
        show_average=True,
        show_individual=True
    )
    
    # 表示（実際の環境では fig.show()）
    print("\n→ 開店日基準の売上推移グラフを生成しました")
    
    # HTMLファイルとして保存
    fig.write_html("aligned_sales_basic.html")
    print("→ aligned_sales_basic.html として保存しました")


def example_2_normalized_comparison():
    """例2: 正規化した比較"""
    print("\n" + "=" * 60)
    print("例2: 正規化した比較")
    print("=" * 60)
    
    # データ生成
    sales_data, opening_dates, store_attributes = generate_store_data_with_patterns()
    
    # 正規化設定でビジュアライザーを作成
    config = AlignmentConfig(
        normalize_sales=True,
        normalization_method="opening_day",  # 開店日の売上で正規化
        show_confidence_band=True,
        confidence_level=0.95
    )
    
    visualizer = SalesAlignmentVisualizer(config)
    
    # アラインメントと正規化
    aligned_data = visualizer.align_sales_data(sales_data, opening_dates)
    
    # 可視化（90日間に限定）
    fig = visualizer.plot_aligned_sales(
        title="正規化売上推移（開店日=1.0）- 最初の90日間",
        filter_days=(0, 90)
    )
    
    print("\n→ 正規化された売上推移グラフを生成しました")
    print("  - 各店舗の開店日売上を1.0として正規化")
    print("  - 成長率の比較が容易に")
    
    fig.write_html("aligned_sales_normalized.html")


def example_3_pattern_analysis():
    """例3: 成長パターン分析"""
    print("\n" + "=" * 60)
    print("例3: 成長パターン分析")
    print("=" * 60)
    
    # データ生成
    sales_data, opening_dates, store_attributes = generate_store_data_with_patterns()
    
    # ビジュアライザー作成
    visualizer = SalesAlignmentVisualizer()
    visualizer.align_sales_data(sales_data, opening_dates, store_attributes)
    
    # 成長パターンの分析
    fig = visualizer.plot_growth_patterns(
        period_days=90,
        n_clusters=3
    )
    
    print("\n→ 成長パターン分析グラフを生成しました")
    print("  - 初期売上と成長率の関係")
    print("  - クラスタ別の平均推移")
    
    fig.write_html("growth_patterns.html")


def example_4_similarity_matrix():
    """例4: 店舗間類似性マトリックス"""
    print("\n" + "=" * 60)
    print("例4: 店舗間類似性マトリックス")
    print("=" * 60)
    
    # データ生成
    sales_data, opening_dates, store_attributes = generate_store_data_with_patterns()
    
    # ビジュアライザー作成
    visualizer = SalesAlignmentVisualizer()
    visualizer.align_sales_data(sales_data, opening_dates)
    
    # 類似性マトリックスの生成
    fig = visualizer.plot_comparison_matrix(
        metric="correlation",
        period_days=60
    )
    
    print("\n→ 店舗間類似性マトリックスを生成しました")
    print("  - 開店後60日間の売上推移の相関")
    print("  - 類似パターンを持つ店舗の特定")
    
    fig.write_html("similarity_matrix.html")


def example_5_interactive_dashboard():
    """例5: インタラクティブダッシュボード"""
    print("\n" + "=" * 60)
    print("例5: インタラクティブダッシュボード")
    print("=" * 60)
    
    # データ生成
    sales_data, opening_dates, store_attributes = generate_store_data_with_patterns()
    
    # ダッシュボード用の設定
    config = AlignmentConfig(
        show_annotations=True,
        enable_zoom=True,
        enable_hover=True,
        theme="plotly_white"
    )
    
    visualizer = SalesAlignmentVisualizer(config)
    visualizer.align_sales_data(sales_data, opening_dates, store_attributes)
    
    # ダッシュボードの作成
    fig = visualizer.create_interactive_dashboard(
        include_plots=['timeline', 'distribution', 'heatmap']
    )
    
    print("\n→ インタラクティブダッシュボードを生成しました")
    print("  - 時系列プロット")
    print("  - 売上分布")
    print("  - 週次ヒートマップ")
    
    fig.write_html("interactive_dashboard.html")


def example_6_custom_styling():
    """例6: カスタムスタイリング"""
    print("\n" + "=" * 60)
    print("例6: カスタムスタイリング")
    print("=" * 60)
    
    # データ生成
    sales_data, opening_dates, store_attributes = generate_store_data_with_patterns()
    
    # カスタム設定
    config = AlignmentConfig(
        # 特定店舗を強調
        highlight_stores=['store_001', 'store_003'],
        # 基準線として表示
        reference_stores=['store_002'],
        # カスタムカラー
        color_palette=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8'],
        # スタイル設定
        line_width=2.5,
        marker_size=8,
        theme="plotly_dark"
    )
    
    visualizer = SalesAlignmentVisualizer(config)
    visualizer.align_sales_data(sales_data, opening_dates)
    
    # カスタムスタイルでプロット
    fig = visualizer.plot_aligned_sales(
        title="カスタムスタイル売上推移",
        filter_days=(0, 180)  # 最初の180日間
    )
    
    # 追加のカスタマイズ
    fig.add_annotation(
        x=30, y=150000,
        text="30日目チェックポイント",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#FF6B6B",
        font=dict(size=14, color="#FF6B6B")
    )
    
    print("\n→ カスタムスタイルのグラフを生成しました")
    print("  - 特定店舗の強調表示")
    print("  - 基準線の表示")
    print("  - カスタムカラーパレット")
    
    fig.write_html("custom_styled_sales.html")


def example_7_export_data():
    """例7: データのエクスポート"""
    print("\n" + "=" * 60)
    print("例7: データのエクスポート")
    print("=" * 60)
    
    # データ生成
    sales_data, opening_dates, store_attributes = generate_store_data_with_patterns()
    
    # ビジュアライザー作成
    visualizer = SalesAlignmentVisualizer()
    visualizer.align_sales_data(sales_data, opening_dates, store_attributes)
    
    # 各形式でエクスポート
    formats = ['csv', 'excel', 'json']
    
    for fmt in formats:
        filename = f"aligned_sales_data.{fmt}"
        if fmt == 'excel':
            filename = "aligned_sales_data.xlsx"
        
        try:
            visualizer.export_aligned_data(
                filepath=filename,
                format=fmt,
                include_metadata=True
            )
            print(f"\n→ {filename} としてエクスポートしました")
        except Exception as e:
            print(f"\n→ {fmt}形式のエクスポートをスキップ: {e}")


def example_8_real_world_scenario():
    """例8: 実際のビジネスシナリオ"""
    print("\n" + "=" * 60)
    print("例8: 実際のビジネスシナリオ")
    print("=" * 60)
    
    # より現実的なデータを生成
    np.random.seed(123)
    
    # 20店舗のデータ
    n_stores = 20
    store_types = ['urban', 'suburban', 'mall', 'roadside']
    
    sales_data = {}
    opening_dates = {}
    store_info = []
    
    base_date = datetime(2022, 1, 1)
    
    for i in range(n_stores):
        store_cd = f'FC{i+1:03d}'
        store_type = np.random.choice(store_types)
        
        # 開店日（過去2年間でランダム）
        days_offset = np.random.randint(0, 730)
        opening_date = base_date + timedelta(days=days_offset)
        opening_dates[store_cd] = opening_date
        
        # 店舗タイプ別の基本売上
        base_sales = {
            'urban': 150000,
            'suburban': 100000,
            'mall': 130000,
            'roadside': 110000
        }[store_type]
        
        # 売上データ生成（開店から現在まで）
        days_open = (datetime.now() - opening_date).days
        days_open = min(days_open, 500)  # 最大500日
        
        if days_open > 0:
            # 初期の立ち上がり
            ramp_up = 1 / (1 + np.exp(-0.1 * (np.arange(days_open) - 30)))
            
            # 季節変動
            seasonal = 0.2 * np.sin(np.arange(days_open) * 2 * np.pi / 365 + np.random.rand() * 2 * np.pi)
            
            # 週次パターン
            weekly = np.array([
                1.0 if i % 7 < 5 else 1.2 
                for i in range(days_open)
            ])
            
            # 売上計算
            sales = base_sales * ramp_up * (1 + seasonal) * weekly
            sales += np.random.normal(0, base_sales * 0.05, days_open)
            sales = np.maximum(sales, 0)
            
            sales_data[store_cd] = sales
            
            store_info.append({
                'store_cd': store_cd,
                'store_type': store_type,
                'opening_date': opening_date,
                'days_open': days_open,
                'avg_sales': np.mean(sales[-30:]) if len(sales) >= 30 else np.mean(sales)
            })
    
    store_attributes = pd.DataFrame(store_info).set_index('store_cd')
    
    # 分析設定
    config = AlignmentConfig(
        normalize_sales=False,
        show_confidence_band=True,
        highlight_stores=[f'FC{i:03d}' for i in [1, 5, 10]],  # 特定店舗を強調
        show_annotations=True
    )
    
    visualizer = SalesAlignmentVisualizer(config)
    visualizer.align_sales_data(sales_data, opening_dates, store_attributes)
    
    # 複合分析図の作成
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            '全店舗売上推移（開店日基準）',
            '店舗タイプ別平均推移',
            '30日目売上分布',
            '成長率ランキング'
        ),
        specs=[[{"type": "scatter"}, {"type": "scatter"}],
               [{"type": "box"}, {"type": "bar"}]],
        row_heights=[0.6, 0.4]
    )
    
    # 1. 全店舗推移（サンプル）
    sample_stores = list(sales_data.keys())[:10]
    for store_cd in sample_stores:
        store_data = visualizer._aligned_data[visualizer._aligned_data['store_cd'] == store_cd]
        fig.add_trace(
            go.Scatter(
                x=store_data['days_from_opening'],
                y=store_data['sales'],
                name=store_cd,
                mode='lines',
                opacity=0.7,
                showlegend=False
            ),
            row=1, col=1
        )
    
    # 2. 店舗タイプ別平均
    for store_type in store_types:
        type_stores = store_attributes[store_attributes['store_type'] == store_type].index
        type_data = visualizer._aligned_data[visualizer._aligned_data['store_cd'].isin(type_stores)]
        
        if len(type_data) > 0:
            avg_by_day = type_data.groupby('days_from_opening')['sales'].mean()
            fig.add_trace(
                go.Scatter(
                    x=avg_by_day.index,
                    y=avg_by_day.values,
                    name=store_type,
                    mode='lines',
                    line=dict(width=3)
                ),
                row=1, col=2
            )
    
    # 3. 30日目売上分布（店舗タイプ別）
    day30_data = []
    for store_type in store_types:
        type_stores = store_attributes[store_attributes['store_type'] == store_type].index
        type_sales = visualizer._aligned_data[
            (visualizer._aligned_data['store_cd'].isin(type_stores)) &
            (visualizer._aligned_data['days_from_opening'].between(25, 35))
        ]['sales']
        
        if len(type_sales) > 0:
            fig.add_trace(
                go.Box(
                    y=type_sales,
                    name=store_type,
                    showlegend=False
                ),
                row=2, col=1
            )
    
    # 4. 成長率ランキング
    growth_rates = []
    for store_cd in store_attributes.index:
        store_data = visualizer._aligned_data[visualizer._aligned_data['store_cd'] == store_cd]
        if len(store_data) >= 60:
            week1_avg = store_data[store_data['days_from_opening'] <= 7]['sales'].mean()
            week8_avg = store_data[store_data['days_from_opening'].between(50, 60)]['sales'].mean()
            if week1_avg > 0:
                growth = (week8_avg - week1_avg) / week1_avg * 100
                growth_rates.append({'store_cd': store_cd, 'growth_rate': growth})
    
    if growth_rates:
        growth_df = pd.DataFrame(growth_rates).sort_values('growth_rate', ascending=True).tail(10)
        fig.add_trace(
            go.Bar(
                x=growth_df['growth_rate'],
                y=growth_df['store_cd'],
                orientation='h',
                text=[f"{g:.1f}%" for g in growth_df['growth_rate']],
                textposition='outside',
                showlegend=False
            ),
            row=2, col=2
        )
    
    # レイアウト設定
    fig.update_layout(
        title="フランチャイズチェーン売上分析ダッシュボード",
        height=900,
        showlegend=True,
        template="plotly_white"
    )
    
    fig.update_xaxes(title_text="開店からの日数", row=1, col=1)
    fig.update_xaxes(title_text="開店からの日数", row=1, col=2)
    fig.update_xaxes(title_text="成長率（%）", row=2, col=2)
    
    fig.update_yaxes(title_text="売上（円）", row=1, col=1)
    fig.update_yaxes(title_text="売上（円）", row=1, col=2)
    fig.update_yaxes(title_text="売上（円）", row=2, col=1)
    
    print("\n→ ビジネス分析ダッシュボードを生成しました")
    print(f"  - 分析店舗数: {len(store_attributes)}店舗")
    print(f"  - 店舗タイプ: {', '.join(store_types)}")
    print("  - 開店日基準の比較分析")
    
    fig.write_html("business_dashboard.html")


if __name__ == "__main__":
    # 全ての例を実行
    examples = [
        example_1_basic_alignment,
        example_2_normalized_comparison,
        example_3_pattern_analysis,
        example_4_similarity_matrix,
        example_5_interactive_dashboard,
        example_6_custom_styling,
        example_7_export_data,
        example_8_real_world_scenario
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
    print("生成されたHTMLファイルをブラウザで開いて確認してください")
    print("=" * 60)