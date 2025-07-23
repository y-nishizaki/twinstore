# TwinStore 可視化ガイド

## 開店日基準の売上推移可視化

SalesAlignmentVisualizerは、複数店舗の売上データを開店日で揃えて比較可能にする可視化機能です。

## 主な特徴

- **開店日アラインメント**: 各店舗の開店日を基準（0日）として売上推移を揃える
- **インタラクティブ可視化**: Plotlyベースの対話的なグラフ
- **正規化オプション**: 売上規模の違いを吸収して成長率を比較
- **パターン分析**: 成長パターンの自動分類と可視化
- **類似性分析**: 店舗間の売上推移の類似性を評価

## 基本的な使い方

### 1. シンプルな例

```python
from twinstore import SalesAlignmentVisualizer

# ビジュアライザーの作成
visualizer = SalesAlignmentVisualizer()

# データのアラインメント
# sales_data: 店舗別売上データ（辞書またはDataFrame）
# opening_dates: 店舗コードと開店日の辞書
aligned_data = visualizer.align_sales_data(
    sales_data,
    opening_dates={'store_001': datetime(2023, 1, 1), ...}
)

# 可視化
fig = visualizer.plot_aligned_sales(
    title="店舗売上推移比較",
    show_average=True
)
fig.show()  # ブラウザで表示
```

### 2. 正規化した比較

```python
from twinstore import AlignmentConfig

# 正規化設定
config = AlignmentConfig(
    normalize_sales=True,
    normalization_method="opening_day"  # 開店日の売上を1.0に
)

visualizer = SalesAlignmentVisualizer(config)
aligned_data = visualizer.align_sales_data(sales_data, opening_dates)

# 成長率の比較が容易に
fig = visualizer.plot_aligned_sales(
    title="正規化売上推移（開店日=1.0）",
    filter_days=(0, 90)  # 最初の90日間
)
```

## AlignmentConfig 設定オプション

### 基本設定

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `reference_point` | "opening_date" | 基準点（"opening_date", "first_sale", "custom"） |
| `max_days_before` | 0 | 基準日より前の表示日数 |
| `max_days_after` | 365 | 基準日より後の表示日数 |
| `aggregate_method` | "mean" | 集計方法（"mean", "median", "sum"） |

### 正規化設定

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `normalize_sales` | False | 売上を正規化するか |
| `normalization_method` | "max" | 正規化方法（"max", "opening_day", "mean"） |

### 表示設定

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `show_confidence_band` | True | 信頼区間を表示 |
| `confidence_level` | 0.95 | 信頼水準 |
| `theme` | "plotly_white" | Plotlyテーマ |
| `highlight_stores` | None | 強調表示する店舗リスト |
| `reference_stores` | None | 基準線として表示する店舗 |

## 主要メソッド

### 1. align_sales_data()
売上データを開店日で揃える

```python
aligned_data = visualizer.align_sales_data(
    sales_data,        # 売上データ（DataFrame or dict）
    opening_dates,     # 開店日辞書（optional）
    store_attributes   # 店舗属性DataFrame（optional）
)
```

### 2. plot_aligned_sales()
アラインメントされた売上推移をプロット

```python
fig = visualizer.plot_aligned_sales(
    title="タイトル",
    show_average=True,      # 平均線を表示
    show_individual=True,   # 個別店舗線を表示
    filter_days=(0, 180)    # 表示期間の指定
)
```

### 3. plot_growth_patterns()
成長パターンの分析

```python
fig = visualizer.plot_growth_patterns(
    period_days=90,    # 分析期間
    n_clusters=3       # クラスタ数
)
```

### 4. plot_comparison_matrix()
店舗間類似性マトリックス

```python
fig = visualizer.plot_comparison_matrix(
    metric="correlation",  # 比較指標
    period_days=60        # 比較期間
)
```

### 5. create_interactive_dashboard()
複合ダッシュボード

```python
fig = visualizer.create_interactive_dashboard(
    include_plots=['timeline', 'distribution', 'heatmap']
)
```

## 実践的な使用例

### 例1: フランチャイズチェーンの分析

```python
# 店舗タイプ別に色分けして表示
config = AlignmentConfig(
    highlight_stores=high_performing_stores,
    reference_stores=benchmark_stores,
    show_annotations=True
)

visualizer = SalesAlignmentVisualizer(config)

# 店舗属性を含めてアラインメント
aligned_data = visualizer.align_sales_data(
    sales_data,
    store_attributes=store_attributes  # type, area等を含む
)

# フィルタリングして表示
urban_stores = store_attributes[
    store_attributes['type'] == 'urban'
].index.tolist()

config.highlight_stores = urban_stores
fig = visualizer.plot_aligned_sales(
    title="都市型店舗の売上推移",
    filter_days=(0, 180)
)
```

### 例2: A/Bテストの結果分析

```python
# 新レイアウト店舗と従来店舗の比較
config = AlignmentConfig(
    normalize_sales=True,
    normalization_method="opening_day",
    highlight_stores=new_layout_stores,
    reference_stores=traditional_stores,
    color_palette=['#FF6B6B', '#4ECDC4']  # カスタムカラー
)

visualizer = SalesAlignmentVisualizer(config)
aligned_data = visualizer.align_sales_data(sales_data)

# 成長率の違いを可視化
fig = visualizer.plot_growth_patterns(
    period_days=60,
    n_clusters=2  # 新/旧の2グループ
)
```

### 例3: 季節性の分析

```python
# 月次の売上パターンを可視化
visualizer = SalesAlignmentVisualizer()
aligned_data = visualizer.align_sales_data(sales_data)

# カスタム分析
import pandas as pd
aligned_df = pd.DataFrame(aligned_data)
aligned_df['month_from_opening'] = aligned_df['days_from_opening'] // 30

# 月次集計
monthly_avg = aligned_df.groupby(
    ['store_cd', 'month_from_opening']
)['sales'].mean().reset_index()

# Plotlyで独自の可視化
import plotly.express as px
fig = px.line(
    monthly_avg,
    x='month_from_opening',
    y='sales',
    color='store_cd',
    title='月次売上推移（開店月基準）'
)
```

## データのエクスポート

### 各形式でのエクスポート

```python
# CSV形式
visualizer.export_aligned_data(
    filepath="aligned_sales.csv",
    format="csv",
    include_metadata=True
)

# Excel形式（複数シート）
visualizer.export_aligned_data(
    filepath="aligned_sales.xlsx",
    format="excel",
    include_metadata=True
)

# JSON形式
visualizer.export_aligned_data(
    filepath="aligned_sales.json",
    format="json",
    include_metadata=True
)
```

## カスタマイズのヒント

### 1. カスタムカラーパレット

```python
config = AlignmentConfig(
    color_palette=px.colors.qualitative.Set3,
    # または具体的な色指定
    color_palette=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
)
```

### 2. アノテーションの追加

```python
fig = visualizer.plot_aligned_sales()

# カスタムアノテーション
fig.add_annotation(
    x=30, y=100000,
    text="30日目チェックポイント",
    showarrow=True,
    arrowhead=2
)

# マイルストーンライン
for day in [7, 30, 90]:
    fig.add_vline(x=day, line_dash="dot", opacity=0.3)
```

### 3. インタラクティブ機能の活用

```python
# ズーム・パン機能
config = AlignmentConfig(
    enable_zoom=True,
    enable_hover=True
)

# カスタムホバー情報
fig.update_traces(
    hovertemplate="<b>%{fullData.name}</b><br>" +
                  "開店後: %{x}日<br>" +
                  "売上: ¥%{y:,.0f}<br>" +
                  "<extra></extra>"
)
```

## トラブルシューティング

### Q: グラフが表示されない
```python
# HTMLファイルとして保存
fig.write_html("output.html")

# 画像として保存（要: kaleido）
fig.write_image("output.png")
```

### Q: メモリエラーが発生する
```python
# 表示期間を限定
fig = visualizer.plot_aligned_sales(
    filter_days=(0, 90),  # 90日間のみ
    show_individual=False  # 個別線を非表示
)
```

### Q: 開店日が不明な店舗がある
```python
# 最初の売上日を開店日として自動設定される
# または手動で設定
opening_dates = {
    'store_001': pd.to_datetime('2023-01-01'),
    'store_002': pd.to_datetime('2023-02-15'),
    # ...
}
```

## パフォーマンスの最適化

1. **大量店舗の場合**
   - `show_individual=False`で個別線を非表示
   - サンプリングして表示店舗数を制限
   - 集計データのみ表示

2. **長期間データの場合**
   - `filter_days`で表示期間を制限
   - 週次・月次に集計してから表示
   - `aggregate_method`を活用

3. **リアルタイム更新**
   - データを事前にアラインメント
   - 差分更新の実装
   - キャッシュの活用