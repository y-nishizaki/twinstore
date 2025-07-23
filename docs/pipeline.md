# TwinStore パイプライン機能

## 概要

TwinStoreのパイプライン機能は、データの検証、前処理、品質チェック、予測、説明生成までの一連の処理を統合的に実行する機能です。

## 主な特徴

- **統合処理**: データ検証から予測結果の保存まで一連の処理を自動実行
- **柔軟な設定**: 各処理ステップの有効/無効や詳細パラメータを設定可能
- **バッチ処理**: 複数店舗の予測を一括実行
- **エラーハンドリング**: 問題のあるデータでも適切に処理
- **結果の保存**: JSON/CSV形式での自動保存

## 基本的な使い方

### 1. シンプルな使用例

```python
from twinstore import PredictionPipeline

# パイプラインの作成
pipeline = PredictionPipeline()

# 過去データで学習
pipeline.fit(historical_data, store_attributes)

# 新規店舗の予測
result = pipeline.predict(new_store_sales, store_name="新宿店")

# 結果の確認
print(f"予測年間売上: {result.prediction.prediction:,.0f}円")
print(f"信頼度: {result.prediction.confidence_score:.2%}")
```

### 2. カスタム設定での使用

```python
from twinstore import PipelineConfig, PredictionPipeline

# カスタム設定
config = PipelineConfig(
    # 前処理設定
    preprocess_data=True,
    handle_missing=True,
    handle_outliers=True,
    
    # 品質チェック設定
    check_quality=True,
    quality_threshold=80.0,
    
    # 予測設定
    similarity_metric="dtw",
    n_similar_stores=5,
    auto_optimize_period=True,
    
    # 出力設定
    save_results=True,
    output_format="json"
)

# パイプライン作成
pipeline = PredictionPipeline(config)
```

### 3. ビルダーパターンでの構築

```python
from twinstore import PipelineBuilder

# ビルダーを使った構築
pipeline = (PipelineBuilder()
    .with_validation(strict=True)
    .with_preprocessing(handle_missing=True, handle_outliers=True)
    .with_quality_check(threshold=75.0)
    .with_prediction(metric="dtw", n_similar=5)
    .with_explanation(language="ja")
    .with_output(save=True, format="json")
    .build()
)
```

## パイプラインの処理フロー

1. **データ検証** (`validate_input=True`)
   - 入力データの形式チェック
   - 必要なデータ量の確認
   - 数値の妥当性検証

2. **前処理** (`preprocess_data=True`)
   - 欠損値の補完 (`handle_missing=True`)
   - 異常値の処理 (`handle_outliers=True`)
   - データの平滑化 (`smooth_data=True`)

3. **品質チェック** (`check_quality=True`)
   - 完全性スコア
   - 一貫性スコア
   - 正確性スコア
   - 適時性スコア

4. **異常検知**
   - リアルタイム異常検出
   - 異常パターンの分析

5. **最適期間の決定** (`auto_optimize_period=True`)
   - マッチング期間の自動最適化
   - 安定性スコアの評価

6. **予測実行**
   - 類似店舗の検索
   - 年間売上の予測
   - 信頼区間の計算

7. **説明生成** (`generate_explanation=True`)
   - 予測根拠の説明
   - 信頼度要因の分析
   - 注意事項の提示

8. **結果保存** (`save_results=True`)
   - JSON/CSV形式での保存
   - タイムスタンプ付きファイル名

## 設定オプション

### PipelineConfig の主要パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `validate_input` | True | 入力データの検証を実行 |
| `strict_validation` | False | 厳格な検証モード（警告もエラーとして扱う） |
| `preprocess_data` | True | データの前処理を実行 |
| `handle_missing` | True | 欠損値を処理 |
| `handle_outliers` | True | 異常値を処理 |
| `smooth_data` | False | データを平滑化 |
| `check_quality` | True | データ品質をチェック |
| `quality_threshold` | 70.0 | 品質スコアの閾値 |
| `similarity_metric` | "dtw" | 類似性指標（"dtw", "cosine", "correlation"） |
| `normalization_method` | "z-score" | 正規化手法 |
| `n_similar_stores` | 5 | 使用する類似店舗数 |
| `auto_optimize_period` | True | マッチング期間を自動最適化 |
| `generate_explanation` | True | 説明文を生成 |
| `save_results` | False | 結果を保存 |
| `output_format` | "json" | 出力形式（"json", "csv"） |

## バッチ処理

複数店舗の予測を一括で実行：

```python
# 複数店舗のデータ
stores_data = {
    "store_001": sales_data_1,
    "store_002": sales_data_2,
    "store_003": sales_data_3,
}

# バッチ予測
results = pipeline.batch_predict(stores_data)

# 結果の確認
for store_cd, result in results.items():
    if result.prediction:
        print(f"{store_cd}: {result.prediction.prediction:,.0f}円")
```

## エラーハンドリング

パイプラインは以下の問題を自動的に処理します：

- **欠損値**: 線形補間、前方/後方補完
- **異常値**: IQR法、Zスコア法での検出と置換
- **負の値**: ゼロまたは周辺値での置換
- **データ不足**: 最小日数チェックとエラー報告

## 結果の構造

`PipelineResult` オブジェクトには以下の情報が含まれます：

```python
result = pipeline.predict(new_store_sales)

# 予測結果
result.prediction.prediction        # 予測年間売上
result.prediction.lower_bound      # 信頼区間下限
result.prediction.upper_bound      # 信頼区間上限
result.prediction.confidence_score # 信頼度スコア

# 検証結果
result.validation_result.is_valid  # 検証合格/不合格
result.validation_result.errors    # エラーリスト

# 品質レポート
result.quality_report.overall_score     # 総合スコア
result.quality_report.completeness_score # 完全性
result.quality_report.consistency_score  # 一貫性
result.quality_report.accuracy_score     # 正確性

# その他
result.explanation      # 予測の説明文
result.execution_time   # 実行時間（秒）
result.warnings        # 警告メッセージ
```

## カスタムパイプラインの作成

独自の処理ステップを含むカスタムパイプライン：

```python
def custom_transform(data, **kwargs):
    # カスタム変換処理
    return transformed_data

def custom_validation(data, **kwargs):
    # カスタム検証処理
    return validated_data

# カスタムパイプラインの作成
custom_pipeline = pipeline.create_custom_pipeline(
    steps=[
        ("validation", custom_validation),
        ("transform", custom_transform),
        ("predict", lambda x: pipeline.predict(x))
    ],
    name="my_custom_pipeline"
)

# 実行
result = custom_pipeline(new_store_sales)
```

## ベストプラクティス

1. **データ品質の確認**: 品質スコアが低い場合は結果の信頼性も低下します
2. **適切な前処理**: データの特性に応じて前処理オプションを調整
3. **十分なデータ量**: 最低でも14日以上、理想的には30日以上のデータを使用
4. **フィルタリング**: 店舗属性でフィルタリングして類似性を向上
5. **結果の保存**: 重要な予測は必ず保存してトレーサビリティを確保

## トラブルシューティング

### 予測精度が低い場合
- データ量を増やす（30日以上推奨）
- 類似店舗数を調整（3-10店舗）
- 異なる類似性指標を試す

### 処理が遅い場合
- `smooth_data=False` で平滑化を無効化
- `auto_optimize_period=False` で期間最適化をスキップ
- バッチサイズを調整

### メモリ不足の場合
- データを分割してバッチ処理
- 不要な処理ステップを無効化