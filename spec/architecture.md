# TwinStore システムアーキテクチャ仕様書

**作成日**: 2025-07-23  
**関連要件**: [requirements.md](./requirements.md)  

## アーキテクチャ概要

TwinStoreは**パイプライン・アーキテクチャ**を中心とした設計で、各処理段階を独立したコンポーネントとして実装しています。この設計により、高い保守性、拡張性、テスト容易性を実現しています。

### 設計原則

1. **Single Responsibility Principle (SRP)**: 各コンポーネントは単一の責務を持つ
2. **Strategy Pattern**: アルゴリズムの切り替えを可能にする
3. **Builder Pattern**: 複雑なオブジェクトの段階的構築
4. **Pipeline Pattern**: データフローの明確な段階分離

## システム構成

### レイヤー構造

```
┌─────────────────────────────────────────┐
│           Application Layer             │
│  ┌─────────────────────────────────────┐│
│  │        Pipeline Management          ││
│  │   (PredictionPipeline, Builder)     ││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│            Business Layer               │
│  ┌───────────┐ ┌───────────┐ ┌─────────┐│
│  │Core Engine│ │Data Proc. │ │Visual.  ││
│  │(Predictor)│ │(Validator)│ │(Charts) ││
│  └───────────┘ └───────────┘ └─────────┘│
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│           Infrastructure Layer          │
│  ┌───────────┐ ┌───────────┐ ┌─────────┐│
│  │Config Mgmt│ │Utilities  │ │Types    ││
│  │(Constants)│ │(Stats)    │ │(Aliases)││
│  └───────────┘ └───────────┘ └─────────┘│
└─────────────────────────────────────────┘
```

### コンポーネント図

```
┌─────────────────────────────────────────────────────────────┐
│                    PredictionPipeline                      │
├─────────────────────────────────────────────────────────────┤
│  1. DataValidator    → 入力データの検証                     │
│  2. DataPreprocessor → 前処理・クリーニング                │
│  3. QualityChecker   → データ品質評価                       │
│  4. AnomalyDetector  → 異常値検出                           │
│  5. SimilarityEngine → 類似店舗検索                         │
│  6. SalesPredictor   → 売上予測実行                         │
│  7. PredictionExplainer → 結果説明生成                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Component Details                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │   Core/     │    │   Data/     │    │   Config/   │    │
│  │             │    │             │    │             │    │
│  │ predictor   │    │ loader      │    │ constants   │    │
│  │ similarity  │    │ validator   │    │ defaults    │    │
│  │ normalizer  │    │ preprocessor│    │ validation  │    │
│  │ explainer   │    │ quality_ch. │    │             │    │
│  └─────────────┘    │ anomaly_det.│    └─────────────┘    │
│                     └─────────────┘                       │
│                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    │
│  │ Visualization/   │   Utils/    │    │   Types/    │    │
│  │             │    │             │    │             │    │
│  │ sales_align.│    │ statistics  │    │ type_aliases│    │
│  │ visualizer  │    │ validation  │    │             │    │
│  │             │    │ data_conv.  │    │             │    │
│  └─────────────┘    └─────────────┘    └─────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## データフロー

### 処理フロー詳細

```
[入力データ] 
    │
    ├─ CSV File ────┐
    ├─ Excel File ──┤
    ├─ JSON ────────┤
    ├─ DataFrame ───┤ → [DataLoader] → 統一形式変換
    ├─ NumPy Array ─┤
    └─ Dictionary ──┘
    │
    ▼
[DataValidator] → ValidationResult(errors, warnings)
    │
    ▼
[DataPreprocessor] → 欠損値補完、外れ値処理
    │
    ▼
[QualityChecker] → 品質スコア(0-100%)
    │
    ▼
[AnomalyDetector] → 異常値フラグ
    │
    ▼
[Normalizer] → 正規化データ
    │
    ▼
[SimilarityEngine] → DTW距離計算 → 類似店舗リスト
    │
    ▼
[SalesPredictor] → 統計予測 → 予測値・信頼区間
    │
    ▼
[PredictionExplainer] → 説明文・メタデータ
    │
    ▼
[PredictionResult] → 最終結果
```

### データ構造

```python
# 主要データ構造
class PredictionResult:
    prediction: float          # 予測値
    confidence_interval: Tuple[float, float]  # 信頼区間
    confidence_score: float    # 信頼度スコア (0-1)
    similar_stores: List[Dict] # 類似店舗情報
    metadata: Dict            # 予測メタデータ

class ValidationResult:
    is_valid: bool            # 検証結果
    errors: List[str]         # エラーメッセージ
    warnings: List[str]       # 警告メッセージ
    quality_score: float      # 品質スコア

class PipelineConfig:
    validation_config: Dict   # バリデーション設定
    preprocessing_config: Dict # 前処理設定
    prediction_config: Dict   # 予測設定
    output_config: Dict       # 出力設定
```

## 主要アルゴリズム

### 類似性計算アルゴリズム

**DTW (Dynamic Time Warping)**
```
目的: 時系列データの類似性計算
実装: dtaidistance ライブラリ
特徴: 
- 時間軸の伸縮を考慮した距離計算
- ウィンドウ制約による計算効率化
- 複数の距離メトリクスをサポート
```

**その他の類似性指標**
- コサイン類似度: ベクトル間の角度による類似性
- 相関係数: 線形関係の強さ
- ユークリッド距離: 多次元空間での直線距離

### 正規化アルゴリズム

1. **Z-Score正規化**: (x - μ) / σ
2. **Min-Max正規化**: (x - min) / (max - min)  
3. **Robust正規化**: (x - median) / IQR
4. **First-Day-Ratio**: 初日売上を基準とした比率
5. **Mean-Ratio**: 平均売上を基準とした比率

### 異常値検出アルゴリズム

1. **IQR方式**: Q1 - 1.5×IQR, Q3 + 1.5×IQR
2. **Z-Score方式**: |z-score| > 3.0
3. **Isolation Forest**: 機械学習による異常検知

## 拡張性設計

### Strategy Pattern による拡張ポイント

```python
# 類似性計算の拡張
class SimilarityCalculator:
    strategies = {
        'dtw': DTWCalculator,
        'cosine': CosineCalculator, 
        'correlation': CorrelationCalculator,
        # 新しいアルゴリズムを追加可能
    }

# 正規化手法の拡張
class Normalizer:
    methods = {
        'z_score': ZScoreNormalizer,
        'min_max': MinMaxNormalizer,
        'robust': RobustNormalizer,
        # 新しい正規化手法を追加可能
    }
```

### 業態別プリセット

```python
PRESET_CONFIGS = {
    'retail': {
        'similarity_method': 'dtw',
        'normalization': 'first_day_ratio',
        'quality_threshold': 70.0
    },
    'restaurant': {
        'similarity_method': 'correlation',
        'normalization': 'mean_ratio', 
        'quality_threshold': 75.0
    },
    'service': {
        'similarity_method': 'cosine',
        'normalization': 'z_score',
        'quality_threshold': 80.0
    }
}
```

## パフォーマンス最適化

### 計算効率化

1. **DTWウィンドウ制約**: 計算量をO(n²)からO(n×w)に削減
2. **類似度キャッシング**: 同一データの再計算を防止
3. **並列処理**: バッチ予測での並列実行
4. **メモリ最適化**: 大容量データの段階的処理

### ボトルネック分析

```
処理時間の内訳（1店舗あたり）:
- データ読み込み: 0.1秒
- データ検証: 0.05秒  
- 前処理: 0.1秒
- DTW計算: 0.6秒 ★主要ボトルネック
- 予測計算: 0.1秒
- 結果生成: 0.05秒
合計: 1.0秒
```

## セキュリティ設計

### 入力検証
- SQL インジェクション対策
- ファイルパス検証
- データ型・範囲チェック

### データ保護
- 個人情報を含まない設計
- 一時ファイルの自動削除
- メモリダンプの防止

## 運用・監視

### ログ設計
```python
logging_config = {
    'version': 1,
    'handlers': {
        'file': {
            'filename': 'twinstore.log',
            'level': 'INFO'
        }
    },
    'loggers': {
        'twinstore.pipeline': {'level': 'DEBUG'},
        'twinstore.predictor': {'level': 'INFO'}
    }
}
```

### メトリクス収集
- 予測精度の監視
- 処理時間の測定
- エラー率の追跡
- リソース使用量の監視

## テスト戦略

### テスト構造
```
tests/
├── unit/           # 単体テスト（各コンポーネント）
├── integration/    # 統合テスト（パイプライン全体）
├── performance/    # 性能テスト
├── fixtures/       # テストデータ
└── conftest.py     # テスト設定
```

### カバレッジ目標
- 単体テスト: 95%以上
- 統合テスト: 85%以上  
- 全体カバレッジ: 90%以上

## 関連ドキュメント

- [全体要件仕様書](./requirements.md)
- [全体設計仕様書](./design.md)
- [実装タスク](./tasks.md)
- [機能別仕様書](./sales-prediction/requirements.md)