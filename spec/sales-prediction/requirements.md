# 売上予測機能 要件仕様書

**作成日**: 2025-07-23  
**最終更新**: 2025-07-23  
**ステータス**: Draft  

## 機能概要

売上予測機能は、TwinStoreの中核となる機能で、新規店舗の限られた売上データから年間売上を高精度で予測します。類似店舗のパターンを活用した統計的手法により、従来手法と比較して20%以上の精度向上を実現します。

## ユーザーストーリー

### US-SP-001: 基本的な売上予測
**As a** 新規店舗の事業責任者  
**I want** 最小限のデータで年間売上予測を実行したい  
**So that** 早期に事業計画を立てられる  

**受け入れ基準:**
- GIVEN 3日分以上の売上データがある
- WHEN predict()メソッドを実行する
- THEN 年間売上予測値と95%信頼区間が返される

### US-SP-002: 予測精度の信頼度評価
**As a** データアナリスト  
**I want** 予測結果の信頼度を数値で評価したい  
**So that** 予測の不確実性を把握できる  

**受け入れ基準:**
- GIVEN 予測実行が完了している
- WHEN 結果を確認する
- THEN 0-1の信頼度スコアが提供される

### US-SP-003: 複数の予測手法の選択
**As a** システム管理者  
**I want** 異なる予測手法を選択したい  
**So that** 業態に応じた最適な予測が可能になる  

**受け入れ基準:**
- GIVEN 予測設定がある
- WHEN 予測手法を指定する
- THEN 指定した手法で予測が実行される

### US-SP-004: バッチ予測処理
**As a** 複数店舗の管理者  
**I want** 複数店舗の予測を一括で実行したい  
**So that** 効率的に予測業務を行える  

**受け入れ基準:**
- GIVEN 複数店舗のデータがある
- WHEN batch_predict()を実行する
- THEN 全店舗の予測結果が一括で返される

## 機能仕様

### 基本機能

1. **単一店舗予測**
   - 最小3日分のデータから年間売上を予測
   - 95%信頼区間の提供
   - 信頼度スコア（0-1）の算出

2. **予測手法選択**
   - weighted_average: 類似店舗の重み付け平均（デフォルト）
   - exponential_growth: 指数成長モデル
   - linear_regression: 線形回帰モデル
   - seasonal_decomposition: 季節分解モデル

3. **正規化オプション**
   - first_day_ratio: 初日売上比率（デフォルト）
   - z_score: Z-score正規化
   - min_max: Min-Max正規化
   - robust: Robust正規化
   - mean_ratio: 平均売上比率

4. **バッチ処理**
   - 複数店舗の並列予測
   - 進捗表示機能
   - エラー店舗の個別報告

### 高度な機能

1. **適応的重み付け**
   - DTW距離による自動重み調整
   - 類似店舗数の動的決定
   - 外れ値店舗の自動除外

2. **不確実性定量化**
   - Monte Carlo シミュレーション
   - ブートストラップ法による信頼区間
   - 予測分布の可視化

3. **成長パターン分析**
   - 成長曲線の自動フィッティング
   - 飽和点の予測
   - 季節性の考慮

## 非機能要件

### パフォーマンス
- **予測時間**: 1店舗あたり1秒以内
- **バッチ処理**: 100店舗を5分以内
- **メモリ使用量**: 予測1件あたり10MB以内

### 精度
- **MAPE目標**: 20%未満
- **信頼区間カバレッジ**: 95%±2%
- **最小精度保証**: 従来手法比10%以上向上

### 可用性
- **エラー率**: 1%未満
- **異常データ対応**: Graceful degradation
- **リソース制限**: メモリ不足時の段階的処理

## 入力仕様

### データ形式
```python
# 基本形式
new_store_sales = [100, 120, 130, ...]  # List[float]

# DataFrame形式
df = pd.DataFrame({'sales': [100, 120, 130, ...]})

# 辞書形式
sales_dict = {'day1': 100, 'day2': 120, 'day3': 130}

# NumPy配列
sales_array = np.array([100, 120, 130, ...])
```

### パラメータ
```python
predict(
    new_store_data,                    # 必須: 売上データ
    method='weighted_average',         # 予測手法
    normalization='first_day_ratio',   # 正規化手法
    confidence_level=0.95,             # 信頼水準
    max_similar_stores=10,             # 最大類似店舗数
    min_similarity_score=0.1           # 最小類似度閾値
)
```

## 出力仕様

### 基本出力
```python
@dataclass
class PredictionResult:
    prediction: float                      # 年間売上予測値
    confidence_interval: Tuple[float, float]  # 信頼区間
    confidence_score: float                # 信頼度スコア (0-1)
    method: str                           # 使用した予測手法
    normalization: str                    # 使用した正規化手法
    similar_stores_count: int             # 使用した類似店舗数
    execution_time: float                 # 実行時間（秒）
```

### 詳細出力（verbose=True）
```python
@dataclass  
class DetailedPredictionResult(PredictionResult):
    similar_stores: List[Dict]            # 類似店舗詳細
    growth_analysis: Dict                 # 成長パターン分析
    quality_metrics: Dict                 # 品質指標
    debug_info: Dict                      # デバッグ情報
```

## エラーハンドリング

### 例外タイプ
```python
class PredictionError(TwinStoreError):
    """予測実行エラー"""
    pass

class InsufficientDataError(PredictionError):
    """データ不足エラー"""
    pass

class NoSimilarStoresError(PredictionError):  
    """類似店舗不足エラー"""
    pass

class ModelFittingError(PredictionError):
    """モデル適合エラー"""
    pass
```

### エラー対応
- **データ不足**: 最小データ要件未満の場合
- **類似店舗不足**: 有効な類似店舗が見つからない場合
- **数値計算エラー**: 計算過程での異常値発生
- **メモリ不足**: 大容量データ処理時

## 品質保証

### テストケース
1. **正常系テスト**
   - 最小データでの予測
   - 十分なデータでの予測
   - 各予測手法の動作確認

2. **異常系テスト**
   - データ不足での例外処理
   - 異常値データでの動作
   - 類似店舗不足での処理

3. **性能テスト**
   - 大容量データでの処理時間
   - メモリ使用量の測定
   - 並列処理の効率確認

### 精度検証
- **Cross-validation**: 5-fold交差検証
- **時系列分割**: 時間軸での訓練・テスト分割
- **A/Bテスト**: 既存手法との比較

## 依存関係

### 前提機能
- 類似店舗マッチング機能（SimilarityEngine）
- データ処理機能（DataValidator, Normalizer）
- 設定管理機能（Config）

### 影響範囲
- 可視化機能（予測結果グラフ）
- レポート生成機能（予測レポート）
- API機能（REST API経由の予測）

## 実装優先度

### 高優先度（Phase 1）
1. 基本予測機能（weighted_average）
2. 信頼区間計算
3. 基本エラーハンドリング

### 中優先度（Phase 2）
1. 複数予測手法の実装
2. バッチ処理機能
3. 性能最適化

### 低優先度（Phase 3）
1. 高度な統計手法
2. 不確実性定量化
3. 詳細診断機能