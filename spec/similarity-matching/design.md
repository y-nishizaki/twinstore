# 類似店舗マッチング機能 設計仕様書

**作成日**: 2025-07-23  
**関連要件**: [requirements.md](./requirements.md)  

## 設計概要

類似店舗マッチング機能は、**Strategy Pattern**を使用して複数の類似性計算アルゴリズムを切り替え可能な設計としています。DTWを中心とした時系列分析により、高精度な類似店舗検索を実現します。

## アーキテクチャ設計

### クラス構成

```python
# 中核クラス
class SimilarityEngine:
    """類似性計算エンジンのメインクラス"""
    def __init__(self, config: SimilarityConfig)
    def find_similar(self, target_data, historical_data) -> List[SimilarStore]
    def calculate_similarity_matrix(self, datasets) -> np.ndarray
    
# 類似性計算アルゴリズム
class SimilarityCalculator(ABC):
    """類似性計算の抽象基底クラス"""
    @abstractmethod
    def calculate(self, x, y) -> float
    
class DTWSimilarityCalculator(SimilarityCalculator):
    """動的時間伸縮法による類似性計算"""
    
class CosineSimilarityCalculator(SimilarityCalculator):
    """コサイン類似度による計算"""
    
class CorrelationSimilarityCalculator(SimilarityCalculator):
    """相関係数による類似性計算"""

# データアライメント
class DataAligner:
    """時間軸調整クラス"""
    def align_periods(self, target_data, reference_data) -> Tuple[np.ndarray, np.ndarray]
    def find_optimal_alignment(self, data1, data2) -> int
```

### データフロー

```
[新規店舗データ] + [過去店舗DB]
    │
    ▼
[DataAligner] → 期間調整
    │
    ▼  
[SimilarityCalculator] → アルゴリズム選択
    │
    ▼
[DTW/Cosine/Correlation] → 類似性計算
    │
    ▼
[スコア計算] → 距離→類似度変換
    │
    ▼
[フィルタリング] → 闾値・数制限
    │
    ▼
[List[SimilarStore]] → 最終結果
```

## 詳細設計

### 1. SimilarityEngine クラス

```python
class SimilarityEngine:
    """類似店舗検索エンジンのメインクラス"""
    
    def __init__(self, config: SimilarityConfig):
        self.config = config
        self.calculator = self._create_calculator(config.method)
        self.data_aligner = DataAligner(config.alignment_config)
        self.normalizer = Normalizer(config.normalization_method)
        
    def find_similar(self, 
                     target_data: np.ndarray,
                     historical_data: Dict[str, np.ndarray],
                     max_results: int = None) -> List[SimilarStore]:
        """類似店舗の検索"""
        
        if not historical_data:
            raise InsufficientHistoricalDataError(
                "Historical data is required for similarity matching")
        
        # データ正規化
        normalized_target = self.normalizer.normalize(target_data)
        
        similarities = []
        
        for store_name, store_data in historical_data.items():
            try:
                # 期間アライメント
                aligned_target, aligned_store = self.data_aligner.align_periods(
                    normalized_target, store_data)
                
                # 類似性計算
                distance = self.calculator.calculate(
                    aligned_target, aligned_store)
                
                # 類似度スコア変換
                similarity_score = self._distance_to_similarity(distance)
                
                # 品質メトリクス計算
                quality_metrics = self._calculate_quality_metrics(
                    aligned_target, aligned_store, distance)
                
                similarities.append(SimilarStore(
                    store_name=store_name,
                    distance=distance,
                    similarity_score=similarity_score,
                    sales_pattern=aligned_store,
                    alignment_period=len(aligned_target),
                    metadata={
                        'quality_metrics': quality_metrics,
                        'alignment_method': self.config.alignment_method,
                        'similarity_method': self.config.method
                    }
                ))
                
            except Exception as e:
                logger.warning(f"Failed to calculate similarity for {store_name}: {e}")
                continue
        
        # 結果フィルタリング
        filtered_similarities = self._filter_results(
            similarities, max_results or self.config.max_results)
        
        # 類似度順でソート
        return sorted(filtered_similarities, 
                     key=lambda x: x.distance)  # DTWは距離が小さいほど類似
    
    def _distance_to_similarity(self, distance: float) -> float:
        """距離を類似度スコアに変換"""
        return 1.0 / (1.0 + distance)
    
    def _filter_results(self, 
                       similarities: List[SimilarStore],
                       max_results: int) -> List[SimilarStore]:
        """結果フィルタリング"""
        
        # 類似度闾値フィルタ
        filtered = [
            s for s in similarities 
            if s.similarity_score >= self.config.similarity_threshold
        ]
        
        # 最大結果数制限
        return filtered[:max_results]
```

### 2. DTWSimilarityCalculator クラス

```python
class DTWSimilarityCalculator(SimilarityCalculator):
    """動的時間伸縮法による類似性計算"""
    
    def __init__(self, config: SimilarityConfig):
        self.config = config
        self.window_constraint = config.dtw_window
        self.distance_metric = config.dtw_distance_metric
        
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """動的時間伸縮法による距離計算"""
        
        try:
            # dtaidistanceライブラリを使用
            from dtaidistance import dtw
            
            # ウィンドウ制約の計算
            window_size = int(max(len(x), len(y)) * self.window_constraint)
            
            # DTW距離計算
            distance = dtw.distance(
                x.astype(np.float64), 
                y.astype(np.float64),
                window=window_size if window_size > 0 else None
            )
            
            return distance
            
        except Exception as e:
            raise DTWComputationError(f"DTW calculation failed: {str(e)}")
    
    def calculate_with_path(self, 
                           x: np.ndarray, 
                           y: np.ndarray) -> Tuple[float, List[Tuple[int, int]]]:
        """パス情報付きDTW計算"""
        
        try:
            from dtaidistance import dtw
            
            window_size = int(max(len(x), len(y)) * self.window_constraint)
            
            # DTW距離とパスを同時計算
            distance, paths = dtw.warping_paths(
                x.astype(np.float64),
                y.astype(np.float64),
                window=window_size if window_size > 0 else None
            )
            
            # 最適パスの抽出
            optimal_path = dtw.best_path(paths)
            
            return distance, optimal_path
            
        except Exception as e:
            raise DTWComputationError(f"DTW path calculation failed: {str(e)}")
    
    def _validate_input(self, x: np.ndarray, y: np.ndarray):
        """入力データの検証"""
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Input arrays cannot be empty")
        
        if np.any(np.isnan(x)) or np.any(np.isnan(y)):
            raise ValueError("Input arrays cannot contain NaN values")
        
        if np.any(np.isinf(x)) or np.any(np.isinf(y)):
            raise ValueError("Input arrays cannot contain infinite values")
```

### 3. DataAligner クラス

```python
class DataAligner:
    """時間軸調整クラス"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.alignment_method = config.get('method', 'truncate')
        self.padding_value = config.get('padding_value', 0.0)
        
    def align_periods(self, 
                     target_data: np.ndarray,
                     reference_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """期間アライメント"""
        
        target_len = len(target_data)
        ref_len = len(reference_data)
        
        if self.alignment_method == 'truncate':
            # 短い方に合わせる
            min_len = min(target_len, ref_len)
            return target_data[:min_len], reference_data[:min_len]
            
        elif self.alignment_method == 'pad':
            # 長い方に合わせてパディング
            max_len = max(target_len, ref_len)
            
            aligned_target = self._pad_array(target_data, max_len)
            aligned_ref = self._pad_array(reference_data, max_len)
            
            return aligned_target, aligned_ref
            
        elif self.alignment_method == 'interpolate':
            # 長い方に合わせて補間
            max_len = max(target_len, ref_len)
            
            aligned_target = self._interpolate_array(target_data, max_len)
            aligned_ref = self._interpolate_array(reference_data, max_len)
            
            return aligned_target, aligned_ref
            
        elif self.alignment_method == 'optimal':
            # 最適アライメント期間を探索
            optimal_period = self.find_optimal_alignment(
                target_data, reference_data)
            
            return target_data[:optimal_period], reference_data[:optimal_period]
            
        else:
            raise ValueError(f"Unknown alignment method: {self.alignment_method}")
    
    def find_optimal_alignment(self, 
                              data1: np.ndarray, 
                              data2: np.ndarray) -> int:
        """最適アライメント期間の探索"""
        
        min_period = min(3, min(len(data1), len(data2)))  # 最小3日
        max_period = min(len(data1), len(data2))
        
        best_period = min_period
        best_correlation = -np.inf
        
        for period in range(min_period, max_period + 1):
            # 指定期間での相関係数を計算
            segment1 = data1[:period]
            segment2 = data2[:period]
            
            if len(segment1) > 1 and len(segment2) > 1:
                correlation = np.corrcoef(segment1, segment2)[0, 1]
                
                if not np.isnan(correlation) and correlation > best_correlation:
                    best_correlation = correlation
                    best_period = period
        
        return best_period
    
    def _pad_array(self, array: np.ndarray, target_length: int) -> np.ndarray:
        """配列のパディング"""
        if len(array) >= target_length:
            return array[:target_length]
        
        padding_length = target_length - len(array)
        padding = np.full(padding_length, self.padding_value)
        
        return np.concatenate([array, padding])
    
    def _interpolate_array(self, array: np.ndarray, target_length: int) -> np.ndarray:
        """配列の補間"""
        if len(array) >= target_length:
            return array[:target_length]
        
        from scipy import interpolate
        
        # 現在のインデックス
        old_indices = np.arange(len(array))
        
        # 新しいインデックス
        new_indices = np.linspace(0, len(array) - 1, target_length)
        
        # 線形補間
        f = interpolate.interp1d(old_indices, array, kind='linear', 
                               fill_value='extrapolate')
        
        return f(new_indices)
```

### 4. その他の類似性計算器

```python
class CosineSimilarityCalculator(SimilarityCalculator):
    """コサイン類似度計算器"""
    
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """コサイン類似度計算"""
        from sklearn.metrics.pairwise import cosine_similarity
        
        # コサイン類似度は1に近いほど類似
        # 距離に変換するため、1-similarityを返す
        similarity = cosine_similarity(
            x.reshape(1, -1), y.reshape(1, -1))[0, 0]
        
        # 距離に変換（小さいほど類似）
        return 1.0 - similarity

class CorrelationSimilarityCalculator(SimilarityCalculator):
    """相関係数類似性計算器"""
    
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """相関係数計算"""
        if len(x) < 2 or len(y) < 2:
            return np.inf  # 計算不可の場合
        
        correlation = np.corrcoef(x, y)[0, 1]
        
        # NaNの場合は無限大距離
        if np.isnan(correlation):
            return np.inf
        
        # 距離に変換（高相関ほど小さい距離）
        return 1.0 - abs(correlation)

class EuclideanSimilarityCalculator(SimilarityCalculator):
    """ユークリッド距離計算器"""
    
    def calculate(self, x: np.ndarray, y: np.ndarray) -> float:
        """ユークリッド距離計算"""
        return np.linalg.norm(x - y)
```

## データモデル設計

### 主要データ構造

```python
@dataclass
class SimilarStore:
    """類似店舗情報クラス"""
    store_name: str                     # 店舗識別名
    distance: float                     # 距離値（小さいほど類似）
    similarity_score: float             # 類似度スコア（0-1）
    sales_pattern: np.ndarray           # 売上パターンデータ
    alignment_period: int               # アライメント期間
    metadata: Dict                      # 追加メタデータ

@dataclass
class SimilarityConfig:
    """類似性計算設定クラス"""
    method: str = 'dtw'                 # 類似性計算手法
    max_results: int = 10               # 最大結果数
    similarity_threshold: float = 0.1   # 類似度闾値
    normalization_method: str = 'z_score' # 正規化手法
    
    # DTW固有パラメータ
    dtw_window: float = 0.1             # ウィンドウ制約（0-1）
    dtw_distance_metric: str = 'euclidean' # DTW距離メトリクス
    
    # アライメント設定
    alignment_config: Dict = field(default_factory=lambda: {
        'method': 'truncate',
        'padding_value': 0.0
    })
```

## パフォーマンス設計

### 計算最適化

1. **DTW最適化**
   - ウィンドウ制約による計算量削減
   - NumPyベクトル化演算の活用
   - メモリ効率的なDPテーブル管理

2. **並列処理**
   - 複数店舗の同時類似性計算
   - ThreadPoolExecutorによる並列化
   - メモリ使用量の制御

3. **キャッシュ機能**
   - 類似性計算結果のキャッシュ
   - アライメント結果の再利用
   - LRUキャッシュによるメモリ管理

### パフォーマンス監視

```python
@performance_monitor
def find_similar(self, target_data, historical_data):
    """パフォーマンス監視付き類似検索"""
    
    start_time = time.time()
    start_memory = psutil.Process().memory_info().rss
    
    result = self._internal_find_similar(target_data, historical_data)
    
    end_time = time.time()
    end_memory = psutil.Process().memory_info().rss
    
    # パフォーマンスメトリクスをログ
    execution_time = end_time - start_time
    memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
    
    logger.info(f"Similarity search completed: "
               f"time={execution_time:.3f}s, memory={memory_used:.2f}MB, "
               f"results={len(result)}")
    
    return result
```

## エラーハンドリング設計

### 例外階層

```python
class SimilarityError(TwinStoreError):
    """類似性計算エラーのベース例外"""
    pass

class InsufficientHistoricalDataError(SimilarityError):
    """過去データ不足エラー"""
    def __init__(self, message: str, required_count: int = None):
        super().__init__(message)
        self.required_count = required_count

class DTWComputationError(SimilarityError):
    """DTW計算エラー"""
    def __init__(self, message: str, data_shapes: Tuple = None):
        super().__init__(message)
        self.data_shapes = data_shapes

class DataAlignmentError(SimilarityError):
    """データアライメントエラー"""
    pass
```

### エラー復旧戦略

```python
def find_similar_with_fallback(self, target_data, historical_data):
    """フォールバック機能付き類似検索"""
    
    try:
        # 第1選択: DTWで計算
        return self.find_similar(target_data, historical_data)
        
    except DTWComputationError as e:
        logger.warning(f"DTW calculation failed, falling back to cosine similarity: {e}")
        
        # フォールバック: コサイン類似度
        fallback_config = self.config.copy()
        fallback_config.method = 'cosine'
        fallback_engine = SimilarityEngine(fallback_config)
        
        try:
            return fallback_engine.find_similar(target_data, historical_data)
        except Exception as e2:
            logger.error(f"Fallback similarity calculation also failed: {e2}")
            
            # 最終フォールバック: 相関係数
            final_config = self.config.copy()
            final_config.method = 'correlation'
            final_engine = SimilarityEngine(final_config)
            
            return final_engine.find_similar(target_data, historical_data)
```

## テスト設計

### 単体テスト

```python
class TestSimilarityEngine:
    def test_dtw_calculation(self):
        """基本的なDTW計算テスト"""
        
    def test_data_alignment(self):
        """データアライメントテスト"""
        
    def test_similarity_ranking(self):
        """類似度ランキングテスト"""
        
    def test_error_handling(self):
        """エラーハンドリングテスト"""
```

### 統合テスト

```python
class TestSimilarityIntegration:
    def test_end_to_end_similarity_search(self):
        """エンドツーエンド類似検索テスト"""
        
    def test_performance_requirements(self):
        """性能要件テスト"""
        
    def test_accuracy_validation(self):
        """精度検証テスト"""
```

## 関連ドキュメント

- [要件仕様書](./requirements.md)
- [実装タスク](./tasks.md)
- [売上予測設計](../sales-prediction/design.md)
- [データ処理設計](../data-processing/design.md)