# 売上予測機能 設計仕様書

**作成日**: 2025-07-23  
**関連要件**: [requirements.md](./requirements.md)  

## 設計概要

売上予測機能は、統計的手法と機械学習アプローチを組み合わせた予測エンジンです。類似店舗の成長パターンを重み付けして統合することで、高精度な年間売上予測を実現します。

## アーキテクチャ設計

### クラス構成

```python
# 中核クラス
class SalesPredictor:
    """売上予測のメインクラス"""
    def __init__(self, config: PredictionConfig)
    def predict(self, data, similar_stores) -> PredictionResult
    def batch_predict(self, datasets) -> List[PredictionResult]
    
# 予測手法の実装
class PredictionMethod(ABC):
    """予測手法の抽象基底クラス"""
    @abstractmethod
    def predict(self, data, similar_stores) -> float
    
class WeightedAveragePredictor(PredictionMethod):
    """重み付け平均による予測"""
    
class ExponentialGrowthPredictor(PredictionMethod):
    """指数成長モデルによる予測"""
    
class LinearRegressionPredictor(PredictionMethod):
    """線形回帰による予測"""

# 信頼区間計算
class ConfidenceIntervalCalculator:
    """信頼区間計算クラス"""
    def calculate_bootstrap_ci(self, predictions, confidence_level)
    def calculate_analytical_ci(self, prediction, std_error, confidence_level)
```

### データフロー

```
[新規店舗データ] + [類似店舗リスト]
    │
    ▼
[データ正規化] → Normalizer
    │
    ▼  
[予測手法選択] → PredictionMethod
    │
    ▼
[基本予測値計算] → 重み付け統合
    │
    ▼
[信頼区間計算] → ConfidenceIntervalCalculator
    │
    ▼
[信頼度スコア算出] → 類似店舗品質評価
    │
    ▼
[PredictionResult] → 最終結果
```

## 詳細設計

### 1. SalesPredictor クラス

```python
class SalesPredictor:
    """売上予測エンジンのメインクラス"""
    
    def __init__(self, config: PredictionConfig):
        self.config = config
        self.normalizer = Normalizer(config.normalization_method)
        self.method = self._create_prediction_method(config.method)
        self.ci_calculator = ConfidenceIntervalCalculator(config.confidence_level)
        
    def predict(self, 
                new_store_data: np.ndarray,
                similar_stores: List[Dict],
                **kwargs) -> PredictionResult:
        """単一店舗の売上予測"""
        
        # 1. データ前処理
        normalized_data = self.normalizer.normalize(new_store_data)
        
        # 2. 類似店舗の品質評価
        quality_scores = self._evaluate_similar_stores(similar_stores)
        
        # 3. 重み計算
        weights = self._calculate_weights(similar_stores, quality_scores)
        
        # 4. 予測実行
        prediction_value = self.method.predict(
            normalized_data, similar_stores, weights)
        
        # 5. 信頼区間計算
        confidence_interval = self.ci_calculator.calculate(
            normalized_data, similar_stores, weights, prediction_value)
        
        # 6. 信頼度スコア計算
        confidence_score = self._calculate_confidence_score(
            similar_stores, weights, quality_scores)
        
        # 7. 結果構築
        return PredictionResult(
            prediction=prediction_value,
            confidence_interval=confidence_interval,
            confidence_score=confidence_score,
            method=self.config.method,
            normalization=self.config.normalization_method,
            similar_stores_count=len(similar_stores),
            execution_time=time.time() - start_time,
            metadata=self._generate_metadata(similar_stores, weights)
        )
    
    def _calculate_weights(self, 
                          similar_stores: List[Dict], 
                          quality_scores: np.ndarray) -> np.ndarray:
        """類似店舗の重み計算"""
        
        # DTW距離から基本重みを計算
        distances = np.array([store['distance'] for store in similar_stores])
        distance_weights = 1.0 / (1.0 + distances)
        
        # 品質スコアで調整
        adjusted_weights = distance_weights * quality_scores
        
        # 正規化
        return adjusted_weights / np.sum(adjusted_weights)
    
    def _calculate_confidence_score(self, 
                                   similar_stores: List[Dict],
                                   weights: np.ndarray,
                                   quality_scores: np.ndarray) -> float:
        """信頼度スコア計算"""
        
        # 要素別スコア
        similarity_score = np.average([1.0 / (1.0 + s['distance']) 
                                     for s in similar_stores], weights=weights)
        quality_score = np.average(quality_scores, weights=weights)
        coverage_score = min(len(similar_stores) / 5.0, 1.0)  # 5店舗で満点
        
        # 統合スコア
        confidence = (similarity_score * 0.4 + 
                     quality_score * 0.4 + 
                     coverage_score * 0.2)
        
        return np.clip(confidence, 0.0, 1.0)
```

### 2. 予測手法の実装

```python
class WeightedAveragePredictor(PredictionMethod):
    """重み付け平均による予測手法"""
    
    def predict(self, 
                normalized_data: np.ndarray,
                similar_stores: List[Dict],
                weights: np.ndarray) -> float:
        """重み付け平均予測"""
        
        predictions = []
        
        for store in similar_stores:
            # 成長率計算
            growth_rate = self._calculate_growth_rate(
                store['sales_pattern'], len(normalized_data))
            
            # 年間売上外挿
            annual_prediction = self._extrapolate_annual_sales(
                normalized_data, growth_rate)
            
            predictions.append(annual_prediction)
        
        # 重み付け平均
        predictions = np.array(predictions)
        return np.average(predictions, weights=weights)
    
    def _calculate_growth_rate(self, 
                              sales_pattern: np.ndarray, 
                              current_days: int) -> float:
        """成長率の計算"""
        
        # 類似期間の成長率を計算
        if len(sales_pattern) <= current_days:
            return 0.0
        
        initial_period = sales_pattern[:current_days]
        next_period = sales_pattern[current_days:current_days*2]
        
        if len(next_period) == 0:
            return 0.0
        
        initial_avg = np.mean(initial_period)
        next_avg = np.mean(next_period)
        
        if initial_avg == 0:
            return 0.0
        
        return (next_avg - initial_avg) / initial_avg
    
    def _extrapolate_annual_sales(self, 
                                 current_data: np.ndarray,
                                 growth_rate: float) -> float:
        """年間売上外挿"""
        
        daily_average = np.mean(current_data)
        
        # 指数成長モデル
        days_in_year = 365
        annual_sales = 0
        
        for day in range(days_in_year):
            daily_sales = daily_average * (1 + growth_rate) ** (day / 30.0)
            annual_sales += daily_sales
        
        return annual_sales

class ExponentialGrowthPredictor(PredictionMethod):
    """指数成長モデルによる予測"""
    
    def predict(self, 
                normalized_data: np.ndarray,
                similar_stores: List[Dict],
                weights: np.ndarray) -> float:
        """指数成長モデル予測"""
        
        # 現在データから成長パラメータを推定
        growth_params = self._fit_exponential_model(normalized_data)
        
        # 類似店舗の成長パラメータと統合
        similar_params = []
        for store in similar_stores:
            params = self._fit_exponential_model(
                store['sales_pattern'][:len(normalized_data)])
            similar_params.append(params)
        
        # 重み付け統合
        integrated_params = self._integrate_parameters(
            growth_params, similar_params, weights)
        
        # 年間予測
        return self._project_annual_sales(integrated_params)
    
    def _fit_exponential_model(self, data: np.ndarray) -> Dict:
        """指数モデルのパラメータ推定"""
        from scipy.optimize import curve_fit
        
        def exponential_func(x, a, b):
            return a * np.exp(b * x)
        
        x = np.arange(len(data))
        try:
            popt, _ = curve_fit(exponential_func, x, data, 
                              bounds=([0.1, -1], [np.inf, 1]))
            return {'a': popt[0], 'b': popt[1]}
        except:
            # フィッティング失敗時は線形近似
            return {'a': data[0], 'b': 0.0}
```

### 3. 信頼区間計算

```python
class ConfidenceIntervalCalculator:
    """信頼区間計算クラス"""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        
    def calculate(self, 
                  normalized_data: np.ndarray,
                  similar_stores: List[Dict],
                  weights: np.ndarray,
                  prediction: float) -> Tuple[float, float]:
        """信頼区間の計算"""
        
        # ブートストラップ法による信頼区間
        if len(similar_stores) >= 5:
            return self._bootstrap_confidence_interval(
                normalized_data, similar_stores, weights)
        else:
            # 解析的手法
            return self._analytical_confidence_interval(
                prediction, similar_stores, weights)
    
    def _bootstrap_confidence_interval(self, 
                                     normalized_data: np.ndarray,
                                     similar_stores: List[Dict],
                                     weights: np.ndarray,
                                     n_bootstrap: int = 1000) -> Tuple[float, float]:
        """ブートストラップ信頼区間"""
        
        bootstrap_predictions = []
        
        for _ in range(n_bootstrap):
            # リサンプリング
            indices = np.random.choice(len(similar_stores), 
                                     size=len(similar_stores), 
                                     replace=True)
            
            resampled_stores = [similar_stores[i] for i in indices]
            resampled_weights = weights[indices]
            resampled_weights /= np.sum(resampled_weights)
            
            # 予測計算
            predictions = []
            for store in resampled_stores:
                growth_rate = self._calculate_growth_rate(
                    store['sales_pattern'], len(normalized_data))
                annual_pred = self._extrapolate_annual_sales(
                    normalized_data, growth_rate)
                predictions.append(annual_pred)
            
            bootstrap_pred = np.average(predictions, weights=resampled_weights)
            bootstrap_predictions.append(bootstrap_pred)
        
        # パーセンタイル法
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        
        lower_ci = np.percentile(bootstrap_predictions, lower_percentile)
        upper_ci = np.percentile(bootstrap_predictions, upper_percentile)
        
        return (lower_ci, upper_ci)
    
    def _analytical_confidence_interval(self, 
                                      prediction: float,
                                      similar_stores: List[Dict],
                                      weights: np.ndarray) -> Tuple[float, float]:
        """解析的信頼区間"""
        
        # 予測値の分散推定
        individual_predictions = []
        for store in similar_stores:
            # 簡易予測値計算
            avg_sales = np.mean(store['sales_pattern'])
            annual_pred = avg_sales * 365
            individual_predictions.append(annual_pred)
        
        # 重み付け分散
        weighted_variance = np.average(
            (np.array(individual_predictions) - prediction) ** 2, 
            weights=weights)
        
        std_error = np.sqrt(weighted_variance)
        
        # t分布による信頼区間
        from scipy.stats import t
        df = len(similar_stores) - 1
        t_value = t.ppf(1 - self.alpha / 2, df)
        
        margin_error = t_value * std_error
        
        return (prediction - margin_error, prediction + margin_error)
```

### 4. バッチ処理設計

```python
class BatchPredictor:
    """バッチ予測処理クラス"""
    
    def __init__(self, predictor: SalesPredictor):
        self.predictor = predictor
        
    def batch_predict(self, 
                     datasets: List[Dict],
                     parallel: bool = True,
                     progress_callback: Optional[Callable] = None) -> List[PredictionResult]:
        """バッチ予測実行"""
        
        if parallel and len(datasets) > 1:
            return self._parallel_predict(datasets, progress_callback)
        else:
            return self._sequential_predict(datasets, progress_callback)
    
    def _parallel_predict(self, 
                         datasets: List[Dict],
                         progress_callback: Optional[Callable]) -> List[PredictionResult]:
        """並列予測処理"""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = [None] * len(datasets)
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # タスク提出
            future_to_index = {
                executor.submit(self._predict_single, dataset): i 
                for i, dataset in enumerate(datasets)
            }
            
            # 結果収集
            completed = 0
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    results[index] = PredictionError(f"Prediction failed: {str(e)}")
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, len(datasets))
        
        return results
    
    def _predict_single(self, dataset: Dict) -> PredictionResult:
        """単一データセットの予測"""
        return self.predictor.predict(
            dataset['new_store_data'],
            dataset['similar_stores']
        )
```

## 設定管理

### PredictionConfig

```python
@dataclass
class PredictionConfig:
    """予測設定クラス"""
    method: str = 'weighted_average'           # 予測手法
    normalization_method: str = 'first_day_ratio'  # 正規化手法
    confidence_level: float = 0.95             # 信頼水準
    max_similar_stores: int = 10               # 最大類似店舗数
    min_similarity_score: float = 0.1          # 最小類似度閾値
    bootstrap_samples: int = 1000              # ブートストラップサンプル数
    parallel_threshold: int = 5                # 並列処理閾値
    
    # 予測手法別パラメータ
    weighted_average_params: Dict = field(default_factory=dict)
    exponential_growth_params: Dict = field(default_factory=dict)
    linear_regression_params: Dict = field(default_factory=dict)
```

## パフォーマンス最適化

### 計算最適化

1. **キャッシング**
   - 成長率計算結果のキャッシュ
   - 類似店舗分析結果の再利用

2. **並列処理**
   - ThreadPoolExecutorによる並列予測
   - NumPyベクトル化演算の活用

3. **メモリ効率**
   - 大容量データの段階的処理
   - 不要オブジェクトの適時削除

### モニタリング

```python
@performance_monitor
def predict(self, data, similar_stores):
    """パフォーマンス監視付き予測"""
    start_memory = psutil.Process().memory_info().rss
    
    result = self._internal_predict(data, similar_stores)
    
    end_memory = psutil.Process().memory_info().rss
    memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
    
    logger.info(f"Prediction completed. Memory used: {memory_used:.2f}MB")
    return result
```

## テスト設計

### 単体テスト
```python
class TestSalesPredictor:
    def test_basic_prediction(self):
        """基本予測のテスト"""
        
    def test_confidence_interval_coverage(self):
        """信頼区間カバレッジのテスト"""
        
    def test_batch_prediction(self):
        """バッチ予測のテスト"""
        
    def test_error_handling(self):
        """エラーハンドリングのテスト"""
```

### 統合テスト
```python
class TestPredictionIntegration:
    def test_end_to_end_prediction(self):
        """エンドツーエンド予測テスト"""
        
    def test_performance_requirements(self):
        """性能要件のテスト"""
```

## 関連ドキュメント

- [要件仕様書](./requirements.md)
- [実装タスク](./tasks.md)
- [類似店舗マッチング設計](../similarity-matching/design.md)