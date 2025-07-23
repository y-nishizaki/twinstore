# API リファレンス

## 目次

- [SalesPredictor](#salespredictor)
- [PredictionPipeline](#predictionpipeline)
- [DataValidator](#datavalidator)
- [QualityChecker](#qualitychecker)
- [AnomalyDetector](#anomalydetector)
- [PredictionExplainer](#predictionexplainer)
- [SalesAlignmentVisualizer](#salesalignmentvisualizer)

## SalesPredictor

売上予測のメインクラス

### コンストラクタ

```python
SalesPredictor(
    similarity_method: str = 'dtw',
    normalization: str = 'zscore',
    distance_metric: str = 'euclidean',
    preset: Optional[str] = None,
    debug: bool = False
)
```

**パラメータ:**
- `similarity_method`: 類似性計算手法 ('dtw', 'correlation', 'euclidean')
- `normalization`: 正規化手法 ('zscore', 'minmax', 'first_value')
- `distance_metric`: DTWの距離計算方法
- `preset`: 業態別プリセット ('retail', 'restaurant', 'service')
- `debug`: デバッグモード

### メソッド

#### fit()
```python
fit(historical_data: Union[Dict, pd.DataFrame], store_attributes: Optional[Dict] = None) -> None
```
過去データで学習

#### predict()
```python
predict(
    new_store_sales: Union[np.ndarray, List, pd.Series],
    n_similar: int = 5,
    confidence_level: float = 0.95,
    filters: Optional[Dict] = None,
    store_attributes: Optional[Dict] = None
) -> PredictionResult
```
売上予測を実行

#### evaluate()
```python
evaluate(
    test_data: Dict[str, Union[np.ndarray, List]],
    metrics: List[str] = ['mape', 'rmse']
) -> Dict[str, float]
```
予測精度を評価

## PredictionPipeline

統合パイプライン処理

### コンストラクタ

```python
PredictionPipeline(config: Optional[PipelineConfig] = None)
```

### 設定クラス

```python
@dataclass
class PipelineConfig:
    validate_data: bool = True
    preprocess_data: bool = True
    handle_missing: bool = True
    detect_anomalies: bool = True
    check_quality: bool = True
    optimize_period: bool = False
    generate_explanation: bool = True
    generate_report: bool = False
    save_results: bool = True
    output_format: str = 'json'
    min_quality_score: float = 60.0
    anomaly_threshold: float = 3.0
    min_days: int = 3
    parallel_processing: bool = True
```

### メソッド

#### run()
```python
run(
    historical_data: Union[Dict, pd.DataFrame],
    new_store_sales: Union[np.ndarray, List, pd.Series],
    store_name: Optional[str] = None,
    store_attributes: Optional[Dict] = None
) -> PipelineResult
```

#### run_batch()
```python
run_batch(
    historical_data: Union[Dict, pd.DataFrame],
    new_stores_data: Dict[str, Union[np.ndarray, List]],
    parallel: bool = True
) -> Dict[str, PipelineResult]
```

## DataValidator

データ検証クラス

### メソッド

#### validate_historical_data()
```python
validate_historical_data(
    data: Union[Dict, pd.DataFrame],
    min_stores: int = 1,
    min_days_per_store: int = 30
) -> ValidationResult
```

#### validate_new_store_data()
```python
validate_new_store_data(
    data: Union[np.ndarray, List, pd.Series],
    min_days: int = 3
) -> ValidationResult
```

## QualityChecker

データ品質評価クラス

### メソッド

#### check_data_quality()
```python
check_data_quality(
    data: Union[pd.DataFrame, Dict, pd.Series],
    reference_data: Optional[Union[pd.DataFrame, Dict]] = None,
    check_items: Optional[List[str]] = None
) -> QualityReport
```

#### check_anomalies()
```python
check_anomalies(
    data: Union[pd.Series, np.ndarray],
    method: str = "zscore"
) -> Tuple[np.ndarray, Dict[str, Any]]
```

## AnomalyDetector

異常値検知クラス

### メソッド

#### detect_anomalies()
```python
detect_anomalies(
    data: Union[pd.Series, np.ndarray, List[float]],
    timestamps: Optional[Union[pd.DatetimeIndex, List[datetime]]] = None,
    return_scores: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
```

#### detect_realtime()
```python
detect_realtime(
    new_value: float,
    timestamp: Optional[datetime] = None
) -> Optional[AnomalyAlert]
```

## PredictionExplainer

予測説明生成クラス

### メソッド

#### generate_explanation()
```python
generate_explanation(
    prediction_result: PredictionResult,
    new_store_sales: np.ndarray,
    historical_data: Dict[str, np.ndarray],
    store_attributes: Optional[Dict[str, Dict]] = None
) -> str
```

#### generate_summary()
```python
generate_summary(
    prediction_result: PredictionResult,
    format: str = "short"
) -> str
```

## SalesAlignmentVisualizer

開店日基準の可視化クラス

### メソッド

#### plot_aligned_sales()
```python
plot_aligned_sales(
    sales_data: Dict[str, Union[List, np.ndarray, pd.Series]],
    config: Optional[AlignmentConfig] = None,
    title: Optional[str] = None,
    highlight_stores: Optional[List[str]] = None
) -> go.Figure
```

#### create_interactive_dashboard()
```python
create_interactive_dashboard(
    store_groups: Dict[str, List[str]],
    sales_data: Dict[str, Union[List, np.ndarray]],
    config: Optional[AlignmentConfig] = None,
    include_growth_analysis: bool = True,
    include_similarity_matrix: bool = True
) -> go.Figure
```