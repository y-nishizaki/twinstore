# パイプライン管理機能 設計仕様書

**作成日**: 2025-07-23  
**関連要件**: [requirements.md](./requirements.md)  

## 設計概要

パイプライン管理機能は**Builder Pattern**と**Chain of Responsibility Pattern**を組み合わせて設計されています。各処理ステップを疎結合に保ちながら、柔軟で拡張可能なパイプラインを構築できる設計となっています。

## アーキテクチャ設計

### クラス構成

```python
# パイプライン管理
class PredictionPipeline:
    """予測パイプラインのメインクラス"""
    def __init__(self, config: PipelineConfig)
    def predict(self, new_store_data, historical_data) -> PipelineResult
    def execute_step(self, step_name: str, data: Any) -> Any
    
class PipelineBuilder:
    """パイプラインのビルダークラス"""
    def with_validation(self, **kwargs) -> 'PipelineBuilder'
    def with_preprocessing(self, **kwargs) -> 'PipelineBuilder'
    def with_quality_check(self, **kwargs) -> 'PipelineBuilder'
    def with_anomaly_detection(self, **kwargs) -> 'PipelineBuilder'
    def with_similarity(self, **kwargs) -> 'PipelineBuilder'
    def with_prediction(self, **kwargs) -> 'PipelineBuilder'
    def build(self) -> PredictionPipeline

# ステップ管理
class PipelineStep(ABC):
    """パイプラインステップの抽象基底クラス"""
    @abstractmethod
    def execute(self, context: PipelineContext) -> StepResult
    @abstractmethod
    def validate_input(self, data: Any) -> bool
    
class PipelineContext:
    """パイプライン実行コンテキスト"""
    def __init__(self)
    def set_data(self, key: str, value: Any)
    def get_data(self, key: str) -> Any
    def add_log(self, message: str, level: str)
    def add_metric(self, name: str, value: float)
```

### パイプラインフロー

```
[Input] → PipelineBuilder → PredictionPipeline
                                  ↓
                          ValidationStep
                                  ↓
                         PreprocessingStep
                                  ↓
                          QualityCheckStep
                                  ↓
                        AnomalyDetectionStep
                                  ↓
                          SimilarityStep
                                  ↓
                          PredictionStep
                                  ↓
                          ExplanationStep
                                  ↓
                            [PipelineResult]
```

## 詳細設計

### 1. PredictionPipeline クラス

```python
class PredictionPipeline:
    """予測パイプラインのメインクラス"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.steps = self._initialize_steps()
        self.context = PipelineContext()
        self.performance_monitor = PerformanceMonitor()
        
    def predict(self, 
                new_store_data: Union[Dict, np.ndarray, pd.DataFrame],
                historical_data: Optional[Dict] = None) -> PipelineResult:
        """パイプライン全体の実行"""
        
        start_time = time.time()
        
        try:
            # コンテキスト初期化
            self.context.set_data('new_store_data', new_store_data)
            self.context.set_data('historical_data', historical_data)
            self.context.set_data('config', self.config)
            
            # 各ステップを順次実行
            for step_name, step in self.steps.items():
                step_result = self._execute_step(step_name, step)
                
                if not step_result.success and self.config.strict_mode:
                    raise StepExecutionError(
                        step_name, 
                        step_result.error_message
                    )
            
            # 結果の構築
            result = self._build_result()
            
            # パフォーマンスメトリクスの追加
            result.performance_metrics = self.performance_monitor.get_metrics()
            
            return result
            
        except Exception as e:
            self._handle_pipeline_error(e)
            raise
        
        finally:
            self.context.add_metric(
                'total_execution_time', 
                time.time() - start_time
            )
    
    def _execute_step(self, step_name: str, step: PipelineStep) -> StepResult:
        """個別ステップの実行"""
        
        start_time = time.time()
        
        try:
            # 入力検証
            if not step.validate_input(self.context):
                raise ValueError(f"Invalid input for step {step_name}")
            
            # ステップ実行
            self.context.add_log(f"Executing {step_name}", "INFO")
            result = step.execute(self.context)
            
            # メトリクス記録
            execution_time = time.time() - start_time
            self.context.add_metric(f"{step_name}_execution_time", execution_time)
            
            return result
            
        except Exception as e:
            self.context.add_log(
                f"Error in {step_name}: {str(e)}", 
                "ERROR"
            )
            
            if self.config.error_handling == 'fail_fast':
                raise
            else:
                return StepResult(
                    success=False,
                    error_message=str(e),
                    data=None
                )
    
    def _initialize_steps(self) -> Dict[str, PipelineStep]:
        """ステップの初期化"""
        
        steps = {}
        
        if self.config.enable_validation:
            steps['validation'] = ValidationStep(self.config.validation_config)
        
        if self.config.enable_preprocessing:
            steps['preprocessing'] = PreprocessingStep(self.config.preprocessing_config)
        
        if self.config.enable_quality_check:
            steps['quality_check'] = QualityCheckStep(self.config.quality_config)
        
        if self.config.enable_anomaly_detection:
            steps['anomaly_detection'] = AnomalyDetectionStep(self.config.anomaly_config)
        
        if self.config.enable_similarity:
            steps['similarity'] = SimilarityStep(self.config.similarity_config)
        
        steps['prediction'] = PredictionStep(self.config.prediction_config)
        
        if self.config.enable_explanation:
            steps['explanation'] = ExplanationStep(self.config.explanation_config)
        
        return steps
```

### 2. PipelineBuilder クラス

```python
class PipelineBuilder:
    """パイプラインのビルダークラス"""
    
    def __init__(self, preset: Optional[str] = None):
        if preset:
            self.config = self._load_preset(preset)
        else:
            self.config = PipelineConfig()
    
    def with_validation(self, 
                       strict: bool = False,
                       min_data_points: int = 3,
                       **kwargs) -> 'PipelineBuilder':
        """検証設定の追加"""
        
        self.config.enable_validation = True
        self.config.validation_config = ValidationConfig(
            strict_mode=strict,
            min_data_points=min_data_points,
            **kwargs
        )
        return self
    
    def with_preprocessing(self, 
                          handle_missing: bool = True,
                          handle_outliers: bool = True,
                          **kwargs) -> 'PipelineBuilder':
        """前処理設定の追加"""
        
        self.config.enable_preprocessing = True
        self.config.preprocessing_config = PreprocessingConfig(
            handle_missing=handle_missing,
            handle_outliers=handle_outliers,
            **kwargs
        )
        return self
    
    def with_similarity(self, 
                       method: str = 'dtw',
                       max_results: int = 10,
                       **kwargs) -> 'PipelineBuilder':
        """類似性計算設定の追加"""
        
        self.config.enable_similarity = True
        self.config.similarity_config = SimilarityConfig(
            method=method,
            max_results=max_results,
            **kwargs
        )
        return self
    
    def with_prediction(self, 
                       method: str = 'weighted_average',
                       confidence_level: float = 0.95,
                       **kwargs) -> 'PipelineBuilder':
        """予測設定の追加"""
        
        self.config.prediction_config = PredictionConfig(
            method=method,
            confidence_level=confidence_level,
            **kwargs
        )
        return self
    
    def build(self) -> PredictionPipeline:
        """パイプラインの構築"""
        
        # 設定の検証
        self._validate_config()
        
        # パイプラインの生成
        return PredictionPipeline(self.config)
    
    def _validate_config(self):
        """設定の整合性検証"""
        
        # 依存関係のチェック
        if self.config.enable_similarity and not self.config.enable_validation:
            raise ConfigurationError(
                "Similarity calculation requires validation to be enabled"
            )
        
        # 設定値の範囲チェック
        if self.config.prediction_config.confidence_level < 0 or \
           self.config.prediction_config.confidence_level > 1:
            raise ConfigurationError(
                "Confidence level must be between 0 and 1"
            )
```

### 3. PipelineStep 実装例

```python
class ValidationStep(PipelineStep):
    """データ検証ステップ"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.validator = DataValidator(config)
    
    def execute(self, context: PipelineContext) -> StepResult:
        """検証の実行"""
        
        data = context.get_data('new_store_data')
        
        # データ検証
        validation_result = self.validator.validate(data)
        
        # 結果の保存
        context.set_data('validation_result', validation_result)
        context.set_data('quality_score', validation_result.quality_score)
        
        # ログ記録
        if validation_result.is_valid:
            context.add_log("Data validation passed", "INFO")
        else:
            context.add_log(
                f"Data validation failed: {validation_result.errors}", 
                "ERROR"
            )
        
        return StepResult(
            success=validation_result.is_valid or not self.config.strict_mode,
            data=validation_result,
            error_message="; ".join(validation_result.errors) if validation_result.errors else None
        )
    
    def validate_input(self, context: PipelineContext) -> bool:
        """入力検証"""
        return context.get_data('new_store_data') is not None
```

### 4. 設定管理システム

```python
@dataclass
class PipelineConfig:
    """パイプライン設定クラス"""
    
    # ステップ有効化フラグ
    enable_validation: bool = True
    enable_preprocessing: bool = True
    enable_quality_check: bool = True
    enable_anomaly_detection: bool = True
    enable_similarity: bool = True
    enable_explanation: bool = True
    
    # エラーハンドリング
    strict_mode: bool = False
    error_handling: str = 'continue'  # 'fail_fast' or 'continue'
    
    # パフォーマンス設定
    enable_performance_monitoring: bool = True
    enable_caching: bool = False
    
    # 各ステップの設定
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    preprocessing_config: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    quality_config: QualityConfig = field(default_factory=QualityConfig)
    anomaly_config: AnomalyConfig = field(default_factory=AnomalyConfig)
    similarity_config: SimilarityConfig = field(default_factory=SimilarityConfig)
    prediction_config: PredictionConfig = field(default_factory=PredictionConfig)
    explanation_config: ExplanationConfig = field(default_factory=ExplanationConfig)

# プリセット設定
PRESET_CONFIGS = {
    'retail': {
        'enable_quality_check': True,
        'enable_anomaly_detection': True,
        'similarity_config': {
            'method': 'dtw',
            'max_results': 10
        },
        'prediction_config': {
            'method': 'weighted_average',
            'normalization': 'first_day_ratio'
        }
    },
    'restaurant': {
        'enable_quality_check': True,
        'enable_anomaly_detection': False,
        'similarity_config': {
            'method': 'correlation',
            'max_results': 15
        },
        'prediction_config': {
            'method': 'exponential_growth',
            'normalization': 'mean_ratio'
        }
    },
    'service': {
        'enable_quality_check': False,
        'enable_anomaly_detection': True,
        'similarity_config': {
            'method': 'cosine',
            'max_results': 20
        },
        'prediction_config': {
            'method': 'linear_regression',
            'normalization': 'z_score'
        }
    }
}
```

### 5. パフォーマンス監視

```python
class PerformanceMonitor:
    """パフォーマンス監視クラス"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.start_times = {}
    
    def start_timer(self, name: str):
        """タイマー開始"""
        self.start_times[name] = time.time()
    
    def stop_timer(self, name: str):
        """タイマー停止と記録"""
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.metrics[f"{name}_time"].append(elapsed)
            del self.start_times[name]
            return elapsed
        return None
    
    def record_metric(self, name: str, value: float):
        """メトリクス記録"""
        self.metrics[name].append(value)
    
    def get_metrics(self) -> Dict[str, Any]:
        """メトリクスサマリーの取得"""
        summary = {}
        
        for name, values in self.metrics.items():
            if values:
                summary[name] = {
                    'count': len(values),
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'total': np.sum(values)
                }
        
        return summary
```

### 6. エラーハンドリング

```python
class PipelineErrorHandler:
    """パイプラインエラーハンドラー"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.error_recovery_strategies = {
            ValidationError: self._handle_validation_error,
            DataProcessingError: self._handle_data_error,
            PredictionError: self._handle_prediction_error,
            ConfigurationError: self._handle_config_error
        }
    
    def handle_error(self, error: Exception, context: PipelineContext) -> Optional[Any]:
        """エラーハンドリング"""
        
        error_type = type(error)
        
        # 特定のエラータイプに対する処理
        if error_type in self.error_recovery_strategies:
            return self.error_recovery_strategies[error_type](error, context)
        
        # デフォルトのエラー処理
        if self.config.error_handling == 'fail_fast':
            raise error
        else:
            context.add_log(f"Unhandled error: {str(error)}", "ERROR")
            return None
    
    def _handle_validation_error(self, error: ValidationError, context: PipelineContext):
        """検証エラーの処理"""
        
        if self.config.strict_mode:
            raise error
        
        # 警告として記録して続行
        context.add_log(f"Validation warning: {str(error)}", "WARNING")
        
        # デフォルト値で続行
        return ValidationResult(
            is_valid=False,
            errors=[str(error)],
            warnings=[],
            quality_score=50.0
        )
```

## テスト設計

### 単体テスト

```python
class TestPredictionPipeline:
    def test_basic_pipeline_execution(self):
        """基本的なパイプライン実行テスト"""
        
    def test_step_failure_handling(self):
        """ステップ失敗時のハンドリングテスト"""
        
    def test_performance_monitoring(self):
        """パフォーマンス監視機能のテスト"""

class TestPipelineBuilder:
    def test_builder_pattern(self):
        """ビルダーパターンのテスト"""
        
    def test_preset_loading(self):
        """プリセット読み込みテスト"""
        
    def test_config_validation(self):
        """設定検証テスト"""
```

### 統合テスト

```python
class TestPipelineIntegration:
    def test_end_to_end_prediction(self):
        """エンドツーエンド予測テスト"""
        
    def test_error_recovery(self):
        """エラー回復機能テスト"""
        
    def test_concurrent_execution(self):
        """並行実行テスト"""
```

## 関連ドキュメント

- [要件仕様書](./requirements.md)
- [実装タスク](./tasks.md)
- [全体アーキテクチャ](../architecture.md)