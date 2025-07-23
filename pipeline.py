"""
TwinStore パイプライン

データの前処理から予測、レポート生成までの一連の処理を
統合的に実行するパイプライン機能を提供する。
"""

from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
import json
import logging
from pathlib import Path
import warnings

from .core.predictor import SalesPredictor, PredictionResult
from .core.similarity import SimilarityEngine
from .core.normalizer import DataNormalizer
from .core.explainer import PredictionExplainer
from .data.validator import DataValidator, ValidationResult
from .data.preprocessor import DataPreprocessor
from .data.quality_checker import QualityChecker, QualityReport
from .data.anomaly_detector import AnomalyDetector
from .data.loader import DataLoader
from .config import TIME_SERIES_CONSTANTS, QUALITY_CONSTANTS


# ロギング設定
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """パイプライン設定"""
    # 検証設定
    validate_input: bool = True
    strict_validation: bool = False
    min_days: int = TIME_SERIES_CONSTANTS['RECOMMENDED_DAYS']  # バリデーション時の最小必要日数
    
    # 前処理設定
    preprocess_data: bool = True
    handle_missing: bool = True
    handle_outliers: bool = True
    smooth_data: bool = False
    
    # 品質チェック設定
    check_quality: bool = True
    quality_threshold: float = QUALITY_CONSTANTS['MIN_ACCEPTABLE_SCORE']
    
    # 予測設定
    similarity_metric: str = "dtw"
    normalization_method: str = "z-score"
    n_similar_stores: int = 5
    confidence_level: float = 0.95
    auto_optimize_period: bool = True
    
    # 説明生成設定
    generate_explanation: bool = True
    explanation_language: str = "ja"
    
    # 出力設定
    save_results: bool = False
    output_dir: Optional[str] = None
    output_format: str = "json"
    
    # ファイルローダー設定
    date_column: Optional[str] = None
    sales_column: Optional[str] = None
    store_cd_column: Optional[str] = None
    auto_detect_columns: bool = True
    
    # ログ設定
    verbose: bool = True
    log_level: str = "INFO"


@dataclass
class PipelineResult:
    """パイプライン実行結果"""
    prediction: PredictionResult
    validation_result: Optional[ValidationResult] = None
    quality_report: Optional[QualityReport] = None
    explanation: Optional[str] = None
    preprocessing_report: Optional[str] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """辞書形式に変換"""
        result_dict = {
            "prediction": {
                "annual_sales": self.prediction.prediction,
                "lower_bound": self.prediction.lower_bound,
                "upper_bound": self.prediction.upper_bound,
                "confidence_score": self.prediction.confidence_score,
                "similar_stores": self.prediction.similar_stores,
                "method": self.prediction.prediction_method,
            },
            "execution_time": self.execution_time,
            "metadata": self.metadata,
            "warnings": self.warnings,
        }
        
        if self.validation_result:
            result_dict["validation"] = {
                "is_valid": self.validation_result.is_valid,
                "errors": self.validation_result.errors,
                "warnings": self.validation_result.warnings,
            }
        
        if self.quality_report:
            result_dict["quality"] = {
                "overall_score": self.quality_report.overall_score,
                "completeness": self.quality_report.completeness_score,
                "consistency": self.quality_report.consistency_score,
                "accuracy": self.quality_report.accuracy_score,
                "issues": len(self.quality_report.issues),
            }
        
        if self.explanation:
            result_dict["explanation"] = self.explanation
        
        return result_dict


class PredictionPipeline:
    """
    売上予測パイプライン
    
    データの検証から予測、レポート生成までの一連の処理を実行する。
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Parameters
        ----------
        config : PipelineConfig, optional
            パイプライン設定（指定しない場合はデフォルト設定）
        """
        self.config = config or PipelineConfig()
        self._setup_logging()
        self._components = {}
        self._initialize_components()
    
    def _setup_logging(self):
        """ロギングの設定"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_components(self):
        """コンポーネントの初期化"""
        self._components["validator"] = DataValidator(strict_mode=self.config.strict_validation)
        self._components["preprocessor"] = DataPreprocessor()
        self._components["quality_checker"] = QualityChecker()
        self._components["predictor"] = SalesPredictor(
            similarity_metric=self.config.similarity_metric,
            normalization_method=self.config.normalization_method
        )
        self._components["explainer"] = PredictionExplainer(language=self.config.explanation_language)
        self._components["anomaly_detector"] = AnomalyDetector()
        self._components["loader"] = DataLoader(
            date_column=self.config.date_column,
            sales_column=self.config.sales_column,
            store_cd_column=self.config.store_cd_column
        )
    
    def fit(
        self,
        historical_data: Union[pd.DataFrame, Dict[str, np.ndarray], str, Path],
        store_attributes: Optional[Union[pd.DataFrame, str, Path]] = None,
        **loader_kwargs: Any
    ) -> "PredictionPipeline":
        """
        過去データで学習
        
        Parameters
        ----------
        historical_data : pd.DataFrame, dict, str, or Path
            過去の店舗売上データ、またはファイルパス
        store_attributes : pd.DataFrame, str, Path, optional
            店舗属性データ、またはファイルパス
        **loader_kwargs
            DataLoaderに渡すオプション引数
            
        Returns
        -------
        self
            学習済みのパイプライン
        """
        logger.info("Starting pipeline training...")
        
        # ファイルパスの場合はDataLoaderで読み込み
        if isinstance(historical_data, (str, Path)):
            logger.info(f"Loading historical data from file: {historical_data}")
            historical_data = self._components["loader"].load_historical_data(
                historical_data, output_format="dict", **loader_kwargs
            )
        
        if isinstance(store_attributes, (str, Path)):
            logger.info(f"Loading store attributes from file: {store_attributes}")
            store_attributes = self._components["loader"].load_store_attributes(
                store_attributes, **loader_kwargs
            )
        
        # データ検証
        if self.config.validate_input:
            val_result = self._components["validator"].validate_sales_data(historical_data)
            if not val_result.is_valid:
                raise ValueError(f"Invalid training data: {val_result.errors}")
        
        # 前処理
        if self.config.preprocess_data:
            if isinstance(historical_data, pd.DataFrame):
                processed_data = self._components["preprocessor"].preprocess(
                    historical_data,
                    handle_missing=self.config.handle_missing,
                    handle_outliers=self.config.handle_outliers,
                )
                historical_data = processed_data
            else:
                # 辞書形式の場合は各店舗ごとに処理
                processed_dict = {}
                for store_cd, data in historical_data.items():
                    processed = self._components["preprocessor"].preprocess(
                        data,
                        handle_missing=self.config.handle_missing,
                        handle_outliers=self.config.handle_outliers,
                    )
                    processed_dict[store_cd] = processed
                historical_data = processed_dict
        
        # 予測器の学習
        self._components["predictor"].fit(historical_data, store_attributes)
        
        # 異常検知器の学習（最初の店舗データを使用）
        if isinstance(historical_data, dict):
            first_store_data = next(iter(historical_data.values()))
        else:
            first_store_data = historical_data.iloc[:, 0]
        self._components["anomaly_detector"].update_model(first_store_data)
        
        logger.info("Pipeline training completed")
        return self
    
    def predict(
        self,
        new_store_sales: Union[np.ndarray, pd.Series, List[float]],
        store_name: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
        save_results: Optional[bool] = None,
    ) -> PipelineResult:
        """
        新規店舗の売上を予測
        
        Parameters
        ----------
        new_store_sales : array-like
            新規店舗の売上データ
        store_name : str, optional
            店舗名（レポート用）
        filters : dict, optional
            類似店舗検索時のフィルタ条件
        save_results : bool, optional
            結果を保存するか（設定を上書き）
            
        Returns
        -------
        PipelineResult
            パイプライン実行結果
        """
        start_time = datetime.now()
        logger.info(f"Starting prediction pipeline for {store_name or 'new store'}...")
        
        result = PipelineResult(
            prediction=None,
            metadata={"store_name": store_name, "start_time": start_time.isoformat()}
        )
        
        try:
            # 前処理を先に実行（バリデーション前に問題を修正）
            processed_sales = new_store_sales
            if self.config.preprocess_data:
                logger.debug("Preprocessing data...")
                processed_sales = self._components["preprocessor"].preprocess(
                    new_store_sales,
                    handle_missing=self.config.handle_missing,
                    handle_outliers=self.config.handle_outliers,
                    smooth_data=self.config.smooth_data,
                )
                result.preprocessing_report = self._components["preprocessor"].get_preprocessing_report()
            
            # 1. データ検証（前処理後のデータに対して）
            if self.config.validate_input:
                logger.debug("Validating input data...")
                val_result = self._components["validator"].validate_prediction_input(
                    processed_sales, 
                    min_days=self.config.min_days
                )
                result.validation_result = val_result
                
                if not val_result.is_valid:
                    raise ValueError(f"Invalid input data: {val_result.errors}")
            
            # 2. 品質チェック（前処理後のデータに対して）
            if self.config.check_quality:
                logger.debug("Checking data quality...")
                quality_report = self._components["quality_checker"].check_data_quality(processed_sales)
                result.quality_report = quality_report
                
                if quality_report.overall_score < self.config.quality_threshold:
                    result.warnings.append(
                        f"Data quality score ({quality_report.overall_score:.1f}) "
                        f"is below threshold ({self.config.quality_threshold})"
                    )
            
            # 3. 異常検知
            anomalies = self._components["anomaly_detector"].detect_anomalies(processed_sales)
            anomaly_ratio = np.mean(anomalies)
            if anomaly_ratio > 0.1:
                result.warnings.append(f"High anomaly ratio detected: {anomaly_ratio:.1%}")
            
            # 4. 最適期間の決定
            if self.config.auto_optimize_period:
                logger.debug("Optimizing matching period...")
                optimal_period = self._components["predictor"].suggest_matching_period(processed_sales)
                matching_days = optimal_period.recommended_days
                result.metadata["optimal_period"] = {
                    "recommended_days": matching_days,
                    "stability_score": optimal_period.stability_score,
                }
            else:
                matching_days = None
            
            # 5. 予測実行
            logger.debug("Executing prediction...")
            prediction = self._components["predictor"].predict(
                processed_sales,
                matching_period_days=matching_days,
                n_similar=self.config.n_similar_stores,
                filters=filters,
                confidence_level=self.config.confidence_level,
            )
            result.prediction = prediction
            
            # 6. 説明生成
            if self.config.generate_explanation:
                logger.debug("Generating explanation...")
                explanation = self._components["explainer"].generate_explanation(
                    prediction,
                    processed_sales,
                    self._components["predictor"].historical_data,
                    self._components["predictor"].store_attributes,
                )
                result.explanation = explanation
            
            # 実行時間の記録
            end_time = datetime.now()
            result.execution_time = (end_time - start_time).total_seconds()
            result.metadata["end_time"] = end_time.isoformat()
            
            # 7. 結果の保存
            if save_results or (save_results is None and self.config.save_results):
                self._save_results(result, store_name)
            
            logger.info(f"Prediction pipeline completed in {result.execution_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            raise
        
        return result
    
    def batch_predict(
        self,
        stores_data: Union[Dict[str, Union[np.ndarray, pd.Series, List[float]]], str, Path],
        filters: Optional[Dict[str, Any]] = None,
        parallel: bool = False,
        **loader_kwargs: Any
    ) -> Dict[str, PipelineResult]:
        """
        複数店舗の一括予測
        
        Parameters
        ----------
        stores_data : dict, str, or Path
            店舗コードと売上データの辞書、またはファイルパス
        filters : dict, optional
            類似店舗検索時のフィルタ条件
        parallel : bool, default=False
            並列処理を使用するか
        **loader_kwargs
            DataLoaderに渡すオプション引数
            
        Returns
        -------
        dict
            店舗コードと予測結果の辞書
        """
        # ファイルパスの場合はDataLoaderで読み込み
        if isinstance(stores_data, (str, Path)):
            logger.info(f"Loading batch data from file: {stores_data}")
            stores_data = self._components["loader"].load_new_store_data(
                stores_data, **loader_kwargs
            )
        
        logger.info(f"Starting batch prediction for {len(stores_data)} stores...")
        results = {}
        
        if parallel:
            # 並列処理（将来の実装）
            warnings.warn("Parallel processing not yet implemented. Using sequential processing.")
        
        # 逐次処理
        for store_cd, sales_data in stores_data.items():
            try:
                result = self.predict(
                    sales_data,
                    store_name=store_cd,
                    filters=filters,
                    save_results=False,  # バッチ処理では個別保存しない
                )
                results[store_cd] = result
            except Exception as e:
                logger.error(f"Failed to predict for store {store_cd}: {str(e)}")
                # エラーの場合も結果に含める
                results[store_cd] = PipelineResult(
                    prediction=None,
                    warnings=[f"Prediction failed: {str(e)}"],
                    metadata={"error": str(e), "store_cd": store_cd}
                )
        
        # バッチ結果の保存
        if self.config.save_results:
            self._save_batch_results(results)
        
        logger.info(f"Batch prediction completed for {len(results)} stores")
        return results
    
    def create_custom_pipeline(
        self,
        steps: List[Tuple[str, Callable[..., Any]]],
        name: str = "custom"
    ) -> Callable[[Any], Any]:
        """
        カスタムパイプラインを作成
        
        Parameters
        ----------
        steps : list
            (ステップ名, 関数) のリスト
        name : str, default='custom'
            パイプライン名
            
        Returns
        -------
        callable
            カスタムパイプライン関数
        """
        def custom_pipeline(data, **kwargs):
            logger.info(f"Executing custom pipeline: {name}")
            result = data
            
            for step_name, step_func in steps:
                logger.debug(f"Executing step: {step_name}")
                try:
                    result = step_func(result, **kwargs.get(step_name, {}))
                except Exception as e:
                    logger.error(f"Step {step_name} failed: {str(e)}")
                    raise
            
            return result
        
        return custom_pipeline
    
    def update_config(self, **kwargs: Any) -> None:
        """
        設定を更新
        
        Parameters
        ----------
        **kwargs
            更新する設定項目
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.debug(f"Updated config: {key} = {value}")
            else:
                warnings.warn(f"Unknown config parameter: {key}")
        
        # コンポーネントの再初期化が必要な場合
        if any(key in ["similarity_metric", "normalization_method", "explanation_language"] for key in kwargs):
            # 学習データを保存
            historical_data = getattr(self._components.get("predictor", None), 'historical_data', None)
            store_attributes = getattr(self._components.get("predictor", None), 'store_attributes', None)
            
            # コンポーネントを再初期化
            self._initialize_components()
            
            # 学習データを復元
            if historical_data is not None:
                self._components["predictor"].historical_data = historical_data
            if store_attributes is not None:
                self._components["predictor"].store_attributes = store_attributes
    
    def _save_results(self, result: PipelineResult, store_name: Optional[str]):
        """結果を保存"""
        if not self.config.output_dir:
            self.config.output_dir = "twinstore_results"
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル名の生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        store_suffix = f"_{store_name}" if store_name else ""
        filename = f"prediction{store_suffix}_{timestamp}"
        
        # 形式に応じて保存
        if self.config.output_format == "json":
            filepath = output_dir / f"{filename}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, ensure_ascii=False, indent=2, default=str)
        elif self.config.output_format == "csv":
            filepath = output_dir / f"{filename}.csv"
            df = pd.DataFrame([result.to_dict()])
            df.to_csv(filepath, index=False)
        else:
            warnings.warn(f"Unknown output format: {self.config.output_format}")
            return
        
        logger.info(f"Results saved to: {filepath}")
    
    def _save_batch_results(self, results: Dict[str, PipelineResult]):
        """バッチ結果を保存"""
        if not self.config.output_dir:
            self.config.output_dir = "twinstore_results"
        
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ファイル名の生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"batch_prediction_{timestamp}"
        
        # 結果を整形
        batch_data = {}
        for store_cd, result in results.items():
            batch_data[store_cd] = result.to_dict()
        
        # 保存
        if self.config.output_format == "json":
            filepath = output_dir / f"{filename}.json"
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(batch_data, f, ensure_ascii=False, indent=2, default=str)
        elif self.config.output_format == "csv":
            filepath = output_dir / f"{filename}.csv"
            # CSVの場合は各店舗を行として保存
            rows = []
            for store_cd, result_dict in batch_data.items():
                row = {"store_cd": store_cd}
                # フラット化
                if "prediction" in result_dict and result_dict["prediction"]:
                    row.update({
                        f"prediction_{k}": v 
                        for k, v in result_dict["prediction"].items()
                        if not isinstance(v, (list, dict))
                    })
                if "quality" in result_dict:
                    row.update({
                        f"quality_{k}": v 
                        for k, v in result_dict["quality"].items()
                    })
                rows.append(row)
            
            df = pd.DataFrame(rows)
            df.to_csv(filepath, index=False)
        
        logger.info(f"Batch results saved to: {filepath}")


class PipelineBuilder:
    """
    パイプラインビルダー
    
    設定を段階的に構築してパイプラインを作成する。
    """
    
    def __init__(self):
        self._config = PipelineConfig()
    
    def with_validation(self, strict: bool = False) -> "PipelineBuilder":
        """検証設定"""
        self._config.validate_input = True
        self._config.strict_validation = strict
        return self
    
    def with_preprocessing(
        self,
        handle_missing: bool = True,
        handle_outliers: bool = True,
        smooth: bool = False
    ) -> "PipelineBuilder":
        """前処理設定"""
        self._config.preprocess_data = True
        self._config.handle_missing = handle_missing
        self._config.handle_outliers = handle_outliers
        self._config.smooth_data = smooth
        return self
    
    def with_quality_check(self, threshold: float = 70.0) -> "PipelineBuilder":
        """品質チェック設定"""
        self._config.check_quality = True
        self._config.quality_threshold = threshold
        return self
    
    def with_prediction(
        self,
        metric: str = "dtw",
        normalization: str = "z-score",
        n_similar: int = 5
    ) -> "PipelineBuilder":
        """予測設定"""
        self._config.similarity_metric = metric
        self._config.normalization_method = normalization
        self._config.n_similar_stores = n_similar
        return self
    
    def with_explanation(self, language: str = "ja") -> "PipelineBuilder":
        """説明生成設定"""
        self._config.generate_explanation = True
        self._config.explanation_language = language
        return self
    
    def with_output(
        self,
        save: bool = True,
        output_dir: str = "twinstore_results",
        format: str = "json"
    ) -> "PipelineBuilder":
        """出力設定"""
        self._config.save_results = save
        self._config.output_dir = output_dir
        self._config.output_format = format
        return self
    
    def build(self) -> PredictionPipeline:
        """パイプラインを構築"""
        return PredictionPipeline(self._config)