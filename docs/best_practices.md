# ベストプラクティス

## 目次

- [データ準備](#データ準備)
- [予測精度の向上](#予測精度の向上)
- [パフォーマンス最適化](#パフォーマンス最適化)
- [エラー処理](#エラー処理)
- [プロダクション環境での使用](#プロダクション環境での使用)

## データ準備

### 1. データ品質の確保

```python
# 必ずデータ品質をチェック
from twinstore.data import QualityChecker

checker = QualityChecker()
report = checker.check_data_quality(your_data)

if report.overall_score < 70:
    print("データ品質に問題があります:")
    print(report.get_summary())
```

### 2. 適切なデータ量の確保

- **最小要件**: 3日分のデータ
- **推奨**: 7日以上のデータ
- **理想的**: 14-30日のデータ

```python
# データ量のチェック
if len(new_store_sales) < 7:
    warnings.warn("データが7日未満です。予測精度が低下する可能性があります。")
```

### 3. 欠損値の処理

```python
# パイプラインで自動処理
config = PipelineConfig(
    handle_missing=True,  # 欠損値を自動補完
    preprocess_data=True  # 前処理を有効化
)
```

## 予測精度の向上

### 1. 類似店舗の選定

```python
# 店舗属性でフィルタリング
result = predictor.predict(
    new_store_sales,
    filters={
        'store_type': new_store_attr['type'],
        'area_range': (new_store_attr['area'] * 0.8, new_store_attr['area'] * 1.2)
    }
)
```

### 2. 期間最適化の活用

```python
# 自動で最適な期間を探索
config = PipelineConfig(optimize_period=True)
pipeline = PredictionPipeline(config)

# または手動で最適化
optimizer = predictor.create_period_optimizer()
optimal_period = optimizer.find_optimal_period(new_store_sales)
```

### 3. 正規化手法の選択

```python
# データの特性に応じて選択
if has_different_scales:
    predictor = SalesPredictor(normalization='zscore')
elif has_growth_trend:
    predictor = SalesPredictor(normalization='first_value')
else:
    predictor = SalesPredictor(normalization='minmax')
```

## パフォーマンス最適化

### 1. バッチ処理の活用

```python
# 複数店舗を並列処理
batch_results = pipeline.run_batch(
    historical_data=historical_data,
    new_stores_data=stores_dict,
    parallel=True
)
```

### 2. キャッシュの利用

```python
# 結果をキャッシュ
import joblib

# 予測結果を保存
joblib.dump(predictor, 'predictor_model.pkl')

# 再利用
predictor = joblib.load('predictor_model.pkl')
```

### 3. メモリ効率の改善

```python
# 大規模データの場合
predictor = SalesPredictor()
predictor.fit_batch(
    historical_data,
    batch_size=100  # 100店舗ずつ処理
)
```

## エラー処理

### 1. 適切な例外処理

```python
try:
    result = predictor.predict(new_store_sales)
except ValueError as e:
    if "Insufficient data" in str(e):
        # データ不足の場合の処理
        result = use_fallback_method()
    else:
        raise
```

### 2. バリデーションの活用

```python
from twinstore.data import DataValidator

validator = DataValidator()
validation_result = validator.validate_new_store_data(new_store_sales)

if not validation_result.is_valid:
    for error in validation_result.errors:
        print(f"エラー: {error}")
```

### 3. ログの活用

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

predictor = SalesPredictor(debug=True)
```

## プロダクション環境での使用

### 1. API化

```python
from fastapi import FastAPI
from twinstore.api import create_app

# FastAPIアプリケーション作成
app = create_app(predictor)

# エンドポイント例
@app.post("/predict")
async def predict_sales(data: PredictionRequest):
    result = predictor.predict(data.sales_data)
    return {"prediction": result.prediction}
```

### 2. モニタリング

```python
# 予測精度のトラッキング
class PredictionMonitor:
    def __init__(self):
        self.predictions = []
        self.actuals = []
    
    def add_prediction(self, store_cd, prediction, actual=None):
        self.predictions.append({
            'store_cd': store_cd,
            'prediction': prediction,
            'actual': actual,
            'timestamp': datetime.now()
        })
    
    def calculate_accuracy(self):
        # MAPEを計算
        valid_data = [(p['prediction'], p['actual']) 
                      for p in self.predictions 
                      if p['actual'] is not None]
        
        if valid_data:
            predictions, actuals = zip(*valid_data)
            mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
            return mape
```

### 3. 継続的改善

```python
# オンライン学習の実装
from twinstore.optimization import OnlineOptimizer

online_optimizer = OnlineOptimizer(predictor)

# 定期的に実行
def update_model(new_data):
    # 新しいデータで学習
    predictor.partial_fit(new_data)
    
    # パラメータの最適化
    if len(new_data) >= 10:
        online_optimizer.optimize_parameters(new_data)
```

### 4. スケーラビリティ

```python
# Redis/Memcachedでのキャッシング
import redis
import json

redis_client = redis.Redis()

def cached_predict(store_cd, sales_data):
    # キャッシュキー
    cache_key = f"prediction:{store_cd}:{hash(str(sales_data))}"
    
    # キャッシュチェック
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)
    
    # 予測実行
    result = predictor.predict(sales_data)
    
    # キャッシュ保存（1時間）
    redis_client.setex(
        cache_key, 
        3600, 
        json.dumps(result.to_dict())
    )
    
    return result
```

## まとめ

これらのベストプラクティスに従うことで：

1. **予測精度の向上**: 適切なデータ準備と手法選択
2. **安定性の確保**: エラー処理とバリデーション
3. **スケーラビリティ**: バッチ処理とキャッシング
4. **保守性**: ログとモニタリング

を実現できます。