# AIアシスタント プロジェクトガイド

このプロジェクトで作業する際のガイダンスを提供します。
下記のAI行動原則は常に遵守してください。

## AI行動原則

1. **応答時**
   - 応答時には、必ずこのAI行動原則を画面出力してから対応する

2. **開発方針**
   - テストを作成しながら進める。
   - テストカバレッジは90％以上を維持する。
   - テスト成功率は100％を維持する。
   - スペック駆動開発システムを遵守して進める。

## スペック駆動開発システム

あなたはスペック駆動開発を行うAIエージェントです。要件→設計→タスク→実装の構造化されたプロセスでアプリケーションを構築します。

### プロジェクト構造

```
project-root/
├── spec/
│   ├── requirements.md      # 全体要件仕様書
│   ├── design.md           # 全体設計仕様書
│   ├── tasks.md            # 全体タスク管理
│   ├── architecture.md     # システムアーキテクチャ
│   ├── authentication/     # 認証機能
│   │   ├── requirements.md
│   │   ├── design.md
│   │   └── tasks.md
│   ├── user-management/     # ユーザー管理機能
│   │   ├── requirements.md
│   │   ├── design.md
│   │   └── tasks.md
│   └── [feature-name]/     # その他の機能
│       ├── requirements.md
│       ├── design.md
│       └── tasks.md
└── [プロジェクト固有のファイル・フォルダ構造]
```

**注意**: APIが必要なプロジェクトの場合は、各機能フォルダに `api-spec.md` を追加してください。

### スペック駆動開発フロー

#### 1. 要件定義フェーズ (Requirements)
ユーザーの要求を受け取ったら、以下の手順で要件を定義：

1. **全体要件の整理**
   - `spec/requirements.md` で全体要件を定義
   - 機能一覧と優先度を明確化

2. **機能別要件の作成**
   - `spec/[feature-name]/requirements.md` で機能要件を詳細化
   - EARS表記法でユーザーストーリーを記述

#### 2. 設計フェーズ (Design)
要件が確定したら、以下の設計ドキュメントを作成：

1. **全体設計**
   - `spec/design.md` でシステム全体設計
   - `spec/architecture.md` でアーキテクチャ詳細

2. **機能別設計**
   - `spec/[feature-name]/design.md` で機能設計
   - 必要に応じて `spec/[feature-name]/api-spec.md` でAPI仕様

#### 3. タスク分解フェーズ (Tasks)
設計が完了したら、実装タスクを分解：

1. **全体タスク管理**
   - `spec/tasks.md` で全体スケジュール管理

2. **機能別タスク**
   - `spec/[feature-name]/tasks.md` で機能別実装タスク
   - Phase別に構造化（基盤→コア→品質→運用）

#### 4. 実装フェーズ (Implementation)
タスクに従って実装を進行：

1. **実装の進行**
   - `spec/[feature-name]/tasks.md` のタスクに従って実装
   - 仕様書との整合性を保持しながら開発

### 機能要件テンプレート

```markdown
# [機能名] 要件仕様書

**作成日**: {DATE}
**最終更新**: {DATE}
**ステータス**: Draft

## 機能概要
[機能の目的と概要]

## ユーザーストーリー
### US-[ID]: [ストーリー名]
**As a** [ユーザー役割]
**I want** [望む機能]
**So that** [目的・価値]

**受け入れ基準:**
- GIVEN [前提条件]
- WHEN [実行条件]
- THEN [期待結果]

## 機能仕様
- [基本機能1]
- [基本機能2]
- [基本機能3]

## 非機能要件
- **パフォーマンス**: [要件]
- **セキュリティ**: [要件]
- **可用性**: [要件]

## 依存関係
- **前提機能**: [依存する他機能]
- **影響範囲**: [影響を与える他機能]
```

### 機能設計テンプレート

```markdown
# [機能名] 設計仕様書

**作成日**: {DATE}
**関連要件**: [requirements.md](./requirements.md)

## 設計概要
[実装方針と設計思想]

## データモデル
```typescript
interface [ModelName] {
  id: string;
  // フィールド定義
}
```

### インターフェース設計
[UIコンポーネント、関数、クラス等の設計]

### API設計（必要に応じて）
| Method | Path | Description | Request | Response |
|--------|------|-------------|---------|----------|
| GET | /api/[resource] | [説明] | [型] | [型] |
| POST | /api/[resource] | [説明] | [型] | [型] |

### 実装方針
- [技術選択]
- [実装パターン]
- [エラーハンドリング]


### 機能タスクテンプレート

```markdown
# [機能名] 実装タスク

**関連仕様**: [requirements.md](./requirements.md) | [design.md](./design.md)

## Phase 1: 基盤構築
- [ ] データベース設計
- [ ] API基盤実装
- [ ] 認証・認可設定

## Phase 2: コア機能実装
- [ ] [機能1] 実装
- [ ] [機能2] 実装
- [ ] [機能3] 実装

## Phase 3: 品質保証
- [ ] ユニットテスト
- [ ] 統合テスト
- [ ] E2Eテスト

## Phase 4: 運用準備
- [ ] 監視・ログ設定
- [ ] セキュリティテスト
- [ ] デプロイメント準備
```

### 実行指針

#### 新規プロジェクト開始時
```markdown
1. **プロジェクト概要をお聞かせください**
   - アプリケーション概要
   - 主要機能リスト
   - 技術的要望

2. **機能の洗い出し**
   - 必要機能の特定
   - 機能別フォルダ作成
   - 機能間依存関係の整理

3. **スペック駆動開発の開始**
   - 全体要件書作成
   - 機能別要件書作成
   - 設計書作成
   - タスク分解
```

#### 機能追加時
```markdown
1. **新機能の要件確認**
   - 機能目的と概要
   - 既存機能との関係

2. **機能フォルダ作成**
   - spec/[new-feature]/ 作成
   - 要件・設計・タスクファイル生成

3. **仕様書作成**
   - requirements.md 作成
   - design.md 作成
   - tasks.md 作成
   - api-spec.md 作成（APIが必要な場合）
```

#### 実装進行時
```markdown
1. **タスク進捗の確認**
   - 現在の実装状況
   - 完了タスクの確認

2. **仕様書との整合性確認**
   - 実装内容が仕様書と一致しているか
   - 必要に応じて仕様書を更新

3. **次のタスクの実行**
   - 優先度に従ったタスク実行
   - 依存関係を考慮した順序
```

---

## プロジェクト概要

TwinStore（ツインストア）は、新規出店後の限られた売上データから年間売上を高精度で予測するPythonパッケージです。類似店舗マッチング技術とDTW（動的時間伸縮法）を使用して、最小3日分のデータから予測が可能です。

## 開発コマンド

### 環境セットアップ
```bash
# 仮想環境の作成と有効化
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# 開発用依存関係のインストール
pip install -e ".[dev]"
```

### テスト
```bash
# 全テストを実行
pytest

# カバレッジ付きでテスト実行
pytest --cov=twinstore

# 特定のテストファイルを実行
pytest tests/test_pipeline.py

# 単一のテストを実行
pytest tests/test_pipeline.py::TestPipelineBuilder::test_build_with_all_options
```

### コード品質
```bash
# コードフォーマット
black .
isort .

# リンティング
flake8 twinstore tests

# 型チェック
mypy twinstore
```

### パッケージビルド
```bash
python -m build
```

## アーキテクチャ概要

### コア設計パターン

1. **パイプライン・アーキテクチャ**
   - `PredictionPipeline`が中心となり、データ検証→前処理→品質チェック→予測→説明生成の流れを管理
   - 各ステップは独立したコンポーネントとして実装（SRP原則）

2. **Strategy パターン**
   - 類似性計算: DTW、コサイン類似度、相関係数を切り替え可能
   - 正規化手法: z-score、min-max、robust等を選択可能
   - 異常値検出: IQR、Z-score、Isolation Forest（実装予定）

3. **Builder パターン**
   - `PipelineBuilder`でパイプラインを段階的に構築
   - メソッドチェーンによる直感的なAPI

### モジュール構造

```
twinstore/
├── core/           # 予測エンジンのコア機能
│   ├── predictor.py      # メイン予測クラス（SalesPredictor）
│   ├── similarity.py     # 類似性計算エンジン（DTW実装）
│   ├── normalizer.py     # データ正規化
│   └── explainer.py      # 予測結果の説明生成
├── data/           # データ処理
│   ├── loader.py         # 多様な形式のデータ読み込み
│   ├── validator.py      # データ検証（ValidationResult）
│   ├── preprocessor.py   # 前処理（欠損値補完等）
│   ├── quality_checker.py # 4つの観点での品質評価
│   └── anomaly_detector.py # 異常値検出
├── pipeline.py     # 統合パイプライン
├── config/         # 設定管理
│   ├── constants.py      # 数値定数
│   ├── defaults.py       # デフォルト設定
│   └── validation.py     # 検証ルール
└── types.py        # 型定義
```

### 重要な設計決定

1. **型安全性**: Pydanticによるスキーマ検証と明確な型ヒント
2. **エラーハンドリング**: ValidationResultでエラーと警告を分離
3. **拡張性**: 業態別プリセット（retail、restaurant、service）をサポート
4. **データ形式**: CSV、Excel、JSON、DataFrame、NumPy配列、辞書をサポート

### データフロー

```
入力データ → DataValidator → DataPreprocessor → QualityChecker → 
→ SimilarityEngine（DTW計算） → SalesPredictor → PredictionExplainer → 結果
```

### 主要なクラスと責務

- `SalesPredictor`: 売上予測のメインクラス
- `SimilarityEngine`: DTWによる類似店舗検索
- `PredictionPipeline`: 全体のワークフロー管理
- `DataLoader`: 多様なデータ形式の統一的な読み込み
- `ValidationResult`: 検証結果の構造化表現

### 拡張ポイント

コード内のTODOコメントで示された将来の拡張：
- 可視化機能の実装（visualization/）
- レポート生成機能
- Isolation Forestの完全実装
- 並列処理の本格対応

## 開発時の注意点

1. **データ品質**: QualityCheckerが70%以上のスコアを要求（デフォルト）
2. **最小データ要件**: 3日分以上のデータが必要
3. **DTWウィンドウ**: 計算効率のため10%のウィンドウ制約を適用
4. **型チェック**: mypyの厳密モードが有効

## テスト戦略

- 単体テスト: 各コンポーネントの独立したテスト
- 統合テスト: パイプライン全体のエンドツーエンドテスト
- フィクスチャ: conftest.pyで共通のテストデータを定義
- カバレッジ目標: 92%以上（現在達成済み）