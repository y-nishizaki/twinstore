"""
DTWベースの類似性計算エンジン

Dynamic Time Warping (DTW) を使用して時系列データの類似性を計算する
モジュール。店舗の売上推移の類似性を評価するために使用される。
"""

from typing import List, Tuple, Optional, Union, Dict, Any
import numpy as np
import pandas as pd
from dtaidistance import dtw
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr
import warnings
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib


class SimilarityEngine:
    """
    時系列データの類似性を計算するエンジンクラス
    
    DTW距離、コサイン類似度、相関係数など複数の類似性指標をサポート
    """
    
    def __init__(
        self,
        metric: str = "dtw",
        window: Optional[int] = None,
        normalize: bool = True,
        cache_size: int = 1024,
    ):
        """
        Parameters
        ----------
        metric : str, default='dtw'
            使用する類似性指標 ('dtw', 'cosine', 'correlation')
        window : int, optional
            DTWのウィンドウ制約（計算速度向上のため）
        normalize : bool, default=True
            計算前にデータを正規化するかどうか
        cache_size : int, default=1024
            類似度計算のキャッシュサイズ
        """
        self.metric = metric
        self.window = window
        self.normalize = normalize
        self._validate_metric()
        
        # キャッシュの初期化
        self._similarity_cache = {}
    
    def _validate_metric(self):
        """メトリクスの検証"""
        valid_metrics = ["dtw", "cosine", "correlation"]
        if self.metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric: {self.metric}. "
                f"Must be one of {valid_metrics}"
            )
    
    def _get_cache_key(self, data1: np.ndarray, data2: np.ndarray) -> str:
        """データのキャッシュキーを生成"""
        # データのハッシュを計算
        hash1 = hashlib.md5(data1.tobytes()).hexdigest()[:16]
        hash2 = hashlib.md5(data2.tobytes()).hexdigest()[:16]
        # 順序に依存しないキーを生成
        if hash1 < hash2:
            return f"{hash1}_{hash2}_{self.metric}"
        else:
            return f"{hash2}_{hash1}_{self.metric}"
    
    def calculate_similarity(
        self,
        series1: Union[List[float], np.ndarray, pd.Series],
        series2: Union[List[float], np.ndarray, pd.Series],
    ) -> float:
        """
        2つの時系列データ間の類似性を計算
        
        Parameters
        ----------
        series1 : array-like
            比較する時系列データ1
        series2 : array-like
            比較する時系列データ2
            
        Returns
        -------
        float
            類似性スコア（メトリクスにより範囲が異なる）
        """
        # numpy配列に変換
        s1 = np.asarray(series1).flatten()
        s2 = np.asarray(series2).flatten()
        
        # データの検証
        if len(s1) == 0 or len(s2) == 0:
            raise ValueError("Input series cannot be empty")
        
        # キャッシュキーを生成
        cache_key = self._get_cache_key(s1, s2)
        
        # キャッシュから取得を試みる
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        # メトリクスに応じた計算
        if self.metric == "dtw":
            # 正規化
            if self.normalize:
                s1 = self._normalize_series(s1)
                s2 = self._normalize_series(s2)
            similarity = self._calculate_dtw(s1, s2)
        elif self.metric == "cosine":
            # コサイン類似度では正規化をスキップ（原データを使用）
            similarity = self._calculate_cosine_similarity(s1, s2)
        elif self.metric == "correlation":
            # 正規化
            if self.normalize:
                s1 = self._normalize_series(s1)
                s2 = self._normalize_series(s2)
            similarity = self._calculate_correlation(s1, s2)
        
        # キャッシュに保存（サイズ制限付き）
        if len(self._similarity_cache) < 10000:  # 10000エントリまで
            self._similarity_cache[cache_key] = similarity
        
        return similarity
    
    def _normalize_series(self, series: np.ndarray) -> np.ndarray:
        """
        時系列データの正規化（Z-score正規化）
        
        Parameters
        ----------
        series : np.ndarray
            正規化する時系列データ
            
        Returns
        -------
        np.ndarray
            正規化されたデータ
        """
        mean = np.mean(series)
        std = np.std(series)
        
        if std == 0:
            # 標準偏差が0の場合は0で埋める
            return np.zeros_like(series)
        
        return (series - mean) / std
    
    def _calculate_dtw(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        DTW距離を計算（値が小さいほど類似）
        
        Parameters
        ----------
        s1, s2 : np.ndarray
            比較する時系列データ
            
        Returns
        -------
        float
            DTW距離
        """
        # ウィンドウ制約が未設定の場合、Sakoe-Chiba bandの推奨値を使用
        window = self.window
        if window is None:
            # 時系列長の10%をデフォルトウィンドウサイズとする（最小1）
            window = max(int(0.1 * max(len(s1), len(s2))), 1)
        
        distance = dtw.distance(s1, s2, window=window)
        
        # 長さで正規化
        normalized_distance = distance / max(len(s1), len(s2))
        
        return normalized_distance
    
    def _calculate_cosine_similarity(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        コサイン類似度を計算（値が大きいほど類似）
        
        Parameters
        ----------
        s1, s2 : np.ndarray
            比較する時系列データ
            
        Returns
        -------
        float
            コサイン類似度 [0, 1]
        """
        # 長さを揃える
        min_len = min(len(s1), len(s2))
        s1_trimmed = s1[:min_len]
        s2_trimmed = s2[:min_len]
        
        # コサイン距離を計算し、類似度に変換
        try:
            cos_distance = cosine(s1_trimmed, s2_trimmed)
            if np.isnan(cos_distance):
                # 一方または両方がゼロベクトルの場合
                return 0.0
            similarity = 1 - cos_distance
        except ValueError:
            # ゼロベクトル同士の場合など
            similarity = 0.0
        
        return similarity
    
    def _calculate_correlation(self, s1: np.ndarray, s2: np.ndarray) -> float:
        """
        ピアソン相関係数を計算
        
        Parameters
        ----------
        s1, s2 : np.ndarray
            比較する時系列データ
            
        Returns
        -------
        float
            相関係数 [-1, 1]
        """
        # 長さを揃える
        min_len = min(len(s1), len(s2))
        s1_trimmed = s1[:min_len]
        s2_trimmed = s2[:min_len]
        
        if len(s1_trimmed) < 2:
            warnings.warn("Series too short for correlation calculation")
            return 0.0
        
        correlation, _ = pearsonr(s1_trimmed, s2_trimmed)
        
        return correlation
    
    def find_similar_stores(
        self,
        target_series: Union[List[float], np.ndarray, pd.Series],
        candidate_series_dict: Dict[str, Union[List[float], np.ndarray, pd.Series]],
        top_k: int = 5,
        threshold: Optional[float] = None,
    ) -> List[Tuple[str, float]]:
        """
        対象店舗に最も類似した店舗を検索
        
        Parameters
        ----------
        target_series : array-like
            対象店舗の売上時系列データ
        candidate_series_dict : dict
            候補店舗の売上時系列データ（店舗ID: 時系列データ）
        top_k : int, default=5
            返す類似店舗数
        threshold : float, optional
            類似度の閾値（メトリクスにより意味が異なる）
            
        Returns
        -------
        List[Tuple[str, float]]
            (店舗ID, 類似度スコア) のリスト（類似度順）
        """
        similarities = []
        
        for store_cd, series in candidate_series_dict.items():
            try:
                similarity = self.calculate_similarity(target_series, series)
                similarities.append((store_cd, similarity))
            except Exception as e:
                warnings.warn(f"Failed to calculate similarity for {store_cd}: {e}")
                continue
        
        # メトリクスに応じてソート
        if self.metric == "dtw":
            # DTWは距離なので昇順（小さいほど類似）
            similarities.sort(key=lambda x: x[1])
        else:
            # コサイン類似度と相関は降順（大きいほど類似）
            similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 閾値によるフィルタリング
        if threshold is not None:
            if self.metric == "dtw":
                similarities = [(sid, sim) for sid, sim in similarities if sim <= threshold]
            else:
                similarities = [(sid, sim) for sid, sim in similarities if sim >= threshold]
        
        # top_k件を返す
        return similarities[:top_k]
    
    def calculate_similarity_matrix(
        self,
        series_dict: Dict[str, Union[List[float], np.ndarray, pd.Series]],
        parallel: bool = True,
        n_jobs: int = -1
    ) -> pd.DataFrame:
        """
        複数の時系列データ間の類似性行列を計算
        
        Parameters
        ----------
        series_dict : dict
            店舗IDと時系列データの辞書
        parallel : bool, default=True
            並列処理を使用するか
        n_jobs : int, default=-1
            並列処理のワーカー数（-1: 全てのCPUを使用）
            
        Returns
        -------
        pd.DataFrame
            類似性行列（対称行列）
        """
        store_cds = list(series_dict.keys())
        n = len(store_cds)
        similarity_matrix = np.zeros((n, n))
        
        # 対角要素を設定
        for i in range(n):
            if self.metric == "dtw":
                similarity_matrix[i, i] = 0.0
            else:
                similarity_matrix[i, i] = 1.0
        
        if parallel and n > 10:  # 10店舗以上の場合のみ並列処理
            # 並列処理で計算
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
            
            if n_jobs == -1:
                import os
                n_jobs = os.cpu_count() or 1
            
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                future_to_pair = {
                    executor.submit(
                        self.calculate_similarity,
                        series_dict[store_cds[i]],
                        series_dict[store_cds[j]]
                    ): (i, j)
                    for i, j in pairs
                }
                
                for future in as_completed(future_to_pair):
                    i, j = future_to_pair[future]
                    sim = future.result()
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        else:
            # 逐次処理
            for i, store_i in enumerate(store_cds):
                for j in range(i + 1, n):
                    sim = self.calculate_similarity(
                        series_dict[store_i],
                        series_dict[store_cds[j]]
                    )
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        # DataFrameに変換
        df = pd.DataFrame(
            similarity_matrix,
            index=store_cds,
            columns=store_cds
        )
        
        return df
    
    def get_warping_path(
        self,
        series1: Union[List[float], np.ndarray, pd.Series],
        series2: Union[List[float], np.ndarray, pd.Series],
    ) -> Optional[List[Tuple[int, int]]]:
        """
        DTWのワーピングパスを取得（可視化用）
        
        Parameters
        ----------
        series1, series2 : array-like
            比較する時系列データ
            
        Returns
        -------
        List[Tuple[int, int]] or None
            ワーピングパス（DTWメトリクスの場合のみ）
        """
        if self.metric != "dtw":
            warnings.warn("Warping path is only available for DTW metric")
            return None
        
        s1 = np.asarray(series1).flatten()
        s2 = np.asarray(series2).flatten()
        
        if self.normalize:
            s1 = self._normalize_series(s1)
            s2 = self._normalize_series(s2)
        
        # dtaidistanceではワーピングパスの取得が限定的なので
        # 必要に応じて他のライブラリを使用
        # ここでは簡略化のためNoneを返す
        return None