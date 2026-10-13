# DocMind —— 检索：向量近邻搜索

"""
向量检索（Phase 1）

【为什么这里不用 Faiss / Milvus？】

生产环境当然应该用专业的向量数据库。但这个项目里我用 numpy 裸写，两个原因：

一、**教学**：暴力矩阵乘法是最近邻搜索最本质的形态。
     `scores = doc_vectors @ query_vector` 一行就是全部核心。
     理解了这行，你再看 Faiss 的 HNSW / IVF-PQ 就会明白它们只是在
     "牺牲一点精度换取速度"，而不是魔法。

二、**规模**：我们的语料是几百到几千个 chunk，1024 维。
     暴力搜索一次矩阵乘法 = 几千×1024 = 几百万次浮点运算，GPU 上是微秒级。
     引入 Faiss 反而增加依赖复杂度和面试时的解释成本。

什么时候该换？
    - chunk 数 > 10 万         -> 上 Faiss HNSW
    - 需要分布式 / 持久化 / 多租户 -> 上 Milvus 或 pgvector
这点在 README 里写清楚，反而显得你有工程判断力。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RetrievalResult:
    """统一的检索返回结构。

    为什么需要它？因为融合阶段我们要追踪：
        - 这个结果是哪一路召回的（便于分析）
        - 它在各自那一路里排第几（RRF 要用）
    光有一个分数是不够的。
    """

    chunk_id: str
    score: float
    source: str = ""       # "bm25" / "vector" / "both"
    vector: object = None  # 对应 Chunk 对象


class VectorRetriever:
    """基于 numpy 的余弦相似度检索器。

    因为存入的向量已经 L2 归一化过，余弦相似度退化为向量点积，
    一次矩阵乘法就能算完所有候选的分数。
    """

    def __init__(self, vectors: np.ndarray, chunk_ids: list[str]):
        """
        参数
        ----
        vectors   : 形状 (N, dim) 的文档向量矩阵，需已归一化
        chunk_ids : 与 vectors 按行对应的 chunk 标识
        """
        self.vectors = vectors.astype(np.float32)
        self.chunk_ids = chunk_ids
        self._id_to_pos = {cid: i for i, cid in enumerate(chunk_ids)}

    def search(self, query_vec: np.ndarray, top_k: int = 10) -> list[RetrievalResult]:
        """检索，返回按相似度降序的前 top_k 条。"""
        if self.vectors.size == 0:
            return []

        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        # 核心就这一行：所有文档向量与查询向量做点积
        scores = self.vectors @ q

        # argpartition 比 argsort 快（O(n) vs O(n log n)），
        # 只需要 top_k 时用它，语料一大差距明显
        k = min(top_k, len(scores))
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [
            RetrievalResult(
                chunk_id=self.chunk_ids[i],
                score=float(scores[i]),
                source="vector",
            )
            for i in top_idx
            if scores[i] > 0
        ]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=self.vectors, chunk_ids=np.array(self.chunk_ids))

    @classmethod
    def load(cls, path: str | Path) -> "VectorRetriever":
        data = np.load(path, allow_pickle=True)
        return cls(vectors=data["vectors"], chunk_ids=list(data["chunk_ids"]))
