# DocMind —— 检索：多路融合

"""
RRF 融合（Phase 1）

【核心问题：两条路的分数怎么合到一起？】

BM25 返回形如：  [("docA", 12.7), ("docB", 4.1), ("docC", 0.3)]
向量返回形如：   [("docX", 0.87), ("docA", 0.62), ("docY", 0.21)]

最直觉的做法是把两个分数加权相加：final = 0.5*bm25 + 0.5*cosine
**这是错的，而且错得很隐蔽**：

    BM25 是无上界的稀疏分数，取值可能是 0.3 也可能是 37.2，
    取决于文档长度和词频；余弦相似度被限制在 [0, 1] 区间。
    两者的**量纲完全不同**，加权平均等于让你用一把量程混乱的尺子量东西。
    更糟的是，你还得给每个新模型重新调一次权重，换批数据就失效。

【RRF（Reciprocal Rank Fusion）的解法】

抛弃原始分数，**只使用排名信息**：

    score(d) = Σ_over_all_lists  1 / (k + rank_of_d_in_list)

其中 k 是一个常数（通常取 60），用来压低排名靠前项的绝对优势。

假设某文档在 BM25 排第 1、在向量排第 3：
    score = 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226

这么做的好处：
    1. 完全不需要关心两路分数的分布 —— 天然免除了归一化问题
    2. 换任何一路新召回源，直接加进来就行，不用调参
    3. 对 recall@k 类指标提升稳定（这是有论文支撑的，Cormack et al. 2009）

这个 60 是哪来的？原始论文的推荐值。它控制"排名靠前到底有多值钱"：
    k 越小 -> 第一名优势越大（更激进）
    k 越大 -> 各名次差异被抹平（更保守）
后面做消融实验时，k ∈ {10, 20, 60, 100} 是必跑的一组。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FusedResult:
    """融合后的单条结果。保留了溯源信息 —— 这是本项目"可解释"的关键。"""

    chunk_id: str
    score: float
    source: str = ""                    # "bm25" / "vector" / "both"
    rank: int = 0
    bm25_rank: int | None = None        # 在 BM25 那一路的排名，None 表示没进 top-k
    vector_rank: int | None = None
    vector_score: float | None = None   # 余弦相似度，用于展示
    chunk: object = None                # 对应的 Chunk 对象，便于上层直接取文本

    @property
    def hit_by_both(self) -> bool:
        return self.bm25_rank is not None and self.vector_rank is not None


def reciprocal_rank_fusion(
    rankings: dict[str, list[str]],
    k: int = 60,
) -> dict[str, float]:
    """对多路排名字典做 RRF 融合。

    参数
    ----
    rankings : {"bm25": [chunk_id, ...], "vector": [chunk_id, ...]}
               每一路按本路排名从高到低排列
    k        : RRF 常数，默认 60

    返回
    ----
    {chunk_id: fused_score}，已按分数降序
    """
    fused: dict[str, float] = {}
    for source, ranked_ids in rankings.items():
        for rank, cid in enumerate(ranked_ids, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)

    return dict(sorted(fused.items(), key=lambda x: x[1], reverse=True))


class HybridRetriever:
    """BM25 + 向量的双路召回与 RRF 融合。

    用法:
        hybrid = HybridRetriever(bm25_retriever, vector_retriever)
        results = hybrid.search("用户问题", top_k=5)
    """

    def __init__(self, bm25_retriever, vector_retriever, embedder, rrf_k: int = 60):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.embedder = embedder
        self.rrf_k = rrf_k

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_k: int = 50,
        mode: str = "hybrid",
    ) -> list[FusedResult]:
        """双路召回 + 融合。

        参数
        ----
        top_k       : 最终返回几条
        candidate_k : 每一路先各取多少候选再做融合。
                      这个值必须 > top_k，否则那些"两路都排第 30 名"的
                      优质结果根本没机会参与融合 —— 这是很多人实现 RRF 时踩的坑。
        mode        : "bm25" / "vector" / "hybrid"（做消融实验用）
        """
        rankings: dict[str, list[str]] = {}
        rank_map: dict[str, dict[str, int]] = {}
        vec_scores: dict[str, float] = {}
        chunk_obj: dict[str, object] = {}

        # --- 第 1 路：BM25 ---
        if mode in ("bm25", "hybrid"):
            hits = self.bm25.search(query, top_k=candidate_k)
            rankings["bm25"] = [h[1].chunk_id for h in hits]
            rank_map["bm25"] = {h[1].chunk_id: i for i, h in enumerate(hits, 1)}
            for _, block in hits:
                chunk_obj[block.chunk_id] = block

        # --- 第 2 路：向量 ---
        if mode in ("vector", "hybrid"):
            qvec = self.embedder.encode_query(query)
            vhits = self.vector.search(qvec, top_k=candidate_k)
            rankings["vector"] = [h.chunk_id for h in vhits]
            rank_map["vector"] = {h.chunk_id: i for i, h in enumerate(vhits, 1)}
            for h in vhits:
                vec_scores[h.chunk_id] = h.score
            # 补全可能只在向量路出现的 chunk 对象
            for h in vhits:
                if h.chunk_id not in chunk_obj and hasattr(self.bm25, "blocks"):
                    pos = self.bm25.block_index.get(h.chunk_id)
                    if pos is not None:
                        chunk_obj[h.chunk_id] = self.bm25.blocks[pos]

        if not rankings:
            return []

        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k)

        results: list[FusedResult] = []
        for rank, (cid, score) in enumerate(list(fused.items())[:top_k], start=1):
            bm25_rank = rank_map.get("bm25", {}).get(cid)
            vec_rank = rank_map.get("vector", {}).get(cid)
            if bm25_rank is not None and vec_rank is not None:
                source = "both"
            elif bm25_rank is not None:
                source = "bm25"
            else:
                source = "vector"
            results.append(
                FusedResult(
                    chunk_id=cid,
                    score=score,
                    source=source,
                    rank=rank,
                    bm25_rank=bm25_rank,
                    vector_rank=vec_rank,
                    vector_score=vec_scores.get(cid),
                    chunk=chunk_obj.get(cid),
                )
            )
        return results
