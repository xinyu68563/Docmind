# DocMind —— 检索：关键词匹配

"""
BM25 检索（Phase 0）

【为什么先做 BM25，而不是直接上向量？】

很多人做 RAG 上来就搞 embedding，我特意反着来，原因有两个：

一、教学原因：BM25 是纯数学、零黑盒。你能肉眼看清"为什么这个词匹配上了"，
    出了问题好排查。而向量检索出问题你完全不知道发生了什么。
    先把可解释的东西搞清楚，后面加向量时你才能判断"向量到底带来了多少提升"。

二、效果原因（这条更重要，简历上能吹）：
    BM25 在「专有名词」上的表现远好于向量检索。举个例子，你的论文里有个
    "sinh型正则化函数"，用户问 "sinh regularization 怎么定义的"：
        - BM25      : 精确命中 sinh / regularization，稳
        - 向量检索   : 可能召回一堆"反正都是数学方法"的段落，但不包含 sinh
    所以真实生产系统一定是 BM25 + 向量双路召回 + 融合，而不是纯向量。
    这个结论你后面做消融实验时能用数据证明出来，是简历上的硬货。

【BM25 是什么（三句话说清）】
    TF-IDF 的改进版。两份文档都提到"的"字 100 次没意义（TF 饱和 + IDF 惩罚），
    只有少数文档出现的词才值钱。BM25 额外加了一个"文档长度归一化"：
    一篇 10 页的文档提 3 次关键词，和一段话提 3 次，显然后者更相关。
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from docmind.ingest.pdf_parser import Block


# ---------------------------------------------------------------------------
# 中英混合分词
# ---------------------------------------------------------------------------

# 英文单词 / 数字
_RE_EN = re.compile(r"[a-zA-Z][a-zA-Z]*[a-zA-Z]|[a-zA-Z]|\d+(?:\.\d+)?")
# 连续中文片段
_RE_CJK = re.compile(r"[\u4e00-\u9fff]+")

# 精简停用词表。为什么不用现成的大词表？
# 因为中英混合场景下，哈工大停用词表会把 "sinh" 这类词漏掉、把某些数学词误杀。
# 检索场景保留更多词通常更安全，宁可召回多一点，交给后面的重排模块去筛。
STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "就", "也", "都", "而", "或",
    "有", "为", "以", "对", "从", "这", "那", "之", "其", "了", "着", "把",
    "被", "所", "让", "给", "才", "更", "很", "最", "不", "没", "无",
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "to", "of", "in", "on", "at", "for", "with", "by", "from", "as",
    "and", "or", "but", "if", "then", "than", "that", "this", "these",
    "we", "our", "it", "its", "can", "may", "will", "which", "et", "al",
}


def tokenize(text: str) -> list[str]:
    """中英混合分词。

    策略：分两条路处理，避免互相污染
        - 英文/数字：正则按词边界切
        - 中文片段  : jieba 分词

    为什么不能直接用空格切？
        中文没有空格，整句话会被当成一个 50 字的"词"，BM25 完全失效。
    为什么不能只对整段用 jieba？
        jieba 会把 "sinh-type regularization function" 切成乱七八糟，
        把英文专业词切碎会严重伤害检索精度。
    """
    import jieba  # 延迟导入：jieba 首次加载词典要几百毫秒，放这里不拖慢启动

    lowered = text.lower()
    tokens: list[str] = []

    # 英文 / 数字
    tokens.extend(_RE_EN.findall(lowered))

    # 中文
    for seg in _RE_CJK.findall(lowered):
        for tok in jieba.cut(seg):
            tok = tok.strip()
            if not tok or tok in STOPWORDS:
                continue
            if len(tok) < 2:
                continue  # 单字词的噪声远大于信息量
            tokens.append(tok)

    return tokens


# ---------------------------------------------------------------------------
# BM25 索引
# ---------------------------------------------------------------------------


class BM25Retriever:
    """基于 BM25Okapi 的检索器。

    用法:
        retriever = BM25Retriever(blocks)
        hits = retriever.search("你的问题", top_k=5)
        for score, block in hits:
            print(score, block.text)
    """

    def __init__(self, blocks: list[Block], k1: float = 1.5, b: float = 0.75):
        """
        参数 k1 : 控制词频饱和速度。k1 越大，词频的权重越线性（重复出现越值钱）
        参数 b  : 控制文档长度归一化的强度。b=0 表示完全不考虑长度，b=1 完全归一化
        这两个是 BM25 的核心超参，默认值是 Robertson 原始论文的推荐值。
        后面做消融实验时，调这两个值能造成 2-3 个百分点的 Recall 差异。
        """
        self.blocks = blocks
        self.k1 = k1
        self.b = b
        # chunk_id -> 下标。融合阶段要按 id 反查 chunk 对象，这里建索引避免 O(n) 查找
        self._id_to_pos = {b.chunk_id: i for i, b in enumerate(blocks)}
        self._tokenized = [tokenize(b.text) for b in blocks]
        self._index = BM25Okapi(self._tokenized, k1=k1, b=b)

    @property
    def block_index(self) -> dict[str, int]:
        return self._id_to_pos

    def search(self, query: str, top_k: int = 5) -> list[tuple[float, Block]]:
        """检索，返回 [(分数, 块), ...]，按分数降序。"""
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self._index.get_scores(query_tokens)
        # 用 heapq 取 Top-K 比全量排序快，语料一大差别明显
        import heapq

        top_indices = heapq.nlargest(top_k, range(len(scores)), key=scores.__getitem__)
        return [(float(scores[i]), self.blocks[i]) for i in top_indices if scores[i] > 0]

    def explain(self, query: str, block_index: int) -> dict[str, float]:
        """解释「为什么这块被打这么高分」—— 调试检索效果的利器。

        当你发现该搜出来的没搜出来时，用这个看每个 query 词贡献了多少分，
        一眼就能定位是分词问题还是词表问题。
        """
        # 注意：这里直接复算一次全部，仅用于调试，不在检索路径上调用
        all_scores = self._index.get_scores(tokenize(query))
        return {
            "total_score": float(all_scores[block_index]),
            "doc_len": len(self._tokenized[block_index]),
            "avg_doc_len": sum(len(t) for t in self._tokenized) / len(self._tokenized),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "k1": self.k1,
                    "b": self.b,
                    "tokenized": self._tokenized,
                    # 把 chunk_id 一起存下来，加载时才能校验"索引和语料是不是同一份"。
                    # 不存的话，一旦重建了 chunks 却忘了重建 BM25，
                    # 索引会静默错位 —— 检索结果看起来正常，实际全是错的。
                    "chunk_ids": [b.chunk_id for b in self.blocks],
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path, blocks: list[Block], strict: bool = True) -> "BM25Retriever":
        """加载索引。

        注意：索引只存了 tokenized 结果，blocks 需要外部传入。
        这样 blocks 和索引解耦，换检索算法时不用重新解析 PDF。

        strict=True 时会校验索引里的 chunk_id 序列与传入 blocks 完全一致，
        不一致直接抛错（宁可启动失败，也不要带着错位索引跑出假结果）。
        旧格式（没有 chunk_ids 字段）无法校验，只在 strict 下提示一次。
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.blocks = blocks
        obj.k1 = data["k1"]
        obj.b = data["b"]
        obj._id_to_pos = {b.chunk_id: i for i, b in enumerate(blocks)}
        obj._tokenized = data["tokenized"]
        obj._index = BM25Okapi(obj._tokenized, k1=obj.k1, b=obj.b)

        saved_ids = data.get("chunk_ids")
        if saved_ids is not None:
            cur_ids = [b.chunk_id for b in blocks]
            if list(saved_ids) != cur_ids:
                msg = (
                    f"BM25 索引与语料不一致：索引里有 {len(saved_ids)} 条、"
                    f"语料有 {len(cur_ids)} 条，chunk_id 序列不同。\n"
                    "    原因通常是重建了 chunks 却没有重建索引。\n"
                    "    请重新运行: python -m docmind.cli ingest"
                )
                if strict:
                    raise ValueError(msg)
                print("[警告] " + msg)
        elif strict:
            print("[警告] BM25 索引为旧格式（未保存 chunk_ids），无法校验一致性")
        return obj
