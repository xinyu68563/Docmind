# DocMind —— 评测指标

"""
检索与生成的量化指标（Phase 3）

【为什么每个指标都要懂它的"盲区"】

指标不是越多越好，是越合适越好。每个指标都有自己的偏见：

    Recall@K  —— 只看"找没找到"，完全不在乎排第几。
                 所以单看 Recall，一个把相关结果排在第 10 名的系统
                 和一个排在第 1 名的系统得分一样。

    MRR       —— 只看第一个相关结果的位置。
                 它对"第一个找到"极度敏感，但完全忽略后面还有没有
                 更有用的结果。适合单答案问答（比如本项目的场景）。

    NDCG@K    —— 兼顾位置和数量，是三者里最全面的。
                 代价是计算稍复杂，且需要定义"相关性等级"。
                 本项目 gold 只有一个，所以退化成二值相关性。

三个一起看，才能判断系统是"找不到"还是"找得到但排得差"。

【本项目为什么只需要 Recall@1 和 MRR】

因为每条评测题只有一个 gold chunk（自监督构造决定的）。
多 gold 场景（比如"列出所有提到 X 的段落"）才需要完整的 NDCG/Precision，
那种场景留到后续扩展。

【生成指标为什么不直接用 RAGAS】

RAGAS、TruLens 这类框架确实方便，但有两个问题：
    1. 它们默认调 OpenAI API，本地离线跑不了
    2. 装了也是黑盒，面试时你讲不清楚指标是怎么算的

这里手写，同时明确标注 LLM-as-judge 的固有局限（见 faithfulness 注释）。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# 检索指标
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list[str], gold_id: str, k: int = 5) -> float:
    """Top-K 召回率。

    本项目 gold 只有 1 个，所以结果只有 0.0 或 1.0。
    多条评测题平均之后才有意义 —— 那个平均值就是"命中率"。
    """
    top = retrieved[:k]
    return 1.0 if gold_id in top else 0.0


def reciprocal_rank(retrieved: list[str], gold_id: str) -> float:
    """倒数排名（单条）。

    gold 排第 1 -> 1.0；第 3 -> 0.333；没找到 -> 0.0
    """
    for i, cid in enumerate(retrieved, start=1):
        if cid == gold_id:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], gold_id: str, k: int = 5) -> float:
    """NDCG@K（二值相关性版本）。

    完整定义：DCG = Σ rel_i / log2(i+1)，IDCG 是理想排序下的 DCG。
    本项目只有一个 gold 且相关性为 1，于是：
        gold 排第 r 位 -> DCG = 1 / log2(r+1)
        IDCG（gold 在第 1 位）= 1 / log2(2) = 1
        所以 NDCG = 1 / log2(r+1)

    举几个直观的数：
        r=1 -> 1.000
        r=2 -> 0.631
        r=5 -> 0.387
    这个衰减比 MRR（1/r）平缓，意味着它对"稍微排后一点"更宽容。
    """
    for r, cid in enumerate(retrieved[:k], start=1):
        if cid == gold_id:
            return 1.0 / math.log2(r + 1)
    return 0.0


def hit_position(retrieved: list[str], gold_id: str) -> int | None:
    """gold 的排名（从 1 开始），没找到返回 None。用于错误分析。"""
    for i, cid in enumerate(retrieved, start=1):
        if cid == gold_id:
            return i
    return None


# ---------------------------------------------------------------------------
# 生成指标
# ---------------------------------------------------------------------------

ABSTAIN_PATTERNS = (
    "无法回答", "不能回答", "没有提及", "未提及", "未提供",
    "无法找到", "不足以回答", "根据当前文献", "文章中未", "文中未",
)


def is_abstention(answer: str) -> bool:
    """判断是否为"拒答"。

    拒答不是坏事 —— 对 RAG 系统来说，**诚实说不知道远比编造有价值**。
    这个指标要单独看：如果某配置的拒答率明显高，
    往往是因为它的检索召回差，导致模型真的找不到依据。
    """
    return any(p in answer for p in ABSTAIN_PATTERNS)


def extract_citations(answer: str) -> list[int]:
    """从回答中提取引用编号 [1] [2][3] —— **不限制位数**。

    【本轮修掉的漏检】

    原正则写作 `\\[(\\d{1,2})\\]`，只匹配 1–2 位数字，于是：
        [99]  -> 两位，能匹配（但编号 99 通常越界）
        [999] -> 三位，**完全匹配不到**，等于这条错误被静默放行
    一个"引用了不存在的编号"的答案，因为编号恰好是三位数就逃过了检查，
    这是典型的"检查存在但形同虚设"。

    现在改成 `\\[(\\d+)\\]`：任何位数的数字编号都会被提出来，
    越界与否交给 `citation_valid_rate()` 判断（合法区间是 1..n_evidence，
    所以 [0] 一定越界）。宁可多提几个（比如正文里出现的参考文献标号 [28]），
    也不能漏掉真正的越界引用 —— 漏检的代价远高于多检。

    【放宽到任意位数的代价，要如实说】

    代价是"年份也被当成引用"：正文里写 `[2020]` 会被判越界。
    这是**故意接受**的假阳性 —— 因为它只会让结论变保守（failed/unknown），
    不会把坏的判成好的；而漏检恰好相反。
    另外实测过 549 条已保存回答，方括号内 4 位以上数字出现 0 次，
    说明在本语料（采样定理数学论文）上这个代价目前没有实际发生。
    """
    nums = re.findall(r"\[(\d+)\]", answer)
    return [int(n) for n in nums]


def citation_valid_rate(answer: str, n_evidence: int) -> tuple[float | None, list[int], list[int]]:
    """引用编号**合法率**（不是事实支撑率）。

    返回 (合法率 | None, 合法引用, 越界引用)

    合法 = 编号落在 [1, n_evidence] 范围内。
    没有引用时返回 None（不适用），**不能**当成 1.0 ——
    "一个引用都没有"和"引用全部合法"是两件完全不同的事，
    旧实现把无引用算成满分，直接把引用合法率抬到 100%，
    这个数字毫无意义且具有误导性。

    【必须和"事实支撑率"分开】
    这个指标只能证明编号**存在且在范围内**，
    不能证明 [3] 指向的内容真的支撑了那句话。
    后者叫 faithfulness / 事实支撑率，需要 LLM-as-judge，
    本项目**尚未实现**，任何报表里都不允许出现它的数值。

    为什么还要这个弱指标？
    因为它零成本、零随机性（纯规则），每次跑结果完全一样，
    适合放在消融结果表里做稳定性参考。
    """
    cites = extract_citations(answer)
    if not cites:
        return None, [], []
    valid = [c for c in cites if 1 <= c <= n_evidence]
    invalid = [c for c in cites if not (1 <= c <= n_evidence)]
    return len(valid) / len(cites), valid, invalid


# 向后兼容别名。新代码请用 citation_valid_rate —— 名字里带 valid 才能
# 提醒读者这只检查编号合法性，不代表内容被证据支撑。
citation_accuracy = citation_valid_rate


def has_citation(answer: str) -> bool:
    """回答是否带任何引用。带引用比例也是个有用的健康度指标。"""
    return len(extract_citations(answer)) > 0


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------

@dataclass
class EvalRow:
    """单条评测结果。

    【口径说明：为什么生成侧字段的默认值是 None 而不是 0/False】

    检索层评测（retrieval）根本不产生答案。旧实现给它填了
    cite_accuracy=1.0 / abstained=False / elapsed=0.0 / tokens=0，
    于是检索层报告里出现了"引用准确率 100%、拒答率 0%、耗时 0 秒" ——
    三个都是凭空编出来的数字。

    现在的规则：**没做的事就是没有数据**，一律 None。
    None 在聚合时被跳过，在表格里显示成 n/a，
    绝不会被当成 0 或满分参与平均。
    """
    qid: str
    question: str
    retrieved: list[str]
    gold_id: str
    answer: str = ""
    n_evidence: int = 0
    # --- 以下只有端到端评测才有值，检索层为 None ---
    rounds: int | None = None
    tokens: int | None = None            # 该次运行的输出 token（预算口径）
    prompt_tokens: int | None = None
    elapsed: float | None = None
    faithfulness: float | None = None    # 事实支撑率：需要 LLM-as-judge，尚未实现
    cite_accuracy: float | None = None   # 引用编号合法率（无引用时为 None）
    abstained: bool | None = None
    cited: bool | None = None
    # --- 端到端可复核信息 ---
    query_used: str | None = None        # 实际送进检索的查询（可能是英文译文）
    evidence_ids: list[str] | None = None
    verification_status: str | None = None   # 综合状态（综合了 LLM 校验与引用合法性）
    llm_verification_status: str | None = None
    citation_validity: str | None = None     # valid / invalid / none
    stop_reason: str | None = None
    answer_status: str | None = None
    citation_issues: list[str] | None = None
    lang: str | None = None
    # evidence: [{id, source, text, score}] —— 端到端保存完整内容与来源，
    # 检索层保存 id/来源/分数（不重抄全文，语料快照另见 run 级 metadata）
    evidence: list[dict] | None = None

    def to_dict(self) -> dict:
        """序列化。

        【必须落盘 `retrieved` 和 `gold_id`】

        第一版只存了算好的 recall/mrr/ndcg，没存原始输入，
        于是拿到 JSON 的人**无法独立复算**指标 —— 只能选择相信我们的汇总值。
        这违背了可复核的基本前提：报告宣称"逐题结果可复算"，
        但文件里根本没有复算所需的输入。现在两个都存，且 retrieved 保持有序。
        """
        return {
            "qid": self.qid,
            "question": self.question,
            "query_used": self.query_used,
            "lang": self.lang,
            # 复算所需的两个原始输入（有序）
            "retrieved": list(self.retrieved),
            "gold_id": self.gold_id,
            "recall@1": recall_at_k(self.retrieved, self.gold_id, 1),
            "recall@3": recall_at_k(self.retrieved, self.gold_id, 3),
            "recall@5": recall_at_k(self.retrieved, self.gold_id, 5),
            "mrr": reciprocal_rank(self.retrieved, self.gold_id),
            "ndcg@5": ndcg_at_k(self.retrieved, self.gold_id, 5),
            "hit_pos": hit_position(self.retrieved, self.gold_id),
            "n_evidence": self.n_evidence,
            "evidence_ids": self.evidence_ids,
            "evidence": self.evidence,
            # 生成侧：检索层整列为 None
            "cite_valid_rate": self.cite_accuracy,
            "abstained": self.abstained,
            "cited": self.cited,
            "rounds": self.rounds,
            "tokens": self.tokens,
            "prompt_tokens": self.prompt_tokens,
            "elapsed": round(self.elapsed, 2) if self.elapsed is not None else None,
            "faithfulness": self.faithfulness,
            "verification_status": self.verification_status,
            "llm_verification_status": self.llm_verification_status,
            "citation_validity": self.citation_validity,
            "stop_reason": self.stop_reason,
            "answer_status": self.answer_status,
            "citation_issues": self.citation_issues,
            "answer": self.answer,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalRow":
        """反序列化（roundtrip 用）。

        为什么要有这个：
        "能存进去"不等于"能读回来"。序列化测试必须验证
        存下来的 dict 能还原成 EvalRow 且指标不变，
        否则将来改字段时静默丢数据的风险很高。
        """
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def recomputed(self) -> dict[str, float]:
        """只依赖 `retrieved` + `gold_id` 重算指标 —— 用于独立校验汇总值。"""
        return {
            "recall@1": recall_at_k(self.retrieved, self.gold_id, 1),
            "recall@3": recall_at_k(self.retrieved, self.gold_id, 3),
            "recall@5": recall_at_k(self.retrieved, self.gold_id, 5),
            "mrr": reciprocal_rank(self.retrieved, self.gold_id),
            "ndcg@5": ndcg_at_k(self.retrieved, self.gold_id, 5),
            "hit_pos": hit_position(self.retrieved, self.gold_id),
        }


@dataclass
class EvalSummary:
    """一组评测的汇总"""
    name: str = ""
    rows: list[dict] = field(default_factory=list)

    @property
    def recall_at_1(self) -> float:
        return self._mean("recall@1")

    @property
    def recall_at_3(self) -> float:
        return self._mean("recall@3")

    @property
    def recall_at_5(self) -> float:
        return self._mean("recall@5")

    @property
    def mrr(self) -> float:
        return self._mean("mrr")

    @property
    def ndcg_at_5(self) -> float:
        return self._mean("ndcg@5")

    @property
    def citation_valid_rate(self) -> float | None:
        """引用编号合法率。

        只在**有引用**的回答上求均值 —— 没有引用的回答不参与，
        否则"全都不引用"能拿满分，指标就废了。
        全部样本都无引用时返回 None（不适用）。
        """
        return self._mean("cite_valid_rate")

    @property
    def abstention_rate(self) -> float | None:
        return self._mean("abstained")

    @property
    def citation_rate(self) -> float | None:
        """非拒答的回答中，带了引用标注的比例。"""
        non_abstain = [r for r in self.rows if r["abstained"] is False]
        if not non_abstain:
            return None
        return sum(1 for r in non_abstain if r["cited"]) / len(non_abstain)

    @property
    def avg_rounds(self) -> float | None:
        return self._mean("rounds")

    @property
    def avg_tokens(self) -> float | None:
        return self._mean("tokens")

    @property
    def avg_latency(self) -> float | None:
        return self._mean("elapsed")

    @property
    def n(self) -> int:
        return len(self.rows)

    def _mean(self, key: str) -> float | None:
        """对某一列求均值。布尔列自动转成 0/1 比例。

        关键改动：**没有任何有效值时返回 None，而不是 0.0。**
        旧实现返回 0.0，于是"检索层没有生成任何答案"被显示成
        "耗时 0.0 秒 / 拒答率 0.000"，看起来像跑得又快又好。
        """
        vals = [r.get(key) for r in self.rows]
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        if isinstance(vals[0], bool):
            return sum(1 for v in vals if v) / len(vals)
        return sum(vals) / len(vals)

    def misses(self) -> list[dict]:
        """检索完全没命中的题目 —— 错误分析用

        比数字更有价值的是"为什么错"。看这些 case，
        往往能发现"专有名词被分词切开"这类具体问题。
        """
        return [r for r in self.rows if r["hit_pos"] is None]

    def exact_means(self) -> dict:
        """未取整的汇总值 —— 供外部独立复算时逐位比对。

        为什么要另存一份：`as_row()` 为了表格好看统一 round 到 3 位，
        于是"从 JSON 复算"只能比对到 5e-4，严格来说仍不是逐位一致。
        有了这一份，复算可以断言到 1e-12。
        """
        keys = ["recall@1", "recall@3", "recall@5", "mrr", "ndcg@5",
                "cite_valid_rate", "abstained", "cited", "rounds", "tokens"]
        out = {k: self._mean(k) for k in keys}
        # 不适用的列保留 None（而不是 0），和 as_row 的 n/a 口径一致
        return {k: v for k, v in out.items()}

    def as_row(self) -> dict:
        """给表格用的一行。不适用的列写 "n/a"，不写成 0。"""
        def f(x, nd: int = 3):
            return "n/a" if x is None else round(x, nd)

        row = {
            "配置": self.name,
            "n": self.n,
            "R@1": f(self.recall_at_1),
            "R@3": f(self.recall_at_3),
            "R@5": f(self.recall_at_5),
            "MRR": f(self.mrr),
            "NDCG@5": f(self.ndcg_at_5),
        }
        # 生成侧指标：检索层没有答案时为 None -> n/a
        if any(r.get("answer") for r in self.rows):
            row["引用编号合法率"] = f(self.citation_valid_rate)
            row["带引用比例"] = f(self.citation_rate)
            row["拒答率"] = f(self.abstention_rate)
            row["平均轮数"] = f(self.avg_rounds, 2)
            row["平均输出token"] = f(self.avg_tokens, 1)
            row["平均耗时s"] = f(self.avg_latency, 1)
        # faithfulness（事实支撑率）尚未实现，这里刻意不输出任何数值
        return row
