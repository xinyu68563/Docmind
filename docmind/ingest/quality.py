# DocMind —— 文档摄取：解析质量评估

"""
解析质量量化评估（W2）

【为什么要单独做这个模块？一个真实的教训】

我最初评估解析质量时，用的指标是"是否存在长度 >= 14 的连续字母片段"，
得出"87% 的 chunk 存在粘连"的结论，看起来问题很严重。

但这个结论是**错的**。查了原始 span 数据才发现：
那些"粘连"绝大多数是 `f(τmn)ψmn(x)`、`sinc(Lt − n)` 这类数学表达式，
它们天生不含空格 —— 因为 LaTeX 数学模式下空格是被忽略的，
字符间距靠定位实现。它们不是缺陷，是正确内容。

教训：**在评估之前，必须先定义清楚"什么算缺陷"**。
指标定义错了，后面所有优化工作都会朝着错误方向使劲。

【解决方案：用字体区分正文与公式】

LaTeX 生成的 PDF 里，数学区域使用专门的字体族：
    CMR10 / CMMI10 / CMSY10 / CMEX10   Computer Modern（LaTeX 默认）
    MSAM / MSBM                        AMS 符号字体
    MTSY / MTMI                        MathTime
识别出这些字体，就能把"公式"从"正文"里剥离出来，
只对正文统计粘连率 —— 这才是真实的解析质量。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# LaTeX / TeX 数学字体标识。字体名通常形如 "CMMI10"、"MSBM10"
_MATH_FONT_PREFIXES = (
    "CMR", "CMMI", "CMSY", "CMEX", "CMBX", "CMTI", "CMSL", "CMU", "CMC",
    "CMSS", "CMTT", "CMFIB",
    "MSAM", "MSBM",           # AMS 符号
    "MTSY", "MTMI", "MTXB",   # MathTime
    "EURM", "EUSM", "EUBM",   # Euclid
    "TXMI", "TXSY", "TXEX",   # TX Fonts
    "LMMI", "LMSY", "LMEX",   # Latin Modern Math
    "PXMI", "PXSY",
)

# 中文字符
_CJK = re.compile(r"[\u4e00-\u9fff]")
_WORD = re.compile(r"[A-Za-z]+")


def is_math_font(font_name: str) -> bool:
    """判断字体是否为数学字体族。"""
    if not font_name:
        return False
    upper = font_name.upper().lstrip("/")
    return any(upper.startswith(p) for p in _MATH_FONT_PREFIXES)


@dataclass
class QualityReport:
    """一份解析结果的质量报告。所有指标都可复现、可对比。"""

    parser_name: str
    n_blocks: int
    n_math_blocks: int
    math_ratio: float            # 数学块占比
    n_text_blocks: int           # 正文块数（参与质量统计）
    glued_blocks: int            # 正文中出现疑似粘连的块数
    glue_rate: float             # 正文粘连率 —— 核心指标
    avg_token_len: float         # 正文平均 token 长度
    max_token_len: int
    cjk_blocks: int              # 含中文的块数

    def as_row(self) -> list[str]:
        return [
            self.parser_name,
            str(self.n_blocks),
            f"{self.math_ratio:.1%}",
            str(self.n_text_blocks),
            f"{self.glue_rate:.1%}",
            f"{self.avg_token_len:.1f}",
            str(self.max_token_len),
        ]


# 连续字母片段的"异常长度"阈值。
# 英文里最长的常用词约 15 字符（如 "internationalization"），
# 超过 18 基本可以断定是多个词被粘连或本身就是数学表达式。
GLUE_THRESHOLD = 18


def evaluate(blocks, parser_name: str) -> QualityReport:
    """评估一批 Block 的解析质量。

    参数 blocks 需要携带 font_name 字段才能做数学区域识别；
    没有该字段时（例如第三方解析器输出），退化为全量统计。
    """
    total = len(blocks)
    math_blocks = 0
    text_blocks: list[str] = []
    glued = 0
    max_len = 0
    token_count = 0
    token_chars = 0
    cjk = 0

    for b in blocks:
        text = b.text or ""
        if not text.strip():
            continue
        font = getattr(b, "font_name", "") or ""
        if _CJK.search(text):
            cjk += 1

        # 数学区域单独计数，不参与正文质量统计
        if font and is_math_font(font):
            math_blocks += 1
            continue
        # 无字体信息时，用符号密度辅助判断是否为公式行
        if not font and _looks_like_formula(text):
            math_blocks += 1
            continue

        text_blocks.append(text)
        words = _WORD.findall(text)
        for w in words:
            token_count += 1
            token_chars += len(w)
            max_len = max(max_len, len(w))
        if any(len(w) >= GLUE_THRESHOLD for w in words):
            glued += 1

    n_text = len(text_blocks)
    return QualityReport(
        parser_name=parser_name,
        n_blocks=total,
        n_math_blocks=math_blocks,
        math_ratio=math_blocks / total if total else 0.0,
        n_text_blocks=n_text,
        glued_blocks=glued,
        glue_rate=glued / n_text if n_text else 0.0,
        avg_token_len=token_chars / token_count if token_count else 0.0,
        max_token_len=max_len,
        cjk_blocks=cjk,
    )


# 常见数学符号。密度高于阈值时认为是公式行而非正文
_MATH_SYMBOLS = set("∫∑∏√∞±≤≥≠≈∂∇∀∃∈∉⊂⊆∪∩→←⇒⇐αβγδεζηθικλμνξπρστυφχψω"
                    "ΓΔΘΛΞΠΣΦΨΩℝℂℕℤ∅‖⟨⟩⋅×÷∂"
                    "^{}()[]|\\/")


def _looks_like_formula(text: str, threshold: float = 0.12) -> bool:
    """无字体信息时，用数学符号密度推断是否为公式行。"""
    if not text:
        return False
    n_sym = sum(1 for ch in text if ch in _MATH_SYMBOLS)
    return n_sym / len(text) > threshold
