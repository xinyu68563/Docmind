# DocMind —— 文档摄取：把 PDF 变成结构化数据

"""
PDF 解析器（Phase 0）

【为什么要单独写这个模块，而不是直接 extract_text()？】

绝大多数人写 RAG 的第一行代码是：
    text = page.extract_text()          # 一行搞定，但是...
这样拿到的是一坨没有结构信息的字符串。你会遇到三个致命问题：

    1. 双栏论文会把左右两栏的文字交错混在一起，语义全乱
    2. 表格被拆成一堆散落的单元格，读不出行列关系
    3. 标题、正文、图注混在一起，切分时无法感知章节边界

而 PyMuPDF 的 get_text("dict") 会给每个文本行返回坐标(bbox)和字体信息，
有了这些「版面信息」，我们才能在后面做：
    - 按坐标排序还原双栏（同一 y 区间的左边<右边）
    - 判断是否标题（字号明显大于正文，或加粗）
    - 归类图注（紧跟图片块下方的 "Figure 1." / "图 1"）

这些是后面 W2 做「多解析器对比」的基础，也是这个项目区别于玩具 demo 的第一道门槛。
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pymupdf as fitz  # 新版本推荐用 pymupdf 名字；旧教程里的 fitz 已弃用


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class Block:
    """文档中的一个「文本块」。

    一个 Block 是我们系统里最小的检索单元候选。注意它携带了坐标和字体信息 ——
    这是纯文本解析器给不了的东西。
    """

    block_id: str          # 全局唯一 ID，格式: {doc_id}-p{页号}-{块序号}
    doc_id: str            # 所属文档
    file_name: str         # 源文件名，方便追溯
    page: int              # 页码，从 1 开始（和你在 PDF 阅读器看到的一致）
    block_no: int          # 该页内的序号，从 0 开始
    bbox: list[float]      # 坐标 [x0, y0, x1, y1]，原点在页面左上角
    font_size: float       # 该行最大字号
    font_name: str         # 主字体名。用于识别数学区域
    is_bold: bool          # 是否加粗
    is_title: bool         # 是否被判定为标题/小标题
    is_math: bool          # 是否为数学区域（由字体族判定）
    text: str              # 清理后的文本

    @property
    def word_count(self) -> int:
        """粗略字数统计：中文按字计，英文按词计。"""
        cjk = len(re.findall(r"[\u4e00-\u9fff]", self.text))
        en = len(re.findall(r"[A-Za-z]+", self.text))
        return cjk + en

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Block":
        return cls(**d)


@dataclass
class ParsedDocument:
    """一份 PDF 解析后的完整结果。"""

    doc_id: str
    file_name: str
    n_pages: int
    blocks: list[Block]

    def to_json(self, out_path: str | Path, ensure_ascii: bool = False) -> None:
        """落盘为 JSON。

        为什么中间产物要落盘？
        因为「解析」和「检索」是两件事。落盘后你可以：
            - 用肉眼/脚本检查解析质量，不用每次都重跑耗时解析
            - 换检索算法时直接复用同一份解析结果，保证对照实验公平
        """
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "doc_id": self.doc_id,
            "file_name": self.file_name,
            "n_pages": self.n_pages,
            "blocks": [b.to_dict() for b in self.blocks],
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=ensure_ascii, indent=2), encoding="utf-8"
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ParsedDocument":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            doc_id=payload["doc_id"],
            file_name=payload["file_name"],
            n_pages=payload["n_pages"],
            blocks=[Block.from_dict(b) for b in payload["blocks"]],
        )

    def stats(self) -> dict[str, Any]:
        """返回一份统计摘要 —— 这是你判断解析质量的第一个抓手。"""
        if not self.blocks:
            return {"pages": self.n_pages, "blocks": 0}
        sizes = [b.word_count for b in self.blocks]
        titles = [b for b in self.blocks if b.is_title]
        short_blocks = [s for s in sizes if s < 5]  # 过短的块通常是噪声（页码、页眉）
        return {
            "pages": self.n_pages,
            "blocks": len(self.blocks),
            "words_total": sum(sizes),
            "words_avg": round(sum(sizes) / len(sizes), 1),
            "blocks_title": len(titles),
            "blocks_short(<5字)": len(short_blocks),
            "noise_ratio": round(len(short_blocks) / len(self.blocks), 3),
        }


# ---------------------------------------------------------------------------
# 文本清洗
# ---------------------------------------------------------------------------

from docmind.ingest.quality import is_math_font

# 匹配页眉页脚里常见的纯页码形态
_RE_PAGE_NUM = re.compile(r"^[\-–—\s]*\d{1,4}[\-–—\s]*$")
_RE_HYPHEN_BREAK = re.compile(r"(\w)-\s+(\w)")
_RE_MULTI_SPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """清理从 PDF 抽取出来的原始文本。

    PDF 里的文本是「按视觉位置摆放」的，不是按阅读顺序流式存储的，
    所以抽出来往往带一堆问题，这里逐个处理：

    1. 软连字符 \\xad        : Word 排版产物，肉眼不可见，但会破坏词
    2. 换行连字符 "inter-\\n national" : 英文排版断词，应该拼回去
    3. 多余空白/换行         : 压缩成单个空格，便于后续切分
    """
    text = text.replace("\xad", "")
    text = _RE_HYPHEN_BREAK.sub(r"\1\2", text)
    text = _RE_MULTI_SPACE.sub(" ", text)
    return text.strip()


def _is_bold_font(font_name: str) -> bool:
    """从字体名判断是否加粗。

    PDF 不会告诉你"这段加粗了"，只能靠字体名里的关键字猜。
    这是行业通行做法，覆盖率大概 80%，剩下的case靠字号兜底。
    """
    lowered = font_name.lower()
    return any(k in lowered for k in ("bold", "black", "heavy", "semibold", "demibold"))


def _is_noise(text: str) -> bool:
    """判断一个块是否是噪声（页码、页眉页脚）。"""
    if not text:
        return True
    if _RE_PAGE_NUM.match(text):
        return True
    return False


# ---------------------------------------------------------------------------
# 主解析流程
# ---------------------------------------------------------------------------


def mark_math_blocks(blocks: list[Block]) -> None:
    """回填 is_math 标记。

    为什么需要单独一个函数？
        因为从 JSON 反序列化回来的 Block 不带 is_math 字段（旧数据），
        需要一个入口统一补齐，避免调用方各写各的判定逻辑。
    """
    for b in blocks:
        b.is_math = is_math_font(getattr(b, "font_name", "") or "")


def _mark_titles(blocks: list[Block]) -> None:
    """在同一页内标记标题。

    判定规则（简单但有效，面试时能讲清楚为什么这么设计）：
        条件一：字号 > 该页字号中位数 x 1.15
        条件二：字数 < 30（标题不会太长，太长说明是某个引文的字号问题）

    注意：这里用「页内中位数」而不是「全篇中位数」，是因为不同 PDF
    的正文基准字号不同，用页内相对值更鲁棒。
    """
    if len(blocks) < 3:
        return
    sizes = sorted(b.font_size for b in blocks)
    median = sizes[len(sizes) // 2]
    for b in blocks:
        if b.font_size > median * 1.15 and b.word_count < 30:
            b.is_title = True


def parse_pdf(pdf_path: str | Path, keep_noise: bool = False) -> ParsedDocument:
    """解析单个 PDF 文件。

    参数
    ----
    pdf_path    : PDF 文件路径
    keep_noise  : 是否保留噪声块（调试解析效果时设为 True 很有用）

    返回
    ----
    ParsedDocument
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"找不到文件: {pdf_path}")

    doc_id = hashlib.md5(pdf_path.read_bytes()).hexdigest()[:12]
    doc = fitz.open(pdf_path)

    blocks: list[Block] = []

    for page_index in range(doc.page_count):
        page = doc[page_index]
        page_no = page_index + 1

        # "dict" 模式会返回带坐标、字体、分块信息的层级结构:
        #   blocks -> lines -> spans
        raw = page.get_text("dict")
        page_blocks: list[Block] = []

        for raw_block in raw.get("blocks", []):
            # type: 0=文本, 1=图片, 2=矢量图形
            # 这里只取文本块；处理图片/表格是 W2 的事
            if raw_block.get("type") != 0:
                continue

            for raw_line in raw_block.get("lines", []):
                # 注意：过滤条件是 `s.get("text")` 而不是 `s.get("text", "").strip()`
                #
                # 这是一个真实踩过的坑：LaTeX 生成的 PDF 会把「空格」编码成
                # 一个独立的 span（text == " "）。如果用 .strip() 判空，
                # 这个 span 会被当成空内容丢掉，导致所有英文单词粘连成
                # "SinhregularizedLagrangiannonuniform" 这种无法检索的垃圾文本。
                # 实测该项目语料上有 87% 的 chunk 受影响。
                spans = [s for s in raw_line.get("spans", []) if s.get("text")]
                if not spans:
                    continue

                text = clean_text("".join(s["text"] for s in spans))
                if not text:
                    continue
                if not keep_noise and _is_noise(text):
                    continue

                x0, y0, x1, y1 = raw_line["bbox"]
                font_size = max(s["size"] for s in spans)
                is_bold = any(_is_bold_font(s["font"]) for s in spans)
                # 取覆盖字符数最多的字体作为该行主字体（比取第一个更稳）
                font_name = max(spans, key=lambda s: len(s["text"]))["font"]

                block_no = len(page_blocks)
                page_blocks.append(
                    Block(
                        block_id=f"{doc_id}-p{page_no}-{block_no}",
                        doc_id=doc_id,
                        file_name=pdf_path.name,
                        page=page_no,
                        block_no=block_no,
                        bbox=[round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
                        font_size=round(font_size, 2),
                        font_name=font_name,
                        is_bold=is_bold,
                        is_title=False,
                        is_math=is_math_font(font_name),
                        text=text,
                    )
                )

        _mark_titles(page_blocks)
        blocks.extend(page_blocks)

    n_pages = doc.page_count
    doc.close()

    return ParsedDocument(
        doc_id=doc_id, file_name=pdf_path.name, n_pages=n_pages, blocks=blocks
    )


def parse_directory(raw_dir: str | Path) -> list[ParsedDocument]:
    """批量解析目录下的所有 PDF。"""
    raw_dir = Path(raw_dir)
    pdfs = sorted(raw_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[警告] {raw_dir} 下没有找到任何 PDF 文件")
        return []
    results = []
    for p in pdfs:
        print(f"  解析中: {p.name}")
        results.append(parse_pdf(p))
    return results
