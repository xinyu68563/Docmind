# DocMind —— 文档摄取：切分（Chunking）

"""
文本切分模块（Phase 0）

【这是整个 RAG 系统中最重要的一个坑，没有之一】

刚才我们遇到的问题：简历里每一行都是一个独立 Block，导致检索时
"我在哪家公司实习过" 命中的是 "实习经历" 这个标题，而不是下面的公司名。

根本原因：**Block（视觉上的一行）≠ Chunk（语义上完整的一段）**

PDF 是按视觉位置排版的，一个自然段的文字在 PDF 里可能就是 5 个互不相邻的
文本行。如果我们直接拿行去检索，会出现两个致命问题：
    1. 语义不完整：一行话往往不构成完整的论据，送给 LLM 也没用
    2. 上下文丢失："相比高斯方法提升了 15%" —— 提升什么的 15%？没有上文就废了

【两种主流切分策略，我先把它们实现出来，W3 用数据对比】

策略 A：固定窗口滑窗（fixed-size sliding window）
    每隔 N 个字切一刀，相邻窗口重叠 M 个字。
    优点：实现简单、chunk 大小均匀、不会丢内容
    缺点：经常把一句话切两半，破坏语义完整性

策略 B：结构感知切分（structure-aware chunking）  ← Phase 0 默认用这个
    利用 PDF 的标题信息（我们上一节用字号检测出来的 is_title）作为切分边界，
    把同一个小标题下的正文合并成一个 chunk。
    优点：chunk 天然对应一个语义单元，检索到的东西LLM能直接用
    缺点：依赖标题识别质量；遇到没有小标题的长段落会失效

真实系统里通常是「结构感知优先 + 超长时降级为滑窗」的混合策略，这也是我们要做的。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from docmind.ingest.pdf_parser import Block


@dataclass
class Chunk:
    """一个检索单元。这是最终送进向量库和 BM25 索引的东西。"""

    chunk_id: str
    doc_id: str
    file_name: str
    page_start: int
    page_end: int
    title: str          # 所属小节标题，检索命中后可以显示给用户做溯源
    text: str
    n_blocks: int       # 由几个原始 block 合并而来
    n_math_blocks: int = 0   # 其中属于数学区域的块数

    @property
    def math_ratio(self) -> float:
        """该 chunk 中数学内容的占比。

        为什么要在 chunk 层面记录这个？
            数学论文里约一半的内容是公式（本项目实测 50.9%）。
            纯公式块的可检索性很差 —— 用户不会用自然语言去搜 "sinc(Lt − n)"。
            有了这个字段，检索层就可以对高公式占比的 chunk 做降权，
            或者在溯源展示时给用户提示"该段落以公式为主"。
            没有它，你根本不知道为什么有些结果看起来"答非所问"。
        """
        return self.n_math_blocks / self.n_blocks if self.n_blocks else 0.0

    @property
    def page(self) -> int:
        """统一接口，方便和 Block 混用在同一个打印逻辑里。"""
        return self.page_start

    @property
    def word_count(self) -> int:
        import re

        return len(re.findall(r"[\u4e00-\u9fff]|[A-Za-z]+", self.text))

    @property
    def is_title(self) -> bool:
        return False

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


def chunk_by_title(
    blocks: list[Block],
    max_words: int = 300,
    min_words: int = 20,
) -> list[Chunk]:
    """按标题边界合并 Block（策略 B）。

    算法：顺序扫描，累积 buffer；
        - 文档切换时              -> 无条件 flush 并重置标题（**必须放在最前面**）
        - 遇到标题且攒够 min_words -> 先 flush，再开新段
        - buffer 超过 max_words   -> 强制 flush（防止超长）
    最后把残余 buffer flush 掉。

    【为什么要显式处理文档边界 —— 这是一个真实修掉的 bug】

    原实现只在"遇到标题"时才 flush。于是当某篇论文的最后几行不足
    min_words（比如只有十几字的致谢或页码），buffer 不会被 flush，
    扫描继续进入下一篇论文，下一篇的开头就被拼进了上一篇的最后一个 chunk。

    实测本项目 335 个 chunk 里有 **5 个** 是这样产生的，它们的
    page_start 是上一篇的末尾页、page_end 是下一篇的第 1 页，
    内容横跨两篇毫不相关的论文。

    这个 bug 很隐蔽：chunk_id 取自 buffer[0]，所以 chunk_id 看起来
    完全"属于"上一篇文档，光看 ID 查不出来；但 chunk 的文本里混进了
    另一篇论文的内容。检索命中它时，LLM 会拿到两段无关文字拼成的证据。

    参数为什么不写死 512？
        因为 max_words 这个值是 W3 消融实验的核心变量之一。
        到时候你会跑 128/256/512/1024 四组，画成表格放进 README。
        现在先给一个偏小的默认值，是因为本地小模型的上下文窗口有限。
    """
    chunks: list[Chunk] = []
    if not blocks:
        return chunks

    # 保证处理顺序正确：按文档 -> 页码 -> 页内序号
    ordered = sorted(blocks, key=lambda b: (b.doc_id, b.page, b.block_no))

    cur_title = ""
    buffer: list[Block] = []

    def _words(bs: list[Block]) -> int:
        return sum(b.word_count for b in bs)

    def _flush() -> None:
        nonlocal buffer
        if not buffer:
            return
        text = " ".join(b.text for b in buffer).strip()
        if not text:
            buffer = []
            return
        first, last = buffer[0], buffer[-1]
        # 防御：万一仍有混排（例如上游传入脏数据），宁可丢弃也不要产出跨文档 chunk
        doc_ids = {b.doc_id for b in buffer}
        if len(doc_ids) > 1:
            keep = first.doc_id
            buffer = [b for b in buffer if b.doc_id == keep]
            if not buffer:
                return
            last = buffer[-1]
            text = " ".join(b.text for b in buffer).strip()
        chunks.append(
            Chunk(
                chunk_id=f"{first.doc_id}-p{first.page}-{first.block_no}",
                doc_id=first.doc_id,
                file_name=first.file_name,
                page_start=first.page,
                page_end=last.page,
                title=cur_title,
                text=text,
                n_blocks=len(buffer),
                n_math_blocks=sum(1 for b in buffer if getattr(b, "is_math", False)),
            )
        )
        buffer = []

    prev_doc: str | None = None
    for b in ordered:
        # 文档边界处理：必须在标题判断之前，否则跨论文污染
        if prev_doc is not None and b.doc_id != prev_doc:
            _flush()
            cur_title = ""      # 标题属于上一篇论文，不能带到下一篇
        prev_doc = b.doc_id

        # 【这里修掉了一个"标题丢失"bug】
        # 原写法是 `b.is_title and _words(buffer) >= min_words`。
        # 当 buffer 为空（刚 flush 过，或文档刚开头）时 _words(buffer)==0，
        # 条件恒为假，于是标题块被当成普通正文塞进 buffer，cur_title 永远不更新 ——
        # 整篇文档的 title 字段全是空字符串，溯源功能等于失效。
        #
        # min_words 的本意是"避免为上一段的短残余单开一个 chunk"，
        # 它只在 buffer 非空时才有意义；buffer 为空时没有残余可保护，
        # 应当直接开新段。所以补一个 `not buffer` 分支。
        is_new_section = b.is_title and (not buffer or _words(buffer) >= min_words)

        if is_new_section:
            _flush()
            cur_title = b.text
            buffer = [b]       # 标题本身也放进 chunk，方便检索时命中章标题
            continue

        buffer.append(b)
        if _words(buffer) >= max_words:
            _flush()

    _flush()
    return chunks


def chunk_fixed_window(blocks: list[Block], size: int = 300, overlap: int = 50) -> list[Chunk]:
    """按固定字数滑窗切分（策略 A）。

    W3 做对照实验时会用到。现在先实现好，两个策略都能跑，
    后面才能在同一个评测集上公平对比。

    【修复说明：原实现有三个问题】

    1. **跨文档拼接**：把所有论文的原文拼成一个大字符串再切，
       于是窗口会横跨论文边界 —— 和 chunk_by_title 同源的污染 bug。
    2. **doc_id / file_name 全部取自第一篇**：后续所有 chunk 都被
       错误地标注成第一篇论文，溯源信息完全错误。
    3. **页码恒为 1、page_end 恒为最后一篇的末页**：
       因为代码直接写了 page_start=1，没有任何真实来源。

    修复后：按文档分组各自滑窗；chunk_id 带 doc_id 前缀保证全局唯一；
    页码由窗口实际覆盖的 block 反推。
    """
    if not blocks:
        return []

    ordered = sorted(blocks, key=lambda b: (b.doc_id, b.page, b.block_no))

    # 按文档分组 —— 这是隔离的第一步
    by_doc: dict[str, list[Block]] = {}
    for b in ordered:
        by_doc.setdefault(b.doc_id, []).append(b)

    chunks: list[Chunk] = []
    step = max(1, size - overlap)

    for doc_id, doc_blocks in sorted(by_doc.items()):
        full_text = "\n".join(b.text for b in doc_blocks)
        if not full_text.strip():
            continue

        # 字符区间 -> block 的映射，用于把窗口还原成真实页码
        spans: list[tuple[int, int, Block]] = []
        pos = 0
        for b in doc_blocks:
            spans.append((pos, pos + len(b.text), b))
            pos += len(b.text) + 1      # +1 是拼接用的换行

        idx = 0
        start = 0
        while start < len(full_text):
            end = start + size
            piece = full_text[start:end]
            if piece.strip():
                # 找出与该窗口有重叠的所有 block，据此确定起止页码
                covered = [b for s, e, b in spans if s < end and e > start]
                page_start = covered[0].page if covered else doc_blocks[0].page
                page_end = covered[-1].page if covered else doc_blocks[0].page
                chunks.append(
                    Chunk(
                        chunk_id=f"{doc_id}-w{idx}",   # 带 doc 前缀，全局唯一
                        doc_id=doc_id,
                        file_name=doc_blocks[0].file_name,
                        page_start=page_start,
                        page_end=page_end,
                        title="",
                        text=piece.strip(),
                        n_blocks=len(covered),
                        n_math_blocks=sum(
                            1 for b in covered if getattr(b, "is_math", False)
                        ),
                    )
                )
            idx += 1
            start += step

    return chunks
