# 切分模块回归测试

"""
重点覆盖跨文档污染（第一轮审核发现的问题 1）。

这些测试不是走过场 —— 它们针对的是真实复现过的 bug：
原实现在 335 个 chunk 里产出了 5 个横跨两篇论文的 chunk，
而 chunk_id 取自 buffer[0]，光看 ID 完全查不出来。

所以这里的断言全部落在 **chunk 内部 block 的来源文档集合** 上，
而不是只看 doc_id 字段。只看字段的话，原来的 bug 依然能通过测试。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docmind.ingest.chunker import chunk_by_title, chunk_fixed_window
from docmind.ingest.pdf_parser import Block


def mk(doc_id: str, page: int, no: int, text: str, title: bool = False) -> Block:
    """构造一个 Block 用于测试。

    word_count 是 property，由 text 推导，所以要保证 text 有内容。
    """
    return Block(
        block_id=f"{doc_id}-p{page}-{no}",
        doc_id=doc_id,
        file_name=f"{doc_id}.pdf",
        page=page,
        block_no=no,
        text=text,
        font_size=14.0 if title else 10.0,
        font_name="Times" if title else "CMMI",   # 标题用正文字体，其余标记为数学字体
        is_bold=title,
        is_title=title,
        is_math=not title,
        bbox=[0.0, 0.0, 100.0, 10.0],
    )


def docs_of(chunk_text: str, blocks: list[Block]) -> set[str]:
    """推断一个 chunk 的文本里出现了哪些文档的独有标记。

    测试里给每个文档的文本加上 "DOC_<id>_" 前缀做标记，
    这样就能从合并后的文本反推出混合了哪些文档。
    """
    return {b.doc_id for b in blocks if f"DOC_{b.doc_id}_" in chunk_text}


# ---------------------------------------------------------------------------
# chunk_by_title：文档边界
# ---------------------------------------------------------------------------

def test_doc_boundary_flushes_residual():
    """前一文档末尾不足 min_words 时，也必须 flush，不能并入下一篇。

    这是最初产生污染的典型场景：论文结尾的致谢/页码只有几个词，
    达不到 min_words=20，于是 buffer 一直攒着，直到下一篇论文的内容进来。
    """
    blocks = [
        mk("A", 1, 0, "DOC_A_" + " ".join(["alpha"] * 60)),
        mk("A", 2, 0, "DOC_A_ Acknowledgments " + " ".join(["x"] * 8)),  # 不足 min_words
        mk("B", 1, 0, "DOC_B_" + " ".join(["beta"] * 60)),
    ]
    chunks = chunk_by_title(blocks, max_words=300, min_words=20)

    for c in chunks:
        assert len(docs_of(c.text, blocks)) <= 1, (
            f"chunk 跨文档: {c.chunk_id} -> {docs_of(c.text, blocks)}"
        )
    assert len(chunks) >= 2, "两篇文档至少应各自产出一个 chunk"


def test_doc_boundary_resets_title():
    """后一文档没有标题时，不能沿用上一篇的小节标题。

    否则溯源会显示"来自论文 B 的 'Numerical experiments' 章节"，
    而实际上论文 B 根本没有那一节。
    """
    blocks = [
        mk("A", 1, 0, "Introduction", title=True),
        mk("A", 1, 1, "DOC_A_" + " ".join(["alpha"] * 40)),
        # 文档 B 没有任何标题块
        mk("B", 1, 0, "DOC_B_" + " ".join(["beta"] * 40)),
    ]
    chunks = chunk_by_title(blocks, max_words=300, min_words=20)
    b_chunks = [c for c in chunks if c.doc_id == "B"]

    assert b_chunks, "文档 B 应有 chunk"
    for c in b_chunks:
        assert c.title == "", f"文档 B 的 chunk 不应带标题，实际为 {c.title!r}"


def test_title_retained_within_doc():
    """同一文档内，标题仍应正确传递给其后的正文 chunk（不能修过头）。"""
    blocks = [
        mk("A", 1, 0, "Numerical experiments", title=True),
        mk("A", 1, 1, "DOC_A_" + " ".join(["alpha"] * 40)),
    ]
    chunks = chunk_by_title(blocks, max_words=300, min_words=20)
    assert any(c.title == "Numerical experiments" for c in chunks), \
        "同文档内标题应正常继承"


def test_no_title_in_second_doc_keeps_empty():
    """后一文档虽有标题，但前一篇残留不足时仍应正确隔离。"""
    blocks = [
        mk("A", 1, 0, "DOC_A_" + " ".join(["a"] * 5)),          # 极短残留
        mk("B", 1, 0, "Method", title=True),
        mk("B", 1, 1, "DOC_B_" + " ".join(["beta"] * 40)),
    ]
    chunks = chunk_by_title(blocks, max_words=300, min_words=20)
    for c in chunks:
        assert len(docs_of(c.text, blocks)) <= 1
    b = [c for c in chunks if c.doc_id == "B"]
    assert any(c.title == "Method" for c in b), "文档 B 自己的标题应生效"


def test_ids_unique_and_pages_within_doc():
    """chunk_id 唯一，且页码必须落在所属文档的页码范围内。"""
    blocks = []
    for d, pages in (("A", (1, 2, 3)), ("B", (1, 2))):
        for p in pages:
            blocks.append(mk(d, p, 0, f"DOC_{d}_ " + " ".join(["w"] * 40)))
    chunks = chunk_by_title(blocks, max_words=300, min_words=20)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_id 必须唯一"

    pages_by_doc = {"A": {1, 2, 3}, "B": {1, 2}}
    for c in chunks:
        assert c.page_start in pages_by_doc[c.doc_id], \
            f"page_start {c.page_start} 不属于文档 {c.doc_id}"
        assert c.page_end in pages_by_doc[c.doc_id], \
            f"page_end {c.page_end} 不属于文档 {c.doc_id}"


# ---------------------------------------------------------------------------
# chunk_fixed_window：文档隔离
# ---------------------------------------------------------------------------

def test_fixed_window_isolates_docs():
    """固定窗口不得跨文档拼接。"""
    blocks = [
        mk("A", 1, 0, "DOC_A_" + " ".join(["alpha"] * 120)),
        mk("A", 3, 0, "DOC_A_" + " ".join(["gamma"] * 120)),
        mk("B", 1, 0, "DOC_B_" + " ".join(["beta"] * 120)),
    ]
    chunks = chunk_fixed_window(blocks, size=200, overlap=50)

    for c in chunks:
        src = docs_of(c.text, blocks)
        assert len(src) <= 1, f"窗口跨文档: {c.chunk_id} -> {src}"


def test_fixed_window_ids_unique_and_doc_attributed():
    """每个窗口的 doc_id/file_name 必须对应它真正的来源文档。"""
    blocks = [
        mk("A", 1, 0, "DOC_A_" + " ".join(["alpha"] * 120)),
        mk("B", 1, 0, "DOC_B_" + " ".join(["beta"] * 120)),
        mk("C", 5, 0, "DOC_C_" + " ".join(["gamma"] * 120)),
    ]
    chunks = chunk_fixed_window(blocks, size=150, overlap=30)

    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk_id 必须唯一"

    seen_docs = {c.doc_id for c in chunks}
    assert seen_docs == {"A", "B", "C"}, f"应覆盖三篇文档，实际 {seen_docs}"

    for c in chunks:
        assert c.file_name == f"{c.doc_id}.pdf", \
            f"file_name 与 doc_id 不匹配: {c.doc_id} vs {c.file_name}"


def test_fixed_window_pages_from_real_blocks():
    """页码必须由窗口实际覆盖的 block 推导，不能恒为 1。"""
    blocks = [
        mk("A", 1, 0, "DOC_A_" + " ".join(["alpha"] * 60)),
        mk("A", 7, 0, "DOC_A_" + " ".join(["mid"] * 60)),
        mk("A", 9, 0, "DOC_A_" + " ".join(["last"] * 60)),
    ]
    chunks = chunk_fixed_window(blocks, size=120, overlap=20)

    all_pages = {c.page_start for c in chunks} | {c.page_end for c in chunks}
    assert all_pages <= {1, 7, 9}, f"页码超出文档实际页范围: {all_pages}"
    assert max(c.page_end for c in chunks) >= 7, \
        "靠后的窗口应覆盖到后面的页码，不能全部标成第 1 页"
