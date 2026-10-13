"""DocMind 命令行入口（Phase 1 + Phase 2）

命令：
    python -m docmind.cli ingest                   解析 PDF -> 切分 -> BM25 索引 -> 向量索引
    python -m docmind.cli ingest --no-embed        只建 BM25，跳过向量（快速迭代用）
    python -m docmind.cli search "问题" --mode hybrid  双路融合检索
    python -m docmind.cli compare "问题"           对比三路召回差异（最能体现技术含量的演示）
    python -m docmind.cli ask "问题"               Agent 闭环问答（需要本地 LLM）
    python -m docmind.cli stats / chunks
"""

from __future__ import annotations

import sys
from pathlib import Path

# Windows 控制台默认编码是 GBK，打印中文容易炸。强制 UTF-8。
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import argparse
import json

from rich.console import Console
from rich.table import Table

from docmind.ingest.chunker import Chunk, chunk_by_title
from docmind.ingest.pdf_parser import Block, ParsedDocument, parse_directory
from docmind.retrieve.bm25 import BM25Retriever, tokenize

console = Console()

ROOT = Path(__file__).resolve().parent.parent
DIR_RAW = ROOT / "data" / "raw"
DIR_PARSED = ROOT / "data" / "processed" / "parsed"
DIR_INDEX = ROOT / "data" / "processed" / "index"
FILE_CORPUS = ROOT / "data" / "processed" / "chunks.json"
FILE_BM25 = DIR_INDEX / "bm25.pkl"
FILE_VEC = DIR_INDEX / "vectors.npz"


def save_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([c.to_dict() for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_chunks(path: Path) -> list[Chunk]:
    if not path.exists():
        return []
    return [Chunk.from_dict(d) for d in json.loads(path.read_text(encoding="utf-8"))]


def _build_vector_index(chunks: list[Chunk], preset: str) -> bool:
    """构建向量索引。失败时返回 False 而不抛异常 —— CLI 不该因为可选功能崩掉。"""
    try:
        from docmind.retrieve.embeddings import EmbeddingModel
    except ImportError as e:
        console.print(f"[yellow]跳过向量索引，缺少依赖: {e}[/yellow]")
        return False
    try:
        model = EmbeddingModel(preset=preset)
        # bge-m3 支持 8192 长度，但论文里单句通常不超 1024。
        # 这里做一个截断保护，防止个别超长 chunk 把显存打爆。
        texts = [c.text[:3000] for c in chunks]
        vecs = model.encode(texts, is_query=False)
        from docmind.retrieve.vector import VectorRetriever

        VectorRetriever(
            vectors=vecs, chunk_ids=[c.chunk_id for c in chunks]
        ).save(FILE_VEC)
        return True
    except Exception as e:
        console.print(f"[red]向量索引构建失败: {e}[/red]")
        console.print("[dim]不影响 BM25 检索，可稍后排查或用 --preset small 换小模型[/dim]")
        return False


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> None:
    DIR_RAW.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(DIR_RAW.glob("*.pdf"))
    if not pdfs:
        console.print(f"[yellow]data/raw 下没有 PDF。请放进:[/yellow] {DIR_RAW}")
        return

    console.print(f"[cyan]发现 {len(pdfs)} 个 PDF，开始解析...[/cyan]")
    docs = parse_directory(DIR_RAW)

    all_blocks: list[Block] = []
    table = Table(title="解析统计（Block = PDF 里的一个文本行）")
    table.add_column("文件名", overflow="fold")
    table.add_column("页数", justify="right")
    table.add_column("文本行", justify="right")
    table.add_column("总字数", justify="right")
    table.add_column("标题数", justify="right")
    for doc in docs:
        doc.to_json(DIR_PARSED / f"{Path(doc.file_name).stem}.json")
        all_blocks.extend(doc.blocks)
        s = doc.stats()
        table.add_row(
            doc.file_name, str(s.get("pages", 0)), str(s.get("blocks", 0)),
            str(s.get("words_total", 0)), str(s.get("blocks_title", 0)),
        )
    console.print(table)

    console.print(f"[cyan]切分（max_words={args.max_words}）...[/cyan]")
    chunks = chunk_by_title(all_blocks, max_words=args.max_words)
    save_chunks(chunks, FILE_CORPUS)

    if chunks:
        sizes = sorted(c.word_count for c in chunks)
        words = sum(c.word_count for c in chunks)
        t2 = Table(title="切分统计")
        t2.add_column("指标"); t2.add_column("值", justify="right")
        t2.add_row("Chunk 总数", str(len(chunks)))
        t2.add_row("平均字数", str(round(words / len(chunks), 1)))
        t2.add_row("最小/中位/最大", f"{sizes[0]} / {sizes[len(sizes)//2]} / {sizes[-1]}")
        console.print(t2)

    console.print("[cyan]构建 BM25 索引...[/cyan]")
    BM25Retriever(chunks).save(FILE_BM25)
    console.print(f"[green]BM25 完成（{len(chunks)} 个 chunk）[/green]")

    if args.no_embed:
        console.print("[dim]已跳过向量索引（--no-embed）[/dim]")
    else:
        console.print(f"[cyan]构建向量索引（模型 preset={args.preset}）...[/cyan]")
        if _build_vector_index(chunks, args.preset):
            console.print("[green]向量索引完成[/green]")

    console.print("\n[green]试试:[/green]  python -m docmind.cli compare \"你的问题\"")


# ---------------------------------------------------------------------------
# stats / chunks
# ---------------------------------------------------------------------------


def cmd_stats(args: argparse.Namespace) -> None:
    chunks = load_chunks(FILE_CORPUS)
    if not chunks:
        console.print("[yellow]还没有语料，请先运行 ingest[/yellow]")
        return
    sizes = sorted(c.word_count for c in chunks)
    total = sum(c.word_count for c in chunks)
    t = Table(title="语料概览")
    t.add_column("指标"); t.add_column("值", justify="right")
    t.add_row("Chunk 数", str(len(chunks)))
    t.add_row("总字数", str(total))
    t.add_row("平均字数", str(round(total / len(chunks), 1)))
    t.add_row("最短/最长", f"{sizes[0]} / {sizes[-1]}")
    t.add_row("来源文档数", str(len({c.file_name for c in chunks})))
    t.add_row("BM25 索引", "有" if FILE_BM25.exists() else "无")
    if FILE_VEC.exists():
        import numpy as np

        d = np.load(FILE_VEC, allow_pickle=True)
        t.add_row("向量索引", f"有 ({d['vectors'].shape[0]} x {d['vectors'].shape[1]})")
    else:
        t.add_row("向量索引", "无")
    console.print(t)


def cmd_chunks(args: argparse.Namespace) -> None:
    chunks = load_chunks(FILE_CORPUS)
    if not chunks:
        console.print("[yellow]还没有语料，请先运行 ingest[/yellow]")
        return
    console.print(f"[dim]共 {len(chunks)} 个 chunk，随机抽查 3 个：[/dim]\n")
    import random

    for c in random.sample(chunks, min(3, len(chunks))):
        console.rule(f"[bold]{c.chunk_id}[/bold]", style="dim")
        console.print(
            f"[dim]{c.file_name} P{c.page_start} | {c.word_count}字 | "
            f"由{c.n_blocks}行合并 | 小节: {c.title or '(无)'}[/dim]"
        )
        console.print(c.text[:400] + ("..." if len(c.text) > 400 else ""))
        console.print()


# ---------------------------------------------------------------------------
# ask —— Agent 问答（Phase 2）
# ---------------------------------------------------------------------------


def cmd_ask(args: argparse.Namespace) -> None:
    """Agent 闭环问答：规划 → 检索 → 反思 → 生成 → 校验。

    这是整个项目的主入口。和 search 的区别：
        search : 只召回，不做生成，用来调检索
        ask    : 完整闭环，Agent 会自己判断证据够不够、要不要再检索
    """
    chunks, bm25, vector, embedder = _load_searchers("hybrid")
    if chunks is None:
        console.print("[yellow]还没有语料，请先运行 ingest[/yellow]")
        return
    if vector is None or embedder is None:
        console.print("[yellow]需要向量索引，请先运行 ingest（不带 --no-embed）[/yellow]")
        return

    from docmind.agent.react import (
        OVERALL_FAILED, OVERALL_PASSED, AgentConfig, DocMindAgent,
    )
    from docmind.generate.llm import LocalLLM
    from docmind.retrieve.fusion import HybridRetriever

    console.print(f"[cyan]加载 LLM ({args.model}, 4-bit 量化)...[/cyan]")

    # 关键一步：先把 embedding 模型挪到 CPU，给 LLM 腾出显存。
    # 8GB 显卡上不这么做，bitsandbytes 会把部分 LLM 层 offload 到 CPU，
    # 生成速度会从 25 tok/s 掉到 3 tok/s。详见 EmbeddingModel.to() 的注释。
    if embedder is not None:
        embedder.to("cpu")
        console.print("[dim]embedding 模型已移至 CPU，显存让给 LLM[/dim]")

    try:
        llm = LocalLLM(model_name=args.model, load_in_4bit=not args.fp16)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return
    except ValueError as e:
        console.print(f"[red]显存不足，部分层被卸载到 CPU: {e}[/red]")
        console.print("[dim]先关掉占用显存的程序（游戏/浏览器硬件加速），或改用 3B："
                      "python scripts/download_llm.py 3b[/dim]")
        return
    console.print(
        f"[green]LLM 就绪[/green] [dim]加载 {llm.load_seconds:.1f}s，"
        f"占用显存 {llm.memory_gb:.2f}GB[/dim]\n"
    )

    hybrid = HybridRetriever(bm25, vector, embedder, rrf_k=60)
    agent = DocMindAgent(
        hybrid,
        llm,
        AgentConfig(max_rounds=args.rounds, enable_verify=not args.no_verify, opaque_citations=True,source_quote_answers=True),
    )

    console.print(f"[cyan]问题:[/cyan] {args.query}\n")
    result = agent.run(args.query)

    # --- 思考轨迹 ---
    console.rule("[bold]Agent 思考过程[/bold]", style="dim cyan")
    console.print(result.trace.summary())

    # --- 答案 ---
    console.rule("[bold]回答[/bold]", style="green")
    console.print(result.answer)

    # --- 综合校验结论 ---
    # 这里显示的是**综合状态**：LLM 校验结果与零成本引用合法性检查合并后的结论。
    # 只要规则查出越界引用，整体就不是通过 —— 哪怕模型自己说"没问题"。
    # （第二轮修复：此前只显示 LLM 的那一票，出现过"查出 [99] 却仍显示通过"的矛盾。）
    st = result.verification_status
    # 两个来源始终都打印，方便一眼看出"到底是模型没过，还是引用越界"
    detail = (f"  [dim]LLM 节点状态: {result.llm_verification_status}；"
              f"引用合法性: {result.citation_validity}[/dim]")
    if st == OVERALL_PASSED:
        console.print("\n[green]模型自校验: 通过；引用编号未越界[/green]")
        console.print(detail)
        console.print("[dim]模型可能漏检错误；此状态不代表答案事实或数学公式已经核定。[/dim]")
    elif st == OVERALL_FAILED:
        console.print("\n[red]校验结论: 未通过（仅报告，未自动修订）[/red]")
        console.print(f"  [dim]原因: {', '.join(result.verification_fail_reasons)}[/dim]")
        console.print(detail)
        for it in result.issues:
            console.print(f"  [red]·[/red] {it}")
    else:
        console.print(f"\n[yellow]校验结论: {st}（不能视为通过）[/yellow]")
        console.print(detail)

    # --- 零成本的引用编号范围检查（无论是否启用 LLM 校验都跑）---
    if result.citation_issues:
        console.print("[red]引用编号越界:[/red]")
        for it in result.citation_issues:
            console.print(f"  [red]·[/red] {it}")

    # --- 证据溯源 ---
    console.rule("[bold]引用来源[/bold]", style="dim")
    for i, e in enumerate(result.evidence, 1):
        tag = "[文件]" if not e.get("title") else f"[文件 · {e['title'][:30]}]"
        console.print(f"[dim][{i}] {e['source'][:70]}  "
                      f"(bm25#{e['bm25_rank']} vec#{e['vector_rank']})[/dim]")

    console.print(
        f"\n[dim]反思 {result.rounds_used} 轮 | 停止原因 {result.stop_reason} | "
        f"输出 {result.output_tokens} token（预算口径）/ 提示 "
        f"{result.prompt_tokens} token | {result.elapsed:.1f}s[/dim]"
    )


# ---------------------------------------------------------------------------
# search / compare
# ---------------------------------------------------------------------------


def _load_searchers(mode: str):
    """按需加载检索器。返回 (chunks, bm25, vector, embedder)"""
    chunks = load_chunks(FILE_CORPUS)
    if not chunks:
        return None, None, None, None
    bm25 = BM25Retriever.load(FILE_BM25, chunks) if FILE_BM25.exists() else None
    vector = None
    embedder = None
    if mode in ("vector", "hybrid") and FILE_VEC.exists():
        from docmind.retrieve.embeddings import EmbeddingModel
        from docmind.retrieve.vector import VectorRetriever

        vector = VectorRetriever.load(FILE_VEC)
        embedder = EmbeddingModel(preset="m3")
    return chunks, bm25, vector, embedder


def cmd_search(args: argparse.Namespace) -> None:
    chunks, bm25, vector, embedder = _load_searchers(args.mode)
    if chunks is None:
        console.print("[yellow]还没有语料，请先运行 ingest[/yellow]")
        return
    if args.mode in ("vector", "hybrid") and (vector is None or embedder is None):
        console.print("[yellow]没有可用的向量索引，退回 BM25 模式。[/yellow]")
        console.print("[dim]运行 ingest（不带 --no-embed）来构建向量索引[/dim]")
        args.mode = "bm25"

    console.print(f"\n[cyan]查询:[/cyan] {args.query}   [dim](模式: {args.mode})[/dim]\n")

    if args.mode == "bm25":
        hits = bm25.search(args.query, top_k=args.top_k)
        for i, (score, c) in enumerate(hits, 1):
            console.rule(f"[bold]#{i}  BM25={score:.3f}[/bold]", style="dim")
            meta = f"{c.file_name} P{c.page_start}"
            if c.title:
                meta += f"  [小节: {c.title}]"
            console.print(f"[dim]{meta}[/dim]")
            console.print(c.text[: args.snippet] + ("..." if len(c.text) > args.snippet else ""))
            console.print()
        return

    # hybrid / vector
    from docmind.retrieve.fusion import HybridRetriever

    hybrid = HybridRetriever(bm25, vector, embedder, rrf_k=args.rrf_k)
    results = hybrid.search(args.query, top_k=args.top_k, mode=args.mode)

    if not results:
        console.print("[yellow]无命中[/yellow]")
        return

    for r in results:
        tag = r.source.upper()
        detail = f"RRF={r.score:.4f}"
        if r.bm25_rank:
            detail += f"  BM25#{r.bm25_rank}"
        if r.vector_rank:
            detail += f"  VEC#{r.vector_rank}"
            if r.vector_score:
                detail += f"(cos={r.vector_score:.3f})"
        color = "green" if r.source == "both" else ("cyan" if r.source == "vector" else "yellow")
        console.rule(f"[bold #{color}]#{r.rank}  {tag}[/bold #{color}]  [dim]{detail}[/dim]", style="dim")
        c = r.chunk
        if c is None:
            console.print(f"[red](找不到 chunk {r.chunk_id})[/red]")
            continue
        meta = f"{c.file_name} P{c.page_start}"
        if c.title:
            meta += f"  [小节: {c.title}]"
        console.print(f"[dim]{meta}[/dim]")
        console.print(c.text[: args.snippet] + ("..." if len(c.text) > args.snippet else ""))
        console.print()


def cmd_compare(args: argparse.Namespace) -> None:
    """三路对比 —— 这个项目最能体现技术含量的一条命令。

    它能直观展示：哪些结果只有 BM25 能找到（专有名词），
    哪些只有向量能找到（同义改写），哪些两路都命中（高置信）。
    """
    chunks, bm25, vector, embedder = _load_searchers("hybrid")
    if chunks is None:
        console.print("[yellow]还没有语料[/yellow]")
        return
    if vector is None or embedder is None:
        console.print("[yellow]需要向量索引，请先运行 ingest（不带 --no-embed）[/yellow]")
        return

    from docmind.retrieve.fusion import HybridRetriever

    hybrid = HybridRetriever(bm25, vector, embedder, rrf_k=60)
    k = args.top_k

    bm25_hits = bm25.search(args.query, top_k=k)
    vec_hits = vector.search(embedder.encode_query(args.query), top_k=k)
    fused = hybrid.search(args.query, top_k=k, mode="hybrid")

    console.print(f"\n[cyan]查询:[/cyan] {args.query}\n")
    console.print("[dim]绿色 = 两路都命中（高置信）  黄色 = 仅 BM25  青色 = 仅向量[/dim]\n")

    def _show(items, title, extract):
        t = Table(title=title)
        t.add_column("#", justify="right", width=3)
        t.add_column("来源")
        t.add_column("分数", justify="right")
        t.add_column("内容预览", overflow="fold")
        for i, it in enumerate(items, 1):
            cid, score, src = extract(it)
            pos = bm25.block_index.get(cid)
            text = chunks[pos].text if pos is not None else "(缺失)"
            short = text[:70].replace("\n", " ")
            cbox = {"both": "[green]双路[/green]", "bm25": "[yellow]BM25[/yellow]"}.get(src, "[cyan]向量[/cyan]")
            t.add_row(str(i), cbox, f"{score:.4f}", short)
        console.print(t)
        console.print()

    _show(bm25_hits, f"仅 BM25 Top-{k}", lambda it: (it[1].chunk_id, it[0], "bm25"))
    _show(vec_hits, f"仅向量 Top-{k}", lambda it: (it.chunk_id, it.score, "vector"))
    _show(fused, f"RRF 融合 Top-{k}", lambda it: (it.chunk_id, it.score, it.source))

    both = [r for r in fused if r.source == "both"]
    console.print(
        f"[green]重合率: {len(both)}/{k} = {len(both)/k:.0%}[/green]"
        "  [dim]（这个数字直观反映了两路召回的互补程度）[/dim]"
    )


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="docmind", description="DocMind —— 长文档深度问答 Agent（命令行版）"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="解析 PDF 并建立索引")
    p_ingest.add_argument("--max-words", type=int, default=300)
    p_ingest.add_argument("--no-embed", action="store_true", help="跳过向量索引")
    p_ingest.add_argument("--preset", default="m3", choices=["small", "base", "m3"])
    p_ingest.set_defaults(func=cmd_ingest)

    sub.add_parser("stats", help="查看语料统计").set_defaults(func=cmd_stats)
    sub.add_parser("chunks", help="抽查切分结果").set_defaults(func=cmd_chunks)

    p_search = sub.add_parser("search", help="检索")
    p_search.add_argument("query")
    p_search.add_argument("-k", "--top-k", type=int, default=5)
    p_search.add_argument("-s", "--snippet", type=int, default=350)
    p_search.add_argument("--mode", default="hybrid", choices=["bm25", "vector", "hybrid"])
    p_search.add_argument("--rrf-k", type=int, default=60)
    p_search.set_defaults(func=cmd_search)

    p_cmp = sub.add_parser("compare", help="对比三路召回")
    p_cmp.add_argument("query")
    p_cmp.add_argument("-k", "--top-k", type=int, default=5)
    p_cmp.set_defaults(func=cmd_compare)

    p_ask = sub.add_parser("ask", help="Agent 闭环问答（规划→检索→反思→生成→校验）")
    p_ask.add_argument("query")
    p_ask.add_argument("--model", default="Qwen2.5-7B-Instruct")
    p_ask.add_argument("--rounds", type=int, default=3, help="最多反思轮数")
    p_ask.add_argument("--no-verify", action="store_true", help="跳过引用校验（省时间）")
    p_ask.add_argument("--fp16", action="store_true", help="不量化（需要 16GB+ 显存）")
    p_ask.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
