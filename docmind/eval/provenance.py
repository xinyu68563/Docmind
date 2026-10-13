# DocMind —— 实验身份与可解析快照

"""
【为什么要有这个模块】

一份实验结果要想在半年后还能被复核，光有数字是不够的，必须能回答：

    1. 当时跑的是**哪份代码**？（改一行切分逻辑，指标含义就变了）
    2. 当时用的是**哪个语料 / 哪个 gold / 哪个索引**？
    3. 当时用的是**哪个 embedding、哪个 LLM、什么量化**？

第三轮审核发现的两个具体问题，都是这个模块要解决的：

    - 检索结果 JSON 里 `model: null`，embedding 身份完全缺失
    - smoke 里证据正文被 `[:400]` 截断（42 条），且没有任何语料版本身份，
      于是"拿交付物复核当时完整证据"这件事根本做不到

【设计原则：能解析，而不是能声明】

记录哈希的价值不在"我写了个哈希"，而在"拿到 JSON 的人能拿它去对上某个东西"。
所以本模块提供 `resolve_snapshot()`：输入一份 metadata，输出每一项
`match / mismatch / missing` 的核对结果。只有 `resolve` 能跑通的记录才算数。

【关于 revision：不能捏造】

本地加载的模型没有 git revision 可言（目录下没有 `.git`、也不是从 hub 拉的）。
所以这里**如实记录 null 并注明原因**，而不是写一个看起来像 revision 的字符串。
造假的版本号比缺失更糟 —— 它会让人以为可以精确复现，实际不能。
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

DIR_DATA = ROOT / "data"
FILE_CHUNKS = DIR_DATA / "processed" / "chunks.json"
FILE_PARSED_DIR = DIR_DATA / "processed" / "parsed"
FILE_BM25 = DIR_DATA / "processed" / "index" / "bm25.pkl"
FILE_VEC = DIR_DATA / "processed" / "index" / "vectors.npz"
FILE_GOLD = DIR_DATA / "eval" / "goldset.json"
DIR_MODELS = DIR_DATA / "models"

# 关键代码文件：这些文件一改，实验数字的含义就可能变。
#
# 第三轮扩充了这一组 —— 原来只有 5 个，缺了 embedding / vector / LLM / prompts
# 和运行脚本本身，于是"检索结果里向量那一路用的是什么"完全无从考证。
CODE_FILES = [
    "docmind/ingest/source_context.py",
    "data/source_reviews/math_v1/notes.json",
    # 解析与切分
    "docmind/ingest/pdf_parser.py",
    "docmind/ingest/chunker.py",
    # 检索
    "docmind/retrieve/bm25.py",
    "docmind/retrieve/embeddings.py",
    "docmind/retrieve/vector.py",
    "docmind/retrieve/fusion.py",
    # 生成与 Agent
    "docmind/generate/llm.py",
    "docmind/agent/prompts.py",
    "docmind/agent/react.py",
    # 评测
    "docmind/eval/metrics.py",
    "docmind/eval/provenance.py",
    # 第十二轮新增：逐调用日志模块**必须**进运行身份。
    # 它决定了日志长什么样（schema 版本、seq 语义、usage 三态），
    # 也就是说"当时能不能留下证据、留下来的证据能不能按 seq 切片"由它决定。
    # 更重要的是：独立校验器要靠"这份快照里有没有 call_journal.py"
    # 来判断该产物**是否具备**日志能力 —— 这是个无法靠删产物字段绕过的证据。
    # 之前只有 AB 脚本通过 ROUND8_EXTRA_FILES 额外带上它，
    # replay 那一路的快照里没有，于是判断能力在两条链路上不一致。
    "docmind/eval/call_journal.py",
    # 运行脚本本身（跑法变了，数字也会变）
    "scripts/run_ablation.py",
    "scripts/smoke_end2end.py",
]


# ---------------------------------------------------------------------------
# 基础哈希
# ---------------------------------------------------------------------------

def sha256_file(path: str | Path, n: int = 16) -> str | None:
    """文件 sha256 前 n 位。文件不存在返回 None（不抛异常）。

    为什么截断到 16 位：元数据要嵌在 JSON 里给人读，64 位太长且没必要 ——
    16 位（64 bit）碰撞概率对"人工核对是否同一份文件"已经绰绰有余。
    """
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def sha256_text(text: str, n: int = 16) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def _read_json(path: str | Path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 各项身份
# ---------------------------------------------------------------------------

def corpus_identity(path: str | Path = FILE_CHUNKS) -> dict:
    """语料身份：路径、哈希、chunk 数、文档数。

    n_docs 用 file_name 去重统计 —— 只记 chunk 数的话，
    "同样 337 个 chunk 但来自不同论文"这种情况分辨不出来。
    """
    p = Path(path)
    out = {"path": _rel(p), "sha256": sha256_file(p), "n_chunks": None, "n_docs": None}
    data = _read_json(p)
    if isinstance(data, list):
        out["n_chunks"] = len(data)
        try:
            out["n_docs"] = len({c.get("file_name") for c in data})
        except Exception:
            pass
    return out


def goldset_identity(path: str | Path = FILE_GOLD) -> dict:
    """评测集身份。

    【第四轮】要同时认两种格式：
        - 旧版 `goldset.json`：顶层就是题目列表
        - v2 `goldset_reviewed_v2.json`：顶层是带元数据的字典，题目在 `items`
    并且额外记录 `dataset_version` / `n_included`（按 evaluation_include 统计），
    这样结果文件自己就能说明"这次用了哪版题集、纳入了多少题"，
    而不是让人去猜文件名。
    """
    p = Path(path)
    out = {"path": _rel(p), "sha256": sha256_file(p), "n_items": None,
           "dataset_version": None, "n_included": None}
    data = _read_json(p)
    if isinstance(data, list):
        out["n_items"] = len(data)
        return out
    if isinstance(data, dict):
        items = data.get("items") or []
        out["n_items"] = len(items)
        out["dataset_version"] = data.get("dataset_version")
        out["n_included"] = len(
            [i for i in items if i.get("evaluation_include") is not False])
    return out


def index_identity(bm25: str | Path = FILE_BM25,
                   vectors: str | Path = FILE_VEC) -> dict:
    return {"bm25": {"path": _rel(bm25), "sha256": sha256_file(bm25)},
            "vectors": {"path": _rel(vectors), "sha256": sha256_file(vectors)}}


def code_identity(root: str | Path = ROOT, files: list[str] | None = None) -> dict:
    root = Path(root)
    return {f: sha256_file(root / f) for f in (files or CODE_FILES)}


def embedding_identity(preset: str | None = None, model_path: str | None = None,
                       dim: int | None = None, device: str | None = None,
                       fp16: bool | None = None,
                       runtime_device: str | None = None,
                       runtime_dtype: str | None = None,
                       load_device: str | None = None,
                       load_dtype: str | None = None) -> dict:
    """embedding 模型身份。

    【不能捏造 revision】本地目录加载没有 revision 可言，
    所以 `revision` 一律为 null 并写明原因；能确证的是解析到的路径、
    config.json 的哈希与里面的架构字段（这些读文件就能拿到）。

    【第四轮修正：运行时 vs 加载时状态必须分开记】
    真实链路是：模型先 load 到 cuda（fp16），随后 end2end 里会调用
    `embedder.to("cpu")` 让出显存给 7B LLM，而 `to()` 会顺带把参数转成 fp32。
    **实际做 encode 的检索调用发生在迁移之后**，所以真正描述本次实验条件的
    是 CPU/fp32。上一轮在 `to("cpu")` 之前采集并记成 cuda/fp16，
    等于把实验条件记错了 —— 这里明确区分：
        runtime_device/dtype : 真正参与向量计算的条件（结果的可比性取决于它）
        load_device/dtype    : 加载时状态，仅作参考
    """
    # 未显式给出时沿用旧参数，保证老调用点行为不变
    if runtime_device is None:
        runtime_device = device
    if runtime_dtype is None and fp16 is not None:
        runtime_dtype = "fp16" if fp16 else "fp32"
    if load_device is None:
        load_device = device
    if load_dtype is None and fp16 is not None:
        load_dtype = "fp16" if fp16 else "fp32"

    # dtype 统一归一化到 fp16 / fp32 / bf16。
    # 为什么必须归一：torch 返回的是 "torch.float16"，
    # 而调用方/旧字段名判断的是 "fp16" —— 不做这层映射的话
    # `"float16" == "fp16"` 恒为 False，兼容字段 fp16 会被记成 False，
    # 于是"半精度跑的"被记成"没用半精度"，又是一次把实验条件记反。
    runtime_dtype = _norm_dtype(runtime_dtype)
    load_dtype = _norm_dtype(load_dtype)

    # preset 给了但没给路径时，自己解析一次 —— 让"只传 preset"也能拿到确证信息。
    # 解析失败也不要炸：身份记录不该成为实验跑不起来的原因。
    if not model_path and preset:
        try:
            from docmind.retrieve.embeddings import resolve_model_path
            model_path = resolve_model_path(preset)
        except Exception:
            model_path = None

    if runtime_device == load_device and runtime_dtype == load_dtype:
        state_note = (f"本次运行未发生设备迁移，运行态与加载态一致 "
                      f"（{runtime_device}/{runtime_dtype}）；"
                      f"实验条件以 runtime_* 为准")
    else:
        state_note = (f"加载时为 {load_device}/{load_dtype}，实际 encode 发生在"
                      f"迁移之后的 {runtime_device}/{runtime_dtype}；"
                      f"实验条件以 runtime_* 为准")

    out = {
        "preset": preset,
        "resolved_path": str(model_path) if model_path else None,
        "dim": dim,
        # device/fp16 保留旧字段名，但语义已修正为**运行时**，向下兼容读取方
        "device": runtime_device,
        "fp16": (runtime_dtype == "fp16") if runtime_dtype else fp16,
        "runtime_device": runtime_device,
        "runtime_dtype": runtime_dtype,
        "load_device": load_device,
        "load_dtype": load_dtype,
        "state_note": state_note,
        "revision": None,
        "revision_note": "本地目录加载，无 git revision；身份由解析路径 + config.json 哈希确证",
        "config_sha256": None,
        "config": {},
    }
    if model_path:
        cfg = _read_json(Path(model_path) / "config.json")
        if cfg:
            out["config_sha256"] = sha256_file(Path(model_path) / "config.json")
            # 只挑对复现真正有用的字段，不要把整个 config 塞进来
            for k in ("model_type", "hidden_size", "num_hidden_layers",
                      "max_seq_length", "max_position_embeddings", "architectures"):
                if k in cfg:
                    out["config"][k] = cfg[k]
        st_cfg = _read_json(Path(model_path) / "config_sentence_transformers.json")
        if st_cfg:
            out["config"]["sentence_transformers_version"] = st_cfg.get("__version__")
        sb_cfg = _read_json(Path(model_path) / "sentence_bert_config.json")
        if sb_cfg:
            out["config"]["max_seq_length"] = sb_cfg.get("max_seq_length")
    return out


def llm_identity(model_name: str | None = None, model_path: str | Path | None = None,
                 load_in_4bit: bool | None = None, device: str | None = None) -> dict:
    """LLM 身份：模型、量化配置、torch/transformers 版本、GPU。

    量化方式必须记：4bit NF4 + double quant 和 fp16 跑出来的答案分布不一样，
    不记等于把"结果不可比"这件事藏起来了。
    """
    out = {
        "model_name": model_name,
        "resolved_path": str(model_path) if model_path else None,
        "load_in_4bit": load_in_4bit,
        "quantization": None,
        "revision": None,
        "revision_note": "本地目录加载，无 git revision；身份由解析路径 + config.json 哈希确证",
        "config_sha256": None,
        "config": {},
        "torch": None,
        "transformers": None,
        "device": device,
        "gpu": None,
    }
    if load_in_4bit:
        out["quantization"] = {
            "method": "bitsandbytes",
            "bits": 4,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16",
        }
    if model_path:
        cfg = _read_json(Path(model_path) / "config.json")
        if cfg:
            out["config_sha256"] = sha256_file(Path(model_path) / "config.json")
            for k in ("model_type", "architectures", "hidden_size",
                      "num_hidden_layers", "num_attention_heads", "vocab_size",
                      "torch_dtype", "quantization_config"):
                if k in cfg:
                    out["config"][k] = cfg[k]
    try:
        import torch
        out["torch"] = torch.__version__
        if torch.cuda.is_available():
            out["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import transformers
        out["transformers"] = transformers.__version__
    except Exception:
        pass
    return out


def env_identity() -> dict:
    return {"python": sys.version.split()[0], "platform": platform.platform()}


# ---------------------------------------------------------------------------
# 组装与解析
# ---------------------------------------------------------------------------

def build_run_metadata(*, kind: str, corpus=None, goldset=None, index=None,
                       embedding=None, llm=None, code=None,
                       extra: dict | None = None) -> dict:
    """把各项身份组装成本次运行的完整身份块。"""
    meta: dict = {
        "schema": "docmind.run_metadata/2",
        "kind": kind,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "env": env_identity(),
        "corpus": corpus if corpus is not None else corpus_identity(),
        "goldset": goldset if goldset is not None else goldset_identity(),
        "index": index if index is not None else index_identity(),
        "embedding": embedding,
        "llm": llm,
        "code_sha256": code if code is not None else code_identity(),
    }
    if extra:
        meta.update(extra)
    return meta


def code_drift_report(before: dict, after: dict) -> dict:
    """对比实验**开始时**与**结束时**的源码哈希，判断本次运行期间代码是否漂移。

    为什么要在开头冻结一份：
    一次实验常常要跑十几分钟（端到端矩阵更久）。如果中途有人改了代码，
    结果文件里记的哈希是**结束时**采的那一份，于是会声称
    "整轮实验都是用这份代码跑的" —— 而前半程用的其实是另一份。

    开头冻结、结尾再采一次，就能把"跑的过程中代码动过"这件事显式暴露出来，
    而不是让它悄悄藏在一堆哈希后面。
    """
    keys = sorted(set(before) | set(after))
    changed, missing = [], []
    for k in keys:
        b, a = before.get(k), after.get(k)
        if b != a:
            changed.append({"file": k, "start": b, "end": a})
        if a is None:
            missing.append(k)
    return {
        "checked_at_start": bool(before),
        "n_files": len(keys),
        "n_changed": len(changed),
        "changed": changed,
        "missing_at_end": missing,
        "drifted": bool(changed),
        "note": ("运行期间源码未发生漂移，结果可用开头采集的哈希代表整轮"
                 if not changed else
                 "运行期间源码发生过改动：本结果文件里的 code_sha256 是**结束时**采集的，"
                 "不能代表整轮实验代码，请以 code_snapshot 归档字节为准"),
    }


def evidence_record(e: dict) -> dict:
    """把一条证据序列化成落盘格式：**完整正文** + 可核对完整性的字段。

    【第三轮修掉的截断】

    旧代码写的是 `e["text"][:400]`。实测 42 条证据被截断，
    于是"拿交付物复核当时喂给模型的到底是什么"根本做不到 ——
    而那正是 smoke 唯一的价值。

    现在一律存全文，并额外给出 `text_chars` 与 `text_sha256`：
    全文很长不想打印时，也能用这两个字段证明"没有被截断/没有被改过"。
    `truncated` 字段显式写 False，让读文件的人不必去猜。
    """
    text = e.get("text", "") or ""
    record = {
        "id": e.get("id"),
        "source": e.get("source"),
        "score": round(float(e.get("score", 0.0)), 6),
        "n_terms": e.get("n_terms"),
        "queries": list(e.get("queries", [])),
        "bm25_rank": e.get("bm25_rank"),
        "vector_rank": e.get("vector_rank"),
        "text": text,
        "text_chars": len(text),
        "text_sha256": sha256_text(text),
        "truncated": False,
    }
    for key in ('source_reviews','model_text'):
        if key in e:
            record[key] = e[key]
    return record


def _norm_dtype(dtype) -> str | None:
    """把 torch 的 dtype 写法归一化成 fp16 / fp32 / bf16。

    不归一化的话，"torch.float16" 和调用方判断用的 "fp16" 永远对不上，
    兼容字段会被静默记成相反的值。
    """
    if dtype is None:
        return None
    s = str(dtype).lower().replace("torch.", "")
    return {
        "float16": "fp16", "half": "fp16", "fp16": "fp16",
        "float32": "fp32", "float": "fp32", "fp32": "fp32",
        "bfloat16": "bf16", "bf16": "bf16",
        "float64": "fp64", "double": "fp64",
    }.get(s, s)


def probe_embedding_state(embedder) -> dict:
    """从**活的 embedder 对象**里读出此刻真实的 device / dtype。

    为什么不直接用调用方传进来的 `device` / `fp16`？
    因为那两个值描述的是"我以为它在哪"，而不是"它此刻在哪"。
    上一轮就是这么记错的：脚本里写着 cuda/fp16，可 `embedder.to("cpu")`
    已经把模型搬到 CPU 并 `.float()` 成 fp32 了，真正做 encode 的是后者。
    只有真的去问模型对象（而不是问我们自己的假设）才不会记错。

    对 fake / 占位对象也安全：拿不到就如实返回 None，不猜。
    """
    dev = getattr(embedder, "device", None)
    dtype = None
    model = getattr(embedder, "model", None)
    if model is not None:
        try:
            p = next(iter(model.parameters()))       # 问参数本身最准
            dtype = str(p.dtype).replace("torch.", "")
            if dev is None:
                dev = str(p.device)
        except Exception:
            d = getattr(model, "dtype", None)
            dtype = str(d).replace("torch.", "") if d is not None else None
    if dev is not None:
        # torch 会写成 "cuda:0"；统一记 "cuda"，免得同一条件被记成两种
        dev = str(dev).split(":")[0]
    # 在源头就归一化：否则后面拼 state_note 会写出 "cuda/float16"
    # 这种与 runtime_dtype("fp16") 不一致的写法，读的人会当成两个不同的值。
    return {"device": dev, "dtype": _norm_dtype(dtype)}


class EmbeddingIdentityRecorder:
    """分两阶段采集 embedding 身份：加载时一次，真正 encode 之前再一次。

    典型用法（端到端链路）::

        rec = EmbeddingIdentityRecorder(preset="m3")
        rec.capture_load(embedder)      # 必须在 embedder.to("cpu") 之前
        embedder.to("cpu")              # 给 LLM 腾显存
        llm = LocalLLM(...)
        ident = rec.build(embedder)     # 在真正检索之前

    `build()` 给出的 `runtime_*` 才是描述本次实验条件的字段，`load_*` 只是参考。
    两者不一致时会在 `state_note` 里写清楚差异，而不是静默认成其中一个。
    """

    def __init__(self, preset: str | None = None):
        self.preset = preset
        self._load: dict | None = None

    def capture_load(self, embedder) -> None:
        """在设备迁移之前调用，记录加载态。"""
        self._load = probe_embedding_state(embedder)

    def build(self, embedder) -> dict:
        """在真正做 encode 之前调用，结果里的 runtime_* 即为实验条件。"""
        runtime = probe_embedding_state(embedder)
        load = self._load or runtime
        ident = embedding_identity(
            preset=self.preset,
            model_path=getattr(embedder, "model_name", None),
            dim=getattr(embedder, "dim", None),
            runtime_device=runtime["device"],
            runtime_dtype=runtime["dtype"],
            load_device=load["device"],
            load_dtype=load["dtype"],
        )
        if load["device"] != runtime["device"] or load["dtype"] != runtime["dtype"]:
            ident["state_note"] = (
                f"加载时为 {load['device']}/{load['dtype']}，"
                f"实际 encode 发生在迁移之后的 {runtime['device']}/{runtime['dtype']}"
                f"（embedder.to() 会把 fp16 参数转回 fp32）；"
                f"实验条件以 runtime_* 为准")
        ident["load_captured"] = self._load is not None
        return ident


def default_of(func, name: str):
    """从函数签名里读出某个参数的默认值，而不是在记录处再手抄一个数字。

    为什么要从签名读：默认值真正生效的地方在被调用方
    （例如 `HybridRetriever.search(candidate_k=50)`）。如果在记录元数据的地方
    再手抄一遍 50，两边迟早漂移 —— 记出来的"有效参数"就不再是真正生效的参数。
    """
    import inspect
    try:
        return inspect.signature(func).parameters[name].default
    except Exception:
        return None


def retrieval_effective(mode: str, top_k: int, rrf_k: int,
                        candidate_k: int) -> dict:
    """检索层的**有效参数**，逐个显式给出，不靠 getattr 现取。

    【第四轮修正】上一版写的是 `vector_top_k: getattr(vector, "top_k", None)`。
    `VectorRetriever` 根本没有 `top_k` 属性，这个字段恒等于 null。
    读结果的人会以为"这次实验没限制候选池"，而实际用的是 candidate_k=50。

    记一个取不到的字段比不记更糟：它看起来像事实，其实是空气。

    另外这里特别区分两个容易混的量：
        candidate_k : 每一路召回各自先取多少条，进入融合前的候选池深度
        top_k       : 融合之后最终返回多少条
    把它们写反会让"召回上限"和"返回条数"两个结论全错。
    """
    return {
        "mode": mode,
        "top_k": top_k,
        "rrf_k": rrf_k,
        "candidate_k": candidate_k,
        "note": ("candidate_k = 每路召回各自取多少条进融合（候选池深度）；"
                 "top_k = 融合后返回多少条。两者含义不同。"),
    }


def config_effective(cfg) -> dict:
    """把一个 dataclass 配置展开成**含默认值**的完整字典。

    为什么不能只存被覆盖的 kwargs：JSON 里如果只有
    `{"enable_plan": false, "max_rounds": 0}`，读结果的人无法知道
    `max_evidence`、`token_budget` 这些当时是什么值 ——
    而这些默认值一改，实验就不可比了。
    """
    import dataclasses
    if dataclasses.is_dataclass(cfg):
        return {f: getattr(cfg, f) for f in cfg.__dataclass_fields__}
    return dict(getattr(cfg, "__dict__", {}))


def read_archived_snapshot(snapshot_dir: str | Path,
                           root: str | Path = ROOT) -> dict:
    """读一份源码快照：返回 {相对路径: sha256}，字节取自快照目录。

    为什么需要它：源码在实验之后还会改，`resolve_snapshot` 拿当前源码去对
    必然 mismatch。要证明"当时确实是那份代码"，只能去看**归档的字节**。
    """
    snap = Path(snapshot_dir)
    if not snap.is_absolute():
        snap = Path(root) / snap
    manifest_path = snap / "manifest.json"
    if not manifest_path.exists():
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    out = {}
    for rel, info in (manifest.get("files") or {}).items():
        fname = info.get("snapshot_file")
        if not fname:
            continue
        f = snap / fname
        out[rel] = sha256_file(f) if f.exists() else None
    return out


def resolve_snapshot(meta: dict, root: str | Path = ROOT) -> dict:
    """拿一份 metadata 去和当前磁盘上的文件对账。

    返回每一项 `match / mismatch / missing` —— 这是"记录可解析"的最低验证：
    只写哈希不提供核对手段，等于让人相信我们没写错。

    【第四轮修正】
    1) 源码部分同时给出两种结果：
         code.current     : 与**当前**源码比对（源码改过就会 mismatch，正常）
         code.archived    : 与 metadata 里 `code_snapshot` 指向的**归档字节**比对
       `code_all_match` 只在**确实有代码清单可比**时才为 True；
       空清单不允许被 `all([])` 判成 True —— 那会让"没检查"伪装成"检查通过"。
    2) 除源码外的语料/索引仍与当前文件比对（它们本来就是冻结的）。
    """
    root = Path(root)
    report: dict = {}

    def check(label: str, path_value, recorded):
        if not path_value or recorded is None:
            report[label] = "missing"
            return
        p = root / path_value if not Path(path_value).is_absolute() else Path(path_value)
        if not p.exists():
            report[label] = "missing"
            return
        report[label] = "match" if sha256_file(p) == recorded else "mismatch"

    corpus = meta.get("corpus") or {}
    check("corpus", corpus.get("path"), corpus.get("sha256"))
    goldset = meta.get("goldset") or {}
    check("goldset", goldset.get("path"), goldset.get("sha256"))

    idx = meta.get("index") or {}
    bm = idx.get("bm25") or {}
    check("index.bm25", bm.get("path"), bm.get("sha256"))
    vec = idx.get("vectors") or {}
    check("index.vectors", vec.get("path"), vec.get("sha256"))

    code = meta.get("code_sha256") or {}
    current = {}
    for f, h in code.items():
        p = root / f
        if not p.exists():
            current[f] = "missing"
        else:
            current[f] = "match" if sha256_file(p) == h else "mismatch"

    snap_value = meta.get("code_snapshot")
    archived_bytes = read_archived_snapshot(snap_value, root) if snap_value else {}
    archived = {}
    for f, h in code.items():
        got = archived_bytes.get(f)
        if got is None:
            archived[f] = "missing"
        else:
            archived[f] = "match" if got == h else "mismatch"

    report["code"] = {
        "current": current,
        # 空清单必须显式说明"没有可比对象"，不能用 all([])==True 假装通过
        "n_files": len(code),
        "note": ("code 清单为空：本次记录未包含任何源码哈希，"
                 "因此无法判定源码是否一致（不是'一致'）") if not code else None,
    }
    report["code"]["current_all_match"] = bool(code) and all(
        v == "match" for v in current.values())
    if snap_value:
        report["code"]["archived"] = archived
        report["code"]["archived_dir"] = str(snap_value)
        report["code"]["archived_all_match"] = bool(code) and all(
            v == "match" for v in archived.values())
        # 兼容旧字段：整体一致 = 当前或归档任一为真
        report["code_all_match"] = (report["code"]["current_all_match"]
                                    or report["code"]["archived_all_match"])
    else:
        report["code"]["archived"] = None
        report["code"]["archived_note"] = "本次记录没有 code_snapshot，无法核对归档字节"
        # 无从核对 -> 记为 None，而不是 False（False 会被读成"不一致"），
        # 更不能被 all([]) 当成 True。
        report["code"]["archived_all_match"] = None
        report["code_all_match"] = report["code"]["current_all_match"]

    # 兼容旧读取方：旧字段语义 = 当前源码是否一致
    report["code_current_all_match"] = report["code"]["current_all_match"]
    return report


def snapshot_code(out_dir: str | Path, files: list[str] | None = None,
                  root: str | Path = ROOT,
                  overwrite: bool = False) -> Path:
    """把关键源码**复制一份**到 `out_dir`，并写出 manifest。

    为什么需要真的复制：
    审核方指出"两份检索结果记录的 react.py/metrics.py 哈希与最终交付源码不同"。
    代码在跑完实验之后还会继续改（改注释也算），哈希自然对不上 ——
    这时唯一诚实的做法是把**当时那份**留下来，而不是回头去改旧 JSON 里的哈希。

    【第四轮修正：默认拒绝覆写】
    上一版对同一个 tag 重跑会把旧快照逐文件覆盖 —— 等于把"上一次实验的代码"
    悄悄替换掉。现在：
        - 目录里已有 manifest.json 时**默认报错**（除非 overwrite=True）；
        - 想重跑就换一个新的 run id / tag，历史快照永远留着。
    这样"哪个哈希对应哪次运行"才真正可查。
    """
    root = Path(root)
    out_dir = Path(out_dir)
    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"快照目录已存在且含 manifest.json: {out_dir}\n"
            f"拒绝覆写历史快照（旧快照是那次实验的唯一代码证据）。\n"
            f"请换一个新的 run id/tag，或显式传 overwrite=True 并说明理由。")
    out_dir.mkdir(parents=True, exist_ok=True)
    files = files or CODE_FILES
    manifest = {}
    for rel in files:
        src = root / rel
        if not src.exists():
            manifest[rel] = {"sha256": None, "status": "missing"}
            continue
        dst = out_dir / rel.replace("/", "__")
        dst.write_bytes(src.read_bytes())
        manifest[rel] = {"sha256": sha256_file(src), "snapshot_file": dst.name}
    manifest_path.write_text(json.dumps(
        {"created": datetime.now().isoformat(timespec="seconds"),
         "root": str(root), "files": manifest},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return out_dir


def unique_snapshot_dir(base: str | Path, run_id: str | None = None,
                        root: str | Path = ROOT) -> Path:
    """给本次运行挑一个**未被占用**的快照目录。

    规则：优先用显式 run_id；没给就用时间戳；万一撞名再加 -2、-3 后缀。
    绝不返回一个已有 manifest 的目录。

    顺带把目录建出来：这相当于"占位"，后续即使同一脚本并发跑第二次，
    也只会拿到 -2 目录，不会写进同一处。
    """
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    stem = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    cand = base / stem
    n = 2
    while (cand / "manifest.json").exists():
        cand = base / f"{stem}-{n}"
        n += 1
    cand.mkdir(parents=True, exist_ok=True)
    return cand


def _rel(p: str | Path) -> str:
    p = Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except Exception:
        return str(p)
