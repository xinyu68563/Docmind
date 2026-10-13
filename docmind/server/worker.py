# DocMind —— Phase 4 服务化：常驻推理 worker（本地 HTTP + JSON）

"""
为什么需要一个常驻 worker，而不是每个请求起一次 CLI

    Phase 0~3 的用法都是"跑一次脚本、加载一次模型、退出"。
    服务化如果沿用这个模式，一次 HTTP 请求 = 加载 bge-m3 + 7B 4bit
    ≈ 1~3 分钟显存搬移，而真正的推理只要十几秒 —— 99% 的时间在加载。
    更糟的是并发两个请求就会加载两份 7B，8GB 显存直接 OOM。

    所以这里必须是"常驻进程 + 队列"：
        - 模型只加载一次（懒加载，第一个任务触发；也可 --load-at-startup）
        - 一个执行线程串行消费，天然杜绝重复 GPU 实例
        - 有界队列：满了立刻 429，而不是无上限堆积把内存吃光

【协议选型：为什么是本地 HTTP/JSON，不是 stdin/stdout 的 JSONL】

    任务书给了二选一，这里选 HTTP，理由三条（都要能在面试里讲出来）：

    1. Java 侧可以直接用 JDK 内置的 java.net.http.HttpClient（JDK 11+），
       不引入任何第三方依赖；JSONL 则要自己写进程管理 + 管道分帧，
       还要处理 stdout 与模型日志混在一起的问题（Windows 上尤其麻烦）。
    2. HTTP 可以用 curl 直接验证，Codex 做独立验收时不必依赖我们写的客户端。
       这是"可独立验证"的硬要求 —— JSONL 协议只能靠我们自己的解析器验。
    3. 有界队列的背压可以用 HTTP 状态码自然表达（429 / 503）；
       管道协议里"队列满"只能靠自定义错误码，双方都要额外约定。

    代价是本地回环多一层序列化开销 —— 相对模型推理的几十秒，可以忽略。

【本模块刻意不做什么】

    - 不做鉴权/账号：默认只绑 127.0.0.1，公网部署是另一件事
    - 不做持久化：任务状态在内存里，重启即失效（下面 TERMINAL 注释有说明）
    - 不做上传入库：语料固定为现有 10 篇 PDF 的索引
    - 不读 gold / 诊断产物：答案只来自检索 + 模型
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse, unquote

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from docmind.eval.call_journal import JOURNAL_SCHEMA, CallJournal  # noqa: E402
from docmind.server.library import LibraryIntegrityError

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

PROTOCOL_VERSION = "docmind.worker/1"

QUEUED = "queued"
RUNNING = "running"
COMPLETED = "completed"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL = frozenset({COMPLETED, FAILED, CANCELLED})

# 输入校验边界。
# 为什么上限要显式存在：LLM 的 prompt 由问题拼接而来，问题无限长就等于
# 把显存和预算交给调用方控制。这里的上限同时被 Java 侧镜像了一份，
# 两边不一致时以更严格的一方为准（见 Java 的 SubmitRequest 校验注释）。
MAX_QUESTION_CHARS = 2000
MIN_QUESTION_CHARS = 1
TOKEN_BUDGET_MIN = 256
TOKEN_BUDGET_MAX = 8000
MAX_ROUNDS_MIN = 0
MAX_ROUNDS_MAX = 5
TOP_K_MIN = 1
TOP_K_MAX = 20

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_QUEUE_MAX = 16
DEFAULT_JOURNAL_DIR = ROOT / "logs" / "service" / "worker"

# 证据正文在服务响应里的节选上限。
# 为什么不返回全文：chunk 最长上千字符，6 条一起返回会让 HTTP 响应变成 MB 级，
# 而前端真正要展示的是"命中了哪一段"。完整正文仍可通过 chunk_id + journal 回溯。
EVIDENCE_EXCERPT_CHARS = 600


# ---------------------------------------------------------------------------
# 输入校验
# ---------------------------------------------------------------------------

class ValidationError(ValueError):
    """请求体不合法。对应 HTTP 400，与"服务不可用"严格分开。"""

    def __init__(self, message: str, *, field: str = "question"):
        super().__init__(message)
        self.message = message
        self.field = field

    def to_dict(self) -> dict:
        return {"error": "invalid_request", "field": self.field,
                "message": self.message}


def validate_submit(payload: Any) -> dict:
    """校验并规范化提交请求，返回可直接交给引擎的参数。

    为什么在这里校验而不是只靠 Java 侧：
        worker 是一个独立可达的 HTTP 服务，Java 只是它的一个客户端。
        把校验只放在 Java 里，等于默认"所有人都会经过 Java"——
        那是一厢情愿。边界服务必须自己守住自己的边界。
    """
    if not isinstance(payload, dict):
        raise ValidationError("请求体必须是 JSON 对象", field="body")

    q = payload.get("question")
    if not isinstance(q, str):
        raise ValidationError("question 必须是字符串", field="question")
    q = q.strip()
    if len(q) < MIN_QUESTION_CHARS:
        raise ValidationError("question 不能为空", field="question")
    if len(q) > MAX_QUESTION_CHARS:
        raise ValidationError(
            f"question 过长：{len(q)} > {MAX_QUESTION_CHARS} 字符", field="question")

    out: dict = {"question": q}
    if payload.get('document_id') is not None:
        document_id=payload['document_id']
        if not isinstance(document_id,str) or not document_id or len(document_id)>100:
            raise ValidationError('document_id 必须是有效文档编号',field='document_id')
        out['document_id']=document_id

    # 数值参数：显式拒绝 bool（Python 里 True 是 int 的子类，
    # 不挡住的话 {"max_rounds": true} 会被当成 1 静默接受）。
    for key, lo, hi, default in (
        ("token_budget", TOKEN_BUDGET_MIN, TOKEN_BUDGET_MAX, None),
        ("max_rounds", MAX_ROUNDS_MIN, MAX_ROUNDS_MAX, None),
        ("top_k", TOP_K_MIN, TOP_K_MAX, None),
    ):
        if key not in payload or payload[key] is None:
            continue
        v = payload[key]
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValidationError(f"{key} 必须是整数", field=key)
        if not (lo <= v <= hi):
            raise ValidationError(f"{key} 必须在 [{lo}, {hi}] 之间，收到 {v}", field=key)
        out[key] = v
    if payload.get('document_ids') is not None:
        ids=payload['document_ids']
        if payload.get('document_id') is not None or not isinstance(ids,list) or not 1<=len(ids)<=10 or any(not isinstance(i,str) or not i or len(i)>100 for i in ids) or len(set(ids))!=len(ids):
            raise ValidationError('请选择 1 到 10 篇不同论文，不能同时使用单篇和多篇参数',field='document_ids')
        out['document_ids']=ids
    return out


# ---------------------------------------------------------------------------
# 任务记录
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


@dataclass
class TaskRecord:
    """一个任务的内存状态。

    events 是**真实状态变化**的序列：由执行线程在阶段切换时追加。
    Java 侧不自己编造阶段事件，只把这里的新增事件转发成 SSE ——
    这样"SSE 说的是真的"这件事有据可查。
    """
    task_id: str
    question: str
    params: dict
    status: str = QUEUED
    stage: str = "queued"
    created_at: str = field(default_factory=_now)
    started_at: str | None = None
    finished_at: str | None = None
    cancel_requested: bool = False
    cancel_mode: str | None = None      # queued / cooperative
    result: dict | None = None
    error: dict | None = None
    events: list[dict] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _on_change: Callable | None = field(default=None, repr=False)

    def add_event(self, type_: str, **payload) -> dict:
        with self._lock:
            ev = {"seq": len(self.events) + 1, "ts": _now(), "type": type_, **payload}
            self.events.append(ev)
            if self._on_change:
                self._on_change(self)
            return ev

    def set_status(self, status: str, stage: str | None = None) -> dict:
        """改状态并追加一条事件 —— 状态和事件**必须**一起变。

        分两步写迟早会漏：状态改了忘了发事件，SSE 客户端就永久悬挂。
        """
        with self._lock:
            self.status = status
            if stage is not None:
                self.stage = stage
            if status == RUNNING and self.started_at is None:
                self.started_at = _now()
            if status in TERMINAL and self.finished_at is None:
                self.finished_at = _now()
            ev = {"seq": len(self.events) + 1, "ts": _now(), "type": f"status.{status}",
                  "status": status, "stage": self.stage}
            self.events.append(ev)
            if self._on_change:
                self._on_change(self)
            return ev

    def to_dict(self, *, include_result: bool = True) -> dict:
        with self._lock:
            d = {
                "task_id": self.task_id,
                "status": self.status,
                "stage": self.stage,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "cancel_requested": self.cancel_requested,
                "cancel_mode": self.cancel_mode,
                "event_count": len(self.events),
                "question_chars": len(self.question),
                # 完整事件序列交给调用方。
                #
                # 为什么不让 Java 自己"猜阶段"：SSE 一旦开始编造阶段名，
                # 客户端看到的东西就和真实执行脱钩了 —— 那比没有 SSE 更糟
                # （用户会以为它在跑某个阶段，实际卡在另一个）。
                # 这里给出带 seq 的真实事件，Java 只做"转发未发过的部分"。
                "events": list(self.events),
            }
            if include_result:
                d["question"] = self.question
                d["params"] = self.params
                if self.result is not None:
                    d["result"] = self.result
                if self.error is not None:
                    d["error"] = self.error
            return d


# ---------------------------------------------------------------------------
# 执行引擎
# ---------------------------------------------------------------------------

class TaskHooks:
    """引擎回写给 worker 的两个回调。

    should_stop : 协作式取消探测，在节点之间被调用
    stage       : 阶段变化上报（会变成一条事件，进而变成一条 SSE）
    """

    def __init__(self, should_stop: Callable[[], bool],
                 stage: Callable[[str, dict | None], None]):
        self.should_stop = should_stop
        self.stage = stage


class Engine:
    """引擎接口。"""

    backend = "unknown"

    def describe(self) -> dict:
        raise NotImplementedError

    def ensure_ready(self) -> None:
        raise NotImplementedError

    def run(self, params: dict, hooks: TaskHooks, journal: CallJournal) -> dict:
        raise NotImplementedError


class StubEngine(Engine):
    """**显式**选择的假后端，只用于打通链路与故障分支测试。

    为什么允许它存在：
        端到端链路（HTTP → 队列 → 执行 → SSE）的正确性和 7B 模型无关，
        每次验证都要加载模型既慢又占显存。
        但假后端必须有三个约束，否则就变成"能造假答案的后门"：
            1. 必须显式 --backend stub 才启用，默认是 real
            2. health 与每个 result 都带 backend="stub" 标记
            3. 它返回的答案里写明了"这是桩"，不冒充模型输出
    """

    backend = "stub"

    def __init__(self, *, stage_delay_ms: int = 120, fail: bool = False):
        self.stage_delay_ms = stage_delay_ms
        self.fail = fail
        # 记录"真正被执行了几次"。取消测试靠它证明排队任务一次都没跑 ——
        # 只看状态是 cancelled 是不够的，状态可以被设置，执行次数不会说谎。
        self.runs = 0

    def describe(self) -> dict:
        return {"backend": "stub", "models": [], "stub": True,
                "stage_delay_ms": self.stage_delay_ms, "fail_mode": self.fail}

    def ensure_ready(self) -> None:
        return None

    def run(self, params: dict, hooks: TaskHooks, journal: CallJournal) -> dict:
        self.runs += 1
        journal.note("stub_engine", note="桩后端：未加载任何模型，未产生任何 LLM 调用")
        for name in ("plan", "retrieve", "reflect", "answer", "verify"):
            if hooks.should_stop():
                return _cancelled_payload(params, backend="stub", stage=name)
            hooks.stage(name, {"stub": True})
            # 分段 sleep 而不是一次 sleep 到底：这样取消才能在"阶段之间"生效，
            # 一次性睡 600ms 会让协作式取消根本没机会被探测到。
            deadline = time.time() + self.stage_delay_ms / 1000.0
            while time.time() < deadline:
                if hooks.should_stop():
                    return _cancelled_payload(params, backend="stub", stage=name)
                time.sleep(min(0.02, max(0.0, deadline - time.time())))

        if self.fail:
            raise RuntimeError("stub 故障注入：模拟模型/检索异常")

        q = params["question"]
        return {
            "answer": ("[STUB 后端 · 非模型输出] 这是一个显式选择的桩后端响应，"
                       f"用于验证链路而非回答问题。收到问题 {len(q)} 字符。"),
            "passed_verification": False,
            "answer_source": "stub",
            "evidence": [],
            "verification_status": "not_run",
            "llm_verification_status": "not_run",
            "citation_validity": "none",
            "verification_fail_reasons": [],
            "issues": [],
            "citation_issues": [],
            "stop_reason": "stub",
            "answer_status": "stub",
            "plan_status": "stub",
            "rounds_used": 0,
            "usage": {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0},
            "budget": {"limit": params.get("token_budget") or 0,
                       "output_tokens": 0, "prompt_tokens": 0, "remaining": None},
            "elapsed_s": 0.0,
            "trace": [],
            "warnings": ["stub backend: 本结果不是模型生成的答案，不可用于任何结论",
                         "校验未执行（not_run），passed_verification 为 False，不等于通过"],
        }


class StageJournal:
    def __init__(self, journal, hooks):
        self.journal, self.hooks = journal, hooks
    def __getattr__(self, name):
        return getattr(self.journal, name)
    def call_start(self, **payload):
        self.hooks.stage(payload["node"], {})
        return self.journal.call_start(**payload)


class RealEngine(Engine):
    """真实后端：复用 Phase 0~2 的检索 + Agent，模型只加载一次。"""

    backend = "real"

    def __init__(self, *, model: str = "Qwen2.5-7B-Instruct",
                 load_in_4bit: bool = True, chunk_path=None, bm25_path=None,
                 vec_path=None):
        self.model = model
        self.load_in_4bit = load_in_4bit
        self.chunk_path = Path(chunk_path or ROOT / "data" / "processed" / "chunks.json")
        self.bm25_path = Path(bm25_path or ROOT / "data" / "processed" / "index" / "bm25.pkl")
        self.vec_path = Path(vec_path or ROOT / "data" / "processed" / "index" / "vectors.npz")
        self._bundle: dict | None = None
        self._lock = threading.Lock()
        self._chunk_meta: dict[str, dict] = {}
        self.library = None

    def describe(self) -> dict:
        b = self._bundle
        return {
            "backend": "real",
            "stub": False,
            "loaded": b is not None,
            "model": self.model,
            "load_in_4bit": self.load_in_4bit,
            "index": {
                "chunks": str(self.chunk_path),
                "bm25": str(self.bm25_path),
                "vectors": str(self.vec_path),
            },
        }

    def ensure_ready(self) -> None:
        self._load()

    def _load(self) -> dict:
        """加载索引 + 模型。用锁保证并发提交只触发一次加载。"""
        with self._lock:
            if self._bundle is not None:
                return self._bundle
            from docmind.ingest.chunker import Chunk
            from docmind.retrieve.bm25 import BM25Retriever
            from docmind.retrieve.fusion import HybridRetriever
            from docmind.retrieve.vector import VectorRetriever

            paths = (self.chunk_path, self.bm25_path, self.vec_path)
            present = [p.exists() for p in paths]
            if any(present) and not all(present):
                raise ValueError('内置索引不完整，请重新建立索引或使用干净的安装目录')
            chunks = ([Chunk.from_dict(d)
                       for d in json.loads(self.chunk_path.read_text(encoding="utf-8"))]
                      if all(present) else [])
            # chunk_id -> 元数据。物理页/页码只能从这里取，
            # 绝不从 chunk_id 的字符串里"猜"（第十三轮验收点：不凭 chunkID 猜页）。
            self._chunk_meta = {
                c.chunk_id: {"file_name": c.file_name, "doc_id": c.doc_id,
                             "title": c.title, "page_start": c.page_start,
                             "page_end": c.page_end}
                for c in chunks
            }
            bm25 = BM25Retriever.load(self.bm25_path, chunks) if chunks else None
            if chunks:
                vector = VectorRetriever.load(self.vec_path)
            else:
                import numpy as np
                vector = VectorRetriever(np.empty((0, 1024), dtype=np.float32), [])
            from .embedding_pool import EmbeddingPool
            embedder = self.library.embedding if self.library else EmbeddingPool()
            # 【Phase 2 踩坑：8GB 显存装不下双模型】
            # embedding 必须在 LLM 加载前搬到 CPU，否则 bitsandbytes 会把
            # LLM 的部分层 offload 到 CPU，速度掉 10 倍且不报错。
            embedder.to("cpu")
            from docmind.generate.llm import LocalLLM
            llm = LocalLLM(model_name=self.model, load_in_4bit=self.load_in_4bit)
            self._bundle = {
                "chunks": chunks, "bm25": bm25, "vector": vector,
                "embedder": embedder, "llm": llm,
                "hybrid": HybridRetriever(bm25, vector, embedder, rrf_k=60) if chunks else None,
            }
            return self._bundle

    def run(self, params: dict, hooks: TaskHooks, journal: CallJournal) -> dict:
        from docmind.agent.react import (AgentConfig, DocMindAgent, STOP_CANCELLED)

        bundle = self._load()
        hybrid, chunk_meta = self._multi_retrieval(bundle, params.get('document_ids') or ([params['document_id']] if params.get('document_id') else []))
        for doc_id in params.get('document_ids') or ([params['document_id']] if params.get('document_id') else []):
            if self.library and self.library.record(doc_id):
                journal.write({'event':'source_index','document_id':doc_id,
                               'artifacts':self.library.record(doc_id).get('artifacts',{})})
        cfg_kwargs = {}
        if "token_budget" in params:
            cfg_kwargs["token_budget"] = params["token_budget"]
        if "max_rounds" in params:
            cfg_kwargs["max_rounds"] = params["max_rounds"]
        if "top_k" in params:
            cfg_kwargs["top_k_per_query"] = params["top_k"]
        cfg = AgentConfig(opaque_citations=True, source_quote_answers=True, **cfg_kwargs)

        agent = DocMindAgent(hybrid, bundle["llm"], cfg,
                             call_journal=StageJournal(journal, hooks),
                             stop_check=hooks.should_stop)

        hooks.stage("agent", {})
        result = agent.run(params["question"])

        journal_errors = list(journal.errors) + list(agent.journal_errors)
        cancelled = result.stop_reason == STOP_CANCELLED
        payload = serialize_answer(result, chunk_meta=chunk_meta,
                                   backend="real", journal_errors=journal_errors)
        payload['document_ids']=params.get('document_ids') or ([params['document_id']] if params.get('document_id') else [])
        payload["answer_source"] = _answer_source(result)
        if cancelled:
            payload["cancelled"] = True
        return payload

    def _multi_retrieval(self, bundle, ids):
        if not ids: return bundle['hybrid'],self._chunk_meta
        from docmind.retrieve.bm25 import BM25Retriever
        from docmind.retrieve.vector import VectorRetriever
        from docmind.retrieve.fusion import HybridRetriever
        import numpy as np
        chunks=[]; vectors=[]; chunk_ids=[]; meta={}
        for doc_id in ids:
            hybrid,one_meta=self._selected_retrieval(bundle,doc_id)
            chunks.extend(hybrid.bm25.blocks);vectors.append(hybrid.vector.vectors)
            chunk_ids.extend(hybrid.vector.chunk_ids);meta.update(one_meta)
        if len(chunk_ids)!=len(set(chunk_ids)):raise ValueError('Duplicate chunk IDs in scope')
        return HybridRetriever(BM25Retriever(chunks),VectorRetriever(np.concatenate(vectors),chunk_ids),bundle['embedder'],rrf_k=60),meta

    def _selected_retrieval(self, bundle, document_id):
        if not document_id:
            return bundle['hybrid'], self._chunk_meta
        from docmind.ingest.chunker import Chunk
        from docmind.retrieve.bm25 import BM25Retriever
        from docmind.retrieve.vector import VectorRetriever
        from docmind.retrieve.fusion import HybridRetriever
        uploaded = self.library.record(document_id) if self.library else None
        if uploaded:
            if uploaded['status'] != 'ready': raise ValueError('论文尚未完成索引')
            chunks=[Chunk.from_dict(c) for c in self.library.chunks(document_id)]
            directory=self.library.path(document_id)
            import hashlib
            for name,digest in uploaded['artifacts'].items():
                if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=digest:
                    raise ValueError('Uploaded source or index changed: '+name)
            bm25=BM25Retriever.load(directory/'bm25.pkl',chunks)
            vector=VectorRetriever.load(directory/'vectors.npz')
            if vector.chunk_ids != [c.chunk_id for c in chunks] or vector.vectors.shape != (len(chunks),1024):
                raise ValueError('Uploaded index does not match document chunks')
        else:
            chunks=[c for c in bundle['chunks'] if c.doc_id==document_id]
            if not chunks: raise ValueError('Document not found')
            bm25=BM25Retriever(chunks)
            selected_ids={c.chunk_id for c in chunks}
            positions=[i for i,cid in enumerate(bundle['vector'].chunk_ids) if cid in selected_ids]
            vector=VectorRetriever(bundle['vector'].vectors[positions], [bundle['vector'].chunk_ids[i] for i in positions])
        meta={c.chunk_id:{'file_name':c.file_name,'doc_id':c.doc_id,'title':c.title,
                         'page_start':c.page_start,'page_end':c.page_end} for c in chunks}
        return HybridRetriever(bm25,vector,bundle['embedder'],rrf_k=60),meta


def _cancelled_payload(params: dict, *, backend: str, stage: str) -> dict:
    return {
        "answer": "",
        "answer_source": "none",
        "passed_verification": False,
        "evidence": [],
        "verification_status": "not_run",
        "llm_verification_status": "not_run",
        "citation_validity": "none",
        "verification_fail_reasons": ["cancelled_by_client"],
        "issues": [],
        "citation_issues": [],
        "stop_reason": "cancelled_by_client",
        "answer_status": "cancelled",
        "plan_status": "cancelled",
        "rounds_used": 0,
        "usage": {"prompt_tokens": 0, "output_tokens": 0, "total_tokens": 0},
        "budget": {"limit": params.get("token_budget") or 0,
                   "output_tokens": 0, "prompt_tokens": 0, "remaining": None},
        "elapsed_s": 0.0,
        "trace": [],
        "cancelled": True,
        "cancelled_at_stage": stage,
        "warnings": [f"任务在 {stage} 阶段之前被取消，未产生模型答案"],
    }


def _answer_source(result) -> str:
    """区分"模型生成"与"程序照搬已核对原文"。

    为什么要区分：原文引用模式下答案是程序渲染的，字符数再长也和
    LLM 输出的 token 无关。混在一起会让"答案很长"被误读成"模型很能写"。
    """
    trace = getattr(result, "trace", None)
    steps = list(getattr(trace, "steps", []) or [])
    for step in steps:
        if step.get("node") == "answer" and step.get("mode") == "reviewed_source_quote":
            return "program_rendered_source_quote"
    return "model"


def serialize_answer(result, *, chunk_meta: dict, backend: str,
                     journal_errors: list[str]) -> dict:
    """把 AgentAnswer 变成服务响应里的结构化结果。

    刻意保留全部校验字段，且**不**把 failed/unknown/not_run 归一成"成功"：
    调用方要能一眼看出这次答案到底有没有通过校验。
    """
    evidence = []
    for e in result.evidence:
        meta = chunk_meta.get(e.get("id"), {})
        reviews = e.get("source_reviews") or []
        text = e.get("text") or ""
        # 传给模型的是 model_text（可能附带人工核对补充），
        # 返回给用户的是原文节选 —— 两者必须分开，否则"用户看到的"
        # 会被当成"模型读到的"，排查证据类问题时就对不上账。
        excerpt = text[:EVIDENCE_EXCERPT_CHARS]
        evidence.append({
            "chunk_id": e.get("id"),
            "file_name": meta.get("file_name") or e.get("file_name"),
            "doc_id": meta.get("doc_id"),
            "title": meta.get("title") or e.get("title"),
            "page_start": meta.get("page_start"),
            "page_end": meta.get("page_end"),
            "physical_pdf_page": meta.get("page_start"),
            "source": e.get("source"),
            "score": e.get("score"),
            "n_terms": e.get("n_terms"),
            "bm25_rank": e.get("bm25_rank"),
            "vector_rank": e.get("vector_rank"),
            "excerpt": excerpt,
            "excerpt_truncated": len(text) > EVIDENCE_EXCERPT_CHARS,
            "text_chars": len(text),
            # 物理页只能来自人工核对记录的独立字段；
            # 没有就写 None 并说明，绝不从 chunk_id 推断。
            "physical_page": (reviews[0].get("physical_page") if reviews else None),
            "source_reviews": [{"id": r.get("id"),
                                "physical_page": r.get("physical_page")}
                               for r in reviews],
            "human_reviewed_source": bool(reviews),
        })

    trace = getattr(result, "trace", None)
    return {
        "answer": result.answer,
        "evidence": evidence,
        "verification_status": result.verification_status,
        "llm_verification_status": result.llm_verification_status,
        "citation_validity": result.citation_validity,
        "verification_fail_reasons": list(result.verification_fail_reasons),
        "issues": list(result.issues),
        "citation_issues": list(result.citation_issues),
        "stop_reason": result.stop_reason,
        "answer_status": result.answer_status,
        "plan_status": result.plan_status,
        "rounds_used": result.rounds_used,
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "output_tokens": result.output_tokens,
            "total_tokens": result.total_tokens,
        },
        "budget": {
            "limit": getattr(result, "token_budget_limit", None),
            "output_tokens": result.output_tokens,
            "prompt_tokens": result.prompt_tokens,
            "remaining": None,
        },
        "elapsed_s": round(result.elapsed, 3),
        "trace": list(trace.steps) if trace is not None else [],
        "backend": backend,
        "passed_verification": bool(result.passed_verification),
        "journal_errors": journal_errors,
        "warnings": _status_warnings(result),
    }


def _status_warnings(result) -> list[str]:
    """把"这个状态意味着什么"讲清楚，避免调用方把 not_run 当成通过。"""
    w = []
    if result.verification_status == "unknown":
        w.append("校验未能读出结论，不等于通过")
    elif result.verification_status == "not_run":
        w.append("校验未执行，passed_verification 为 False")
    elif result.verification_status == "failed":
        w.append("校验未通过；passed 仅为模型自评，不是事实保证")
    elif result.verification_status == "passed":
        w.append("passed 仅表示模型自校验通过，不是事实正确性保证")
    return w


# ---------------------------------------------------------------------------
# Worker：有界队列 + 单执行线程
# ---------------------------------------------------------------------------

class Worker:
    """任务调度器。

    为什么只有一个执行线程：
        一个 7B 4bit 模型 + embedding 已经吃掉 8GB 显存的大半。
        两个任务并行 = 两份模型 = OOM；就算显存够，GPU 也是串行执行的，
        并行只会让两个任务都变慢。串行 + 有界队列是唯一诚实的选择。
    """

    def __init__(self, engine: Engine, *, queue_max: int = DEFAULT_QUEUE_MAX,
                 journal_dir: Path = DEFAULT_JOURNAL_DIR, state_dir: Path | None = None):
        self.engine = engine
        self.journal_dir = Path(journal_dir)
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        self._queue: "queue.Queue[str]" = queue.Queue(maxsize=queue_max)
        self._tasks: dict[str, TaskRecord] = {}
        self._lock = threading.Lock()
        self._fatal: str | None = None          # 非空 = worker 不可用
        self._executor: threading.Thread | None = None
        self._stop = threading.Event()
        from .task_store import TaskStore
        self.store = TaskStore(state_dir, engine.backend) if state_dir is not None else None
        self.recovery_errors = []
        self.library = None

    # ---------------- 生命周期 ----------------

    def start(self) -> None:
        if self._executor is not None and self._executor.is_alive():
            raise RuntimeError("Worker already started")
        if self.store:
            self.store.acquire()
            try:
                snapshots, self.recovery_errors = self.store.load()
                for data in snapshots:
                    rec = TaskRecord(**{k:v for k,v in data.items() if k in {
                        "task_id","question","params","status","stage","created_at",
                        "started_at","finished_at","cancel_requested","cancel_mode","result","error","events"}})
                    self._tasks[rec.task_id] = rec
                    rec._on_change = self._persist
                    if rec.status not in TERMINAL:
                        rec.error = {"error_type":"InterruptedByRestart", "message":"任务被重启中断，未自动重跑；请重新提交。"}
                        rec.result = {"answer":"", "backend":self.engine.backend,
                            "verification_status":"unknown", "passed_verification":False,
                            "evidence":[], "usage":{"prompt_tokens":None,"output_tokens":None,"total_tokens":None}}
                        rec.set_status(FAILED, stage="interrupted_by_restart")
                if self.recovery_errors:
                    self.mark_unavailable("Task recovery failed: " + "; ".join(self.recovery_errors))
            except BaseException:
                self.store.close()
                raise
        self._executor = threading.Thread(target=self._loop, name="docmind-worker",
                                          daemon=True)
        self._executor.start()

    def shutdown(self) -> None:
        self._stop.set()
        if self._executor is not None:
            self._executor.join(timeout=5)
        if self.store and (self._executor is None or not self._executor.is_alive()):
            self.store.close()

    def _persist(self, rec):
        try:
            self.store.save(rec.to_dict())
        except Exception as exc:
            self.mark_unavailable(f"Task state write failed: {type(exc).__name__}: {exc}")
            rec.error = {"error_type":"PersistenceError", "message":self._fatal}
            rec.status = FAILED
            rec.stage = "persistence_failed"
            rec.cancel_requested = True

    def mark_unavailable(self, reason: str) -> None:
        """把 worker 标记为不可用（**仅**用于故障注入测试与致命加载失败）。

        为什么做成显式方法而不是随便改个字段：
        503 是"服务明确不可用"，它必须有一个可追溯的理由字符串，
        而不是一个布尔量 —— 后者会让排查时只剩"它挂了"这一句话。
        """
        self._fatal = reason

    # ---------------- 查询 ----------------

    def health(self) -> dict:
        executor_alive = bool(self._executor is not None and self._executor.is_alive())
        with self._lock:
            pending = sum(1 for t in self._tasks.values() if t.status == QUEUED)
            running = sum(1 for t in self._tasks.values() if t.status == RUNNING)
        unavailable = self._fatal
        if not executor_alive:
            unavailable = unavailable or "executor thread not running"
        return {
            "service": "docmind-python-worker",
            "protocol": PROTOCOL_VERSION,
            "journal_schema": JOURNAL_SCHEMA,
            "status": "unavailable" if unavailable else ("ready" if self._ready() else "idle"),
            "ready": not unavailable,
            "unavailable_reason": unavailable,
            "backend": self.engine.backend,
            "stub": self.engine.backend == "stub",
            "engine": self.engine.describe(),
            "queue": {"size": self._queue.qsize(), "max": self._queue.maxsize,
                      "pending": pending, "running": running},
            "executor_alive": executor_alive,
            "journal_dir": str(self.journal_dir),
            "tasks_total": len(self._tasks),
            "library": {"enabled":self.library is not None,
                        "recovery_errors":self.library.recovery_errors if self.library else [],
                        "embedding":self.library.embedding.describe() if self.library else None},
            "persistence": {"enabled": self.store is not None,
                            "state_dir": str(self.store.directory) if self.store else None,
                            "recovery_errors": self.recovery_errors},
        }

    def _ready(self) -> bool:
        """真实后端在模型加载完成前不算 ready。"""
        if self.engine.backend == "stub":
            return True
        return bool(self.engine.describe().get("loaded"))

    def get_task(self, task_id: str) -> TaskRecord | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list_tasks(self) -> list[dict]:
        with self._lock:
            return [{"task_id":t.task_id,"question":t.question,"status":t.status,
                     "stage":t.stage,"created_at":t.created_at,"backend":self.engine.backend}
                    for t in sorted(self._tasks.values(), key=lambda x: x.created_at, reverse=True)][:100]

    # ---------------- 提交 / 取消 ----------------

    def submit(self, payload: Any) -> tuple[dict, int]:
        """提交任务。返回 (响应体, HTTP 状态码)。"""
        params = validate_submit(payload)          # 不合法 -> ValidationError -> 400
        if isinstance(self.engine, RealEngine) and not (params.get('document_ids') or params.get('document_id')):
            from .sources import corpus
            if not corpus():
                raise ValidationError('文档库为空，请先上传 PDF 并选择就绪的论文。', field='document_ids')
        for document_id in params.get('document_ids') or ([params['document_id']] if params.get('document_id') else []):
            from .sources import documents
            doc=next((d for d in documents() if d['doc_id']==document_id),None)
            if not doc or doc.get('status')!='ready':
                raise ValidationError('论文不存在或索引尚未完成，请稍后选择就绪论文。',field='document_id')
            if self.library and self.library.record(document_id):
                for artifact in ('source.pdf','chunks.json','bm25.pkl','vectors.npz'):
                    self.library.verified_path(document_id,artifact)

        if self._fatal:
            return {"error": "worker_unavailable",
                    "message": self._fatal}, 503
        if self._executor is None or not self._executor.is_alive():
            return {"error": "worker_unavailable",
                    "message": "执行线程已退出，worker 不再接受新任务"}, 503

        task_id = uuid.uuid4().hex
        rec = TaskRecord(task_id=task_id, question=params["question"], params=params)
        with self._lock:
            self._tasks[task_id] = rec
        rec._on_change = self._persist if self.store else None
        rec.add_event("created", status=QUEUED)
        if self._fatal:
            return {"error":"worker_unavailable","message":self._fatal},503
        try:
            # 有界队列：put_nowait 满了就抛。
            # 为什么不用阻塞 put：HTTP 请求线程会被无限挂住，
            # 客户端既得不到 202 也得不到 429，只能等超时 —— 那是最差的结果。
            self._queue.put_nowait(task_id)
        except queue.Full:
            with self._lock:
                self._tasks.pop(task_id, None)
            if self.store:
                self.store.discard(task_id)
            return {
                "error": "queue_full",
                "message": "任务队列已满，请稍后重试",
                "queue": {"size": self._queue.qsize(), "max": self._queue.maxsize},
            }, 429
        return {"task_id": task_id, "status": QUEUED,
                "protocol": PROTOCOL_VERSION,
                "queue": {"size": self._queue.qsize(), "max": self._queue.maxsize}}, 202

    def cancel(self, task_id: str) -> tuple[dict, int]:
        rec = self.get_task(task_id)
        if rec is None:
            return {"error": "task_not_found", "task_id": task_id}, 404
        with rec._lock:
            if rec.status in TERMINAL:
                return {"error": "task_already_terminal", "task_id": task_id,
                        "status": rec.status}, 409

            rec.cancel_requested = True
            if rec.status == QUEUED:
                # 排队任务：直接置终态，执行线程取到它会跳过 —— 一次都不会运行。
                rec.cancel_mode = "queued"
                rec.set_status(CANCELLED, stage="cancelled_while_queued")
                return {"task_id": task_id, "status": CANCELLED,
                        "cancel_mode": "queued",
                        "note": "任务仍在队列中，已直接取消，不会被执行"}, 200

            # 运行中的任务：只能**协作式**停止。
            # 必须如实说明能力边界 —— 一次 generate() 已经发出去就无法安全掐断，
            # 声称"已停止 GPU"而实际还在跑就是在撒谎。
            rec.cancel_mode = "cooperative"
            rec.add_event("cancel_requested", status=RUNNING, mode="cooperative")
            return {"task_id": task_id, "status": RUNNING,
                    "cancel_mode": "cooperative",
                    "note": ("已发出取消请求，将在当前阶段结束后停止；"
                             "正在进行的模型生成无法立即中断，"
                             "期间 GPU 仍在工作")}, 202

    # ---------------- 执行循环 ----------------

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                task_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._run_one(task_id)
            except BaseException as exc:  # noqa: BLE001 - 一个任务崩了不能带走整个 worker
                rec = self.get_task(task_id)
                if rec is not None and rec.status not in TERMINAL:
                    rec.error = {"error_type": type(exc).__name__,
                                 "message": str(exc)}
                    rec.set_status(FAILED, stage="failed")
            finally:
                self._queue.task_done()

    def _run_one(self, task_id: str) -> None:
        rec = self.get_task(task_id)
        if rec is None:
            return
        with rec._lock:
            if rec.cancel_requested or rec.status == CANCELLED:
                return
            # Claim atomically with cancellation; no cancelled -> running transition.
            rec.status = RUNNING
        rec.set_status(RUNNING, stage="starting")
        journal_path = self.journal_dir / task_id / "calls.jsonl"
        journal = CallJournal(journal_path)
        errors = []
        result = None
        terminal = FAILED
        def jcall(fn, *args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                errors.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
        def stage(name, extra=None):
            rec.stage = name
            rec.add_event("stage", status=RUNNING, stage=name, **(extra or {}))
        hooks = TaskHooks(lambda: rec.cancel_requested, stage)
        try:
            jcall(journal.run_start, protocol=PROTOCOL_VERSION, task_id=task_id,
                  backend=self.engine.backend, params=rec.params)
            jcall(journal.row_start, group="service", qid=task_id, row_seq=1)
            result = self.engine.run(rec.params, hooks, journal)
            terminal = CANCELLED if result.pop("cancelled", False) or rec.cancel_requested else COMPLETED
            jcall(journal.row_end, "cancelled" if terminal == CANCELLED else "ok")
            jcall(journal.run_end, "cancelled" if terminal == CANCELLED else "complete")
        except BaseException as exc:
            rec.error = {"error_type": type(exc).__name__, "message": str(exc)}
            # An exception may follow successful generation; do not invent zero usage.
            result = {"answer": "", "answer_source": "none", "evidence": [],
                      "passed_verification": False, "verification_status": "not_run",
                      "llm_verification_status": "not_run", "citation_validity": "none",
                      "verification_fail_reasons": ["engine_error"],
                      "stop_reason": "engine_error", "answer_status": "error",
                      "issues": [str(exc)], "usage": {"prompt_tokens": None,
                      "output_tokens": None, "total_tokens": None},
                      "warnings": ["任务执行失败；用量未知处不能当作零。"]}
            jcall(journal.row_end, "error", error=rec.error)
            jcall(journal.run_end, "error", error=rec.error)
        finally:
            jcall(journal.close)
        errors = list(journal.errors) + errors
        result.setdefault("backend", self.engine.backend)
        result["journal"] = {"path": str(journal_path), "schema": JOURNAL_SCHEMA,
                             "n_records": journal.n_records, "closed": journal.closed,
                             "errors": errors}
        result["journal_errors"] = errors
        if self._fatal:
            errors.append(self._fatal)
        if errors:
            terminal = FAILED
            rec.error = {"error_type": "JournalError", "message": "; ".join(errors)}
            result["passed_verification"] = False
            result["verification_status"] = "unknown"
            result.setdefault("verification_fail_reasons", []).append("journal_error")
        with rec._lock:
            if rec.cancel_requested and terminal == COMPLETED:
                terminal = CANCELLED
            rec.result = result
            rec.set_status(terminal, stage={COMPLETED:"done", CANCELLED:"cancelled", FAILED:"failed"}[terminal])


# ---------------------------------------------------------------------------
# HTTP 层
# ---------------------------------------------------------------------------

class WorkerHandler(BaseHTTPRequestHandler):
    """极薄的 HTTP 适配层。所有业务逻辑都在 Worker 里，这里只做协议翻译。"""

    worker: Worker
    server_version = "DocMindWorker/1"

    # 关掉默认日志到 stderr 的写法？不，保留反而便于排障，只是加上时间戳前缀。
    def log_message(self, fmt: str, *args) -> None:  # noqa: D102
        sys.stderr.write("%s %s - %s\n" % (self.log_date_time_string(),
                                           self.address_string(), fmt % args))

    # ---------------- 输出 ----------------

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> Any:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 32768:
            raise ValidationError("Request body too large", field="body")
        if n <= 0:
            return {}
        raw = self.rfile.read(n)
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValidationError(f"请求体不是合法 JSON：{exc}", field="body") from exc

    # ---------------- 路由 ----------------

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            if path == "/v1/documents":
                from .sources import documents
                self._send(200, {"documents": documents()})
            elif path.startswith('/v1/imports/'):
                record=self.worker.library.record(path.removeprefix('/v1/imports/')) if self.worker.library else None
                self._send(200 if record else 404,record or {'error':'import_not_found'})
            elif path.startswith("/v1/chunks/"):
                from .sources import chunk
                c = chunk(unquote(path.removeprefix("/v1/chunks/")))
                self._send(200 if c else 404, c or {"error": "chunk_not_found"})
            elif path.startswith('/v1/documents/') and '/pages/' in path:
                from .sources import render_page,document_path
                parts=path.split('/')
                if len(parts)!=6 or document_path(unquote(parts[3])) is None:
                    self._send(404,{'error':'document_not_found'});return
                try: body,pages=render_page(unquote(parts[3]),int(parts[5]))
                except (ValueError,RuntimeError) as exc:
                    self._send(400,{'error':'invalid_page','message':str(exc)});return
                self.send_response(200);self.send_header('Content-Type','image/png')
                self.send_header('Content-Length',str(len(body)));self.send_header('X-PDF-Pages',str(pages))
                self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(body)
            elif path.startswith("/v1/documents/") and path.endswith("/pdf"):
                from .sources import document_path
                p = document_path(unquote(path[len("/v1/documents/"):-4]))
                if p is None:
                    self._send(404, {"error": "document_not_found"})
                else:
                    body=p.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/pdf")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("X-Content-Type-Options", "nosniff")
                    self.end_headers()
                    self.wfile.write(body)
            elif path == "/v1/health":
                self._send(200, self.worker.health())
            elif path == "/v1/tasks":
                self._send(200, {"tasks": self.worker.list_tasks()})
            elif path.startswith("/v1/tasks/"):
                rec = self.worker.get_task(path.split("/")[-1])
                if rec is None:
                    self._send(404, {"error": "task_not_found", "task_id": path.split("/")[-1]})
                else:
                    self._send(200, rec.to_dict())
            else:
                self._send(404, {"error": "not_found", "path": path})
        except LibraryIntegrityError as exc:
            self._send(409,{'error':'source_integrity_error','message':str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": "internal_error",
                             "error_type": type(exc).__name__, "message": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/") or "/"
        try:
            if path == '/v1/imports':
                from .library import MAX_BYTES, ImportErrorDetail
                if not self.worker.library:
                    self._send(503,{'error':'library_unavailable','message':'资料库未配置'}); return
                n=int(self.headers.get('Content-Length') or 0)
                if not 0<n<=MAX_BYTES:
                    self._send(413,{'error':'upload_too_large','message':'PDF 大小必须在 1 字节到 20 MB 之间。'}); return
                try:
                    body,code=self.worker.library.submit(self.rfile.read(n),unquote(self.headers.get('X-File-Name','')))
                except ImportErrorDetail as exc:
                    self._send(400,{'error':'invalid_pdf','message':str(exc)}); return
                self._send(code,body); return
            payload = self._read_json()
            if path == "/v1/tasks":
                body, code = self.worker.submit(payload)
                self._send(code, body)
            elif path.startswith("/v1/tasks/") and path.endswith("/cancel"):
                task_id = path.split("/")[-2]
                body, code = self.worker.cancel(task_id)
                self._send(code, body)
            else:
                self._send(404, {"error": "not_found", "path": path})
        except ValidationError as exc:
            self._send(400, exc.to_dict())
        except LibraryIntegrityError as exc:
            self._send(409,{'error':'source_integrity_error','message':str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": "internal_error",
                             "error_type": type(exc).__name__, "message": str(exc)})


def make_server(worker: Worker, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    handler = type("BoundWorkerHandler", (WorkerHandler,), {"worker": worker})
    return ThreadingHTTPServer((host, port), handler)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m docmind.server.worker",
        description="DocMind 常驻推理 worker（本地 HTTP，默认只绑 127.0.0.1）")
    p.add_argument("--host", default=DEFAULT_HOST)
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--queue-max", type=int, default=DEFAULT_QUEUE_MAX)
    p.add_argument("--journal-dir", default=str(DEFAULT_JOURNAL_DIR))
    p.add_argument("--state-dir", default=None, help="持久任务目录；默认按后端隔离于logs/service/tasks")
    p.add_argument("--backend", choices=["real", "stub"], default="real",
                   help="real = 真实模型；stub = 显式桩后端（会在 health/result 标记）")
    p.add_argument("--model", default="Qwen2.5-7B-Instruct")
    p.add_argument("--load-at-startup", action="store_true",
                   help="启动时就加载模型（否则第一个任务触发懒加载）")
    # 以下三个只服务于故障分支测试，生产无需理会
    p.add_argument("--stub-stage-delay-ms", type=int, default=120)
    p.add_argument("--stub-fail", action="store_true",
                   help="桩后端在最后一个阶段抛异常，用于验证 failed 传播")
    p.add_argument("--force-unavailable", metavar="REASON", default=None,
                   help="启动即标记不可用（故障注入），所有提交返回 503")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.backend == "stub":
        engine: Engine = StubEngine(stage_delay_ms=args.stub_stage_delay_ms,
                                    fail=args.stub_fail)
    else:
        engine = RealEngine(model=args.model)

    worker = Worker(engine, queue_max=args.queue_max,
                    journal_dir=Path(args.journal_dir),
                    state_dir=Path(args.state_dir) if args.state_dir else ROOT/"logs/service/tasks"/args.backend)
    if args.force_unavailable:
        worker.mark_unavailable(args.force_unavailable)
    elif args.load_at_startup:
        try:
            engine.ensure_ready()
        except Exception as exc:  # noqa: BLE001
            # 加载失败必须变成"不可用"而不是"假装能跑"。
            worker.mark_unavailable(f"model load failed: {type(exc).__name__}: {exc}")

    worker.start()
    from .library import Library
    from . import sources
    encoder = None
    if args.backend == 'stub':
        from .demo_encoder import DemoEncoder
        encoder = DemoEncoder()
    worker.library=Library(ROOT/'library'/args.backend,encoder=encoder,exclusive=True)
    sources.library=worker.library
    if isinstance(engine,RealEngine): engine.library=worker.library
    httpd = make_server(worker, args.host, args.port)
    print(f"[worker] 监听 http://{args.host}:{args.port}  backend={args.backend}  "
          f"queue_max={args.queue_max}", flush=True)
    print(f"[worker] 日志目录 {worker.journal_dir}", flush=True)
    try:
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("[worker] 收到中断，关闭中…", flush=True)
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
