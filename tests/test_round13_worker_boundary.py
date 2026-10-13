"""第十三轮：Python 推理 worker 的边界测试（**不加载任何模型**）。

覆盖的是"服务化之后新出现的那部分正确性"：
    - 有界队列的背压（429）与不可用（503）
    - 生命周期五态是否真的互斥
    - 取消的两种语义（排队取消一次都不跑 / 运行取消只能协作停止）
    - 每任务唯一日志、日志失败要报出来而不是静默吞
    - failed / unknown 不得被包装成通过

为什么用桩后端而不是真模型：
    上面每一条都和 7B 模型无关，用真模型跑一遍要几十分钟还占显存。
    真模型的验证单独走"一个问题的真实冒烟"，不在这里重复。
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from docmind.eval.call_journal import JOURNAL_SCHEMA, CallJournal
from docmind.server.worker import (CANCELLED, COMPLETED, FAILED, QUEUED, RUNNING,
                                   StubEngine, Worker, make_server,
                                   validate_submit, ValidationError)


# ---------------------------------------------------------------------------
# 测试脚手架
# ---------------------------------------------------------------------------

class Client:
    """极简 HTTP 客户端（只用标准库，避免给测试引入 requests 依赖）。"""

    def __init__(self, port: int):
        self.base = f"http://127.0.0.1:{port}"

    def _req(self, method: str, path: str, body=None):
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body, ensure_ascii=False).encode("utf-8")
            headers["Content-Type"] = "application/json; charset=utf-8"
        req = urllib.request.Request(self.base + path, data=data, headers=headers,
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw = resp.read().decode("utf-8")
                return resp.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8")
            try:
                return e.code, json.loads(raw)
            except json.JSONDecodeError:
                return e.code, {"raw": raw}

    def get(self, path):
        return self._req("GET", path)

    def post(self, path, body=None):
        return self._req("POST", path, body if body is not None else {})


def _serve(worker: Worker):
    httpd = make_server(worker, "127.0.0.1", 0)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05},
                         daemon=True)
    t.start()
    return httpd, Client(port)


@pytest.fixture
def stub_worker(tmp_path):
    """一个跑桩后端的 worker，监听随机端口。"""
    engine = StubEngine(stage_delay_ms=60)
    worker = Worker(engine, queue_max=4, journal_dir=tmp_path / "journals")
    worker.start()
    httpd, client = _serve(worker)
    yield worker, client, engine
    httpd.shutdown()
    httpd.server_close()
    worker.shutdown()


def _wait_terminal(client, task_id, timeout=20.0):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        code, body = client.get(f"/v1/tasks/{task_id}")
        assert code == 200, body
        last = body
        if body["status"] in (COMPLETED, FAILED, CANCELLED):
            return body
        time.sleep(0.05)
    raise AssertionError(f"任务未在 {timeout}s 内进入终态，最后状态 {last}")


def _wait_status(client, task_id, status, timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        _, body = client.get(f"/v1/tasks/{task_id}")
        if body["status"] == status:
            return body
        time.sleep(0.02)
    raise AssertionError(f"未观察到状态 {status}")


# ---------------------------------------------------------------------------
# 1. 健康与后端标记
# ---------------------------------------------------------------------------

def test_health_marks_stub_backend_explicitly(stub_worker):
    worker, client, _ = stub_worker
    code, body = client.get("/v1/health")
    assert code == 200
    assert body["backend"] == "stub"
    assert body["stub"] is True, "桩后端必须被显式标记，否则假答案会混进生产"
    assert body["journal_schema"] == JOURNAL_SCHEMA == 2
    assert body["executor_alive"] is True


# ---------------------------------------------------------------------------
# 2. 输入校验
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("payload", [
    {"question": ""},
    {"question": "   "},
    {},
    {"question": 123},
    {"question": "x" * 2001},
    {"question": "abc", "token_budget": 10},
    {"question": "abc", "token_budget": 99999},
    {"question": "abc", "max_rounds": 99},
    {"question": "abc", "top_k": 0},
])
def test_invalid_submit_returns_400(stub_worker, payload):
    _, client, _ = stub_worker
    code, body = client.post("/v1/tasks", payload)
    assert code == 400, body
    assert body["error"] == "invalid_request"


def test_bool_is_not_silently_accepted_as_int():
    """{"max_rounds": true} 不能被当成 1 静默接受。"""
    with pytest.raises(ValidationError):
        validate_submit({"question": "a", "max_rounds": True})
    with pytest.raises(ValidationError):
        validate_submit({"question": "a", "token_budget": True})


def test_valid_boundaries_accepted():
    out = validate_submit({"question": "a", "token_budget": 256, "max_rounds": 0, "top_k": 1})
    assert out["token_budget"] == 256 and out["max_rounds"] == 0 and out["top_k"] == 1


# ---------------------------------------------------------------------------
# 3. 完整通路
# ---------------------------------------------------------------------------

def test_full_path_submit_poll_result(stub_worker):
    _, client, _ = stub_worker
    code, body = client.post("/v1/tasks", {"question": "sinh 正则化的收敛速度？"})
    assert code == 202, body
    task_id = body["task_id"]
    assert body["status"] == QUEUED

    final = _wait_terminal(client, task_id)
    assert final["status"] == COMPLETED
    result = final["result"]
    # 结果必须结构完整，且桩后端标记一路带到底
    for key in ("answer", "evidence", "verification_status", "llm_verification_status",
                "citation_validity", "verification_fail_reasons", "issues",
                "stop_reason", "usage", "trace", "backend"):
        assert key in result, f"结果缺少字段 {key}"
    assert result["backend"] == "stub"
    assert result["verification_status"] == "not_run"
    assert result["passed_verification"] is False
    # 校验未执行必须在 warnings 里讲明白，不能让调用方把 not_run 读成通过
    assert any("not_run" in w or "未执行" in w for w in result["warnings"])


def test_two_tasks_run_serially_never_concurrently(tmp_path):
    engine = StubEngine(stage_delay_ms=150)
    worker = Worker(engine, queue_max=4, journal_dir=tmp_path / "j")
    worker.start()
    httpd, client = _serve(worker)
    try:
        _, a = client.post("/v1/tasks", {"question": "第一个问题"})
        _, b = client.post("/v1/tasks", {"question": "第二个问题"})
        seen = []
        deadline = time.time() + 20
        while time.time() < deadline:
            _, va = client.get(f"/v1/tasks/{a['task_id']}")
            _, vb = client.get(f"/v1/tasks/{b['task_id']}")
            seen.append((va["status"], vb["status"]))
            if va["status"] in (COMPLETED, FAILED) and vb["status"] in (COMPLETED, FAILED):
                break
            time.sleep(0.02)
        # 核心断言：任何时刻都不允许两个任务同时 running
        assert not any(sa == RUNNING and sb == RUNNING for sa, sb in seen), \
            "单执行线程保证了串行；出现并发说明队列/执行器设计被破坏"
        assert any(sa == RUNNING and sb == QUEUED for sa, sb in seen), \
            "应当观察到第二个任务在排队（否则没真正测到串行）"
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.shutdown()


# ---------------------------------------------------------------------------
# 4. 背压：队列满 / worker 不可用
# ---------------------------------------------------------------------------

def test_queue_full_returns_429(tmp_path):
    engine = StubEngine(stage_delay_ms=400)     # 让第一个任务占住执行器
    worker = Worker(engine, queue_max=1, journal_dir=tmp_path / "j")
    worker.start()
    httpd, client = _serve(worker)
    try:
        code1, a = client.post("/v1/tasks", {"question": "占住执行器"})
        assert code1 == 202
        _wait_status(client, a["task_id"], RUNNING)
        code2, b = client.post("/v1/tasks", {"question": "排队"})
        assert code2 == 202, b            # 队列容量 1，这个刚好排进去
        code3, c = client.post("/v1/tasks", {"question": "应该被拒绝"})
        assert code3 == 429, c
        assert c["error"] == "queue_full"
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.shutdown()


def test_worker_unavailable_returns_503(stub_worker):
    worker, client, _ = stub_worker
    worker.mark_unavailable("故障注入：模型加载失败")
    code, body = client.post("/v1/tasks", {"question": "a"})
    assert code == 503
    assert body["error"] == "worker_unavailable"
    assert "模型加载失败" in body["message"], "503 必须给出可排障的理由，不能只有一句话"


# ---------------------------------------------------------------------------
# 5. 取消
# ---------------------------------------------------------------------------

def test_cancel_queued_task_never_runs(tmp_path):
    engine = StubEngine(stage_delay_ms=500)
    worker = Worker(engine, queue_max=4, journal_dir=tmp_path / "j")
    worker.start()
    httpd, client = _serve(worker)
    try:
        _, a = client.post("/v1/tasks", {"question": "先跑起来的"})
        _wait_status(client, a["task_id"], RUNNING)
        _, b = client.post("/v1/tasks", {"question": "排队中，将被取消"})

        code, body = client.post(f"/v1/tasks/{b['task_id']}/cancel")
        assert code == 200, body
        assert body["cancel_mode"] == "queued"
        assert body["status"] == CANCELLED

        _wait_terminal(client, a["task_id"])
        time.sleep(0.3)
        # 关键：执行次数必须是 1 —— B 从未进入引擎。
        # 只看状态是 cancelled 证明不了"没跑过"（状态可以被直接设置）。
        assert engine.runs == 1, f"排队任务被执行了 {engine.runs} 次，取消没有真正生效"
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.shutdown()


def test_cancel_running_task_is_cooperative_and_ends_cancelled(tmp_path):
    engine = StubEngine(stage_delay_ms=400)
    worker = Worker(engine, queue_max=4, journal_dir=tmp_path / "j")
    worker.start()
    httpd, client = _serve(worker)
    try:
        _, a = client.post("/v1/tasks", {"question": "运行中取消"})
        _wait_status(client, a["task_id"], RUNNING)

        code, body = client.post(f"/v1/tasks/{a['task_id']}/cancel")
        assert code == 202, body
        assert body["cancel_mode"] == "cooperative"
        assert body["status"] == RUNNING, \
            "运行中的任务不能声称已停止 —— 只能承诺在下一个阶段边界停"

        final = _wait_terminal(client, a["task_id"])
        assert final["status"] == CANCELLED
        assert final["result"]["stop_reason"] == "cancelled_by_client"
        assert final["result"]["verification_status"] == "not_run"
        assert final["result"]["passed_verification"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.shutdown()


def test_cancelling_terminal_task_returns_409(stub_worker):
    _, client, _ = stub_worker
    _, a = client.post("/v1/tasks", {"question": "a"})
    _wait_terminal(client, a["task_id"])
    code, body = client.post(f"/v1/tasks/{a['task_id']}/cancel")
    assert code == 409
    assert body["error"] == "task_already_terminal"


def test_cancel_unknown_task_returns_404(stub_worker):
    _, client, _ = stub_worker
    code, body = client.post("/v1/tasks/does-not-exist/cancel")
    assert code == 404


# ---------------------------------------------------------------------------
# 6. 失败传播
# ---------------------------------------------------------------------------

def test_engine_exception_becomes_failed_not_completed(tmp_path):
    engine = StubEngine(stage_delay_ms=10, fail=True)   # 故障注入
    worker = Worker(engine, queue_max=4, journal_dir=tmp_path / "j")
    worker.start()
    httpd, client = _serve(worker)
    try:
        _, a = client.post("/v1/tasks", {"question": "会失败的"})
        final = _wait_terminal(client, a["task_id"])
        assert final["status"] == FAILED
        assert final["result"]["answer_status"] == "error"
        assert final["result"]["verification_status"] == "not_run"
        assert final["result"]["passed_verification"] is False
        assert "RuntimeError" in str(final["error"])
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.shutdown()


def test_unknown_task_status_returns_404(stub_worker):
    _, client, _ = stub_worker
    code, _ = client.get("/v1/tasks/nope")
    assert code == 404


# ---------------------------------------------------------------------------
# 7. 每任务唯一日志
# ---------------------------------------------------------------------------

def test_each_task_gets_its_own_journal_file(stub_worker):
    worker, client, _ = stub_worker
    _, a = client.post("/v1/tasks", {"question": "第一个"})
    _, b = client.post("/v1/tasks", {"question": "第二个"})
    fa = _wait_terminal(client, a["task_id"])
    fb = _wait_terminal(client, b["task_id"])

    pa = Path(fa["result"]["journal"]["path"])
    pb = Path(fb["result"]["journal"]["path"])
    assert pa != pb, "两个任务不能共用一份日志"
    assert pa.is_file() and pb.is_file()
    assert pa.read_text(encoding="utf-8").strip(), "日志不能是空的"
    assert fa["result"]["journal"]["schema"] == JOURNAL_SCHEMA
    assert fa["result"]["journal"]["n_records"] > 0


def test_journal_write_failure_is_reported_not_swallowed(stub_worker, monkeypatch):
    """日志写不进去时：任务照常完成，但必须把留痕失败报出来。

    这里测的是第十轮就定下的原则——留痕失败不能静默吞，
    也不能因为记日志而把一次本可成功的推理打成 failed。
    """
    worker, client, _ = stub_worker

    # 故障注入点在**文件句柄**上，而不是直接替换 write()。
    # 直接替换 write() 会连它内部那个"接住 I/O 异常"的兜底一起换掉，
    # 那测的是我们自己造出来的坏实现，不是真实路径。
    # 真实路径是：write() 接住 OSError -> 攒进 journal.errors -> 由 worker 报出来。
    original_init = CallJournal.__init__

    class _BrokenWriter:
        def write(self, _s):
            raise OSError("disk full（故障注入）")

        def flush(self):
            pass

    def broken_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._fh = _BrokenWriter()

    monkeypatch.setattr(CallJournal, "__init__", broken_init)
    _, a = client.post("/v1/tasks", {"question": "日志会失败的"})
    final = _wait_terminal(client, a["task_id"])

    assert final["status"] == FAILED, "缺失可靠日志的任务不能认证为完整成功"
    errs = final["result"]["journal_errors"]
    assert errs and any("disk full" in e for e in errs), \
        f"留痕失败必须出现在结果里，实际 journal_errors={errs}"
