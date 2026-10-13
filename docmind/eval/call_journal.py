# DocMind —— 逐调用持久化日志

"""
为什么要有这个模块（第十轮修掉的可复现缺陷）

第九轮的诊断脚本 `replay_answer_review.py` 是**逐行**写检查点的：
一行 = 一次（回答 + 校验）的完整组合。于是只要这一行没走完，
**已经花掉的 token、已经发出的 prompt、模型已经给出的原始回复、
以及"程序到底选了哪一条来源"全都只存在于内存里**，进程一崩就全部消失。

复现方式（第十轮要求的桩测试就是照这个写的）：

    桩 LLM 在 source_select 节点返回 {"choice": 1} 和真实 usage
    -> 程序据此渲染了已核对的原文公式
    -> 下一个节点 verify 抛异常
    -> 旧实现：这一行什么都不留，检查点里既没有那次调用的账，也没有那个选择

这类"缺一半"的记录最危险的地方不是少了几行日志，而是**读的人无法判断
这次失败到底走了多远**：是选择之前就崩了，还是选完之后才崩的？
预算还剩多少？模型到底看到了什么 prompt？没有逐调用日志，这三个问题都答不了。

所以本模块把留痕的粒度下沉到**每一次 LLM 调用**：

    row_start           一行开始
      call_start        调用**之前**先落盘（prompt 全文 + 哈希 + 预算余额）
        call_end        调用**之后**落盘（原回复 + 实际 usage + 新的余额）
        call_end        异常时同样落盘，usage 记 None（见下）
      note              节点级的补充说明（例如"原文引用渲染了多少字符"）
    row_end             行结束，status ∈ {ok, error, cancelled}
  run_end               整轮结束，status ∈ {complete, error, cancelled, ...}

【为什么"未知 usage 记 null 而不是 0"】

把没拿到的 usage 写成 0，读的人会以为"这次调用没花钱"，
于是账本看起来是平的、预算看起来没超 —— 而真实情况是**不知道**。
"不知道"和"确定是 0"是两种完全不同的证据强度，混在一起就等于编数据。
所以本模块对每次调用都显式给 `usage_known`，未知时 token 字段一律为 None。

【为什么用 JSONL 且每条立即 flush】

JSONL 的每一行都是自包含的，追加写不会破坏前面已经写好的内容；
进程被 kill 时最多丢掉**当前这一条**，而不是整个文件。
每条写完立刻 flush，保证字节落到操作系统（进程被杀也留得下）。
注意：flush 只能保证扛住进程终止，扛不住机器断电 —— 这一点如实写明，
不假装 fsync 过的日志等于"永不丢失"。

【为什么路径默认拒绝覆写】

留痕文件一旦被后面的运行覆盖，就等于把上一次失败的证据擦掉了。
所以 `CallJournal` 默认用 'x' 模式打开（文件已存在就报错），
想复用路径必须显式传 `overwrite=True` 并说明理由。

【第十一轮新增：保留字段不可覆盖】

第十轮的日志里 114 条记录中有 14 条 `row_start` 的 `seq` 是**题序**
（1..14），而不是全局事件号。原因就在 `write()` 里：

    rec = {"seq": self._n, "ts": ..., **record}    # ← record 在后面，会覆盖 seq

于是第二行的 `seq` 被写成 2，而它真实的事件位置早已远大于 2。
后果是 seq **重复且不单调**，`tail_since(seq)` 这种"取回某一行之后所有
证据"的切片就彻底失效了 —— 它可能取回别处的记录，也可能漏掉本行的。

现在改成 `{**record, "seq": ..., "ts": ...}`，并且一旦 payload 里出现
`seq` / `ts` / `event` 就直接抛 `JournalRecordError`：
宁可让写日志的人当场失败，也不要悄悄产出一份序号自相矛盾的证据。

【第十一轮新增：强杀留下的半条末行】

被 kill -9 时，最后一行往往只写了一半。旧实现 `read_records()` 对整份
文件做 `json.loads`，遇到这种半行直接抛 `JSONDecodeError`，
于是**前面已经 flush 的 113 条完整记录一起读不回来** ——
一条坏记录毁掉整份证据，这恰恰是最需要证据的那一刻。

现在改成逐行解析 + 逐行记录损坏情况：完整的部分照常返回，
损坏行的行号与原文前缀一并报出来，`summarize()` 里显式标注
`torn_tail` / `journal_complete=False`。既不静默丢弃，也不假装完整。
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# 通用修复：非法 JSON 转义
# ---------------------------------------------------------------------------

# JSON 规范（RFC 8259）允许的转义字符只有这 9 种；
# 其余全部非法。7B 模型在 JSON 字符串值里写 LaTeX 时必然踩到：
#   \sigma  \delta  \pi  \frac  \sqrt  \(  \)  \[  \]  \mathbb  \text ...
_VALID_JSON_ESCAPE = re.compile(r'\\(?:["\\/bfnrt]|u[0-9a-fA-F]{4})')


def repair_invalid_json_escapes(text: str) -> tuple[str, int]:
    r"""把**非法**的反斜杠转义补成合法转义，返回 (修复后文本, 修复处数)。

    【为什么要修，以及为什么这不是"帮模型圆谎"】

    `json.loads('{"a":"\sigma"}')` 抛 `Invalid \escape`，
    整段输出被判 parse_failed -> 综合 `unknown`。
    但模型**其实明确表态了**：它写了 `"passed": false` 和具体理由，
    只是理由里夹了 LaTeX 宏。把 `\sigma` 读成字面量 `\sigma`
    正是模型原本想写的东西 —— 修复只改变"能否解析"，不改变任何语义。

    【为什么必须是"逐字符扫描"而不是全局替换】

    `text.replace('\\', '\\\\')` 会把**已经合法**的 `\"`、`\\`、`\n`、
    `\u00e9` 一起破坏掉，把 "a"b" 变成 "a\\"b"。
    所以这里逐字符走，遇到合法转义就整段原样搬过去，只补非法的那些。

    【为什么返回修复处数】

    修了却不说，读产物的人会以为模型输出了严格 JSON，
    从而高估输出的规整程度。调用方要把这个数写进产物（见 react.py）。
    """
    if "\\" not in text:
        return text, 0

    out: list[str] = []
    n_repairs = 0
    i, n = 0, len(text)
    while i < n:
        if text[i] == "\\":
            m = _VALID_JSON_ESCAPE.match(text, i)
            if m:
                out.append(text[i:m.end()])     # 合法转义：一个字节都不动
                i = m.end()
                continue
            out.append("\\\\")                    # 非法转义：\ -> \\
            n_repairs += 1
            i += 1
            continue
        out.append(text[i])
        i += 1
    return "".join(out), n_repairs


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: 当前写入的日志格式版本。产物把它记进 `run.call_journal_schema`，
#: 独立校验脚本据此决定"这份日志能核对到什么程度"。
#:
#: 【为什么要有版本号】
#: 日志格式会演进，而旧产物是证据、不能改。校验器若不分版本，
#: 要么拿新规则误判旧产物，要么为了兼容旧产物而不敢加新断言。
#: 有了版本号就可以**分级**：v1 已知缺陷 -> 跳过 seq 类断言并显式报缺陷；
#: 无版本号（第九轮及更早）-> 本来就没有日志，直接跳过。
#:
#: v1：第十轮。致命缺陷是 `row_start(seq=题序)` 覆盖了全局事件序号。
#: v2：第十一轮。seq 由 `write()` 统一生成且 payload 禁止覆盖，
#:     题序改记 `row_seq`；读取侧容忍并报告强杀留下的半条末行。
JOURNAL_SCHEMA = 2


class JournalRecordError(ValueError):
    """payload 试图覆盖保留的事件字段（seq / ts / event）。

    为什么抛错而不是"让调用方赢"：
    seq 是全局唯一递增的事件序号，是 tail_since 切片的唯一依据。
    一旦被业务字段（比如"第几题"）覆盖，序号就不再唯一也不再单调，
    整套"按 seq 取回某一行证据"的语义就崩了，而崩的方式非常隐蔽 ——
    日志看起来还在，只是取出来的东西是错的。
    """


class CallJournal:
    """逐调用持久化日志。

    典型用法（实验脚本里）::

        journal = CallJournal(path)
        journal.run_start(seed=..., code_at_start=..., ...)
        for ...:
            journal.row_start(row_seq=..., group=..., qid=...)
            try:
                ...                      # journal 由 Agent 在每次 _call 里自动写入
                journal.row_end("ok", row=payload)
            except Exception as exc:
                journal.row_end("error", error={...})
        journal.run_end("complete")

    写入失败不会被静默吞掉：异常被记进 `self.errors`，
    由调用方在最终产物里报出来（留痕失败本身也是要报告的事实）。
    """

    #: 保留字段：这些由日志对象自己写，payload 一律不得携带。
    RESERVED_FIELDS = ("seq", "ts", "event")

    def __init__(self, path: str | Path, *, overwrite: bool = False,
                 encoding: str = "utf-8"):
        self.path = Path(path)
        self.encoding = encoding
        self.errors: list[str] = []
        self._n = 0
        # 内存里也留一份：让"这一行从哪条记录开始"可以被切片取回，
        # 用于把崩溃行之前已落盘的调用原样附进产物（不必重新读文件）。
        self._records: list[dict] = []
        mode = "w" if overwrite else "x"
        # 父目录先建好；'x' 保证路径唯一（见模块头"为什么默认拒绝覆写"）
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open(mode, encoding=encoding)

    # ---------------- 底层写入 ----------------

    def write(self, record: dict) -> dict | None:
        """写一条记录并立即 flush。返回写出去的那条（失败返回 None）。

        【第十一轮关键改动：保留字段的顺序】

        原来是 `{"seq": n, "ts": ..., **record}` —— record 在**后面**，
        于是调用方传进来的 `seq` 会覆盖我们自己生成的全局序号。
        现在反过来：`{**record, "seq": n, "ts": ...}`，
        `seq` / `ts` 永远由本方法定，谁都覆盖不了。

        I/O 失败不抛出（不能因为记日志把实验打断），但攒进 `self.errors`
        由调用方报出来 —— 留痕失败本身也是要报告的事实。
        """
        if not isinstance(record, dict):
            raise JournalRecordError(f"record 必须是 dict，实际 {type(record).__name__}")

        # seq / ts 是 write() 自己的产物，调用方带了就是编程错误。
        # 为什么在这里也查一遍而不只靠 _event()：
        # 直接调 write() 的调用点（例如驱动脚本写 finalize_error）
        # 同样可能手滑带上这两个字段。
        clash = [k for k in ("seq", "ts") if k in record]
        if clash:
            raise JournalRecordError(
                f"record 不得自带保留字段 {clash}；seq/ts 一律由日志对象生成，"
                f"题序请改用业务字段（例如 row_seq）")

        self._n += 1
        # 保留字段放**最后**：保证 seq / ts 永远是本方法生成的值。
        rec = {**record, "seq": self._n, "ts": datetime.now().isoformat(timespec="seconds")}
        try:
            self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fh.flush()
        except Exception as exc:      # noqa: BLE001 - 留痕失败不能让实验半途崩掉
            self.errors.append(f"{type(exc).__name__}: {exc}")
            return None
        self._records.append(rec)
        return rec

    def _event(self, event_name: str, **payload) -> dict | None:
        """结构化事件的统一入口。

        【为什么 payload 不许带 seq / ts / event】

        第十轮的 14 条 `row_start` 传了 `seq=题序`，把全局事件号覆盖成 1..14，
        于是日志里出现重复且回退的序号，`tail_since(seq)` 这种
        "取回某一行之后的所有证据"就失效了 —— 而它正是崩溃时我们唯一能依靠的东西。

        字段冲突属于**编程错误**，必须当场抛出来让人改；
        只有 I/O 失败才静默进 `self.errors`（见 `write`）。

        参数名为什么叫 `event_name` 而不是 `event`：
        `_event('row_start', event=...)` 若参数同名会先在 Python 层报
        "got multiple values"，我们自定义的 JournalRecordError 根本没机会抛出。
        """
        clash = [k for k in self.RESERVED_FIELDS if k in payload]
        if clash:
            raise JournalRecordError(
                f"payload 不得覆盖保留事件字段 {clash}；"
                f"题序请改用 row_seq，事件类型由各事件方法自动写入"
            )
        return self.write({"event": event_name, **payload})

    def tail_since(self, start_seq: int) -> list[dict]:
        """取回从某个全局事件序号起（**含该序号**）之后写入的记录。

        语义前提：`seq` 全局唯一递增。第十一轮之前这个前提被破坏
        （row_start 用题序覆盖了 seq），所以本方法当时不可靠。

        注意"含"这个边界：想取回"从下一行开始"，要传
        `journal.n_records + 1`（n_records 是已经写出的最后一条的 seq），
        传 `n_records` 会把上一条记录也带出来 —— 那样"本行的账本"里
        就混进了上一行的 row_end（第十一轮自查时抓到的 off-by-one）。
        """
        return [r for r in self._records if r["seq"] >= start_seq]

    @property
    def records(self) -> list[dict]:
        return list(self._records)

    @property
    def closed(self) -> bool:
        """句柄是否已关闭；用于断言"异常路径也没泄漏句柄"。"""
        return bool(getattr(self._fh, "closed", True))

    def close(self) -> None:
        try:
            if not self._fh.closed:
                self._fh.close()
        except Exception as exc:
            self.errors.append(f"close {type(exc).__name__}: {exc}")

    def __enter__(self) -> "CallJournal":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @property
    def n_records(self) -> int:
        return self._n

    # ---------------- 结构化事件 ----------------

    def run_start(self, **payload) -> dict | None:
        """整轮开始。**必须在任何耗时初始化之前**调用。

        为什么：如果先加载 7B 模型再写 run_start，那么模型加载失败的那一刻
        连"这轮开始了"都没记录 —— 而那正是最需要留痕的失败。
        """
        return self._event("run_start", **payload)

    def row_start(self, *, group: str, qid: str, row_seq: int | None = None,
                  **payload) -> dict | None:
        """一行开始。

        `row_seq` 是**题序**（第几题），只作为业务字段记录，
        **绝不能**写成 `seq` —— 那会覆盖全局事件序号（第十轮的 bug）。
        """
        return self._event("row_start", group=group, qid=qid, row_seq=row_seq,
                           **payload)

    def call_start(self, *, node: str, attempt: int, prompt: str,
                   max_new_tokens: int, budget_limit: int,
                   budget_remaining_before: int, **payload) -> dict | None:
        """调用**之前**落盘。

        必须在调用之前写：一旦 `llm.chat()` 抛异常，prompt 就再也没有第二次
        机会被记下来了 —— 而"模型到底被问了什么"恰恰是事后唯一能复盘的东西。
        """
        return self._event(
            "call_start", node=node, attempt=attempt,
            prompt=prompt,
            prompt_chars=len(prompt),
            prompt_sha256=_sha256_text(prompt),
            max_new_tokens_cap=max_new_tokens,
            budget_limit=budget_limit,
            budget_remaining_before=budget_remaining_before,
            **payload,
        )

    def call_end(self, *, node: str, attempt: int, status: str,
                 reply: str | None = None,
                 prompt_tokens: int | None = None,
                 output_tokens: int | None = None,
                 elapsed: float | None = None,
                 budget_remaining_after: int | None = None,
                 budget_calls: int | None = None,
                 usage_known: bool = True,
                 error_type: str | None = None,
                 error_message: str | None = None,
                 **payload) -> dict | None:
        """调用之后落盘。

        `status` ∈ {ok, skipped_budget, error}：
            ok              —— 真的调用了，usage 已知
            skipped_budget  —— 预算不够、根本没调用，此时 usage 确定为 0（不是"未知"）
            error           —— 调用抛异常，usage **未知**，一律记 None
        """
        return self._event(
            "call_end", node=node, attempt=attempt, status=status,
            reply=reply,
            reply_chars=None if reply is None else len(reply),
            reply_sha256=None if reply is None else _sha256_text(reply),
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            elapsed=elapsed,
            usage_known=usage_known,
            budget_remaining_after=budget_remaining_after,
            budget_calls=budget_calls,
            error_type=error_type,
            error_message=error_message,
            **payload,
        )

    def note(self, kind: str, **payload) -> dict | None:
        """节点级补充说明，用于记录"不是 LLM 生成"的那部分内容。

        典型用途：原文引用模式下，答案是**程序渲染**已核对来源得到的，
        它的字符数（`rendered_source_chars`）和 LLM 实际输出的 token 完全不是一回事。
        两者必须分开记，否则"答案很长"会被误读成"模型生成了很多"。
        """
        return self._event("note", kind=kind, **payload)

    def row_end(self, status: str, *, row: dict | None = None,
                error: dict | None = None, **payload) -> dict | None:
        return self._event("row_end", status=status, row=row, error=error,
                           **payload)

    def run_end(self, status: str, **payload) -> dict | None:
        return self._event("run_end", status=status, **payload)


# ---------------------------------------------------------------------------
# 读取：容忍强杀留下的半条末行，但**必须**报出来
# ---------------------------------------------------------------------------

def read_scan(path: str | Path) -> dict:
    """逐行读取一份日志，返回记录与**逐行**损坏情况。

    【为什么不能遇错就抛】
    强杀留下的半条末行会让 `json.loads` 抛 `JSONDecodeError`，
    旧实现因此整份读不回来 —— 一条坏记录毁掉 113 条完整证据，
    而这 113 条正是唯一能说明"崩之前走到哪"的东西。

    【为什么不能默默跳过】
    跳过了却不说，读的人会以为日志完整。
    所以损坏行的行号、错误、原文前缀全部列进 `corrupted`，
    `summarize()` 再据此把 `journal_complete` 置为 False。
    """
    p = Path(path)
    raw = p.read_text(encoding="utf-8") if p.exists() else ""

    records: list[dict] = []
    corrupted: list[dict] = []
    nonempty: list[int] = []

    for lineno, line in enumerate(raw.splitlines(), 1):
        s = line.strip()
        if not s:
            continue
        nonempty.append(lineno)
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as exc:
            corrupted.append({
                "line_no": lineno,
                "error": f"JSONDecodeError: {exc}",
                "raw_head": s[:120],
                "raw_chars": len(s),
            })
            continue
        if not isinstance(obj, dict):
            corrupted.append({
                "line_no": lineno,
                "error": "not a JSON object",
                "raw_head": s[:120],
                "raw_chars": len(s),
            })
            continue
        records.append(obj)

    # 尾行损坏：最后一个非空行正好是最后一条损坏记录。
    # 这是进程被强杀最典型的特征（写到一半被杀，文件不以换行结尾）。
    torn_tail = bool(corrupted) and bool(nonempty) and corrupted[-1]["line_no"] == nonempty[-1]

    # 完整前缀：从 seq 1 开始连续、且中间没有损坏行。
    # seq 可能缺失或非整数（手工拼出来的日志、被截断的记录），
    # 这时一律按"不连续"处理 —— 不能让 TypeError 把整份读取打崩。
    prefix_ok = bool(records)
    if prefix_ok:
        prefix_ok = records[0].get("seq") == 1
        for a, b in zip(records, records[1:]):
            sa, sb = a.get("seq"), b.get("seq")
            if not isinstance(sa, int) or not isinstance(sb, int) or sb != sa + 1:
                prefix_ok = False
                break

    return {
        "path": str(path),
        "exists": p.exists(),
        "n_lines": len(nonempty),
        "records": records,
        "n_records": len(records),
        "corrupted": corrupted,
        "n_corrupted": len(corrupted),
        "torn_tail": torn_tail,
        "prefix_continuous": prefix_ok,
    }


def read_records(path: str | Path, *, strict: bool = False) -> list[dict]:
    """读回一份日志；用于独立复核（不依赖写日志的那份代码）。

    `strict=False`（默认）：容忍半条末行，返回所有能解析的记录。
        损坏情况请用 `read_scan()` 查询 —— 本函数只返回记录，不报损坏，
        所以"有没有损坏"这件事必须靠 `read_scan()`/`summarize()` 回答，
        不能因为这里返回了 113 条就以为日志是完整的。
    `strict=True`：遇到任何非法行直接抛 `json.JSONDecodeError`，
        用于"必须保证日志完整"的场合（例如生成最终交付物时）。
    """
    if strict:
        out: list[dict] = []
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if not isinstance(obj, dict):
                        raise json.JSONDecodeError("not a JSON object", line, 0)
                    out.append(obj)
        return out
    return read_scan(path)["records"]


def summarize(path: str | Path) -> dict:
    """把日志汇总成"读者第一眼要的东西"：状态、每行调用数、账本、未知 usage。

    为什么单独提供这个：日志本身是流水账，
    而"这轮到底完没完"必须有一个一眼可见的答案 ——
    尤其要能区分 **run_end 缺失**（被强制终止，按未完成处理）和
    run_end 写了 cancelled / error（被捕获地中止）。

    【第十一轮新增的三个字段】
        torn_tail            最后一行是不完整的（强杀特征）
        n_calls_unclosed     call_start 之后没有配对 call_end 的调用次数
        journal_complete     文件存在 + 无损坏行 + 有 run_end + 状态 complete

    未闭合的调用也要计入 unknown usage：它**发起过**，但我们不知道花了多少，
    这与"确定没花（skipped_budget）"是两回事，绝不能合并统计。
    """
    scan = read_scan(path)
    records = scan["records"]

    summary = {
        "path": str(path),
        "exists": scan["exists"],
        "n_records": len(records),
        "n_corrupted_lines": scan["n_corrupted"],
        "corrupted_lines": scan["corrupted"],
        "torn_tail": scan["torn_tail"],
        "prefix_continuous": scan["prefix_continuous"],
        "run_status": None,          # None = 没有 run_end，按未完成处理
        "run_end_present": False,
        "rows": [],
        "n_calls": 0,
        "n_calls_unknown_usage": 0,
        "n_calls_error": 0,
        "n_calls_unclosed": 0,
        "output_tokens_known_sum": 0,
        "output_tokens_complete": True,
    }
    current: dict | None = None
    open_calls: list[str] = []
    for rec in records:
        ev = rec.get("event")
        if ev == "run_start":
            summary["run_start"] = {k: v for k, v in rec.items() if k not in ("event", "seq", "ts")}
        elif ev == "row_start":
            current = {"group": rec.get("group"), "qid": rec.get("qid"),
                       "row_seq": rec.get("row_seq"),
                       "calls": 0, "errors": 0, "unknown_usage": 0,
                               "unclosed": 0, "status": None, "closed": False}
            summary["rows"].append(current)
            open_calls = []
        elif ev == "call_start" and current is not None:
            current["calls"] += 1
            summary["n_calls"] += 1
            open_calls.append(f"{rec.get('node')}#{rec.get('attempt')}")
        elif ev == "call_end" and current is not None:
            if rec.get("status") == "error":
                current["errors"] += 1
                summary["n_calls_error"] += 1
            if rec.get("usage_known") is False:
                current["unknown_usage"] += 1
                summary["n_calls_unknown_usage"] += 1
            else:
                summary["output_tokens_known_sum"] += int(rec.get("output_tokens") or 0)
            key = f"{rec.get('node')}#{rec.get('attempt')}"
            if key in open_calls:
                open_calls.remove(key)
        elif ev == "row_end" and current is not None:
            current["status"] = rec.get("status")
            current["closed"] = True
            current["unclosed"] = len(open_calls)
            summary["n_calls_unclosed"] += len(open_calls)
            summary["n_calls_unknown_usage"] += len(open_calls)
            summary["output_tokens_complete"] = False
            open_calls = []
        elif ev == "run_end":
            summary["run_status"] = rec.get("status")
            summary["run_end_present"] = True
    # 文件在 run_end 之后被截断（连 run_end 都没写出来）也要算未闭合
    summary["n_calls_unclosed"] += len(open_calls)
    if open_calls:
        summary["n_calls_unknown_usage"] += len(open_calls)
        summary["output_tokens_complete"] = False

    summary["n_rows_closed"] = sum(1 for r in summary["rows"] if r["closed"])
    summary["n_rows_unclosed"] = len(summary["rows"]) - summary["n_rows_closed"]
    summary["journal_complete"] = (
        scan["exists"]
        and scan["n_corrupted"] == 0
        and summary["run_end_present"]
        and summary["run_status"] == "complete"
        and summary["n_calls_unclosed"] == 0
    )
    return summary
