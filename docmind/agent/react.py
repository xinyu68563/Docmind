# DocMind —— Agent 编排：反思式检索循环

"""
Plan → Retrieve → Reflect → Answer → Verify 闭环（Phase 2）

【为什么需要 Agent，直接"检索一次 + 生成"不行吗？】

单行 RAG（retrieve once → generate）有三个致命问题，而且是**一定会遇到**的：

    1. 用户的真实问题和检索词不一样
       用户问："sinh 正则化比高斯收敛快吗？"
       直接拿这句去检索，BM25 会被"吗"、"比"这些虚词干扰，
       向量检索则被整句的语义平均化，丢失了核心术语的权重。

    2. 一次检索召回不全
       复杂问题需要多角度证据。比如"对比 A 和 B 的收敛速度"，
       需要分别有 A 的收敛性证明、B 的收敛性证明、以及实验对比数据。
       一次 Top-5 召回几乎不可能同时覆盖这三类。

    3. 无法判断"没找到"这件事
       单行 RAG 检索不到就会硬着头皮生成，于是开始编造 —— 这就是幻觉。
       用户体验上表现为"回答很流畅但全是错的"，比直接说不知道更糟糕。

Agent 循环怎么解决这三个问题：
    问题 1 -> PLAN 节点：让 LLM 把原问题改写成适合检索的查询
    问题 2 -> PLAN 拆分 + RETRIEVE 多路：每个子查询各自召回
    问题 3 -> REFLECT 节点：LLM 显式判断"证据够不够"，不够就再来一轮

【为什么用手写状态机而不是 LangGraph】

这不是偷懒。原因有三：

    1. 循环里每个节点的输入输出你都能打印出来看，调试成本极低。
       LangGraph 的 StateGraph 封装后，出错时你看到的是一张图的异常，
       对初学者极不友好。
    2. 本项目的终止条件是**多重**的（见下），
       用 LangGraph 的条件边表达起来反而绕。
    3. 面试时你要能讲清楚"为什么这么设计"。手写状态机意味着每一个
       分支都是你自己写的，你讲得出来；用框架则容易变成"我调了个库"。

等你完全跑通这套逻辑后，把它翻译成 LangGraph 大约只需要 30 分钟
（节点 = 函数，边 = 这里的 if/else）。那时候你会真正理解框架在封装什么。

【终止条件（每一条都要能说清为什么存在）】

    max_rounds      : 最多反思几轮。硬上限，防止死循环烧 token。
    token_budget    : 累计**输出 token** 上限。成本控制（口径见 Budget 注释）。
    sufficient      : LLM 自己判断证据够了 —— 这是"提前退出"，
                      让简单问题不要浪费 3 轮。
    tried（去重）   : 追问和已试过的查询重复就停。防"反思退化"，见代码内注释。
    no_new_evidence : 追问检索回来的全是已见过的 chunk，再问也没用。
                      注意"没有新增条目"≠"这次检索没生效"：重复命中的
                      chunk 分数仍会累加、n_terms 也会 +1。trace 里为此
                      专门记了一条 note，说明和状态必须对齐。
    parse_failed    : 反思节点输出解析不出来 -> 保守停止，不再盲检索。

为什么每个都不能少？
    - 只有 max_rounds：所有问题都被迫跑满，慢且贵
    - 只有 sufficient：模型可能永远觉得不够，死循环
    - 没有 token_budget：无法做成本预估，线上会失控
    - 没有去重：模型会反复生成同一句追问的近似复述，
      实测问"用了什么深度学习模型"时连跑 3 轮、后 2 轮命中数都是 0
    - 没有 no_new_evidence：明知召回池没变化还要再花一轮 LLM 调用

【一个必须讲清楚的限制：证据分数不是"校准过的相关度"】

本模块把多条子查询的 RRF 分数**相加**作为 chunk 的分值。这是 RRF 在多路召回
下的标准做法，但你必须知道它是什么：
    - 它是**名次融合启发式**，不是概率、不是相似度、没有跨查询可比性
    - 两路检索器对不同查询给出的分数分布本来就不同，相加只是在说
      "这个 chunk 在多次检索里都排在前面"，仅此而已
    - 所以本模块**不会**声称 score 是"全局相关度"，也不会拿它做阈值判断，
      只用它排序。任何需要阈值的判断都交给 REFLECT 节点（LLM）去做。

【另一个必须讲清楚的限制：校验节点只报告，不修订】

VERIFY 节点发现问题时，当前实现只把 issues 记录下来并返回，
**不会**据此重新生成答案。也就是说这里不是一个"自我修订闭环"。
理由：7B 模型做修订经常把原本正确的句子改坏，且缺少可靠的自动判据，
贸然上线会引入新的错误。要真做修订需要额外的判据与人工抽检，属于后续工作。
README 与交付文档都必须如实描述这一点，不能宣称"已完成答案修订闭环"。
"""

from __future__ import annotations

from docmind.ingest.source_context import prepare_source_evidence, model_text
from docmind.agent.prompts import evidence_label, normalize_evidence_citations
from docmind.agent.prompts import SOURCE_SELECTOR
from docmind.ingest.source_context import formal_source_request, source_review_options, render_source_quote
from docmind.ingest.source_context import format_source_options, selected_source_option

import json
import re
import time
from dataclasses import dataclass, field

from docmind.agent.prompts import (
    ANSWERER,
    PLANNER,
    REFLECTOR,
    VERIFIER,
    format_evidence,
)
from docmind.eval.metrics import extract_citations
# 第十一轮：JSON 非法转义修复（只在"解析不了"这条路上用，见 parse_json_object）
from docmind.eval.call_journal import repair_invalid_json_escapes


# ---------------------------------------------------------------------------
# 常量：终止原因 / 校验状态 / 答案状态
#
# 为什么要用常量而不是裸字符串？
#   因为 terminate reason 是要写进评测结果、拿去做错误分析的。
#   裸字符串拼错一个字母，统计时就会静默漏掉一整类样本。
# ---------------------------------------------------------------------------

STOP_SUFFICIENT = "sufficient"                  # 模型判定证据已充分
STOP_MAX_ROUNDS = "max_rounds"                  # 用满轮次
STOP_NO_FOLLOWUP = "no_followup"                # 模型没给出可用的追问
STOP_DUPLICATE_QUERY = "duplicate_query"        # 追问与历史查询重复（反思退化）
STOP_NO_NEW_EVIDENCE = "no_new_evidence"        # 追问没有召回任何新证据
STOP_TOKEN_BUDGET = "token_budget"              # 输出 token 预算耗尽
STOP_REFLECT_FAILED = "reflect_failed"          # 反思输出无法解析 -> 保守停止
STOP_REFLECTION_DISABLED = "reflection_disabled"  # 配置关闭了反思（单行 RAG 基线）
# 第十三轮（Phase 4 服务化）新增：客户端在服务侧取消了任务。
#
# 为什么必须是一个独立的终止原因，而不是复用 max_rounds？
#   取消是"外部介入"，不是"策略自然结束"。混在一起会让实验统计无法区分
#   "模型跑完了但没答好"和"用户中途叫停"——后者根本不该计入质量指标。
#
# 【必须讲清楚的能力边界】这个常量只代表"在两个节点之间停下来"。
# 模型的一次 generate() 一旦开始，本地推理进程无法安全地把它掐断
# （强行 kill 线程会留下 GPU 显存与 CUDA 上下文的脏状态）。
# 因此服务层不得对外宣称"已停止 GPU"，只能说"当前节点结束后停止"。
STOP_CANCELLED = "cancelled_by_client"

# --- LLM 校验节点自身的状态（只描述"模型那一步"）---
VERIFY_PASSED = "passed"
VERIFY_FAILED = "failed"
VERIFY_PARSE_FAILED = "parse_failed"        # 校验未真正执行 —— 不等同通过
VERIFY_NOT_RUN = "not_run"                  # 配置关闭
VERIFY_SKIPPED_BUDGET = "skipped_budget"    # 想校验但预算没了 —— 同样不等同通过

# --- 综合校验状态（外面看到的唯一结论）---
OVERALL_PASSED = "passed"
OVERALL_FAILED = "failed"
OVERALL_NOT_RUN = "not_run"     # 校验节点压根没跑（但仍然会做零成本引用检查）
OVERALL_UNKNOWN = "unknown"     # 校验没跑成 / 结果读不出来 —— 绝不等同通过

# --- 引用编号合法性（零成本规则判定，与 LLM 校验完全独立）---
CITE_VALID = "valid"
CITE_INVALID = "invalid"        # 存在越界编号，如 [99] / [999] / [0]
CITE_NONE = "none"              # 答案里没有任何引用（拒答常见，不算错）

ANSWER_GENERATED = "generated"
ANSWER_SKIPPED_BUDGET = "skipped_budget"    # 预算在生成前就耗尽，明确不生成

PLAN_OK = "ok"
PLAN_FALLBACK = "fallback"                  # 解析失败/空 -> 退回原问题（允许）
PLAN_DISABLED = "disabled"                  # 配置关闭规划（单行 RAG 基线）
PLAN_SKIPPED_BUDGET = "skipped_budget"      # 预算耗尽，不调用规划


# ---------------------------------------------------------------------------
# JSON 解析与字段类型校验
# ---------------------------------------------------------------------------

_TRUE_WORDS = {"true", "yes", "y", "1", "是", "够", "充分"}
_FALSE_WORDS = {"false", "no", "n", "0", "否", "不够", "不足"}


def parse_json_object(text: str) -> tuple[dict | None, str]:
    """从 LLM 输出里提取一个 JSON **对象**。

    返回 (对象 | None, 状态)，状态 ∈ {ok, empty, not_object, regex, failed}

    【为什么不能用 json.loads 一把梭】

    7B 模型输出 JSON 时常见的"不乖"行为：
        1. 用 markdown 代码块包起来：  ```json {...} ```
        2. 前面加了一句废话：          好的，结果是：{...}
        3. 用了单引号或尾逗号：        {'sufficient': True,}
        4. 直接输出多个 JSON 对象：    {...} {...}

    【最关键的一点：合法 JSON ≠ 合法结果】

    `[]`、`null`、`"true"`、`123` 都能被 json.loads 成功解析，
    但它们不是我们要的对象。原实现只 try/except JSONDecodeError，
    于是 `[]` 会被当成 `{}` 继续往下走，`.get()` 全部取到默认值 ——
    表面看流程正常，实际是"模型什么都没说，我们替它做了决定"。
    所以这里显式区分 not_object，让调用方知道"这次解析是失败的"。
    """
    if not text or not text.strip():
        return None, "empty"

    # 第 1 级：去掉 markdown 代码块围栏
    cleaned = re.sub(r"```(?:json)?\s*", "", text).strip()

    # 第 2 级：直接解析
    # 注意不能用 `data is None` 判断是否解析成功 ——
    # json.loads("null") 是**成功**的，只是结果是 None。
    # 必须用 try/except 的成败来区分"解析失败"和"解析出一个非对象"。
    try:
        data = json.loads(cleaned)
        parsed = True
    except json.JSONDecodeError:
        data, parsed = None, False

    if parsed:
        return (data, "ok") if isinstance(data, dict) else (None, "not_object")

    # 第 2.5 级（第十一轮新增）：非法 JSON 转义修复
    #
    # 【为什么需要这一级 —— 一个真实病例】
    # 第十轮产物 agent/q008 的校验节点**明明表态了**：
    #     {"passed": false, "issues": ["... 文中 \sigma^2 = \frac{m}{\pi-\delta} ..."]}
    # 但 `\sigma` / `\frac` 不是合法 JSON 转义，json.loads 直接抛
    # `Invalid \escape` -> parse_failed -> 综合结论 unknown。
    # 也就是说：**模型说"不通过"，我们却记成了"不知道"**。
    # 把 `\sigma` 当字面量读正是模型原本想写的东西，
    # 所以这一级只改变"能否解析"，不改写任何内容。
    #
    # 【为什么放在第 2 级之后、第 3 级之前】
    # 能正常解析的输出**一个字节都不动**（本分支只在解析失败后才进来），
    # 因此不会改变任何现有成功路径的行为。
    #
    # 【为什么要返回 "repaired:N" 而不是静默成功】
    # 修了却不说，读产物的人会以为模型输出了严格 JSON，
    # 从而高估输出的规整程度。N 会被写进 trace（见 _verify）。
    repaired, n_repairs = repair_invalid_json_escapes(cleaned)
    if n_repairs:
        try:
            data = json.loads(repaired)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return data, f"repaired:{n_repairs}"

    # 第 3 级：正则抓第一个 {...} 块（非贪婪不够用，JSON 里可能有嵌套）
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            # 正则抓出来的块同样可能带非法转义（最常出现在 issues 里的 LaTeX）
            fixed, n2 = repair_invalid_json_escapes(match.group(0))
            if n2:
                try:
                    data = json.loads(fixed)
                except json.JSONDecodeError:
                    return None, "failed"
                return (data, f"repaired:{n2}") if isinstance(data, dict) else (None, "not_object")
            return None, "failed"
        return (data, "regex") if isinstance(data, dict) else (None, "not_object")

    return None, "failed"


def parse_json_robust(text: str, default: dict) -> dict:
    """兼容旧接口：解析失败就返回 default。

    只给"解析失败可以安全退化"的调用方用（例如批量造题脚本）。
    Agent 主流程不要用这个 —— 它把"模型没说话"和"模型说了 false"
    混成了同一件事，正是本轮要修的问题。
    """
    data, _ = parse_json_object(text)
    return data if data is not None else default


def as_bool(value) -> bool | None:
    """把 LLM 给的字段安全地转成布尔，**无法判定时返回 None**。

    【这是本轮修掉的一个真实缺陷】

    原实现写的是 `bool(data.get("sufficient", True))`。Python 里：
        bool("false")  -> True      ← 字符串非空即真！
        bool("0")      -> True
        bool("")       -> False
    也就是说模型老老实实输出 `{"sufficient": "false"}`，
    我们却把它读成了"证据充分"，直接退出循环去生成答案。
    这类 bug 不会报错，只会让系统在你最需要它谨慎的时候变得激进。

    返回 None 表示"不知道"，调用方必须按未知处理，不能当 True 用。
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        # 只接受 0/1 这两个语义明确的数字，2、-1、0.5 都视为未知
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    if isinstance(value, str):
        s = value.strip().lower()
        if s in _TRUE_WORDS:
            return True
        if s in _FALSE_WORDS:
            return False
    return None


def as_str_list(value, max_items: int | None = None) -> list[str]:
    """把 LLM 给的字段安全地转成字符串列表，非字符串元素直接丢弃。

    为什么不能只写 `data.get("queries", [])`：
    模型可能给你 `{"queries": [123]}` 或 `{"queries": null}` 或
    `{"queries": {"a": "b"}}`，这些都会让后面的 `q.strip()` 抛异常，
    或者更糟 —— 静默产出 `[123]` 拿去检索。
    """
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for x in value:
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
    if max_items is not None:
        out = out[:max_items]
    return out


def _short(x, n: int = 40) -> str:
    """把任意值压成短字符串，用于写进 trace 的原因说明。"""
    try:
        s = repr(x)
    except Exception:
        s = "<unrepr>"
    return s if len(s) <= n else s[:n] + "…"


def issues_schema_error(value, max_items: int | None = None) -> str | None:
    """检查 `issues` 字段是不是"有效非空字符串的列表"。合法返回 None，否则返回原因。

    【第三轮修掉的静默放行】

    第二轮只校验了 `isinstance(raw_issues, list)`，随后交给 `as_str_list()`
    —— 而 as_str_list 的设计就是**丢弃非法元素**。于是：

        {"passed": true, "issues": [123]}                  -> [] -> passed
        {"passed": true, "issues": [null]}                 -> [] -> passed
        {"passed": true, "issues": [{"reason": "..."}]}    -> [] -> passed

    三个都"通过了校验"。这不是边缘情况：模型输出半结构化内容（编号、null、
    嵌套对象）非常常见，而"过滤掉看不懂的东西"恰好把**它其实在报告问题**
    变成了"它说没问题"。过滤只适用于"少了这项也能正常工作"的字段
    （比如 planning 的 queries），不适用于"有没有问题"这种判定性字段。

    所以这里改成严格 schema：
        必须是 list；每个元素必须是 str 且 strip() 后非空。
        空列表是合法的（"确实没问题"），但**任何**非法元素都不允许。
    """
    if not isinstance(value, list):
        return ("issues 字段缺失" if value is None
                else f"issues 字段类型非法: {type(value).__name__}")
    for i, x in enumerate(value):
        if not isinstance(x, str):
            # 把实际值也写进原因里：trace 要能让人一眼看出"模型到底输出了什么"，
            # 只说"不是字符串"等于让人去猜。
            return f"issues[{i}] 不是字符串: {type(x).__name__}（值 {_short(x)}）"
        if not x.strip():
            return f"issues[{i}] 是空白字符串（原值 {_short(x)}）"
    if max_items is not None and len(value) > max_items:
        # 超出上限不算 schema 错误（内容本身合法），只截断，
        # 但要在 trace 里说明"报告的是截断后的版本"。
        return None
    return None


#: 拒答的开场白：这类答案**本来就没有任何引用**，不能算"来源不可追溯"。
#: 少了这条豁免，"模型诚实地说不知道"也会被引用规则判失败 ——
#: 那是在惩罚正确的拒绝行为，会反过来逼模型硬编一个答案出来。
REFUSAL_PREFIXES = ('根据当前文献无法回答', '当前文献无法回答', '无法根据当前文献回答')


def citation_format_invalid(answer: str, *, opaque: bool) -> bool:
    """不透明句柄模式下，答案是否**来源不可追溯**。

    为什么单独抽成函数（第十一轮）：
    判"零引用 -> failed"的规则原先内联在 `_answer()` 里，无法直接测试；
    而 q029 恰恰是这条规则判出来的失败。抽出来之后，
    "拒答不该被误杀"这个反向用例才测得到 —— 否则测试只能把实现抄一遍，
    那种"测试"和没测一样。

    判定：
        没开不透明句柄        -> 不适用（False）
        用了 [1] 这类数字编号  -> 不可追溯（模型没用我们给的句柄）
        既没有 [EVID_x] 也不是拒答 -> 不可追溯
        是拒答               -> 合法（豁免）
    """
    if not opaque:
        return False
    text = answer or ''
    handles = re.findall(r'\[EVID_[A-Z]+\]', text, flags=re.IGNORECASE)
    refusal = text.strip().startswith(REFUSAL_PREFIXES)
    return bool(extract_citations(text)) or (not handles and not refusal)


def citation_issues(answer: str, n_evidence: int) -> list[str]:
    """零成本的引用编号范围检查。

    为什么值得单独做：LLM 校验要一次完整生成（几百到上千 token），
    而"引用了不存在的编号 [7]（一共只有 6 条证据）"这种错误
    用一条正则就能 100% 确定地查出来，且零随机性、可复现。
    这不是 faithfulness —— 它只能证明编号存在，不能证明内容支撑，
    两者必须分开统计、分开命名。

    合法区间是 1..n_evidence，所以 [0] 必定越界。
    """
    bad = sorted({c for c in extract_citations(answer) if not (1 <= c <= n_evidence)})
    return [f"引用编号 [{c}] 超出证据范围 1..{n_evidence}" for c in bad]


def citation_validity(answer: str, n_evidence: int) -> str:
    """给出引用合法性的三态判定：valid / invalid / none。

    为什么是三态而不是布尔：
    "没有引用"和"引用全对"含义完全不同（前者可能是拒答，也可能漏标），
    压成 True/False 会让指标失真 —— 这是第一轮已经踩过的坑。
    """
    cites = extract_citations(answer)
    if not cites:
        return CITE_NONE
    return CITE_INVALID if any(not (1 <= c <= n_evidence) for c in cites) else CITE_VALID


def combine_verification(llm_status: str, cite_state: str, issues: list[str]) -> tuple[str, list[str]]:
    """把"LLM 校验结果"和"引用合法性规则"合并成一个**外部唯一结论**。

    为什么必须合并（第二轮审核发现的核心问题）：

    第一轮里 `passed_verification` 只依赖 LLM 的 passed 字段，
    `citation_issues` 虽然查出了 [99] 却只是"记录在案"——
    于是出现荒谬的状态：系统明确知道答案引用了不存在的编号，
    最终结论仍然是"校验通过"。查出来和正确处理是两件事。

    合并规则（保守优先）：
        1. 规则判定存在越界引用        -> failed（规则压过模型的自我认可）
        2. LLM 判定 failed            -> failed
        3. LLM 没跑成 / 结果读不出来  -> unknown（不是 passed）
        4. 节点压根没启用             -> not_run
        5. 其余                       -> passed
    """
    reasons: list[str] = []
    if cite_state == CITE_INVALID:
        reasons.append("citation")
    if llm_status == VERIFY_FAILED:
        reasons.append("llm")

    if reasons:
        return OVERALL_FAILED, reasons
    if llm_status == VERIFY_NOT_RUN:
        return OVERALL_NOT_RUN, []
    if llm_status == VERIFY_PASSED:
        return OVERALL_PASSED, []
    # parse_failed / skipped_budget：校验没真正跑完，结论是"不知道"
    return OVERALL_UNKNOWN, [llm_status]


# ---------------------------------------------------------------------------
# token 预算
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    """一次 run 的 token 账本。

    【口径必须写明，否则预算形同虚设】

    token_budget = **累计输出 token 上限**（不含 prompt）。
    选输出口径的理由：prompt 长度由证据条数决定、是被动的，
    而输出长度才是我们能主动控制、也最容易失控的部分。
    prompt token 也照样累计并落盘，但不参与预算判定。

    【本轮修掉的 bug】

    原实现里 `tokens_used` 只累加了 `_answer()` 的返回值，
    而规划、反思、校验三次 LLM 调用的 GenerationStats 全部被丢弃
    （代码里写的是 `reply, _ = self.llm.chat(...)`）。
    于是循环内的 `if tokens_used >= cfg.token_budget` 恒为假，
    token_budget 这个"成本控制"参数从来没生效过。

    【为什么 Budget 是每次 run 新建的对象】

    如果把它挂在 self 上，同一个 Agent 实例跑第二题就会带着第一题的
    消耗继续判预算，第二题直接被判定超预算 —— 评测时表现为
    "越往后跑质量越差"，非常难排查。所以 run() 里新建，绝不复用。
    """
    limit: int
    prompt_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.output_tokens)

    def spend(self, stats) -> None:
        """记账。只依赖 stats 的属性，不依赖具体类型（方便测试用假对象）。"""
        self.prompt_tokens += int(getattr(stats, "prompt_tokens", 0) or 0)
        self.output_tokens += int(getattr(stats, "output_tokens", 0) or 0)
        self.calls += 1

    def to_dict(self) -> dict:
        return {
            "limit": self.limit,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.prompt_tokens + self.output_tokens,
            "calls": self.calls,
            "remaining": self.remaining,
        }


# ---------------------------------------------------------------------------
# 轨迹
# ---------------------------------------------------------------------------

@dataclass
class AgentTrace:
    """一次运行的完整轨迹。

    为什么要存这个？
    1. 调试：某次回答质量差，你能回看是"检索没召回"还是"反思判断错了"
    2. 评测：Phase 3 要统计"平均反思几轮"、"每轮召回了什么"、"为什么停"
    3. 展示：前端可以画出一个"Agent 思考过程"的时间线，
       这是这类项目最直观的卖点
    """
    steps: list[dict] = field(default_factory=list)

    def add(self, node: str, **payload) -> None:
        self.steps.append({"node": node, **payload})

    def summary(self) -> str:
        lines = []
        for i, s in enumerate(self.steps, 1):
            node = s["node"]
            if node == "plan":
                tag = {"ok": "规划", "fallback": "规划失败退回原问题",
                       "disabled": "未启用规划", "skipped_budget": "预算不足跳过规划"}.get(
                           s.get("status", ""), "规划")
                lines.append(f"{i}. [{tag}] {len(s['queries'])} 条查询: {s['queries']}")
            elif node == "retrieve":
                lines.append(
                    f"{i}. [检索] 「{s['query']}」-> 新增 {s['n_new']} 条 "
                    f"(证据池 {s['n_pool']} 条)"
                )
            elif node == "reflect":
                suf = s.get("sufficient")
                mark = "充分" if suf is True else ("不足" if suf is False else "未知")
                detail = s.get("missing", "") or s.get("note", "")
                lines.append(f"{i}. [反思] 第{s['round']}轮: 证据{mark} {detail}")
            elif node == "answer":
                if s.get("status") == ANSWER_SKIPPED_BUDGET:
                    lines.append(f"{i}. [生成] 预算已耗尽，未生成答案")
                elif s.get('mode')=='reviewed_source_quote':
                    lines.append(f"{i}. [原文引用] {s['source_review_id']}，{s['rendered_source_chars']} 字符（程序呈现）")
                else:
                    lines.append(f"{i}. [生成] {s['tokens']} tokens, {s['seconds']:.1f}s")
            elif node == 'source_select':
                lines.append(f"{i}. [来源选择] {s.get('selected_id') or '未选择，回退普通回答'}")
            elif node == "verify":
                status = s.get("status", "")
                if status == VERIFY_PASSED:
                    lines.append(f"{i}. [校验] 通过")
                elif status == VERIFY_FAILED:
                    lines.append(f"{i}. [校验] 发现{len(s['issues'])}个问题（仅报告，未修订）")
                elif status == VERIFY_NOT_RUN:
                    lines.append(f"{i}. [校验] 未启用")
                else:
                    lines.append(f"{i}. [校验] 未执行（{status}），不能视为通过")
            elif node == "citation_check":
                lines.append(f"{i}. [引用检查] 引用 {s['n_citations']} 处，"
                             f"合法性 {s.get('validity')}，越界 {len(s['issues'])} 处")
            elif node == "verification_summary":
                lines.append(f"{i}. [综合结论] {s['overall']}"
                             + (f"（原因: {', '.join(s['fail_reasons'])}）"
                                if s["fail_reasons"] else ""))
            elif node == "llm_call":
                # 逐次调用的详细账目保留在 trace.steps 里供核对；
                # 摘要视图不逐条打印，避免刷屏。
                continue
        return "\n".join(lines)

    def llm_usage(self) -> dict:
        """从 trace 里把所有 llm_call 的 token 加总 —— 用于和总账对账。

        为什么要能独立加总：如果只有 Agent 自己报一个数，
        "统计漏记"这类 bug 永远发现不了。有了这条，
        测试可以直接断言 sum(trace) == result.output_tokens。
        """
        calls = [s for s in self.steps
                 if s["node"] == "llm_call" and s.get("status") == "ok"]
        return {
            "calls": len(calls),
            "prompt_tokens": sum(s["prompt_tokens"] for s in calls),
            "output_tokens": sum(s["output_tokens"] for s in calls),
            "elapsed": round(sum(s["elapsed"] for s in calls), 4),
            # 按"谁调用的"分组（plan/reflect/verify/answer），
            # 这样才能看出开销大头在哪个节点，而不只是拿到一个总数。
            "by_node": {
                n: sum(s["output_tokens"] for s in calls if s.get("caller") == n)
                for n in sorted({s.get("caller", "?") for s in calls})
            },
        }


# ---------------------------------------------------------------------------
# 配置与输出
# ---------------------------------------------------------------------------

@dataclass
class AgentConfig:
    """Agent 的可调参数。

    全部暴露出来是为了 Phase 3 的消融实验：
    你可以固定其他参数，只改 max_rounds，看它对答案质量的影响。

    【消融铁律：一次只改一个变量】
    下面三个开关就是"单行 RAG 基线"和"完整 Agent"的唯一差异点：
        enable_plan=False, max_rounds=0, enable_verify=False  -> 真正的单次 RAG
        enable_plan=True,  max_rounds=3, enable_verify=False  -> Agent 无校验
        enable_plan=True,  max_rounds=3, enable_verify=True   -> Agent + 校验
    注意旧版把"0 轮基线"实现成 max_rounds=1 且仍然规划 ——
    那样跑的是"规划 + 一次检索"，根本不是单行 RAG，消融结论不成立。
    """
    max_rounds: int = 3             # 最多反思轮数；<=0 表示完全不反思
    token_budget: int = 4000        # 累计输出 token 预算（口径见 Budget）
    top_k_per_query: int = 5        # 每条子查询召回几条
    max_evidence: int = 6           # 最终喂给模型的证据条数上限
    max_answer_words: int = 300
    enable_verify: bool = True      # 是否启用引用校验（只报告，不修订）
    opaque_citations: bool = False  # CLI/new experiments opt in; legacy library configs stay compatible.
    source_quote_answers: bool = False  # Preserve complete checked mathematical sections when relevant.
    enable_plan: bool = True        # 是否启用查询规划；关闭 = 直接用原问题检索
    max_node_retries: int = 1       # 反思/校验解析失败后的重试次数（计入预算）
    # 【已废弃，保留字段只为不破坏旧配置】
    # 第一轮曾用它"给未展示过的证据预留 k//2 个槽位"，第二轮改为
    # tier1(fresh)/tier2(unseen)/tier3(按分数) 分层优先后不再读取。
    # 之所以不直接删：删字段会让已有的 AgentConfig(..., new_evidence_slots=2)
    # 调用点直接 TypeError。诚实做法是标记为废弃并说明，而不是假装它还在生效。
    new_evidence_slots: int | None = None
    max_new_tokens_answer: int | None = None  # 生成节点单次上限，None = 按字数估算


@dataclass
class AgentAnswer:
    """Agent 的最终输出。

    passed_verification 只有 VERIFY 真正执行且判定通过时才为 True；
    未执行/解析失败/预算不足一律为 False，具体原因看 verification_status。
    """
    answer: str
    evidence: list[dict]            # [{id, text, source, score, ...}]
    rounds_used: int
    total_tokens: int               # prompt + output（总计费口径）
    prompt_tokens: int
    output_tokens: int              # 预算口径
    elapsed: float
    passed_verification: bool
    verification_status: str          # 综合结论：passed / failed / not_run / unknown
    llm_verification_status: str      # LLM 校验节点自己的状态
    citation_validity: str            # valid / invalid / none（零成本规则）
    verification_fail_reasons: list[str]
    issues: list[str]
    citation_issues: list[str]
    stop_reason: str
    answer_status: str
    plan_status: str
    trace: AgentTrace


# ---------------------------------------------------------------------------
# 主体
# ---------------------------------------------------------------------------

class DocMindAgent:
    """反思式检索 Agent。

    用法:
        agent = DocMindAgent(hybrid_retriever, llm)
        result = agent.run("sinh 正则化比高斯收敛快吗？")
        print(result.answer)
        print(result.trace.summary())
    """

    def __init__(self, retriever, llm, config: AgentConfig | None = None,
                 call_journal=None, stop_check=None):
        self.retriever = retriever
        self.llm = llm
        self.config = config or AgentConfig()
        # 【第十三轮：协作式取消钩子】
        # stop_check 是一个 `() -> bool` 的可调用对象，在"节点之间"被调用。
        #
        # 为什么是可选参数而不是必填：
        #   默认 None 时本类的行为与第十三轮之前**逐字节相同**，
        #   339 项既有回归依赖这一点。只有服务化路径才会传。
        #
        # 为什么不放进 AgentConfig：
        #   AgentConfig 会被 provenance.config_effective() 展开成完整字典写进产物。
        #   把一个函数塞进"有效配置"既不可序列化，也会让实验身份里出现
        #   一个每次运行都不同的东西——那会毁掉可复现性。
        #
        # 为什么只检查节点边界：
        #   见 STOP_CANCELLED 的注释。正在进行的 generate() 无法安全中断，
        #   声称能做到就是在撒谎。
        self._stop_check = stop_check
        # 逐调用持久化日志（第十轮）。
        #
        # 为什么挂在 Agent 上而不是塞进 AgentConfig：
        # AgentConfig 会被 config_effective() 展开成完整字典写进产物 JSON，
        # 把一个文件句柄塞进去既不可序列化，也会污染"有效配置"的语义
        # （配置应该只描述"怎么跑"，不该夹带"往哪写"）。
        # 放在 Agent 上、默认 None，则既有调用点一行都不用改。
        #
        # 为什么必须是可选的：留痕是实验需求，不是产品功能。
        # 默认 None 时本类行为与第十轮之前完全一致（286 项回归依赖这一点）。
        self.call_journal = call_journal
        # 留痕自身失败时不能静默：攒在这里，由实验脚本写进最终产物。
        self.journal_errors: list[str] = []

    # ---------------- 逐调用留痕（第十轮） ----------------

    def _journal(self, method: str, **payload) -> None:
        """把一次事件交给日志对象；没有日志就直接返回。

        【为什么失败不抛出】
        留痕的目的是"崩了也能留下东西"，如果留痕自己抛异常，
        反而会把一次本来能跑完的实验打断 —— 本末倒置。
        但也不能悄悄吞掉，所以攒进 journal_errors，由调用方报告。
        """
        journal = getattr(self, "call_journal", None)
        if journal is None:
            return
        try:
            getattr(journal, method)(**payload)
        except Exception as exc:      # noqa: BLE001 - 留痕失败要留痕，不能中断实验
            self.journal_errors.append(
                f"{method} {type(exc).__name__}: {exc}")

    # ---------------- 统一的 LLM 调用入口（预算在这里生效） ----------------

    def _call(self, budget: Budget, prompt: str, max_new_tokens: int,
              temperature: float | None, node: str, trace: AgentTrace,
              attempt: int = 0) -> tuple[str | None, object | None, str]:
        """所有 LLM 调用都必须走这里。返回 (reply|None, stats|None, 状态)。

        为什么要有这个统一入口？
        因为"预算检查"这件事一旦散落在各个节点里，就一定会漏。
        原实现里三个节点各自调 llm.chat，只有 _answer 记了账，
        于是规划/反思/校验的开销完全消失在统计之外。

        两个硬规则：
        1. 调用前先看剩余预算，不够就**根本不调**（返回 None），
           而不是调完再后悔 —— 超预算的生成已经花钱了。
        2. max_new_tokens 用剩余预算封顶，
           避免"最后一调用掉了 3 倍预算"这种越界。

        另外每次调用都写一条 `llm_call` trace（节点名、重试序号、token、耗时），
        trace 里所有 llm_call 的 token 之和必须等于最终报告的用量 ——
        这是可核对性的最低要求：说花了多少 token，就要能一笔一笔对上。
        trace 里**不存 prompt 正文**（只存长度），避免把大段文本塞进轨迹。

        【第十轮：逐调用留痕也在这里，而且必须包住异常】

        为什么必须放在这个统一入口：预算、调用、记账三件事在这里是一体的，
        留痕若放到各节点就会漏掉某一个（这是第一轮已经踩过的同款坑）。

        为什么调用**之前**就要写一条：一旦 `llm.chat()` 抛异常，
        prompt 就再也记不下来了，而"模型到底被问了什么"正是事后唯一能复盘的东西。
        第九轮的实际故障就是这样丢的：source_select 已经成功返回、
        程序已经据此选定了原文，紧接着 verify 抛异常 —— 那一整行在内存里蒸发。

        为什么异常时的 usage 记 None 而不是 0：
        0 会被读成"这次调用没花钱"，账本看着是平的、预算看着没超，
        而真实情况是**不知道**。"不知道"和"确定是 0"是两种证据强度，
        混在一起等于编数据。
        """
        if budget.remaining <= 0:
            trace.add("llm_call", caller=node, attempt=attempt, status="skipped_budget",
                      max_new_tokens=0, prompt_tokens=0, output_tokens=0,
                      elapsed=0.0, prompt_chars=len(prompt))
            # 没调用 -> usage 是**确定**的 0，不是"未知"
            self._journal("call_end", node=node, attempt=attempt, status="skipped_budget",
                          reply=None, prompt_tokens=0, output_tokens=0, elapsed=0.0,
                          usage_known=True, budget_remaining_after=budget.remaining,
                          budget_calls=budget.calls,
                          note="预算不足，未发起调用")
            return None, None, "budget_exhausted"
        cap = max(1, min(int(max_new_tokens), budget.remaining))
        self._journal("call_start", node=node, attempt=attempt, prompt=prompt,
                      max_new_tokens=cap, budget_limit=budget.limit,
                      budget_remaining_before=budget.remaining,
                      temperature_requested=temperature)
        try:
            reply, stats = self.llm.chat(prompt, max_new_tokens=cap, temperature=temperature)
        except BaseException as exc:      # noqa: BLE001 - 记录后原样抛出，不吞异常
            self._journal("call_end", node=node, attempt=attempt, status="error",
                          reply=None, prompt_tokens=None, output_tokens=None,
                          elapsed=None, usage_known=False,
                          budget_remaining_after=budget.remaining,
                          budget_calls=budget.calls,
                          error_type=type(exc).__name__, error_message=str(exc))
            raise
        budget.spend(stats)
        trace.add("llm_call", caller=node, attempt=attempt, status="ok",
                  max_new_tokens=cap,
                  prompt_tokens=int(getattr(stats, "prompt_tokens", 0) or 0),
                  output_tokens=int(getattr(stats, "output_tokens", 0) or 0),
                  elapsed=round(float(getattr(stats, "elapsed", 0.0) or 0.0), 4),
                  prompt_chars=len(prompt))
        self._journal("call_end", node=node, attempt=attempt, status="ok",
                      reply=reply,
                      prompt_tokens=int(getattr(stats, "prompt_tokens", 0) or 0),
                      output_tokens=int(getattr(stats, "output_tokens", 0) or 0),
                      elapsed=round(float(getattr(stats, "elapsed", 0.0) or 0.0), 4),
                      usage_known=True, budget_remaining_after=budget.remaining,
                      budget_calls=budget.calls)
        return reply, stats, "ok"

    # ---------------- 节点：规划 ----------------

    def _plan(self, question: str, budget: Budget, trace: AgentTrace) -> tuple[list[str], str]:
        """把用户问题改写成 1-3 条检索查询。

        规划失败**允许**退回原问题（这是唯一被允许的"乐观兜底"）：
        规划只是优化检索词，最坏情况退化为单行 RAG，不会引入幻觉。
        """
        if not self.config.enable_plan:
            return [question], PLAN_DISABLED

        prompt = PLANNER.format(question=question)
        reply, _, status = self._call(budget, prompt, max_new_tokens=256, temperature=0.0,
                                      node="plan", trace=trace)
        if reply is None:
            trace.add("plan", queries=[question], status=PLAN_SKIPPED_BUDGET,
                      note=f"预算剩余 {budget.remaining}，跳过规划")
            return [question], PLAN_SKIPPED_BUDGET

        data, _ = parse_json_object(reply)
        queries = as_str_list(data.get("queries") if data else None, max_items=3)

        if not queries:
            # 模型没给出合法结果 -> 用原问题当查询，退化为单行 RAG
            queries = [question]
            st = PLAN_FALLBACK
        else:
            st = PLAN_OK
        trace.add("plan", queries=queries, status=st)
        return queries, st

    # ---------------- 节点：检索 ----------------

    def _retrieve(self, query: str, pool: dict[str, dict]) -> list[dict]:
        """检索一次，把结果并入证据池；返回本次**新增**的条目。

        【证据池为什么是 dict 而不是 list】

        多条子查询很可能召回同一个 chunk。原实现用一个 `seen` 集合去重，
        命中过的直接丢弃 —— 但这个信号被浪费了：一个 chunk 被 3 条不同
        子查询同时召回，说明它很可能真的相关。

        所以这里改成**累加**：同一 chunk 多次命中时把各次 RRF 分数相加
        （这正是 RRF 在多路召回下的定义：Σ 1/(k + rank_i)）。
        同时记录命中次数与来源查询，方便错误分析时看"它是被哪条查询带出来的"。

        重申：这个分数是名次融合启发式，不是校准过的相关度。
        只用于排序，绝不用于阈值判断。
        """
        hits = self.retriever.search(query, top_k=self.config.top_k_per_query)
        new: list[dict] = []
        for h in hits:
            cid = h.chunk_id
            if cid in pool:
                pool[cid]["score"] += h.score
                pool[cid]["n_terms"] += 1
                pool[cid]["queries"].append(query)
                continue
            chunk = h.chunk
            # 溯源信息拼成 "文件名 · 小节标题"。
            # 只给文件名是不够的 —— 一篇 30 页的论文里，用户看到
            # "来自 XXX.pdf" 根本无法定位。带上小节标题才能真的追溯。
            title = getattr(chunk, "title", "") or ""
            fname = getattr(chunk, "file_name", "") or "unknown"
            pool[cid] = {
                "id": cid,
                "text": getattr(chunk, "text", ""),
                "source": f"{fname} · {title}" if title else fname,
                "title": title,
                "file_name": fname,
                "score": h.score,
                "n_terms": 1,
                "queries": [query],
                "bm25_rank": h.bm25_rank,
                "vector_rank": h.vector_rank,
            }
            new.append(pool[cid])
        return new

    # ---------------- 证据视图选择 ----------------

    def _select_view(self, pool: dict[str, dict], shown: set[str],
                     fresh: set[str], k: int) -> list[dict]:
        """挑出本轮要展示给 REFLECT 节点的证据。

        【第一轮修掉的 bug：反思永远看不到新证据】

        原实现写的是 `self._reflect(question, evidence[:cfg.max_evidence])`
        —— 永远取证据列表的**前 k 个**。首轮检索完列表已经有 5-15 条，
        第二轮补充检索的新证据被追加到列表**尾部**，于是反思节点
        看到的还是首轮那几条旧证据，据此判断"还是不够"。

        【第二轮修掉的残余问题：只按分数排，"低分新证据"仍会被挤掉】

        第一轮的修法是"给未展示过的证据预留 k//2 个槽位"，
        但槽位内部仍然**按分数排序**。于是当首轮候选数 > max_evidence 时，
        未展示的旧候选分数普遍高于刚召回的新证据，预留槽位被旧候选占满，
        新证据（尤其是分数低的）依然看不见 —— 而它恰恰是这一轮检索的全部意义。

        所以这里改成**分层优先**，而不是单一分数排序：

            tier 1  本轮新增（fresh）—— 不论分数，优先进入视野
                    理由：它存在的原因就是"上一轮觉得缺、专门去找的"，
                    分数低只是因为它来自不同查询，量纲本来就不可比
                    （见模块顶部关于 RRF 分数不可跨查询比较的说明）
            tier 2  从未展示过（unseen）
            tier 3  其余，按分数补齐到 k

        注意"本轮新增"和"从未展示"是两个不同的集合：
        一条证据可能从未展示过但也不是本轮新增（首轮没排进视野的旧候选）。
        """
        ranked = sorted(pool.values(), key=lambda e: e["score"], reverse=True)

        picked: list[dict] = []
        picked_ids: set[str] = set()

        def take(cands: list[dict]) -> None:
            for e in cands:
                if len(picked) >= k:
                    break
                if e["id"] in picked_ids:
                    continue
                picked.append(e)
                picked_ids.add(e["id"])

        take([e for e in ranked if e["id"] in fresh])
        take([e for e in ranked if e["id"] not in shown])
        take(ranked)
        return picked

    def _final_evidence(self, pool: dict[str, dict], sufficient_view: list[str],
                        k: int) -> list[dict]:
        """选出最终喂给 ANSWER 的证据。

        【第二轮修掉的 bug：促成"充分"判定的证据被最终回答丢掉】

        复现场景：max_evidence=2，首轮两条旧证据得分 10/9，
        追问召回两条新证据得分 0.001；反思看到新证据后判定 sufficient=true，
        但最终选择又回到"按分数截断"，只留下旧的两条 ——
        ANSWER 根本没看到让模型认为"够了"的那条新证据。

        这不只是"少看了一条"的问题：引用编号是在最终列表上重新编的，
        所以模型的判断依据和它实际能引用的证据对不上，
        是答案质量与可信度双输。

        修法：**若反思明确判定充分，就把那一轮的视图作为证据主干**，
        剩余槽位再按分数补齐。这样：
            - 判定依据一定在答案里
            - 视图内的相对顺序保持原样，引用编号与反思时看到的一致
        若循环是因为轮次/预算/无新增而结束（没有 sufficient 判定），
        则退回按分数取 top-k —— 那时没有任何"已认可"的视图可言。
        """
        if sufficient_view:
            picked = [pool[i] for i in sufficient_view if i in pool]
            ids = {e["id"] for e in picked}
            ranked = sorted(pool.values(), key=lambda e: e["score"], reverse=True)
            for e in ranked:
                if len(picked) >= k:
                    break
                if e["id"] in ids:
                    continue
                picked.append(e)
                ids.add(e["id"])
            return picked[:k]

        ranked = sorted(pool.values(), key=lambda e: e["score"], reverse=True)
        return ranked[:k]

    # ---------------- 节点：反思 ----------------

    def _reflect(self, question: str, view: list[dict], budget: Budget,
                 trace: AgentTrace, r: int) -> tuple[bool | None, str, str, bool]:
        """判断证据是否充分。

        返回 (sufficient | None, missing, follow_up, 是否成功解析)

        sufficient 为 None 表示**未知**。调用方必须按"不知道"处理，
        绝不能当成 True 去提前退出循环（这是本轮修的核心缺陷之一）。

        解析失败时做**有限重试**（max_node_retries 次），重试同样走预算。
        重试后仍然失败就返回解析失败，由调用方保守终止 ——
        而不是像原实现那样默认 {"sufficient": True} 直接放行。
        """
        view = prepare_source_evidence(view)
        ev = format_evidence([(i, model_text(e), e["source"]) for i, e in enumerate(view, 1)])
        prompt = REFLECTOR.format(question=question, evidence=ev)
        # 每次反思都要留痕：轮次、判定、缺失说明、追问、本次看到的证据 ID。
        # 第一轮只在"异常分支"记 trace，导致正常反思在轨迹里完全消失 ——
        # 事后根本无法回答"它到底看了哪些证据才说够了"。
        view_ids = [e["id"] for e in view]

        max_new = 256
        for attempt in range(self.config.max_node_retries + 1):
            reply, _, status = self._call(budget, prompt, max_new, temperature=0.0,
                                          node="reflect", trace=trace, attempt=attempt)
            if reply is None:
                trace.add("reflect", round=r, sufficient=None, missing="", follow_up="",
                          view_ids=view_ids,
                          note=f"预算剩余 {budget.remaining}，无法反思")
                return None, "", "", False

            data, pstatus = parse_json_object(reply)
            if data is None:
                if attempt < self.config.max_node_retries:
                    continue    # 有限重试，代价计入预算
                trace.add("reflect", round=r, sufficient=None, missing="", follow_up="",
                          view_ids=view_ids,
                          note=f"反思输出无法解析（{pstatus}），保守停止检索")
                return None, "", "", False

            sufficient = as_bool(data.get("sufficient"))
            if sufficient is None:
                # 字段缺失 / 类型不对 / 语义不明 —— 同样按未知处理
                if attempt < self.config.max_node_retries:
                    continue
                trace.add("reflect", round=r, sufficient=None, missing="", follow_up="",
                          view_ids=view_ids,
                          note="反思未给出可判定的 sufficient 字段，保守停止检索")
                return None, "", "", False

            missing = str(data.get("missing", "") or "")[:200]
            follow_up = as_str_list([data.get("follow_up")])
            follow_up = follow_up[0][:200] if follow_up else ""
            # 正常分支同样入 trace（这是第二轮补上的）
            trace.add("reflect", round=r, sufficient=sufficient, missing=missing,
                      follow_up=follow_up, view_ids=view_ids)
            return sufficient, missing, follow_up, True

        trace.add("reflect", round=r, sufficient=None, missing="", follow_up="",
                  view_ids=view_ids, note="反思重试后仍无有效输出")
        return None, "", "", False

    # ---------------- 节点：生成 ----------------

    def _answer(self, question: str, evidence: list[dict], budget: Budget,
                trace: AgentTrace) -> tuple[str, str]:
        """生成答案。返回 (答案, 状态)。

        预算不足时**不生成**，返回明确的说明文本。
        为什么不能硬着头皮生成：预算耗尽说明前面已经烧掉很多轮，
        此时再生成一次长答案既不可控，也会让 trace 里的统计失真。
        """
        if budget.remaining <= 0:
            trace.add("answer", status=ANSWER_SKIPPED_BUDGET, tokens=0, seconds=0.0)
            return "（已达 token 预算上限，本次未生成回答）", ANSWER_SKIPPED_BUDGET

        options = source_review_options(evidence)
        if self.config.source_quote_answers and options and formal_source_request(question):
            prompt = SOURCE_SELECTOR.format(question=question, options=format_source_options(options))
            reply, _, _ = self._call(budget,prompt,128,temperature=0.0,node='source_select',trace=trace)
            decision, parse_status = parse_json_object(reply or '')
            selected = selected_source_option(decision,options)
            trace.add('source_select',selected_id=selected['source_review_id'] if selected else None,
                      decision=decision,parse_status=parse_status)
            if selected is not None:
                answer = render_source_quote(selected)
                trace.add('answer',status=ANSWER_GENERATED,mode='reviewed_source_quote',
                          source_review_id=selected['source_review_id'],citation_encoding_invalid=False,
                          tokens=0,seconds=0.0,rendered_source_chars=len(answer))
                # 【第十轮】原文引用模式下"答案长度"和"模型输出"必须分开记：
                # 这段答案是**程序**把已核对原文渲染出来的，没有一个 token 是模型生成的。
                # 若不显式区分，读产物的人会把 rendered_source_chars 当成模型产出，
                # 进而把"程序照抄原文"包装成"模型恢复了排版"—— 那是对结果的实质性误读。
                self._journal("note", kind="rendered_source_quote", node="answer",
                              mode="reviewed_source_quote",
                              source_review_id=selected['source_review_id'],
                              evidence_number=selected.get('evidence_number'),
                              rendered_source_chars=len(answer),
                              llm_output_tokens_for_answer=0,
                              note="答案为程序渲染的已核对来源原文，模型本次未生成答案 token")
                return answer,ANSWER_GENERATED
            if budget.remaining <= 0:
                trace.add('answer',status=ANSWER_SKIPPED_BUDGET,tokens=0,seconds=0.0)
                return '（已达 token 预算上限，本次未生成回答）',ANSWER_SKIPPED_BUDGET

        ev = format_evidence([(i, model_text(e), e["source"]) for i, e in enumerate(evidence, 1)],
                             opaque_labels=self.config.opaque_citations)
        prompt = ANSWERER.format(
            question=question,
            evidence=ev,
            max_words=self.config.max_answer_words,
            cite_example=evidence_label(1) if self.config.opaque_citations else '1',
        )
        want = self.config.max_new_tokens_answer or self.config.max_answer_words * 3
        reply, stats, _ = self._call(budget, prompt, want, temperature=0.1,
                                     node="answer", trace=trace)
        opaque = self.config.opaque_citations
        original = reply or ''
        invalid_format = citation_format_invalid(original, opaque=opaque)
        trace.add("answer", status=ANSWER_GENERATED, citation_encoding_invalid=invalid_format,
                  tokens=getattr(stats, "output_tokens", 0),
                  seconds=getattr(stats, "elapsed", 0.0))
        return (normalize_evidence_citations(original) if opaque else original), ANSWER_GENERATED

    # ---------------- 节点：校验 ----------------

    def _verify(self, evidence: list[dict], answer: str, budget: Budget,
                trace: AgentTrace, question: str = "") -> tuple[str, list[str]]:
        """校验答案。返回 (LLM 节点状态, 问题列表)。

        注意这里返回的只是**模型那一票**。最终结论由 `combine_verification()`
        综合引用合法性规则后给出 —— 见那里的注释。

        三种保守处理（第二轮新增）：
            1. `issues` 字段缺失或类型非法（比如给了个字符串）-> 按解析失败处理，
               因为"模型说通过但没给出问题列表"和"模型说通过且确实没问题"
               是两种不同的可信度，不能一律当后者。
            2. `passed=true` 却带着非空 issues -> **自相矛盾**，按 failed 处理。
               模型经常一边说通过一边列问题，这种信号必须显式降级而不是静默接受。
            3. 其余解析失败/预算不足 -> 不明说通过。

        【第三轮补强：严格校验 issues 的**元素**】

        第二轮只查了"是不是列表"，元素合法性交给 as_str_list 静默过滤，
        于是 `issues=[123]` / `[null]` / `[{"reason":...}]` 全被过滤成 `[]`
        并最终 passed。现在改用 `issues_schema_error()` 逐元素校验：
        任何非法元素都算 schema 异常 -> 重试 -> 仍失败则 parse_failed（unknown）。
        空列表 `[]` 依然合法（"确实没问题"）。
        """
        if not self.config.enable_verify:
            trace.add("verify", status=VERIFY_NOT_RUN, issues=[])
            return VERIFY_NOT_RUN, []

        if budget.remaining <= 0:
            trace.add("verify", status=VERIFY_SKIPPED_BUDGET, issues=[])
            return VERIFY_SKIPPED_BUDGET, []

        # Only cited blocks may justify a cited answer. Other retrieved blocks can
        # contain the right fact while the actual citation points to the wrong paper.
        # Keep original numbering; uncited answers/refusals still see all evidence.
        cited = set(extract_citations(answer))
        numbered = [(i, e) for i, e in enumerate(evidence, 1) if not cited or i in cited]
        trace.add("verify_context", view_ids=[e["id"] for _, e in numbered],
                  citation_scope="cited_only" if cited else "all_for_uncited")
        ev = format_evidence([(i, model_text(e), e["source"]) for i, e in numbered])
        prompt = VERIFIER.format(evidence=ev, answer=answer, question=question)

        for attempt in range(self.config.max_node_retries + 1):
            reply, _, status = self._call(budget, prompt, 256, temperature=0.0,
                                          node="verify", trace=trace, attempt=attempt)
            if reply is None:
                trace.add("verify", status=VERIFY_SKIPPED_BUDGET, issues=[])
                return VERIFY_SKIPPED_BUDGET, []

            data, pstatus = parse_json_object(reply)
            # 修复留痕：解析是靠"补转义"才成功的，必须写进 trace。
            # 否则读产物的人会以为模型输出了严格 JSON，高估输出规整程度。
            repair_note = ""
            if isinstance(pstatus, str) and pstatus.startswith("repaired:"):
                n = pstatus.split(":", 1)[1]
                repair_note = (f"原始输出含 {n} 处非法 JSON 转义（模型在字符串值里写了 LaTeX），"
                               f"已按字面量修复后解析；只影响能否解析，未改写任何内容")
            if data is None:
                if attempt < self.config.max_node_retries:
                    continue
                trace.add("verify", status=VERIFY_PARSE_FAILED, issues=[], note=repair_note)
                return VERIFY_PARSE_FAILED, []

            passed = as_bool(data.get("passed"))
            raw_issues = data.get("issues")
            # issues 必须是列表才算"模型明确回答了问题清单"；
            # 缺失 / 字符串 / 数字都视为未给出有效信息。
            #
            # 为什么"缺失"也要算失败（而不是当成空列表放行）：
            # 模型说"通过"却没给出问题清单，和"通过且确认无问题"
            # 是两种不同的可信度 —— 前者往往是输出被截断或跑偏了。
            # 一律当后者会让我们把"没答完"记成"答得很完整"。
            schema_err = issues_schema_error(raw_issues, max_items=5)
            if schema_err is not None:
                if attempt < self.config.max_node_retries:
                    continue    # 重试同样计入预算
                trace.add("verify", status=VERIFY_PARSE_FAILED, issues=[],
                          note=f"issues 不符合 schema（{schema_err}），按未通过处理")
                return VERIFY_PARSE_FAILED, []

            issues = as_str_list(raw_issues, max_items=5)
            truncated = (len(raw_issues) > len(issues))

            if passed is None:
                if attempt < self.config.max_node_retries:
                    continue
                trace.add("verify", status=VERIFY_PARSE_FAILED, issues=issues,
                          note=repair_note)
                return VERIFY_PARSE_FAILED, issues

            if passed and issues:
                # passed=true 却列出问题：自相矛盾，按未通过处理
                trace.add("verify", status=VERIFY_FAILED, issues=issues,
                          note="模型判定通过但同时列出了问题，按未通过处理")
                return VERIFY_FAILED, issues

            status = VERIFY_PASSED if passed else VERIFY_FAILED
            # 结尾 note 里把"转义修复"和"issues 截断"都写上 —— 两者都是
            # "这份判定没那么干净"的证据，漏记任何一个都会让人高估可信度。
            tail_note = repair_note
            if truncated:
                tail_note = (tail_note + "；" if tail_note else "") + \
                    f"issues 被截断到前 {len(issues)} 条（原 {len(raw_issues)} 条）"
            trace.add("verify", status=status, issues=issues, note=tail_note)
            return status, issues

        trace.add("verify", status=VERIFY_PARSE_FAILED, issues=[])
        return VERIFY_PARSE_FAILED, []

    # ---------------- 主循环 ----------------

    def _stopped(self) -> bool:
        """协作式取消探测：True = 调用方希望尽快停止。

        只在节点边界被调用。没有钩子（默认）时恒为 False，
        因此既有调用点的行为与第十三轮之前完全一致。
        """
        if self._stop_check is None:
            return False
        return bool(self._stop_check())

    def _cancelled_answer(self, question: str, trace: AgentTrace, rounds: int,
                          t_start: float, budget: Budget, *,
                          answer: str = "", answer_status: str = "cancelled",
                          evidence: list[dict] | None = None) -> AgentAnswer:
        """构造"被取消"这一终态的返回值。

        为什么不抛异常：
            抛异常会让已经跑出来的 trace / 预算 / 证据全部丢失，
            调用方（服务层）就没法告诉用户"取消发生在哪一步、花了多少 token"。
            返回值的语义是"我停下来是因为你让我停"，而不是"我崩了"。

        为什么校验状态一律是 not_run 而不是 passed/unknown：
            not_run = 校验确实没有执行；unknown = 校验跑了但读不出来。
            取消属于前者。写成 unknown 会让"没做完"被误读成"质量不确定"。
        """
        evidence = evidence or []
        return AgentAnswer(
            answer=answer,
            evidence=evidence,
            rounds_used=rounds,
            total_tokens=budget.prompt_tokens + budget.output_tokens,
            prompt_tokens=budget.prompt_tokens,
            output_tokens=budget.output_tokens,
            elapsed=time.time() - t_start,
            passed_verification=False,          # 取消绝不等同通过
            verification_status=OVERALL_NOT_RUN,
            llm_verification_status=VERIFY_NOT_RUN,
            citation_validity=citation_validity(answer, len(evidence)),
            verification_fail_reasons=["cancelled_by_client"],
            issues=[],
            citation_issues=citation_issues(answer, len(evidence)),
            stop_reason=STOP_CANCELLED,
            answer_status=answer_status,
            plan_status="cancelled",
            trace=trace,
        )

    def run(self, question: str) -> AgentAnswer:
        cfg = self.config
        trace = AgentTrace()
        # 每次 run 新建预算对象 —— 杜绝跨题串用状态
        budget = Budget(cfg.token_budget)
        t_start = time.time()

        pool: dict[str, dict] = {}      # chunk_id -> 证据条目（分数为多查询累加）
        shown: set[str] = set()         # 曾经展示给反思节点的 chunk_id
        fresh: set[str] = set()         # **本轮新增**、还没被反思看过的 chunk_id
        sufficient_view: list[str] = [] # 被判定为"充分"那一轮的视图 ID（顺序保留）

        # ===== 节点 1: PLAN =====
        # 协作式取消点 1：还没花任何 token，直接停。
        if self._stopped():
            trace.add("cancelled", at="before_plan", note="客户端取消了任务，未执行任何检索")
            return self._cancelled_answer(question, trace, 0, t_start, budget)

        queries, plan_status = self._plan(question, budget, trace)

        # ===== 节点 2: RETRIEVE（首轮）=====
        for q in queries:
            new = self._retrieve(q, pool)
            trace.add("retrieve", query=q, n_new=len(new), n_pool=len(pool))

        # ===== 节点 3: REFLECT 循环 =====
        rounds = 0
        stop_reason = ""
        tried: set[str] = set(queries)   # 已试过的查询，拦截"反思退化"

        if cfg.max_rounds <= 0:
            stop_reason = STOP_REFLECTION_DISABLED
        else:
            for r in range(1, cfg.max_rounds + 1):
                rounds = r

                # 协作式取消点 2：每一轮反思开始之前。
                # 放在这里而不是循环外，是因为一轮反思 = 一次 LLM 调用，
                # 能在它开始前停下就有实际意义（省掉整轮 token）。
                if self._stopped():
                    trace.add("cancelled", at=f"reflect_round_{r}",
                              note="客户端取消了任务，停止后续反思")
                    return self._cancelled_answer(question, trace, rounds, t_start, budget)

                if budget.remaining <= 0:
                    stop_reason = STOP_TOKEN_BUDGET
                    trace.add("reflect", round=r, sufficient=None, missing="",
                              note=f"预算已耗尽（输出 {budget.output_tokens}/{budget.limit}），停止反思")
                    break

                view = self._select_view(pool, shown, fresh, cfg.max_evidence)
                sufficient, missing, follow_up, ok = self._reflect(
                    question, view, budget, trace, r
                )
                # 这一轮展示过的条目记下来；本轮新增的已经看过了，清空
                shown.update(e["id"] for e in view)
                fresh.clear()

                if not ok:
                    stop_reason = STOP_REFLECT_FAILED
                    break

                if sufficient is True:
                    # 记住"是哪一组证据让模型说够了" —— 最终回答必须包含它们
                    sufficient_view = [e["id"] for e in view]
                    stop_reason = STOP_SUFFICIENT
                    break

                # 终止：没给出有效追问，再检索也没意义
                if not follow_up:
                    trace.add("reflect", round=r, sufficient=False,
                              missing="模型未给出追问查询，提前退出")
                    stop_reason = STOP_NO_FOLLOWUP
                    break

                # 终止：反思退化 —— 追问和之前试过的一样
                #
                # 这是实测发现的真实问题：问"论文用了什么深度学习模型"时，
                # 7B 模型连续 3 轮生成的追问都是原问题的近似复述，
                # 检索命中 0 条却仍要跑满 max_rounds，白白浪费 2 轮时间和几千 token。
                # 根因是模型缺乏"换个角度想"的元认知能力，只会重新表述问题。
                if follow_up in tried:
                    trace.add("reflect", round=r, sufficient=False,
                              missing="追问查询与之前重复，已无新角度可试，提前退出")
                    stop_reason = STOP_DUPLICATE_QUERY
                    break
                tried.add(follow_up)

                # 【第二轮修掉的问题：最后一轮还做补充检索】
                # 补充检索的唯一目的是"给下一轮反思看"。如果这一轮已经是最后一轮，
                # 或者预算已经不够再反思一次，那么检索出来的东西
                # 永远不会进入反思视野 —— 纯属浪费。
                # 复现：max_rounds=1 且反思返回 false+follow_up 时，
                # 旧实现仍然执行了第二次检索（queries=['initial','supplementary']）。
                if r >= cfg.max_rounds:
                    stop_reason = STOP_MAX_ROUNDS
                    break
                if budget.remaining <= 0:
                    stop_reason = STOP_TOKEN_BUDGET
                    trace.add("reflect", round=r, sufficient=False, missing="",
                              follow_up=follow_up,
                              note="输出预算已耗尽，无法再进行下一轮反思，停止检索")
                    break

                # 补充检索
                new = self._retrieve(follow_up, pool)
                fresh.update(e["id"] for e in new)

                # 【第二轮修掉的措辞不一致】
                # 终止原因叫 no_new_evidence，但重复命中的 chunk 分数确实被
                # 累加了（_retrieve 里 pool[cid]["score"] += h.score，
                # n_terms 也会 +1）。也就是说"没有新增条目"≠"什么都没变"。
                # 说明必须把两件事都讲清楚，否则看 trace 的人会以为
                # 这次检索完全没生效 —— 所以下面这条 note 不是可选的装饰。
                note = ""
                if not new:
                    bumped = sum(1 for e in pool.values() if e["n_terms"] > 1)
                    note = ("没有召回新条目；重复命中的证据分数已累加"
                            f"（{bumped} 条 n_terms>1）")
                trace.add("retrieve", query=follow_up, n_new=len(new),
                          n_pool=len(pool), note=note)

                if not new:
                    stop_reason = STOP_NO_NEW_EVIDENCE
                    break

        if not stop_reason:
            stop_reason = STOP_MAX_ROUNDS

        # 最终证据：以"被判定充分的那一轮视图"为主干，其余按分数补齐。
        # 详见 _final_evidence() 的注释（第二轮修复）。
        evidence = prepare_source_evidence(self._final_evidence(pool, sufficient_view, cfg.max_evidence))

        # ===== 节点 4: ANSWER =====
        # 协作式取消点 3：检索已经做完，生成是最贵的一步，这是最后一个能省的地方。
        if self._stopped():
            trace.add("cancelled", at="before_answer", note="客户端取消了任务，未生成答案")
            return self._cancelled_answer(question, trace, rounds, t_start, budget)

        answer, answer_status = self._answer(question, evidence, budget, trace)

        # ===== 节点 5: VERIFY（只报告，不修订）=====
        # 协作式取消点 4：答案已经生成出来了。
        #
        # 这里有个必须讲清楚的取舍：取消发生在 VERIFY 之前时，答案本身是完整的，
        # 丢掉它等于白烧了前面所有 token。所以这里保留答案，
        # 但校验状态强制为 not_run —— 保留答案 ≠ 声明它通过了校验。
        if self._stopped():
            trace.add("cancelled", at="before_verify",
                      note="客户端取消了任务，已生成的答案予以保留，但校验未执行")
            return self._cancelled_answer(question, trace, rounds, t_start, budget,
                                          answer=answer, answer_status=answer_status,
                                          evidence=evidence)

        llm_status, issues = self._verify(evidence, answer, budget, trace, question=question)

        # ===== 零成本引用编号检查（无论是否启用 LLM 校验都跑）=====
        cites = citation_issues(answer, len(evidence))
        cite_state = citation_validity(answer, len(evidence))
        trace.add("citation_check", n_citations=len(extract_citations(answer)),
                  validity=cite_state, issues=cites)

        # ===== 综合结论：规则判定优先于模型的自我认可 =====
        # 只要查出越界引用，整体就不能是 passed —— 查出来和正确处理是两件事。
        overall, fail_reasons = combine_verification(llm_status, cite_state, issues)
        if any(step.get('node')=='answer' and step.get('citation_encoding_invalid') for step in trace.steps):
            overall = OVERALL_FAILED
            fail_reasons.append('citation_format')
        trace.add("verification_summary", llm_status=llm_status,
                  citation_validity=cite_state, overall=overall,
                  fail_reasons=fail_reasons)

        return AgentAnswer(
            answer=answer,
            evidence=evidence,
            rounds_used=rounds,
            total_tokens=budget.prompt_tokens + budget.output_tokens,
            prompt_tokens=budget.prompt_tokens,
            output_tokens=budget.output_tokens,
            elapsed=time.time() - t_start,
            passed_verification=(overall == OVERALL_PASSED),
            verification_status=overall,
            llm_verification_status=llm_status,
            citation_validity=cite_state,
            verification_fail_reasons=fail_reasons,
            issues=issues,
            citation_issues=cites,
            stop_reason=stop_reason,
            answer_status=answer_status,
            plan_status=plan_status,
            trace=trace,
        )
