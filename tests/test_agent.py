# Agent 层回归测试（第 2/3/4/5 条审核要求）

"""
用 fake LLM + fake retriever 覆盖第一轮审核发现的缺陷。

【为什么必须用 fake，而不是真的加载 7B 模型】

    1. 真的跑一次要几十秒到几分钟，回归测试必须秒级
    2. 真模型输出不稳定，测试会变 flaky
    3. 最重要的是：这里要测的是**控制流**（预算、证据可见性、解析失败兜底），
       不是模型能力。用可控的假输出才能精确断言"在这种输出下系统该怎么做"。

每个测试都对应一个真实复现过的缺陷，注释里写明"原来会怎样"。
断言一律写死具体值，不为通过而放宽。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from docmind.agent.react import (
    ANSWER_GENERATED,
    ANSWER_SKIPPED_BUDGET,
    CITE_INVALID,
    OVERALL_FAILED,
    OVERALL_NOT_RUN,
    OVERALL_PASSED,
    OVERALL_UNKNOWN,
    STOP_DUPLICATE_QUERY,
    STOP_NO_NEW_EVIDENCE,
    STOP_REFLECT_FAILED,
    STOP_REFLECTION_DISABLED,
    STOP_SUFFICIENT,
    STOP_TOKEN_BUDGET,
    VERIFY_NOT_RUN,
    VERIFY_PARSE_FAILED,
    VERIFY_PASSED,
    VERIFY_SKIPPED_BUDGET,
    AgentConfig,
    DocMindAgent,
    as_bool,
    as_str_list,
    citation_issues,
    parse_json_object,
)

# 【第二轮起的字段语义变化，不要退回旧写法】
#   verification_status      = 综合结论（引用规则 + 模型那一票），对外唯一口径
#   llm_verification_status  = 模型校验节点自己的状态
# 旧测试里写的 `verification_status == VERIFY_*` 是在断言模型那一票，
# 现在必须断言 llm_verification_status，同时补上综合结论不能是 passed。


# ---------------------------------------------------------------------------
# Fake 基础设施
# ---------------------------------------------------------------------------

@dataclass
class FakeStats:
    """鸭子类型的 GenerationStats，避免测试导入 torch。"""
    prompt_tokens: int = 40
    output_tokens: int = 100
    elapsed: float = 0.01


class FakeLLM:
    """按脚本顺序返回回复的假 LLM，并记录每一次调用。"""

    def __init__(self, replies: list[str], output_tokens: int = 100):
        self.replies = list(replies)
        self.output_tokens = output_tokens
        self.calls: list[dict] = []      # [{prompt, max_new_tokens, temperature}]

    def chat(self, prompt, max_new_tokens=None, temperature=None):
        self.calls.append({
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        })
        reply = self.replies.pop(0) if self.replies else ""
        return reply, FakeStats(output_tokens=self.output_tokens)

    @property
    def n_calls(self) -> int:
        return len(self.calls)

    def prompt_of(self, i: int) -> str:
        return self.calls[i]["prompt"]


@dataclass
class FakeChunk:
    chunk_id: str
    text: str
    file_name: str = "paper.pdf"
    title: str = ""
    page: int = 1


@dataclass
class FakeHit:
    chunk_id: str
    chunk: FakeChunk
    score: float = 0.5
    bm25_rank: int | None = 1
    vector_rank: int | None = 1


class FakeRetriever:
    """按 query 返回预设 chunk 列表的假检索器。"""

    def __init__(self, script: dict[str, list[str]], default: list[str] | None = None):
        self.script = script
        self.default = default or []
        self.calls: list[str] = []

    def search(self, query, top_k=5):
        self.calls.append(query)
        ids = self.script.get(query, self.default)
        return [
            FakeHit(cid, FakeChunk(cid, f"TEXT_OF_{cid} 内容若干"),
                    score=1.0 - 0.01 * i)
            for i, cid in enumerate(ids[:top_k])
        ]

    @property
    def n_calls(self) -> int:
        return len(self.calls)


def mk_agent(llm, retriever, **cfg) -> DocMindAgent:
    return DocMindAgent(retriever, llm, AgentConfig(**cfg))


# ---------------------------------------------------------------------------
# 第 4 条：JSON 解析与字段类型校验
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("true", True), ("True", True), ("yes", True), ("1", True),
    ("false", False), ("FALSE", False), ("no", False), ("0", False),
])
def test_as_bool_strings(raw, expected):
    """字符串必须按语义解析。

    bool("false") == True 是 Python 的经典陷阱，
    原实现正是这么写的，导致模型输出 sufficient:"false" 被当成"充分"。
    """
    assert as_bool(raw) is expected


@pytest.mark.parametrize("raw", ["maybe", "", "不知道", None, 2, -1, 0.5, [], {}])
def test_as_bool_unknown_returns_none(raw):
    """无法判定时返回 None，绝不能默认 True。

    调用方拿到 None 必须按"未知"处理，这是本轮的核心修复之一。
    """
    assert as_bool(raw) is None


def test_as_bool_native_types():
    assert as_bool(True) is True
    assert as_bool(False) is False
    assert as_bool(1) is True
    assert as_bool(0) is False


def test_as_str_list_filters_non_strings():
    """queries:[123] 这种输出不能变成检索词。"""
    assert as_str_list([123, "ok", None, "  ", "另一个"]) == ["ok", "另一个"]
    assert as_str_list(None) == []
    assert as_str_list("not a list") == []
    assert as_str_list({"a": "b"}) == []
    assert as_str_list(["a", "b", "c"], max_items=2) == ["a", "b"]


def test_parse_json_object_rejects_non_dict():
    """合法 JSON 不等于合法结果：[] / null / 数字都不是对象。"""
    assert parse_json_object("[]") == (None, "not_object")
    assert parse_json_object("null") == (None, "not_object")
    assert parse_json_object("123") == (None, "not_object")
    assert parse_json_object('"true"') == (None, "not_object")
    assert parse_json_object("") == (None, "empty")


def test_parse_json_object_extracts_from_noise():
    data, status = parse_json_object('好的，结果是：```json\n{"sufficient": false}\n```')
    assert status in ("ok", "regex")
    assert data["sufficient"] is False


def test_citation_issues_detects_out_of_range():
    """零成本检查：引用了不存在的编号。"""
    assert citation_issues("结论见 [1] 和 [3]", 3) == []
    issues = citation_issues("结论见 [7]", 3)
    assert issues and "7" in issues[0]


# ---------------------------------------------------------------------------
# 第 2 条：token 预算
# ---------------------------------------------------------------------------

def test_budget_zero_makes_no_llm_call():
    """预算为 0 时一次 LLM 都不能调，且不能凭空生成答案。

    原实现里预算检查形同虚设（tokens_used 恒为 0），
    即使 budget=0 也会照常规划、反思、生成。
    """
    llm = FakeLLM(['{"queries": ["Q"]}', '{"sufficient": true}', "答案"])
    retr = FakeRetriever({"Q": ["A"]})
    agent = mk_agent(llm, retr, token_budget=0, max_rounds=2, enable_verify=True)

    res = agent.run("问题？")
    assert llm.n_calls == 0, "预算为 0 不应发起任何 LLM 调用"
    assert res.answer_status == ANSWER_SKIPPED_BUDGET
    assert res.stop_reason == STOP_TOKEN_BUDGET
    assert res.llm_verification_status == VERIFY_SKIPPED_BUDGET
    assert res.verification_status == OVERALL_UNKNOWN
    assert res.passed_verification is False, "没执行校验绝不等于通过"


def test_budget_caps_max_new_tokens():
    """剩余预算不足时，单次调用的上限必须被压到剩余量。"""
    llm = FakeLLM(['{"queries": ["Q"]}', '{"sufficient": true}', "答案"],
                  output_tokens=100)
    retr = FakeRetriever({"Q": ["A"]})
    # 预算 150：第 1 次调用后剩 50，第 2 次最多只能要 50
    agent = mk_agent(llm, retr, token_budget=150, max_rounds=2, enable_verify=False)

    res = agent.run("问题？")
    # 规划节点想生成 256 token，但总预算只有 150 -> 首次调用就被压到 150
    assert llm.calls[0]["max_new_tokens"] == 150
    # 第 1 次调用花掉 100，只剩 50 -> 第 2 次最多只能要 50
    assert llm.calls[1]["max_new_tokens"] == 50
    assert res.output_tokens == 200


def test_budget_accumulates_all_nodes():
    """规划/反思/生成/校验的 token 都要算进去。

    原实现只累加了 _answer 的返回值，规划/反思/校验的 stats 被丢弃，
    于是 total_tokens 严重低估真实开销。
    """
    llm = FakeLLM([
        '{"queries": ["Q"]}',                                  # plan
        '{"sufficient": false, "follow_up": "F"}',             # reflect r1
        '{"sufficient": true}',                                # reflect r2
        "答案 [1]",                                             # answer
        '{"passed": true, "issues": []}',                      # verify
    ], output_tokens=100)
    retr = FakeRetriever({"Q": ["A"], "F": ["B"]})
    agent = mk_agent(llm, retr, token_budget=5000, max_rounds=3, enable_verify=True)

    res = agent.run("问题？")
    assert llm.n_calls == 5, "规划1 + 反思2 + 生成1 + 校验1"
    assert res.output_tokens == 500
    assert res.prompt_tokens == 5 * 40
    assert res.total_tokens == 500 + 200
    assert res.llm_verification_status == VERIFY_PASSED
    assert res.verification_status == OVERALL_PASSED


def test_verify_skipped_when_budget_runs_out():
    """预算恰好在校验前耗尽：校验状态必须是 skipped_budget，不是 passed。"""
    llm = FakeLLM([
        '{"queries": ["Q"]}',
        '{"sufficient": true}',
        "答案",
        '{"passed": true, "issues": []}',
    ], output_tokens=100)
    retr = FakeRetriever({"Q": ["A"]})
    # 预算 300：plan(100) + reflect(100) + answer(100) 后用尽
    agent = mk_agent(llm, retr, token_budget=300, max_rounds=2, enable_verify=True)

    res = agent.run("问题？")
    assert llm.n_calls == 3, "第 4 次（校验）不该发生"
    assert res.llm_verification_status == VERIFY_SKIPPED_BUDGET
    assert res.verification_status == OVERALL_UNKNOWN
    assert res.passed_verification is False


def test_repeated_runs_do_not_share_budget():
    """同一个 Agent 实例跑两题，第二题的预算必须是全新的。

    如果 Budget 挂在 self 上，第二题会带着第一题的消耗继续判预算，
    表现为"越往后跑质量越差"，且极难排查。
    """
    llm = FakeLLM(['{"queries": ["Q"]}', '{"sufficient": true}', "答案"] * 2,
                  output_tokens=100)
    retr = FakeRetriever({"Q": ["A"]})
    agent = mk_agent(llm, retr, token_budget=300, max_rounds=2, enable_verify=False)

    r1 = agent.run("第一题？")
    r2 = agent.run("第二题？")
    assert r1.output_tokens == r2.output_tokens == 300
    assert r2.stop_reason == STOP_SUFFICIENT, "第二题不应被上一题的消耗误判为超预算"


# ---------------------------------------------------------------------------
# 第 3 条：反思必须能看到新证据
# ---------------------------------------------------------------------------

def test_reflection_round_two_sees_new_evidence():
    """第二轮反思的输入里必须包含补充检索到的新证据。

    原实现写死 `evidence[:max_evidence]`，永远取最前面的几条旧证据，
    于是新召回的内容在反思阶段根本不可见 —— 这也是实测中
    "第二轮反思后命中数没有提升"的直接原因。
    """
    llm = FakeLLM([
        '{"sufficient": false, "missing": "缺X", "follow_up": "F"}',
        '{"sufficient": true}',
        "答案 [1]",
    ])
    # 首轮召回 4 条（正好等于 max_evidence），补充检索再召回 2 条全新的
    retr = FakeRetriever({"问题？": ["A", "B", "C", "D"], "F": ["E", "G"]})
    agent = mk_agent(llm, retr, max_rounds=2, max_evidence=4,
                     enable_plan=False, enable_verify=False)

    res = agent.run("问题？")
    round2_prompt = llm.prompt_of(1)      # 第 2 次 LLM 调用 = 第 2 轮反思
    assert "TEXT_OF_E" in round2_prompt, "第二轮反思看不到新证据 E"
    assert "TEXT_OF_G" in round2_prompt, "第二轮反思看不到新证据 G"
    assert "TEXT_OF_A" in round2_prompt, "旧证据也不该被全部挤掉"
    assert res.stop_reason == STOP_SUFFICIENT


def test_stop_when_no_new_evidence():
    """追问检索回来的全是旧 chunk 时，不要无意义地再反思。"""
    llm = FakeLLM([
        '{"sufficient": false, "follow_up": "F"}',
        '{"sufficient": true}',
        "答案",
    ])
    retr = FakeRetriever({"问题？": ["A", "B"], "F": ["A", "B"]})   # 追问召回重复
    agent = mk_agent(llm, retr, max_rounds=3, enable_plan=False, enable_verify=False)

    res = agent.run("问题？")
    assert res.stop_reason == STOP_NO_NEW_EVIDENCE
    assert retr.n_calls == 2, "无新增证据后不应再检索"
    assert llm.n_calls == 2, "无新增证据后不应再调反思"


def test_stop_on_duplicate_query():
    """追问与已试过的查询重复（反思退化）时提前退出。"""
    llm = FakeLLM([
        '{"sufficient": false, "follow_up": "问题？"}',   # 追问 = 原问题
        "答案",
    ])
    retr = FakeRetriever({"问题？": ["A", "B"]})
    agent = mk_agent(llm, retr, max_rounds=3, enable_plan=False, enable_verify=False)

    res = agent.run("问题？")
    assert res.stop_reason == STOP_DUPLICATE_QUERY
    assert retr.n_calls == 1


# ---------------------------------------------------------------------------
# 第 4 条：解析失败不能默认放行
# ---------------------------------------------------------------------------

def test_reflect_parse_failure_stops_conservatively():
    """反思输出解析不出来 -> 保守停止检索，且不能当成"证据充分"。"""
    # 三次回复：反思、反思重试（各失败一次）、生成
    llm = FakeLLM(["这段话不是 JSON，模型跑偏了", "还是不是 JSON", "答案"])
    retr = FakeRetriever({"问题？": ["A"]})
    agent = mk_agent(llm, retr, max_rounds=3, enable_plan=False, enable_verify=False,
                     max_node_retries=1)

    res = agent.run("问题？")
    assert res.stop_reason == STOP_REFLECT_FAILED
    assert retr.n_calls == 1, "解析失败后不应继续检索"
    assert llm.n_calls == 3, "反思失败后应有限重试一次，然后进入生成"
    assert res.answer == "答案"   # 仍然会基于已有证据生成，但不再盲检索


def test_reflect_string_false_is_not_sufficient():
    """sufficient:"false" 必须被识别为"不充分"。"""
    llm = FakeLLM(['{"sufficient": "false", "follow_up": "F"}',
                   '{"sufficient": "true"}', "答案"])
    retr = FakeRetriever({"问题？": ["A"], "F": ["B"]})
    agent = mk_agent(llm, retr, max_rounds=2, enable_plan=False, enable_verify=False)

    res = agent.run("问题？")
    assert retr.n_calls == 2, "sufficient:'false' 应触发补充检索"
    assert res.rounds_used == 2


def test_verify_parse_failure_is_not_a_pass():
    """校验输出乱七八糟 -> 状态必须是 parse_failed，不能等同通过。"""
    llm = FakeLLM(['{"queries": ["Q"]}', '{"sufficient": true}', "答案 [1]",
                   "校验器坏了，输出不是 JSON"])
    retr = FakeRetriever({"Q": ["A"]})
    agent = mk_agent(llm, retr, max_rounds=2, enable_verify=True, max_node_retries=0)

    res = agent.run("问题？")
    assert res.llm_verification_status == VERIFY_PARSE_FAILED
    assert res.verification_status == OVERALL_UNKNOWN
    assert res.passed_verification is False


def test_answer_out_of_range_citation_is_reported():
    """答案引用了不存在的编号，零成本检查必须报出来。

    【第二轮加强】第一轮只断言"查出来了"。但审核方指出：
    查出来 ≠ 正确处理 —— 综合结论也必须因此不为 passed。
    """
    llm = FakeLLM(['{"queries": ["Q"]}', '{"sufficient": true}', "结论见 [9]"])
    retr = FakeRetriever({"Q": ["A", "B"]})
    agent = mk_agent(llm, retr, max_rounds=2, enable_verify=False)

    res = agent.run("问题？")
    assert res.citation_issues, "引用 [9] 但只有 2 条证据，应当被查出"
    assert res.citation_validity == CITE_INVALID
    assert res.verification_status == OVERALL_FAILED
    assert res.passed_verification is False


# ---------------------------------------------------------------------------
# 第 5 条：真正的单次 RAG 基线
# ---------------------------------------------------------------------------

def test_single_pass_baseline_calls_llm_once():
    """基线：不规划 / 不反思 / 不校验 —— 只应该有一次 LLM 调用（生成）。

    原实现的"0 轮基线"是 max_rounds=1 且仍然规划，
    实际跑的是"规划 + 一次检索"，根本不是单行 RAG，消融结论不成立。
    """
    llm = FakeLLM(["答案 [1]"])
    retr = FakeRetriever({"问题？": ["A", "B"]})
    agent = mk_agent(llm, retr, enable_plan=False, max_rounds=0,
                     enable_verify=False)

    res = agent.run("问题？")
    assert llm.n_calls == 1, f"基线只应调用一次 LLM，实际 {llm.n_calls} 次"
    assert res.rounds_used == 0
    assert res.stop_reason == STOP_REFLECTION_DISABLED
    assert res.llm_verification_status == VERIFY_NOT_RUN
    assert res.verification_status == OVERALL_NOT_RUN
    assert res.plan_status == "disabled"
    assert retr.calls == ["问题？"], "基线必须直接用原问题检索，不做查询改写"


def test_agent_with_plan_and_rounds_calls_more():
    """对照组：开启规划与反思后，调用次数显著多于基线（证明两者确实不同）。"""
    llm = FakeLLM([
        '{"queries": ["Q1", "Q2"]}',
        '{"sufficient": true}',
        "答案",
    ])
    retr = FakeRetriever({"Q1": ["A"], "Q2": ["B"]})
    agent = mk_agent(llm, retr, enable_plan=True, max_rounds=1, enable_verify=False)

    res = agent.run("问题？")
    assert llm.n_calls == 3       # 规划 + 反思 + 生成
    assert retr.n_calls == 2      # 两条子查询各检索一次
    assert res.rounds_used == 1
