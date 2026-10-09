"""同骨架单写者 S0/S1 搜索协议（KAGGLE-19 §2/§4，KAGGLE-22 §3.4/§4）。

关键设计（决定了四臂消融能不能成立）：

- **S0 与 S1 必须共用同一个骨架**（同一 `SearchSession`、同一工具白名单、同一预算计量）。
  S0 = 冻结的原始搜索内容"搬进同一骨架"，S1 = 新版四步五阶段协议。
  只有这样才能把「搜索协议」的贡献和「LoRA」的贡献分开。
- 首轮四臂统一**单个 LlmAgent、官方工具、不带强制分析员**；拓扑固定后才能拆分贡献。
- 预算：搜索类调用 ≤22、升级 ≤2 轮、首个有证据候选 ≤6 次调用、
  单次输出目标 ≤1500 字符（硬上限 5000）、≥55% 时间为软收敛提示。
- **证据不足时允许明确失败/空补丁**，不因预算强迫盲改。
- shell 参数必须安全转义，**不得把 issue 文本直接插值进命令**。
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..common.canonical import sha256_json
from ..common.errors import PolicyViolation
from .lexsearch import HARD_OUTPUT_CHARS, TARGET_OUTPUT_CHARS

#: 预算常量（KAGGLE-22 §4.1）。
SEARCH_CALL_BUDGET = 22
ESCALATION_BUDGET = 2
CALLS_TO_FIRST_TARGET = 6
SOFT_CONVERGENCE_RATIO = 0.55
RESERVE_TOOL_CALLS = 25  # 剩余 tool_calls ≤ 25 时必须进入编辑

#: 升级阶梯（便宜 → 贵）。L4/L5 不在首轮四臂内。
LADDER = ("L1", "L2", "L3")

#: 触发信号（任一成立即升级，不靠模型的"感觉"）。
TRIGGER_IDS = ("E-a", "E-b", "E-c", "E-d", "E-e")


@dataclass
class SearchBudget:
    total_tool_calls: int = 100
    max_time_minutes: int = 4

    def remaining(self, used_tool_calls: int) -> int:
        return max(0, self.total_tool_calls - used_tool_calls)


@dataclass
class SearchSession:
    """单写者会话状态：S0/S1 共用。"""

    task_id: str
    budget: SearchBudget = field(default_factory=SearchBudget)
    tool_calls_used: int = 0
    search_calls_used: int = 0
    calls_to_first_candidate: int | None = None
    escalation_rounds: int = 0
    elapsed_ratio: float = 0.0
    candidates: list[dict] = field(default_factory=list)
    chosen: dict | None = None
    events: list[dict] = field(default_factory=list)
    output_flags: list[str] = field(default_factory=list)
    escalate_to: str | None = None

    # ---- 计量 ----

    def note_call(self, tool_name: str, output_chars: int = 0, bills: bool = True) -> dict:
        """记录一次工具调用；搜索类调用单独计数并检查预算。"""
        if bills:
            self.tool_calls_used += 1
        if tool_name in ("run_command", "read_file", "get_code_neighbors", "search_similar_code", "get_code_subgraph"):
            self.search_calls_used += 1
            if self.search_calls_used > SEARCH_CALL_BUDGET:
                raise PolicyViolation(
                    "search_budget_exceeded",
                    "搜索调用 %d 次超过预算 %d" % (self.search_calls_used, SEARCH_CALL_BUDGET),
                )
        if output_chars > HARD_OUTPUT_CHARS:
            # 超限意味着官方会按字母序截断，信息被系统性扭曲 —— 必须记 flag。
            self.output_flags.append("hard_truncation_%d" % output_chars)
        elif output_chars > TARGET_OUTPUT_CHARS:
            self.output_flags.append("over_target_%d" % output_chars)
        event = {
            "tool_name": tool_name,
            "tool_calls_used": self.tool_calls_used,
            "search_calls_used": self.search_calls_used,
            "output_chars": output_chars,
        }
        self.events.append(event)
        return event

    def note_candidate(self, candidate: Mapping) -> None:
        self.candidates.append(dict(candidate))
        if self.calls_to_first_candidate is None:
            self.calls_to_first_candidate = self.search_calls_used

    def note_chosen(self, candidate: Mapping) -> None:
        self.chosen = dict(candidate)

    # ---- 触发信号 ----

    def signals(self, ctx: Mapping | None = None) -> list[str]:
        """按 KAGGLE-22 §4.2 的判据计算触发信号。"""
        ctx = dict(ctx or {})
        out: list[str] = []
        if ctx.get("rounds_without_src_hit", 0) >= 2:
            out.append("E-a")
        if ctx.get("candidate_behaviour_mismatch"):
            out.append("E-b")
        if ctx.get("behaviour_already_correct"):
            out.append("E-c")
        if self.search_calls_used >= 8 and self.chosen is None:
            out.append("E-d")
        if ctx.get("candidate_has_no_caller"):
            out.append("E-e")
        return out

    # ---- 阶梯 ----

    def next_rung(self) -> str | None:
        if self.escalation_rounds >= ESCALATION_BUDGET:
            return None
        if self.escalation_rounds >= len(LADDER):
            return None
        return LADDER[self.escalation_rounds]

    def escalate(self, ctx: Mapping | None = None) -> str:
        rung = self.next_rung()
        if rung is None:
            raise PolicyViolation(
                "escalation_exhausted",
                "升级超过 %d 轮：这更可能是理解问题而不是检索问题" % ESCALATION_BUDGET,
            )
        triggers = self.signals(ctx)
        if not triggers:
            raise PolicyViolation(
                "escalation_without_trigger", "没有触发信号时不得升级（不靠模型的「感觉」）"
            )
        self.escalation_rounds += 1
        self.escalate_to = rung
        self.events.append({"escalate": rung, "triggers": triggers})
        return rung

    # ---- 停止条件 ----

    def must_edit(self) -> bool:
        """显式停止条件：预算或时间到 → 必须收敛到当前最优候选。"""
        if self.chosen is not None:
            return True
        if self.search_calls_used >= SEARCH_CALL_BUDGET:
            return True
        if self.budget.remaining(self.tool_calls_used) <= RESERVE_TOOL_CALLS:
            return True
        return self.elapsed_ratio >= SOFT_CONVERGENCE_RATIO

    def must_fail_closed(self) -> bool:
        """证据不足时允许明确失败/空补丁，不因预算强迫盲改。"""
        return self.chosen is None and self.next_rung() is None and self.must_edit()

    def next_action(self) -> str:
        if self.chosen is not None:
            return "edit"
        if self.calls_to_first_candidate is None and self.search_calls_used >= CALLS_TO_FIRST_TARGET:
            return "escalate" if self.next_rung() else "fail"
        if self.next_rung() is not None and self.search_calls_used >= 8:
            return "escalate"
        if self.must_edit():
            return "fail"
        return "search"

    # ---- 快照 ----

    def snapshot(self) -> dict:
        return {
            "task_id": self.task_id,
            "tool_calls_used": self.tool_calls_used,
            "search_calls_used": self.search_calls_used,
            "search_share": self.search_calls_used / float(max(1, self.tool_calls_used)),
            "calls_to_first_candidate": self.calls_to_first_candidate,
            "escalation_rounds": self.escalation_rounds,
            "escalate_to": self.escalate_to,
            "chosen": self.chosen,
            "candidate_count": len(self.candidates),
            "output_flags": list(self.output_flags),
            "next_action": self.next_action(),
            "events_sha256": sha256_json(self.events),
        }


class SearchPolicy:
    """策略接口：S0/S1 必须实现同一组方法，保证骨架一致。"""

    name = "base"

    def plan(self, session: SearchSession, ctx: Mapping) -> dict:
        raise NotImplementedError

    def rank_candidates(self, session: SearchSession) -> list[dict]:
        raise NotImplementedError


class S0Policy(SearchPolicy):
    """冻结基线：原始搜索内容搬进同骨架（不剔除测试候选、单轮查询、无升级阶梯）。"""

    name = "S0"

    def plan(self, session: SearchSession, ctx: Mapping) -> dict:
        if session.search_calls_used == 0:
            return {"action": "search", "channel": "C1", "note": "单轮字面检索"}
        if session.candidates and session.chosen is None:
            return {"action": "read", "channel": "read_file"}
        if session.chosen is not None:
            return {"action": "edit", "channel": "submit_patch"}
        return {"action": "fail", "channel": "none", "note": "S0 无升级阶梯"}

    def rank_candidates(self, session: SearchSession) -> list[dict]:
        return sorted(session.candidates, key=lambda c: (-c.get("hits", 0), c.get("path", "")))


class S1Policy(SearchPolicy):
    """新版：四步五阶段，候选剔除测试文件 + 规则排序 + 升级阶梯。"""

    name = "S1"

    def plan(self, session: SearchSession, ctx: Mapping) -> dict:
        if session.search_calls_used == 0 and not ctx.get("anchors_empty"):
            return {"action": "search", "channel": "C2", "note": "先出文件:计数表"}
        signals = session.signals(ctx)
        if signals and session.next_rung() is not None:
            return {"action": "escalate", "rung": session.next_rung(), "triggers": signals}
        if session.candidates and session.chosen is None:
            return {"action": "read", "channel": "read_file", "note": "阶段 3 函数级确认"}
        if session.chosen is not None:
            return {"action": "edit", "channel": "submit_patch"}
        if session.must_fail_closed():
            return {"action": "fail", "channel": "none", "note": "证据不足，明确失败"}
        return {"action": "search", "channel": "C1"}

    def rank_candidates(self, session: SearchSession) -> list[dict]:
        """③ 非测试优先；测试文件只作桥，不作候选。"""
        from .lexsearch import is_test_path, src_side_priority

        filtered = [c for c in session.candidates if not is_test_path(c.get("path", ""))]
        return sorted(
            filtered,
            key=lambda c: (
                -int(c.get("distinct_queries", 0)),
                -float(c.get("concentration", 0.0)),
                src_side_priority(c.get("path", "")),
                str(c.get("path", "")).encode("utf-8"),
            ),
        )


def policies() -> dict[str, SearchPolicy]:
    """四臂里的搜索侧两档：A/B 用 S0，C/D 用 S1（共用同一骨架）。"""
    return {"S0": S0Policy(), "S1": S1Policy()}


def assert_same_skeleton(left: SearchPolicy, right: SearchPolicy) -> None:
    """S0/S1 必须实现同一接口 —— 否则消融不是单变量。"""
    required = ("plan", "rank_candidates")
    for name in required:
        if not callable(getattr(left, name, None)) or not callable(getattr(right, name, None)):
            raise PolicyViolation(
                "skeleton_mismatch", "策略缺少接口 %s，S0/S1 必须共用同一骨架" % name
            )


def escape_shell_arg(value: str) -> str:
    """安全转义单个 shell 参数（issue 文本不得直接插值进命令）。"""
    return shlex.quote(str(value))


def build_grep_command(pattern: str, root: str = "/workspace", max_files: int = 20, literal: bool = True) -> str:
    """两段式检索的第一段：先只出文件名（≤20 行）。"""
    flag = "-F" if literal else "-E"
    return "grep -rln %s --include=*.py -- %s %s | head -%d" % (
        flag,
        escape_shell_arg(pattern),
        escape_shell_arg(root),
        int(max_files),
    )


def build_count_command(pattern: str, root: str = "/workspace", max_lines: int = 10) -> str:
    """C2：文件:计数表（只给计数，不吐正文）。"""
    return "grep -rc -F -- %s %s | sort -t: -k2 -nr | head -%d" % (
        escape_shell_arg(pattern),
        escape_shell_arg(root),
        int(max_lines),
    )


def build_read_command(path: str, start: int, end: int) -> str:
    """read_file 必须有行号范围（单文件 150 行 / 10000 字符上限）。"""
    if end - start + 1 > 150:
        raise PolicyViolation("read_range_too_wide", "read_file 区间超过 150 行上限")
    return "read_file(filepath=%s, start_line=%d, end_line=%d)" % (
        escape_shell_arg(path),
        int(start),
        int(end),
    )


def four_arm_matrix() -> list[dict]:
    """四臂定义（KAGGLE-19 §6）：A=W0/S0，B=W0/S1，C=W0+L1/S0，D=W0+L1/S1。"""
    return [
        {"arm": "A", "policy": "S0", "adapter": None, "prompt": "W0"},
        {"arm": "B", "policy": "S1", "adapter": None, "prompt": "W0"},
        {"arm": "C", "policy": "S0", "adapter": "L1", "prompt": "W0"},
        {"arm": "D", "policy": "S1", "adapter": "L1", "prompt": "W0"},
    ]


def assert_single_writer(arms: Sequence[Mapping]) -> None:
    """首轮四臂必须共用单个 LlmAgent、同一工具集、不带强制分析员。"""
    problems = []
    if len({arm.get("policy_owner", "single_agent") for arm in arms}) != 1:
        problems.append("写者不唯一")
    for arm in arms:
        if arm.get("forced_analyzer"):
            problems.append("臂 %s 带了强制分析员" % arm.get("arm"))
    if problems:
        raise PolicyViolation("not_single_writer", "四臂拓扑不满足单写者约束", problems=problems)
