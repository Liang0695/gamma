"""20 个 mask / 模板兼容 fixture（KAGGLE-20 §4 验收项）。

覆盖设计稿逐条要求的场景：空搜索、多 tool_call、失败恢复、thinking 开关、
tool 返回内包含标记文本、Unicode 路径/引号、未知工具、参数 roundtrip、BOS 唯一、
padding 全 mask、历史 assistant 只作 context、恢复观察与正确后继同窗、
以及"渲染与 serving 一致"必须显式报 unverified。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from ..common.errors import FailClosed, PolicyViolation
from .masks import render_and_mask
from .template import BOS_TOKEN_ID, ShimRenderer, assert_roundtrip_arguments


@dataclass
class Fixture:
    fixture_id: str
    purpose: str
    messages: list[dict]
    thinking: bool = False
    pad_to: int | None = None
    expect_raise: str | None = None
    extra_check: Callable[[dict], dict] | None = None
    tags: list[str] = field(default_factory=list)


def _asst(content: str, *, step: int, tool: str, supervised: bool, phase: str, args: Mapping | None = None, reason: str = "") -> dict:
    return {
        "role": "assistant",
        "content": content,
        "tool_name": tool,
        "arguments": dict(args) if args is not None else None,
        "step": step,
        "loss_eligible": supervised,
        "phase": phase,
        "reason": reason,
    }


def _tool(content: str, *, step: int, tool: str = "run_command") -> dict:
    return {"role": "tool", "content": content, "tool_name": tool, "step": step, "phase": "localize"}


def _user(content: str) -> dict:
    return {"role": "user", "content": content}


def _system(content: str) -> dict:
    return {"role": "system", "content": content}


def _check_unknown_tool(stats: dict) -> dict:
    from ..t0.official import OFFICIAL_TOOLS

    used = stats.get("tools_used") or []
    unknown = [name for name in used if name not in OFFICIAL_TOOLS]
    if unknown:
        raise PolicyViolation("unknown_tool", "fixture 使用了未知工具：%s" % unknown)
    stats["unknown_tools"] = 0
    return stats


def _check_args_roundtrip(stats: dict) -> dict:
    """F11：对 fixture 里**真实那条** assistant 消息的 args 做渲染级 roundtrip。

    KAGGLE-26 / Q0 报告 Y2：旧实现传的是硬编码字典，且不接触 renderer，断言恒真。
    现在改成从 `stats["assistant_arguments"]`（由 `run_fixture` 从 fixture 自己的
    messages 抽出）取参数，并且**必须经过 ShimRenderer.render()** 才能通过。
    """
    items = stats.get("assistant_arguments") or []
    if not items:
        raise PolicyViolation(
            "args_roundtrip_no_arguments", "F11 里没有带结构化参数的 assistant 消息"
        )
    spans = []
    for item in items:
        outcome = assert_roundtrip_arguments(
            ShimRenderer(), item["arguments"], tool_name=item.get("tool_name") or "run_command"
        )
        spans.append(outcome["span"])
    stats["args_roundtrip_lossless"] = True
    stats["args_roundtrip_cases"] = len(items)
    stats["args_roundtrip_spans"] = spans
    return stats


def _check_thinking_reasoning_channel(stats: dict) -> dict:
    """F04：thinking 打开时必须有独立 reasoning 通道，且该通道不参与 loss。"""
    if not stats.get("reasoning_span_count"):
        raise PolicyViolation(
            "thinking_channel_missing", "thinking=True 却没有渲染出 reasoning 通道"
        )
    if stats.get("reasoning_context_tokens", 0) <= 0:
        raise PolicyViolation("thinking_channel_empty", "reasoning 通道长度为 0")
    stats["thinking_channel_verified"] = "reasoning_context_only"
    return stats


def _check_thinking_off_no_reasoning(stats: dict) -> dict:
    """F05：thinking 关闭时不得产生 reasoning 通道（与 F04 必须可区分）。"""
    if stats.get("reasoning_span_count"):
        raise PolicyViolation(
            "thinking_channel_unexpected", "thinking=False 却出现了 reasoning 通道"
        )
    stats["thinking_channel_verified"] = "no_reasoning_channel"
    return stats


def _check_recovery_same_window(stats: dict) -> dict:
    from ..data.exporter import _is_illegal_boundary

    steps = [
        {"role": "assistant", "phase": "recover", "loss_eligible": False, "finish_reason": "error", "step": 1},
        {"role": "assistant", "phase": "recover", "loss_eligible": True, "finish_reason": None, "step": 2},
    ]
    if not _is_illegal_boundary(steps, 1):
        raise PolicyViolation("window_boundary_leak", "恢复观察与正确后继被允许切窗")
    stats["recovery_boundary_protected"] = True
    return stats


def _check_serving_render_parity(stats: dict) -> dict:
    if stats.get("official_template_verified"):
        raise PolicyViolation(
            "false_parity_claim", "shim 渲染器不得声称与 serving 渲染一致"
        )
    stats["serving_render_parity"] = "unverified"
    return stats


def build_fixtures() -> list[Fixture]:
    """构造 20 个 fixture。全部为合成数据，不含任何比赛 gold。"""
    long_output = "".join("line %d of tool output\n" % i for i in range(400))
    return [
        Fixture(
            "F01-empty-search",
            "空搜索：无候选时输出明确失败动作，而不是编造位置",
            [
                _system("You are a locating agent."),
                _user("Something is broken but no anchors are given."),
                _tool("(empty result)", step=1),
                _asst(
                    "LOCATE v1\nanchor: none\ncandidates:\nchosen: none\nsearch: {queries: 0, tool_calls: 1, escalate: none}\nunknown: no anchors",
                    step=2,
                    tool="run_command",
                    supervised=True,
                    phase="localize",
                ),
            ],
            tags=["empty_search", "fail_closed"],
        ),
        Fixture(
            "F02-multi-tool-call",
            "多 tool_call：一个 assistant turn 里两次工具调用，全部结构化为 object",
            [
                _user("Two files may be involved."),
                _asst(
                    "search both",
                    step=1,
                    tool="run_command",
                    supervised=True,
                    phase="localize",
                    args={"command": "grep -rln -F -- 'alpha' /workspace | head -20"},
                ),
                _tool("a.py\nb.py", step=2),
                _asst(
                    "call graph tool too",
                    step=3,
                    tool="get_code_neighbors",
                    supervised=True,
                    phase="localize",
                    args={"node": "pkg.mod.alpha"},
                ),
                _tool("pkg.mod.alpha -> pkg.mod.beta", step=4, tool="get_code_neighbors"),
            ],
            tags=["multi_tool_call"],
            extra_check=_check_unknown_tool,
        ),
        Fixture(
            "F03-failure-recovery",
            "失败恢复：失败动作 mask，恢复后的正确动作进入 loss",
            [
                _user("The tool call failed, recover."),
                _asst(
                    "bad call",
                    step=1,
                    tool="read_file",
                    supervised=False,
                    phase="recover",
                    args={"filepath": "missing.py"},
                    reason="bad_action_context_only",
                ),
                _tool("FileNotFoundError: missing.py", step=2, tool="read_file"),
                _asst(
                    "recover with grep",
                    step=3,
                    tool="run_command",
                    supervised=True,
                    phase="recover",
                    args={"command": "grep -rln -F -- 'missing' /workspace | head -20"},
                ),
            ],
            tags=["recovery"],
        ),
        Fixture(
            "F04-thinking-on",
            "thinking 开关打开：content 走独立 reasoning 通道，只作 context 不施加 loss；"
            "工具名+参数走 action 通道并参与监督",
            [
                _user("Find the failing function."),
                _asst(
                    "thinking... then act",
                    step=1,
                    tool="run_command",
                    supervised=True,
                    phase="localize",
                    args={"command": "grep -rc -F -- 'timeout' /workspace | sort -t: -k2 -nr | head -10"},
                ),
            ],
            thinking=True,
            tags=["thinking"],
            extra_check=_check_thinking_reasoning_channel,
        ),
        Fixture(
            "F05-thinking-off",
            "thinking 开关关闭：不产生 reasoning 通道，content 并入 action 通道并参与监督"
            "（与 F04 的 input_ids 必须不同）",
            [
                _user("Find the failing function."),
                _asst(
                    "act",
                    step=1,
                    tool="run_command",
                    supervised=True,
                    phase="localize",
                    args={"command": "grep -rln -F -- 'timeout' /workspace | head -20"},
                ),
            ],
            thinking=False,
            tags=["thinking"],
            extra_check=_check_thinking_off_no_reasoning,
        ),
        Fixture(
            "F06-marker-in-tool-output",
            "tool 返回内含 <|assistant|> 标记文本：不得污染 mask 偏移",
            [
                _user("Tool output contains markers."),
                _tool("the file contains <|assistant|> literally", step=1),
                _asst(
                    "cite it",
                    step=2,
                    tool="read_file",
                    supervised=True,
                    phase="localize",
                    args={"filepath": "pkg/mod.py", "start_line": 10, "end_line": 20},
                ),
            ],
            tags=["marker_injection"],
        ),
        Fixture(
            "F07-end-of-turn-in-tool-output",
            "tool 返回内含 <end_of_turn>：不得提前截断 turn",
            [
                _user("Marker inside observation."),
                _tool("prefix <end_of_turn> suffix", step=1),
                _asst(
                    "continue",
                    step=2,
                    tool="run_command",
                    supervised=True,
                    phase="localize",
                    args={"command": "grep -rln -F -- 'x' /workspace | head -20"},
                ),
            ],
            tags=["marker_injection"],
        ),
        Fixture(
            "F08-unicode-path",
            "Unicode 路径：中文目录名不得破坏 span 偏移",
            [
                _user("中文路径下的问题"),
                _asst(
                    "read",
                    step=1,
                    tool="read_file",
                    supervised=True,
                    phase="localize",
                    args={"filepath": "源码/模块/处理.py", "start_line": 1, "end_line": 30},
                ),
            ],
            tags=["unicode"],
        ),
        Fixture(
            "F09-unicode-quotes",
            "Unicode 引号与反引号：锚点逐字引用不被改写",
            [
                _user("报错信息是 “unexpected token” 出现在 `--strict` 模式"),
                _asst(
                    "locate",
                    step=1,
                    tool="run_command",
                    supervised=True,
                    phase="localize",
                    args={"command": "grep -rln -F -- '--strict' /workspace | head -20"},
                ),
            ],
            tags=["unicode"],
        ),
        Fixture(
            "F10-special-token-in-user-text",
            "用户文本里出现特殊标记：仍按 role 切 span",
            [
                _user("The literal text <start_of_turn> appears in my issue."),
                _asst(
                    "act",
                    step=1,
                    tool="run_command",
                    supervised=True,
                    phase="localize",
                    args={"command": "grep -rln -F -- 'start_of_turn' /workspace | head -20"},
                ),
            ],
            tags=["marker_injection"],
        ),
        Fixture(
            "F11-args-roundtrip",
            "工具参数 roundtrip 无损：嵌套 object 保持结构",
            [
                _user("Nested args."),
                _asst(
                    "edit",
                    step=1,
                    tool="edit_file",
                    supervised=True,
                    phase="patch_verify",
                    args={"filepath": "pkg/mod.py", "new_content": "x = 1", "start_line": 5, "end_line": 5},
                ),
            ],
            tags=["args"],
            extra_check=_check_args_roundtrip,
        ),
        Fixture(
            "F12-bos-single",
            "BOS 恰好一次：不得重复添加",
            [_user("Check BOS."), _asst("act", step=1, tool="run_command", supervised=True, phase="localize", args={"command": "ls"})],
            tags=["bos"],
        ),
        Fixture(
            "F13-padding-masked",
            "padding 全部 -100",
            [_user("Pad this."), _asst("act", step=1, tool="run_command", supervised=True, phase="localize", args={"command": "ls"})],
            pad_to=96,
            tags=["padding"],
        ),
        Fixture(
            "F14-history-assistant-context",
            "历史 assistant 动作只作 context，不参与 loss",
            [
                _user("Multi turn."),
                _asst("earlier plan", step=1, tool="run_command", supervised=False, phase="localize", args={"command": "ls"}, reason="history_context"),
                _tool("ok", step=2),
                _asst("now locate", step=3, tool="run_command", supervised=True, phase="localize", args={"command": "grep -rln -F -- 'q' /workspace | head -20"}),
            ],
            tags=["context"],
        ),
        Fixture(
            "F15-truncated-observation",
            "被截断的工具观测仍不参与 loss，且保持 truncated 标记语义",
            [
                _user("Huge output."),
                _tool(long_output[:4000] + "\n[truncated]", step=1),
                _asst("narrow output", step=2, tool="run_command", supervised=True, phase="localize", args={"command": "grep -rc -F -- 'line' /workspace | head -10"}),
            ],
            tags=["truncation"],
        ),
        Fixture(
            "F16-recovery-window-boundary",
            "恢复观察与正确后继必须同窗（窗口切分边界检查）",
            [_user("Recovery boundary check."), _asst("recover", step=1, tool="run_command", supervised=True, phase="recover", args={"command": "ls"})],
            tags=["windowing"],
            extra_check=_check_recovery_same_window,
        ),
        Fixture(
            "F17-serving-parity-unverified",
            "渲染与 serving 一致：shim 下必须显式报告 unverified",
            [_user("Parity."), _asst("act", step=1, tool="run_command", supervised=True, phase="localize", args={"command": "ls"})],
            tags=["parity"],
            extra_check=_check_serving_render_parity,
        ),
        Fixture(
            "F18-patch-verify-phase",
            "补丁/验证阶段动作进入监督桶",
            [
                _user("Apply patch."),
                _asst("edit", step=1, tool="edit_file", supervised=True, phase="patch_verify", args={"filepath": "pkg/mod.py", "new_content": "y = 2", "start_line": 9, "end_line": 9}),
                _tool("ok", step=2, tool="edit_file"),
                _asst("validate", step=3, tool="run_command", supervised=True, phase="patch_verify", args={"command": "python -m pytest -q tests/test_mod.py"}),
            ],
            tags=["patch_verify"],
        ),
        Fixture(
            "F19-empty-patch-explicit-failure",
            "证据不足时允许明确失败/空补丁，不强迫盲改",
            [
                _user("No evidence available."),
                _tool("(empty)", step=1),
                _asst(
                    "LOCATE v1\nanchor: none\ncandidates:\nchosen: none\nsearch: {queries: 0, tool_calls: 1, escalate: none}\nunknown: insufficient evidence",
                    step=2,
                    tool="submit_patch",
                    supervised=True,
                    phase="finish",
                    args={},
                ),
            ],
            tags=["fail_closed"],
        ),
        Fixture(
            "F20-tool-role-never-loss",
            "连续多个 tool 观测都不得进入 loss",
            [
                _user("Many observations."),
                _tool("obs-1", step=1),
                _tool("obs-2", step=2),
                _tool("obs-3", step=3),
                _asst("finish", step=4, tool="submit_patch", supervised=True, phase="finish", args={}),
            ],
            tags=["tool_masking"],
        ),
    ]


def run_fixture(fixture: Fixture, renderer=None) -> dict:
    """跑一个 fixture；返回通过/失败与统计（失败不抛给调用方，由调用方汇总）。"""
    renderer = renderer or ShimRenderer()
    tools_used = [m.get("tool_name") for m in fixture.messages if m.get("role") == "assistant" and m.get("tool_name")]
    try:
        rendered, labels, stats = render_and_mask(
            renderer, fixture.messages, thinking=fixture.thinking, pad_to=fixture.pad_to
        )
        stats["tools_used"] = tools_used
        # fixture 自己携带的结构化参数（供 Y2 的渲染级 roundtrip 使用，而不是硬编码字典）
        stats["assistant_arguments"] = [
            {"tool_name": message.get("tool_name"), "arguments": message.get("arguments")}
            for message in fixture.messages
            if message.get("role") == "assistant" and message.get("arguments") is not None
        ]
        if fixture.extra_check is not None:
            stats = fixture.extra_check(stats)
        return {
            "fixture_id": fixture.fixture_id,
            "purpose": fixture.purpose,
            "status": "pass",
            "tags": list(fixture.tags),
            "stats": stats,
            "rendered_length": rendered.length,
            "supervised_tokens": labels.supervised_tokens,
        }
    except FailClosed as exc:
        return {
            "fixture_id": fixture.fixture_id,
            "purpose": fixture.purpose,
            "status": "fail",
            "tags": list(fixture.tags),
            "error": exc.to_dict(),
        }


def run_all(renderer=None) -> dict:
    """跑全部 fixture，给出汇总与必须显式声明为未验证的项。"""
    results = [run_fixture(fixture, renderer=renderer) for fixture in build_fixtures()]
    passed = sum(1 for item in results if item["status"] == "pass")
    unverified = sorted(
        {
            "official_template_byte_equality",
            "serving_render_parity",
            "real_tokenizer_offsets",
        }
    )
    return {
        "fixture_count": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "all_passed": passed == len(results),
        "results": results,
        "unverified_claims": unverified,
        "renderer": (renderer.name if renderer is not None else "shim-deterministic"),
        "note": "mask 工程回归通过 ≠ 官方模板渲染一致通过；后者需要真实 tokenizer + canonical template。",
    }


def dump_report(report: Mapping) -> str:
    return json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
