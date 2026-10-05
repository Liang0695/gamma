"""导出训练视图（trajectory → 训练窗口），并给出导出 manifest。

实现 KAGGLE-21 §5/§6/§7 与 KAGGLE-20 §4 的导出规则：

- actor 输入只含当时可见信息；gold / test_patch / 未来提交一律不进 prompt；
- 监督按 **loss_eligible** 逐动作决定，历史 assistant 与全部 tool/system/user 为 context；
- 窗口只监督目标后继动作；**不得从 JSON / patch / 函数调用中间截断**；
- **恢复观察与正确后继动作必须同时留在同一窗口**；
- 训练窗口不得带未来 chosen、gold 或未发生的测试结果；
- 混合比例以**有 loss 的 assistant token**计（定位 30 / 工具 15 / 补丁 30 / 恢复 25），
  黄金辅助 ≤20%、变异 ≤40%、单仓库 ≤30%；冲突时减少规模并报告，不放宽隔离门槛。
"""

from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

from ..common.canonical import sha256_json
from ..common.errors import MissingInput, PolicyViolation
from .faces import FACE_PUBLIC, FACE_TRAJECTORY, assert_no_oracle_in_actor_input, validate_face

#: 监督目标配比（有 loss 的 assistant token 占比）。
SUPERVISION_TARGETS = {
    "localize": 0.30,
    "tool_exit": 0.15,
    "patch_verify": 0.30,
    "recover": 0.25,
}

#: 隔离与配比上限。
GOLD_ASSISTED_MAX = 0.20
MUTATION_MAX = 0.40
SINGLE_REPO_MAX = 0.30

PHASE_TO_BUCKET = {
    "localize": "localize",
    "tool": "tool_exit",
    "finish": "tool_exit",
    "patch_verify": "patch_verify",
    "edit": "patch_verify",
    "validate": "patch_verify",
    "recover": "recover",
}

DEFAULT_WINDOW_TOKENS = 2048


def _step_bucket(step: Mapping) -> str:
    phase = step.get("phase")
    if phase not in PHASE_TO_BUCKET:
        raise PolicyViolation("bad_phase", "未知 phase：%r" % (phase,))
    return PHASE_TO_BUCKET[phase]


def steps_to_messages(public: Mapping, steps: Sequence[Mapping]) -> list[dict]:
    """把固定 problem statement 与逐动作轨迹拼成 messages。

    只有 problem statement 是 prompt 的近端文本；剩余上下文来自真实发生过的动作历史。
    """
    messages: list[dict] = [{"role": "user", "content": public["problem_statement"]}]
    for step in steps:
        role = step["role"]
        messages.append(
            {
                "role": role,
                "content": step.get("content", ""),
                "tool_name": step.get("tool_name"),
                "arguments": step.get("arguments"),
                "observation_ref": step.get("observation_ref"),
                "loss_eligible": bool(step.get("loss_eligible")),
                "phase": step.get("phase"),
                "step": step.get("step"),
            }
        )
    return messages


def _is_illegal_boundary(steps: Sequence[Mapping], index: int) -> bool:
    """窗口能否在 steps[index-1] 与 steps[index] 之间切开？

    禁止：切在一个失败的 assistant 动作与它的恢复后继之间（恢复观察与正确后继必须同窗）。
    """
    left = steps[index - 1]
    right = steps[index]
    if left.get("phase") in ("recover",) and right.get("loss_eligible"):
        return True
    if left.get("role") == "assistant" and left.get("finish_reason") == "error":
        return True
    return False


def build_windows(
    public: Mapping,
    trajectory: Mapping,
    count_tokens: Callable[[Sequence[dict]], int],
    window_tokens: int = DEFAULT_WINDOW_TOKENS,
) -> list[dict]:
    """贪心切窗：完整 turn 为界，不切工具回环，不跨非法边界。"""
    validate_face(FACE_PUBLIC, public)
    validate_face(FACE_TRAJECTORY, trajectory)
    steps = list(trajectory["steps"])
    if not steps:
        raise MissingInput("empty_trajectory", "轨迹没有动作，无法导出窗口")

    windows: list[dict] = []
    start = 0
    while start < len(steps):
        end = start + 1
        while end < len(steps):
            if _is_illegal_boundary(steps, end):
                end += 1
                continue
            if count_tokens(steps_to_messages(public, steps[start : end + 1])) > window_tokens:
                break
            end += 1
        chunk = steps[start:end]
        supervised = [s for s in chunk if s.get("loss_eligible")]
        if supervised:
            windows.append(
                {
                    "task_id": public["task_id"],
                    "trace_id": trajectory["trace_id"],
                    "base_commit": public["base_commit"],
                    "messages": steps_to_messages(public, chunk),
                    "step_range": [chunk[0]["step"], chunk[-1]["step"]],
                    "supervised_steps": [s["step"] for s in supervised],
                    "supervision_buckets": sorted({_step_bucket(s) for s in supervised}),
                    "gold_access": bool(trajectory["gold_access"]),
                    "split": public["split"],
                    "repo_family": public["repo_family"],
                }
            )
        start = end
    if not windows:
        raise PolicyViolation(
            "no_supervised_window", "轨迹中没有任何可监督动作，不得作为成功监督"
        )
    return windows


def audit_mix(windows: Iterable[Mapping]) -> dict:
    """按有 loss 的 assistant token 统计配比与隔离上限（token 口径由调用方加权）。"""
    totals = {key: 0 for key in SUPERVISION_TARGETS}
    grand = 0
    gold_tokens = 0
    mutation_tokens = 0
    by_repo: dict[str, int] = {}
    for window in windows:
        weight = int(window.get("supervised_token_count", 0))
        if weight <= 0:
            continue
        buckets = window.get("supervision_buckets") or []
        if not buckets:
            continue
        share = weight / float(len(buckets))
        for bucket in buckets:
            totals[bucket] += share
        grand += weight
        if window.get("gold_access"):
            gold_tokens += weight
        if window.get("origin_kind") == "mutation":
            mutation_tokens += weight
        repo = str(window.get("repo_family", "unknown"))
        by_repo[repo] = by_repo.get(repo, 0) + weight

    if grand == 0:
        return {"effective_tokens": 0, "status": "empty"}
    ratios = {key: totals[key] / grand for key in SUPERVISION_TARGETS}
    violations = []
    for key, target in SUPERVISION_TARGETS.items():
        if abs(ratios[key] - target) > 0.10:
            violations.append({"bucket": key, "actual": ratios[key], "target": target})
    limits = {
        "gold_assisted": gold_tokens / grand,
        "mutation": mutation_tokens / grand,
        "max_single_repo": (max(by_repo.values()) / grand) if by_repo else 0.0,
    }
    hard = []
    if limits["gold_assisted"] > GOLD_ASSISTED_MAX:
        hard.append("gold_assisted>20%")
    if limits["mutation"] > MUTATION_MAX:
        hard.append("mutation>40%")
    if limits["max_single_repo"] > SINGLE_REPO_MAX:
        hard.append("single_repo>30%")
    return {
        "effective_tokens": grand,
        "supervision_ratios": ratios,
        "isolation_limits": limits,
        "ratio_violations": violations,
        "hard_violations": hard,
        "status": "fail" if hard else ("warn" if violations else "ok"),
        "note": "配比冲突时减少可用规模并报告，不放宽许可/隔离门槛。",
    }


def build_export(
    records: Sequence[Mapping],
    count_tokens: Callable[[Sequence[dict]], int],
    window_tokens: int = DEFAULT_WINDOW_TOKENS,
    export_config: Mapping | None = None,
) -> dict:
    """主入口：records = [{"public":..., "trajectory":...}, ...]。"""
    windows: list[dict] = []
    for record in records:
        public = record["public"]
        trajectory = record["trajectory"]
        for window in build_windows(public, trajectory, count_tokens, window_tokens):
            window["origin_kind"] = public["origin_kind"]
            window["supervised_token_count"] = count_tokens(
                [
                    {"role": "assistant", "content": str(step)}
                    for step in window["messages"]
                    if step.get("loss_eligible")
                ]
            )
            assert_no_oracle_in_actor_input(window)
            windows.append(window)
    if not windows:
        raise PolicyViolation("empty_export", "导出为空：不得发布空训练集")
    mix = audit_mix(windows)
    if mix.get("status") == "fail":
        raise PolicyViolation(
            "export_mix_violation",
            "导出配比违反隔离上限：%s" % mix.get("hard_violations"),
            audit=mix,
        )
    manifest = {
        "export_config": dict(export_config or {}),
        "window_tokens": window_tokens,
        "window_count": len(windows),
        "task_count": len({w["task_id"] for w in windows}),
        "trace_count": len({w["trace_id"] for w in windows}),
        "audit": mix,
        "windows_sha256": sha256_json(windows),
    }
    return {"windows": windows, "export_manifest": manifest}
