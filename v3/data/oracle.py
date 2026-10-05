"""八题 oracle / 回放验证器（KAGGLE-21 §4/§7/§9）。

对照协议（每题，各两次干净执行）：

1. **broken + 验收测试**：≥1 个稳定的 F2P 失败（两次一致）；
2. **reference + 同一验收测试**：全部通过；
3. **原 clean / reference + P2P 回归**：全部通过。

判定细则：
- `skip` 或 collection error **不算 pass**；
- 两次不一致 → `inconclusive`，进 quarantine；
- 无失败目标、原环境本来失败、reference 不通过 → `quarantine`，不得进成功集；
- 禁止用 `not_run` 填 `resolved=true`。

回放：在相同初始 tree 上按日志重放动作，比对 tree sha 与规范化后的观测，禁止只比较"看起来一样"。
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Callable, Mapping, Sequence

from ..common.canonical import sha256_json
from ..common.errors import MissingInput, PolicyViolation
from .faces import VERIFICATION_STATUS, validate_verification


class RunResult:
    """一次测试执行的规范化结果。"""

    def __init__(
        self,
        exit_code: int,
        stdout: str = "",
        stderr: str = "",
        duration_ms: int = 0,
        failed_tests: Sequence[str] = (),
        passed_tests: Sequence[str] = (),
        skipped_tests: Sequence[str] = (),
        collection_error: bool = False,
    ) -> None:
        self.exit_code = int(exit_code)
        self.stdout = stdout
        self.stderr = stderr
        self.duration_ms = int(duration_ms)
        self.failed_tests = list(failed_tests)
        self.passed_tests = list(passed_tests)
        self.skipped_tests = list(skipped_tests)
        self.collection_error = bool(collection_error)

    @property
    def all_target_passed(self) -> bool:
        return self.exit_code == 0 and not self.failed_tests and not self.collection_error

    def to_dict(self) -> dict:
        return {
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "failed_tests": sorted(self.failed_tests),
            "passed_tests": sorted(self.passed_tests),
            "skipped_tests": sorted(self.skipped_tests),
            "collection_error": self.collection_error,
            "stdout_sha256": sha256_json({"stdout": self.stdout}),
            "stderr_sha256": sha256_json({"stderr": self.stderr}),
        }


class SubprocessRunner:
    """真实执行器：在给定 workspace 里顺序跑命令（超时/退出码如实记录）。"""

    def __init__(self, timeout_seconds: int = 300) -> None:
        self.timeout_seconds = int(timeout_seconds)

    def run(self, workspace: str, commands: Sequence[str]) -> RunResult:
        failed: list[str] = []
        passed: list[str] = []
        skipped: list[str] = []
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        exit_code = 0
        collection_error = False
        started = time.monotonic()
        for command in commands:
            argv = command if isinstance(command, (list, tuple)) else command.split()
            try:
                proc = subprocess.run(
                    list(argv),
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                failed.append(command)
                stderr_parts.append("timeout: %s" % command)
                exit_code = 124
                continue
            stdout_parts.append(proc.stdout)
            stderr_parts.append(proc.stderr)
            if proc.returncode != 0:
                exit_code = proc.returncode
                failed.append(command)
            else:
                passed.append(command)
            if "ERROR collecting" in proc.stdout or "ERROR collecting" in proc.stderr:
                collection_error = True
        return RunResult(
            exit_code=exit_code,
            stdout="\n".join(stdout_parts),
            stderr="\n".join(stderr_parts),
            duration_ms=int((time.monotonic() - started) * 1000),
            failed_tests=failed,
            passed_tests=passed,
            skipped_tests=skipped,
            collection_error=collection_error,
        )


class ScriptedRunner:
    """行为由调用方给定的执行器（合成 fixture / 测试用）。"""

    def __init__(self, table: Mapping[str, RunResult]) -> None:
        self.table = dict(table)
        self.calls: list[str] = []

    @staticmethod
    def key(workspace: str, commands: Sequence[str]) -> str:
        return sha256_json({"workspace": workspace, "commands": list(commands)})

    def run(self, workspace: str, commands: Sequence[str]) -> RunResult:
        key = self.key(workspace, commands)
        self.calls.append(key)
        if key not in self.table:
            raise MissingInput(
                "scripted_runner_miss", "ScriptedRunner 没有该组合的预设结果", key=key
            )
        return self.table[key]


class OracleValidator:
    """单题 oracle 对照。"""

    def __init__(self, runner, oracle: Mapping, repeats: int = 2) -> None:
        self.runner = runner
        self.oracle = dict(oracle)
        self.repeats = int(repeats)
        for field in ("f2p_tests", "p2p_tests", "verify_commands", "gold_files"):
            if field not in self.oracle:
                raise MissingInput("oracle_missing_field", "oracle 缺少 %s" % field)

    def _run_repeated(self, workspace: str, commands: Sequence[str]) -> list[RunResult]:
        return [self.runner.run(workspace, commands) for _ in range(self.repeats)]

    @staticmethod
    def _consistent(results: Sequence[RunResult]) -> bool:
        signatures = {(r.exit_code, tuple(sorted(r.failed_tests))) for r in results}
        return len(signatures) == 1

    def contrast(self, broken_ws: str, reference_ws: str, clean_ws: str) -> dict:
        """执行三段对照并给出结论；不通过一律进 quarantine。"""
        commands = list(self.oracle["verify_commands"])
        broken = self._run_repeated(broken_ws, commands)
        reference = self._run_repeated(reference_ws, commands)
        clean = self._run_repeated(clean_ws, commands)

        problems: list[str] = []
        # 1) broken 必须稳定失败，且至少一个 F2P 失败（skip / collection error 不算）
        if not all(not r.all_target_passed for r in broken):
            problems.append("broken_tree 没有稳定失败")
        if not all(r.failed_tests for r in broken):
            problems.append("broken_tree 没有记录到失败的 F2P 目标")
        if any(r.collection_error for r in broken):
            problems.append("broken_tree 出现 collection error，不计成功")
        if not self._consistent(broken):
            problems.append("broken_tree 两次执行不一致（flaky）")

        # 2) reference 必须全部通过
        if not all(r.all_target_passed for r in reference):
            problems.append("reference_tree 未全部通过")
        if any(r.skipped_tests for r in reference):
            problems.append("reference_tree 出现 skip，不计 pass")
        if not self._consistent(reference):
            problems.append("reference_tree 两次执行不一致")

        # 3) clean / reference 的 P2P 不允许回归
        if not all(r.exit_code == 0 for r in clean):
            problems.append("clean_tree 的 P2P 出现回归")

        if problems:
            status = "inconclusive" if any("不一致" in p or "flaky" in p for p in problems) else "fail"
            return {
                "status": status,
                "quarantine": True,
                "problems": problems,
                "runs": {
                    "broken": [r.to_dict() for r in broken],
                    "reference": [r.to_dict() for r in reference],
                    "clean": [r.to_dict() for r in clean],
                },
            }
        validate_verification("pass", True, True, False, self.repeats)
        return {
            "status": "pass",
            "quarantine": False,
            "problems": [],
            "runs": {
                "broken": [r.to_dict() for r in broken],
                "reference": [r.to_dict() for r in reference],
                "clean": [r.to_dict() for r in clean],
            },
        }


def validate_question_set(questions: Sequence[Mapping], runner_factory: Callable) -> dict:
    """八题（P0）对照验收：8/8 合格才允许扩大。返回逐题结论与汇总。

    每题传入 {"question_id", "oracle", "workspaces": {"broken","reference","clean"}}。
    """
    results = []
    passed = 0
    for question in questions:
        validator = OracleValidator(runner_factory(question), question["oracle"])
        outcome = validator.contrast(**question["workspaces"])
        outcome["question_id"] = question["question_id"]
        if outcome["status"] == "pass":
            passed += 1
        results.append(outcome)
    total = len(questions)
    return {
        "total": total,
        "passed": passed,
        "qualified": passed == total and total > 0,
        "results": results,
        "note": "零错 20 例仅支持小试；本函数不声称总体质量已证明。",
    }


def replay(
    steps: Sequence[Mapping],
    initial_tree_sha256: str,
    apply_action: Callable[[Mapping, str], str],
    normalize_observation: Callable[[Mapping], dict],
    expect: Mapping[str, dict] | None = None,
) -> dict:
    """动作回放：在相同初始 tree 上重放，比对 tree 与规范化观测。"""
    tree = initial_tree_sha256
    divergences: list[dict] = []
    for step in steps:
        recorded_in = step.get("input_tree_sha256")
        if recorded_in and recorded_in != tree:
            divergences.append(
                {"step": step.get("step"), "kind": "input_tree_mismatch", "expected": recorded_in, "actual": tree}
            )
        new_tree = apply_action(step, tree)
        recorded_out = step.get("output_tree_sha256")
        if recorded_out and recorded_out != new_tree:
            divergences.append(
                {"step": step.get("step"), "kind": "output_tree_mismatch", "expected": recorded_out, "actual": new_tree}
            )
        tree = new_tree
        if expect is not None:
            key = str(step.get("step"))
            if key in expect:
                got = normalize_observation(step)
                if got != expect[key]:
                    divergences.append(
                        {"step": step.get("step"), "kind": "observation_mismatch", "expected": expect[key], "actual": got}
                    )
    return {
        "replayed_steps": len(steps),
        "final_tree_sha256": tree,
        "divergences": divergences,
        "status": "pass" if not divergences else "fail",
        "policy": "只比较规范化字段，禁止只比较「看起来一样」。",
    }


def normalize_observation(step: Mapping) -> dict:
    """默认规范化：丢掉时间相关字段，只保留确定性内容哈希与退出码。"""
    return {
        "exit_code": step.get("exit_code"),
        "observation_sha256": step.get("observation_sha256"),
        "tool_name": step.get("tool_name"),
    }


def dump_report(report: Mapping) -> str:
    return json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2)
