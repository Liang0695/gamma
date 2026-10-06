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
from ..common.errors import Blocked, MissingInput, PolicyViolation
from .faces import VERIFICATION_STATUS, validate_verification

#: pytest 短摘要（`-rA`）里逐节点状态行的前缀。
_PYTEST_NODE_STATUSES = ("PASSED", "FAILED", "ERROR", "SKIPPED")

#: pytest 明确报告"没有收集到用例"的文本（用于把"解析失败"与"确实零用例"分开）。
_PYTEST_NO_TESTS_MARKERS = ("no tests ran", "no tests collected", "collected 0 items")


class RunResult:
    """一次测试执行的规范化结果。

    `failed_tests` / `passed_tests` / `skipped_tests` 装的是**测试节点名**（pytest 的
    `path::test`）；跑了哪些**命令**记录在 `commands_run` 里。两者不再混为一谈。
    `node_ids_available` 为 False 表示这条结果只有命令级信息（非 pytest 命令），
    其 `passed_tests` / `failed_tests` 里装的是命令字符串 —— 这种结果**不能**用来
    证明"没有 skip"。
    """

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
        commands_run: Sequence[str] = (),
        node_ids_available: bool = True,
    ) -> None:
        self.exit_code = int(exit_code)
        self.stdout = stdout
        self.stderr = stderr
        self.duration_ms = int(duration_ms)
        self.failed_tests = list(failed_tests)
        self.passed_tests = list(passed_tests)
        self.skipped_tests = list(skipped_tests)
        self.collection_error = bool(collection_error)
        self.commands_run = list(commands_run)
        self.node_ids_available = bool(node_ids_available)

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
            "commands_run": list(self.commands_run),
            "node_ids_available": self.node_ids_available,
            "stdout_sha256": sha256_json({"stdout": self.stdout}),
            "stderr_sha256": sha256_json({"stderr": self.stderr}),
        }


def _looks_like_pytest(argv: Sequence[str]) -> bool:
    """判定一条命令是不是 pytest 调用（决定走节点级解析还是命令级记录）。"""
    parts = [str(a) for a in argv]
    if not parts:
        return False
    for index, part in enumerate(parts):
        base = part.replace("\\", "/").rsplit("/", 1)[-1].lower()
        if base in ("pytest", "pytest.exe", "py.test"):
            return True
        if base in ("pytest.py",):
            return True
        # `python -m pytest ...` / `py -m pytest ...`
        if part == "-m" and index + 1 < len(parts) and parts[index + 1].strip() == "pytest":
            return True
    return False


def _pytest_argv_with_report(argv: Sequence[str]) -> list[str]:
    """给 pytest 命令追加逐节点报告参数（`-rA -q`）；已有则不重复追加。"""
    out = [str(a) for a in argv]
    if not _looks_like_pytest(out):
        return out
    if not any(a == "-rA" or a.startswith("-r") or a.startswith("--report") for a in out):
        out.append("-rA")
    if not any(a in ("-q", "--quiet") for a in out):
        out.append("-q")
    return out


def _parse_pytest_report(stdout: str) -> dict:
    """从 pytest 文本输出解析逐节点结果（纯函数，可脱离子进程单测）。

    识别 `-rA` 短摘要里的 `PASSED/FAILED/ERROR/SKIPPED <nodeid>` 行；
    没有短摘要时回退到通用扫描。`parsable=False` 表示"没解析到任何节点名，
    也没有明确的零用例标记"，调用方必须据此 fail-closed。
    """
    passed: list[str] = []
    failed: list[str] = []
    skipped: list[str] = []
    text = stdout or ""
    no_tests = any(marker in text for marker in _PYTEST_NO_TESTS_MARKERS)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        head, sep, rest = line.partition(" ")
        if head not in _PYTEST_NODE_STATUSES or not sep:
            continue
        node = rest.strip()
        # 短摘要尾部可能跟 " - <原因>"，只取节点名本身。
        node = node.split(" - ", 1)[0].strip()
        # SKIPPED 行可能带原因计数前缀，例如 "SKIPPED [1] tests/a.py::test_x"。
        if node.startswith("["):
            _tail = node[1:].split("]", 1)
            node = _tail[1].strip() if len(_tail) == 2 else node
        if not node or "::" not in node and "/" not in node and "\\" not in node:
            continue
        # 同一节点会同时出现在进度行与短摘要里，按桶去重。
        if head == "PASSED":
            bucket = passed
        elif head in ("FAILED", "ERROR"):
            bucket = failed
        else:
            bucket = skipped
        if node not in bucket:
            bucket.append(node)
    return {
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "no_tests": no_tests,
        "parsable": bool(passed or failed or skipped or no_tests),
    }


def _pytest_output_signals(stdout: str, stderr: str = "") -> bool:
    """判定输出里是否出现 **collection error**（真收集/导入错误）。

    只认 pytest 自己的收集错误标记，**不**把普通断言失败（`E   AssertionError`）
    误判成 collection error —— 后者是正常的 F2P 失败，会导致 broken 段永远算 pass。
    识别：`ERROR collecting <file>`、错误摘要区 `=== ERRORS ===`、
    或摘要行 `ERROR <nodeid>`。
    """
    text = "%s\n%s" % (stdout or "", stderr or "")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "ERROR collecting" in line:
            return True
        if line.startswith("=") and "ERRORS" in line:
            return True
        if line.startswith("ERROR ") and ("::" in line or ".py" in line):
            return True
    return False


class SubprocessRunner:
    """真实执行器：在给定 workspace 里顺序跑命令（超时/退出码如实记录）。"""

    def __init__(self, timeout_seconds: int = 300) -> None:
        self.timeout_seconds = int(timeout_seconds)

    def run(self, workspace: str, commands: Sequence[str]) -> RunResult:
        """逐命令执行；pytest 命令解析出**节点级**结果，其余按命令级记录。

        硬规则：看起来是 pytest 的命令若非零退出却解析不出任何节点名，
        抛 `Blocked("test_report_unparsable")` —— 不允许用空列表冒充"无 skip"。
        """
        failed: list[str] = []
        passed: list[str] = []
        skipped: list[str] = []
        commands_run: list[str] = []
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        exit_code = 0
        collection_error = False
        node_ids_available = True
        started = time.monotonic()
        for command in commands:
            command_text = command if isinstance(command, str) else " ".join(str(a) for a in command)
            argv = [str(a) for a in command] if isinstance(command, (list, tuple)) else command.split()
            commands_run.append(command_text)
            is_pytest = _looks_like_pytest(argv)
            if is_pytest:
                argv = _pytest_argv_with_report(argv)
            try:
                proc = subprocess.run(
                    list(argv),
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                # 超时：没有节点级证据，按命令级失败记录。
                failed.append(command_text)
                node_ids_available = False
                stderr_parts.append("timeout: %s" % command_text)
                exit_code = 124
                continue
            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            stdout_parts.append(stdout)
            stderr_parts.append(stderr)
            if proc.returncode != 0:
                exit_code = proc.returncode

            if not is_pytest:
                # 非 pytest：只有命令级证据，不得声称拿到了节点结果。
                node_ids_available = False
                if proc.returncode != 0:
                    failed.append(command_text)
                else:
                    passed.append(command_text)
                continue

            parsed = _parse_pytest_report(stdout)
            if not parsed["parsable"]:
                # 零退出也一样 fail-closed：解析不出节点名就没有"无 skip"的证据，
                # 不允许退化成一个看起来成功的空结果。
                raise Blocked(
                    "test_report_unparsable",
                    "pytest 命令的输出解析不出任何节点名，不得冒充「无 skip」",
                    command=command_text,
                    workspace=workspace,
                    exit_code=proc.returncode,
                )
            if _pytest_output_signals(stdout, stderr):
                collection_error = True
            passed.extend(parsed["passed"])
            failed.extend(parsed["failed"])
            skipped.extend(parsed["skipped"])
        return RunResult(
            exit_code=exit_code,
            stdout="\n".join(stdout_parts),
            stderr="\n".join(stderr_parts),
            duration_ms=int((time.monotonic() - started) * 1000),
            failed_tests=failed,
            passed_tests=passed,
            skipped_tests=skipped,
            collection_error=collection_error,
            commands_run=commands_run,
            node_ids_available=node_ids_available,
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
        # 空的 F2P / P2P 列表等于没有对照约束，必须 fail-closed（否则 P2P 段形同虚设）。
        for field in ("f2p_tests", "p2p_tests"):
            value = self.oracle[field]
            if isinstance(value, str):
                value = [value]
            if value is None:
                value = []
            entries = [v for v in value if str(v).strip()]
            if not entries:
                raise MissingInput(
                    "oracle_empty_test_list",
                    "oracle 的 %s 为空，不得用空列表冒充「无需对照」" % field,
                    field=field,
                )

    def _run_repeated(self, workspace: str, commands: Sequence[str]) -> list[RunResult]:
        return [self.runner.run(workspace, commands) for _ in range(self.repeats)]

    @staticmethod
    def _consistent(results: Sequence[RunResult]) -> bool:
        signatures = {(r.exit_code, tuple(sorted(r.failed_tests))) for r in results}
        return len(signatures) == 1

    @staticmethod
    def _failing_labels(results: Sequence[RunResult]) -> list[str]:
        """一段执行里可归因的失败标识：节点名优先，没有节点名时退回命令名。"""
        labels: list[str] = []
        for result in results:
            labels.extend(result.failed_tests)
            if not result.failed_tests and result.exit_code != 0:
                labels.extend(result.commands_run or ["exit_code=%d" % result.exit_code])
        return sorted(set(labels))

    def contrast(self, broken_ws: str, reference_ws: str, clean_ws: str) -> dict:
        """执行对照并给出结论；不通过一律进 quarantine。

        F2P 与 P2P 是两条独立判定线：
        - F2P（`verify_commands`）：broken 必须稳定失败、reference 必须全通过；
        - P2P（`p2p_tests`）：在 clean 与 reference 两个工作区各跑 `repeats` 次，
          任一失败即回归。
        """
        commands = list(self.oracle["verify_commands"])
        p2p_commands = list(self.oracle["p2p_tests"])
        f2p_declared = list(self.oracle["f2p_tests"])
        broken = self._run_repeated(broken_ws, commands)
        reference = self._run_repeated(reference_ws, commands)
        clean = self._run_repeated(clean_ws, commands)
        p2p_reference = self._run_repeated(reference_ws, p2p_commands)
        p2p_clean = self._run_repeated(clean_ws, p2p_commands)

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
        # 1b) broken 的失败必须真的覆盖声明的每一个 F2P 节点。
        # 只有"确实记录了失败、但不是声明的那个节点"才算未复现（证据不足）；
        # 完全没有失败由上面的检查按明确 fail 处理，不在这里重复判定。
        for index, result in enumerate(broken, start=1):
            if not result.failed_tests:
                continue
            missing = [node for node in f2p_declared if node not in result.failed_tests]
            if missing:
                problems.append(
                    "broken_tree 第 %d 次执行未复现声明的 F2P 目标：%s" % (index, ", ".join(missing))
                )

        # 2) reference 必须全部通过，且不得出现 skip
        if not all(r.all_target_passed for r in reference):
            problems.append("reference_tree 未全部通过")
        if any(r.skipped_tests for r in reference):
            problems.append("reference_tree 出现 skip，不计 pass")
        if not self._consistent(reference):
            problems.append("reference_tree 两次执行不一致")

        # 3) clean / reference 的 F2P + P2P 不允许回归
        if not all(r.exit_code == 0 for r in clean):
            problems.append("clean_tree 的 F2P 出现回归")
        for label, segment in (("clean", p2p_clean), ("reference", p2p_reference)):
            if not all(r.all_target_passed for r in segment):
                problems.append(
                    "P2P 出现回归（%s_tree）：%s" % (label, "、".join(self._failing_labels(segment)))
                )
        # skip 出现在任何一段都算问题，且不依赖节点名是否可解析。
        skip_signal = (
            any(r.skipped_tests for r in broken)
            or any(r.skipped_tests for r in reference)
            or any(r.skipped_tests for r in clean)
            or any(r.skipped_tests for r in p2p_clean)
            or any(r.skipped_tests for r in p2p_reference)
        )
        if skip_signal:
            problems.append("某一段出现 skip，skip 不算 pass")

        runs = {
            "broken": [r.to_dict() for r in broken],
            "reference": [r.to_dict() for r in reference],
            "clean": [r.to_dict() for r in clean],
            "p2p_clean": [r.to_dict() for r in p2p_clean],
            "p2p_reference": [r.to_dict() for r in p2p_reference],
        }
        if problems:
            # F2P 未复现 / 两次不一致 / flaky → 证据不足，inconclusive；其余为明确 fail。
            # 但 **P2P 回归是明确失败**，优先级高于 inconclusive：把一个真的出现回归的
            # 结果降级成"证据不足"，会让"P2P 无回归"这条验收口径失去意义。
            p2p_regression = any("P2P" in p for p in problems)
            inconclusive = any(
                "不一致" in p or "flaky" in p or "未复现" in p for p in problems
            )
            return {
                "status": "fail" if (p2p_regression or not inconclusive) else "inconclusive",
                "quarantine": True,
                "problems": problems,
                "p2p_regression": p2p_regression,
                "runs": runs,
            }
        validate_verification("pass", True, True, False, self.repeats)
        return {
            "status": "pass",
            "quarantine": False,
            "problems": [],
            "runs": runs,
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
