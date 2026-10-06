"""Q0 独立代码审查两项缺陷的回归测试（oracle 对照协议）。

覆盖：
- R2（阻断）：`OracleValidator.contrast()` 必须**真的执行** `p2p_tests`，
  且 P2P 失败必须进 quarantine；
- R2 配套：`f2p_tests` / `p2p_tests` 空列表必须 fail-closed；
  broken 段未复现声明的 F2P 节点时必须 `inconclusive`；
- Y12：`SubprocessRunner` 的逐节点解析（skip 可识别、失败装节点名而非命令名、
  解析不出来必须 `Blocked`），以及 `RunResult.to_dict()` 的 `commands_run` / `node_ids_available`。

注意：本机沙箱下子进程管道 stdio 可能不通，所以解析逻辑全部走纯函数
`_parse_pytest_report`，测试里**不起任何子进程**。
"""

from __future__ import annotations

import unittest

from v3.common.errors import Blocked, MissingInput
from v3.data.oracle import (
    OracleValidator,
    RunResult,
    SubprocessRunner,
    _looks_like_pytest,
    _parse_pytest_report,
    _pytest_argv_with_report,
    _pytest_output_signals,
)

#: 反例脚本 R2 用的"必然失败"命令。
REFERENCE_FAILING_COMMAND = ["python -c \"raise SystemExit(1)\""]

F2P_NODE = "tests/test_routing.py::test_timeout"
P2P_NODE = "tests/test_routing.py::test_ok"
F2P_TEST_COMMAND = "python -m pytest -q tests/test_routing.py"


def make_oracle(
    f2p_tests=None,
    p2p_tests=None,
    verify_commands=None,
    gold_files=None,
) -> dict:
    """合成一份最小可用的 oracle 面（只带 OracleValidator 消费的字段）。"""
    return {
        "f2p_tests": [F2P_NODE] if f2p_tests is None else f2p_tests,
        "p2p_tests": [P2P_NODE] if p2p_tests is None else p2p_tests,
        "verify_commands": (
            [F2P_TEST_COMMAND] if verify_commands is None else verify_commands
        ),
        "gold_files": ["alpha/routing.py"] if gold_files is None else gold_files,
    }


class RecordingRunner:
    """按工作区给结果的假执行器，并把每次调用如实记录下来。

    `failing_commands` 里的命令一律返回非零退出 —— 用来模拟"必然失败的命令"，
    这样 P2P 是否被真的执行会直接体现到结论上。
    """

    def __init__(self, broken_failed_tests=None, failing_commands=None) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.broken_failed_tests = (
            [F2P_NODE] if broken_failed_tests is None else list(broken_failed_tests)
        )
        self.failing_commands = set(failing_commands or ())

    def executed_commands(self) -> set[str]:
        return {command for _, commands in self.calls for command in commands}

    def run(self, workspace, commands) -> RunResult:
        commands = list(commands)
        self.calls.append((workspace, tuple(commands)))
        for command in commands:
            if command in self.failing_commands:
                return RunResult(exit_code=1, commands_run=commands)
        # broken 树：稳定失败，且失败节点= 声明的 F2P（或调用方指定的替代节点）
        if workspace == "broken":
            return RunResult(
                exit_code=1,
                failed_tests=list(self.broken_failed_tests),
                commands_run=commands,
            )
        return RunResult(exit_code=0, passed_tests=list(commands), commands_run=commands)


def reference_report_text() -> str:
    """伪造一段真实形态的 pytest `-rA -q` 输出。"""
    return "\n".join(
        [
            "tests/test_routing.py::test_ok PASSED",
            "tests/test_routing.py::test_skip_me SKIPPED [1] tests/conftest.py:9: 平台相关，跳过",
            "tests/test_routing.py::test_timeout FAILED",
            "=================== short test summary info ===================",
            "PASSED tests/test_routing.py::test_ok",
            "SKIPPED [1] tests/test_routing.py::test_skip_me",
            "FAILED tests/test_routing.py::test_timeout - AssertionError: handle(None) is not None",
            "=================== 1 failed, 1 passed, 1 skipped in 0.42s ===================",
        ]
    )


class P2PExecutionTests(unittest.TestCase):
    """R2 阻断项：P2P 必须真的被执行，失败必须进 quarantine。"""

    def test_failing_p2p_command_is_executed_and_quarantined(self) -> None:
        """反例：p2p_tests 给一条必然失败的命令，结论不得为 pass。"""
        oracle = make_oracle(p2p_tests=list(REFERENCE_FAILING_COMMAND))
        runner = RecordingRunner(failing_commands=REFERENCE_FAILING_COMMAND)
        outcome = OracleValidator(runner, oracle, repeats=2).contrast(
            "broken", "reference", "clean"
        )

        # 1) 结论必须是否定的（修复前恒为 pass，因为 p2p_tests 从未被执行）
        self.assertNotEqual(outcome["status"], "pass")
        self.assertEqual(outcome["status"], "fail")
        self.assertTrue(outcome["quarantine"])
        self.assertTrue(any("P2P" in problem for problem in outcome["problems"]))

        # 2) P2P 命令真的被送进执行器（证明不是被忽略）
        self.assertIn(
            REFERENCE_FAILING_COMMAND[0],
            runner.executed_commands(),
            "p2p_tests 里的命令从未被提交给执行器",
        )

    def test_p2p_runs_on_both_clean_and_reference(self) -> None:
        """P2P 必须在 clean 与 reference 两个工作区各跑 repeats 次。"""
        oracle = make_oracle()
        runner = RecordingRunner()
        outcome = OracleValidator(runner, oracle, repeats=2).contrast(
            "broken", "reference", "clean"
        )
        self.assertEqual(outcome["status"], "pass")

        p2p_calls = [
            (workspace, commands)
            for workspace, commands in runner.calls
            if commands == tuple(oracle["p2p_tests"])
        ]
        self.assertEqual(len(p2p_calls), 4, "P2P 应在 clean/reference 各跑 2 次")
        self.assertEqual({ws for ws, _ in p2p_calls}, {"clean", "reference"})

        # 返回结构里能看出 P2P 真的跑过
        for key in ("p2p_clean", "p2p_reference"):
            self.assertIn(key, outcome["runs"])
            self.assertEqual(len(outcome["runs"][key]), 2)
        for key in ("broken", "reference", "clean"):
            self.assertIn(key, outcome["runs"])

    def test_empty_p2p_test_list_is_missing_input(self) -> None:
        """空的 p2p_tests 必须 fail-closed，不得当成"无需对照"。"""
        with self.assertRaises(MissingInput) as ctx:
            OracleValidator(RecordingRunner(), make_oracle(p2p_tests=[]))
        self.assertEqual(ctx.exception.code, "oracle_empty_test_list")

        empty_string = make_oracle(p2p_tests=["   "])
        with self.assertRaises(MissingInput) as ctx2:
            OracleValidator(RecordingRunner(), empty_string)
        self.assertEqual(ctx2.exception.code, "oracle_empty_test_list")

    def test_empty_f2p_test_list_is_missing_input(self) -> None:
        """空的 f2p_tests 同样必须 fail-closed。"""
        with self.assertRaises(MissingInput) as ctx:
            OracleValidator(RecordingRunner(), make_oracle(f2p_tests=[]))
        self.assertEqual(ctx.exception.code, "oracle_empty_test_list")

    def test_broken_without_declared_f2p_node_is_inconclusive(self) -> None:
        """broken 段失败的不是声明的 F2P 节点 → 结论 inconclusive（不是 pass）。"""
        oracle = make_oracle()
        runner = RecordingRunner(broken_failed_tests=["tests/test_routing.py::test_other"])
        outcome = OracleValidator(runner, oracle, repeats=2).contrast(
            "broken", "reference", "clean"
        )
        self.assertEqual(outcome["status"], "inconclusive")
        self.assertTrue(outcome["quarantine"])
        self.assertTrue(any("未复现" in problem for problem in outcome["problems"]))

    def test_skip_in_p2p_segment_is_a_problem(self) -> None:
        """skip 出现在 P2P 段也算问题（不止 reference 段的 F2P）。

        fake runner 构造"P2P 命令退出码 0 但全部被 skip"的执行结果。
        source 里 `skipped.append` 必须真实存在，否则这条硬规则无法落地。
        """

        class SkipOnlyP2PRunner(RecordingRunner):
            def run(self, workspace, commands) -> RunResult:
                commands = list(commands)
                self.calls.append((workspace, tuple(commands)))
                if workspace == "broken":
                    return RunResult(
                        exit_code=1, failed_tests=[F2P_NODE], commands_run=commands
                    )
                if commands == [F2P_TEST_COMMAND]:
                    return RunResult(exit_code=0, commands_run=commands)
                return RunResult(
                    exit_code=0,
                    skipped_tests=[node for node in commands],
                    commands_run=commands,
                )

        oracle = make_oracle()
        runner = SkipOnlyP2PRunner()
        outcome = OracleValidator(runner, oracle, repeats=2).contrast(
            "broken", "reference", "clean"
        )
        self.assertNotEqual(outcome["status"], "pass")
        self.assertTrue(outcome["quarantine"])
        self.assertTrue(any("skip" in problem for problem in outcome["problems"]))


class PytestReportParsingTests(unittest.TestCase):
    """Y12：逐节点解析必须是可脱离子进程单测的纯函数。"""

    def test_parse_extracts_node_names_including_skips(self) -> None:
        parsed = _parse_pytest_report(reference_report_text())
        self.assertTrue(parsed["parsable"])
        self.assertIn("tests/test_routing.py::test_ok", parsed["passed"])
        self.assertIn("tests/test_routing.py::test_timeout", parsed["failed"])
        # skip 必须被识别出来（修复前 skipped 永远是空的）
        self.assertIn("tests/test_routing.py::test_skip_me", parsed["skipped"])

    def test_parse_dedupes_summary_and_progress_lines(self) -> None:
        parsed = _parse_pytest_report(reference_report_text())
        self.assertEqual(len(parsed["passed"]), 1)
        self.assertEqual(len(parsed["failed"]), 1)
        self.assertEqual(len(parsed["skipped"]), 1)

    def test_parse_marks_output_without_nodes_as_unparsable(self) -> None:
        parsed = _parse_pytest_report("段错误 (核心已转储)\n")
        self.assertFalse(parsed["parsable"])
        self.assertEqual(parsed["passed"], [])
        self.assertEqual(parsed["failed"], [])
        self.assertEqual(parsed["skipped"], [])

    def test_parse_recognises_zero_collected_as_parsable(self) -> None:
        """确实零用例是"有结论"，不是"解析失败"。"""
        parsed = _parse_pytest_report("collected 0 items\n\nno tests ran in 0.01s\n")
        self.assertTrue(parsed["parsable"])
        self.assertTrue(parsed["no_tests"])

    def test_unparsable_pytest_failure_raises_blocked(self) -> None:
        """解析不出来必须 fail-closed：Blocked('test_report_unparsable')。"""
        import subprocess as _subprocess

        class FakeCompleted:
            returncode = 4
            stdout = "内部错误，没有任何测试节点行\n"
            stderr = ""

        original_run = _subprocess.run
        _subprocess.run = lambda *a, **kw: FakeCompleted()  # type: ignore[assignment]
        try:
            with self.assertRaises(Blocked) as ctx:
                SubprocessRunner().run("/tmp/ws", ["python -m pytest -q tests/test_routing.py"])
        finally:
            _subprocess.run = original_run  # type: ignore[assignment]
        self.assertEqual(ctx.exception.code, "test_report_unparsable")

    def test_unparsable_pytest_zero_exit_also_raises_blocked(self) -> None:
        """零退出但解析不出节点名同样 fail-closed，不得冒充"无 skip"。"""
        import subprocess as _subprocess

        class FakeCompleted:
            returncode = 0
            stdout = ""
            stderr = ""

        original_run = _subprocess.run
        _subprocess.run = lambda *a, **kw: FakeCompleted()  # type: ignore[assignment]
        try:
            with self.assertRaises(Blocked) as ctx:
                SubprocessRunner().run("/tmp/ws", ["python -m pytest -q tests/empty.py"])
        finally:
            _subprocess.run = original_run  # type: ignore[assignment]
        self.assertEqual(ctx.exception.code, "test_report_unparsable")


class RunnerCommandLineTests(unittest.TestCase):
    """pytest 判定与参数追加（同样不依赖子进程）。"""

    def test_looks_like_pytest(self) -> None:
        self.assertTrue(_looks_like_pytest(["python", "-m", "pytest", "-q", "tests"]))
        self.assertTrue(_looks_like_pytest(["pytest", "tests/test_a.py"]))
        self.assertTrue(_looks_like_pytest([r"C:\py\Scripts\pytest.exe", "-q"]))
        self.assertFalse(_looks_like_pytest(["python", "-c", "raise SystemExit(1)"]))
        self.assertFalse(_looks_like_pytest(["ls"]))
        self.assertFalse(_looks_like_pytest([]))

    def test_pytest_argv_gets_per_node_report_flags(self) -> None:
        argv = _pytest_argv_with_report(["python", "-m", "pytest", "tests/"])
        self.assertIn("-rA", argv)
        self.assertIn("-q", argv)
        # 不重复追加，也不改动非 pytest 命令
        self.assertEqual(_pytest_argv_with_report(argv).count("-rA"), 1)
        self.assertEqual(_pytest_argv_with_report(["ls", "-l"]), ["ls", "-l"])

    def test_non_pytest_command_is_recorded_at_command_level(self) -> None:
        """非 pytest 命令：退出码 0 → passed_tests 装命令字符串且 node_ids_available=False。"""
        import subprocess as _subprocess

        class FakeCompleted:
            returncode = 0
            stdout = ""
            stderr = ""

        original_run = _subprocess.run
        _subprocess.run = lambda *a, **kw: FakeCompleted()  # type: ignore[assignment]
        try:
            result = SubprocessRunner().run("/tmp/ws", ["ls"])
        finally:
            _subprocess.run = original_run  # type: ignore[assignment]
        self.assertEqual(result.passed_tests, ["ls"])
        self.assertFalse(result.node_ids_available)
        self.assertEqual(result.to_dict()["commands_run"], ["ls"])


class CollectionErrorTests(unittest.TestCase):
    """collection error 识别不得误伤普通断言失败。"""

    def test_collection_error_is_recognised(self) -> None:
        self.assertTrue(
            _pytest_output_signals(
                "ERROR collecting tests/test_a.py\nImportError: no module named alpha\n"
            )
        )
        self.assertTrue(
            _pytest_output_signals("============ ERRORS ============\n")
        )

    def test_plain_assertion_failure_is_not_a_collection_error(self) -> None:
        """断言失败是正常的 F2P 失败，误判成 collection error 会让 broken 段永远算 pass。"""
        report = "\n".join(
            [
                "FAILED tests/test_routing.py::test_timeout - AssertionError: handle(None) is not None",
                "________________________________ test_timeout ________________________________",
                "    def test_timeout():",
                ">       assert handle(None) is None",
                "E       AssertionError: assert 1 is None",
                "=== 1 failed, 1 passed in 0.42s ===",
            ]
        )
        self.assertFalse(_pytest_output_signals(report))

    def test_missing_symbol_at_import_time_is_a_collection_error(self) -> None:
        """pytest 收集期导入失败会打 `ERROR <nodeid>` 行。"""
        self.assertTrue(
            _pytest_output_signals(
                "ERROR tests/test_routing.py::test_timeout - ImportError: cannot import name 'handle'\n"
            )
        )


class RunResultContractTests(unittest.TestCase):
    """RunResult 契约：命令与节点名分开记录。"""

    def test_to_dict_carries_commands_run(self) -> None:
        result = RunResult(
            exit_code=1,
            failed_tests=[F2P_NODE],
            commands_run=["python -m pytest -q tests/test_routing.py"],
        )
        dumped = result.to_dict()
        self.assertIn("commands_run", dumped)
        self.assertEqual(dumped["commands_run"], ["python -m pytest -q tests/test_routing.py"])
        self.assertEqual(dumped["failed_tests"], [F2P_NODE])
        self.assertIs(dumped["node_ids_available"], True)

    def test_node_ids_available_defaults_true_for_synthetic_results(self) -> None:
        """合成结果默认按"节点名可用"处理，保持既有 ScriptedRunner fixture 语义。"""
        self.assertTrue(RunResult(exit_code=0).to_dict()["node_ids_available"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
