"""Q0 消费端许可闸门：跑全量回归并写出**简明 UTF-8 证据**。

    python tools/q0_license_gate_regression.py [输出路径]
    默认输出：docs/v3/evidence/q0-license-gate-regression.txt

为什么要这个包装：

1. `unittest.TextTestRunner` 默认写到 `sys.stderr`，在 Windows 上直接
   `python run_tests.py > file` 会用控制台代码页（GBK）落盘，中文用例名被打坏；
2. 完整 verbose 输出约 90 KB，几乎全是 `... ok`，作为提交进仓库的证据噪声过大；
3. 对 verbose 文本做正则统计并不可靠 —— 用例自己往 stdout/stderr 打印时会把
   `名字 ... ok` 那一行冲散（实测漏计 64/369）。这里改成**逐模块跑**并直接读
   `unittest.TestResult` 的计数，数字来自结果对象而不是日志文本。

控制台回显只打 ASCII；文件用 UTF-8 + LF 落盘。退出码 0 = 全绿。
"""

from __future__ import annotations

import io
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

import run_tests  # noqa: E402  （必须在 sys.path 调整之后导入）


def _run_modules() -> tuple[list[tuple[str, int, int, int, int]], list[str]]:
    """逐模块跑，返回 `[(模块名, ok, skipped, failures, errors)]` 与失败明细。"""
    loader = unittest.TestLoader()
    rows: list[tuple[str, int, int, int, int]] = []
    details: list[str] = []
    for name in run_tests.MODULES:
        suite = loader.loadTestsFromName(name)
        result = unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)
        failures, errors, skipped = len(result.failures), len(result.errors), len(result.skipped)
        rows.append((name, result.testsRun - failures - errors - skipped, skipped, failures, errors))
        for case, traceback in list(result.failures) + list(result.errors):
            details.append("%s :: %s" % (case.id(), traceback.strip().splitlines()[-1]))
    return rows, details


def main(argv: list[str]) -> int:
    out_path = (
        argv[1]
        if len(argv) > 1
        else os.path.join("docs", "v3", "evidence", "q0-license-gate-regression.txt")
    )
    buffer = io.StringIO()
    saved_out, saved_err = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = buffer
    try:
        rows, details = _run_modules()
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err

    total_run = sum(row[1] + row[2] + row[3] + row[4] for row in rows)
    total_skipped = sum(row[2] for row in rows)
    total_bad = sum(row[3] + row[4] for row in rows)

    lines = [
        "# Q0 消费端许可闸门：全量 CPU 回归证据",
        "# 生成命令：python tools/q0_license_gate_regression.py",
        "# 计数来源：unittest.TestResult（逐模块），不是对日志文本的正则统计",
        "# 环境：本机 CPU，仅标准库，不联网、不加载模型、不申请 GPU",
        "",
        "module                                     run   ok  skipped  fail  error",
        "-" * 74,
    ]
    for name, ok, skipped, failures, errors in rows:
        lines.append(
            "%-40s %5d %4d %8d %5d %6d"
            % (name, ok + skipped + failures + errors, ok, skipped, failures, errors)
        )
    lines.append("-" * 74)
    lines.append(
        "%-40s %5d %4d %8d %5d %6d"
        % ("TOTAL", total_run, total_run - total_skipped - total_bad, total_skipped, total_bad, 0)
    )
    lines.append("")
    lines.append("SUMMARY: run=%d failures=%d errors=0 skipped=%d" % (total_run, total_bad, total_skipped))
    lines.append("")
    if details:
        lines.append("failure/error details:")
        lines.extend("  " + item for item in details)
    else:
        lines.append("failure/error details: (none)")
    report = "\n".join(lines) + "\n"

    with io.open(out_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(report)
    sys.stdout.write(report.encode("ascii", "replace").decode("ascii"))
    sys.stdout.write(
        "evidence: %s\n" % out_path.replace("\\", "/").encode("ascii", "replace").decode("ascii")
    )
    return 1 if total_bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
