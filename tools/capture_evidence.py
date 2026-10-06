"""重新生成 `docs/v3/evidence/` 下的证据快照（可复现、可复核）。

为什么要这个脚本：
- 证据必须来自**真实执行**，不能手写；
- 但执行输出里会出现**本机绝对路径**（本运行时的 sandbox 目录），
  运行时本地路径不能作为交付物出现，所以统一替换成 `<workdir>`；
- 顺带保证写出的文件是 UTF-8 无 BOM、LF 换行。

    python tools/capture_evidence.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EVIDENCE_DIR = os.path.join(REPO_ROOT, "docs", "v3", "evidence")

#: 本机绝对路径 → 占位符（含反斜杠与正斜杠两种写法）。
SANITIZE = (
    (re.compile(r"L:\\\\[^\"']*?(?=\\\\|[\"'\n])"), "<workdir>"),
    (re.compile(r"L:/[^\"'\s]*"), "<workdir>"),
    (re.compile(r"C:\\\\Users\\\\[^\"'\\]*"), "<home>"),
)


def sanitize(text: str) -> str:
    for pattern, replacement in SANITIZE:
        text = pattern.sub(replacement, text)
    return text


def capture(commands: list[list[str]], out_name: str) -> int:
    # 显式 UTF-8 解码：本机默认 locale 是 GBK，子解释器输出的是 UTF-8，
    # 用默认编码解码会抛 UnicodeDecodeError（或让 stdout 变成 None）。
    proc = subprocess.run(
        commands, cwd=REPO_ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    payload = sanitize(proc.stdout or "")
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(os.path.join(EVIDENCE_DIR, out_name), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(payload)
    print("%-34s exit=%d bytes=%d" % (out_name, proc.returncode, len(payload)))
    return proc.returncode


def main() -> int:
    py = sys.executable
    jobs = [
        ([py, "run_tests.py"], "tests-summary.txt"),
        ([py, "-m", "v3.cli", "train-preflight"], "train-preflight.json"),
        # Q0 §7：可执行训练入口的 CPU 自检证据（前向/反向/优化器步进/保存重载）。
        ([py, "-m", "v3.train.entry", "--smoke"], "train-entry-smoke.json"),
        ([py, "-m", "v3.cli", "deps"], "deps-status.json"),
        ([py, "-m", "v3.cli", "exp1", "--output-dir", EVIDENCE_DIR], "exp1-synthetic-stdout.json"),
        ([py, "-m", "v3.cli", "audit", "--release", "does-not-exist.json"], "audit-missing.json"),
        # 🔴-A：用 D0 冻结副本（65aaa16 的真实产物）跑正向 ingest，退出码必须是 0。
        (
            [
                py,
                "-m",
                "v3.cli",
                "ingest",
                "--source-lock",
                "docs/v3/design/d0-source-lock-65aaa16.json",
                "--train-only",
            ],
            "ingest-d0-source-lock.json",
        ),
    ]
    exit_codes = {name: capture(cmd, name) for cmd, name in jobs}

    # 源路径被替换后，报告里也同步替换 bin 内那份 JSON。
    report = os.path.join(EVIDENCE_DIR, "exp1-report.json")
    if os.path.exists(report):
        with open(report, "r", encoding="utf-8") as handle:
            payload = sanitize(handle.read())
        with open(report, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
        print("%-34s sanitized" % "exp1-report.json")

    print("\ncaptured exit codes:", exit_codes)
    print("说明：train-preflight / deps / audit 的**非零退出码是设计要求的 fail-closed 行为**。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
