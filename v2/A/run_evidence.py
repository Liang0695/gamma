#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KAGGLE-16：一键复跑全部证据并把日志以 **UTF-8** 落盘。

为什么需要它：本机默认 shell 是 Windows PowerShell 5.1，用 ``*> file.txt`` 重定向会把
子进程的 UTF-8 输出重新按控制台编码写盘，中文会变成乱码/非法字节。这里改为在
Python 进程内用 ``contextlib.redirect_stdout`` 直接写 UTF-8 文件，日志可被任何 UTF-8
阅读器正常打开，也可以直接作为证据附件。

用法
----
    python run_evidence.py --b0-zip submission.zip --root ./out \
        --e1-compile-script ../e1_env/kaggle15-e1/e1_compile_b0.py \
        --e1-venv-python <path-to-venv-python>
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import sys
from pathlib import Path

import build_a
import compare_compiles
import rollback_to_b0
import verify_a


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run_capture(fn, argv: list[str], log_path: Path) -> int:
    """在进程内调用，stdout 直写 UTF-8 文件（不经控制台编码）。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = fn(argv)
    log_path.write_text(buf.getvalue(), encoding="utf-8")
    return code


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0-zip", required=True)
    ap.add_argument("--root", required=True, help="输出根目录，如 ./out")
    args = ap.parse_args(argv)

    root = Path(args.root)
    art = root / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    a_zip = root / "A-submission.zip"
    codes: dict[str, int] = {}

    # 1) 构建 A
    codes["build_a"] = run_capture(
        build_a.main,
        ["--b0-zip", args.b0_zip, "--out-a", str(a_zip),
         "--work", str(root / "work"), "--artifacts", str(art)],
        art / "build-a.txt",
    )

    # 2) 独立校验
    codes["verify_a"] = run_capture(
        verify_a.main,
        ["--b0-zip", args.b0_zip, "--a-zip", str(a_zip),
         "--out", str(art / "verify-a.json")],
        art / "verify-a.txt",
    )

    # 3) 对象树对照
    codes["compare_compiles"] = run_capture(
        compare_compiles.main,
        ["--b0-report", str(root / "compile/B0/E1-b0-compile-report.json"),
         "--a-report", str(root / "compile/A/E1-b0-compile-report.json"),
         "--out", str(art / "compile-compare.json")],
        art / "compile-compare.txt",
    )

    # 4) 回退
    codes["rollback_to_b0"] = run_capture(
        rollback_to_b0.main,
        ["--b0-zip", args.b0_zip, "--dest", str(root / "rollback-test"), "--force"],
        art / "rollback-test.txt",
    )

    # 5) 输入哈希账
    lines = []
    for label, path in (
        ("B0 submission.zip", Path(args.b0_zip)),
        ("A A-submission.zip", a_zip),
    ):
        lines.append(f"{sha256_file(path)}  {path.name}  ({label})")
    (art / "inputs-hashes.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("exit codes:", codes)
    return 0 if all(c == 0 for c in codes.values()) else 3


if __name__ == "__main__":
    sys.exit(main())
