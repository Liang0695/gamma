#!/usr/bin/env python3
"""KAGGLE-32 v4 ② 截止控制包装器（纯标准库，DRAFT）

为什么需要它（Mika 第 1 项）：

* v3 用 bash 的 `timeout` 做"停止新工作"，但
  **CPU 草案的 timeout 没有 `kill-after`** —— 子进程忽略 TERM 时会越过小额预算；
* 本机**没有任何 POSIX shell**，所以 bash 里的超时行为**无法用 fixture 证明**。

把它挪到 Python 后，超时、终止、收尾都能在本地用短时、无模型的 fixture **真跑**
并断言。sbatch 草案改为调用本包装器，不再裸用 `timeout`。

行为
----
1. 启动子命令（**不用管道**，stdout/stderr 直接写日志文件，避免死锁）；
2. 等到 `--deadline-seconds`；
3. 未结束 → 终止：
   * POSIX：`start_new_session=True` + `killpg(SIGTERM)` → 等 `--kill-after` → `killpg(SIGKILL)`
   * Windows：`terminate()` → 等 `--kill-after` → `kill()`
4. 无论哪条路径，**收尾都有上限**（`--wrapup-seconds`），超了也要返回；
5. 把每个阶段事件追加写入 `--ledger`（JSONL，含 `time_source`）。

退出码
------
子进程自己的退出码；被 TERM 结束用 124；被 KILL 结束用 137。
参数/环境错误用 60+。

**未验证边界（必须声明）**：POSIX 的 SIGTERM/SIGKILL 语义**没有**在本机跑过
（本机是 Windows）。Windows 上 `terminate()` 走 TerminateProcess，
**无法被目标进程忽略**，所以"忽略 TERM 后升格 KILL"这条路径在 Windows 上
测不到。测试脚本会如实报告 `posix_signal_semantics_verified=false`。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time

EXIT_TERM = 124
EXIT_KILL = 137


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_ledger(path, record):
    if not path:
        return
    record = dict(record)
    record.setdefault("utc", utc_now())
    record.setdefault("time_source", "wrapper-wallclock-epoch")
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def terminate_child(proc, kill_after_seconds, ledger_path, deadline_hit_at, label=None):
    """终止并**等待**，返回 `(exit_code, stop_reason, escalated)`。

    每个事件都带 `label`，否则调用方按 label 过滤时会把这一段的账目漏掉
    （v4 初版就漏了 `terminate_sent`）。
    """
    def note(event, **kw):
        record = {"event": event, "label": label, "pid": proc.pid}
        record.update(kw)
        append_ledger(ledger_path, record)

    escalated = False
    if os.name == "posix":
        method = "killpg:SIGTERM"
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError) as exc:
            note("term_failed", error=str(exc))
            method = "term_failed"
    else:
        method = "terminate:TerminateProcess"
        try:
            proc.terminate()
        except OSError as exc:
            note("term_failed", error=str(exc))
            method = "term_failed"

    note("terminate_sent", method=method,
         seconds_after_deadline=round(time.time() - deadline_hit_at, 3))

    grace_started = time.time()
    grace_end = grace_started + kill_after_seconds
    while time.time() < grace_end:
        if proc.poll() is not None:
            code = proc.returncode
            note("exited_after_term", exit_code=code,
                 waited_seconds=round(time.time() - grace_started, 3))
            return EXIT_TERM, "stopped_by_term", escalated
        time.sleep(0.05)

    escalated = True
    if os.name == "posix":
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            kill_method = "killpg:SIGKILL"
        except (ProcessLookupError, PermissionError) as exc:
            note("kill_failed", error=str(exc))
            kill_method = "kill_failed"
    else:
        proc.kill()
        kill_method = "kill:TerminateProcess"

    note("kill_sent", method=kill_method)
    try:
        proc.wait(timeout=max(1.0, kill_after_seconds))
        exited = True
    except subprocess.TimeoutExpired:
        exited = False
    note("kill_settled", exited=exited)
    return EXIT_KILL, "stopped_by_kill", escalated


def main(argv=None):
    parser = argparse.ArgumentParser(description="KAGGLE-32 v4 截止控制包装器")
    parser.add_argument("--deadline-seconds", type=int, required=True)
    parser.add_argument("--kill-after", type=int, default=30)
    parser.add_argument("--wrapup-seconds", type=int, default=60)
    parser.add_argument("--ledger", default=None)
    parser.add_argument("--label", default="main")
    parser.add_argument("--stdout-file", default=None, help="子进程 stdout 直接写文件（不用管道）")
    parser.add_argument("--stderr-file", default=None, help="子进程 stderr 直接写文件（不用管道）")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = args.command
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("FATAL: 没有给出要运行的命令", file=sys.stderr)
        return 60
    if args.deadline_seconds <= 0:
        print("FATAL: --deadline-seconds 必须 > 0", file=sys.stderr)
        return 61

    started = time.time()
    append_ledger(args.ledger, {
        "event": "start", "label": args.label, "command": command,
        "deadline_seconds": args.deadline_seconds, "kill_after": args.kill_after,
        "wrapup_seconds": args.wrapup_seconds,
    })

    # 关键：不用 PIPE（避免管道死锁与沙箱管道限制）。输出直接写文件。
    kwargs = {}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    stream_handles = []
    if args.stdout_file:
        handle = open(args.stdout_file, "ab")
        stream_handles.append(handle)
        kwargs["stdout"] = handle
    if args.stderr_file:
        handle = open(args.stderr_file, "ab")
        stream_handles.append(handle)
        kwargs["stderr"] = handle
    proc = subprocess.Popen(command, **kwargs)

    deadline_at = started + args.deadline_seconds
    exit_code = None
    stop_reason = "completed_within_deadline"
    escalated = False

    while True:
        remaining = deadline_at - time.time()
        if remaining <= 0:
            break
        try:
            exit_code = proc.wait(timeout=remaining)
            break
        except subprocess.TimeoutExpired:
            continue

    if exit_code is None:
        deadline_hit_at = time.time()
        append_ledger(args.ledger, {
            "event": "deadline_hit", "label": args.label,
            "elapsed_seconds": round(deadline_hit_at - started, 3),
        })
        exit_code, stop_reason, escalated = terminate_child(
            proc, args.kill_after, args.ledger, deadline_hit_at, label=args.label
        )

    elapsed = time.time() - started
    # 收尾：把结论写账，并保证收尾时间有上限（这里只是记账，故远低于上限）
    wrapup_deadline = time.time() + args.wrapup_seconds
    append_ledger(args.ledger, {
        "event": "end", "label": args.label, "exit_code": exit_code,
        "stop_reason": stop_reason, "escalated_to_kill": escalated,
        "elapsed_seconds": round(elapsed, 3),
        "wrapup_within_budget": time.time() < wrapup_deadline,
        "total_seconds_upper_bound": args.deadline_seconds + args.kill_after + args.wrapup_seconds,
    })
    for handle in stream_handles:
        try:
            handle.close()
        except OSError:
            pass
    print("deadline_wrapper: label=%s exit=%s reason=%s elapsed=%.3fs"
          % (args.label, exit_code, stop_reason, elapsed))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
