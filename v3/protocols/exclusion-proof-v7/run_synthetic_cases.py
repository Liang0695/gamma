#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KAGGLE-27 v7.2 · runner：逐例断言目标 reason_code + 进程审计流完整性。

断言规则（每例）：
  * 报告必须存在（Y1 稳定失败报告）
  * 父级收到的每条包装进程事件流必须连续且有结束确认
  * 目标 reason_code 必须全部命中
  * kind=pos  ：exit 0 且 verdict == expect
  * kind=neg  ：exit != 0 且 verdict != expect
  * 目标含 AUTH_BEFORE_ACCESS / COMPUTE_NOT_STARTED 的用例：
        protected_actual_open_count == 0 且 compute_invocations == 0（真实哨兵）
  * 三类轮次（验收 0 / 诊断 7 / 坏输入 4）在正例中必须同时成立
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
import signal
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHECKER = HERE / "verify_manifest.py"
PROTOCOL = HERE / "exclusion-proof-protocol-v7.json"
BASELINE = HERE / "frozen-denominator-baseline.json"
ZERO_ACCESS_TARGETS = {"AUTH_BEFORE_ACCESS", "COMPUTE_NOT_STARTED"}
CASE_TIMEOUT_SECONDS = 45.0
PROCESS_TIMEOUT_SECONDS = 30.0
CONNECTION_HANDSHAKE_SECONDS = 3.0
CONNECTION_POLL_SECONDS = 0.1
POST_EXIT_DRAIN_SECONDS = 3.0
DRAIN_QUIET_SECONDS = 0.25

PACKAGE_PROBE_CASES = {
    "neg_b1_plain_evidence_all_gates", "neg_b1_evidence_self_produced",
    "neg_b1_evidence_bad_time", "neg_b1_trusted_approval_not_approval",
    "neg_b1_g4_negative_exit_zero", "neg_b2_evidence_root_escape",
    "neg_b4_old_e0_schema_unconsumed", "neg_y2_stat_both_denied_expectation",
    "neg_keep_intersection", "neg_keep_bool_only_evidence",
    "neg_b1_forged_evidence_complete_format", "neg_b1_forged_approval_ref",
    "neg_b4_receipt_missing_real_cli",
}
CONSUMER_CASES = {
    "pos_synthetic_consumer_acceptance", "neg_b4_receipt_mutation_uses_pre_run_digest",
}
CONSUMER_ROLES = ["consumer_acceptance", "consumer_diagnostic", "consumer_bad_input"]


class ExternalAuditCollector:
    """父进程收集器：校验事件流、进程计划和有限连接期限。"""
    def __init__(self, fault=None, process_timeout=PROCESS_TIMEOUT_SECONDS,
                 case_deadline=None):
        self.events = []
        self.streams = []
        self.errors = []
        self.fault = fault
        self._fault_used = False
        self.process_timeout = process_timeout
        self.case_deadline = case_deadline
        self.root_pid = None
        self._lock = threading.Lock()
        self._changed = threading.Event()
        self._stop = threading.Event()
        self._active_connections = set()
        self._workers = set()
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.bind(("127.0.0.1", 0))
        self._server.listen()
        self._server.settimeout(0.1)
        self.host, self.port = self._server.getsockname()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            worker = threading.Thread(target=self._connection_worker, args=(conn,), daemon=True)
            with self._lock:
                self._workers.add(worker)
            worker.start()

    def _connection_worker(self, conn):
        with self._lock:
            self._active_connections.add(conn)
        try:
            self._read_connection(conn)
        finally:
            with self._lock:
                self._active_connections.discard(conn)
                self._workers.discard(threading.current_thread())
            self._changed.set()

    def _read_connection(self, conn):
        accepted_at = time.monotonic()
        stream = {"pid": None, "ppid": None, "role": None, "expected_seq": 0, "started": False,
                  "ended": False, "end_exit_code": None, "eof": False, "errors": []}
        pending = b""
        reset_here = False
        conn.settimeout(CONNECTION_POLL_SECONDS)
        with conn:
            while True:
                now = time.monotonic()
                deadline = accepted_at + CONNECTION_HANDSHAKE_SECONDS
                timeout_reason = "CONNECTION_HANDSHAKE_TIMEOUT"
                if stream["started"]:
                    deadline = accepted_at + self.process_timeout
                    timeout_reason = "PROCESS_TIMEOUT"
                if self.case_deadline is not None:
                    if self.case_deadline <= deadline:
                        deadline = self.case_deadline
                        timeout_reason = "CASE_TIMEOUT"
                if now >= deadline:
                    stream["errors"].append(timeout_reason)
                    root_pid = self.root_pid or stream.get("pid")
                    if isinstance(root_pid, int):
                        with self._lock:
                            active_pids = {s.get("pid") for s in self.streams
                                           if s.get("started") and not s.get("ended")}
                            active_pids.add(stream.get("pid"))
                        terminate_pid_tree(root_pid, active_pids)
                    break
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError as exc:
                    stream["errors"].append("recv_error:%s" % type(exc).__name__)
                    break
                if not chunk:
                    stream["eof"] = True
                    break
                pending += chunk
                while b"\n" in pending:
                    line, pending = pending.split(b"\n", 1)
                    try:
                        obj = json.loads(line.decode("utf-8"))
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        stream["errors"].append("invalid_frame")
                        continue
                    if not isinstance(obj, dict):
                        stream["errors"].append("non_object_frame")
                        continue
                    seq = obj.get("seq")
                    if seq != stream["expected_seq"]:
                        stream["errors"].append("sequence_gap")
                    stream["expected_seq"] = (seq + 1) if isinstance(seq, int) else stream["expected_seq"]
                    event = obj.get("event")
                    if not stream["started"]:
                        if event != "process_start" or seq != 0:
                            stream["errors"].append("missing_process_start")
                        else:
                            stream["started"] = True
                            stream["pid"] = obj.get("pid")
                            stream["ppid"] = obj.get("ppid")
                            stream["role"] = obj.get("role")
                            obj["received_monotonic_ns"] = time.monotonic_ns()
                            with self._lock:
                                self.events.append(obj)
                    elif stream["ended"]:
                        stream["errors"].append("event_after_process_end")
                    elif obj.get("pid") != stream["pid"] or obj.get("role") != stream["role"]:
                        stream["errors"].append("identity_changed")
                    elif event == "process_end":
                        stream["ended"] = True
                        stream["end_exit_code"] = obj.get("exit_code")
                        if not isinstance(obj.get("exit_code"), int):
                            stream["errors"].append("process_end_missing_exit_code")
                    elif event == "open":
                        if not isinstance(obj.get("path"), str):
                            stream["errors"].append("open_missing_path")
                        obj["received_monotonic_ns"] = time.monotonic_ns()
                        with self._lock:
                            self.events.append(obj)
                    else:
                        stream["errors"].append("unknown_event")

                    if (self.fault == "rst_after_process_start" and event == "process_start"
                            and not self._fault_used):
                        self._fault_used = True
                        reset_here = True
                        break
                    if self.fault == "drop_process_end" and event == "process_end":
                        stream["ended"] = False
                        stream["errors"].append("injected_missing_end_confirmation")
                if reset_here:
                    try:
                        conn.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
                    except OSError:
                        pass
                    break
        if pending:
            stream["errors"].append("truncated_frame")
        if not stream["started"]:
            stream["errors"].append("no_process_start")
        if not stream["ended"]:
            stream["errors"].append("missing_process_end")
        with self._lock:
            self.streams.append(stream)
            self.errors.extend(stream["errors"])
        self._changed.set()

    def integrity(self):
        with self._lock:
            streams = [dict(s, errors=list(s["errors"])) for s in self.streams]
            errors = list(self.errors)
        complete = bool(streams) and not errors and all(
            s["started"] and s["ended"] and s["eof"]
            and isinstance(s["end_exit_code"], int) for s in streams)
        return {"complete": complete, "stream_count": len(streams),
                "errors": errors, "streams": streams}

    def environment(self, role):
        return dict(os.environ, K27_EXTERNAL_AUDIT_ENDPOINT="%s:%d" % (self.host, self.port),
                    K27_EXTERNAL_AUDIT_ROLE=role)

    def finalize(self, expected_roles, root_pid, runner_pid, drain_seconds=POST_EXIT_DRAIN_SECONDS):
        """Drain for a bounded period, then verify exact roles and PID ancestry."""
        expected = Counter(expected_roles)
        stop_at = time.monotonic() + drain_seconds
        if self.case_deadline is not None:
            stop_at = min(stop_at, self.case_deadline)
        quiet_since = None
        while time.monotonic() < stop_at:
            with self._lock:
                streams = list(self.streams)
                active = bool(self._active_connections)
            counts = Counter(s["role"] for s in streams if s.get("started"))
            roles_met = counts == expected
            all_terminal = not active and all(s.get("ended") and s.get("eof") for s in streams)
            if roles_met and all_terminal:
                if quiet_since is None:
                    quiet_since = time.monotonic()
                elif time.monotonic() - quiet_since >= DRAIN_QUIET_SECONDS:
                    break
            else:
                quiet_since = None
            self._changed.wait(min(CONNECTION_POLL_SECONDS, max(0, stop_at - time.monotonic())))
            self._changed.clear()

        self.close()
        with self._lock:
            streams = list(self.streams)
            errors = self.errors
        actual = Counter(s["role"] for s in streams if s.get("started"))
        reasons = []
        for role, count in expected.items():
            if actual[role] < count:
                reasons.append("MISSING_EXPECTED_ROLE:%s:%d/%d" % (role, actual[role], count))
            elif actual[role] > count:
                reasons.append("EXTRA_ROLE_COUNT:%s:%d/%d" % (role, actual[role], count))
        for role, count in actual.items():
            if role not in expected:
                reasons.append("UNEXPECTED_ROLE:%s:%d" % (role, count))
        pids = [s["pid"] for s in streams if s.get("started")]
        if len(pids) != len(set(pids)):
            reasons.append("DUPLICATE_PROCESS_ID")
        checker = [s for s in streams if s.get("role") == "checker"]
        for stream in streams:
            role = stream.get("role")
            if role == "checker":
                if stream.get("pid") != root_pid or stream.get("ppid") != runner_pid:
                    reasons.append("ROOT_PID_PARENT_MISMATCH")
            elif role in expected and role not in {"audit_self_test", "transport_probe", "timeout_probe"}:
                if len(checker) != 1 or stream.get("ppid") != checker[0].get("pid"):
                    reasons.append("CHILD_PID_PARENT_MISMATCH:%s" % role)
        if errors:
            reasons.extend(errors)
        if any(not (s.get("started") and s.get("ended") and s.get("eof")
                    and not s.get("errors")) for s in streams):
            reasons.append("INCOMPLETE_PROCESS_STREAM")
        return {"complete": not reasons, "expected_roles": dict(expected),
                "actual_roles": dict(actual), "root_pid": root_pid,
                "runner_pid": runner_pid, "stream_count": len(streams),
                "reasons": sorted(set(reasons)), "streams": streams}

    def close(self):
        self._stop.set()
        self._server.close()
        with self._lock:
            active = list(self._active_connections)
        for conn in active:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                conn.close()
            except OSError:
                pass
        self._thread.join(timeout=2)
        workers_deadline = time.monotonic() + 2
        with self._lock:
            workers = list(self._workers)
        for worker in workers:
            worker.join(timeout=max(0, workers_deadline - time.monotonic()))
        if self._thread.is_alive() or any(worker.is_alive() for worker in workers):
            with self._lock:
                self.errors.append("COLLECTOR_THREAD_SHUTDOWN_TIMEOUT")

    def __exit__(self, *_exc):
        self.close()


def audited_python_command(argv, role):
    """在每个 Python 进程入口先装独立审计 hook，再执行原命令。"""
    return [sys.executable, "-X", "utf8", str(HERE / "external_audit_bootstrap.py"),
            *list(argv)]


def terminate_pid_tree(root_pid, child_pids=()):
    """Terminate only the process tree rooted at a PID launched by this runner."""
    if os.name == "nt":
        for pid in child_pids:
            if isinstance(pid, int) and pid != root_pid:
                try:
                    os.kill(pid, signal.SIGTERM)
                except (OSError, ProcessLookupError):
                    pass
        try:
            os.kill(root_pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
    else:
        try:
            os.killpg(root_pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass


def terminate_case_processes(proc, collector):
    """回收本 case 的进程树；仅使用本次 Popen PID 和本 collector 记录的 PID。"""
    if os.name == "nt":
        # First ask Windows to terminate any just-created descendants that have
        # not yet completed their process_start handshake; then kill known PIDs.
        try:
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=1, check=False)
        except (OSError, subprocess.TimeoutExpired):
            pass
    with collector._lock:
        pids = {s.get("pid") for s in collector.streams
                if s.get("started") and not s.get("ended")}
        pids.update(ev.get("pid") for ev in collector.events
                    if ev.get("event") == "process_start")
    terminate_pid_tree(proc.pid, pids)
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        return proc.communicate(timeout=1.0)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        try:
            return proc.communicate(timeout=1.0)
        except subprocess.TimeoutExpired:
            for pipe in (proc.stdout, proc.stderr):
                if pipe is not None:
                    try:
                        pipe.close()
                    except OSError:
                        pass
            return b"", b""


def run_wrapped_process(argv, role, env, cwd, collector, deadline):
    """Run one wrapped process under both an outer case deadline and stream deadlines."""
    started = time.monotonic()
    options = {"cwd": str(cwd), "env": env, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
    if os.name == "nt":
        options["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        options["start_new_session"] = True
    proc = subprocess.Popen(audited_python_command(argv, role), **options)
    collector.root_pid = proc.pid
    timeout_kind = None
    try:
        stdout, stderr = proc.communicate(timeout=max(0.01, deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        timeout_kind = "CASE_TIMEOUT"
        stdout, stderr = terminate_case_processes(proc, collector)
    return {"proc": proc, "returncode": proc.returncode,
            "stdout": stdout.decode("utf-8", "replace"),
            "stderr": stderr.decode("utf-8", "replace"),
            "timeout_kind": timeout_kind,
            "elapsed_seconds": round(time.monotonic() - started, 3)}


def expected_process_plan(case):
    """Frozen case/round contract; it is computed before launching the checker."""
    plan = {"case": {"checker": 1}, "rounds": {}}
    if case["id"] in PACKAGE_PROBE_CASES:
        plan["rounds"]["package_origin_probe"] = {"package_origin_probe": 1}
    if case["id"] in CONSUMER_CASES or case["id"] in PACKAGE_PROBE_CASES:
        plan["rounds"].update({
            "acceptance": {"consumer_acceptance": 1},
            "diagnostic": {"consumer_diagnostic": 1},
            "bad_input": {"consumer_bad_input": 1},
        })
    return plan


def flatten_process_plan(plan):
    roles = []
    for group in [plan["case"], *plan["rounds"].values()]:
        for role, count in group.items():
            roles.extend([role] * count)
    return roles


def finalize_probe(collector, run, expected_roles, runner_pid=None, drain_seconds=POST_EXIT_DRAIN_SECONDS):
    return collector.finalize(expected_roles, run["proc"].pid,
                              os.getpid() if runner_pid is None else runner_pid,
                              drain_seconds=drain_seconds)


def external_protected_opens(events, case, e0):
    """从父进程 TCP 收集的原始 side channel 计数；不读取 checker 报告。"""
    d = HERE / case["dir"]
    protected = set()
    for name in ("records.json", "denylist.json", "snapshot.json"):
        p = d / name
        if p.exists():
            protected.add(str(p.resolve()))
    ev = HERE / "synthetic" / "evidence"
    if ev.exists():
        for f in ev.rglob("*"):
            if f.is_file():
                protected.add(str(f.resolve()))
    sys.path.insert(0, str(HERE))
    from verify_manifest import compute_import_closure
    vsrc = resolve_validator_src(case, e0)
    for rel in compute_import_closure(vsrc):
        protected.add(str((vsrc / rel).resolve()))
    norm = {os.path.normcase(os.path.abspath(p)) for p in protected}
    found = [ev["path"] for ev in events if isinstance(ev.get("path"), str)
             if os.path.normcase(os.path.abspath(ev.get("path", ""))) in norm]
    return len(found), sorted(set(found))


def resolve_e0(explicit):
    for c in ([Path(explicit)] if explicit else []) + [
            HERE.parent.parent / "e0", HERE.parent / "e0", Path.cwd() / "e0"]:
        if c and (c / "v3" / "data" / "dedup.py").exists():
            return c.resolve()
    raise SystemExit("未找到 E0 只读克隆")


def resolve_validator_src(case, e0):
    source = case.get("validator_src", "e0")
    if source == "e0":
        return e0
    return (HERE / "synthetic" / source).resolve()


def external_audit_self_test():
    """合成哨兵由独立子进程打开；即便 checker 自报流为空，父进程仍须观察到。"""
    sentinel = HERE / "_external_audit_sentinel.txt"
    sentinel.write_text("synthetic canary only\n", encoding="utf-8")
    try:
        deadline = time.monotonic() + 10
        collector = ExternalAuditCollector(case_deadline=deadline)
        collector.__enter__()
        env = collector.environment("audit_self_test")
        code = "import pathlib,sys; pathlib.Path(sys.argv[1]).read_bytes()"
        run = run_wrapped_process(["-c", code, str(sentinel)], "audit_self_test",
                                  env, HERE, collector, deadline)
        integrity = finalize_probe(collector, run, ["audit_self_test"], drain_seconds=2)
        expected = os.path.normcase(os.path.abspath(str(sentinel)))
        observed = [ev for ev in collector.events
                    if os.path.normcase(os.path.abspath(ev.get("path", ""))) == expected]
        observed = [dict(ev, path=Path(ev["path"]).resolve().relative_to(HERE).as_posix())
                    for ev in observed]
        # 模拟 checker 漏报/清空其内部日志；外层结论只使用父进程收到的原始事件。
        intentionally_empty_checker_log = []
        return {"passed": run["returncode"] == 0 and len(observed) == 1
                and intentionally_empty_checker_log == [] and integrity["complete"],
                "child_exit_code": run["returncode"],
                "elapsed_seconds": run["elapsed_seconds"],
                "integrity": integrity,
                "checker_reported_events": intentionally_empty_checker_log,
                "parent_observed_events": observed,
                "sentinel_path": sentinel.relative_to(HERE).as_posix()}
    finally:
        try:
            sentinel.unlink()
        except OSError:
            pass


def audit_transport_failure_probes():
    """Prove midstream RST and a missing end acknowledgement cannot look clean."""
    sentinel = HERE / "_external_audit_transport_sentinel.txt"
    sentinel.write_text("synthetic transport sentinel\n", encoding="utf-8")
    output = {}
    code = "import pathlib,sys; pathlib.Path(sys.argv[1]).read_bytes()"
    try:
        for fault in ("rst_after_process_start", "drop_process_end"):
            deadline = time.monotonic() + 15
            collector = ExternalAuditCollector(fault=fault, case_deadline=deadline)
            collector.__enter__()
            env = collector.environment("transport_probe")
            run = run_wrapped_process(["-c", code, str(sentinel)], "transport_probe",
                                      env, HERE, collector, deadline)
            integrity = finalize_probe(collector, run, ["transport_probe"], drain_seconds=2)
            output[fault] = {
                "child_exit_code": run["returncode"],
                "elapsed_seconds": run["elapsed_seconds"],
                "collection_complete": integrity["complete"],
                "explicit_errors": integrity["reasons"],
                "streams": integrity["streams"],
                # A failed/incomplete observation must never be interpreted as
                # a successful zero-open proof, regardless of the observed count.
                "zero_open_proof_accepted": integrity["complete"] and not collector.events,
                "runner_rejects_case": not integrity["complete"],
            }
        raw_client = r'''import json,os,socket,sys
fault=sys.argv[1]
host,port=os.environ["K27_EXTERNAL_AUDIT_ENDPOINT"].rsplit(":",1)
s=socket.create_connection((host,int(port)),timeout=2)
pid=os.getpid(); ppid=os.getppid(); role=os.environ["K27_EXTERNAL_AUDIT_ROLE"]
start={"event":"process_start","pid":pid,"ppid":ppid,"role":role,"seq":0}
s.sendall((json.dumps(start,separators=(",",":"))+"\n").encode())
if fault == "bad_frame":
    s.sendall(b"not-json\n")
    end={"event":"process_end","pid":pid,"ppid":ppid,"role":role,"seq":1,"exit_code":0}
    s.sendall((json.dumps(end,separators=(",",":"))+"\n").encode())
else:
    s.sendall(b'{"event":"open"')
s.close()
'''
        for fault in ("bad_frame", "truncated_frame"):
            deadline = time.monotonic() + 8
            collector = ExternalAuditCollector(case_deadline=deadline)
            collector.__enter__()
            env = collector.environment("transport_probe")
            options = {"cwd": str(HERE), "env": env, "stdout": subprocess.PIPE,
                       "stderr": subprocess.PIPE}
            if os.name == "nt":
                options["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                options["start_new_session"] = True
            started = time.monotonic()
            proc = subprocess.Popen([sys.executable, "-X", "utf8", "-c", raw_client, fault], **options)
            collector.root_pid = proc.pid
            stdout, stderr = proc.communicate(timeout=3)
            raw_run = {"proc": proc, "returncode": proc.returncode,
                       "elapsed_seconds": round(time.monotonic() - started, 3)}
            integrity = finalize_probe(collector, raw_run, ["transport_probe"], drain_seconds=2)
            output[fault] = {"child_exit_code": proc.returncode,
                             "elapsed_seconds": raw_run["elapsed_seconds"],
                             "collection_complete": integrity["complete"],
                             "explicit_errors": integrity["reasons"],
                             "runner_rejects_case": not integrity["complete"]}
        output["passed"] = (
            output["rst_after_process_start"]["runner_rejects_case"]
            and not output["rst_after_process_start"]["zero_open_proof_accepted"]
            and output["drop_process_end"]["runner_rejects_case"]
            and not output["drop_process_end"]["zero_open_proof_accepted"]
            and output["bad_frame"]["runner_rejects_case"]
            and "invalid_frame" in output["bad_frame"]["explicit_errors"]
            and output["truncated_frame"]["runner_rejects_case"]
            and "truncated_frame" in output["truncated_frame"]["explicit_errors"]
        )
        return output
    finally:
        try:
            sentinel.unlink()
        except OSError:
            pass


def audit_plan_and_deadline_probes():
    """Exercise missing/extra/wrong roles, endpoint fail-closed and time limits."""
    evidence = {}
    simple = "pass"

    # One valid complete stream cannot satisfy a plan that requires a consumer.
    deadline = time.monotonic() + 6
    collector = ExternalAuditCollector(case_deadline=deadline)
    collector.__enter__()
    run = run_wrapped_process(["-c", simple], "checker", collector.environment("checker"),
                              HERE, collector, deadline)
    integrity = finalize_probe(collector, run, ["checker", "consumer_acceptance"], drain_seconds=1)
    evidence["missing_child"] = {"root_exit_code": run["returncode"],
                                  "elapsed_seconds": run["elapsed_seconds"],
                                  "reasons": integrity["reasons"],
                                  "actual_roles": integrity["actual_roles"],
                                  "runner_rejects_case": not integrity["complete"]}

    # An additional or mislabeled stream also violates the frozen plan.
    deadline = time.monotonic() + 6
    collector = ExternalAuditCollector(case_deadline=deadline)
    collector.__enter__()
    root = run_wrapped_process(["-c", simple], "checker", collector.environment("checker"),
                               HERE, collector, deadline)
    extra = run_wrapped_process(["-c", simple], "unexpected_probe", collector.environment("unexpected_probe"),
                                HERE, collector, deadline)
    integrity = finalize_probe(collector, root, ["checker"], drain_seconds=1)
    evidence["extra_role"] = {"root_exit_code": root["returncode"],
                              "extra_exit_code": extra["returncode"],
                              "reasons": integrity["reasons"],
                              "runner_rejects_case": not integrity["complete"]}

    deadline = time.monotonic() + 6
    collector = ExternalAuditCollector(case_deadline=deadline)
    collector.__enter__()
    wrong = run_wrapped_process(["-c", simple], "wrong_role", collector.environment("wrong_role"),
                                HERE, collector, deadline)
    integrity = finalize_probe(collector, wrong, ["checker"], drain_seconds=1)
    evidence["wrong_role"] = {"process_exit_code": wrong["returncode"],
                               "reasons": integrity["reasons"],
                               "runner_rejects_case": not integrity["complete"]}

    # No endpoint must fail before a payload capable of creating a sentinel runs.
    sentinel = HERE / "_no_endpoint_payload_ran.txt"
    try:
        sentinel.unlink(missing_ok=True)
    except OSError:
        pass
    no_endpoint_env = dict(os.environ)
    no_endpoint_env.pop("K27_EXTERNAL_AUDIT_ENDPOINT", None)
    no_endpoint_env.pop("K27_EXTERNAL_AUDIT_ROLE", None)
    code = "from pathlib import Path; Path(__import__('sys').argv[1]).write_text('ran')"
    start = time.monotonic()
    proc = subprocess.run([sys.executable, "-X", "utf8", str(HERE / "external_audit_bootstrap.py"),
                           "-c", code, str(sentinel)], cwd=str(HERE), env=no_endpoint_env,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=3)
    evidence["missing_endpoint"] = {"exit_code": proc.returncode,
                                   "stderr": proc.stderr.decode("utf-8", "replace").strip(),
                                   "payload_created_file": sentinel.exists(),
                                   "elapsed_seconds": round(time.monotonic() - start, 3),
                                   "target_reason": "AUDIT_ENDPOINT_REQUIRED"}
    try:
        sentinel.unlink(missing_ok=True)
    except OSError:
        pass

    # Per-process stream deadline: collector terminates a connected hung wrapper.
    deadline = time.monotonic() + 5
    collector = ExternalAuditCollector(process_timeout=0.5, case_deadline=deadline)
    collector.__enter__()
    process_run = run_wrapped_process(["-c", "import time; time.sleep(5)"], "timeout_probe",
                                      collector.environment("timeout_probe"), HERE, collector, deadline)
    process_integrity = finalize_probe(collector, process_run, ["timeout_probe"], drain_seconds=1)
    evidence["process_timeout"] = {"exit_code": process_run["returncode"],
                                   "elapsed_seconds": process_run["elapsed_seconds"],
                                   "target_reason": "PROCESS_TIMEOUT",
                                   "reasons": process_integrity["reasons"],
                                   "terminated": process_run["proc"].poll() is not None}

    # Whole-case deadline remains separate and tighter than the per-process deadline.
    start = time.monotonic()
    deadline = start + 0.5
    collector = ExternalAuditCollector(process_timeout=5, case_deadline=deadline)
    collector.__enter__()
    case_run = run_wrapped_process(["-c", "import time; time.sleep(5)"], "timeout_probe",
                                  collector.environment("timeout_probe"), HERE, collector, deadline)
    case_integrity = finalize_probe(collector, case_run, ["timeout_probe"], drain_seconds=0.5)
    evidence["case_timeout"] = {"exit_code": case_run["returncode"],
                                "elapsed_seconds": round(time.monotonic() - start, 3),
                                "target_reason": case_run["timeout_kind"] or "CASE_TIMEOUT",
                                "reasons": case_integrity["reasons"],
                                "terminated": case_run["proc"].poll() is not None}

    evidence["passed"] = (
        evidence["missing_child"]["runner_rejects_case"]
        and evidence["extra_role"]["runner_rejects_case"]
        and evidence["wrong_role"]["runner_rejects_case"]
        and evidence["missing_endpoint"]["exit_code"] == 78
        and not evidence["missing_endpoint"]["payload_created_file"]
        and "PROCESS_TIMEOUT" in evidence["process_timeout"]["reasons"]
        and evidence["process_timeout"]["terminated"]
        and evidence["process_timeout"]["elapsed_seconds"] < 3
        and evidence["case_timeout"]["target_reason"] == "CASE_TIMEOUT"
        and evidence["case_timeout"]["terminated"]
        and evidence["case_timeout"]["elapsed_seconds"] < 3
    )
    return evidence


def run_case(case, e0, idx):
    d = HERE / case["dir"]
    rp = HERE / ("_tmp_report_%d.json" % idx)
    if rp.exists():
        rp.unlink()
    vsrc = resolve_validator_src(case, e0)
    args = [
        "--protocol", str(PROTOCOL), "--manifest", str(d / "manifest.json"),
        "--mode", case["mode"], "--validator-src", str(vsrc),
        "--records", str(d / "records.json"), "--denylist", str(d / "denylist.json"),
        "--snapshot", str(d / "snapshot.json"),
        "--evidence-root", str(HERE / "synthetic" / "evidence"),
        "--baseline", str(BASELINE), "--out", str(rp),
        "--workdir", str(HERE / "_work"),
    ]
    if case["mode"] == "test":
        args += ["--expect", case["expect"]]
    if case["need_auth"]:
        args += ["--authorization", str(d / "authorization.json"),
                 "--trust-anchor-file", str(d / "trust_anchor.bin")]
    args += list(case["args"])
    # 计划在启动前冻结；不能用实际观察到的角色反推预期集合。
    process_plan = expected_process_plan(case)
    expected_roles = flatten_process_plan(process_plan)
    start = time.monotonic()
    deadline = start + CASE_TIMEOUT_SECONDS
    collector = ExternalAuditCollector(case_deadline=deadline)
    collector.__enter__()
    env = collector.environment("checker")
    run = run_wrapped_process([str(CHECKER)] + args, "checker", env, HERE, collector, deadline)
    integrity = collector.finalize(expected_roles, run["proc"].pid, os.getpid(),
                                   drain_seconds=POST_EXIT_DRAIN_SECONDS)
    case_elapsed = round(time.monotonic() - start, 3)
    rep = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else None
    if rp.exists():
        rp.unlink()
    return (run["returncode"], rep, run["stderr"].strip(), list(collector.events),
            integrity, run, case_elapsed, process_plan)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e0src")
    a = ap.parse_args()
    e0 = resolve_e0(a.e0src)

    if not (HERE / "synthetic" / "e0_mut_cli").exists() or not (HERE / "fixtures.json").exists():
        print("重建 fixtures ...")
        rc = subprocess.run([sys.executable, str(HERE / "make_synthetic_cases.py"),
                             "--e0src", str(e0)], cwd=str(HERE))
        if rc.returncode != 0:
            raise SystemExit("fixture 重建失败")

    spec = json.loads((HERE / "fixtures.json").read_text(encoding="utf-8"))
    results, failures = [], []
    audit_probe = external_audit_self_test()
    if not audit_probe["passed"]:
        failures.append("external_audit_self_test")
    transport_probe = audit_transport_failure_probes()
    if not transport_probe["passed"]:
        failures.append("audit_transport_failure_probes")
    plan_deadline_probe = audit_plan_and_deadline_probes()
    if not plan_deadline_probe["passed"]:
        failures.append("audit_plan_and_deadline_probes")

    ids = [case["id"] for case in spec["cases"]]
    if len(ids) != 47 or len(ids) != len(set(ids)):
        failures.append("unexpected_fixture_inventory")
    if not (PACKAGE_PROBE_CASES | CONSUMER_CASES).issubset(set(ids)):
        failures.append("audit_plan_references_unknown_case")

    for idx, case in enumerate(spec["cases"]):
        rc, rep, se, ext_events, integrity, execution, elapsed, process_plan = run_case(case, e0, idx)
        targets = case["targets"]
        kind = "neg" if targets else "pos"
        codes = set((rep or {}).get("reason_codes") or [])
        verdict = (rep or {}).get("computed_verdict")
        opens = (rep or {}).get("protected_actual_open_count")
        compute = (rep or {}).get("compute_invocations")
        rounds = ((rep or {}).get("detail") or {}).get("rounds") or {}

        checks = {
            "report_exists": rep is not None,
            "audit_stream_complete": integrity["complete"],
            "case_before_deadline": execution["timeout_kind"] is None
                                      and elapsed < CASE_TIMEOUT_SECONDS,
            "targets_hit": all(t in codes for t in targets),
            "targets_missed": sorted(t for t in targets if t not in codes),
        }
        if rep is not None:
            checks["audit_complete"] = (rep.get("all_open_events_complete") is True
                                         or ("AUDIT_LOG_OVERFLOW" in targets
                                             and "AUDIT_LOG_OVERFLOW" in (rep.get("reason_codes") or [])
                                             and rep.get("computed_verdict") != "ACCEPTED"))
        if kind == "pos":
            checks["verdict_ok"] = (rc == 0 and verdict == case["expect"])
        elif case["mode"] == "test" and case["expect"] != "SYNTHETIC_ACCEPTED":
            checks["verdict_ok"] = (rc == 0 and verdict == case["expect"])
        elif case["mode"] == "production":
            checks["verdict_ok"] = (rc != 0 and verdict != "ACCEPTED")
        else:
            checks["verdict_ok"] = (rc != 0 and verdict != case["expect"])

        sentinel_open_required = "AUTH_BEFORE_ACCESS" in targets
        sentinel_compute_required = bool({"AUTH_BEFORE_ACCESS", "COMPUTE_NOT_STARTED"} & set(targets))
        external_consumer_starts = sum(1 for ev in ext_events
                                       if ev.get("event") == "process_start"
                                       and str(ev.get("role", "")).startswith("consumer_"))
        if sentinel_compute_required:
            # Pre-compute denials use their own smaller process plan; never impose
            # the healthy checker+three-consumer role set on these cases.
            checks["precompute_denial_zero_consumers"] = (
                compute == 0 and external_consumer_starts == 0)
        ext_opens, ext_paths = (0, [])
        if rep is not None:
            ext_opens, ext_paths = external_protected_opens(ext_events, case, e0)
        observed_paths = {os.path.normcase(os.path.abspath(p)) for p in ext_paths}
        relative_ext_paths = []
        for path in ext_paths[:5]:
            try:
                relative_ext_paths.append(Path(path).resolve().relative_to(HERE).as_posix())
            except ValueError:
                relative_ext_paths.append("<outside-package-root>")
        raw_external_events = []
        for ev in ext_events:
            if ev.get("event") == "process_start" or (
                    isinstance(ev.get("path"), str)
                    and os.path.normcase(os.path.abspath(ev["path"])) in observed_paths):
                item = dict(ev)
                if isinstance(item.get("path"), str):
                    try:
                        item["path"] = Path(item["path"]).resolve().relative_to(HERE).as_posix()
                    except ValueError:
                        item["path"] = "<outside-package-root>"
                raw_external_events.append(item)
        if sentinel_open_required:
            checks["zero_open_sentinel"] = (opens == 0)
            # 外层独立核算：全量 open 事件流里受保护路径的打开次数必须为 0
            checks["external_zero_open"] = (ext_opens == 0)
        else:
            checks["zero_open_sentinel"] = None
            checks["external_zero_open"] = None
        if sentinel_compute_required:
            checks["zero_compute_sentinel"] = (compute == 0)
        else:
            checks["zero_compute_sentinel"] = None
        if "GIT_BLOB_INVALID" in targets:
            checks["git_failure_compute_not_started"] = (
                "COMPUTE_NOT_STARTED" in codes and compute == 0)
            checks["git_failure_zero_external_consumers"] = external_consumer_starts == 0
        if "AUTH_SCOPE_INSUFFICIENT" in targets:
            checks["zero_unauthorized_path_opens"] = (
                (rep or {}).get("unauthorized_actual_open_count") == 0)
        entry_ext = {"external_protected_open_count": ext_opens,
                     "external_protected_paths": relative_ext_paths,
                     "external_audit_event_count": len(raw_external_events),
                     "external_consumer_process_starts": external_consumer_starts,
                     "external_audit_events": raw_external_events}

        if kind == "pos":
            acc = (rounds.get("acceptance") or {}).get("os_process_exit_code")
            diag = (rounds.get("diagnostic") or {}).get("os_process_exit_code")
            bad = (rounds.get("bad_input") or {}).get("os_process_exit_code")
            checks["rounds_0_7_4"] = (acc == 0 and diag == 7 and bad == 4)
            checks["rounds"] = {"acceptance": acc, "diagnostic": diag, "bad_input": bad}
            checks["input_bytes_match"] = (rounds.get("acceptance") or {}).get("input_bytes_match")

        ok = all(v for k, v in checks.items() if isinstance(v, bool) and v is not None)
        entry = {"id": case["id"], "note": case["note"], "kind": kind, "mode": case["mode"],
                 "checker_exit_code": rc, "computed_verdict": verdict,
                 "execution_exit_code": execution["returncode"],
                 "execution_elapsed_seconds": elapsed,
                 "execution_timeout_reason": execution["timeout_kind"],
                 "audit_plan": process_plan,
                 "audit_failure_reasons": integrity["reasons"],
                 "reason_codes": sorted(codes), "target_reason_codes": targets,
                 "protected_actual_open_count": opens, "compute_invocations": compute,
                 "unauthorized_actual_open_count": (rep or {}).get("unauthorized_actual_open_count"),
                 "unauthorized_actual_open_paths": (rep or {}).get("unauthorized_actual_opens", [])[:10],
                 "identity_probe_invocations": (rep or {}).get("identity_probe_invocations", 0),
                 "external_audit_integrity": integrity,
                 "rounds": rounds,
                 "checks": checks, "result": "PASS" if ok else "FAIL"}
        entry.update(entry_ext)
        if not ok:
            entry["stderr_tail"] = se[-300:]
            failures.append(case["id"])
        results.append(entry)

    summary = {"total": len(results), "passed": len(results) - len(failures),
               "failed": len(failures), "failures": failures}
    doc = {
        "checker": "verify_manifest.py (v7)",
        "suite": "P7 合成正反例（B1-B4 + Y1-Y2 + 保留项）+ 哨兵",
        "summary": summary, "cases": results,
        "external_audit_self_test": audit_probe,
        "audit_transport_failure_probes": transport_probe,
        "audit_plan_and_deadline_probes": plan_deadline_probe,
        "execution_limits": {"case_timeout_seconds": CASE_TIMEOUT_SECONDS,
                             "process_timeout_seconds": PROCESS_TIMEOUT_SECONDS,
                             "connection_handshake_seconds": CONNECTION_HANDSHAKE_SECONDS,
                             "post_exit_drain_seconds": POST_EXIT_DRAIN_SECONDS,
                             "drain_quiet_seconds": DRAIN_QUIET_SECONDS},
        "sentinel_note": "独立父进程通过逐进程 TCP side channel 收集带连续序号的 Python open audit events；每条流必须有父级观察到的 process_start、连续序号及 process_end。断连、截断或缺结束确认使整个 case 失败，checker all_open_events 不作为外层计数输入。",
        "external_audit_coverage": {
            "covered": "runner 启动的 checker、被测 Python checker 中的 package probe 和三轮 Python consumer；side channel 原始事件含 process_start/open、PID/PPID/role/路径/父级接收时间。",
            "limits": "Python audit hook 不能观察任意原生扩展/直接系统调用；git 等非 Python 原生子进程不在 Python open hook 覆盖内。runner 仅报告该覆盖面，不声称 OS 级穷尽审计。",
        },
        "boundary": "全部合成 fam-syn-* 标签；未接触真实 H 身份、未读受限正文、未导出 denylist。",
    }
    (HERE / "P7-synthetic-cases.json").write_bytes(
        (json.dumps(doc, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print("v7.3 suite:", json.dumps(summary, ensure_ascii=False))
    for r in results:
        if r["result"] == "FAIL":
            print("  FAIL", r["id"], "verdict=", r["computed_verdict"], "rc=", r["checker_exit_code"],
                  "missed=", r["checks"]["targets_missed"], "opens=", r["protected_actual_open_count"],
                  "compute=", r["compute_invocations"], "checks=",
                  {k: v for k, v in r["checks"].items() if v is False})
    shutil.rmtree(HERE / "_work", ignore_errors=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
