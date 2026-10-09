#!/usr/bin/env python3
"""v6: one monotonic deadline, including pre-write setup and audit IO.

The parent performs no filesystem IO. Plan loading, guard/hash/path checks,
mkdir, phase log opening and ledger writing are killable child operations.
--job-mode is the actual sbatch entry; --plan is the no-model fixture entry.
Neither starts a fresh budget after setup. No kill grace extends the budget.
POSIX process-group cleanup is implemented but remains unverified on Windows.
"""
import time
ENTRY_STARTED = time.monotonic()

import argparse
import datetime
import json
import math
import os
import signal
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
HELPER = os.path.join(HERE, "shard_job.py")
# All are reserves INSIDE the budget, never additions to it.
AUDIT_RESERVE = 0.50
REAP_RESERVE = 0.15
SCHEDULING_RESERVE = 0.10
OBSERVATION_TOLERANCE = 0.20  # diagnostic only; over-budget still fails
MIN_PHASE_SECONDS = 0.02
CANCEL_SIGNAL = None
ACTIVE_WAIT = False


class JobSignal(Exception):
    pass


def handle_signal(number, frame):
    global CANCEL_SIGNAL
    CANCEL_SIGNAL = number
    if ACTIVE_WAIT:
        raise JobSignal(number)


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_plan(path):
    with open(path, encoding="utf-8") as handle:
        plan = json.load(handle)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if not isinstance(plan, dict) or not isinstance(plan.get("phases"), list):
        raise ValueError("plan must contain phases")
    phases = plan["phases"]
    # One required phase of each kind, in order. No wrapup can disappear.
    if [p.get("kind") for p in phases if isinstance(p, dict)] != ["prep", "main", "wrapup"]:
        raise ValueError("required order: prep, main, wrapup")
    for phase in phases:
        name = phase.get("name")
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
            raise ValueError("unsafe phase name")
        cmd = phase.get("command")
        if not isinstance(cmd, list) or not cmd or not all(isinstance(c, str) and c for c in cmd):
            raise ValueError("command must be a nonempty string list")
    if len({p["name"] for p in phases}) != 3:
        raise ValueError("duplicate phase name")


def send_signal(proc, force=False):
    try:
        if os.name == "posix":
            # Popen(start_new_session=True) guarantees pgid == pid; the group
            # can still exist after its leader exits.
            os.killpg(proc.pid, signal.SIGKILL if force else signal.SIGTERM)
        elif proc.poll() is None:
            proc.kill() if force else proc.terminate()
    except ProcessLookupError:
        pass


def terminate_child(proc, grace, hard_end, emit, phase):
    """Grace AND post-KILL reap must fit before hard_end."""
    send_signal(proc)
    grace_end = min(time.monotonic() + grace, hard_end - REAP_RESERVE)
    emit("terminate_sent", phase=phase, grace_seconds=max(0, grace_end-time.monotonic()))
    try:
        proc.wait(timeout=max(0, grace_end-time.monotonic()))
        escalated = False
    except (subprocess.TimeoutExpired, JobSignal):
        escalated = True
    # Also clean descendants after TERM killed just the group leader.
    if os.name == "posix" or escalated:
        send_signal(proc, force=True)
    try:
        proc.wait(timeout=max(0, min(REAP_RESERVE, hard_end-time.monotonic())))
        settled = True
    except (subprocess.TimeoutExpired, JobSignal):
        settled = False
    emit("kill_settled", phase=phase, escalated_to_kill=escalated, exited=settled,
         descendants_may_remain=(os.name != "posix"))
    return (137 if escalated else 124), ("stopped_by_kill" if escalated else "stopped_by_term"), settled


def run_child(command, execution_end, hard_end, grace, emit, phase, env=None, capture=False, input_text=None):
    """No file opens in parent; all potentially blocking IO is in children."""
    global ACTIVE_WAIT
    kwargs = {"env": env}
    if os.name == "posix":
        kwargs["start_new_session"] = True
    if capture:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                      encoding="utf-8", errors="replace")
    if input_text is not None:
        kwargs["stdin"] = subprocess.PIPE
    began = time.monotonic()
    try:
        proc = subprocess.Popen(command, **kwargs)
    except OSError as exc:
        return {"exit_code": 65, "stop_reason": "spawn_failed", "error": str(exc), "stdout": ""}
    emit("child_started", phase=phase, pid=proc.pid, execution_remaining_seconds=execution_end-time.monotonic())
    output = ""
    error = ""
    reason = "completed_within_deadline"
    settled = True
    try:
        ACTIVE_WAIT = True
        timeout = max(0, execution_end-time.monotonic())
        if capture or input_text is not None:
            output, error = proc.communicate(input=input_text, timeout=timeout)
        else:
            proc.wait(timeout=timeout)
        code = proc.returncode
    except (subprocess.TimeoutExpired, JobSignal) as exc:
        emit("phase_deadline_hit", phase=phase, elapsed_seconds=time.monotonic()-began)
        code, reason, settled = terminate_child(proc, grace, hard_end, emit, phase)
        if isinstance(exc, JobSignal):
            code, reason = 128+exc.args[0], "interrupted_by_signal"
        if capture:
            # No second unbounded communicate/wait. Reader data is unnecessary
            # for a timed-out setup; close pipes without waiting for descendants.
            proc.stdout.close()
            proc.stderr.close()
    finally:
        ACTIVE_WAIT = False
        if os.name == "posix":
            send_signal(proc, force=True)
    return {"exit_code": code, "stop_reason": reason, "stdout": output or "",
            "stderr": error or "", "elapsed_seconds": time.monotonic()-began,
            "child_settled": settled}


def build_parser():
    parser = argparse.ArgumentParser(description="v6 whole-job monotonic supervisor")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--job-mode", choices=("gpu", "cpu"))
    mode.add_argument("--plan")
    parser.add_argument("--total-budget", type=float)
    parser.add_argument("--started-monotonic", type=float,
                        help="trusted caller's original monotonic start; never a future time")
    parser.add_argument("--ledger")
    parser.add_argument("--run-dir")
    parser.add_argument("--kill-after", type=float, default=30)
    parser.add_argument("--prep-cap", type=float, default=120)
    parser.add_argument("--wrapup-cap", type=float, default=180)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, handle_signal)
    records = []
    start = args.started_monotonic if args.started_monotonic is not None else ENTRY_STARTED
    try:
        raw_budget = args.total_budget if args.total_budget is not None else float(os.environ.get("V3_TOTAL_BUDGET_SECONDS", ""))
        if not math.isfinite(raw_budget) or raw_budget <= 0:
            raise ValueError("positive finite total budget required")
        if not math.isfinite(start) or start > ENTRY_STARTED:
            raise ValueError("start must be a past monotonic time")
        if not all(math.isfinite(v) and v >= 0 for v in (args.kill_after, args.prep_cap, args.wrapup_cap)):
            raise ValueError("invalid phase/grace caps")
        budget = raw_budget
        if args.job_mode:
            # Provisional bound before guard; rejection is still guard's job.
            budget = min(budget, 13500 if args.job_mode == "gpu" else 1800)
            if args.job_mode == "gpu":
                hours = float(os.environ.get("V3_GPU_HOURS", ""))
                if not math.isfinite(hours) or hours <= 0:
                    raise ValueError("positive finite GPU hours required")
                budget = min(budget, hours*3600)
        deadline = start + budget
    except ValueError as exc:
        print(json.dumps({"event": "job_result", "overall_exit_code": 61,
                          "overall_status": "invalid_parameters", "error": str(exc)}))
        return 61

    def emit(event, **values):
        record = {"event": event, "utc": utc_now(), "time_source": "monotonic",
                  "total_elapsed_seconds": time.monotonic()-start,
                  "remaining_seconds": deadline-time.monotonic()}
        record.update(values)
        records.append(record)

    emit("supervisor_start", total_budget_seconds=budget, started_monotonic=start,
         deadline_monotonic=deadline, kill_after=args.kill_after,
         scheduling_reserve_seconds=SCHEDULING_RESERVE,
         observation_tolerance_seconds=OBSERVATION_TOLERANCE)
    # Setup includes plan loading or ALL job guard/hash/path/mkdir/plan work.
    setup_hard_end = min(start+args.prep_cap, deadline-AUDIT_RESERVE-args.wrapup_cap-SCHEDULING_RESERVE)
    setup_grace = min(args.kill_after, max(0, (setup_hard_end-time.monotonic()-REAP_RESERVE)/2))
    setup_execution_end = setup_hard_end-setup_grace-REAP_RESERVE
    command = [sys.executable, "-B", HELPER, "setup", json.dumps(vars(args))]
    ledger = args.ledger
    main_exit = None
    wrapup_exit = None
    aborted = False
    setup_code = 125
    all_settled = True
    if setup_execution_end <= time.monotonic():
        emit("setup_skipped", reason="budget_exhausted")
        setup = None
    else:
        setup_result = run_child(command, setup_execution_end, setup_hard_end, setup_grace, emit, "setup", capture=True)
        setup_code = setup_result["exit_code"]
        all_settled = setup_result.get("child_settled", True)
        emit("setup_end", exit_code=setup_code, stop_reason=setup_result["stop_reason"])
        setup = None
        if setup_code == 0:
            try:
                setup = json.loads(setup_result["stdout"])
                validate_plan(setup["plan"])
                # Tighten the SAME deadline; do not reset elapsed time.
                budget = min(budget, setup.get("total_budget", budget))
                deadline = start+budget
                ledger = setup["ledger"]
                args.prep_cap = min(args.prep_cap, setup.get("prep_cap", args.prep_cap))
                args.wrapup_cap = min(args.wrapup_cap, setup.get("wrapup_cap", args.wrapup_cap))
                args.kill_after = min(args.kill_after, setup.get("kill_after", args.kill_after))
                emit("setup_ready", deadline_monotonic=deadline, guard=setup.get("guard"))
            except (ValueError, KeyError, TypeError) as exc:
                setup_code = 62
                emit("setup_invalid", error=str(exc))
                setup = None
        else:
            emit("setup_output", stdout=setup_result["stdout"], stderr=setup_result.get("stderr"))

    if setup:
        env = dict(os.environ)
        env.update(setup.get("env", {}))
        for phase in setup["plan"]["phases"]:
            kind, name = phase["kind"], phase["name"]
            now = time.monotonic()
            # Prep cap includes setup elapsed; main reserves the full wrapup
            # cap. Every cap includes termination and KILL reap, not just work.
            future = args.wrapup_cap if kind != "wrapup" else 0
            hard_end = deadline-AUDIT_RESERVE-SCHEDULING_RESERVE-future
            if kind == "prep":
                hard_end = min(hard_end, start+args.prep_cap)
            if kind == "wrapup":
                hard_end = min(hard_end, now+args.wrapup_cap)
            grace = min(args.kill_after, max(0, (hard_end-now-REAP_RESERVE)/2))
            execution_end = hard_end-grace-REAP_RESERVE
            if (kind == "main" and (aborted or CANCEL_SIGNAL)) or execution_end-now < MIN_PHASE_SECONDS:
                emit("phase_skipped", phase=name, kind=kind,
                     reason="prep_failed_or_interrupted" if (aborted or CANCEL_SIGNAL) and kind == "main" else "budget_exhausted")
                if kind == "prep":
                    aborted = True
                continue
            emit("phase_start", phase=name, kind=kind,
                 deadline_seconds=execution_end-now, termination_grace_seconds=grace)
            # The IO helper opens phase logs, then invokes the command in the
            # SAME process group. Outer timeout also kills that shell's children.
            spec = {"command": phase["command"], "stdout_file": os.path.join(setup["run_dir"], name+".out")}
            if os.name == "posix":
                result = run_child([sys.executable, "-B", HELPER, "phase", json.dumps(spec)],
                                   execution_end, hard_end, grace, emit, name, env=env, capture=True)
                if result["stop_reason"] == "completed_within_deadline" and result["stdout"]:
                    result["exit_code"] = json.loads(result["stdout"])["raw_child_exit_code"]
            else:
                # Windows fixture branch starts the command directly, so
                # TerminateProcess stops the real fixture, not an IO proxy.
                # Arbitrary Windows descendants remain explicitly unverified.
                result = run_child(phase["command"], execution_end, hard_end,
                                   grace, emit, name, env=env, capture=True)
                emit("phase_output", phase=name, stdout_file=spec["stdout_file"],
                     stdout=result.get("stdout", ""), stderr=result.get("stderr", ""))
            all_settled = all_settled and result.get("child_settled", True)
            emit("phase_end", phase=name, kind=kind,
                 **{k: v for k, v in result.items() if k not in ("stdout", "stderr")})
            if kind == "prep" and result["exit_code"] != 0:
                aborted = True
            if kind == "main":
                main_exit = result["exit_code"]
            if kind == "wrapup":
                wrapup_exit = result["exit_code"]

    def outcome():
        if time.monotonic() > deadline:
            return 126, "total_budget_exceeded"
        if not all_settled:
            return 137, "child_cleanup_unsettled"
        if CANCEL_SIGNAL:
            return 128+CANCEL_SIGNAL, "interrupted_by_signal"
        if setup is None:
            return setup_code or 62, "setup_failed"
        if wrapup_exit is None:
            return 125, "required_wrapup_not_executed"
        if wrapup_exit != 0:
            return (wrapup_exit if wrapup_exit > 0 else 128-wrapup_exit), "required_wrapup_failed"
        if main_exit is None:
            return 125, "main_not_executed"
        if main_exit != 0:
            return (main_exit if main_exit > 0 else 128-main_exit), "main_failed"
        return 0, "succeeded"

    code, status = outcome()
    emit("supervisor_end", main_exit_code=main_exit, main_skipped=main_exit is None,
         wrapup_exit_code=wrapup_exit, overall_exit_code=code, overall_status=status,
         total_budget_seconds=budget, budget_respected=time.monotonic() <= deadline,
         tentative=True, finalized=False)
    # A failed guard does NOT authorize writes to unvalidated ledger/run paths.
    # Account via stdout + post-job sacct fallback in that case.
    ledger_written = False
    if setup and ledger:
        audit_hard_end = deadline-SCHEDULING_RESERVE
        audit_grace = min(0.05, max(0, (audit_hard_end-time.monotonic()-REAP_RESERVE)/2))
        audit_result = run_child([sys.executable, "-B", HELPER, "audit", ledger],
                                 audit_hard_end-audit_grace-REAP_RESERVE, audit_hard_end,
                                 audit_grace, emit, "audit", capture=True,
                                 input_text=json.dumps(records, ensure_ascii=False))
        if audit_result["exit_code"] != 0:
            code, status = 66, "audit_failed"
        else:
            ledger_written = True
        all_settled = all_settled and audit_result.get("child_settled", True)
    # Charge diagnostic output before measuring the authoritative final result.
    # Windows fixture output bodies are already in the ledger, not duplicated.
    for record in records:
        if record["event"] != "phase_output":
            print(json.dumps(record, ensure_ascii=False), flush=True)
    final_code, final_status = outcome()
    if final_code == 126 or (code == 0 and final_code != 0):
        code, status = final_code, final_status
    emit("job_result", main_exit_code=main_exit, wrapup_exit_code=wrapup_exit,
         overall_exit_code=code, overall_status=status, total_budget_seconds=budget,
         budget_respected=time.monotonic() <= deadline,
         within_observation_tolerance=time.monotonic() <= deadline+OBSERVATION_TOLERANCE,
         ledger_written=ledger_written,
         ledger_write_in_budget=(ledger_written and time.monotonic() <= deadline),
         posix_signal_semantics_verified=False, descendants_may_remain=(os.name != "posix"))
    # JSON stdout is the authoritative whole-job result, including audit time.
    print(json.dumps(records[-1], ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
