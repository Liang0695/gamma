#!/usr/bin/env python3
"""Killable IO/setup worker for shard_supervisor.py; never run standalone jobs.

Imports, realpath/hash checks, guard, mkdir, plan preparation, phase log opens
and tentative ledger writes run here, outside the supervising parent process.
"""
import contextlib
import datetime
import hashlib
import io
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def prepare_job(mode):
    import shard_guard as guard
    env = os.environ
    home = os.path.expanduser("~")
    repo = env.get("V3_REPO", os.path.join(home, "v3", "gamma"))
    stage = env.get("V3_STAGE", "start" if mode == "gpu" else "preflight")
    arguments = ["--mode", mode, "--repo", repo,
                 "--out-root", env.get("V3_OUT_ROOT", os.path.join(home, "v3", "runs")),
                 "--ledger-dir", env.get("V3_LEDGER_DIR", os.path.join(home, "v3", "ledger")),
                 "--stage", stage, "--total-budget-seconds", env.get("V3_TOTAL_BUDGET_SECONDS", ""),
                 "--already-elapsed", "0"]
    if mode == "gpu":
        arguments += ["--gpu-hours", env.get("V3_GPU_HOURS", ""),
                      "--approval-evidence", env.get("V3_APPROVAL_EVIDENCE", ""),
                      "--approval-expected-sha256", env.get("V3_APPROVAL_EXPECTED_SHA256", ""),
                      "--approval-evidence-ref", env.get("V3_APPROVAL_EVIDENCE_REF", "")]
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        code = guard.main(arguments)
    if code:
        print(output.getvalue())
        raise SystemExit(code)
    checked = json.loads(output.getvalue())
    if mode == "gpu" and not (checked.get("approval_verified") is True and
                               (checked.get("approval") or {}).get("approval_id")):
        raise SystemExit(44)
    # No writes have occurred. Use guard-returned canonical paths exclusively.
    guard.check_abs_outside_repo(HERE, "TOOLS_DIR", checked["repo_real"])
    root, ledger_dir = checked["out_root"], checked["ledger_dir"]
    run_id = env.get("SLURM_JOB_ID", "local")
    if not run_id.isascii() or not run_id.isdigit() and run_id != "local":
        raise ValueError("unsafe SLURM_JOB_ID")
    run_id += "-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = os.path.join(root, stage, run_id)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=False)
    os.makedirs(ledger_dir, exist_ok=True)
    py = env.get("V3_PYTHON", os.path.join(home, "miniconda3", "envs", "h3", "bin", "python"))
    with open(os.path.join(HERE, "stage_scripts.json"), encoding="utf-8") as handle:
        scripts = json.load(handle)[mode]
    bindings = {"V3_RUN_DIR": run_dir, "V3_REPO_DIR": checked["repo_real"],
                "V3_PIN_COMMIT": guard.PIN_COMMIT, "V3_PY": py,
                "V3_ENTRY_FLAG": "--"+stage,
                "V3_OPERATOR": env.get("V3_OPERATOR", ""),
                "V3_GPU_HOURS_ARG": env.get("V3_GPU_HOURS", ""),
                "V3_MEMORY_PROFILE": env.get("V3_MEMORY_PROFILE", ""),
                "V3_EXPORT_MANIFEST": env.get("V3_EXPORT_MANIFEST", "")}
    shell = env.get("V3_SHELL", "bash")
    plan = {"phases": [
        {"name": "prep", "kind": "prep", "command": [shell, "-c", scripts["prep"]]},
        {"name": "main", "kind": "main", "command": [shell, "-c", scripts["main"]]},
        {"name": "wrapup", "kind": "wrapup", "command": [sys.executable, "-B", __file__, "hash", run_dir]},
    ]}
    with open(os.path.join(run_dir, "plan.json"), "w", encoding="utf-8") as handle:
        json.dump(plan, handle, ensure_ascii=False, indent=2)
    return {"plan": plan, "env": bindings, "run_dir": run_dir,
            "ledger": os.path.join(run_dir, "tentative-ledger.jsonl"),
            "total_budget": checked["supervisor_total_budget_seconds"],
            "prep_cap": checked["supervisor_prep_cap_seconds"],
            "wrapup_cap": checked["supervisor_wrapup_cap_seconds"],
            "kill_after": checked["supervisor_kill_after_seconds"], "guard": checked}


def hash_artifacts(run_dir):
    # Required wrapup: any list/read/write/hash error propagates nonzero.
    # No WARN-and-success branch, and only publish the final manifest atomically.
    temp_path = os.path.join(run_dir, "sha256-final.txt.tmp")
    final_path = os.path.join(run_dir, "sha256-final.txt")
    def walk_error(error):
        raise error
    with open(temp_path, "w", encoding="utf-8") as output:
        for root, dirs, files in os.walk(run_dir, onerror=walk_error):
            dirs.sort()
            for name in sorted(files):
                if name.startswith("sha256-") or name.endswith((".err", ".jsonl", ".out")):
                    continue
                path = os.path.join(root, name)
                digest = hashlib.sha256()
                with open(path, "rb") as handle:
                    for chunk in iter(lambda: handle.read(1024*1024), b""):
                        digest.update(chunk)
                output.write(digest.hexdigest()+"  "+os.path.relpath(path, run_dir)+"\n")
    os.replace(temp_path, final_path)


def main():
    action, value = sys.argv[1:3]
    if action == "setup":
        args = json.loads(value)
        if args["job_mode"]:
            payload = prepare_job(args["job_mode"])
        else:
            from shard_supervisor import load_plan
            if not args["run_dir"] or not os.path.isdir(args["run_dir"]):
                raise ValueError("run_dir must exist")
            payload = {"plan": load_plan(args["plan"]), "run_dir": args["run_dir"],
                       "ledger": args["ledger"]}
        print(json.dumps(payload, ensure_ascii=False), flush=True)
    elif action == "phase":
        spec = json.loads(value)
        with open(spec["stdout_file"], "ab") as handle:
            # Same group as this worker: supervisor can kill shell descendants.
            proc = subprocess.Popen(spec["command"], stdout=handle, stderr=handle)
            code = proc.wait()
            print(json.dumps({"raw_child_exit_code": code}), flush=True)
            if code < 0:
                code = 128-code
            return code
    elif action == "audit":
        records = json.load(sys.stdin)
        for record in records:
            if record.get("event") == "phase_output":
                with open(record["stdout_file"], "a", encoding="utf-8") as output:
                    output.write(record.get("stdout", "")+record.get("stderr", ""))
        with open(value, "a", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False)+"\n")
    elif action == "hash":
        hash_artifacts(value)
    else:
        raise ValueError("unknown worker action")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(62 if sys.argv[1] == "setup" else 67)
