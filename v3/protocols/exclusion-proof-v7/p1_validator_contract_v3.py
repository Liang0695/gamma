#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KAGGLE-27 · P1 探针 v3（修正版）：契约核对 + **真实子进程退出码**。

对 v2 探针的两处缺陷的修正（Mika 2026-10-06 裁定第 3 条）：
  1. v2 的 `OUT["errors"]` 全程恒为 `[]`，探针自身退出码永远是 0、不携带信息。
     本版为每个用例写**期望值**，实测与期望不符即累加 errors；main() 返回
     `1 if errors else 0`，探针自身失败时**非零退出**。
  2. v2 的 exit_code 来自 `getattr(exc, "exit_code")` —— 那是**类属性映射**，
     不是操作系统进程退出码。本版用 `subprocess.run(...).returncode` 实测，
     并在输出里把两种来源**分开命名**（exception_attribute / os_process）。

全部使用合成 fam-syn-* 标签；不接触真实 H 身份、不读受限正文、不改 E0 任何文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

SYN_DENY = ["fam-syn-d1", "fam-syn-d2", "fam-syn-h1"]

DIRECT_CHILD = r"""
import json, sys
src = sys.argv[1]
payload = json.load(sys.stdin)
sys.path.insert(0, src)
from v3.common.errors import FailClosed
from v3.data.dedup import assert_no_split_leak
try:
    assert_no_split_leak(payload["records"], payload["denylist"])
except FailClosed as exc:
    sys.stderr.write("[%s] %s\n" % (exc.code, exc.message))
    raise SystemExit(exc.exit_code)
raise SystemExit(0)
"""

OUT: dict = {
    "probe": "KAGGLE-27 P1-v3: frozen validator contract + real subprocess exit codes",
    "validator_pin": {},
    "normalization": {},
    "direct_cases": [],
    "production_entry_cases": [],
    "exit_code_provenance": {},
    "errors": [],
    "boundary": "全部合成 fam-syn-* 标签；未接触真实 H 身份、未读受限正文、未导出 denylist。",
}


def rec(fid, repo_family="repo-syn-x", split="train", **extra):
    d = {"task_id": "synth-task", "problem_family_id": fid,
         "repo_family": repo_family, "split": split}
    d.update(extra)
    return d


# ---------------------------------------------------------------- direct cases
# (tag, records, denylist, expected_os_exit_code, expected_error_code, note)
DIRECT_CASES = [
    ("clean_no_intersection", [rec("fam-syn-ok")], SYN_DENY, 0, None, "无交集基线"),
    ("intersection_problem_family", [rec("fam-syn-d1")], SYN_DENY, 7,
     "v2_denylist_intersection", "命中 problem_family_id"),
    ("intersection_repo_family", [rec("fam-syn-ok", repo_family="fam-syn-h1")], SYN_DENY, 7,
     "v2_denylist_intersection", "命中 repo_family"),
    ("intersection_case_and_space", [rec("  FAM-SYN-D1  ")], SYN_DENY, 7,
     "v2_denylist_intersection", "归一后仍命中（strip+lower）"),
    ("missing_key_problem_family", [{"task_id": "t", "repo_family": "repo-syn-x",
                                     "split": "train"}], SYN_DENY, 4,
     "split_leak_check_missing_family", "缺键"),
    ("missing_key_repo_family", [{"task_id": "t", "problem_family_id": "fam-syn-ok",
                                  "split": "train"}], SYN_DENY, 4,
     "split_leak_check_missing_family", "缺 repo_family"),
    ("null_problem_family", [rec(None)], SYN_DENY, 4, "split_leak_check_missing_family", "null"),
    ("empty_problem_family", [rec("")], SYN_DENY, 4, "split_leak_check_missing_family", "空串"),
    ("blank_problem_family", [rec("   ")], SYN_DENY, 4, "split_leak_check_missing_family", "纯空白"),
    ("sealed_still_checked", [rec("fam-syn-d1", split="sealed")], SYN_DENY, 7,
     "v2_denylist_intersection", "sealed 也纳入检查"),
    ("unknown_split_still_checked", [rec("fam-syn-d1", split="totally-unknown")], SYN_DENY, 7,
     "v2_denylist_intersection", "未知 split 也纳入检查"),
    ("legacy_test_split_still_checked", [rec("fam-syn-d1", split="test")], SYN_DENY, 7,
     "v2_denylist_intersection", "历史别名 test 也纳入检查"),
    ("empty_denylist_no_intersection", [rec("fam-syn-d1")], [], 0, None,
     "空 denylist 时什么都不命中（正是需要协议层整批拒绝的情形）"),
    ("empty_records_passes", [], SYN_DENY, 0, None, "空记录通过"),
    ("literal_none_vs_null", [rec("none")], ["none"], 7, "v2_denylist_intersection",
     "合法字面量 'none' 正确命中"),
    ("denylist_none_becomes_literal_none", [rec("none")], [None], 7,
     "v2_denylist_intersection", "【缺陷】null 项退化为字面 'none' 造成假阳性"),
    ("denylist_none_silently_no_hit", [rec("fam-syn-ok")], [None], 0, None,
     "【缺陷】null 项静默进入比对集合但不命中，调用方无任何丢项信号"),
    ("denylist_blank_silently_dropped", [rec("fam-syn-ok")],
     ["fam-syn-d1", "", "   ", "\t"], 0, None,
     "【缺陷】空白项被静默丢弃，无错误无计数"),
    ("denylist_int_str_coerced", [rec("42")], [42], 7, "v2_denylist_intersection",
     "【缺陷】int 被 str() 兜底成 '42' 并参与比对"),
    ("one_bad_among_good", [rec("fam-syn-ok"), {"task_id": "t2", "problem_family_id": "fam-syn-ok",
                                                "split": "train"}], SYN_DENY, 4,
     "split_leak_check_missing_family", "批量中存在一条缺字段即整批失败"),
    ("duplicates_do_not_break", [rec("fam-syn-ok"), rec("fam-syn-ok")],
     ["fam-syn-d1", "fam-syn-d1", "FAM-SYN-D1"], 0, None, "重复项不破坏判定"),
]


def run_direct(e0src: Path, records, denylist):
    payload = json.dumps({"records": records, "denylist": denylist}, ensure_ascii=False)
    p = subprocess.run([sys.executable, "-c", DIRECT_CHILD, str(e0src)],
                       input=payload.encode("utf-8"),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return p.returncode, p.stderr.decode("utf-8", "replace").strip()


# ------------------------------------------------------------ production entry


def run_cli(e0src: Path, registry: dict, denylist_obj, tmp: Path, use_denylist=True):
    """真实入口 `python -m v3.cli split`，返回 (os_exit_code, stderr_tail)。"""
    for rec_ in registry.get("tasks", []):
        rec_.setdefault("problem_statement", "syn statement for %s" % rec_.get("task_id"))
    reg = tmp / "syn-registry.json"
    reg.write_bytes((json.dumps(registry, ensure_ascii=False) + "\n").encode("utf-8"))
    cmd = [sys.executable, "-m", "v3.cli", "split", "--registry", str(reg),
           "--out", str(tmp / "syn-split-out.json")]
    if use_denylist:
        dl = tmp / "syn-denylist.json"
        if isinstance(denylist_obj, str):
            dl.write_bytes(denylist_obj.encode("utf-8"))
        else:
            dl.write_bytes((json.dumps(denylist_obj, ensure_ascii=False) + "\n").encode("utf-8"))
        cmd += ["--denylist", str(dl)]
    p = subprocess.run(cmd, cwd=str(e0src), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    detail = (p.stdout.decode("utf-8", "replace") + "\n" +
              p.stderr.decode("utf-8", "replace")).strip()
    return p.returncode, detail[-300:]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--e0src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tmp", default=None)
    args = ap.parse_args()

    src = Path(args.e0src).resolve()
    sys.path.insert(0, str(src))
    tmp = Path(args.tmp).resolve() if args.tmp else Path(args.out).resolve().parent / "_p1_tmp"
    tmp.mkdir(parents=True, exist_ok=True)

    vbytes = (src / "v3" / "data" / "dedup.py").read_bytes()
    OUT["validator_pin"] = {
        "path": "v3/data/dedup.py",
        "bytes": len(vbytes),
        "file_bytes_sha256": hashlib.sha256(vbytes).hexdigest(),
        "file_bytes_sha256_lf_normalized": hashlib.sha256(vbytes.replace(b"\r\n", b"\n")).hexdigest(),
        "newline_mode": "crlf" if b"\r\n" in vbytes else "lf",
        "git_commit_sha": "ffc37b4f3ecde6579117b0012ab4f69b4cff16ab",
        "entrypoint": "assert_no_split_leak",
        "python": sys.version,
    }
    if len(vbytes) != 12144:
        OUT["errors"].append("validator_pin: 字节数 %d != 12144，版本不是 ffc37b4f" % len(vbytes))

    # --- 归一行为
    from v3.data import dedup
    nrm = dedup._normalize_match_token
    def normalize_sample(value):
        try:
            return {"value": nrm(value)}
        except Exception as exc:  # probe records stable behavior; it must not abort before report output
            return {"error_code": getattr(exc, "code", type(exc).__name__)}
    OUT["normalization"] = {
        "function_source": "E0-pinned _normalize_match_token; exceptions are captured as structured probe output",
        "samples": {repr(s): normalize_sample(s) for s in
                    ["FAM-ALPHA", "  fam-alpha  ", "fam\u00a0alpha", "fam alpha", "fam  alpha",
                     "caf\u00e9", "cafe\u0301", "", "   ", None, 123]},
    }
    OUT["normalization"]["nfc_nfd_collapsed"] = (
        OUT["normalization"]["samples"][repr("caf\u00e9")].get("value")
        == OUT["normalization"]["samples"][repr("cafe\u0301")].get("value"))
    OUT["normalization"]["nbsp_equals_space"] = (
        OUT["normalization"]["samples"][repr("fam\u00a0alpha")].get("value")
        == OUT["normalization"]["samples"][repr("fam alpha")].get("value"))
    if OUT["normalization"]["samples"][repr(None)].get("value") != "none":
        OUT["errors"].append("normalization: None 的实际归一结果与记录值不符")

    # --- 直接调用：真实子进程退出码
    for tag, records, denylist, exp_code, exp_err, note in DIRECT_CASES:
        code, err = run_direct(src, records, denylist)
        observed_err = None
        for marker in ("[%s]" % exp_err,) if exp_err else ():
            if marker in err:
                observed_err = exp_err
        entry = {
            "tag": tag,
            "note": note,
            "denylist": [repr(d) for d in denylist],
            "expected_os_exit_code": exp_code,
            "os_process_exit_code": code,
            "exit_code_source": "os_process",
            "stderr_tail": err[-200:],
            "match": code == exp_code,
        }
        OUT["direct_cases"].append(entry)
        if code != exp_code:
            OUT["errors"].append("direct/%s: os_exit=%r 期望=%r" % (tag, code, exp_code))

    # --- 真实生产入口
    good_reg = {"tasks": [dict(rec("fam-syn-ok", repo_family="repo-syn-ok"), task_id="syn-1",
                               problem_statement="syn statement one"),
                          dict(rec("fam-syn-ok2", repo_family="repo-syn-ok2"), task_id="syn-2",
                               problem_statement="syn statement two")], "quota": {"*": 1}}
    inter_reg = {"tasks": [dict(rec("fam-syn-d1", repo_family="repo-syn-ok"), task_id="syn-1",
                                problem_statement="syn statement one")], "quota": {"*": 1}}

    v2_schema = {"denylist_schema_version": "v2-dh-exclusion-denylist-1",
                 "d_family_ids": ["fam-syn-d1"], "h_family_ids": ["fam-syn-h1"],
                 "exposed_family_ids": []}
    prod_schema = {"families": ["fam-syn-d1"], "repos": ["repo-syn-x"]}

    prod_cases = [
        ("no_denylist_arg", good_reg, None, False, 0,
         "【fail-open】--denylist 缺省（cli.py:298 未设 required=True）⇒ 不做排除也 PASS"),
        ("protocol_three_array_schema", inter_reg, v2_schema, True, 0,
         "【fail-open】喂协议三数组 schema、且记录真的相交 ⇒ 仍 exit 0，零交集是伪结论"),
        ("production_schema_intersection", inter_reg, prod_schema, True, 7,
         "生产 schema 下真交集可命中"),
        ("production_clean", good_reg, prod_schema, True, 0, "生产 schema 下无交集 PASS"),
        ("missing_field_registry", {"tasks": [{"task_id": "s", "problem_family_id": "fam-syn-ok"}],
                                    "quota": {"*": 1}}, prod_schema, True, 4,
         "记录缺 repo_family ⇒ MissingInput"),
        ("denylist_file_absent", good_reg, "not-a-path", True, 4, "denylist 路径不存在"),
        ("denylist_file_is_null", good_reg, "null", True, 1,
         "【未定义失败模式】denylist 内容为 null ⇒ AttributeError，退出码 1（非 FailClosed）"),
    ]
    for tag, reg_obj, deny, use, exp_code, note in prod_cases:
        if isinstance(deny, str) and deny not in ("null", "not-a-path"):
            pass
        if deny == "not-a-path":
            reg = tmp / "syn-registry2.json"
            reg.write_bytes((json.dumps(reg_obj, ensure_ascii=False) + "\n").encode("utf-8"))
            p = subprocess.run([sys.executable, "-m", "v3.cli", "split",
                                "--registry", str(reg), "--denylist", str(tmp / "does-not-exist.json"),
                                "--out", str(tmp / "o.json")], cwd=str(src),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            code, err = p.returncode, p.stderr.decode("utf-8", "replace").strip()[-300:]
        else:
            code, err = run_cli(src, reg_obj, deny, tmp, use_denylist=use)
        entry = {
            "tag": tag, "note": note,
            "expected_os_exit_code": exp_code,
            "os_process_exit_code": code,
            "exit_code_source": "os_process",
            "stderr_tail": err[-200:],
            "match": code == exp_code,
        }
        OUT["production_entry_cases"].append(entry)
        if code != exp_code:
            OUT["errors"].append("prod/%s: os_exit=%r 期望=%r" % (tag, code, exp_code))

    # --- 退出码来源对照
    from v3.common.errors import MissingInput, PolicyViolation
    OUT["exit_code_provenance"] = {
        "exception_attribute": {
            "MissingInput.exit_code": getattr(MissingInput, "exit_code", None),
            "PolicyViolation.exit_code": getattr(PolicyViolation, "exit_code", None),
            "note": "类属性映射，**不是**操作系统进程退出码；不得写成『进程退出 4/7』",
        },
        "os_process": {
            "clean": 0, "missing_field": 4, "intersection": 7,
            "measured_by": "subprocess.run(...).returncode",
        },
        "rule": "报告与 manifest 中必须分开标注 exception_attribute / os_process；ACCEPTED 只接受 os_process。",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_bytes((json.dumps(OUT, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))

    ok = sum(1 for c in OUT["direct_cases"] if c["match"])
    okp = sum(1 for c in OUT["production_entry_cases"] if c["match"])
    print("direct=%d/%d production=%d/%d errors=%d"
          % (ok, len(OUT["direct_cases"]), okp, len(OUT["production_entry_cases"]), len(OUT["errors"])))
    for e in OUT["errors"][:20]:
        print("  - " + e)
    return 1 if OUT["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
