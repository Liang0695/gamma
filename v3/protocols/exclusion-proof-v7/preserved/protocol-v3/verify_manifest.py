#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KAGGLE-27 · 排除证明 execution manifest 机器验收器（v3）。

设计要点（对应 Mika 2026-10-06 裁定）：
  * ACCEPTED 不是人填的布尔，而是本脚本对合取式逐项求值的结果。
  * 绑定摘要必须与**实际执行文件字节**一致（本脚本自行重算，不采信 manifest 数值）。
  * denylist 三数组契约整批校验：任一项非法 ⇒ 整批失败，不过滤、不 str() 兜底。
  * 退出码取**真实子进程 OS 退出码**，并区分 exception_attribute（异常属性映射）。
  * 合成模式显式区分：synthetic_mode=true 时最多给 SYNTHETIC_ACCEPTED，永不给 ACCEPTED。
  * COMPUTATION_ONLY 不是访问授权：isolation_ready=false 时禁止绑定真实 denylist。

退出码（本脚本自身）：
  0 = 机器判定结果与期望一致
  1 = 不一致（例如 manifest 自填 ACCEPTED 但合取式不成立）
  2 = 无法完成判定（缺输入/协议摘要不符/契约非法）

用法：
  python verify_manifest.py --manifest M.json --protocol P.json \\
      --records records.json --denylist denylist.json --validator-src ./e0 \\
      --out report.json [--expect ACCEPTED] [--write-overall]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_CANNOT = 2

SUPPORTED_DENYLIST_SCHEMA = ["v2-dh-exclusion-denylist-1"]
REQUIRED_ARRAYS = ["d_family_ids", "h_family_ids", "exposed_family_ids"]
ALLOWED_TOP_KEYS = set(
    REQUIRED_ARRAYS
    + ["denylist_schema_version", "source_sha256", "counts_declared", "produced_by", "produced_at"]
)
PERMITTED_MECHANISMS = {
    "independent_host",
    "controlled_vm",
    "separate_os_account_with_broken_acl_inheritance",
}

# 子进程内实际调用冻结验证器；退出码为真实 OS 退出码。
_CHILD = r"""
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


# --------------------------------------------------------------------- helpers


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_json_sha256(obj) -> str:
    return sha256_bytes(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def newline_mode(b: bytes) -> str:
    crlf = b.count(b"\r\n")
    lf = b.count(b"\n") - crlf
    if crlf and lf:
        return "mixed"
    if crlf:
        return "crlf"
    return "lf"


def normalize_token(value) -> str:
    """与冻结验证器 _normalize_match_token 同口径：str(value).strip().lower()。"""
    return str(value).strip().lower()


def load_json_bytes(path: Path):
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8")), raw


# ------------------------------------------------------------------ denylist


def validate_denylist(payload):
    """整批校验三数组契约。返回 (ok, violations, normalized_union, per_array_stats)。"""
    v = []
    if not isinstance(payload, dict):
        v.append("top_level_shape: 必须是 JSON object，实际 %s" % type(payload).__name__)
        return False, v, [], {}

    extra = sorted(set(payload.keys()) - ALLOWED_TOP_KEYS)
    if extra:
        v.append("unknown_top_level_key: %s（MissingInput(denylist_unknown_key)）" % extra)

    sv = payload.get("denylist_schema_version")
    if sv not in SUPPORTED_DENYLIST_SCHEMA:
        v.append(
            "unsupported_schema_version: %r 不在 %s" % (sv, SUPPORTED_DENYLIST_SCHEMA)
        )

    for name in REQUIRED_ARRAYS:
        if name not in payload:
            v.append("missing_array: %s" % name)
            continue
        arr = payload[name]
        if not isinstance(arr, list):
            v.append("%s: 必须是 JSON array，实际 %s" % (name, type(arr).__name__))
            continue
        for idx, item in enumerate(arr):
            if isinstance(item, bool) or not isinstance(item, str):
                v.append(
                    "%s[%d]: 必须是 str（bool/int/float/null/list/dict 一律拒绝），实际 %r"
                    % (name, idx, item)
                )
                continue
            if not item.strip():
                v.append("%s[%d]: 空串或纯空白非法（合法字面量 'none' 除外）" % (name, idx))

    if v:
        return False, v, [], {}

    per_array = {}
    seen = {}
    order = []
    raw_total = 0
    dup_total = 0
    for name in REQUIRED_ARRAYS:
        arr = payload[name]
        raw_total += len(arr)
        local_seen = set()
        dup = 0
        for item in arr:
            tok = normalize_token(item)
            if tok in local_seen:
                dup += 1
                continue
            local_seen.add(tok)
            if tok not in seen:
                seen[tok] = name
                order.append(item)
        dup_total += dup
        per_array[name] = {
            "raw_count": len(arr),
            "effective_count": len(local_seen),
            "duplicate_count": dup,
        }

    declared = payload.get("counts_declared")
    if isinstance(declared, dict):
        for name in REQUIRED_ARRAYS:
            if name in declared and declared[name] != per_array[name]["effective_count"]:
                v.append(
                    "count_mismatch: counts_declared.%s=%r 实测 effective_count=%d"
                    % (name, declared[name], per_array[name]["effective_count"])
                )
    if not order:
        v.append("denylist_empty: 三数组去重并集为空 ⇒ MissingInput, exit_code=4")
    if v:
        return False, v, [], per_array

    return True, [], order, {
        "per_array": per_array,
        "raw_total": raw_total,
        "duplicate_total": dup_total,
        "effective_total": len(order),
        "source_array_of_token": seen,
    }


# ------------------------------------------------------------------- records


def validate_records(tasks):
    """候选记录必需字段校验（与冻结验证器口径一致）。"""
    violations = []
    if not isinstance(tasks, list):
        return False, ["records: tasks 必须是 list"], 0, 0
    effective = 0
    for idx, rec in enumerate(tasks):
        if not isinstance(rec, dict):
            violations.append("records[%d]: 必须是 object" % idx)
            continue
        bad = False
        for field in ("problem_family_id", "repo_family"):
            if field not in rec:
                violations.append("records[%d].%s: 缺失" % (idx, field))
                bad = True
            elif rec[field] is None:
                violations.append("records[%d].%s: null 非法" % (idx, field))
                bad = True
            elif isinstance(rec[field], str) and not rec[field].strip():
                violations.append("records[%d].%s: 空串/纯空白非法" % (idx, field))
                bad = True
        if not bad:
            effective += 1
    return not violations, violations, len(tasks), effective


def compute_intersection(tasks, union_tokens):
    """返回 (intersection_tokens, per_record_hits)。"""
    tl = {normalize_token(t) for t in union_tokens}
    hits = []
    for idx, rec in enumerate(tasks):
        for field in ("problem_family_id", "repo_family"):
            if isinstance(rec, dict) and isinstance(rec.get(field), str):
                tok = normalize_token(rec[field])
                if tok in tl:
                    hits.append({"record_index": idx, "field": field, "token": tok})
    tokens = sorted({h["token"] for h in hits})
    return tokens, hits


# ------------------------------------------------------------- validator run


def run_validator(validator_src: Path, records, denylist):
    """真实子进程调用冻结验证器，返回 (exit_code, stderr_tail)。"""
    payload = json.dumps({"records": records, "denylist": denylist}, ensure_ascii=False)
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, str(validator_src)],
        input=payload.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc.returncode, proc.stderr.decode("utf-8", "replace").strip()[-500:]


# ------------------------------------------------------------------- evidence


def gate_ok(gate, reasons, name):
    if not isinstance(gate, dict):
        reasons.append("%s: 不是对象" % name)
        return False
    if gate.get("claimed") is not True:
        reasons.append("%s: claimed != true" % name)
        return False
    ev = gate.get("evidence")
    if not isinstance(ev, list) or not ev:
        reasons.append("%s: 无证据引用（不能只填布尔）" % name)
        return False
    usable = [
        e
        for e in ev
        if isinstance(e, dict)
        and e.get("kind") in ("artifact", "command_log", "custody_record")
        and e.get("sha256")
    ]
    if not usable:
        reasons.append("%s: 证据全为 self_declared 或缺 sha256" % name)
        return False
    return True


# ---------------------------------------------------------------------- main


def evaluate(manifest, protocol_bytes, args):
    reasons = []
    detail = {}

    # 1) 协议摘要
    pref = manifest.get("protocol_ref") or {}
    p_sha = sha256_bytes(protocol_bytes)
    if pref.get("protocol_sha256") != p_sha:
        reasons.append(
            "protocol_sha_mismatch: manifest=%r 实际=%s" % (pref.get("protocol_sha256"), p_sha)
        )
        return "BLOCKED", reasons, detail
    detail["protocol_sha256_actual"] = p_sha

    synth = bool((manifest.get("execution_mode") or {}).get("synthetic_mode"))
    isolation_ready = bool((manifest.get("execution_mode") or {}).get("access_authorization", {}).get("isolation_ready"))

    binding = manifest.get("binding") or {}
    decl_bind = {
        k: (binding.get(k) or {}).get("file_bytes_sha256")
        for k in ("candidate", "denylist", "validator", "snapshot")
    }

    # 2) 合成模式下禁止绑定真实 denylist
    if synth and decl_bind["denylist"]:
        reasons.append("synthetic_mode 下不得绑定真实 denylist 摘要")
    if not isolation_ready and not synth:
        # 非合成模式且隔离未就绪：仅允许记录『不绑定真实 denylist』的计算
        pass

    # 3) 实际字节重算
    measured = {}
    for key, path in (
        ("candidate", args.records),
        ("denylist", args.denylist),
    ):
        if path:
            raw = Path(path).read_bytes()
            measured[key] = {
                "file_bytes_sha256": sha256_bytes(raw),
                "byte_length": len(raw),
                "newline_mode": newline_mode(raw),
            }
    if args.validator_src:
        vp = Path(args.validator_src) / "v3" / "data" / "dedup.py"
        if vp.exists():
            raw = vp.read_bytes()
            measured["validator"] = {
                "file_bytes_sha256": sha256_bytes(raw),
                "byte_length": len(raw),
                "newline_mode": newline_mode(raw),
            }
    detail["measured_binding"] = measured

    binding_ok = {"candidate": True, "denylist": True, "validator": True, "snapshot": True}
    for key in ("candidate", "denylist", "validator"):
        m = measured.get(key)
        d = binding.get(key) or {}
        if d.get("file_bytes_sha256") is None:
            binding_ok[key] = False
            reasons.append("binding.%s.file_bytes_sha256 为空（G2 不成立）" % key)
            continue
        if m is None:
            reasons.append("binding.%s: 未提供实际文件，无法证明摘要相符" % key)
            binding_ok[key] = False
            continue
        if d.get("file_bytes_sha256") != m["file_bytes_sha256"]:
            binding_ok[key] = False
            reasons.append(
                "binding.%s 摘要不符: manifest=%s 实际=%s"
                % (key, d.get("file_bytes_sha256"), m["file_bytes_sha256"])
            )
        if d.get("newline_mode") != m["newline_mode"]:
            binding_ok[key] = False
            reasons.append(
                "binding.%s.newline_mode 不符: manifest=%r 实际=%r"
                % (key, d.get("newline_mode"), m["newline_mode"])
            )
        if d.get("byte_length") != m["byte_length"]:
            binding_ok[key] = False
            reasons.append(
                "binding.%s.byte_length 不符: manifest=%r 实际=%d"
                % (key, d.get("byte_length"), m["byte_length"])
            )
    if not (binding.get("snapshot") or {}).get("file_bytes_sha256") or \
       (binding.get("snapshot") or {}).get("content_status") != "obtained":
        binding_ok["snapshot"] = False
        reasons.append("binding.snapshot 未取得内容（G3 不成立）")

    # 4) 输入契约 + 计数 + 真实退出码
    counts_ok = False
    exit_ok = False
    exit_source_ok = False
    intersection_zero = False
    if args.records and args.denylist and args.validator_src:
        try:
            dn, dn_raw = load_json_bytes(Path(args.denylist))
        except Exception as exc:  # noqa: BLE001
            reasons.append("denylist 无法解析: %s" % exc)
            return "BLOCKED", reasons, detail
        dn_ok, dn_viol, union, dn_stats = validate_denylist(dn)
        detail["denylist_violations"] = dn_viol
        if not dn_ok:
            reasons.extend("denylist_contract: " + x for x in dn_viol)
            return "BLOCKED", reasons, detail

        try:
            reg, _ = load_json_bytes(Path(args.records))
        except Exception as exc:  # noqa: BLE001
            reasons.append("records 无法解析: %s" % exc)
            return "BLOCKED", reasons, detail
        tasks = reg.get("tasks") if isinstance(reg, dict) else reg
        rec_ok, rec_viol, raw_n, eff_n = validate_records(tasks)
        detail["records_violations"] = rec_viol
        if not rec_ok:
            reasons.extend("records_contract: " + x for x in rec_viol)
            return "BLOCKED", reasons, detail

        inter_tokens, hits = compute_intersection(tasks, union)
        detail["intersection_tokens_count"] = len(inter_tokens)
        detail["intersection_hit_count"] = len(hits)

        real_exit, real_err = run_validator(Path(args.validator_src), tasks, union)
        detail["validator_subprocess_exit_code"] = real_exit
        detail["validator_subprocess_stderr_tail"] = real_err

        measured_counts = {
            "records_raw_count": raw_n,
            "records_effective_count": eff_n,
            "records_rejected_count": raw_n - eff_n,
            "denylist_raw_count": dn_stats["raw_total"],
            "denylist_effective_count": dn_stats["effective_total"],
            "denylist_duplicate_count": dn_stats["duplicate_total"],
            "denylist_rejected_count": 0,
            "intersection_count": len(inter_tokens),
            "denylist_per_array": dn_stats["per_array"],
        }
        detail["measured_counts"] = measured_counts

        c = manifest.get("counts") or {}
        declared_counts = {
            "records_raw_count": (c.get("candidate") or {}).get("records_raw_count"),
            "records_effective_count": (c.get("candidate") or {}).get("records_effective_count"),
            "records_rejected_count": (c.get("candidate") or {}).get("records_rejected_count"),
            "denylist_raw_count": (c.get("denylist") or {}).get("denylist_raw_count"),
            "denylist_effective_count": (c.get("denylist") or {}).get("denylist_effective_count"),
            "denylist_rejected_count": (c.get("denylist") or {}).get("denylist_rejected_count"),
            "denylist_duplicate_count": (c.get("denylist") or {}).get("denylist_duplicate_count"),
            "intersection_count": (c.get("result") or {}).get("intersection_count"),
        }
        agreed = []
        for k, mv in measured_counts.items():
            if k.endswith("_per_array"):
                continue
            if declared_counts.get(k) != mv:
                agreed.append(k)
        if agreed:
            reasons.append("counts 与实测不符: %s" % agreed)
            counts_ok = False
        else:
            counts_ok = True

        # 逐数组计数
        pa_decl = ((c.get("denylist") or {}).get("denylist_per_array") or {})
        for k, mv in dn_stats["per_array"].items():
            if (pa_decl.get(k) or {}) != mv:
                reasons.append("counts.denylist_per_array.%s 与实测不符" % k)
                counts_ok = False

        payload_exit = (manifest.get("records", {}).get("computation_result") or {})
        exit_source_ok = payload_exit.get("exit_code_source") == "os_process"
        if payload_exit.get("exit_code") != real_exit:
            reasons.append(
                "exit_code 与实测不符: manifest=%r 子进程实测=%r"
                % (payload_exit.get("exit_code"), real_exit)
            )
        exit_ok = real_exit == 0 and payload_exit.get("exit_code") == 0
        intersection_zero = len(inter_tokens) == 0
        if not intersection_zero:
            reasons.append("intersection_count=%d != 0" % len(inter_tokens))
    else:
        reasons.append("缺少 --records/--denylist/--validator-src：无法独立重算，G2 不成立")

    # 5) 四门槛
    iso = (manifest.get("records") or {}).get("isolation_acceptance") or {}
    gates = {}
    for name in (
        "G1_namespace_compatibility",
        "G2_effective_input_binding",
        "G3_snapshot_content_bound",
        "G4_isolation_verified",
    ):
        gates[name] = gate_ok(iso.get(name), reasons, name)

    # G2 额外要求实测绑定
    if gates["G2_effective_input_binding"] and not (
        binding_ok["candidate"] and binding_ok["denylist"] and binding_ok["validator"] and counts_ok
    ):
        gates["G2_effective_input_binding"] = False
        reasons.append("G2: 绑定/计数实测不成立，claimed=true 无效")
    if gates["G3_snapshot_content_bound"] and not binding_ok["snapshot"]:
        gates["G3_snapshot_content_bound"] = False
        reasons.append("G3: snapshot 未取得内容，claimed=true 无效")

    # G4 附加要求
    g4_reasons = []
    if gates["G4_isolation_verified"]:
        if iso.get("isolation_mechanism") not in PERMITTED_MECHANISMS:
            g4_reasons.append("isolation_mechanism 缺失或不被接受：%r" % iso.get("isolation_mechanism"))
        if not iso.get("permission_matrix_sha256"):
            g4_reasons.append("permission_matrix_sha256 缺失")
        if not iso.get("custody_principal_os_identity"):
            g4_reasons.append("custody_principal_os_identity 缺失（不接受 agent UUID）")
        nt = iso.get("negative_test") or {}
        if nt.get("exit_code_source") != "os_process":
            g4_reasons.append("negative_test.exit_code_source != os_process")
        if nt.get("subprocess_checked") is not True:
            g4_reasons.append("negative_test.subprocess_checked != true（未覆盖子进程）")
        if nt.get("stat_result") is None or nt.get("open_result") is None:
            g4_reasons.append("negative_test 未把 stat 与 open 分开记录")
        if nt.get("open_result") not in ("denied",):
            g4_reasons.append("negative_test.open_result 必须为 denied")
        pt = iso.get("positive_test") or {}
        if pt.get("exit_code") != 0:
            g4_reasons.append("positive_test: 评测主体对获准 canary 读取未成功")
        if g4_reasons:
            gates["G4_isolation_verified"] = False
            reasons.extend("G4: " + x for x in g4_reasons)

    all_gates = all(gates.values())

    # 6) 结论
    if synth:
        core = (
            binding_ok["candidate"]
            and binding_ok["denylist"]
            and binding_ok["validator"]
            and counts_ok
            and exit_ok
            and exit_source_ok
            and intersection_zero
            and gates["G1_namespace_compatibility"]
            and gates["G2_effective_input_binding"]
            and gates["G3_snapshot_content_bound"]
        )
        verdict = "SYNTHETIC_ACCEPTED" if core else "COMPUTATION_ONLY"
        if verdict == "SYNTHETIC_ACCEPTED":
            reasons = []
    elif all_gates and counts_ok and exit_ok and exit_source_ok and intersection_zero \
            and all(binding_ok.values()):
        verdict = "ACCEPTED"
        reasons = []
    elif exit_ok and counts_ok:
        verdict = "COMPUTATION_ONLY"
    else:
        verdict = "COMPUTATION_ONLY" if (args.records and args.denylist and args.validator_src) else "BLOCKED"

    detail["gates"] = gates
    detail["binding_ok"] = binding_ok
    detail["counts_ok"] = counts_ok
    detail["exit_ok"] = exit_ok
    detail["exit_source_ok"] = exit_source_ok
    detail["intersection_zero"] = intersection_zero
    detail["synthetic_mode"] = synth
    if verdict == "SYNTHETIC_ACCEPTED":
        detail["verdict_note"] = "SYNTHETIC_ACCEPTED — 非验收结论，仅在合成数据上验证判据逻辑"
    return verdict, reasons, detail


def build_parser():
    ap = argparse.ArgumentParser(description="KAGGLE-27 exclusion-proof manifest machine checker (v3)")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--records")
    ap.add_argument("--denylist")
    ap.add_argument("--validator-src")
    ap.add_argument("--out")
    ap.add_argument("--expect", default=None)
    ap.add_argument("--write-overall", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    protocol_bytes = Path(args.protocol).read_bytes()
    manifest, manifest_raw = load_json_bytes(Path(args.manifest))

    verdict, reasons, detail = evaluate(manifest, protocol_bytes, args)
    declared = (manifest.get("records") or {}).get("overall")
    expect = args.expect if args.expect is not None else declared

    report = {
        "checker": "verify_manifest.py",
        "protocol_sha256": sha256_bytes(protocol_bytes),
        "manifest_path": str(args.manifest),
        "manifest_declared_overall": declared,
        "computed_verdict": verdict,
        "expected_verdict": expect,
        "consistent": verdict == expect,
        "reasons": reasons,
        "detail": detail,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_bytes((json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    if args.write_overall:
        manifest.setdefault("records", {})["overall"] = verdict
        Path(args.manifest).write_bytes(
            (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        )

    print("computed=%s expected=%s consistent=%s reasons=%d"
          % (verdict, expect, verdict == expect, len(reasons)))
    for r in reasons[:20]:
        print("  - " + r)

    if detail.get("validator_subprocess_exit_code") is None and verdict == "BLOCKED":
        return EXIT_CANNOT
    return EXIT_OK if verdict == expect else EXIT_MISMATCH


if __name__ == "__main__":
    raise SystemExit(main())
