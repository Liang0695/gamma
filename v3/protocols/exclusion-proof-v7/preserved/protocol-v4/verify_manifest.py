#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KAGGLE-27 · 排除证明 execution manifest 机器验收器（v4）。

本版针对 Q0 在真实冻结 E0 上复现的 16 项缺陷（见协议 defect_register）重写：

  D1  证据必须可解析（格式/存在/实算相符/produced_by/produced_at）+ 覆盖门槛含义
  D2  snapshot 必须实测重算（--snapshot），不看 status
  D3  授权判定**前置于一切受保护输入读取**；未通过 ⇒ 零读取（read_log 可证）
  D4  非空候选门槛
  D5  counts_declared 三数组全部必需
  D6  计数守恒 raw == eff + dup（含跨数组重复）+ 多值来源映射
  D7  同上
  D8  经**真实 CLI 入口**执行，并检验协议 schema 是否真被消费
  D9  绑定**导入闭包**而非单个文件
  D10 协议冻结的独立期望分母交叉核对
  D11 模式开关严格布尔（禁 bool() 强转）
  D12 OS 身份格式校验（拒绝 agent UUID）
  D13 每个文件单次读取，哈希与解析同一份字节
  D14 全部 violation 带稳定 reason_code
  D15 生产模式 exit 0 只代表真实 ACCEPTED；期望不得取自被检清单自述
  D16 G4 机制不足清单

用法：
  生产：python verify_manifest.py --protocol P --manifest M --mode production \\
            --records R --denylist D --validator-src E0 --evidence-root EV \\
            --authorization AUTH --authorization-trust-anchor <sha256> --out report.json
  测试：... --mode test --expect SYNTHETIC_ACCEPTED --validator-mode test_double
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_CANNOT = 2

SUPPORTED_DENYLIST_SCHEMA = ["v2-dh-exclusion-denylist-1"]
REQUIRED_ARRAYS = ["d_family_ids", "h_family_ids", "exposed_family_ids"]
ALLOWED_TOP_KEYS = set(
    REQUIRED_ARRAYS + ["denylist_schema_version", "counts_declared",
                       "source_sha256", "produced_by", "produced_at"]
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
SID_RE = re.compile(r"^S-1-[0-9-]+$")
ACCOUNT_RE = re.compile(r"^[A-Za-z0-9._\\-]+\\[A-Za-z0-9._$-]+$")
POSIX_RE = re.compile(r"^(uid=)?[0-9]+(\.[0-9]+)?(\([^)]*\))?$")

PERMITTED_MECHANISMS = {
    "independent_host",
    "controlled_vm",
    "separate_os_account_with_broken_acl_inheritance",
}
REQUIRED_CLOSURE = ["v3/data/dedup.py", "v3/common/errors.py", "v3/common/canonical.py"]
EVIDENCE_KINDS = {"artifact", "command_log", "custody_record", "trusted_approval"}

CLI_CHILD = None  # 真实 CLI 用 `python -m v3.cli split`；测试替身见下
TEST_DOUBLE = r"""
import json, sys
payload = json.load(sys.stdin)
deny = {str(x).strip().lower() for x in payload.get("denylist", []) if str(x).strip()}
for rec in payload.get("records", []):
    for f in ("problem_family_id", "repo_family"):
        v = rec.get(f)
        if v is None or str(v).strip() == "":
            sys.stderr.write("[split_leak_check_missing_family]\n")
            raise SystemExit(4)
for rec in payload.get("records", []):
    for f in ("problem_family_id", "repo_family"):
        if str(rec.get(f)).strip().lower() in deny:
            sys.stderr.write("[v2_denylist_intersection]\n")
            raise SystemExit(7)
raise SystemExit(0)
"""


# ------------------------------------------------------------------ plumbing


class Violations:
    def __init__(self):
        self.items = []

    def add(self, code, message):
        self.items.append({"reason_code": code, "message": message})

    def codes(self):
        return [i["reason_code"] for i in self.items]

    def has(self, code):
        return code in self.codes()

    def __len__(self):
        return len(self.items)


class ReadLog:
    """受保护输入的读取记录。每个路径只允许读一次（D13）。"""

    def __init__(self):
        self.entries = []
        self._cache = {}
        self._counts = {}

    def read(self, path, phase, purpose):
        p = str(Path(path).resolve())
        self._counts[p] = self._counts.get(p, 0) + 1
        self.entries.append({
            "phase": phase, "purpose": purpose, "path": p,
            "read_ordinal": self._counts[p],
        })
        if p not in self._cache:
            self._cache[p] = Path(p).read_bytes()
        return self._cache[p], self._counts[p]

    def read_count(self, path):
        return self._counts.get(str(Path(path).resolve()), 0)

    def protected_entries(self):
        return [e for e in self.entries if e["phase"] >= 2]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def canonical_json_sha256(obj):
    return sha256_bytes(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                   separators=(",", ":")).encode("utf-8"))


def newline_mode(b):
    crlf = b.count(b"\r\n")
    lf = b.count(b"\n") - crlf
    return "mixed" if (crlf and lf) else ("crlf" if crlf else "lf")


def norm_token(v):
    return str(v).strip().lower()


def valid_os_identity(s):
    if not isinstance(s, str) or not s.strip():
        return False
    if UUID_RE.match(s.strip()):
        return False
    return bool(SID_RE.match(s.strip()) or ACCOUNT_RE.match(s.strip()) or POSIX_RE.match(s.strip()))


# ------------------------------------------------------------ denylist (D5-D7)


def validate_denylist(payload, v):
    """整批校验三数组契约 + 计数守恒 + 多值来源映射。返回 (ok, union, stats)。"""
    if not isinstance(payload, dict):
        v.add("DENYLIST_UNKNOWN_KEY", "顶层必须是 object，实际 %s" % type(payload).__name__)
        return False, [], {}

    extra = sorted(set(payload.keys()) - ALLOWED_TOP_KEYS)
    if extra:
        v.add("DENYLIST_UNKNOWN_KEY", "未知顶层键: %s" % extra)
    if payload.get("denylist_schema_version") not in SUPPORTED_DENYLIST_SCHEMA:
        v.add("DENYLIST_SCHEMA_UNSUPPORTED",
              "denylist_schema_version=%r 不在 %s" % (payload.get("denylist_schema_version"),
                                                     SUPPORTED_DENYLIST_SCHEMA))

    for name in REQUIRED_ARRAYS:
        if name not in payload:
            v.add("DENYLIST_ITEM_INVALID", "缺少必需数组 %s" % name)
            continue
        arr = payload[name]
        if not isinstance(arr, list):
            v.add("DENYLIST_ITEM_INVALID", "%s 必须是 array" % name)
            continue
        for i, it in enumerate(arr):
            if isinstance(it, bool) or not isinstance(it, str):
                v.add("DENYLIST_ITEM_INVALID",
                      "%s[%d]=%r 类型非法（bool/int/float/null/list/dict 一律拒绝）" % (name, i, it))
            elif not it.strip():
                v.add("DENYLIST_ITEM_INVALID", "%s[%d] 空串/纯空白非法" % (name, i))

    declared = payload.get("counts_declared")
    if declared is None:
        v.add("DENYLIST_COUNTS_DECLARED_MISSING", "counts_declared 整块缺失")
    elif not isinstance(declared, dict):
        v.add("DENYLIST_COUNTS_DECLARED_MISSING", "counts_declared 必须是 object")
    else:
        missing = [n for n in REQUIRED_ARRAYS if n not in declared]
        if missing:
            v.add("DENYLIST_COUNTS_PARTIAL", "counts_declared 未声明: %s" % missing)

    if v.has("DENYLIST_ITEM_INVALID") or v.has("DENYLIST_SCHEMA_UNSUPPORTED") or v.has("DENYLIST_UNKNOWN_KEY"):
        return False, [], {}

    arrays_of_token = {}
    per = {}
    raw_total = 0
    within_dupes = 0
    order = []
    for name in REQUIRED_ARRAYS:
        arr = payload[name]
        raw_total += len(arr)
        seen = set()
        within = 0
        for it in arr:
            t = norm_token(it)
            if t in seen:
                within += 1
            else:
                seen.add(t)
                if t not in arrays_of_token:
                    arrays_of_token[t] = []
                    order.append(it)
                arrays_of_token[t].append(name)
        within_dupes += within
        per[name] = {"raw_count": len(arr), "effective_count": len(seen), "duplicate_count": within}

    effective_total = len(arrays_of_token)
    cross = sum(len(a) - 1 for a in arrays_of_token.values() if len(a) > 1)
    duplicate_total = within_dupes + cross

    if raw_total != effective_total + duplicate_total:
        v.add("COUNT_INVARIANT_BROKEN",
              "raw=%d != eff=%d + dup=%d" % (raw_total, effective_total, duplicate_total))

    if declared and isinstance(declared, dict):
        for n in REQUIRED_ARRAYS:
            if n in declared and declared[n] != per[n]["effective_count"]:
                v.add("DENYLIST_COUNT_MISMATCH",
                      "counts_declared.%s=%r 实测=%d" % (n, declared[n], per[n]["effective_count"]))

    if effective_total == 0:
        v.add("DENYLIST_EMPTY", "三数组去重并集为空")

    stats = {
        "per_array": per,
        "raw_total": raw_total,
        "effective_total": effective_total,
        "duplicate_total": duplicate_total,
        "within_array_duplicate_total": within_dupes,
        "cross_array_duplicate_total": cross,
        "source_array_of_token": {k: sorted(set(x)) for k, x in arrays_of_token.items()},
        "source_map_is_multi_valued": all(isinstance(x, list) for x in arrays_of_token.values()),
    }
    ok = not (v.has("COUNT_INVARIANT_BROKEN") or v.has("DENYLIST_COUNT_MISMATCH")
              or v.has("DENYLIST_EMPTY") or v.has("DENYLIST_COUNTS_DECLARED_MISSING")
              or v.has("DENYLIST_COUNTS_PARTIAL"))
    return ok, order, stats


def validate_records(tasks, v):
    if not isinstance(tasks, list):
        v.add("RECORD_MISSING_FIELD", "records 必须是 list")
        return False, 0, 0
    if len(tasks) == 0:
        v.add("CANDIDATE_EMPTY", "候选记录 0 条；空候选的交集恒为 0，等于没验")
    effective = 0
    for i, rec in enumerate(tasks):
        bad = False
        if not isinstance(rec, dict):
            v.add("RECORD_MISSING_FIELD", "records[%d] 不是 object" % i)
            continue
        for f in ("problem_family_id", "repo_family"):
            if f not in rec:
                v.add("RECORD_MISSING_FIELD", "records[%d].%s 缺失" % (i, f)); bad = True
            elif rec[f] is None:
                v.add("RECORD_MISSING_FIELD", "records[%d].%s 为 null" % (i, f)); bad = True
            elif isinstance(rec[f], str) and not rec[f].strip():
                v.add("RECORD_MISSING_FIELD", "records[%d].%s 空串/纯空白" % (i, f)); bad = True
        if not bad:
            effective += 1
    return True, len(tasks), effective


# --------------------------------------------------------------- evidence (D1)


def resolve_evidence(evidence, evidence_root, read_log, v, gate_name):
    """解析证据：格式 / 存在 / 实算相符 / 生产者 / 覆盖面。"""
    if not isinstance(evidence, list) or not evidence:
        v.add("EVIDENCE_SELF_DECLARED", "%s: 无证据引用" % gate_name)
        return False
    usable = 0
    for e in evidence:
        if not isinstance(e, dict):
            v.add("EVIDENCE_NOT_RESOLVABLE", "%s: 证据项不是 object" % gate_name); continue
        kind = e.get("kind")
        if kind == "self_declared" or kind not in EVIDENCE_KINDS:
            v.add("EVIDENCE_SELF_DECLARED", "%s: kind=%r 永不足够" % (gate_name, kind)); continue
        digest = e.get("sha256")
        if not isinstance(digest, str) or not HEX64.match(digest):
            v.add("EVIDENCE_HASH_INVALID", "%s: sha256=%r 不是 64 位小写十六进制" % (gate_name, digest)); continue
        if not e.get("produced_by") or not e.get("produced_at"):
            v.add("EVIDENCE_PRODUCER_MISSING", "%s: 缺 produced_by/produced_at" % gate_name); continue
        rel = e.get("path")
        if not isinstance(rel, str) or not rel:
            v.add("EVIDENCE_NOT_RESOLVABLE", "%s: path 缺失" % gate_name); continue
        if evidence_root is None:
            v.add("EVIDENCE_NOT_RESOLVABLE", "%s: 未提供 --evidence-root" % gate_name); continue
        root = Path(evidence_root).resolve()
        target = (root / rel).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            v.add("EVIDENCE_NOT_RESOLVABLE", "%s: path 逃出 evidence root: %s" % (gate_name, rel)); continue
        if not target.exists():
            v.add("EVIDENCE_NOT_RESOLVABLE", "%s: 路径不存在: %s" % (gate_name, rel)); continue
        raw, _ = read_log.read(target, phase=2, purpose="evidence")
        if sha256_bytes(raw) != digest:
            v.add("EVIDENCE_HASH_MISMATCH",
                  "%s: 证据 %s 实算 %s != 声明 %s" % (gate_name, rel, sha256_bytes(raw)[:16], digest[:16]))
            continue
        usable += 1
    if usable == 0:
        v.add("GATE_MEANING_UNSUPPORTED", "%s: 无可解析证据，门槛含义未被覆盖" % gate_name)
        return False
    return True


def gate_claimed(gate, gate_name, v):
    if not isinstance(gate, dict):
        v.add("EVIDENCE_SELF_DECLARED", "%s: 不是对象" % gate_name); return False
    if gate.get("claimed") is not True:
        v.add("EVIDENCE_SELF_DECLARED", "%s: claimed != true" % gate_name); return False
    return True


# ----------------------------------------------------------------- CLI (D8)


def run_real_cli(validator_src, tasks, denylist_union, tmpdir):
    """真实入口 python -m v3.cli split，两轮：
       A) 协议三数组 schema 的 denylist（含与候选相交的家族）→ 必须非 0，否则未消费；
       B) 生产 schema families/repos → 用于对照。
       返回 dict。
    """
    reg = Path(tmpdir) / "reg.json"
    reg.write_bytes((json.dumps({"tasks": tasks, "quota": {"*": 1}}, ensure_ascii=False) + "\n").encode())

    proto_deny = Path(tmpdir) / "deny_protocol_schema.json"
    proto_deny.write_bytes((json.dumps({
        "denylist_schema_version": "v2-dh-exclusion-denylist-1",
        "d_family_ids": list(denylist_union),
        "h_family_ids": [], "exposed_family_ids": [],
        "counts_declared": {"d_family_ids": len(set(denylist_union)),
                            "h_family_ids": 0, "exposed_family_ids": 0},
    }, ensure_ascii=False) + "\n").encode())

    prod_deny = Path(tmpdir) / "deny_production_schema.json"
    prod_deny.write_bytes((json.dumps({
        "families": list(denylist_union), "repos": [],
    }, ensure_ascii=False) + "\n").encode())

    out = {}
    for tag, deny in (("protocol_schema", proto_deny), ("production_schema", prod_deny)):
        p = subprocess.run(
            [sys.executable, "-m", "v3.cli", "split", "--registry", str(reg),
             "--denylist", str(deny), "--out", str(Path(tmpdir) / ("o_%s.json" % tag))],
            cwd=str(validator_src), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        out[tag] = {
            "os_process_exit_code": p.returncode,
            "stderr_tail": p.stderr.decode("utf-8", "replace").strip()[-300:],
        }
    return out


def run_test_double(tasks, denylist_union):
    payload = json.dumps({"records": tasks, "denylist": list(denylist_union)}, ensure_ascii=False)
    p = subprocess.run([sys.executable, "-c", TEST_DOUBLE], input=payload.encode(),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return {"test_double": {"os_process_exit_code": p.returncode,
                            "stderr_tail": p.stderr.decode("utf-8", "replace").strip()[-200:]},
            "note": "显式测试替身：只验证判据逻辑，不冒充 E0 真实子进程"}


# ---------------------------------------------------------------- evaluation


def evaluate(args, read_log, v):
    detail = {}
    checks = {}

    # ---------- 阶段 0：协议 / 授权记录（预先获准的非敏感配置）
    protocol_bytes, _ = read_log.read(args.protocol, phase=0, purpose="protocol")
    protocol = json.loads(protocol_bytes.decode("utf-8"))
    manifest_bytes, _ = read_log.read(args.manifest, phase=0, purpose="manifest")
    manifest = json.loads(manifest_bytes.decode("utf-8"))

    pref = manifest.get("protocol_ref") or {}
    if pref.get("protocol_sha256") != sha256_bytes(protocol_bytes):
        v.add("PROTOCOL_SHA_MISMATCH",
              "manifest=%r 实际=%s" % (pref.get("protocol_sha256"), sha256_bytes(protocol_bytes)))
        return "BLOCKED", detail, checks

    auth = None
    auth_bytes = None
    if args.authorization:
        auth_bytes, _ = read_log.read(args.authorization, phase=0, purpose="authorization")
        auth = json.loads(auth_bytes.decode("utf-8"))

    # ---------- 模式严格布尔（D11）
    mode = manifest.get("execution_mode") or {}
    synth_raw = mode.get("synthetic_mode")
    if not isinstance(synth_raw, bool):
        v.add("MODE_TYPE_INVALID", "synthetic_mode=%r 不是 JSON 布尔（禁 bool() 强转）" % (synth_raw,))
    synthetic = synth_raw is True
    detail["synthetic_mode"] = synthetic

    # ---------- 授权判定（D3），必须先于任何受保护输入读取
    authorized = False
    if synthetic:
        authorized = True
        checks["authorization"] = "synthetic_mode ⇒ 测试替身，结论一律非验收结论"
    else:
        if auth is None:
            v.add("AUTH_MISSING", "非合成模式缺少 --authorization 可信授权记录")
        else:
            anchor = args.authorization_trust_anchor
            if not anchor:
                v.add("AUTH_UNTRUSTED", "未提供 --authorization-trust-anchor，授权记录不可信")
            elif not HEX64.match(anchor):
                v.add("AUTH_UNTRUSTED", "信任锚不是 64 位小写十六进制")
            elif sha256_bytes(auth_bytes) != anchor:
                v.add("AUTH_UNTRUSTED", "授权记录实算 %s != 信任锚 %s"
                      % (sha256_bytes(auth_bytes)[:16], anchor[:16]))
            else:
                scope = auth.get("scope") or {}
                if not isinstance(scope.get("read_permitted"), bool):
                    v.add("MODE_TYPE_INVALID", "authorization.scope.read_permitted 不是布尔")
                need = {"candidate", "denylist", "validator"}
                have = set(scope.get("inputs") or [])
                if scope.get("read_permitted") is True and not (need - have):
                    authorized = True
                else:
                    v.add("AUTH_SCOPE_INSUFFICIENT", "授权范围 %s 未覆盖 %s" % (sorted(have), sorted(need)))
        checks["authorization"] = "authorized" if authorized else "denied"

    detail["read_log_before_gate"] = list(read_log.entries)
    if not authorized:
        v.add("AUTH_BEFORE_ACCESS", "未通过授权判定：受保护输入零读取")
        declared_now = (manifest.get("records") or {}).get("overall")
        if declared_now == "ACCEPTED":
            v.add("OVERALL_SELF_DECLARED",
                  "被检清单自述 overall=ACCEPTED，但机器判定 BLOCKED；自述不参与验收")
        return "BLOCKED", detail, checks

    # ---------- 阶段 2：受保护输入
    binding = manifest.get("binding") or {}
    measured = {}

    def measure(key, path, purpose):
        raw, cnt = read_log.read(path, phase=2, purpose=purpose)
        if cnt > 1:
            v.add("BYTES_REBOUND", "%s 被读 %d 次（哈希与解析必须用同一份字节）" % (key, cnt))
        measured[key] = {"file_bytes_sha256": sha256_bytes(raw), "byte_length": len(raw),
                         "newline_mode": newline_mode(raw), "_raw": raw}
        return raw

    rec_raw = measure("candidate", args.records, "candidate") if args.records else None
    den_raw = measure("denylist", args.denylist, "denylist") if args.denylist else None
    snap_raw = measure("snapshot", args.snapshot, "snapshot") if args.snapshot else None

    if rec_raw is None or den_raw is None:
        v.add("BINDING_INCOMPLETE", "缺少 --records / --denylist")
        return "BLOCKED", detail, checks

    # 绑定核对
    for key, spec in (("candidate", binding.get("candidate") or {}),
                      ("denylist", binding.get("denylist") or {})):
        m = measured[key]
        if spec.get("file_bytes_sha256") is None:
            v.add("BINDING_INCOMPLETE", "binding.%s.file_bytes_sha256 为空" % key)
        elif spec.get("file_bytes_sha256") != m["file_bytes_sha256"]:
            v.add("BINDING_MISMATCH", "binding.%s 摘要不符" % key)
        if spec.get("byte_length") != m["byte_length"]:
            v.add("BINDING_MISMATCH", "binding.%s.byte_length 不符" % key)
        if spec.get("newline_mode") != m["newline_mode"]:
            v.add("BINDING_MISMATCH", "binding.%s.newline_mode 不符" % key)

    # ---------- 导入闭包绑定（D9）
    if args.validator_src:
        vsrc = Path(args.validator_src)
        declared_set = ((binding.get("validator") or {}).get("file_set") or [])
        declared_paths = {e.get("path") for e in declared_set if isinstance(e, dict)}
        missing = [p for p in REQUIRED_CLOSURE if p not in declared_paths]
        if missing:
            v.add("BINDING_INCOMPLETE", "导入闭包未完整绑定，缺: %s" % missing)
        for e in declared_set:
            if not isinstance(e, dict) or not e.get("path"):
                v.add("BINDING_INCOMPLETE", "file_set 项缺 path"); continue
            f = vsrc / e["path"]
            if not f.exists():
                v.add("BINDING_INCOMPLETE", "file_set 路径不存在: %s" % e["path"]); continue
            raw, cnt = read_log.read(f, phase=2, purpose="validator_closure")
            if cnt > 1:
                v.add("BYTES_REBOUND", "validator 闭包文件 %s 被读 %d 次" % (e["path"], cnt))
            if sha256_bytes(raw) != e.get("file_bytes_sha256"):
                v.add("BINDING_MISMATCH",
                      "闭包文件 %s 实算 %s != 声明 %s"
                      % (e["path"], sha256_bytes(raw)[:16], str(e.get("file_bytes_sha256"))[:16]))
            if not e.get("git_blob_sha1"):
                v.add("BINDING_INCOMPLETE", "闭包文件 %s 缺 git_blob_sha1" % e["path"])
        checks["closure_files_checked"] = len(declared_set)
    else:
        v.add("BINDING_INCOMPLETE", "缺少 --validator-src，无法核对导入闭包")

    # ---------- denylist 契约 / 计数 / 不变式
    den = json.loads(den_raw.decode("utf-8"))
    dn_ok, union, dn_stats = validate_denylist(den, v)
    detail["denylist_stats"] = {k: val for k, val in dn_stats.items() if k != "source_array_of_token"}
    detail["source_array_map_sample"] = dict(list(dn_stats.get("source_array_of_token", {}).items())[:5])

    tasks = json.loads(rec_raw.decode("utf-8")).get("tasks")
    rec_ok, raw_n, eff_n = validate_records(tasks, v)
    detail["records"] = {"raw": raw_n, "effective": eff_n}

    # ---------- 分母基线（D10）
    baseline = (protocol.get("denominator_baseline") or {}).get("expected_public_counts") or {}
    dcheck = manifest.get("denominator_check") or {}
    compared = dcheck.get("compared_denominators")
    if not isinstance(compared, list) or not compared:
        v.add("DENOMINATOR_MISSING", "denominator_check.compared_denominators 未声明")
    else:
        unknown = [c for c in compared if c not in baseline]
        if unknown:
            v.add("DENOMINATOR_MISSING", "对照了协议中不存在的分母口径: %s" % unknown)
        for name in compared:
            exp = (baseline.get(name) or {}).get("value")
            got = (dcheck.get("measured") or {}).get(name)
            if exp is None:
                v.add("DENOMINATOR_MISSING", "%s 在协议中无期望值" % name)
            elif got is None:
                v.add("DENOMINATOR_MISSING", "%s 未给出实测值" % name)
            elif isinstance(got, int) and got < exp:
                v.add("DENOMINATOR_SHORTFALL", "%s 实测 %d < 期望 %d" % (name, got, exp))
        checks["denominators_compared"] = compared

    # ---------- 真实 CLI / 测试替身（D8）
    validator_mode = args.validator_mode or "real_cli"
    detail["validator_mode"] = validator_mode
    exit_code = None
    if validator_mode == "real_cli":
        if not union:
            v.add("DENYLIST_EMPTY", "无并集可喂入真实入口")
        else:
            base = Path(args.workdir) if args.workdir else (
                (Path(args.out).parent if args.out else Path.cwd()) / "_k27v4_tmp")
            # 注意：不使用 tempfile.mkdtemp —— 沙箱下 mkdtemp 生成的随机目录不可写。
            tmp = base / ("run_%d" % os.getpid())
            if tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True, exist_ok=True)
            try:
                res = run_real_cli(Path(args.validator_src), tasks, union, tmp)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            detail["real_cli"] = {k: val for k, val in res.items()}
            proto_code = res["protocol_schema"]["os_process_exit_code"]
            prod_code = res["production_schema"]["os_process_exit_code"]
            if prod_code != 7:
                v.add("DENYLIST_SCHEMA_NOT_CONSUMED",
                      "对照轮（生产 schema，含真实相交家族）退出码 %d != 7，入口未按预期判定" % prod_code)
            if proto_code == 0:
                v.add("DENYLIST_SCHEMA_NOT_CONSUMED",
                      "协议 schema 的排除清单被真实入口忽略（exit 0，fail-open），零交集是伪结论")
            exit_code = prod_code
    else:
        res = run_test_double(tasks, union)
        detail["test_double"] = res
        exit_code = res["test_double"]["os_process_exit_code"]

    detail["validator_exit_code"] = exit_code

    # ---------- 计数与实测对照
    decl = manifest.get("counts") or {}
    d_dn = decl.get("denylist") or {}
    d_cd = decl.get("candidate") or {}
    inv = decl.get("invariant") or {}
    mcounts = {
        "records_raw_count": raw_n,
        "records_effective_count": eff_n,
        "records_rejected_count": raw_n - eff_n,
        "denylist_raw_count": dn_stats.get("raw_total"),
        "denylist_effective_count": dn_stats.get("effective_total"),
        "denylist_duplicate_count": dn_stats.get("duplicate_total"),
        "denylist_cross_array_duplicate_count": dn_stats.get("cross_array_duplicate_total"),
    }
    for k, mv in mcounts.items():
        target = d_cd if k.startswith("records_") else d_dn
        if target.get(k) != mv:
            v.add("COUNT_INVARIANT_BROKEN", "counts.%s 声明 %r != 实测 %r" % (k, target.get(k), mv))
    if (d_dn.get("denylist_per_array") or {}) != dn_stats.get("per_array"):
        v.add("COUNT_INVARIANT_BROKEN", "counts.denylist_per_array 与实测不符")
    if d_dn.get("source_array_map_is_multi_valued") is not True:
        v.add("COUNT_INVARIANT_BROKEN", "source_array_map_is_multi_valued 未声明为 true（来源映射必须多值）")
    if inv.get("raw_equals_effective_plus_duplicate") is not True:
        v.add("COUNT_INVARIANT_BROKEN", "invariant.raw_equals_effective_plus_duplicate 未为 true")

    # 交集
    tl = {norm_token(t) for t in union}
    inter = {norm_token(r.get(f)) for r in tasks if isinstance(r, dict)
             for f in ("problem_family_id", "repo_family")
             if isinstance(r.get(f), str) and norm_token(r.get(f)) in tl}
    detail["intersection_count"] = len(inter)
    if inter:
        v.add("INTERSECTION_NONZERO", "交集 %d 项" % len(inter))

    comp = (manifest.get("records") or {}).get("computation_result") or {}
    if comp.get("exit_code") != exit_code:
        v.add("VALIDATOR_EXIT_MISMATCH", "manifest exit_code=%r 实测=%r" % (comp.get("exit_code"), exit_code))
    if comp.get("exit_code_source") != "os_process":
        v.add("EXIT_CODE_NOT_OS_PROCESS", "exit_code_source=%r" % comp.get("exit_code_source"))
    if validator_mode == "real_cli" and comp.get("denylist_schema_consumed") is not True:
        v.add("DENYLIST_SCHEMA_NOT_CONSUMED", "manifest 未声明真实入口消费了协议 schema 清单")

    # ---------- snapshot（D2）
    snap = binding.get("snapshot") or {}
    if snap.get("content_status") != "obtained" or snap_raw is None:
        v.add("SNAPSHOT_UNVERIFIED", "snapshot 未取得内容或未提供 --snapshot")
    elif sha256_bytes(snap_raw) != snap.get("file_bytes_sha256"):
        v.add("SNAPSHOT_UNVERIFIED", "snapshot 实算摘要 != 声明")

    # ---------- 四门槛
    iso = (manifest.get("records") or {}).get("isolation_acceptance") or {}
    gates = {}
    for gname in ("G1_namespace_compatibility", "G2_effective_input_binding",
                  "G3_snapshot_content_bound", "G4_isolation_verified"):
        claimed = gate_claimed(iso.get(gname), gname, v)
        resolved = resolve_evidence((iso.get(gname) or {}).get("evidence"),
                                    args.evidence_root, read_log, v, gname) if claimed else False
        gates[gname] = bool(claimed and resolved)

    if gates["G3_snapshot_content_bound"] and v.has("SNAPSHOT_UNVERIFIED"):
        gates["G3_snapshot_content_bound"] = False
    if gates["G2_effective_input_binding"] and (v.has("BINDING_MISMATCH") or v.has("BINDING_INCOMPLETE")
                                                or v.has("COUNT_INVARIANT_BROKEN")
                                                or v.has("DENYLIST_SCHEMA_NOT_CONSUMED")):
        gates["G2_effective_input_binding"] = False
        v.add("GATE_MEANING_UNSUPPORTED", "G2 claimed 但绑定/计数/入口实测不成立")

    # OS 身份格式校验（无条件，独立于 G4 是否 claimed）
    nt_all = iso.get("negative_test") or {}
    for fld, val in (("custody_principal_os_identity", iso.get("custody_principal_os_identity")),
                     ("negative_test.subject_os_identity", nt_all.get("subject_os_identity")),
                     ("positive_test.subject_os_identity",
                      (iso.get("positive_test") or {}).get("subject_os_identity"))):
        if not valid_os_identity(val):
            v.add("OS_IDENTITY_INVALID",
                  "%s=%r 不是可接受的 OS 身份（拒绝 agent UUID/空值）" % (fld, val))

    if gates["G4_isolation_verified"]:
        g4 = []
        if iso.get("isolation_mechanism") not in PERMITTED_MECHANISMS:
            g4.append("isolation_mechanism=%r" % iso.get("isolation_mechanism"))
        if not iso.get("permission_matrix_sha256"):
            g4.append("缺 permission_matrix_sha256")
        if iso.get("training_context_cannot_self_authorize") is not True:
            g4.append("未证明训练主体不能自行授权（管理员/owner/daemon 路径）")
        if iso.get("admin_owner_daemon_paths_excluded") is not True:
            g4.append("未证明管理员/owner/daemon 路径已排除")
        nt = iso.get("negative_test") or {}
        if nt.get("exit_code_source") != "os_process" or nt.get("subprocess_checked") is not True:
            g4.append("negative_test 未覆盖子进程或非 os_process")
        if nt.get("stat_result") is None or nt.get("open_result") is None:
            g4.append("negative_test 未分开记录 stat 与 open")
        if nt.get("open_result") != "denied":
            g4.append("negative_test.open_result != denied")
        if (iso.get("positive_test") or {}).get("exit_code") != 0:
            g4.append("positive_test 未成功")
        if v.has("OS_IDENTITY_INVALID"):
            g4.append("OS 身份字段非法")
        if g4:
            gates["G4_isolation_verified"] = False
            v.add("G4_MECHANISM_INSUFFICIENT", "; ".join(g4))

    detail["gates"] = gates

    # ---------- 结论（D15）
    # 任何阻断级 reason_code 出现 ⇒ 不得 ACCEPTED / SYNTHETIC_ACCEPTED。
    BLOCKING = {
        "PROTOCOL_SHA_MISMATCH", "MODE_TYPE_INVALID",
        "AUTH_MISSING", "AUTH_UNTRUSTED", "AUTH_SCOPE_INSUFFICIENT", "AUTH_BEFORE_ACCESS",
        "EVIDENCE_SELF_DECLARED", "EVIDENCE_NOT_RESOLVABLE", "EVIDENCE_HASH_INVALID",
        "EVIDENCE_HASH_MISMATCH", "EVIDENCE_PRODUCER_MISSING", "GATE_MEANING_UNSUPPORTED",
        "CANDIDATE_EMPTY", "RECORD_MISSING_FIELD",
        "DENYLIST_SCHEMA_UNSUPPORTED", "DENYLIST_UNKNOWN_KEY", "DENYLIST_ITEM_INVALID",
        "DENYLIST_EMPTY", "DENYLIST_COUNTS_DECLARED_MISSING", "DENYLIST_COUNTS_PARTIAL",
        "DENYLIST_COUNT_MISMATCH", "COUNT_INVARIANT_BROKEN",
        "DENOMINATOR_MISSING", "DENOMINATOR_SHORTFALL",
        "BINDING_INCOMPLETE", "BINDING_MISMATCH", "BYTES_REBOUND",
        "DENYLIST_SCHEMA_NOT_CONSUMED", "VALIDATOR_EXIT_MISMATCH", "EXIT_CODE_NOT_OS_PROCESS",
        "INTERSECTION_NONZERO", "SNAPSHOT_UNVERIFIED", "OS_IDENTITY_INVALID",
        "G4_MECHANISM_INSUFFICIENT", "OVERALL_SELF_DECLARED",
    }
    blocking_hit = sorted(set(v.codes()) & BLOCKING)
    detail["blocking_reason_codes"] = blocking_hit

    core_ok = (exit_code == 0) and not blocking_hit and not v.has("EXIT_CODE_NOT_OS_PROCESS")

    if synthetic:
        verdict = "SYNTHETIC_ACCEPTED" if (core_ok and gates["G1_namespace_compatibility"]
                                           and gates["G2_effective_input_binding"]
                                           and gates["G3_snapshot_content_bound"]) else "COMPUTATION_ONLY"
    else:
        verdict = "ACCEPTED" if (core_ok and all(gates.values())) else "COMPUTATION_ONLY"

    declared_overall = (manifest.get("records") or {}).get("overall")
    if declared_overall not in (None, verdict):
        v.add("OVERALL_SELF_DECLARED",
              "被检清单自述 overall=%r != 机器判定 %r；自述不参与验收" % (declared_overall, verdict))
    elif declared_overall == "ACCEPTED" and verdict != "ACCEPTED":
        v.add("OVERALL_SELF_DECLARED", "被检清单自述 ACCEPTED，机器判定不同")

    if verdict != "ACCEPTED" and not v.items and synthetic:
        pass
    detail["core_ok"] = core_ok
    return verdict, detail, checks


# ---------------------------------------------------------------------- main


def build_parser():
    ap = argparse.ArgumentParser(description="KAGGLE-27 exclusion-proof manifest checker (v4)")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mode", choices=["production", "test"], default="production")
    ap.add_argument("--expect", default=None)
    ap.add_argument("--records")
    ap.add_argument("--denylist")
    ap.add_argument("--snapshot")
    ap.add_argument("--validator-src")
    ap.add_argument("--validator-mode", choices=["real_cli", "test_double"], default="real_cli")
    ap.add_argument("--evidence-root")
    ap.add_argument("--authorization")
    ap.add_argument("--authorization-trust-anchor")
    ap.add_argument("--out")
    ap.add_argument("--workdir", default=None,
                    help="临时工作目录（必须位于可写工作区内；默认 <--out 同级>/_k27v4_tmp）")
    return ap


def main() -> int:
    args = build_parser().parse_args()

    if args.mode == "test" and args.expect is None:
        print("--mode test 必须显式提供 --expect；禁止从被检 overall 取期望")
        return EXIT_CANNOT
    if args.mode == "production" and args.expect is not None:
        print("--mode production 不接受 --expect：生产 exit 0 只代表真实 ACCEPTED")
        return EXIT_CANNOT

    read_log = ReadLog()
    v = Violations()
    try:
        verdict, detail, checks = evaluate(args, read_log, v)
    except json.JSONDecodeError as exc:
        verdict, detail, checks = "BLOCKED", {}, {}
        v.add("EVIDENCE_NOT_RESOLVABLE", "输入无法解析: %s" % exc)

    report = {
        "checker": "verify_manifest.py",
        "checker_version": "v4",
        "mode": args.mode,
        "protocol_sha256": sha256_bytes(Path(args.protocol).read_bytes()),
        "computed_verdict": verdict,
        "expected_verdict": args.expect,
        "violations": v.items,
        "reason_codes": sorted(set(v.codes())),
        "read_log": read_log.entries,
        "checks": checks,
        "detail": {k: val for k, val in detail.items() if not k.startswith("_")},
        "exit_semantics": {
            "production": "exit 0 当且仅当机器判定为真实 ACCEPTED",
            "test": "exit 0 当且仅当机器判定等于调用方显式给出的 --expect",
            "expect_source": "调用方" if args.mode == "test" else "n/a（生产模式不取期望）",
        },
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_bytes((json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n").encode())

    if args.mode == "production":
        rc = EXIT_OK if verdict == "ACCEPTED" else EXIT_MISMATCH
        if verdict == "BLOCKED" and not read_log.protected_entries():
            rc = EXIT_MISMATCH
    else:
        rc = EXIT_OK if verdict == args.expect else EXIT_MISMATCH

    print("mode=%s computed=%s expect=%r exit=%d violations=%d codes=%s protected_reads=%d"
          % (args.mode, verdict, args.expect, rc, len(v), ",".join(sorted(set(v.codes()))) or "-",
             len(read_log.protected_entries())))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
