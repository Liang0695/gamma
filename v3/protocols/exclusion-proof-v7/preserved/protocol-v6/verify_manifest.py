#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KAGGLE-27 · 排除证明 execution manifest 机器验收器（v5）。

针对 Q0 v4 集中复核（报告 01a1125c-46f4-762d-87e6-f35d6646544b）的四组阻断与两项应修：

  B1 证据含义/可信来源 + 生产禁替身
     - 生产模式拒绝任何测试替身（test_double / synthetic_consumer）
     - 证据必须**解析正文**并满足门槛含义契约；produced_by 必须是合法主体且**不同于候选作者**；
       produced_at 必须是合法 RFC3339；kind=trusted_approval 必须解析为批准记录
     - G4 的 negative_test 必须是**真实 OS 退出码的非零拒绝**
  B2 全部实际对象的读取前授权
     - 授权 scope 改为**对象级**（kind + 规范路径 + 字节摘要 + 用途），必须覆盖本次**将要打开的每一个**受保护对象
     - 合成模式只能由**调用方**（--mode test --allow-synthetic --fixture-root）开启，被检 manifest 不能自行选择
     - 所有路径做 root containment（拒绝 ../ 与绝对路径越界）
     - **输入/执行绑定失败在启动计算之前终止**
  B3 独立基线与严格输入计数
     - 独立冻结基线必须由 --baseline 提供且摘要命中协议记录
     - 必比项由基线**冻结**，不得自选；分母**从运行输入实算**，不得取清单自述
     - 计数必须是严格非负整数；记录身份必须是非空字符串；rejected_count 必须与实测一致
  B4 完整 CLI 与原/派生字节绑定、可成功的验收轮
     - 绑定 `v3/cli.py` 的**完整导入闭包**，并快照到不可变执行副本，从快照运行
     - 验收轮必须使用**原始字节**（不再重写输入）
     - **诊断轮**（必有交集，预期 7）与**验收轮**（真实输入，预期 0）分开记录
  Y1 records 顶层为 list 等形态错误 → 稳定 reason_code + 始终输出报告
  Y2 stat 与 open 口径分离（stat 单列）

附加：真实 open 哨兵（audit hook）与计算哨兵，与内部 read_log 请求数分开报告。
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

EXIT_OK = 0
EXIT_MISMATCH = 1
EXIT_CANNOT = 2

SUPPORTED_DENYLIST_SCHEMA = ["v2-dh-exclusion-denylist-1"]
SUPPORTED_AUTH_SCHEMA = ["v3-exclusion-proof-read-authorization-2"]
REQUIRED_ARRAYS = ["d_family_ids", "h_family_ids", "exposed_family_ids"]
ALLOWED_TOP_KEYS = set(REQUIRED_ARRAYS + ["denylist_schema_version", "counts_declared",
                                          "source_sha256", "produced_by", "produced_at"])
HEX64 = re.compile(r"^[0-9a-f]{64}$")
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
SID_RE = re.compile(r"^S-1-[0-9-]+$")
ACCOUNT_RE = re.compile(r"^[A-Za-z0-9._\\-]+\\[A-Za-z0-9._$-]+$")
POSIX_RE = re.compile(r"^(uid=)?[0-9]+(\.[0-9]+)?(\([^)]*\))?$")
RFC3339 = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})$")

PERMITTED_MECHANISMS = {"independent_host", "controlled_vm",
                        "separate_os_account_with_broken_acl_inheritance"}
EVIDENCE_KINDS = {"artifact", "command_log", "custody_record", "trusted_approval"}
TEST_DOUBLE_MODES = {"test_double", "synthetic_consumer"}
PROTECTED_KINDS = {"candidate", "denylist", "validator_closure", "snapshot", "evidence"}

# 每个门槛在证据正文里必须出现的字段（含义契约）
GATE_PAYLOAD_REQUIRED = {
    "G1_namespace_compatibility": ["namespace_v3", "namespace_v2", "derivation_rule",
                                   "extractor_id", "sample_count"],
    "G2_effective_input_binding": ["bound_objects", "approved_by"],
    "G3_snapshot_content_bound": ["snapshot_sha256"],
    "G4_isolation_verified": ["mechanism", "evaluator_subject_os_identity",
                              "training_context_cannot_self_authorize",
                              "admin_owner_daemon_paths_excluded", "negative_test", "positive_test"],
}

_AUDIT = {"protected": set(), "protected_opens": [], "compute_invocations": 0,
          "all_opens": []}
_HOOK_INSTALLED = False
_ALL_OPEN_CAP = 20000


def _install_audit_hook():
    """真实 open 哨兵：记录**每一次** open 事件路径，覆盖整个进程启动过程。

    `all_opens` 是全量事件流，供**外部**独立核算受保护对象的实际打开次数，
    不受本验收器内部"受保护集合"定义影响。
    """
    global _HOOK_INSTALLED
    if _HOOK_INSTALLED:
        return
    _HOOK_INSTALLED = True

    def hook(event, args):
        if event == "open":
            try:
                p = os.path.abspath(str(args[0]))
            except Exception:
                return
            if len(_AUDIT["all_opens"]) < _ALL_OPEN_CAP:
                _AUDIT["all_opens"].append(p)
            if p in _AUDIT["protected"]:
                _AUDIT["protected_opens"].append(p)

    sys.addaudithook(hook)


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
    def __init__(self):
        self.entries = []
        self._cache = {}
        self._counts = {}

    def read(self, path, phase, purpose):
        p = str(Path(path).resolve())
        self._counts[p] = self._counts.get(p, 0) + 1
        self.entries.append({"phase": phase, "purpose": purpose, "path": p,
                             "read_ordinal": self._counts[p]})
        if p not in self._cache:
            self._cache[p] = Path(p).read_bytes()
        return self._cache[p], self._counts[p]

    def protected_entries(self):
        return [e for e in self.entries if e["phase"] >= 2]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def newline_mode(b):
    crlf = b.count(b"\r\n")
    lf = b.count(b"\n") - crlf
    return "mixed" if (crlf and lf) else ("crlf" if crlf else "lf")


def norm_token(v):
    return str(v).strip().lower()


def valid_os_identity(s):
    if not isinstance(s, str) or not s.strip():
        return False
    s = s.strip()
    if UUID_RE.match(s):
        return False
    return bool(SID_RE.match(s) or ACCOUNT_RE.match(s) or POSIX_RE.match(s))


def is_nonneg_int(x):
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def contained(path: Path, root: Path):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _atomic_write(path: Path, data: bytes):
    """把**已核验的 buffer**原子写出（先写临时件再替换），避免从可变源重新复制。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _dedupe(pairs):
    seen, out = set(), []
    for k, p in pairs:
        key = (k, str(Path(p)))
        if key in seen:
            continue
        seen.add(key)
        out.append((k, p))
    return out


def is_real_git_blob_id(src: Path, rel: str, expected: str) -> bool:
    """按 Git blob 规范核对：sha1(b\"blob <len>\\0\" + bytes) == expected。

    拒绝全零、SHA256 截断等伪造形态。
    """
    if not isinstance(expected, str) or not re.match(r"^[0-9a-f]{40}$", expected):
        return False
    if expected == "0" * 40:
        return False
    fp = src / rel
    if not fp.exists():
        return False
    b = fp.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() == expected


def safe_json(raw, what, v, code="INPUT_CONTAINER_INVALID"):
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        v.add("INPUT_PARSE_ERROR", "%s 无法解析: %s" % (what, exc))
        return None
    return obj


# ------------------------------------------------------ CLI closure (B4)


def compute_import_closure(src: Path, entry="v3/cli.py"):
    """静态扫描 v3 包内导入（含相对导入），返回相对路径集合。

    必须是**实际会被执行**的文件集：`v3/cli.py` 及其递归导入的包内模块。
    """
    seen = set()
    queue = [entry]
    # `python -m v3.cli` 会隐式导入根包初始化文件；必须显式纳入，不能退化为 namespace package
    if (src / "v3" / "__init__.py").exists():
        queue.append("v3/__init__.py")
    pat_abs = re.compile(r"^\s*(?:from\s+(v3[\w\.]*)\s+import|import\s+(v3[\w\.]*))", re.M)
    pat_rel = re.compile(r"^\s*from\s+(\.+)([\w\.]*)\s+import", re.M)
    while queue:
        rel = queue.pop()
        if rel in seen:
            continue
        f = src / rel
        if not f.exists():
            continue
        seen.add(rel)
        try:
            text = f.read_text(encoding="utf-8")
        except Exception:  # noqa: BLE001
            continue
        pkg_dir = Path(rel).parent  # 相对导入基准目录
        cands = []
        for m in pat_abs.finditer(text):
            cands.append((m.group(1) or m.group(2)).strip().replace(".", "/"))
        for m in pat_rel.finditer(text):
            dots, mod = m.group(1), m.group(2)
            base = pkg_dir
            for _ in range(len(dots) - 1):
                base = base.parent
            cands.append((base / mod.replace(".", "/")).as_posix() if mod else base.as_posix())
        for base in cands:
            for cand in (base + ".py", base + "/__init__.py"):
                if (src / cand).exists():
                    queue.append(cand)
    return sorted(seen)


# ---------------------------------------------------------------- denylist


def validate_denylist(payload, v):
    if not isinstance(payload, dict):
        v.add("DENYLIST_CONTAINER_INVALID", "顶层必须是 object，实际 %s" % type(payload).__name__)
        return False, [], {}
    extra = sorted(set(payload.keys()) - ALLOWED_TOP_KEYS)
    if extra:
        v.add("DENYLIST_UNKNOWN_KEY", "未知顶层键: %s" % extra)
    if payload.get("denylist_schema_version") not in SUPPORTED_DENYLIST_SCHEMA:
        v.add("DENYLIST_SCHEMA_UNSUPPORTED", "schema=%r" % payload.get("denylist_schema_version"))
    for name in REQUIRED_ARRAYS:
        if name not in payload:
            v.add("DENYLIST_ITEM_INVALID", "缺少必需数组 %s" % name); continue
        arr = payload[name]
        if not isinstance(arr, list):
            v.add("DENYLIST_ITEM_INVALID", "%s 必须是 array" % name); continue
        for i, it in enumerate(arr):
            if isinstance(it, bool) or not isinstance(it, str):
                v.add("DENYLIST_ITEM_INVALID", "%s[%d]=%r 类型非法" % (name, i, it))
            elif not it.strip():
                v.add("DENYLIST_ITEM_INVALID", "%s[%d] 空串/纯空白非法" % (name, i))

    declared = payload.get("counts_declared")
    if declared is None:
        v.add("DENYLIST_COUNTS_DECLARED_MISSING", "counts_declared 缺失")
    elif not isinstance(declared, dict):
        v.add("DENYLIST_COUNTS_DECLARED_MISSING", "counts_declared 必须是 object")
    else:
        miss = [n for n in REQUIRED_ARRAYS if n not in declared]
        if miss:
            v.add("DENYLIST_COUNTS_PARTIAL", "未声明: %s" % miss)
        for n in REQUIRED_ARRAYS:
            if n in declared and not is_nonneg_int(declared[n]):
                v.add("COUNT_TYPE_INVALID", "counts_declared.%s=%r 不是严格非负整数" % (n, declared[n]))

    fatal = [c for c in ("DENYLIST_ITEM_INVALID", "DENYLIST_SCHEMA_UNSUPPORTED",
                         "DENYLIST_UNKNOWN_KEY", "DENYLIST_CONTAINER_INVALID") if v.has(c)]
    if fatal:
        return False, [], {}

    arrays_of, per, order = {}, {}, []
    raw_total = within_total = 0
    for name in REQUIRED_ARRAYS:
        arr = payload[name]
        raw_total += len(arr)
        seen, within = set(), 0
        for it in arr:
            t = norm_token(it)
            if t in seen:
                within += 1
            else:
                seen.add(t)
                if t not in arrays_of:
                    arrays_of[t] = []
                    order.append(it)
                arrays_of[t].append(name)
        within_total += within
        per[name] = {"raw_count": len(arr), "effective_count": len(seen), "duplicate_count": within}

    eff = len(arrays_of)
    cross = sum(len(a) - 1 for a in arrays_of.values() if len(a) > 1)
    dup = within_total + cross
    if raw_total != eff + dup:
        v.add("COUNT_INVARIANT_BROKEN", "raw=%d != eff=%d + dup=%d" % (raw_total, eff, dup))
    if declared and isinstance(declared, dict):
        for n in REQUIRED_ARRAYS:
            if n in declared and declared[n] != per[n]["effective_count"]:
                v.add("DENYLIST_COUNT_MISMATCH", "%s 声明 %r 实测 %d"
                      % (n, declared[n], per[n]["effective_count"]))
    if eff == 0:
        v.add("DENYLIST_EMPTY", "三数组去重并集为空")
    return True, order, {
        "per_array": per, "raw_total": raw_total, "effective_total": eff,
        "duplicate_total": dup, "cross_array_duplicate_total": cross,
        "source_array_of_token": {k: sorted(set(x)) for k, x in arrays_of.items()},
    }


def validate_records(tasks, v):
    """严格身份与计数：身份必须是非空字符串；任一非法 ⇒ 整批失败。"""
    if not isinstance(tasks, list):
        v.add("RECORD_CONTAINER_INVALID", "records 顶层必须是 list，实际 %s" % type(tasks).__name__)
        return 0, 0
    if len(tasks) == 0:
        v.add("CANDIDATE_EMPTY", "候选记录 0 条")
    eff = 0
    for i, rec in enumerate(tasks):
        if not isinstance(rec, dict):
            v.add("RECORD_CONTAINER_INVALID", "records[%d] 不是 object" % i); continue
        bad = False
        for f in ("problem_family_id", "repo_family"):
            val = rec.get(f, None)
            if f not in rec:
                v.add("RECORD_MISSING_FIELD", "records[%d].%s 缺失" % (i, f)); bad = True
            elif not isinstance(val, str):
                v.add("RECORD_IDENTITY_TYPE_INVALID",
                      "records[%d].%s=%r 不是字符串（dict/list/bool/int 一律拒绝）" % (i, f, val))
                bad = True
            elif not val.strip():
                v.add("RECORD_MISSING_FIELD", "records[%d].%s 空串/纯空白" % (i, f)); bad = True
        if not bad:
            eff += 1
    return len(tasks), eff


# ---------------------------------------------------------------- evidence


def resolve_and_parse_evidence(evidence, root, read_log, v, gate_name, produced_by_forbidden,
                               trusted_principals=None, approvals=None, synthetic=False):
    """解析证据正文并核对门槛含义（B1）。返回 (ok, parsed_payloads)。"""
    if not isinstance(evidence, list) or not evidence:
        v.add("EVIDENCE_SELF_DECLARED", "%s: 无证据引用" % gate_name)
        return False, []
    required = GATE_PAYLOAD_REQUIRED.get(gate_name, [])
    parsed, usable = [], 0
    for e in evidence:
        if not isinstance(e, dict):
            v.add("EVIDENCE_NOT_RESOLVABLE", "%s: 证据项不是 object" % gate_name); continue
        kind = e.get("kind")
        if kind == "self_declared" or kind not in EVIDENCE_KINDS:
            v.add("EVIDENCE_SELF_DECLARED", "%s: kind=%r 永不足够" % (gate_name, kind)); continue
        digest = e.get("sha256")
        if not isinstance(digest, str) or not HEX64.match(digest):
            v.add("EVIDENCE_HASH_INVALID", "%s: sha256=%r" % (gate_name, digest)); continue
        pb, pa = e.get("produced_by"), e.get("produced_at")
        if not isinstance(pb, str) or not pb.strip():
            v.add("EVIDENCE_PRODUCER_MISSING", "%s: 缺 produced_by" % gate_name); continue
        if UUID_RE.match(pb.strip()):
            v.add("EVIDENCE_PRODUCER_MISSING", "%s: produced_by 是 agent UUID" % gate_name); continue
        if not isinstance(pa, str) or not RFC3339.match(pa.strip()):
            v.add("EVIDENCE_TIME_INVALID", "%s: produced_at=%r 不是合法 RFC3339" % (gate_name, pa)); continue
        if produced_by_forbidden and pb.strip().lower() == produced_by_forbidden.strip().lower():
            v.add("EVIDENCE_SELF_PRODUCED",
                  "%s: 证据由候选作者本人产出（%s），不构成独立证据" % (gate_name, pb)); continue
        rel = e.get("path")
        if not isinstance(rel, str) or not rel:
            v.add("EVIDENCE_NOT_RESOLVABLE", "%s: path 缺失" % gate_name); continue
        if root is None:
            v.add("EVIDENCE_NOT_RESOLVABLE", "%s: 未提供 --evidence-root" % gate_name); continue
        try:
            target = Path(rel) if Path(rel).is_absolute() else (Path(root) / rel)
            target = target.resolve()
        except Exception:  # noqa: BLE001
            v.add("EVIDENCE_NOT_RESOLVABLE", "%s: path 非法 %r" % (gate_name, rel)); continue
        if not contained(target, Path(root)):
            v.add("EVIDENCE_NOT_RESOLVABLE", "%s: path 逃出 evidence root: %s" % (gate_name, rel)); continue
        if not target.exists():
            v.add("EVIDENCE_NOT_RESOLVABLE", "%s: 路径不存在: %s" % (gate_name, rel)); continue
        raw, cnt = read_log.read(target, phase=2, purpose="evidence")
        if cnt > 1:
            v.add("BYTES_REBOUND", "证据 %s 被读 %d 次" % (rel, cnt))
        if sha256_bytes(raw) != digest:
            v.add("EVIDENCE_HASH_MISMATCH", "%s: 证据 %s 摘要不符" % (gate_name, rel)); continue
        payload = safe_json(raw, "evidence %s" % rel, v)
        if payload is None:
            continue
        if not isinstance(payload, dict):
            v.add("EVIDENCE_MEANING_UNSUPPORTED", "%s: 证据正文必须是 object" % gate_name); continue
        missing = [k for k in required if k not in payload]
        if missing:
            v.add("EVIDENCE_MEANING_UNSUPPORTED",
                  "%s: 证据正文缺含义字段 %s（仅存在与哈希相等不足够）" % (gate_name, missing))
            continue
        if kind == "trusted_approval" and not (payload.get("approved_by") and payload.get("approval_id")):
            v.add("EVIDENCE_MEANING_UNSUPPORTED",
                  "%s: kind=trusted_approval 但正文不是批准记录" % gate_name)
            continue
        # 可信生产者 / 批准者必须由**调用方受控配置**核验，不能靠自填字段或"名字不同"
        if trusted_principals is None:
            if synthetic:
                pass
            else:
                v.add("EVIDENCE_TRUST_ROOT_MISSING",
                      "%s: 未提供 --trusted-principals 受控主体注册表，无法核验证据来源" % gate_name)
                continue
        else:
            if pb.strip() not in trusted_principals:
                v.add("EVIDENCE_UNTRUSTED_PRODUCER",
                      "%s: produced_by=%r 不在调用方受控主体注册表内（自填的名字不构成独立来源）"
                      % (gate_name, pb))
                continue
            if produced_by_forbidden and pb.strip().lower() == produced_by_forbidden.strip().lower():
                v.add("EVIDENCE_SELF_PRODUCED", "%s: 证据由候选作者本人产出" % gate_name)
                continue
        if kind == "trusted_approval":
            if approvals is None and not synthetic:
                v.add("EVIDENCE_APPROVAL_UNTRUSTED",
                      "%s: kind=trusted_approval 但未提供 --approvals 受控批准注册表" % gate_name)
                continue
            if approvals is not None:
                aref = payload.get("approval_id")
                if not isinstance(aref, str) or aref not in approvals:
                    v.add("EVIDENCE_APPROVAL_UNTRUSTED",
                          "%s: approval_id=%r 不在调用方受控批准注册表内" % (gate_name, aref))
                    continue
                appr = approvals[aref]
                if appr.get("approved_by") and payload.get("approved_by") \
                        and appr["approved_by"] != payload.get("approved_by"):
                    v.add("EVIDENCE_APPROVAL_UNTRUSTED",
                          "%s: 证据 approved_by 与受控批准记录不一致" % gate_name)
                    continue
        # 主体必须是受信注册表里声明过的 OS 身份
        if trusted_principals is not None:
            rec = trusted_principals[pb.strip()]
            if not valid_os_identity((rec or {}).get("os_identity")):
                v.add("EVIDENCE_UNTRUSTED_PRODUCER",
                      "%s: 受控注册表中 %r 的 os_identity 非法或无记录" % (gate_name, pb))
                continue
        parsed.append(payload)
        usable += 1
    if usable == 0:
        v.add("EVIDENCE_MEANING_UNSUPPORTED", "%s: 无一条证据同时满足格式/哈希/含义" % gate_name)
        return False, []
    return True, parsed


# ------------------------------------------------------------- authorization


def check_authorization(auth, auth_bytes, anchor_bytes, v, planned):
    """对象级授权校验（B2）。只比较**路径身份**，不打开任何受保护对象。

    planned: [(kind, path)] —— 纯路径字符串。
    返回 (ok, granted)，granted[(kind, canonical_path)] = sha256（供授权后逐字节核对）。
    """
    if auth is None:
        v.add("AUTH_MISSING", "非合成模式缺少 --authorization 可信授权记录")
        return False, {}
    if auth.get("authorization_schema_version") not in SUPPORTED_AUTH_SCHEMA:
        v.add("AUTH_SCHEMA_UNSUPPORTED", "authorization_schema_version=%r"
              % auth.get("authorization_schema_version"))
        return False, {}
    if anchor_bytes is None:
        v.add("AUTH_UNTRUSTED", "未提供 --trust-anchor-file（信任锚必须由受控调用方持有）")
        return False, {}
    if sha256_bytes(auth_bytes) != sha256_bytes(anchor_bytes):
        v.add("AUTH_UNTRUSTED", "授权记录摘要与调用方信任锚不符")
        return False, {}
    issued = auth.get("issued_by") or {}
    if not valid_os_identity(issued.get("os_identity")):
        v.add("AUTH_IDENTITY_INVALID", "issued_by.os_identity=%r 非法" % issued.get("os_identity"))
        return False, {}
    if not isinstance(issued.get("role"), str) or not issued.get("role").strip():
        v.add("AUTH_IDENTITY_INVALID", "issued_by.role 缺失")
        return False, {}
    if not isinstance(auth.get("approved_at"), str) or not RFC3339.match(str(auth.get("approved_at")).strip()):
        v.add("AUTH_IDENTITY_INVALID", "approved_at=%r 不是合法 RFC3339" % auth.get("approved_at"))
        return False, {}
    iso = auth.get("isolation") or {}
    if iso.get("verified") is not True:
        v.add("AUTH_ISOLATION_UNVERIFIED", "authorization.isolation.verified != true")
        return False, {}
    if iso.get("mechanism") not in PERMITTED_MECHANISMS:
        v.add("AUTH_ISOLATION_UNVERIFIED", "isolation.mechanism=%r" % iso.get("mechanism"))
        return False, {}
    if iso.get("exit_code_source") != "os_process":
        v.add("AUTH_ISOLATION_UNVERIFIED", "isolation.exit_code_source != os_process")
        return False, {}
    scope = auth.get("scope") or {}
    if scope.get("read_permitted") is not True:
        v.add("AUTH_SCOPE_INSUFFICIENT", "scope.read_permitted != true")
        return False, {}
    objects = scope.get("objects")
    if not isinstance(objects, list) or not objects:
        v.add("AUTH_OBJECTS_MISSING", "scope.objects 必须以对象级清单给出（kind/path/sha256/purpose）")
        return False, {}
    granted = {}
    for o in objects:
        if not isinstance(o, dict):
            v.add("AUTH_OBJECTS_MISSING", "scope.objects 项不是 object"); return False, {}
        if o.get("kind") not in PROTECTED_KINDS or not o.get("path") or not o.get("purpose"):
            v.add("AUTH_OBJECTS_MISSING", "scope.objects 项字段不全: %r" % o); return False, {}
        if not isinstance(o.get("sha256"), str) or not HEX64.match(str(o.get("sha256"))):
            v.add("AUTH_OBJECTS_MISSING", "scope.objects 项 sha256 非法: %r" % o); return False, {}
        try:
            key = (o["kind"], str(Path(o["path"]).resolve()))
        except Exception:  # noqa: BLE001
            v.add("AUTH_OBJECTS_MISSING", "scope.objects 项 path 非法"); return False, {}
        granted[key] = o["sha256"]
    missing = []
    for kind, path in planned:
        key = (kind, str(Path(path).resolve()))
        if key not in granted:
            missing.append("%s:%s" % (kind, Path(path).name))
    if missing:
        v.add("AUTH_SCOPE_INSUFFICIENT",
              "授权未覆盖本次实际要打开的对象: %s（类别级 scope 不证明任意新路径或字节获准）"
              % sorted(set(missing)))
        return False, {}
    return True, granted


# -------------------------------------------------------------------- main


class Ctx:
    pass


def evaluate(args, read_log, v):
    detail, checks = {}, {}
    _install_audit_hook()

    # ---------- 阶段 0：协议 / 清单 / 授权记录 / 信任锚 / 冻结基线（预先获准的非敏感件）
    protocol_bytes, _ = read_log.read(args.protocol, phase=0, purpose="protocol")
    checks["protocol_sha256"] = sha256_bytes(protocol_bytes)
    protocol = safe_json(protocol_bytes, "protocol", v)
    if protocol is None:
        return "BLOCKED", detail, checks

    manifest_bytes, _ = read_log.read(args.manifest, phase=0, purpose="manifest")
    manifest = safe_json(manifest_bytes, "manifest", v)
    if not isinstance(manifest, dict):
        v.add("INPUT_CONTAINER_INVALID", "manifest 顶层必须是 object")
        return "BLOCKED", detail, checks

    pref = manifest.get("protocol_ref") or {}
    if pref.get("protocol_sha256") != sha256_bytes(protocol_bytes):
        v.add("PROTOCOL_SHA_MISMATCH", "manifest=%r 实际=%s"
              % (pref.get("protocol_sha256"), sha256_bytes(protocol_bytes)))
        return "BLOCKED", detail, checks

    auth_bytes = auth = None
    if args.authorization:
        auth_bytes, _ = read_log.read(args.authorization, phase=0, purpose="authorization")
        auth = safe_json(auth_bytes, "authorization", v)
    anchor_bytes = None
    if args.trust_anchor_file:
        anchor_bytes, _ = read_log.read(args.trust_anchor_file, phase=0, purpose="trust_anchor")
    trusted_principals = None
    if args.trusted_principals:
        tp_raw, _ = read_log.read(args.trusted_principals, phase=0, purpose="trusted_principals")
        tp = safe_json(tp_raw, "trusted principals", v)
        if isinstance(tp, dict):
            trusted_principals = {k: val for k, val in tp.items()
                                  if isinstance(k, str) and isinstance(val, dict)}
    approvals = None
    if args.approvals:
        ap_raw, _ = read_log.read(args.approvals, phase=0, purpose="approvals")
        ap = safe_json(ap_raw, "approvals", v)
        if isinstance(ap, dict):
            approvals = {k: val for k, val in ap.items()
                         if isinstance(k, str) and isinstance(val, dict)}

    # ---------- 模式：只能由调用方开启（B2）
    mode_obj = manifest.get("execution_mode") or {}
    synth_raw = mode_obj.get("synthetic_mode")
    if not isinstance(synth_raw, bool):
        v.add("MODE_TYPE_INVALID", "synthetic_mode=%r 不是 JSON 布尔" % (synth_raw,))
    manifest_claims_synth = synth_raw is True
    if args.mode == "production" and manifest_claims_synth:
        v.add("SYNTHETIC_NOT_PERMITTED_IN_PRODUCTION",
              "被检清单自填 synthetic_mode=true；合成模式只能由调用方以 --mode test --allow-synthetic 开启")
    synthetic = (args.mode == "test") and args.allow_synthetic and manifest_claims_synth
    if synthetic and not args.fixture_root:
        v.add("FIXTURE_ROOT_MISSING", "合成模式必须由调用方给出 --fixture-root")
    detail["synthetic_mode"] = synthetic
    detail["validator_mode"] = args.validator_mode

    if args.mode == "production" and args.validator_mode in TEST_DOUBLE_MODES:
        v.add("PRODUCTION_FORBIDS_TEST_DOUBLE",
              "生产模式禁止测试替身（--validator-mode=%s）" % args.validator_mode)

    # ---------- 阶段 1：受保护对象清单
    # 只用**调用方受信的预期路径与摘要**核对授权；此阶段**不打开任何受保护对象、不执行导入扫描**。
    planned_manifest = []
    if args.records:
        planned_manifest.append(("candidate", args.records))
    if args.denylist:
        planned_manifest.append(("denylist", args.denylist))
    if args.snapshot:
        planned_manifest.append(("snapshot", args.snapshot))
    iso_pre = (manifest.get("records") or {}).get("isolation_acceptance") or {}
    for gname in GATE_PAYLOAD_REQUIRED:
        for e in ((iso_pre.get(gname) or {}).get("evidence") or []):
            if isinstance(e, dict) and isinstance(e.get("path"), str) and e["path"]:
                p = Path(e["path"])
                if not p.is_absolute():
                    p = Path(args.evidence_root or ".") / e["path"]
                planned_manifest.append(("evidence", str(p)))
    planned_manifest = _dedupe(planned_manifest)

    # 授权记录中声明的闭包对象（调用方受信清单；不读取源码即可比对）
    declared_closure = []
    if isinstance(auth, dict):
        for o in ((auth.get("scope") or {}).get("objects") or []):
            if isinstance(o, dict) and o.get("kind") == "validator_closure" and o.get("path"):
                declared_closure.append(str(Path(o["path"])))
    planned = _dedupe(planned_manifest + [("validator_closure", p) for p in declared_closure])
    if not declared_closure and not synthetic:
        v.add("AUTH_OBJECTS_MISSING",
              "授权记录未声明 validator_closure 对象；授权前不得导入扫描，故闭包清单必须由调用方给出")

    # root containment（纯路径判断，不读取内容）
    root = Path(args.fixture_root or args.authorized_root).resolve() \
        if (args.fixture_root or args.authorized_root) else None
    if root is None:
        v.add("ROOT_NOT_DECLARED", "必须由调用方给出 --fixture-root（合成）或 --authorized-root（生产）")
    else:
        for p in [args.records, args.denylist, args.snapshot, args.evidence_root]:
            if p and not contained(Path(p), root):
                v.add("PATH_OUTSIDE_ROOT", "输入路径越界（root containment 失败）: %s" % p)

    # 受信注册表必须由调用方提供，且不得落在 fixture/evidence 根之下
    for label, fp in (("--trusted-principals", args.trusted_principals),
                      ("--approvals", args.approvals)):
        if fp:
            for bad_root, bad_name in ((root, "fixture/authorized root"),
                                       (Path(args.evidence_root) if args.evidence_root else None,
                                        "evidence root")):
                if bad_root and contained(Path(fp), bad_root):
                    v.add("EVIDENCE_TRUST_ROOT_MISSING",
                          "%s 落在被检输入范围（%s）内，不构成受控信任根" % (label, bad_name))

    detail["planned_objects"] = [{"kind": k, "path": Path(p).name} for k, p in planned]

    # ---------- 授权判定（在任何受保护对象被打开之前完成）
    authorized = False
    granted = {}
    if synthetic:
        authorized = True
        checks["authorization"] = "synthetic fixture root 由调用方授权"
    else:
        authorized, granted = check_authorization(auth, auth_bytes, anchor_bytes, v, planned)
        checks["authorization"] = "authorized" if authorized else "denied"

    if not authorized:
        v.add("AUTH_BEFORE_ACCESS", "未通过对象级授权判定：受保护对象零打开、零计算")
        declared = (manifest.get("records") or {}).get("overall") \
            if isinstance(manifest.get("records"), dict) else None
        if declared == "ACCEPTED":
            v.add("OVERALL_SELF_DECLARED", "被检清单自述 ACCEPTED，机器判定 BLOCKED")
        return "BLOCKED", detail, checks

    # ---------- 阶段 2：先登记哨兵集合，再打开（单次读取）
    for kind, p in planned:
        _AUDIT["protected"].add(str(Path(p).resolve()))

    measured = {}

    def measure(key, path):
        raw, cnt = read_log.read(path, phase=2, purpose=key)
        if cnt > 1:
            v.add("BYTES_REBOUND", "%s 被读 %d 次（哈希与解析必须同一份字节）" % (key, cnt))
        measured[key] = {"file_bytes_sha256": sha256_bytes(raw), "byte_length": len(raw),
                         "newline_mode": newline_mode(raw)}
        return raw

    rec_raw = measure("candidate", args.records)
    den_raw = measure("denylist", args.denylist)
    snap_raw = measure("snapshot", args.snapshot) if args.snapshot else None

    binding = manifest.get("binding") or {}
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

    # ---------- 完整 CLI 闭包绑定 + 不可变执行快照（B4）
    # 授权已通过 ⇒ 现在才做导入扫描（会读取源码）。
    vsrc = Path(args.validator_src)
    # 哨兵必须在**扫描之前**登记：导入扫描本身会打开源码文件
    if vsrc.exists():
        for f in vsrc.rglob("*.py"):
            _AUDIT["protected"].add(str(f.resolve()))
    closure = compute_import_closure(vsrc) if args.validator_src else []
    authorized_closure = {str(Path(p).resolve()) for k, p in planned if k == "validator_closure"}
    unauth = [r for r in closure
              if str((vsrc / r).resolve()) not in authorized_closure]
    if unauth and not synthetic:
        v.add("AUTH_SCOPE_INSUFFICIENT",
              "实际导入闭包含未获授权的文件（授权前无法扫描，故须由调用方全量声明）: %s" % unauth)
    declared_set = ((binding.get("validator") or {}).get("file_set") or [])
    declared_paths = {e.get("path") for e in declared_set if isinstance(e, dict)}
    missing_closure = [p for p in closure if p not in declared_paths]
    if missing_closure:
        v.add("BINDING_INCOMPLETE", "完整 CLI 导入闭包未绑定，缺: %s" % missing_closure)
    closure_buffers = {}
    for rel in closure:
        _AUDIT["protected"].add(str((vsrc / rel).resolve()))
    for e in declared_set:
        if not isinstance(e, dict) or not e.get("path"):
            v.add("BINDING_INCOMPLETE", "file_set 项缺 path"); continue
        rel = e["path"]
        if Path(rel).is_absolute() or ".." in Path(rel).parts:
            v.add("PATH_OUTSIDE_ROOT", "file_set 越界路径: %s" % rel); continue
        f = (vsrc / rel).resolve()
        if not contained(f, vsrc) or not f.exists():
            v.add("BINDING_INCOMPLETE", "file_set 路径不存在或越界: %s" % rel); continue
        raw, cnt = read_log.read(f, phase=2, purpose="validator_closure")
        if cnt > 1:
            v.add("BYTES_REBOUND", "闭包文件 %s 被读 %d 次" % (rel, cnt))
        closure_buffers[rel] = raw
        if sha256_bytes(raw) != e.get("file_bytes_sha256"):
            v.add("BINDING_MISMATCH", "闭包文件 %s 摘要不符" % rel)
        if e.get("byte_length") != len(raw):
            v.add("BINDING_MISMATCH", "闭包文件 %s byte_length 不符" % rel)
        if e.get("newline_mode") != newline_mode(raw):
            v.add("BINDING_MISMATCH", "闭包文件 %s newline_mode 不符" % rel)
        # Git blob 必须是真实对象身份：按 blob 规范重算 SHA-1，拒绝全零/截断伪造
        if not is_real_git_blob_id(vsrc, rel, e.get("git_blob_sha1")):
            v.add("GIT_BLOB_INVALID",
                  "闭包文件 %s 的 git_blob_sha1 不是真实 Git 对象身份（全零/SHA256 截断一律拒绝）" % rel)
    # 根包初始化文件必须显式绑定（`python -m v3.cli` 会隐式导入它）
    root_init = "v3/__init__.py"
    if (vsrc / root_init).exists() and root_init not in declared_paths:
        v.add("BINDING_INCOMPLETE", "根包初始化文件未绑定: %s（不能退化为 namespace package）" % root_init)
    checks["closure_files"] = len(closure)

    # ---------- 输入契约
    den = safe_json(den_raw, "denylist", v)
    dn_ok, union, dstat = (False, [], {})
    if den is not None:
        dn_ok, union, dstat = validate_denylist(den, v)

    reg = safe_json(rec_raw, "records", v)
    tasks = reg.get("tasks") if isinstance(reg, dict) else reg
    raw_n, eff_n = validate_records(tasks, v)

    if snap_raw is not None:
        snap = binding.get("snapshot") or {}
        if snap.get("content_status") != "obtained":
            v.add("SNAPSHOT_UNVERIFIED", "content_status=%r" % snap.get("content_status"))
        elif sha256_bytes(snap_raw) != snap.get("file_bytes_sha256"):
            v.add("SNAPSHOT_UNVERIFIED", "snapshot 实算摘要 != 声明")

    # ---------- 独立冻结基线（B3）
    baseline = None
    if args.baseline:
        braw, _ = read_log.read(args.baseline, phase=2, purpose="baseline")
        bref = (protocol.get("denominator_baseline") or {}).get("baseline_ref") or {}
        if bref.get("file_bytes_sha256") != sha256_bytes(braw):
            v.add("BASELINE_UNVERIFIED", "冻结基线摘要 != 协议记录值")
        else:
            baseline = safe_json(braw, "baseline", v)
    else:
        v.add("BASELINE_UNVERIFIED", "未提供 --baseline 独立冻结基线")

    dcheck = manifest.get("denominator_check") or {}
    if baseline is None:
        v.add("DENOMINATOR_MISSING", "无独立基线，无法交叉核对")
    else:
        required = baseline.get("required_comparisons") or {}
        compared = dcheck.get("compared_denominators")
        if not isinstance(compared, list) or sorted(compared) != sorted(required.keys()):
            v.add("DENOMINATOR_SET_MISMATCH",
                  "必需比较项由基线冻结；声明 %r != 冻结 %r" % (compared, sorted(required.keys())))
        med = dcheck.get("measured")
        if not isinstance(med, dict):
            v.add("DENOMINATOR_TYPE_INVALID", "denominator_check.measured 必须是 object")
            med = {}
        # 从运行输入实算分母
        computed = {
            "v2_d8_count": dstat.get("per_array", {}).get("d_family_ids", {}).get("effective_count"),
            "v2_h24_count": dstat.get("per_array", {}).get("h_family_ids", {}).get("effective_count"),
            "v2_exposed_ids_in_union": dstat.get("per_array", {}).get("exposed_family_ids", {}).get("effective_count"),
            "v3_candidate_count": eff_n,
        }
        detail["computed_denominators"] = computed
        for name, exp in required.items():
            got = computed.get(name)
            declared = med.get(name)
            if not is_nonneg_int(declared):
                v.add("DENOMINATOR_TYPE_INVALID", "measured.%s=%r 不是严格非负整数" % (name, declared))
                continue
            if got is None:
                v.add("DENOMINATOR_MISSING", "%s 无法从运行输入实算" % name)
                continue
            if declared != got:
                v.add("DENOMINATOR_COUNT_MISMATCH", "%s 自述 %r != 实算 %r" % (name, declared, got))
            if isinstance(exp, int) and got < exp:
                v.add("DENOMINATOR_SHORTFALL", "%s 实算 %d < 冻结基线 %d" % (name, got, exp))

    # ---------- 计数严格类型与实测对照
    decl = manifest.get("counts") or {}
    d_dn, d_cd = decl.get("denylist") or {}, decl.get("candidate") or {}
    mcounts = {
        "records_raw_count": raw_n, "records_effective_count": eff_n,
        "records_rejected_count": raw_n - eff_n,
        "denylist_raw_count": dstat.get("raw_total"),
        "denylist_effective_count": dstat.get("effective_total"),
        "denylist_duplicate_count": dstat.get("duplicate_total"),
        "denylist_cross_array_duplicate_count": dstat.get("cross_array_duplicate_total"),
        "denylist_rejected_count": 0,
    }
    for k, mv in mcounts.items():
        target = d_cd if k.startswith("records_") else d_dn
        dv = target.get(k)
        if not is_nonneg_int(dv):
            v.add("COUNT_TYPE_INVALID", "counts.%s=%r 不是严格非负整数" % (k, dv))
        elif dv != mv:
            v.add("COUNT_INVARIANT_BROKEN", "counts.%s 声明 %r != 实测 %r" % (k, dv, mv))
    if (d_dn.get("denylist_per_array") or {}) != dstat.get("per_array"):
        v.add("COUNT_INVARIANT_BROKEN", "counts.denylist_per_array 与实测不符")
    if d_dn.get("source_array_map_is_multi_valued") is not True:
        v.add("COUNT_INVARIANT_BROKEN", "source_array_map_is_multi_valued 未为 true")
    if (decl.get("invariant") or {}).get("raw_equals_effective_plus_duplicate") is not True:
        v.add("COUNT_INVARIANT_BROKEN", "invariant.raw_equals_effective_plus_duplicate 未为 true")

    tl = {norm_token(t) for t in union}
    inter = {norm_token(r.get(f)) for r in tasks if isinstance(r, dict)
             for f in ("problem_family_id", "repo_family")
             if isinstance(r.get(f), str) and norm_token(r.get(f)) in tl}
    detail["intersection_count"] = len(inter)

    # ---------- B4：绑定全部通过后才启动计算
    pre_compute_blocking = {"BINDING_MISMATCH", "BINDING_INCOMPLETE", "PATH_OUTSIDE_ROOT",
                            "COUNT_INVARIANT_BROKEN", "COUNT_TYPE_INVALID",
                            "RECORD_CONTAINER_INVALID", "RECORD_IDENTITY_TYPE_INVALID",
                            "DENYLIST_CONTAINER_INVALID", "INPUT_CONTAINER_INVALID",
                            "INPUT_PARSE_ERROR", "BASELINE_UNVERIFIED",
                            "DENOMINATOR_MISSING", "DENOMINATOR_SET_MISMATCH",
                            "DENOMINATOR_TYPE_INVALID", "DENOMINATOR_COUNT_MISMATCH",
                            "DENOMINATOR_SHORTFALL", "SNAPSHOT_UNVERIFIED"}
    hits = sorted(set(v.codes()) & pre_compute_blocking)
    detail["pre_compute_blocking"] = hits
    if hits or not dn_ok:
        v.add("COMPUTE_NOT_STARTED", "输入/执行绑定未通过，已在启动计算前终止")
        return "COMPUTATION_ONLY" if not hits else "BLOCKED", detail, checks

    # 不可变执行快照：**从已核验的 buffer 原子写出**，不再从可变源路径复制
    work = Path(args.workdir or ((Path(args.out).parent if args.out else Path.cwd()) / "_k27v6_tmp"))
    snapdir = work / ("exec_%d" % os.getpid())
    if snapdir.exists():
        shutil.rmtree(snapdir, ignore_errors=True)
    snapdir.mkdir(parents=True, exist_ok=True)
    exec_sha = {}
    for rel in closure:
        if rel not in closure_buffers:
            v.add("BINDING_INCOMPLETE", "闭包文件 %s 无可核验 buffer，无法构建执行快照" % rel)
            continue
        dst = snapdir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(dst, closure_buffers[rel])
        written = dst.read_bytes()
        # 快照字节必须与已授权、已绑定摘要**逐字节相等**，否则不得启动
        if written != closure_buffers[rel]:
            v.add("EXEC_SNAPSHOT_MISMATCH", "执行快照 %s 字节与已核验 buffer 不等" % rel)
            continue
        exec_sha[rel] = sha256_bytes(written)
        decl = next((e for e in declared_set
                     if isinstance(e, dict) and e.get("path") == rel), None)
        if decl and decl.get("file_bytes_sha256") != exec_sha[rel]:
            v.add("EXEC_SNAPSHOT_MISMATCH", "执行快照 %s 摘要 != manifest 已绑定摘要" % rel)
        if not synthetic and rel in closure:
            key = ("validator_closure", str((vsrc / rel).resolve()))
            if key in granted and granted[key] != exec_sha[rel]:
                v.add("EXEC_SNAPSHOT_MISMATCH", "执行快照 %s 摘要 != 授权记录声明摘要" % rel)
    detail["exec_snapshot_sha256"] = {k: v2[:16] for k, v2 in exec_sha.items()}
    if v.has("EXEC_SNAPSHOT_MISMATCH"):
        v.add("COMPUTE_NOT_STARTED", "执行快照与已核验字节不符，已在启动计算前终止")

    # 输入快照：同样从已核验 buffer 写出，交给消费者的就是通过校验的那份字节
    indir = work / ("inputs_%d" % os.getpid())
    if indir.exists():
        shutil.rmtree(indir, ignore_errors=True)
    indir.mkdir(parents=True, exist_ok=True)
    in_records, in_denylist = indir / "candidate.json", indir / "denylist.json"
    _atomic_write(in_records, rec_raw)
    _atomic_write(in_denylist, den_raw)
    if in_records.read_bytes() != rec_raw or in_denylist.read_bytes() != den_raw:
        v.add("INPUT_BYTES_MISMATCH", "输入快照字节与已核验 buffer 不等")

    if v.has("COMPUTE_NOT_STARTED"):
        return "BLOCKED", detail, checks

    exit_code, rounds = run_consumer(snapdir, tasks, union, args, work,
                                     records_path=str(in_records), denylist_path=str(in_denylist))
    detail["rounds"] = rounds
    detail["validator_exit_code"] = exit_code
    detail["compute_invocations"] = _AUDIT["compute_invocations"]

    comp = (manifest.get("records") or {}).get("computation_result") or {}
    if comp.get("exit_code") != exit_code:
        v.add("VALIDATOR_EXIT_MISMATCH", "manifest exit_code=%r 验收轮实测=%r"
              % (comp.get("exit_code"), exit_code))
    if comp.get("exit_code_source") != "os_process":
        v.add("EXIT_CODE_NOT_OS_PROCESS", "exit_code_source=%r" % comp.get("exit_code_source"))
    diag = rounds.get("diagnostic", {}).get("os_process_exit_code")
    if diag != 7:
        v.add("DENYLIST_SCHEMA_NOT_CONSUMED",
              "诊断轮（必有交集、预期 7）实测 %r ⇒ 消费者未按协议 schema 排除" % diag)
    bad = rounds.get("bad_input", {}).get("os_process_exit_code")
    if bad != 4:
        v.add("BAD_INPUT_NOT_REJECTED", "坏输入轮（预期 4）实测 %r" % bad)
    if inter:
        v.add("INTERSECTION_NONZERO", "交集 %d 项" % len(inter))
    # 所有消费者模式都必须回传消费回执；缺失/null/摘要不符一律阻断
    acc = rounds.get("acceptance") or {}
    if acc.get("receipt") is None:
        v.add("CONSUMPTION_RECEIPT_MISSING",
              "验收轮缺少消费回执（%s 模式未回传输入摘要）⇒ 不得通过" % args.validator_mode)
    elif acc.get("input_bytes_match") is not True:
        v.add("INPUT_BYTES_MISMATCH", "消费者回执的输入摘要 != 原始绑定字节")

    # ---------- 门槛证据（含含义）
    iso = (manifest.get("records") or {}).get("isolation_acceptance") or {}
    author_principal = iso.get("candidate_author_principal")
    gates = {}
    gate_payloads = {}
    for gname in ("G1_namespace_compatibility", "G2_effective_input_binding",
                  "G3_snapshot_content_bound", "G4_isolation_verified"):
        gate = iso.get(gname)
        claimed = isinstance(gate, dict) and gate.get("claimed") is True
        if not claimed:
            v.add("EVIDENCE_SELF_DECLARED", "%s: claimed != true" % gname)
            gates[gname] = False
            continue
        ok, payloads = resolve_and_parse_evidence(gate.get("evidence"), args.evidence_root,
                                                 read_log, v, gname, author_principal,
                                                 trusted_principals, approvals, synthetic)
        gate_payloads[gname] = payloads
        gates[gname] = ok

    # G3 含义：证据正文里的 snapshot_sha256 必须等于实测
    if gates["G3_snapshot_content_bound"] and snap_raw is not None:
        declared_snap = {p.get("snapshot_sha256") for p in gate_payloads.get("G3_snapshot_content_bound", [])}
        if sha256_bytes(snap_raw) not in declared_snap:
            gates["G3_snapshot_content_bound"] = False
            v.add("SNAPSHOT_UNVERIFIED", "G3 证据正文未记录实测 snapshot 摘要")

    # ---------- OS 身份（stat 单列口径）
    nt = iso.get("negative_test") or {}
    for fld, val in (("custody_principal_os_identity", iso.get("custody_principal_os_identity")),
                     ("negative_test.subject_os_identity", nt.get("subject_os_identity")),
                     ("positive_test.subject_os_identity",
                      (iso.get("positive_test") or {}).get("subject_os_identity"))):
        if not valid_os_identity(val):
            v.add("OS_IDENTITY_INVALID", "%s=%r 非法（拒绝 agent UUID/空值）" % (fld, val))

    if gates["G4_isolation_verified"]:
        g4 = []
        if iso.get("isolation_mechanism") not in PERMITTED_MECHANISMS:
            g4.append("isolation_mechanism=%r" % iso.get("isolation_mechanism"))
        if not iso.get("permission_matrix_sha256"):
            g4.append("缺 permission_matrix_sha256")
        if iso.get("training_context_cannot_self_authorize") is not True:
            g4.append("未证明训练主体不能自行授权")
        if iso.get("admin_owner_daemon_paths_excluded") is not True:
            g4.append("未证明管理员/owner/daemon 路径已排除")
        # stat 单列：stat 可见或拒绝都可；open/枚举/自行授权必须拒绝
        if nt.get("open_result") != "denied":
            g4.append("negative_test.open_result != denied")
        if nt.get("enumerate_result") != "denied":
            g4.append("negative_test.enumerate_result != denied")
        if nt.get("self_authorize_result") != "denied":
            g4.append("negative_test.self_authorize_result != denied")
        if nt.get("stat_recorded_separately") is not True:
            g4.append("negative_test.stat 未单列（stat_recorded_separately != true）")
        if nt.get("stat_result") not in ("visible", "denied"):
            g4.append("negative_test.stat_result 未记录")
        if nt.get("exit_code_source") != "os_process" or nt.get("subprocess_checked") is not True:
            g4.append("negative_test 未覆盖子进程或非 os_process")
        if not is_nonneg_int(nt.get("exit_code")) or nt.get("exit_code") == 0:
            g4.append("negative_test.exit_code 必须是实测的非零拒绝码，实际 %r" % nt.get("exit_code"))
        if (iso.get("positive_test") or {}).get("exit_code") != 0:
            g4.append("positive_test 未成功（评测主体必须能读获准 canary）")
        if v.has("OS_IDENTITY_INVALID"):
            g4.append("OS 身份字段非法")
        if g4:
            gates["G4_isolation_verified"] = False
            v.add("G4_MECHANISM_INSUFFICIENT", "; ".join(g4))

    detail["gates"] = gates

    # ---------- 结论
    BLOCKING = {
        "PROTOCOL_SHA_MISMATCH", "MODE_TYPE_INVALID", "SYNTHETIC_NOT_PERMITTED_IN_PRODUCTION",
        "FIXTURE_ROOT_MISSING", "ROOT_NOT_DECLARED", "PATH_OUTSIDE_ROOT",
        "PRODUCTION_FORBIDS_TEST_DOUBLE",
        "AUTH_MISSING", "AUTH_UNTRUSTED", "AUTH_SCOPE_INSUFFICIENT", "AUTH_OBJECTS_MISSING",
        "AUTH_SCHEMA_UNSUPPORTED", "AUTH_IDENTITY_INVALID", "AUTH_ISOLATION_UNVERIFIED",
        "AUTH_BEFORE_ACCESS",
        "EVIDENCE_SELF_DECLARED", "EVIDENCE_NOT_RESOLVABLE", "EVIDENCE_HASH_INVALID",
        "EVIDENCE_HASH_MISMATCH", "EVIDENCE_PRODUCER_MISSING", "EVIDENCE_TIME_INVALID",
        "EVIDENCE_SELF_PRODUCED", "EVIDENCE_MEANING_UNSUPPORTED", "GATE_MEANING_UNSUPPORTED",
        "CANDIDATE_EMPTY", "RECORD_MISSING_FIELD", "RECORD_IDENTITY_TYPE_INVALID",
        "RECORD_CONTAINER_INVALID", "INPUT_CONTAINER_INVALID", "INPUT_PARSE_ERROR",
        "DENYLIST_SCHEMA_UNSUPPORTED", "DENYLIST_UNKNOWN_KEY", "DENYLIST_ITEM_INVALID",
        "DENYLIST_EMPTY", "DENYLIST_CONTAINER_INVALID", "DENYLIST_COUNTS_DECLARED_MISSING",
        "DENYLIST_COUNTS_PARTIAL", "DENYLIST_COUNT_MISMATCH", "COUNT_INVARIANT_BROKEN",
        "COUNT_TYPE_INVALID", "DENOMINATOR_MISSING", "DENOMINATOR_SHORTFALL",
        "DENOMINATOR_SET_MISMATCH", "DENOMINATOR_TYPE_INVALID", "DENOMINATOR_COUNT_MISMATCH",
        "BASELINE_UNVERIFIED", "BINDING_INCOMPLETE", "BINDING_MISMATCH", "BYTES_REBOUND",
        "GIT_BLOB_INVALID", "EXEC_SNAPSHOT_MISMATCH", "CONSUMPTION_RECEIPT_MISSING",
        "EVIDENCE_TRUST_ROOT_MISSING", "EVIDENCE_UNTRUSTED_PRODUCER", "EVIDENCE_APPROVAL_UNTRUSTED",
        "DENYLIST_SCHEMA_NOT_CONSUMED", "BAD_INPUT_NOT_REJECTED", "VALIDATOR_EXIT_MISMATCH",
        "EXIT_CODE_NOT_OS_PROCESS", "INTERSECTION_NONZERO", "INPUT_BYTES_MISMATCH",
        "SNAPSHOT_UNVERIFIED", "OS_IDENTITY_INVALID", "G4_MECHANISM_INSUFFICIENT",
        "OVERALL_SELF_DECLARED",
    }
    blocking = sorted(set(v.codes()) & BLOCKING)
    detail["blocking_reason_codes"] = blocking
    core_ok = (exit_code == 0) and not blocking

    if synthetic:
        verdict = "SYNTHETIC_ACCEPTED" if (core_ok and all(gates.values())) else "COMPUTATION_ONLY"
    else:
        verdict = "ACCEPTED" if (core_ok and all(gates.values())) else "COMPUTATION_ONLY"

    declared_overall = (manifest.get("records") or {}).get("overall")
    if declared_overall not in (None, verdict):
        v.add("OVERALL_SELF_DECLARED", "自述 overall=%r != 机器判定 %r" % (declared_overall, verdict))
    return verdict, detail, checks


# --------------------------------------------------------------- consumers


SYNTHETIC_CONSUMER = r'''
"""显式合成消费者（新 schema）：只用于测试替身，不是 E0 G2 集成验收。"""
import json, sys, hashlib, pathlib
reg_p, den_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]
rb, db = pathlib.Path(reg_p).read_bytes(), pathlib.Path(den_p).read_bytes()
sys.stderr.write(json.dumps({"synthetic_consumer": True,
                             "registry_sha256": hashlib.sha256(rb).hexdigest(),
                             "denylist_sha256": hashlib.sha256(db).hexdigest()}) + "\n")
reg, den = json.loads(rb.decode("utf-8")), json.loads(db.decode("utf-8"))
if den.get("denylist_schema_version") != "v2-dh-exclusion-denylist-1":
    raise SystemExit(4)
arrays = ("d_family_ids", "h_family_ids", "exposed_family_ids")
for k in arrays:
    for it in den.get(k) or []:
        if not isinstance(it, str) or not it.strip():
            raise SystemExit(4)
tasks = reg.get("tasks")
if not isinstance(tasks, list) or not tasks:
    raise SystemExit(4)
deny = {str(x).strip().lower() for k in arrays for x in (den.get(k) or [])}
for t in tasks:
    for f in ("problem_family_id", "repo_family"):
        v = t.get(f)
        if not isinstance(v, str) or not v.strip():
            raise SystemExit(4)
        if v.strip().lower() in deny:
            pathlib.Path(out_p).write_text(json.dumps({"intersection": 1}))
            raise SystemExit(7)
pathlib.Path(out_p).write_text(json.dumps({"intersection": 0}))
raise SystemExit(0)
'''

TEST_DOUBLE = r'''
import json, sys, pathlib
payload = json.load(sys.stdin)
deny = {str(x).strip().lower() for x in payload.get("denylist", []) if str(x).strip()}
for rec in payload.get("records", []):
    for f in ("problem_family_id", "repo_family"):
        v = rec.get(f)
        if not isinstance(v, str) or not v.strip():
            sys.stderr.write("[split_leak_check_missing_family]\n"); raise SystemExit(4)
for rec in payload.get("records", []):
    for f in ("problem_family_id", "repo_family"):
        if str(rec.get(f)).strip().lower() in deny:
            sys.stderr.write("[v2_denylist_intersection]\n"); raise SystemExit(7)
raise SystemExit(0)
'''


def run_consumer(execroot, tasks, union, args, work, records_path=None, denylist_path=None):
    """诊断轮 / 验收轮 / 坏输入轮，分开记录。

    所有轮次都消费**已核验 buffer 写出的输入快照**（records_path / denylist_path）。
    """
    rounds = {}
    tmp = work / ("rounds_%d" % os.getpid())
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    acc_records = records_path or args.records
    acc_denylist = denylist_path or args.denylist

    def spawn(tag, reg_path, den_path):
        _AUDIT["compute_invocations"] += 1
        out = tmp / ("out_%s.json" % tag)
        if args.validator_mode == "real_cli":
            cmd = [sys.executable, "-m", "v3.cli", "split", "--registry", str(reg_path),
                   "--denylist", str(den_path), "--out", str(out)]
            env = dict(os.environ, PYTHONPATH=str(execroot))
            p = subprocess.run(cmd, cwd=str(execroot), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, env=env)
        elif args.validator_mode == "synthetic_consumer":
            script = tmp / "syn_consumer.py"
            script.write_text(SYNTHETIC_CONSUMER, encoding="utf-8")
            p = subprocess.run([sys.executable, str(script), str(reg_path), str(den_path), str(out)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            payload = json.dumps({"records": tasks, "denylist": list(union)}, ensure_ascii=False)
            p = subprocess.run([sys.executable, "-c", TEST_DOUBLE], input=payload.encode(),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stderr = p.stderr.decode("utf-8", "replace").strip()
        return {"os_process_exit_code": p.returncode, "stderr_tail": stderr[-300:],
                "input_bytes_match": None, "receipt": None}

    # 验收轮：消费已核验 buffer 写出的输入快照
    acc = spawn("acceptance", acc_records, acc_denylist)
    rounds["acceptance"] = acc

    # 诊断轮：注入一个必与 denylist 相交的家族（同样写成不可变快照）
    if union:
        diag = json.loads(json.dumps({"tasks": list(tasks)}))
        diag["tasks"] = list(diag["tasks"]) + [{
            "task_id": "diag-injected", "problem_statement": "diag injected",
            "problem_family_id": union[0], "repo_family": "repo-diag", "split": "train"}]
        dr = tmp / "diag_registry.json"
        _atomic_write(dr, (json.dumps(diag, ensure_ascii=False) + "\n").encode())
        rounds["diagnostic"] = spawn("diagnostic", dr, acc_denylist)

    # 坏输入轮：身份非法
    br = tmp / "bad_registry.json"
    _atomic_write(br, (json.dumps({"tasks": [{"task_id": "bad", "problem_statement": "x",
                                             "problem_family_id": 123,
                                             "repo_family": "repo-bad"}]},
                                  ensure_ascii=False) + "\n").encode())
    rounds["bad_input"] = spawn("bad_input", br, acc_denylist)

    # 回执核对：所有模式都必须回传与预运行摘要一致的消费回执
    receipt = None
    for line in reversed((rounds["acceptance"].get("stderr_tail") or "").splitlines()):
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and obj.get("synthetic_consumer"):
                receipt = obj
                break
        except Exception:  # noqa: BLE001
            continue
    if receipt:
        receipt["original_registry_sha256"] = sha256_bytes(Path(acc_records).read_bytes())
        receipt["original_denylist_sha256"] = sha256_bytes(Path(acc_denylist).read_bytes())
        rounds["acceptance"]["receipt"] = receipt
        rounds["acceptance"]["input_bytes_match"] = (
            receipt.get("registry_sha256") == receipt["original_registry_sha256"]
            and receipt.get("denylist_sha256") == receipt["original_denylist_sha256"])

    shutil.rmtree(tmp, ignore_errors=True)
    return rounds["acceptance"]["os_process_exit_code"], rounds


# ---------------------------------------------------------------------- main


def build_parser():
    ap = argparse.ArgumentParser(description="KAGGLE-27 exclusion-proof manifest checker (v5)")
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--mode", choices=["production", "test"], default="production")
    ap.add_argument("--expect", default=None)
    ap.add_argument("--allow-synthetic", action="store_true")
    ap.add_argument("--fixture-root")
    ap.add_argument("--authorized-root")
    ap.add_argument("--records")
    ap.add_argument("--denylist")
    ap.add_argument("--snapshot")
    ap.add_argument("--validator-src")
    ap.add_argument("--validator-mode",
                    choices=["real_cli", "synthetic_consumer", "test_double"], default="real_cli")
    ap.add_argument("--evidence-root")
    ap.add_argument("--authorization")
    ap.add_argument("--trust-anchor-file")
    ap.add_argument("--trusted-principals")
    ap.add_argument("--approvals")
    ap.add_argument("--baseline")
    ap.add_argument("--workdir")
    ap.add_argument("--out")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.mode == "test" and args.expect is None:
        print("--mode test 必须显式提供 --expect")
        return EXIT_CANNOT
    if args.mode == "production" and args.expect is not None:
        print("--mode production 不接受 --expect")
        return EXIT_CANNOT

    _install_audit_hook()
    read_log = ReadLog()
    v = Violations()
    try:
        verdict, detail, checks = evaluate(args, read_log, v)
    except Exception as exc:  # noqa: BLE001
        import traceback
        verdict, detail, checks = "BLOCKED", {"exception": traceback.format_exc()[-800:]}, {}
        v.add("CHECKER_EXCEPTION", "%s: %s" % (type(exc).__name__, exc))

    report = {
        "checker": "verify_manifest.py", "checker_version": "v5",
        "mode": args.mode, "validator_mode": args.validator_mode,
        "computed_verdict": verdict, "expected_verdict": args.expect,
        "violations": v.items, "reason_codes": sorted(set(v.codes())),
        "read_log": read_log.entries,
        "read_log_protected_requests": len(read_log.protected_entries()),
        "protected_actual_opens": sorted(set(_AUDIT["protected_opens"])),
        "protected_actual_open_count": len(_AUDIT["protected_opens"]),
        "all_open_events": _AUDIT["all_opens"],
        "all_open_event_count": len(_AUDIT["all_opens"]),
        "compute_invocations": _AUDIT["compute_invocations"],
        "checks": checks, "detail": detail,
        "exit_semantics": {
            "production": "exit 0 当且仅当机器判定为真实 ACCEPTED",
            "test": "exit 0 当且仅当机器判定等于调用方显式给出的 --expect",
        },
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_bytes(
            (json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n").encode())

    rc = (EXIT_OK if verdict == "ACCEPTED" else EXIT_MISMATCH) if args.mode == "production" \
        else (EXIT_OK if verdict == args.expect else EXIT_MISMATCH)
    print("mode=%s vm=%s computed=%s expect=%r exit=%d violations=%d codes=%s opens=%d compute=%d"
          % (args.mode, args.validator_mode, verdict, args.expect, rc, len(v),
             ",".join(sorted(set(v.codes()))) or "-",
             len(_AUDIT["protected_opens"]), _AUDIT["compute_invocations"]))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
