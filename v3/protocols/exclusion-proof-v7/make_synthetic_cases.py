#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KAGGLE-27 v7 · 生成合成正反例 fixture 与用例规格。

覆盖 Q0 v4 复核（报告 01a1125c-46f4-762d-87e6-f35d6646544b）的
四组阻断 B1-B4 与两项应修 Y1-Y2，并保留 v4 已通过的反例。

产物：synthetic/<case>/... · synthetic/evidence/... · synthetic/e0_mut_cli/ · fixtures.json
全部合成 fam-syn-* 标签；零真实 H 身份、零受限正文。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SYN = HERE / "synthetic"
LEGACY_E0_COMMIT = "ffc37b4f3ecde6579117b0012ab4f69b4cff16ab"
E0_FIXED_COMMIT = "128d9b98b05ddf128c2e77b599e078de65675b8a"
PROTOCOL = HERE / "exclusion-proof-protocol-v7.json"
BASELINE = HERE / "frozen-denominator-baseline.json"
E0 = None

REQUIRED_ARRAYS = ["d_family_ids", "h_family_ids", "exposed_family_ids"]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def write(path, obj=None, raw=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if raw is not None:
        b = raw if isinstance(raw, bytes) else raw.encode()
    elif isinstance(obj, (bytes, bytearray)):
        b = bytes(obj)
    else:
        b = (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode()
    path.write_bytes(b)
    return b


def nl(b):
    crlf = b.count(b"\r\n"); lf = b.count(b"\n") - crlf
    return "mixed" if (crlf and lf) else ("crlf" if crlf else "lf")


def measured(p: Path):
    b = p.read_bytes()
    return {"file_bytes_sha256": sha(b), "byte_length": len(b), "newline_mode": nl(b)}


# ------------------------------------------------------------------ base data
GOOD_TASKS = [
    {"task_id": "syn-1", "problem_statement": "one", "problem_family_id": "fam-syn-alpha",
     "repo_family": "repo-syn-a", "split": "train"},
    {"task_id": "syn-2", "problem_statement": "two", "problem_family_id": "fam-syn-beta",
     "repo_family": "repo-syn-b", "split": "dev"},
    {"task_id": "syn-3", "problem_statement": "three", "problem_family_id": "FAM-SYN-GAMMA",
     "repo_family": "repo-syn-c", "split": "sealed"},
    {"task_id": "syn-4", "problem_statement": "four", "problem_family_id": "fam-syn-delta",
     "repo_family": "repo-syn-d", "split": "unknown"},
]
D_FAMS = ["fam-syn-d%02d" % i for i in range(1, 9)]      # 8
H_FAMS = ["fam-syn-h%02d" % i for i in range(1, 25)]     # 24
E_FAMS = ["fam-syn-e%02d" % i for i in range(1, 18)]     # 17
GOOD_DENYLIST = {
    "denylist_schema_version": "v2-dh-exclusion-denylist-1",
    "d_family_ids": D_FAMS, "h_family_ids": H_FAMS, "exposed_family_ids": E_FAMS,
    "counts_declared": {"d_family_ids": 8, "h_family_ids": 24, "exposed_family_ids": 17},
    "produced_by": "KAGGLE-27 v7 synthetic fixture generator",
    "produced_at": "2026-10-07T02:00:00Z",
}
INTERSECTING_TASKS = GOOD_TASKS[:1] + [
    {"task_id": "syn-9", "problem_statement": "nine", "problem_family_id": "FAM-SYN-H01",
     "repo_family": "repo-syn-h", "split": "sealed"}]


def dn_stats(dn):
    arrays_of, per = {}, {}
    raw = within = 0
    for name in REQUIRED_ARRAYS:
        arr = dn.get(name) or []
        raw += len(arr)
        seen, w = set(), 0
        for it in arr:
            t = str(it).strip().lower()
            if t in seen:
                w += 1
            else:
                seen.add(t)
                arrays_of.setdefault(t, []).append(name)
        within += w
        per[name] = {"raw_count": len(arr), "effective_count": len(seen), "duplicate_count": w}
    eff = len(arrays_of)
    cross = sum(len(a) - 1 for a in arrays_of.values() if len(a) > 1)
    return {"per_array": per, "raw_total": raw, "effective_total": eff,
            "duplicate_total": within + cross, "cross_array_duplicate_total": cross}


def closure_paths(e0: Path):
    """与验收器同源的完整 CLI 导入闭包（含相对导入）。"""
    sys.path.insert(0, str(HERE))
    from verify_manifest import compute_import_closure
    return compute_import_closure(e0)


def file_set(e0: Path, paths):
    out = []
    for rel in paths:
        b = (e0 / rel).read_bytes()
        out.append({"path": rel, "file_bytes_sha256": sha(b), "byte_length": len(b),
                    "newline_mode": nl(b), "git_commit": None,
                    "git_blob_sha1": None, "git_source": "not_git",
                    "git_checkout_newline_mapping": "not_git"})
    return out


# ------------------------------------------------------------------- evidence
def ev_files(root: Path):
    """四个门槛各自的含义完整证据文件。"""
    write(root / "g1_namespace.json", {
        "namespace_v3": "problem_family_id/repo_family", "namespace_v2": "v2 family id",
        "derivation_rule": "strip+lower 比对 token", "extractor_id": "syn-extractor-1",
        "sample_count": 12})
    write(root / "g2_binding.json", {
        "bound_objects": ["candidate", "denylist", "validator_closure"],
        "approved_by": "syn-independent-reviewer"})
    write(root / "g4_isolation.json", {
        "mechanism": "separate_os_account_with_broken_acl_inheritance",
        "evaluator_subject_os_identity": "S-1-5-21-0-0-1-1003",
        "training_context_cannot_self_authorize": True,
        "admin_owner_daemon_paths_excluded": True,
        "negative_test": {"open_result": "denied"},
        "positive_test": {"exit_code": 0}})
    write(root / "plain_note.json", {"note": "普通无关文本，不是任何门槛的证据"})


def ev_entry(name, digest, kind="artifact", producer="syn-independent-reviewer",
             produced_at="2026-10-07T02:00:00Z"):
    return {"kind": kind, "path": name, "sha256": digest, "produced_by": producer,
            "produced_at": produced_at}


def good_evidence(root: Path, snapshot_sha):
    write(root / "g3_snapshot.json", {"snapshot_sha256": snapshot_sha})
    out = {}
    for g, names in (("G1_namespace_compatibility", ["g1_namespace.json"]),
                     ("G2_effective_input_binding", ["g2_binding.json"]),
                     ("G3_snapshot_content_bound", ["g3_snapshot.json"]),
                     ("G4_isolation_verified", ["g4_isolation.json"])):
        out[g] = [ev_entry(n, sha((root / n).read_bytes())) for n in names]
    return out


# ------------------------------------------------------------------ manifest
def base_manifest(case_dir: Path, records_p, denylist_p, snapshot_p, e0, closure,
                  ev, *, synt=True, overall=None, declared_exit=0, counts=None,
                  denom_measured=None, denom_compared=None, closure_paths_used=None,
                  file_set_from=None):
    tasks = json.loads(records_p.read_text(encoding="utf-8"))["tasks"]
    dn = json.loads(denylist_p.read_text(encoding="utf-8"))
    st = dn_stats(dn)
    snap_sha = sha(snapshot_p.read_bytes())
    counts = counts or {}
    m = {
        "schema_version": "v3-v2-exclusion-proof-execution-manifest-6",
        "protocol_ref": {"schema_version": "v3-v2-exclusion-proof-protocol-7",
                         "protocol_sha256": sha(PROTOCOL.read_bytes()),
                         "protocol_newline_mode": "lf"},
        "execution_mode": {
            "synthetic_mode": bool(synt),
            "authorization_ref": {"present": True, "authorization_id": "syn-auth-0001",
                                  "file_bytes_sha256": None,
                                  "trust_anchor_source": "caller_supplied_file"},
        },
        "binding": {
            "candidate": measured(records_p), "denylist": measured(denylist_p),
            "validator": {"cli_entrypoint": "python -m v3.cli split",
                          "source_commit": E0_FIXED_COMMIT,
                          "file_set": file_set(file_set_from or e0, closure_paths_used or closure)},
            "snapshot": {"content_status": "obtained", "file_bytes_sha256": snap_sha,
                         "byte_length": len(snapshot_p.read_bytes())},
        },
        "counts": {
            "candidate": {"records_raw_count": len(tasks), "records_effective_count": len(tasks),
                          "records_rejected_count": 0},
            "denylist": {"denylist_raw_count": st["raw_total"],
                         "denylist_effective_count": st["effective_total"],
                         "denylist_rejected_count": 0,
                         "denylist_duplicate_count": st["duplicate_total"],
                         "denylist_cross_array_duplicate_count": st["cross_array_duplicate_total"],
                         "denylist_per_array": st["per_array"],
                         "source_array_map_is_multi_valued": True},
            "result": {"intersection_count": 0},
            "invariant": {"raw_equals_effective_plus_duplicate": True},
        },
        "denominator_check": {
            "baseline_artifact_id": "v2-dh-custody-frozen-baseline-1",
            "baseline_file_bytes_sha256": sha(BASELINE.read_bytes()),
            "compared_denominators": denom_compared or ["v2_d8_count", "v2_h24_count",
                                                        "v2_exposed_ids_in_union", "v3_candidate_count"],
            "measured": denom_measured or {"v2_d8_count": 8, "v2_h24_count": 24,
                                           "v2_exposed_ids_in_union": 17, "v3_candidate_count": len(tasks)},
        },
        "records": {
            "computation_result": {"label": "SYNTHETIC_ONLY", "synthetic_mode": bool(synt),
                                   "exit_code": declared_exit, "exit_code_source": "os_process",
                                   "entrypoint": "python -m v3.cli split",
                                   "raised_error_code": None,
                                   "emitted_at": "2026-10-07T02:00:00Z"},
            "isolation_acceptance": {
                "status": "NOT_ACCEPTED",
                "candidate_author_principal": "syn-candidate-author",
                "G1_namespace_compatibility": {"claimed": True, "evidence": ev["G1_namespace_compatibility"]},
                "G2_effective_input_binding": {"claimed": True, "evidence": ev["G2_effective_input_binding"]},
                "G3_snapshot_content_bound": {"claimed": True, "evidence": ev["G3_snapshot_content_bound"]},
                "G4_isolation_verified": {"claimed": True, "evidence": ev["G4_isolation_verified"]},
                "isolation_mechanism": "separate_os_account_with_broken_acl_inheritance",
                "permission_matrix_sha256": sha(b"syn-matrix"),
                "training_context_cannot_self_authorize": True,
                "admin_owner_daemon_paths_excluded": True,
                "negative_test": {"exit_code": 1, "exit_code_source": "os_process",
                                  "stat_result": "visible", "stat_recorded_separately": True,
                                  "open_result": "denied", "enumerate_result": "denied",
                                  "self_authorize_result": "denied",
                                  "subject_os_identity": "S-1-5-21-0-0-1-1002",
                                  "subprocess_checked": True},
                "positive_test": {"exit_code": 0, "subject_os_identity": "S-1-5-21-0-0-1-1003"},
                "custody_principal_os_identity": "S-1-5-21-0-0-1-1004",
            },
            "overall": overall,
        },
        "fill_rules": ["机器结论写入独立输出件"],
    }
    for dotted, val in counts.items():
        node = m["counts"]
        ks = dotted.split(".")
        for k in ks[:-1]:
            node = node.setdefault(k, {})
        node[ks[-1]] = val
    return m


CASES = []


def emit(cid, manifest, *, records, denylist, snapshot, tmp_root, args_extra=(),
         mode="test", expect="SYNTHETIC_ACCEPTED", targets=(), note="",
         auth_objects=None, trust_anchor_ok=True, need_auth=True, validator_src=None,
         auth_patch=None):
    d = SYN / cid
    d.mkdir(parents=True, exist_ok=True)
    write(d / "manifest.json", manifest)
    for name, obj in (("records.json", records), ("denylist.json", denylist),
                      ("snapshot.json", snapshot)):
        write(d / name, obj)
    if need_auth:
        objs = auth_objects if auth_objects is not None else []
        auth = {"authorization_schema_version": "v3-exclusion-proof-read-authorization-2",
                "authorization_id": "syn-auth-0001",
                "issued_by": {"principal": "syn-independent-approver",
                              "role": "designated_evaluator",
                              "os_identity": "S-1-5-21-0-0-1-1005"},
                "approved_at": "2026-10-07T02:00:00Z",
                "scope": {"read_permitted": True, "objects": objs},
                "isolation": {"verified": True,
                              "mechanism": "separate_os_account_with_broken_acl_inheritance",
                              "exit_code_source": "os_process",
                              "training_context_cannot_self_authorize": True,
                              "admin_owner_daemon_paths_excluded": True},
                "test_double_note": "显式合成授权替身，不是真实批准记录"}
        if auth_patch:
            auth_patch(auth)
        ab = write(d / "authorization.json", auth)
        write(d / "trust_anchor.bin", raw=ab if trust_anchor_ok else b"forged-anchor")
    CASES.append({"id": cid, "note": note, "mode": mode, "expect": expect,
                  "targets": list(targets), "args": args_extra,
                  "dir": "synthetic/%s" % cid,
                  "validator_src": validator_src or "e0",
                  "need_auth": need_auth, "trust_anchor_ok": trust_anchor_ok})


def auth_objects_for(records_p, denylist_p, snapshot_p, ev_dir, ev_names, e0, closure):
    objs = []
    for kind, p in (("candidate", records_p), ("denylist", denylist_p), ("snapshot", snapshot_p)):
        objs.append({"kind": kind, "path": str(p.resolve()),
                     "sha256": sha(p.read_bytes()), "purpose": "synthetic judgement test"})
    for n in ev_names:
        p = ev_dir / n
        objs.append({"kind": "evidence", "path": str(p.resolve()),
                     "sha256": sha(p.read_bytes()), "purpose": "gate evidence"})
    for rel in closure:
        p = e0 / rel
        objs.append({"kind": "validator_closure", "path": str(p.resolve()),
                     "sha256": sha(p.read_bytes()), "purpose": "execution closure"})
    return objs


def main():
    global E0
    ap = argparse.ArgumentParser()
    ap.add_argument("--e0src", default=None)
    a = ap.parse_args()
    cands = ([Path(a.e0src)] if a.e0src else []) + [
        HERE.parent.parent / "e0", HERE.parent / "e0", Path.cwd() / "e0"]
    for c in cands:
        if c and (c / "v3" / "data" / "dedup.py").exists():
            E0 = c.resolve(); break
    if E0 is None:
        raise SystemExit("需要 E0 只读克隆")

    if SYN.exists():
        shutil.rmtree(SYN)
    SYN.mkdir(parents=True)
    closure = closure_paths(E0)
    # Version-pinned legacy negative control from Git object storage, never HEAD.
    archive = subprocess.run(["git", "-C", str(E0), "archive", "--format=tar",
                              LEGACY_E0_COMMIT, "v3"], check=True, stdout=subprocess.PIPE).stdout
    legacy_root = SYN / "e0_legacy"
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:") as tf:
        for member in tf.getmembers():
            rel = Path(member.name)
            if rel.is_absolute() or ".." in rel.parts or not member.isfile():
                continue
            target = legacy_root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            source = tf.extractfile(member)
            if source is not None:
                target.write_bytes(source.read())
    evd = SYN / "evidence"
    ev_files(evd)
    snapshot_b = b"syn-snapshot-content"
    (SYN / "snapshot_content.bin").write_bytes(snapshot_b)
    snap_sha = sha(snapshot_b)
    EV = good_evidence(evd, snap_sha)
    ev_names = ["g1_namespace.json", "g2_binding.json", "g3_snapshot.json", "g4_isolation.json"]

    # 变异 CLI 副本（只改 cli.py，用于"CLI 语义变化无 BINDING 告警"反例）
    mut = SYN / "e0_mut_cli"
    for rel in closure:
        source = E0 / rel
        if source.is_file():
            target = mut / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    clip = mut / "v3" / "cli.py"
    t = clip.read_text(encoding="utf-8")
    before_mutation_sha = sha(clip.read_bytes())
    clip.write_text(t + "\n# explicit v7 mutation marker: execution identity regression\n",
                    encoding="utf-8", newline="\n")
    if sha(clip.read_bytes()) == before_mutation_sha:
        raise RuntimeError("CLI mutation did not change bytes")
    write(SYN / "e0_mutation_record.json", {
        "cli_sha256_before": sha((E0 / "v3" / "cli.py").read_bytes()),
        "cli_sha256_after": sha(clip.read_bytes()),
        "mutation_changed_bytes": sha((E0 / "v3" / "cli.py").read_bytes()) != sha(clip.read_bytes()),
        "dedup_sha256_unchanged": sha((E0 / "v3" / "data" / "dedup.py").read_bytes())})

    # ---------------- 正例：合成消费者验收轮
    cid = "pos_synthetic_consumer_acceptance"
    d = SYN / cid
    rp, dp, sp = d / "records.json", d / "denylist.json", d / "snapshot.json"
    write(rp, {"tasks": GOOD_TASKS}); write(dp, GOOD_DENYLIST); write(sp, raw=snapshot_b)
    m = base_manifest(d, rp, dp, sp, E0, closure, EV, overall="SYNTHETIC_ACCEPTED")
    emit(cid, m, records={"tasks": GOOD_TASKS}, denylist=GOOD_DENYLIST, snapshot=snapshot_b,
         tmp_root=SYN, expect="SYNTHETIC_ACCEPTED", note="合成消费者：验收轮 0 / 诊断轮 7 / 坏输入轮 4",
         args_extra=["--allow-synthetic", "--fixture-root", str(SYN),
                     "--validator-mode", "synthetic_consumer"],
         auth_objects=auth_objects_for(rp, dp, sp, evd, ev_names, E0, closure))

    def mk(cid, note, targets, *, tasks=GOOD_TASKS, dn=None, mutate=None, synt=True,
           mode="test", declared_exit=0, e0dir=None, closure_used=None, ev=EV,
           counts=None, denom_measured=None, denom_compared=None, auth_objs=None,
           trust_anchor_ok=True, validator_src=None, expect="SYNTHETIC_ACCEPTED",
           extra=None, auth_drop_kinds=(), auth_patch=None):
        d = SYN / cid
        rp, dp, sp = d / "records.json", d / "denylist.json", d / "snapshot.json"
        write(rp, {"tasks": tasks}); write(dp, dn or GOOD_DENYLIST); write(sp, raw=snapshot_b)
        use_e0 = e0dir or E0
        cl = closure_used if closure_used is not None else closure_paths(use_e0)
        m = base_manifest(d, rp, dp, sp, use_e0, closure, ev, synt=synt,
                          declared_exit=declared_exit, counts=counts,
                          denom_measured=denom_measured, denom_compared=denom_compared,
                          closure_paths_used=cl, file_set_from=use_e0)
        if mutate:
            mutate(m)
        ao = auth_objs if auth_objs is not None else auth_objects_for(
            rp, dp, sp, evd, ev_names, use_e0, closure_paths(use_e0))
        if auth_drop_kinds:
            ao = [o for o in ao if o["kind"] not in auth_drop_kinds]
        args = list(extra or [])
        if mode == "test":
            args += ["--allow-synthetic", "--fixture-root", str(SYN)]
        emit(cid, m, records={"tasks": tasks}, denylist=(dn or GOOD_DENYLIST), snapshot=sp.read_bytes(),
             tmp_root=SYN, mode=mode, expect=(expect if mode == "test" else None),
             targets=targets, note=note, auth_objects=ao, trust_anchor_ok=trust_anchor_ok,
             validator_src=validator_src, auth_patch=auth_patch)
        CASES[-1]["args"] = args

    # ============ B1 ============
    plain = {g: [ev_entry("plain_note.json", sha((evd / "plain_note.json").read_bytes()))]
             for g in EV}
    mk("neg_b1_plain_evidence_all_gates",
       "同一无关普通文本冒充四门槛（合法摘要、伪 producer、伪 kind）",
       ["EVIDENCE_MEANING_UNSUPPORTED"], ev=plain)
    CASES[-1]["need_auth"] = False
    selfp = {g: [ev_entry(n, sha((evd / n).read_bytes()), producer="syn-candidate-author")
                 for n in ns] for g, ns in
             (("G1_namespace_compatibility", ["g1_namespace.json"]),
              ("G2_effective_input_binding", ["g2_binding.json"]),
              ("G3_snapshot_content_bound", ["g3_snapshot.json"]),
              ("G4_isolation_verified", ["g4_isolation.json"]))}
    mk("neg_b1_evidence_self_produced", "证据由候选作者本人产出 ⇒ 不构成独立证据",
       ["EVIDENCE_SELF_PRODUCED"], ev=selfp)
    badtime = {g: [ev_entry(n, sha((evd / n).read_bytes()), produced_at="not-a-time")
                   for n in ns] for g, ns in
               (("G1_namespace_compatibility", ["g1_namespace.json"]),
                ("G2_effective_input_binding", ["g2_binding.json"]),
                ("G3_snapshot_content_bound", ["g3_snapshot.json"]),
                ("G4_isolation_verified", ["g4_isolation.json"]))}
    mk("neg_b1_evidence_bad_time", "produced_at 非法时间被接受（v4 缺陷）",
       ["EVIDENCE_TIME_INVALID"], ev=badtime)
    appr = copy.deepcopy(EV)
    appr["G1_namespace_compatibility"] = [ev_entry("g1_namespace.json",
                                                   sha((evd / "g1_namespace.json").read_bytes()),
                                                   kind="trusted_approval")]
    mk("neg_b1_trusted_approval_not_approval",
       "kind=trusted_approval 但正文不是批准记录",
       ["EVIDENCE_MEANING_UNSUPPORTED"], ev=appr)
    mk("neg_b1_production_test_double",
       "生产模式使用测试替身（v4 缺陷：可得到 ACCEPTED/exit 0）",
       ["PRODUCTION_FORBIDS_TEST_DOUBLE"], synt=False, mode="production",
       extra=["--validator-mode", "test_double"])
    CASES[-1]["args"] = ["--validator-mode", "test_double"]
    mk("neg_b1_g4_negative_exit_zero", "negative_test.exit_code=0 ⇒ 未发生拒绝，G4 必须失败",
       ["G4_MECHANISM_INSUFFICIENT"],
       mutate=lambda m: m["records"]["isolation_acceptance"]["negative_test"].update({"exit_code": 0}))

    # ============ B2 ============
    mk("neg_b2_scope_omits_snapshot", "授权对象清单漏 snapshot ⇒ 拒绝前零打开",
       ["AUTH_SCOPE_INSUFFICIENT", "AUTH_BEFORE_ACCESS"],
       synt=False, mode="production", extra=[], auth_drop_kinds=("snapshot",))
    CASES[-1]["args"] = []
    mk("neg_b2_production_manifest_selects_synthetic",
       "生产模式下被检清单自填 synthetic_mode=true（v4 缺陷：可绕过授权读取并启动替身）",
       ["SYNTHETIC_NOT_PERMITTED_IN_PRODUCTION", "AUTH_MISSING", "AUTH_BEFORE_ACCESS"],
       synt=True, mode="production", extra=[])
    CASES[-1]["args"] = []
    CASES[-1]["need_auth"] = False
    mk("neg_b2_relative_path_escape", "file_set 用 ../ 越界读取 root 外文件",
       ["PATH_OUTSIDE_ROOT"],
       mutate=lambda m: m["binding"]["validator"]["file_set"].__setitem__(
           0, {"path": "../../../e0/v3/data/dedup.py", "file_bytes_sha256": "a" * 64,
               "byte_length": 1, "newline_mode": "lf", "git_blob_sha1": "b" * 40}))
    mk("neg_b2_absolute_path_escape", "file_set 用绝对路径越界",
       ["PATH_OUTSIDE_ROOT"],
       mutate=lambda m: m["binding"]["validator"]["file_set"].__setitem__(
           0, {"path": str((E0 / "v3" / "data" / "dedup.py").resolve()),
               "file_bytes_sha256": "a" * 64, "byte_length": 1, "newline_mode": "lf",
               "git_blob_sha1": "b" * 40}))
    mk("neg_b2_evidence_root_escape", "证据路径 ../ 逃出 evidence root",
       ["EVIDENCE_NOT_RESOLVABLE"],
       ev={g: [ev_entry("../plain_note.json", sha((evd / "plain_note.json").read_bytes()))]
           for g in EV})
    CASES[-1]["need_auth"] = False
    mk("neg_b2_auth_wrong_isolation", "授权记录 isolation.verified=false",
       ["AUTH_ISOLATION_UNVERIFIED", "AUTH_BEFORE_ACCESS"], synt=False, mode="production",
       extra=[], auth_patch=lambda a: a["isolation"].update({"verified": False}))
    CASES[-1]["args"] = []
    mk("neg_b2_binding_failure_no_compute",
       "绑定摘要失败 ⇒ 必须在启动计算前终止（compute_invocations 必须为 0）",
       ["COMPUTE_NOT_STARTED", "BINDING_MISMATCH"],
       mutate=lambda m: m["binding"]["denylist"].update({"file_bytes_sha256": "f" * 64}))
    mk("neg_b2_grant_sha_mismatch",
       "候选授权 SHA 为全零、但同路径 manifest/实读字节为另一摘要 ⇒ 消费者不得启动",
       ["AUTH_OBJECT_HASH_MISMATCH", "COMPUTE_NOT_STARTED"])
    auth_path = SYN / "neg_b2_grant_sha_mismatch" / "authorization.json"
    auth_doc = json.loads(auth_path.read_text(encoding="utf-8"))
    next(o for o in auth_doc["scope"]["objects"] if o["kind"] == "candidate")["sha256"] = "0" * 64
    auth_bytes = write(auth_path, auth_doc)
    write(SYN / "neg_b2_grant_sha_mismatch" / "trust_anchor.bin", raw=auth_bytes)
    mk("neg_b2_unauthorized_dependency_zero_open",
       "授权闭包遗漏 faces.py ⇒ 目标 scope reason、该未授权路径零 open、consumer 零启动",
       ["AUTH_SCOPE_INSUFFICIENT", "COMPUTE_NOT_STARTED"])
    auth_path = SYN / "neg_b2_unauthorized_dependency_zero_open" / "authorization.json"
    auth_doc = json.loads(auth_path.read_text(encoding="utf-8"))
    auth_doc["scope"]["objects"] = [o for o in auth_doc["scope"]["objects"]
                                     if not (o["kind"] == "validator_closure"
                                             and o["path"].replace("\\", "/").endswith("/v3/data/faces.py"))]
    auth_bytes = write(auth_path, auth_doc)
    write(SYN / "neg_b2_unauthorized_dependency_zero_open" / "trust_anchor.bin", raw=auth_bytes)

    # ============ B3 ============
    for tag, val in (("string", "0"), ("bool", True), ("float", 0.0), ("list", []), ("dict", {})):
        mk("neg_b3_denominator_%s" % tag, "分母声明值类型非法（%s）" % tag,
           ["DENOMINATOR_TYPE_INVALID"],
           denom_measured={"v2_d8_count": val, "v2_h24_count": 24,
                           "v2_exposed_ids_in_union": 17, "v3_candidate_count": 4})
    mk("neg_b3_choose_unrelated_denominator", "自行只比较无关分母项 ⇒ 必比项由基线冻结",
       ["DENOMINATOR_SET_MISMATCH"], denom_compared=["v3_candidate_count"],
       denom_measured={"v3_candidate_count": 4})
    short_dn = dict(GOOD_DENYLIST)
    short_dn["h_family_ids"] = H_FAMS[:1]
    short_dn["counts_declared"] = {"d_family_ids": 8, "h_family_ids": 1, "exposed_family_ids": 17}
    st = dn_stats(short_dn)
    mk("neg_b3_shrink_input_sync_counts", "输入与 counts 同步缩小 ⇒ 必须被冻结基线挡住",
       ["DENOMINATOR_SHORTFALL"],
       dn=short_dn,
       denom_measured={"v2_d8_count": 8, "v2_h24_count": 24,
                       "v2_exposed_ids_in_union": 17, "v3_candidate_count": 4},
       counts={"denylist.denylist_raw_count": st["raw_total"],
               "denylist.denylist_effective_count": st["effective_total"],
               "denylist.denylist_duplicate_count": st["duplicate_total"],
               "denylist.denylist_cross_array_duplicate_count": st["cross_array_duplicate_total"],
               "denylist.denylist_per_array": st["per_array"]})
    mk("neg_b3_candidate_dict_identity", "记录族字段是 dict ⇒ 身份类型非法，整批失败",
       ["RECORD_IDENTITY_TYPE_INVALID"],
       tasks=[{"task_id": "t", "problem_statement": "x",
               "problem_family_id": {"fam": True}, "repo_family": "repo-a"}])
    mk("neg_b3_rejected_count_99", "denylist_rejected_count=99 被忽略（v4 缺陷）",
       ["COUNT_INVARIANT_BROKEN"], counts={"denylist.denylist_rejected_count": 99})
    mk("neg_b3_count_type_string", "计数写成字符串",
       ["COUNT_TYPE_INVALID"], counts={"candidate.records_raw_count": "4"})

    # ============ B4 ============
    mk("neg_b4_closure_incomplete", "file_set 漏 v3/cli.py（v4 只绑三文件）",
       ["BINDING_INCOMPLETE"],
       closure_used=[p for p in closure_paths(E0) if p != "v3/cli.py"])
    mk("neg_b4_cli_mutation_unbound", "只改 v3/cli.py 而绑定摘要不变（v4 无 BINDING 告警）",
       ["BINDING_MISMATCH"], e0dir=SYN / "e0_mut_cli")
    CASES[-1]["need_auth"] = False
    mk("neg_b4_required_package_init_missing",
       "固定 E0 REQUIRED_EXECUTION_PATHS 中 v3/data/__init__.py 漏绑定 ⇒ 常规包闭包不完整",
       ["BINDING_INCOMPLETE"],
       closure_used=[p for p in closure_paths(E0) if p != "v3/data/__init__.py"])
    mk("neg_b4_old_e0_schema_unconsumed", "固定 ffc37b4 旧 E0 未消费新协议 schema ⇒ 诊断轮命中未消费",
       ["DENYLIST_SCHEMA_NOT_CONSUMED"], synt=True, mode="test", expect="COMPUTATION_ONLY",
       e0dir=legacy_root, validator_src="e0_legacy", extra=["--validator-mode", "real_cli"])
    CASES[-1]["validator_src"] = "e0_legacy"
    CASES[-1]["e0_ref"] = LEGACY_E0_COMMIT
    legacy_manifest_path = SYN / "neg_b4_old_e0_schema_unconsumed" / "manifest.json"
    legacy_manifest = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
    legacy_manifest["binding"]["validator"]["source_commit"] = LEGACY_E0_COMMIT
    write(legacy_manifest_path, legacy_manifest)

    # ============ Y1 / Y2 ============
    mk("neg_y1_records_top_level_list",
       "records 顶层为 JSON list（v4 在此抛 AttributeError、exit 1、无报告）⇒ 必须给稳定码且仍输出报告",
       ["CANDIDATE_EMPTY"], tasks=[],
       extra=["--allow-synthetic", "--fixture-root", str(SYN)])
    CASES[-1]["args"] = ["--allow-synthetic", "--fixture-root", str(SYN)]
    d = SYN / "neg_y1_records_top_level_list"
    write(d / "records.json", [], raw="[]\n")
    m = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    m["binding"]["candidate"] = measured(d / "records.json")
    write(d / "manifest.json", m)

    mk("neg_y2_stat_both_denied_expectation",
       "stat 未单列（stat_recorded_separately=false）⇒ 与 stat 单列口径冲突",
       ["G4_MECHANISM_INSUFFICIENT"],
       mutate=lambda m: m["records"]["isolation_acceptance"]["negative_test"].update(
           {"stat_recorded_separately": False, "stat_result": None}))

    # ============ 保留的 v4 已通过反例 ============
    mk("neg_keep_intersection", "候选与 denylist 真实相交 ⇒ 诊断语义保留",
       ["INTERSECTION_NONZERO"], tasks=INTERSECTING_TASKS)
    null_dn = copy.deepcopy(GOOD_DENYLIST)
    null_dn["h_family_ids"] = H_FAMS[:5] + [None]
    mk("neg_keep_denylist_null_item", "denylist 含 null 项 ⇒ 整批失败",
       ["DENYLIST_ITEM_INVALID"], dn=null_dn)
    mk("neg_keep_candidate_empty", "候选 0 条", ["CANDIDATE_EMPTY"], tasks=[])
    mk("neg_keep_bool_only_evidence",
       "四门槛只填布尔、证据为空",
       ["EVIDENCE_SELF_DECLARED"], ev={g: [] for g in EV})
    CASES[-1]["need_auth"] = False

    # ============ v6 新增：受信来源 / Git blob / 根包 / 回执 ============
    TRUST = HERE / "trust"
    write(TRUST / "trusted_principals.json",
          {"syn-independent-reviewer": {"role": "designated_evaluator",
                                        "os_identity": "S-1-5-21-0-0-1-1005",
                                        "approved_scope": ["G1", "G2", "G3", "G4"]}})
    write(TRUST / "approvals.json",
          {"syn-appr-0001": {"approved_by": "syn-independent-reviewer", "scope": "G1-G4"}})
    TP = ["--trusted-principals", str(TRUST / "trusted_principals.json")]
    AP = ["--approvals", str(TRUST / "approvals.json")]

    # 正例同时走受信来源核验路径
    CASES[0]["args"] = CASES[0]["args"] + TP + AP

    forged = {g: [ev_entry(n, sha((evd / n).read_bytes()),
                           producer="invented-independent-reviewer")
                  for n in ns] for g, ns in
              (("G1_namespace_compatibility", ["g1_namespace.json"]),
               ("G2_effective_input_binding", ["g2_binding.json"]),
               ("G3_snapshot_content_bound", ["g3_snapshot.json"]),
               ("G4_isolation_verified", ["g4_isolation.json"]))}
    mk("neg_b1_forged_evidence_complete_format",
       "格式完整的伪造证据：字段齐全、摘要正确、时间合法、producer 只是另一个字符串 ⇒ 受控注册表核验必须拒绝",
       ["EVIDENCE_UNTRUSTED_PRODUCER"], ev=forged, extra=TP)
    CASES[-1]["args"] = list(CASES[-1]["args"]) + TP

    write(evd / "g1_approval_forged.json", {
        "namespace_v3": "problem_family_id/repo_family", "namespace_v2": "v2 family id",
        "derivation_rule": "strip+lower 比对 token", "extractor_id": "syn-extractor-1",
        "sample_count": 12,
        "approved_by": "syn-independent-reviewer", "approval_id": "forged-appr-9999"})
    forged_appr = copy.deepcopy(EV)
    forged_appr["G1_namespace_compatibility"] = [
        ev_entry("g1_approval_forged.json",
                 sha((evd / "g1_approval_forged.json").read_bytes()),
                 kind="trusted_approval")]
    mk("neg_b1_forged_approval_ref",
       "kind=trusted_approval 且引用不在受控批准注册表内的 approval_id ⇒ 拒绝",
       ["EVIDENCE_APPROVAL_UNTRUSTED"], ev=forged_appr, extra=TP + AP)
    CASES[-1]["args"] = list(CASES[-1]["args"]) + TP + AP
    CASES[-1]["need_auth"] = False

    mk("neg_b1_production_requires_real_trust_root",
       "生产模式未提供受控主体注册表 ⇒ 不得仅凭自填字段认定可信来源",
       ["REAL_TRUST_PROVIDER_UNAVAILABLE", "AUTH_BEFORE_ACCESS"], synt=False, mode="production", extra=[])
    CASES[-1]["args"] = []
    mk("neg_b1_approval_scope_forgery",
       "approved_scope/approvals.scope 只有自述字符串、未绑定 gate/object SHA/code/主体/有效期 ⇒ 零读取 BLOCKED",
       ["EVIDENCE_APPROVAL_SCOPE_MISMATCH", "REAL_TRUST_PROVIDER_UNAVAILABLE", "AUTH_BEFORE_ACCESS"],
       synt=False, mode="production", extra=TP + AP)

    mk("neg_b4_receipt_missing_real_cli",
       "固定 E0 授权 stub 的真实 CLI 未回传消费回执 ⇒ 合成测试模式仍拒绝验收",
       ["CONSUMPTION_RECEIPT_MISSING"], synt=True, mode="test", expect="COMPUTATION_ONLY",
       validator_src="e0", extra=["--validator-mode", "real_cli"])
    mk("neg_b4_receipt_mutation_uses_pre_run_digest",
       "恶意合成 consumer 改写 candidate 后回执新 SHA ⇒ 必须对照运行前冻结 digest 拒绝",
       ["INPUT_BYTES_MISMATCH"], expect="COMPUTATION_ONLY",
       extra=["--validator-mode", "synthetic_consumer", "--simulate-consumer-mutation"])
    mk("neg_y_audit_log_overflow",
       "open 审计超过 20,000 条后显式 AUDIT_LOG_OVERFLOW 并拒绝验收",
       ["AUDIT_LOG_OVERFLOW"], expect="COMPUTATION_ONLY",
       extra=["--audit-open-probe-count", "20005"])

    mk("neg_git_blob_zero_placeholder",
       "git_blob_sha1 用 40 个零占位 ⇒ 不是真实 Git 对象身份",
       ["GIT_BLOB_INVALID"],
       mutate=lambda m: [e.update({"git_blob_sha1": "0" * 40})
                         for e in m["binding"]["validator"]["file_set"]])

    mk("neg_git_commit_identity_mismatch",
       "blob 值正确但 git_commit 指向旧提交 ⇒ 固定 commit:path 身份失败后不得启动消费者",
       ["GIT_BLOB_INVALID"],
       mutate=lambda m: [e.update({"git_commit": "ffc37b4f3ecde6579117b0012ab4f69b4cff16ab"})
                         for e in m["binding"]["validator"]["file_set"]])

    mk("neg_git_newline_mapping_mismatch",
       "固定 blob 正确但换行映射声明错误 ⇒ 身份失败后不得启动消费者",
       ["GIT_BLOB_INVALID"],
       mutate=lambda m: [e.update({"git_checkout_newline_mapping": "forged_mapping"})
                         for e in m["binding"]["validator"]["file_set"]])

    mk("neg_root_init_unbound",
       "file_set 漏根包初始化文件 v3/__init__.py（会退化为 namespace package）",
       ["BINDING_INCOMPLETE"],
       closure_used=[p for p in closure_paths(E0) if p != "v3/__init__.py"])

    write(HERE / "fixtures.json", {"cases": CASES, "closure": closure,
                                   "evidence_names": ev_names})
    print("fixtures written:", len(CASES), "cases; closure files:", len(closure))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
