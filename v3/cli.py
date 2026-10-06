"""V3 数据/训练流水线 CLI（KAGGLE-21 §7 的契约实现，本轮为 fail-closed 骨架）。

    python -m v3.cli ingest --source-lock locks/source.lock.json --out out/source_manifest.json
    python -m v3.cli split --registry data/registry.json --denylist locks/v2-denylist.json
    python -m v3.cli validate-env --manifest data/env.lock.json
    python -m v3.cli verify --question-set data/p0-eight.json --repeat 2 --fresh
    python -m v3.cli export --records data/records.json --template-lock locks/template.lock.json --out out/export.json
    python -m v3.cli audit --release data/release.json
    python -m v3.cli rollout --train-only --budget locks/budget.json
    python -m v3.cli exp1 --corpus data/corpus.json --output-dir out/
    python -m v3.cli train-preflight

每步都做到「输入 hash → 输出 manifest，失败非零退出且写状态」，
并且**不自动扩大预算、不补缺题、不启动第二个 GPU 作业**。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import subprocess
import sys
from typing import Mapping, Sequence

from .common.canonical import sha256_json, write_canonical
from .common.errors import (
    Blocked,
    FailClosed,
    IntegrityError,
    MissingInput,
    PolicyViolation,
    UnverifiedLock,
)
from .data.dedup import (
    assert_no_split_leak,
    validate_denylist_payload,
    dedup_report,
    deterministically_select,
    selection_manifest,
)
from .data.g2_integrity import (
    BASELINE_FILE_BYTES_SHA256,
    BASELINE_RELATIVE_PATH,
    EXECUTION_MANIFEST_SCHEMA_VERSION,
    EXPECTED_EXECUTION_SOURCE_LF_SHA256,
    PROTOCOL_ARTIFACT_ID,
    PROTOCOL_SCHEMA_VERSION,
    PROTOCOL_ZIP_SHA256,
    REQUIRED_EXECUTION_PATHS,
)
from .data.exporter import build_export
from .data.faces import FACE_ENV_LOCK, validate_face, validate_release
from .data.oracle import SubprocessRunner, validate_question_set
from .data.source_lock import adapt, assert_train_only, ingest_manifest
from .exp.exp1 import build_synthetic_corpus, load_corpus, run_exp1
from .t0.deps import DependencyLock
from .train.template import CANONICAL_TEMPLATE_SHA256

# Snapshot the local import closure at CLI module load time. Later tests or callers may
# import unrelated V3 modules into the same process; those are not split dependencies.
_SPLIT_MODULE_NAMES_AT_LOAD = frozenset(sys.modules)

EXIT_OK = 0
EXIT_BLOCKED = 5


def _emit_json(payload: Mapping) -> None:
    """Write structured UTF-8 JSON independently of the Windows console code page."""
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    stream = getattr(sys.stdout, "buffer", None)
    if stream is not None:
        stream.write(rendered.encode("utf-8"))
        stream.flush()
    else:
        print(rendered, end="")


def _read_json(path: str | None, what: str, required: bool = True) -> dict:
    if not path:
        if required:
            raise MissingInput("missing_argument", "缺少必需的 %s" % what)
        return {}
    if not os.path.exists(path):
        raise MissingInput("file_not_found", "%s 不存在：%s" % (what, path), path=path)
    # utf-8-sig：本仓库写出的 JSON 一律无 BOM，但同伴用 PowerShell 导出时会带 BOM，
    # 直接拒收会让交接莫名其妙地失败。
    with open(path, "r", encoding="utf-8-sig") as handle:
        text = handle.read()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise PolicyViolation(
            "invalid_json", "%s 不是合法 JSON：%s" % (what, exc), path=path
        )


def _newline_mode(raw: bytes) -> str:
    without_crlf = raw.replace(b"\r\n", b"")
    has_lf = b"\n" in without_crlf
    has_crlf = b"\r\n" in raw
    if has_lf and has_crlf:
        return "mixed"
    return "crlf" if has_crlf else "lf"


def _sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _split_execution_identity(argv: Sequence[str]) -> dict:
    package_root = pathlib.Path(__file__).resolve().parent.parent
    loaded: dict[str, list[str]] = {}
    for name in _SPLIT_MODULE_NAMES_AT_LOAD:
        module = sys.modules.get(name)
        if module is None:
            continue
        path = getattr(module, "__file__", None)
        if not path or not str(path).endswith(".py"):
            continue
        try:
            relative = pathlib.Path(path).resolve().relative_to(package_root).as_posix()
        except (OSError, ValueError):
            continue
        if relative not in REQUIRED_EXECUTION_PATHS and relative != "v3/data/g2_integrity.py":
            continue
        loaded.setdefault(relative, []).append(name)
    pin_anchor_path = "v3/data/g2_integrity.py"
    # Under `python -m v3.cli`, the actual entrypoint is __main__, not v3.cli.
    # Both aliases point at one path; retain the real runtime module name.
    cli_relative = pathlib.Path(__file__).resolve().relative_to(package_root).as_posix()
    if cli_relative not in loaded:
        loaded[cli_relative] = ["__main__"]
    missing = sorted(set(REQUIRED_EXECUTION_PATHS) - set(loaded))
    unpinned = sorted(set(loaded) - set(EXPECTED_EXECUTION_SOURCE_LF_SHA256) - {pin_anchor_path})
    files: dict[str, dict] = {}
    unpinned_paths: list[str] = []
    root = None
    commit = None
    git_root_reason = None
    try:
        root = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True).stdout.strip()
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_root_reason = "no_git_worktree"

    for relative in sorted(loaded):
        full_path = package_root / pathlib.PurePosixPath(relative)
        try:
            raw = full_path.read_bytes()
        except OSError:
            raise MissingInput("BINDING_INCOMPLETE", "执行闭包文件无法读取", path=relative)
        digest = _sha256_bytes(raw)
        normalized_raw = raw.replace(b"\r\n", b"\n")
        source_digest = _sha256_bytes(normalized_raw)
        expected = EXPECTED_EXECUTION_SOURCE_LF_SHA256.get(relative)
        if expected is None:
            if relative != pin_anchor_path:
                unpinned_paths.append(relative)
        elif source_digest != expected:
            raise IntegrityError("BINDING_CHANGED", "执行文件内容与固定闭包基线不一致", path=relative, expected=expected, actual=source_digest)
        blob = None
        if root:
            repo_relative = os.path.relpath(full_path, root).replace(os.sep, "/")
            try:
                blob = subprocess.run(["git", "rev-parse", "HEAD:%s" % repo_relative], check=True, capture_output=True, text=True).stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                pass
        computed_blob = hashlib.sha1(b"blob %d\0" % len(normalized_raw) + normalized_raw).hexdigest()
        files[relative] = {
            "module_names": sorted(loaded[relative]),
            "file_bytes_sha256": digest,
            "source_lf_sha256": source_digest,
            "byte_length": len(raw),
            "newline_mode": _newline_mode(raw),
            "git_blob_sha1": blob,
            "worktree_matches_git_blob": blob == computed_blob if blob else False,
        }
    missing_git_blobs = sorted(path for path, item in files.items() if item["git_blob_sha1"] is None or not item["worktree_matches_git_blob"])
    git_binding_complete = bool(commit) and not missing_git_blobs
    return {
        "argv": list(argv),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "entrypoint_module": "__main__" if "__main__" in loaded.get(cli_relative, []) else "v3.cli",
        "entrypoint_path": cli_relative,
        "package_root": str(package_root),
        "git_commit_sha": commit if git_binding_complete else None,
        "git_repository_head_sha": commit,
        "git_root_status": git_root_reason,
        "files": files,
        "required_paths_missing": missing,
        "unpinned_paths": unpinned_paths,
        "untracked_or_changed_paths": missing_git_blobs,
        "closure_complete": not missing and not unpinned and (set(loaded) - {pin_anchor_path}) == set(EXPECTED_EXECUTION_SOURCE_LF_SHA256) and pin_anchor_path in loaded,
        "git_binding_complete": git_binding_complete,
        "execution_pin_complete": not missing and not unpinned and not unpinned_paths,
        "pin_anchor_file": pin_anchor_path,
    }


def _validate_authorization(path: str | None, expected_sha256: str | None) -> dict:
    """当前运行时无可信批准提供器，真实模式必须在读取任何输入前停止。

    调用者给出的授权路径/摘要不能建立信任；本函数不打开授权文件、证据或数据输入。
    可信 custody / platform verifier 就绪后，应以独立实现替换此 fail-closed stub。
    """
    del path, expected_sha256
    raise PolicyViolation(
        "AUTH_BEFORE_ACCESS",
        "可信授权提供器未配置；为保证受保护输入零读取，拒绝真实模式",
        trusted_approval_provider="unavailable",
        protected_inputs_read=[],
        read_log=[{"phase": "authorization", "resource": "protected-inputs", "action": "not-read"}],
    )


def _write(path: str | None, payload: Mapping) -> str | None:
    if not path:
        return None
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    digest = write_canonical(path, payload)
    return digest


def _finalize(payload: Mapping) -> tuple[dict, str]:
    """先对**不含自身 hash** 的正文求 hash，再把 hash 写进 manifest（外部账口径）。"""
    digest = sha256_json(payload)
    final = dict(payload)
    final["manifest_sha256"] = digest
    return final, digest


def _count_tokens(messages) -> int:
    return sum(len(str(message.get("content", "")).split()) for message in messages) + len(messages)


# ---------------------------------------------------------------- ingest


def cmd_ingest(args) -> int:
    """校验来源锁：固定 revision + 许可 approved，逐条不留空。

    支持两种形状：本仓库的 `{"sources": [...]}` 与 D0（KAGGLE-23）的
    `{"repos": {...}}`（见 v3/data/source_lock.py）。归一后统一校验；不通过即非零退出。

    `--train-only` 生成训练侧视图：全量记录都做许可/revision 判定，
    只有 `split_role="train"` 的记录进入 `sources`（其余列在 `excluded_non_train`）。
    """
    lock = _read_json(args.source_lock, "--source-lock")
    manifest = ingest_manifest(lock, train_only=bool(args.train_only))
    manifest, digest = _finalize(manifest)
    _write(args.out, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK


# ---------------------------------------------------------------- split


def _run_split_payloads(
    registry: object,
    registry_raw: bytes,
    denylist_payload: object,
    denylist_raw: bytes,
    *,
    case_name: str,
    baseline: Mapping,
    argv: Sequence[str],
) -> int:
    """Same production split logic, supplied only with fixed hash-pinned fixtures."""
    if not isinstance(registry, dict):
        raise MissingInput("CANDIDATE_SHAPE_INVALID", "registry 必须是 JSON object")
    records = registry.get("tasks")
    if not isinstance(records, list) or not records:
        raise MissingInput("CANDIDATE_EMPTY", "registry.tasks 必须是非空 JSON array")
    expected_task_count = baseline.get("candidate_task_count")
    if type(expected_task_count) is not int or len(records) != expected_task_count:
        raise MissingInput("DENOMINATOR_SHORTFALL", "candidate task count 不匹配独立基线", expected=expected_task_count, actual=len(records))
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise MissingInput("CANDIDATE_RECORD_INVALID", "候选记录必须是 JSON object", index=index)
        required = ("task_id", "split", "problem_family_id", "repo_family", "problem_statement")
        missing = [field for field in required if field not in record]
        if missing:
            raise MissingInput("CANDIDATE_RECORD_INVALID", "候选记录缺少必需字段", index=index, fields=missing)
        if any(not isinstance(record[field], str) or not record[field].strip() for field in required):
            raise MissingInput("CANDIDATE_RECORD_INVALID", "候选必需字段必须是非空字符串", index=index)

    expected_counts = baseline.get("denylist_effective_counts")
    denylist_stats = validate_denylist_payload(denylist_payload, expected_counts=expected_counts)
    assert_no_split_leak(records, denylist_stats["effective"])
    report = dedup_report(records)
    if report["status"] != "clean":
        raise PolicyViolation("DEDUP_REVIEW_REQUIRED", "合成 candidate 触发重复审查")
    selected = deterministically_select(records, registry.get("quota") or {"*": 1}, salt="k24-g2-fixture-v1")
    identity = _split_execution_identity(argv)
    if not identity["execution_pin_complete"]:
        raise IntegrityError("BINDING_INCOMPLETE", "执行闭包未完整绑定", unpinned_paths=identity["unpinned_paths"], missing=identity["required_paths_missing"], unpinned=identity["untracked_or_changed_paths"])

    def input_binding(raw: bytes, value: object, name: str) -> dict:
        return {
            "fixture_path": "tests/fixtures/g2/%s/%s.json" % (case_name, name),
            "file_bytes_sha256": _sha256_bytes(raw),
            "canonical_json_sha256": sha256_json(value),
            "byte_length": len(raw),
            "newline_mode": _newline_mode(raw),
            "bytes_hashed": len(raw),
            "bytes_parsed": len(raw),
        }

    manifest = {
        "schema_version": EXECUTION_MANIFEST_SCHEMA_VERSION,
        "protocol": {
            "schema_version": PROTOCOL_SCHEMA_VERSION,
            "artifact_id": PROTOCOL_ARTIFACT_ID,
            "artifact_zip_sha256": PROTOCOL_ZIP_SHA256,
            "review_status": "candidate_not_independently_reviewed",
        },
        "execution_mode": {"mode": "test", "fixture_case": case_name, "expect": {"exit_code": baseline.get("expected_exit_code"), "reason_code": baseline.get("expected_reason_code"), "source": "pinned_fixture_baseline"}, "production_claim": False},
        "computation_result": "COMPUTATION_ONLY",
        "isolation_acceptance": "NOT_ACCEPTED",
        "overall": "NOT_ACCEPTED",
        "authorization": {"kind": "bounded_checked_in_fixture", "read_log": [
            {"phase": "fixture-config", "resource": "pinned-baseline-and-public-fixture", "action": "read"},
            {"phase": "protected-input", "resource": "real-candidate-denylist-snapshot", "action": "not-read"},
        ]},
        "binding": {
            "candidate": input_binding(registry_raw, registry, "registry"),
            "denylist": {**input_binding(denylist_raw, denylist_payload, "denylist"), "denylist_schema_version": denylist_payload["denylist_schema_version"]},
            "integrity_baseline": {"artifact_id": "k24-g2-synthetic-input-baseline-1", "file_bytes_sha256": BASELINE_FILE_BYTES_SHA256, "byte_length": baseline.get("baseline_byte_length"), "bytes_hashed": baseline.get("baseline_byte_length"), "bytes_parsed": baseline.get("baseline_byte_length"), "scope": "synthetic-fixtures-only"},
            "validator": {
                "entrypoint": "python -m v3.cli split --fixture-case %s" % case_name,
                "files": identity["files"],
                "git_commit_sha": identity["git_commit_sha"],
                "git_binding_complete": identity["git_binding_complete"],
                "closure_complete": identity["closure_complete"],
            },
            "execution_identity": identity,
        },
        "counts": {
            "candidate": {"records_raw_count": len(records), "records_effective_count": len(records), "records_rejected_count": 0, "unit": "tasks"},
            "denylist": {key: value for key, value in denylist_stats.items() if key not in {"effective", "source_mapping"}},
            "intersection_count": 0,
            "count_semantics": {"candidate": "tasks", "denylist_arrays": "effective unique family ids", "denylist_union": "families"},
        },
        "source_array_of_token": denylist_stats["source_mapping"],
        "gates": {
            "G1_namespace_compatibility": "pending",
            "G2_effective_input_binding": "fixture_only_not_real_inputs",
            "G3_snapshot_content_bound": "pending",
            "G4_isolation_verified": "pending",
        },
        "reason_code": None,
    }
    final, _ = _finalize(manifest)
    _emit_json(final)
    return EXIT_OK


def _parse_json_buffer(raw: bytes, label: str) -> object:
    try:
        return json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MissingInput("INVALID_JSON", "%s 不是合法 UTF-8 JSON" % label, error=type(exc).__name__)


def _run_fixture_case(case_name: str, argv: Sequence[str]) -> int:
    source_root = pathlib.Path(__file__).resolve().parent.parent
    baseline_path = source_root / BASELINE_RELATIVE_PATH
    try:
        baseline_raw = baseline_path.read_bytes()
    except OSError as exc:
        raise IntegrityError("BASELINE_NOT_RESOLVABLE", "合成完整性基线缺失", error=type(exc).__name__)
    if _sha256_bytes(baseline_raw) != BASELINE_FILE_BYTES_SHA256:
        raise IntegrityError("BASELINE_HASH_MISMATCH", "合成完整性基线字节摘要不符")
    baseline_doc = _parse_json_buffer(baseline_raw, "synthetic-baseline")
    if not isinstance(baseline_doc, dict) or baseline_doc.get("baseline_schema_version") != "k24-g2-synthetic-input-baseline-1":
        raise IntegrityError("BASELINE_SCHEMA_INVALID", "合成完整性基线 schema 错误")
    if baseline_doc.get("provenance", {}).get("protocol_zip_sha256") != PROTOCOL_ZIP_SHA256:
        raise IntegrityError("BASELINE_PROTOCOL_MISMATCH", "合成计数基线未绑定当前协议候选")
    case = baseline_doc.get("cases", {}).get(case_name)
    if not isinstance(case, dict):
        raise MissingInput("FIXTURE_CASE_UNKNOWN", "没有该合成用例")
    case = dict(case)
    case["baseline_byte_length"] = len(baseline_raw)
    case["baseline_file_bytes_sha256"] = _sha256_bytes(baseline_raw)
    case_root = source_root / "tests" / "fixtures" / "g2" / case_name
    try:
        registry_raw = (case_root / "registry.json").read_bytes()
    except OSError as exc:
        raise MissingInput("CANDIDATE_NOT_RESOLVABLE", "固定合成 candidate 不存在", error=type(exc).__name__)
    if _sha256_bytes(registry_raw) != case.get("registry_sha256"):
        raise IntegrityError("CANDIDATE_BASELINE_MISMATCH", "candidate 字节与独立固定基线不符")
    registry = _parse_json_buffer(registry_raw, "candidate")
    if isinstance(registry, dict):
        tasks = registry.get("tasks")
        if not isinstance(tasks, list) or len(tasks) != case.get("candidate_task_count"):
            raise MissingInput("DENOMINATOR_SHORTFALL", "candidate 数量与独立固定基线不符")

    denylist_path = case_root / "denylist.json"
    if case.get("denylist_sha256") is None:
        if denylist_path.exists():
            raise IntegrityError("FIXTURE_LAYOUT_CHANGED", "missing 用例不应包含 denylist")
        raise MissingInput("file_not_found", "固定 fixture 缺少 denylist")
    try:
        denylist_raw = denylist_path.read_bytes()
    except OSError as exc:
        raise MissingInput("file_not_found", "固定 fixture 缺少 denylist", error=type(exc).__name__)
    if _sha256_bytes(denylist_raw) != case.get("denylist_sha256"):
        raise IntegrityError("DENYLIST_BASELINE_MISMATCH", "denylist 字节与独立固定基线不符")
    denylist_payload = _parse_json_buffer(denylist_raw, "denylist")
    return _run_split_payloads(registry, registry_raw, denylist_payload, denylist_raw, case_name=case_name, baseline=case, argv=argv)


def cmd_split(args) -> int:
    """真实输入默认在读取前阻断；只允许预置固定 SHA 的合成 fixture。"""
    argv = list(sys.argv[1:] if args._argv is None else args._argv)
    if args.fixture_case:
        real_args = (args.registry, args.denylist, args.registry_sha256, args.denylist_sha256,
                     args.authorization_manifest, args.authorization_sha256, args.out)
        if any(value is not None for value in real_args) or args.allow_review:
            raise PolicyViolation("FIXTURE_INPUT_OVERRIDE", "fixture 模式不接受调用者输入、摘要、授权或输出路径")
        return _run_fixture_case(args.fixture_case, argv)
    _validate_authorization(args.authorization_manifest, args.authorization_sha256)
    raise PolicyViolation("AUTH_BEFORE_ACCESS", "真实模式在读取 protected input 前停止", protected_inputs_read=[])


# ---------------------------------------------------------------- validate-env


def cmd_validate_env(args) -> int:
    manifest = _read_json(args.manifest, "--manifest")
    validate_face(FACE_ENV_LOCK, manifest)
    summary = {
        "stage": "validate-env",
        "env_id_sha256": sha256_json(manifest),
        "cpu": manifest.get("cpu"),
        "ram_bytes": manifest.get("ram_bytes"),
        "concurrency": 1,
        "note": "不是仅 pip freeze：os/arch、容器 digest、wheel hash、测试选集与 timeout 都必须齐备。",
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK


# ---------------------------------------------------------------- verify


def cmd_verify(args) -> int:
    question_set = _read_json(args.question_set, "--question-set")
    questions = question_set.get("questions")
    if not questions:
        raise MissingInput("empty_question_set", "题目集为空")
    if not args.fresh:
        raise PolicyViolation("verify_requires_fresh", "verify 必须带 --fresh（干净 workspace）")
    if args.repeat < 2:
        raise PolicyViolation("verify_requires_two_runs", "每题需要至少两次复验，--repeat 2")

    runner = SubprocessRunner(timeout_seconds=question_set.get("timeout_seconds", 300))
    report = validate_question_set(questions, lambda question: runner)
    report["stage"] = "verify"
    report["repeat"] = args.repeat
    report["question_set_sha256"] = sha256_json(question_set)
    digest = _write(args.out, report)
    report["manifest_sha256"] = digest
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK if report["qualified"] else EXIT_BLOCKED


# ---------------------------------------------------------------- export


def cmd_export(args) -> int:
    payload = _read_json(args.records, "--records")
    records = payload.get("records")
    if not records:
        raise MissingInput("empty_records", "records 为空")
    template_lock = _read_json(args.template_lock, "--template-lock", required=False)
    if not template_lock.get("template_sha256"):
        raise UnverifiedLock(
            "template_lock_missing",
            "缺少模板锁定（template_sha256）：不得用未锁定模板导出训练视图",
        )
    # Q0 🔵-7：旧实现只要求 template_sha256 **非空**，既不与 canonical SHA 比对、
    # 也不要求 verified=true —— 等于"填个字符串就能导出"。现在两条都强制。
    if not template_lock.get("verified"):
        raise UnverifiedLock(
            "template_lock_unverified",
            "模板锁 verified != true：未核实的模板不得用于导出训练视图",
            template_lock=template_lock,
        )
    if template_lock["template_sha256"] != CANONICAL_TEMPLATE_SHA256:
        raise IntegrityError(
            "template_sha_mismatch",
            "模板锁里的 template_sha256 与 canonical 值不一致",
            expected=CANONICAL_TEMPLATE_SHA256,
            actual=template_lock["template_sha256"],
        )
    mask_config = _read_json(args.mask_config, "--mask-config", required=False)
    export = build_export(
        records,
        _count_tokens,
        window_tokens=int(args.window_tokens),
        export_config={"template_lock": template_lock, "mask_config": mask_config},
    )
    export["export_manifest"]["stage"] = "export"
    digest = _write(args.out, export["export_manifest"])
    export["export_manifest"]["manifest_sha256"] = digest
    print(json.dumps(export["export_manifest"], ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK


# ---------------------------------------------------------------- audit


def cmd_audit(args) -> int:
    release = _read_json(args.release, "--release")
    summary = validate_release(release)
    summary["stage"] = "audit"
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK


# ---------------------------------------------------------------- rollout


def cmd_rollout(args) -> int:
    """轨迹生成：本任务不具备模型与预算，如实阻断，不自动扩预算。"""
    if not args.train_only:
        raise PolicyViolation("rollout_train_only", "rollout 只允许对 train 生成（--train-only）")
    budget = _read_json(args.budget, "--budget")
    if budget.get("gpu_hours_released", 0) <= 0:
        raise PolicyViolation(
            "no_gpu_budget", "未被释放 GPU 预算：不自动扩大预算、不启动第二个 GPU 作业"
        )
    raise Blocked(
        "rollout_requires_teacher_model",
        "轨迹生成需要许可的教师模型与目标平台环境，交 107 运行统筹执行",
    )


# ---------------------------------------------------------------- exp1 / preflight


def cmd_exp1(args) -> int:
    corpus = load_corpus(args.corpus) if args.corpus else build_synthetic_corpus()
    report = run_exp1(corpus, output_dir=args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK


def cmd_train_preflight(args) -> int:
    from .train.entry import preflight

    report = preflight(config_path=args.config)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK if not report["blockers"] else EXIT_BLOCKED


def cmd_deps(args) -> int:
    train = DependencyLock.from_file(args.train_lock).summary()
    serving = DependencyLock.from_file(args.serving_lock).summary()
    print(json.dumps({"train": train, "serving": serving}, ensure_ascii=False, indent=2, sort_keys=True))
    return EXIT_OK if (train["verified"] and serving["verified"]) else EXIT_BLOCKED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="v3", description="V3 数据/训练流水线（fail-closed）")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="校验来源锁（支持 D0 版本化 manifest）")
    p.add_argument("--source-lock", required=True)
    p.add_argument("--train-only", action="store_true", help="拒绝非 train 角色的来源")
    p.add_argument("--out")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("split", help="家族/时间去重与确定性选择")
    p.add_argument("--fixture-case", choices=("clean", "missing", "intersection", "invalid"), help="运行固定哈希锁定的公开合成 fixture，不接受路径参数")
    p.add_argument("--registry", help="真实模式；当前可信授权提供器缺失，输入读取会在授权阶段前置阻断")
    p.add_argument("--registry-sha256", help="仅调用者声明摘要，不能代替独立基线")
    p.add_argument("--denylist", help="真实模式；当前可信授权提供器缺失，输入读取会在授权阶段前置阻断")
    p.add_argument("--denylist-sha256", help="仅调用者声明摘要，不能代替独立基线")
    p.add_argument("--authorization-manifest", help="调用者文件不构成可信授权；当前不会读取")
    p.add_argument("--authorization-sha256", help="调用者摘要不构成信任锚；当前不会读取")
    p.add_argument("--salt", default="KAGGLE-21-V3-data-v1")
    p.add_argument("--allow-review", action="store_true", help="重复项已人工审查后放行")
    p.add_argument("--out")
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("validate-env", help="校验 env.lock")
    p.add_argument("--manifest", required=True)
    p.set_defaults(func=cmd_validate_env)

    p = sub.add_parser("verify", help="broken/reference/P2P 对照复验")
    p.add_argument("--question-set", required=True)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--out")
    p.set_defaults(func=cmd_verify)

    p = sub.add_parser("export", help="导出训练视图")
    p.add_argument("--records", required=True)
    p.add_argument("--template-lock", required=True)
    # Q0 🔵-7：设计稿 KAGGLE-21 §7 的 export 契约里有 --mask-config，旧实现漏了。
    p.add_argument("--mask-config", default=None, help="mask 配置的 JSON 路径（设计稿 §7 要求）")
    p.add_argument("--window-tokens", type=int, default=2048)
    p.add_argument("--out")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("audit", help="发布前审计")
    p.add_argument("--release", required=True)
    p.set_defaults(func=cmd_audit)

    p = sub.add_parser("rollout", help="轨迹生成（需教师模型与预算）")
    p.add_argument("--train-only", action="store_true")
    p.add_argument("--budget", required=True)
    p.set_defaults(func=cmd_rollout)

    p = sub.add_parser("exp1", help="CPU EXP-1")
    p.add_argument("--corpus")
    p.add_argument("--output-dir")
    p.set_defaults(func=cmd_exp1)

    p = sub.add_parser("train-preflight", help="训练静态 preflight")
    p.add_argument("--config")
    p.set_defaults(func=cmd_train_preflight)

    p = sub.add_parser("deps", help="打印两侧依赖锁状态")
    p.add_argument("--train-lock", default=os.path.join(os.path.dirname(__file__), "locks", "train.lock.json"))
    p.add_argument("--serving-lock", default=os.path.join(os.path.dirname(__file__), "locks", "serving.lock.json"))
    p.set_defaults(func=cmd_deps)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parsed_argv = list(argv) if argv is not None else list(sys.argv[1:])
    args = parser.parse_args(parsed_argv)
    args._argv = parsed_argv
    try:
        return args.func(args)
    except FailClosed as exc:
        result = exc.to_dict()
        result["reason_code"] = exc.code
        _emit_json(result)
        return exc.exit_code


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
