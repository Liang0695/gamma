"""四面数据契约（task.public / task.audit / oracle.private / trajectory）。

实现 KAGGLE-21 §5 的表格与 KAGGLE-19 §5：
- 每个面上的"最少字段"缺一即非法（省略不是默认值）；
- `verification.status` 与 `release_state` 是枚举；
- **训练/搜索作者只能读 task.public 与 trajectory 的 train 视图**，
  `oracle.private` 的 gold 与封存 split 记录不得进入 actor 输入；
- `release_state=released` 要求全部引用可解析、hash 非空匹配、许可 approved、split 冻结、复验 pass，
  且禁止示例字符串充当 ID/哈希。
"""

from __future__ import annotations

import re
from typing import Mapping

from ..common.canonical import problem_family_id, sha256_json
from ..common.errors import MissingInput, PolicyViolation

FACE_PUBLIC = "task.public"
FACE_AUDIT = "task.audit"
FACE_ORACLE = "oracle.private"
FACE_TRAJECTORY = "trajectory"
FACE_ENV_LOCK = "env.lock"
FACE_RUN_PROTOCOL = "run.protocol"

REQUIRED_FIELDS: dict[str, tuple[str, ...]] = {
    FACE_PUBLIC: (
        "task_id",
        "origin_kind",
        "repo_url",
        "repo_family",
        "base_commit",
        "source_time",
        "problem_statement",
        "statement_sha256",
        "snapshot_sha256",
        "env_id",
        "public_test_commands",
        "split",
        "problem_family_id",
        "difficulty",
        "parent_ids",
    ),
    FACE_AUDIT: (
        "source_urls",
        "pr_ids",
        "original_file_sha256",
        "license_spdx",
        "license_text_sha256",
        "notice_sha256",
        "authorization_scope",
        "acquired_at",
        "transforms",
        "generator_revision",
        "generator_seed",
        "lineage",
        "dedup_evidence",
        "exclusions",
        "reviewer",
        "reviewed_at",
    ),
    FACE_ORACLE: (
        "gold_patch_sha256",
        "test_patch_sha256",
        "f2p_tests",
        "p2p_tests",
        "verify_commands",
        "timeout_seconds",
        "expected_behaviour",
        "gold_files",
        "gold_symbols",
        "acceptable_locations",
        "validator_version",
        "contrast_runs",
    ),
    FACE_TRAJECTORY: (
        "trace_id",
        "task_id",
        "teacher_source",
        "teacher_revision",
        "template_lock",
        "parser",
        "tool_schema_sha256",
        "seed",
        "gold_access",
        "steps",
        "before_tree_sha256",
        "after_tree_sha256",
        "elapsed_ms",
        "token_counts",
        "candidate_patch_sha256",
        "verification_ref",
        "failure_labels",
        "review_status",
        "loss_mask_ref",
    ),
    FACE_ENV_LOCK: (
        "os",
        "arch",
        "container_digest",
        "python_version",
        "test_runner_version",
        "dependency_wheel_hashes",
        "install_commands",
        "locale",
        "timezone",
        "env_allowlist",
        "network_policy",
        "cpu",
        "ram_bytes",
        "seed",
        "test_selection",
        "timeout_seconds",
        "snapshot_tree_sha256",
    ),
    FACE_RUN_PROTOCOL: (
        "model_id",
        "model_revision",
        "tokenizer_sha256",
        "template_sha256",
        "parser",
        "serving_wheel_hashes",
        "actual_flags",
    ),
}

SPLITS = ("train", "dev", "test")
ORIGIN_KINDS = ("real_history", "mutation", "synthetic_behaviour")
VERIFICATION_STATUS = ("pass", "fail", "invalid_env", "inconclusive", "not_run")
RELEASE_STATES = ("design_only", "candidate", "quarantine", "released")
PHASES = ("localize", "edit", "validate", "recover", "finish")
LOCALIZE_ACTIONS = ("localize", "tool", "patch_verify", "recover")

#: 「示例字符串」不能当 ID/哈希：这些显式标记一律拒绝进入 released。
PLACEHOLDER_VALUES = frozenset(
    {
        "",
        "example",
        "EXAMPLE",
        "todo",
        "TODO",
        "tbd",
        "TBD",
        "xxx",
        "placeholder",
        "design_only",
        "PENDING",
        "none",
        "null",
    }
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def is_hex64(value) -> bool:
    return isinstance(value, str) and bool(_HEX64.match(value))


def is_placeholder(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() in PLACEHOLDER_VALUES or value.strip() == ""
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    return False


def validate_face(face: str, record: Mapping) -> None:
    """字段级校验：缺字段 / 枚举越界 / 空占位即抛异常。"""
    if face not in REQUIRED_FIELDS:
        raise MissingInput("unknown_face", "未知数据面：%s" % face)
    if not isinstance(record, Mapping):
        raise MissingInput("bad_face_record", "%s 记录不是对象" % face)
    missing = [field for field in REQUIRED_FIELDS[face] if field not in record]
    if missing:
        raise MissingInput(
            "face_missing_fields", "%s 缺少必需字段" % face, missing=missing
        )
    if face == FACE_PUBLIC:
        if record["split"] not in SPLITS:
            raise PolicyViolation("bad_split", "非法 split：%r" % (record["split"],))
        if record["origin_kind"] not in ORIGIN_KINDS:
            raise PolicyViolation(
                "bad_origin_kind", "非法 origin_kind：%r" % (record["origin_kind"],)
            )
        expected = problem_family_id(record.get("parent_ids") or [record["task_id"]])
        if record.get("problem_family_id") and record["problem_family_id"] != expected:
            raise PolicyViolation(
                "family_id_mismatch",
                "problem_family_id 与 canonical(sorted origin IDs) 不一致",
                recorded=record["problem_family_id"],
                computed=expected,
            )
    if face == FACE_TRAJECTORY:
        for step in record.get("steps") or []:
            validate_step(step)


def validate_step(step: Mapping) -> None:
    """逐动作字段（KAGGLE-21 §5 末段）。"""
    required = (
        "step",
        "role",
        "phase",
        "tool_name",
        "arguments",
        "observation_ref",
        "observation_sha256",
        "exit_code",
        "started_at",
        "duration_ms",
        "input_tree_sha256",
        "output_tree_sha256",
        "finish_reason",
        "loss_eligible",
    )
    missing = [field for field in required if field not in step]
    if missing:
        raise MissingInput("step_missing_fields", "轨迹动作缺少字段", missing=missing)
    if step["phase"] not in PHASES:
        raise PolicyViolation("bad_phase", "非法 phase：%r" % (step["phase"],))


def assert_no_oracle_in_actor_input(actor_payload: Mapping) -> None:
    """gold / test_patch / 未来信息绝不能出现在 actor 输入里。

    这是 §5「不将 oracle 字段拼进 actor 输入」与 §6 的机械检查。
    """
    forbidden_keys = (
        "gold_patch",
        "gold_patch_sha256",
        "test_patch",
        "test_patch_sha256",
        "gold_files",
        "gold_hunks",
        "gold_symbols",
        "acceptable_locations",
        "f2p_tests",
        "p2p_tests",
        "future_commits",
        "hints_text",
    )
    found = []

    def walk(node, path: str) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                if key in forbidden_keys:
                    found.append("%s.%s" % (path, key))
                walk(value, "%s.%s" % (path, key))
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, "%s[%d]" % (path, index))

    walk(actor_payload, "actor")
    if found:
        raise PolicyViolation(
            "oracle_leak",
            "actor 输入里出现了 oracle 域字段：%s" % ", ".join(sorted(found)),
            fields=sorted(found),
        )


def validate_verification(status: str, f2p_all_pass: bool, p2p_clean: bool, tampered: bool, repeats: int) -> None:
    """`verification.status` 与 resolved 的合法组合。"""
    if status not in VERIFICATION_STATUS:
        raise PolicyViolation("bad_verification_status", "非法 status：%r" % (status,))
    if status == "pass":
        if not (f2p_all_pass and p2p_clean):
            raise PolicyViolation(
                "verification_pass_inconsistent",
                "status=pass 但 F2P/P2P 未全部满足",
            )
        if tampered:
            raise PolicyViolation("verification_tampered", "验收工具/测试被篡改，不得算 pass")
        if repeats < 2:
            raise PolicyViolation(
                "verification_repeats", "pass 需要两次新鲜 workspace 复验，实际 %d" % repeats
            )


def validate_release(bundle: Mapping) -> dict:
    """发布级校验。返回可供 manifest 使用的摘要。

    `bundle` 形状：
      {"release_id": str, "release_state": str, "faces": {face: record}, "frozen": {...}}
    """
    state = bundle.get("release_state")
    if state not in RELEASE_STATES:
        raise PolicyViolation("bad_release_state", "非法 release_state：%r" % (state,))
    faces = bundle.get("faces") or {}
    for face, record in faces.items():
        validate_face(face, record)

    frozen = bundle.get("frozen") or {}
    public = faces.get(FACE_PUBLIC) or {}
    audit = faces.get(FACE_AUDIT) or {}

    problems: list[str] = []
    if is_placeholder(bundle.get("release_id")):
        problems.append("release_id 是空占位")
    if state == "released":
        for face in (FACE_PUBLIC, FACE_AUDIT, FACE_ORACLE, FACE_TRAJECTORY):
            if face not in faces:
                problems.append("released 缺少数据面 %s" % face)
        for field in ("statement_sha256", "snapshot_sha256"):
            if not is_hex64(public.get(field)):
                problems.append("task.public.%s 不是可解析的 64 位 hex" % field)
        if not is_hex64((faces.get(FACE_ORACLE) or {}).get("gold_patch_sha256")):
            problems.append("oracle.private.gold_patch_sha256 为空或不可解析")
        if str(audit.get("authorization_scope", "")).lower() not in ("approved", "train_allowed"):
            problems.append("task.audit.authorization_scope 不是 approved")
        if not frozen.get("ids_frozen"):
            problems.append("ids_frozen 未置位")
        if not frozen.get("content_verified"):
            problems.append("content_verified 未置位")
        if not frozen.get("oracle_verified"):
            problems.append("oracle_verified 未置位")
        if not frozen.get("export_frozen"):
            problems.append("export_frozen 未置位")
        if not frozen.get("protocol_frozen"):
            problems.append("protocol_frozen 未置位")

    if problems:
        raise PolicyViolation(
            "release_not_releasable",
            "release_state=%s 不满足发布条件（%d 项）" % (state, len(problems)),
            problems=problems,
        )
    return {
        "release_id": bundle.get("release_id"),
        "release_state": state,
        "faces": sorted(faces),
        "bundle_sha256": sha256_json(
            {"release_id": bundle.get("release_id"), "faces": faces}
        ),
    }
