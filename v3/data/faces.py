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

#: V3 的**权威** split 三值是 train / dev / sealed —— 来源：D0 split manifest
#: （D0 台账逐仓库登记 `split_role ∈ {train, dev, sealed}`）与 Mika 2026-10-05 裁决。
#: 旧代码把第三个值写成 `test`，导致 `split="sealed"` 被判 `bad_split`（Q0 Y8 ①）。
V3_SPLITS = ("train", "dev", "sealed")
#: `test` 是历史遗留别名（早期契约定稿与既有下游仍在使用），为不破坏现有用法而保留。
#: 新增记录**不得**使用 `test`；V3 正式口径只有 V3_SPLITS 三值。
LEGACY_SPLITS = ("test",)
SPLITS = V3_SPLITS + LEGACY_SPLITS
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


# --------------------------------------------------------------- actor 可见面（Q0 Y1）

#: actor 可见面的**顶层键白名单**（唯一真源）。
#:
#: 依据（2026-10-05 逐个读现有调用方与契约后确定，不是猜的）：
#: - `v3/data/exporter.py:119-132` `build_windows()` 产出的 window 键：
#:   task_id / trace_id / base_commit / messages / step_range / supervised_steps /
#:   supervision_buckets / gold_access / split / repo_family；`build_export()` 另外补写
#:   origin_kind / supervised_token_count；
#: - `v3/data/exporter.py:63-73` `steps_to_messages()` 产出的 message 键：
#:   role / content / tool_name / arguments / observation_ref / loss_eligible / phase / step；
#: - `v3/train/fixtures.py` 的动作消息沿用同一组 message 键；
#: - `REQUIRED_FIELDS[FACE_PUBLIC]` 中允许进入 actor 输入的展示字段：
#:   task_id / problem_statement / split / repo_family / origin_kind / base_commit；
#: - `public` / `actions` / `observation` / `split_role` 是留给其它 actor 输入装配点
#:   （question set / exp1 语料视图 / D0 视图）的登记键，暂时没有生产调用方。
#:
#: 未登记的键一律拒绝，**不做前缀、子串或"看起来像正常字段"的放行**。
ACTOR_VISIBLE_KEYS = frozenset(
    {
        "task_id",
        "trace_id",
        "problem_statement",
        "public",
        "messages",
        "actions",
        "observation",
        "observation_ref",
        "tool_name",
        "arguments",
        "role",
        "content",
        "step",
        "step_range",
        "supervised_steps",
        "supervision_buckets",
        "phase",
        "split",
        "split_role",
        "repo_family",
        "origin_kind",
        "base_commit",
        "gold_access",
        "loss_eligible",
        "finish_reason",
        "exit_code",
        "supervised_token_count",
    }
)

#: 键名黑名单（沿用原 `forbidden_keys`，并补齐 Q0 点名的改名变体）。
ORACLE_LEAK_KEYS = frozenset(
    {
        "gold_patch",
        "gold_patch_sha256",
        "gold_patch_text",
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
        "solution",
        "solution_patch",
        "reference_fix",
        "expected_patch",
        "patch",
        "answer",
        "fix",
    }
)

#: 词语级黑名单：把键名按 camelCase / snake_case / kebab-case 切成词后**逐段精确比对**。
#: 刻意不做子串匹配 —— 子串匹配会把 `prefix`（含 fix）、`golden` 之类正常键误伤，
#: 而误伤会让下游为了跑通而绕过检查，反而更不安全（见函数 docstring 的定位说明）。
ORACLE_LEAK_TOKENS = frozenset(
    {
        "gold",
        "oracle",
        "solution",
        "reference",
        "answer",
        "hint",
        "future",
        "patch",
        "fix",
        "expected",
    }
)

#: 这些键**本身**就以 hex 摘要 / 提交号作为合法取值（actor 可见的自身仓库状态、
#: 自身观察摘要），因此不对其做"40/64 位 hex 即拒绝"的值级检查。其余键一律检查。
HASH_VALUED_KEYS = frozenset(
    {
        "base_commit",
        "commit",
        "pinned_commit",
        "sha",
        "sha1",
        "sha256",
        "tree_sha",
        "before_tree_sha256",
        "after_tree_sha256",
        "input_tree_sha256",
        "output_tree_sha256",
        "observation_sha256",
        "candidate_patch_sha256",
        "template_sha256",
        "tool_schema_sha256",
        "statement_sha256",
        "snapshot_sha256",
    }
)

#: 值级：40 位 hex（可疑 commit SHA）、64 位 hex（可疑 patch 片段哈希）。
_VALUE_SHA40 = re.compile(r"\b[0-9a-fA-F]{40}\b")
_VALUE_HEX64 = re.compile(r"\b[0-9a-fA-F]{64}\b")
#: 值级：明显的补丁文本特征（行首 `@@ -`、`diff --git`、统一 diff 的文件头）。
_VALUE_PATCH_TEXT = re.compile(r"diff --git |^@@ -|^--- a/|^\+\+\+ b/", re.MULTILINE)
#: 值级：把 oracle 结论直接写成自然语言的措辞（"the fix is at routing.py line 12" 这类）。
#: 这是**启发式**，只求不漏放到明显情形，不承诺覆盖改写后的自然语言。
ORACLE_HINT_MARKERS = (
    "the fix is",
    "fix is at",
    "the answer is",
    "apply this patch",
    "gold patch",
    "gold_patch",
    "solution_patch",
    "reference_fix",
    "expected_patch",
    "test_patch",
)

#: 键名切词 + 归一（camelCase → snake，再按非字母数字切分）。
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_NON_WORD = re.compile(r"[^a-z0-9]+")


def _key_words(key) -> list[str]:
    text = _CAMEL_BOUNDARY.sub("_", str(key))
    return [word for word in _NON_WORD.split(text.lower()) if word]


def _is_leak_key(key) -> bool:
    """键名是否命中 oracle 域黑名单（精确名或切词后逐段命中）。"""
    lowered = str(key).strip().lower()
    if lowered in ORACLE_LEAK_KEYS:
        return True
    return any(word in ORACLE_LEAK_TOKENS for word in _key_words(key))


def _is_hash_valued_key(key) -> bool:
    lowered = str(key).strip().lower()
    return (
        lowered in HASH_VALUED_KEYS
        or lowered.endswith("_sha256")
        or lowered.endswith("_sha")
    )


def assert_no_oracle_in_actor_input(actor_payload: Mapping, max_depth: int = 3) -> None:
    """gold / test_patch / 未来信息绝不能出现在 actor 输入里。

    这是 §5「不将 oracle 字段拼进 actor 输入」与 §6 的机械检查，分三层：

    1. **顶层白名单**：顶层键必须登记在 `ACTOR_VISIBLE_KEYS` 里，否则
       `PolicyViolation("oracle_leak_unknown_field")`（异常里带键名）；
    2. **递归键名黑名单**（递归至少到 `max_depth=3`）：嵌套 dict 的键名按
       `ORACLE_LEAK_KEYS` / `ORACLE_LEAK_TOKENS` 判定 →
       `PolicyViolation("oracle_leak")`（含 `reference_fix` / `solution_patch` /
       `expected_patch` 之类改名变体）；
    3. **值级检查**：字符串值里出现 40 位 hex commit SHA、64 位 hex patch 片段哈希、
       补丁文本特征（`diff --git ` / `@@ -` / `--- a/` / `+++ b/` 行头）或
       "the fix is" 这类 oracle 结论措辞 → `PolicyViolation("oracle_leak_value")`。

    .. warning::
       这是**机械防误拼检查，不是隔离机制**。它只拦"已知名字 / 已知取值形状"，
       无法阻止改写、编码、自然语言转述后的泄漏，也不证明任何隔离性质。
       **下游（评审、发布闸门、报告）不得把它当作隔离证明引用**：隔离必须由
       oracle.private 与 actor 输入在装配层就物理分离来保证（Q0 报告 🔵-4）。
    """
    if not isinstance(actor_payload, Mapping):
        raise MissingInput(
            "bad_actor_payload", "actor 输入不是对象：%r" % (type(actor_payload).__name__,)
        )

    unknown = sorted(str(key) for key in actor_payload if key not in ACTOR_VISIBLE_KEYS)
    if unknown:
        raise PolicyViolation(
            "oracle_leak_unknown_field",
            "actor 输入顶层出现未登记字段（白名单外一律拒绝）：%s" % ", ".join(unknown),
            fields=unknown,
        )

    leaked_keys: list[str] = []
    leaked_values: list[str] = []

    def check_value(value: str, path: str, key) -> None:
        if _VALUE_PATCH_TEXT.search(value):
            leaked_values.append("%s（补丁文本特征）" % path)
            return
        lowered = value.lower()
        if any(marker in lowered for marker in ORACLE_HINT_MARKERS):
            leaked_values.append("%s（oracle 结论措辞）" % path)
            return
        if _is_hash_valued_key(key):
            return
        if _VALUE_SHA40.search(value):
            leaked_values.append("%s（40 位 hex commit SHA）" % path)
        elif _VALUE_HEX64.search(value):
            leaked_values.append("%s（64 位 hex patch 片段哈希）" % path)

    def walk(node, path: str, depth: int) -> None:
        """depth = dict 嵌套层数：顶层映射为 0，其值的映射为 1 ……"""
        if isinstance(node, Mapping):
            for key, value in node.items():
                child = "%s.%s" % (path, key)
                # 顶层键（depth 0）已由白名单覆盖，所以键名黑名单从 depth 1 起算；
                # 超过 max_depth 就不再查键名（黑名单承诺范围即 max_depth），
                # 但字符串值仍然全量检查。
                if 1 <= depth <= max_depth and _is_leak_key(key):
                    leaked_keys.append(child)
                walk(value, child, depth + 1)
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, "%s[%d]" % (path, index), depth)
        elif isinstance(node, str):
            check_value(node, path, path.rsplit(".", 1)[-1])

    walk(actor_payload, "actor", 0)

    if leaked_keys:
        raise PolicyViolation(
            "oracle_leak",
            "actor 输入里出现了 oracle 域字段：%s" % ", ".join(sorted(leaked_keys)),
            fields=sorted(leaked_keys),
        )
    if leaked_values:
        raise PolicyViolation(
            "oracle_leak_value",
            "actor 输入里出现了 oracle 取值特征：%s" % ", ".join(sorted(leaked_values)),
            fields=sorted(leaked_values),
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
