"""来源锁适配器：让 D0（KAGGLE-23）的版本化 manifest 能被 ingest 直接消费。

动机：设计稿要求"与数据生产者通过版本化 manifest 交接"，但 D0 的
`d0/out/source-lock.json` 与代码最初假设的形状不同：

- 本仓库最初假设：`{"sources": [{"name", "repo_url", "commit", "license_spdx",
  "authorization_scope", "acquired_at"}]}`；
- D0 实际产出：`{"repos": {"<name>": {"upstream_slug", "mirror_url", "pinned_tag",
  "pinned_commit", "tree_sha", "split_role", "design_license_expectation",
  "spdx_headers_found", "packaging_metadata"}}}`。

本模块把两种形状都归一成同一份内部结构，并且**不猜许可结论**。

### D0 v2 许可契约（2026-10-06 Mika 裁定冻结，🔴-A）

契约以 **D0 的形状为准**，E0 侧适配：

    repos.<name>.license_review.decision                  ∈ {approved, rejected, pending}
    repos.<name>.license_review.decided_against_revision  == repos.<name>.pinned_commit
    repos.<name>.license_review.independent_review.status == "pending"（占位，不是独立签字）

出处是 D0 manifest 自带的 `license_review_schema`：`gate` 明确"no repository may be
used as a family source unless `license_review.decision == 'approved'` for the exact
revision in `decided_against_revision`"，`approval_is_revision_bound` 明确"an approval
covers one pinned commit; re-pinning a repository invalidates it"。

三条硬规则：

1. **revision 绑定**：`decided_against_revision` 必须等于该记录的 `pinned_commit`；
   不等（或取不到 pinned commit）即拒绝 —— 重新 pin 会让旧批准失效。
2. **任一拒绝/冲突信号优先拒绝**：`decision != approved`、copyleft/restrictive 命中、
   `osi_permissive` 非真、`approved_spdx` 为空、`independent_review.status == "rejected"`，
   任意一条命中即整条记录 `unverified`，**不看键序、不看是否另有 approved 信号**。
3. **导入 ≠ 批准**：`independent_review.status="pending"` 原样保留为事实字段；
   许可元数据被 ingest **不等于**独立批准，也不等于任何 split 被 `released`。
   本模块**不写入**任何顶层的无条件 `approved` 别名（那会绕过审核）。
"""

from __future__ import annotations

from typing import Mapping

from ..common.canonical import sha256_json
from ..common.errors import MissingInput, PolicyViolation

#: 平铺形状里可以用来声明"许可已批准"的键（任一层出现即可）。
APPROVAL_KEYS = ("authorization_scope", "license_approved", "approved", "license_status")
APPROVED_VALUES = ("approved", "train_allowed", "cleared", True)
#: 明确的**拒绝/未决**信号：任何一个出现即整体判 `unverified`。
_DENY_WORDS = ("denied", "pending", "unverified", "rejected", "forbidden", "unlicensed", "none")

# ---------------------------------------------------------------- D0 v2 契约
#: D0 把逐仓库的机器可读批准放在**嵌套**的 `license_review` 块里（🔴-A 冻结的字段名）。
LICENSE_REVIEW_KEY = "license_review"
#: 嵌套批准块里承载结论的键名（`decision`，不是 `approved`）。
LICENSE_REVIEW_DECISION_KEYS = ("decision", "license_decision")
#: 取值域（D0 `license_review_schema.decision_domain`）。
LICENSE_REVIEW_DECISIONS = ("approved", "rejected", "pending")
#: 独立复核占位块的取值域（`rejected` 是拒绝信号，`pending` 只是"还没签"）。
INDEPENDENT_REVIEW_STATUSES = ("pending", "approved", "rejected")
#: 批准块的必备字段（D0 `license_review_schema.required_fields`，11 项）。
LICENSE_REVIEW_REQUIRED_FIELDS = (
    "decision",
    "approved_spdx",
    "osi_permissive",
    "copyleft_marker_hits",
    "restrictive_marker_hits",
    "evidence",
    "decision_basis",
    "decided_by",
    "decided_at",
    "decided_against_revision",
    "independent_review",
)
#: 拒绝批准块的**冲突**信号：出现任一非空命中即"自述批准与证据冲突"→ 拒绝。
LICENSE_REVIEW_CONFLICT_KEYS = ("copyleft_marker_hits", "restrictive_marker_hits")


def _is_sha40(value) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 40 and all(char in "0123456789abcdef" for char in text)


def assess_license_review(review, pinned_commit: str | None, *, required: bool = True) -> dict:
    """判定一个 D0 `license_review` 块：**任一拒绝/冲突信号即拒绝**。

    `required=False` 表示平铺形状（`v3-source-lock/1`）的调用：那里没有批准块是合法的，
    结论由 `authorization_scope` 之类的平铺键给出，本函数只返回"未出现"。

    返回值：:

        {"present": bool, "decision": str|None, "status": "approved"|"rejected",
         "independent_review_status": str|None, "approved_spdx": str|None,
         "decided_against_revision": str|None, "revision_bound": bool|None,
         "approval_revision_matches_pin": bool|None, "problems": [str, ...]}
    """
    result = {
        "present": False,
        "decision": None,
        "status": "rejected",
        "independent_review_status": None,
        "approved_spdx": None,
        "decided_against_revision": None,
        "revision_bound": None,
        "approval_revision_matches_pin": None,
        "problems": [],
    }
    if review is None:
        if required:
            result["problems"].append("缺少 %s 批准块" % LICENSE_REVIEW_KEY)
        else:
            result["status"] = "absent"
        return result
    if not isinstance(review, Mapping):
        result["present"] = True
        result["problems"].append("%s 不是对象" % LICENSE_REVIEW_KEY)
        return result

    result["present"] = True
    missing = [field for field in LICENSE_REVIEW_REQUIRED_FIELDS if field not in review]
    if missing:
        result["problems"].append(
            "%s 缺少必备字段：%s" % (LICENSE_REVIEW_KEY, ", ".join(sorted(missing)))
        )

    decision = None
    for key in LICENSE_REVIEW_DECISION_KEYS:
        if key in review:
            decision = str(review.get(key)).strip().lower()
            break
    result["decision"] = decision
    if decision is None:
        result["problems"].append(
            "%s 里没有 %s 键（不得用顶层无条件 approved 别名代替）"
            % (LICENSE_REVIEW_KEY, LICENSE_REVIEW_DECISION_KEYS[0])
        )
    elif decision not in LICENSE_REVIEW_DECISIONS:
        result["problems"].append(
            "%s.decision=%r 不在取值域 %s 内"
            % (LICENSE_REVIEW_KEY, review.get("decision"), list(LICENSE_REVIEW_DECISIONS))
        )
    elif decision != "approved":
        result["problems"].append(
            "%s.decision=%s：拒绝/未决信号优先，不得进入训练侧"
            % (LICENSE_REVIEW_KEY, decision)
        )

    # ---- revision 绑定（approval_is_revision_bound）----
    approved_revision = review.get("decided_against_revision")
    result["decided_against_revision"] = (
        None if approved_revision is None else str(approved_revision).strip().lower()
    )
    pinned = str(pinned_commit or "").strip().lower()
    if not _is_sha40(approved_revision):
        result["revision_bound"] = False
        result["problems"].append(
            "%s.decided_against_revision=%r 不是 40 位 hex：批准未绑定到具体 revision"
            % (LICENSE_REVIEW_KEY, approved_revision)
        )
    elif not _is_sha40(pinned):
        result["revision_bound"] = False
        result["problems"].append(
            "%s 已批准 %s，但本记录的 pinned_commit=%r 不是 40 位 hex：无法核对 revision 绑定"
            % (LICENSE_REVIEW_KEY, result["decided_against_revision"], pinned_commit)
        )
    else:
        result["revision_bound"] = True
        matches = result["decided_against_revision"] == pinned
        result["approval_revision_matches_pin"] = matches
        if not matches:
            result["problems"].append(
                "%s.decided_against_revision=%s != pinned_commit=%s：批准不覆盖当前 revision，"
                "重新 pin 会让旧批准失效"
                % (LICENSE_REVIEW_KEY, result["decided_against_revision"], pinned)
            )

    # ---- 冲突信号（自述 approved 与证据矛盾）----
    for key in LICENSE_REVIEW_CONFLICT_KEYS:
        hits = review.get(key)
        if hits:
            result["problems"].append(
                "%s.%s 命中 %r：自述批准与许可证据冲突"
                % (LICENSE_REVIEW_KEY, key, list(hits)[:5])
            )
    if "osi_permissive" in review and review.get("osi_permissive") is not True:
        result["problems"].append(
            "%s.osi_permissive=%r 非真：非 OSI 宽松许可不得批准"
            % (LICENSE_REVIEW_KEY, review.get("osi_permissive"))
        )
    spdx = review.get("approved_spdx")
    result["approved_spdx"] = None if spdx is None else str(spdx).strip()
    if not result["approved_spdx"]:
        result["problems"].append("%s.approved_spdx 为空：批准必须指明许可表达式" % LICENSE_REVIEW_KEY)

    # ---- 独立复核占位（不是签字，也不阻断 ingest）----
    independent = review.get("independent_review")
    if isinstance(independent, Mapping):
        status = independent.get("status")
        status_text = None if status is None else str(status).strip().lower()
        result["independent_review_status"] = status_text
        if status_text not in INDEPENDENT_REVIEW_STATUSES:
            result["problems"].append(
                "%s.independent_review.status=%r 不在取值域 %s 内"
                % (LICENSE_REVIEW_KEY, status, list(INDEPENDENT_REVIEW_STATUSES))
            )
        elif status_text == "rejected":
            result["problems"].append(
                "%s.independent_review.status=rejected：独立复核是否定信号，优先拒绝"
                % LICENSE_REVIEW_KEY
            )
    else:
        result["problems"].append(
            "%s.independent_review 缺失或不是对象：必须保留该事实字段（pending 是合法值）"
            % LICENSE_REVIEW_KEY
        )

    result["status"] = "approved" if not result["problems"] else "rejected"
    return result


def _approval_signal(value) -> str:
    """把单个许可信号归一成 `approved` / `rejected` / `unknown`。"""
    if value is True:
        return "approved"
    if value is False or value is None:
        return "rejected"
    text = str(value).strip().lower()
    if text in ("approved", "train_allowed", "cleared"):
        return "approved"
    if text == "" or text in _DENY_WORDS:
        return "rejected"
    return "unknown"

FORMAT_D0 = "d0-source-lock/1"
FORMAT_FLAT = "v3-source-lock/1"


def _approval_of(entry: Mapping, top_level: Mapping | None = None) -> str:
    """判定记录的许可结论：**任一拒绝信号即拒绝**（Q0 Y6）。

    旧实现按键序返回首个命中的键，于是 ``{"approved": true, "license_status": "denied"}``
    被 `approved` 放行，而反向键序才给 `unverified` —— 结果取决于字典顺序，是 fail-open。

    现在：扫**全部**许可键（entry 与 top_level 两层）；
    - 出现任何 `rejected` 或 `unknown` 信号 → `unverified`；
    - 只有全部出现的信号都是 `approved` 且至少有一个 → `approved`；
    - 一个信号都没有 → `unverified`。

    🔴-A 追加：D0 v2 的结论不在平铺键上，而在嵌套的 `license_review.decision`。
    只要 entry（或 top_level）出现该块，**它自己就是唯一权威**：
    块内任何拒绝/冲突信号都让整条记录 `unverified`，平铺键上的 `approved` 不能把它翻回来
    （否则等于用一个无条件别名绕过审核）。
    """
    for source in (entry, top_level or {}):
        if isinstance(source, Mapping) and LICENSE_REVIEW_KEY in source:
            review = assess_license_review(
                source.get(LICENSE_REVIEW_KEY), source.get("pinned_commit") or source.get("commit")
            )
            return "approved" if review["status"] == "approved" else "unverified"

    signals: list[str] = []
    for source in (entry, top_level or {}):
        if not isinstance(source, Mapping):
            continue
        for key in APPROVAL_KEYS:
            if key in source:
                signals.append(_approval_signal(source[key]))
    if not signals:
        return "unverified"
    if any(signal != "approved" for signal in signals):
        return "unverified"
    return "approved"


def _normalize_entry(name: str, entry: Mapping, top_level: Mapping | None = None) -> dict:
    commit = (
        entry.get("commit")
        or entry.get("pinned_commit")
        or entry.get("base_commit")
        or ""
    )
    origin = entry.get("_origin_format", FORMAT_FLAT)
    review_block = entry.get(LICENSE_REVIEW_KEY)
    if review_block is None and isinstance(top_level, Mapping):
        review_block = top_level.get(LICENSE_REVIEW_KEY)
    # D0 形状（`repos.<name>`）**必须**带批准块：否则是"形状对的旧 manifest"，
    # 只能用平铺 authorization_scope 冒充批准 —— 这正是要禁掉的无条件别名。
    review = assess_license_review(review_block, commit, required=(origin == FORMAT_D0))
    scope = _approval_of(entry, top_level)
    if origin == FORMAT_D0 and not review["present"]:
        # 平铺别名不得翻盘：D0 记录的结论只能来自嵌套批准块。
        scope = "unverified"
    return {
        "name": name,
        "repo_url": entry.get("repo_url") or entry.get("mirror_url") or entry.get("upstream_slug") or "",
        "upstream_slug": entry.get("upstream_slug"),
        "commit": str(commit),
        "pinned_tag": entry.get("pinned_tag"),
        "tree_sha": entry.get("tree_sha"),
        "split_role": entry.get("split_role"),
        "license_spdx": entry.get("license_spdx") or entry.get("design_license_expectation"),
        "spdx_headers_found": list(entry.get("spdx_headers_found") or []),
        "acquired_at": entry.get("acquired_at") or entry.get("commit_date"),
        "authorization_scope": scope,
        "origin_format": origin,
        # 🔴-A：把 D0 批准块的**事实**原样带出来（导入 ≠ 独立批准，也 ≠ released）。
        "role_class": entry.get("role_class"),
        "license_review_present": bool(review["present"]),
        "license_decision": review["decision"],
        "license_decision_source": (
            "d0-license-review.decision" if review["present"] else "flat-approval-key"
        ),
        "license_approved_spdx": review["approved_spdx"],
        "approval_revision": review["decided_against_revision"],
        "approval_revision_matches_pin": review["approval_revision_matches_pin"],
        "independent_review_status": review["independent_review_status"],
        "license_review_problems": list(review["problems"]),
    }


def adapt(payload: Mapping) -> dict:
    """把任意支持形状的来源锁归一成 `{"format", "sources": [...]}`。"""
    if not isinstance(payload, Mapping):
        raise MissingInput("bad_source_lock", "来源锁不是对象")
    if "repos" in payload and isinstance(payload["repos"], Mapping):
        sources = [
            _normalize_entry(name, dict(entry, _origin_format=FORMAT_D0), payload)
            for name, entry in sorted(payload["repos"].items())
        ]
        fmt = FORMAT_D0
    elif "sources" in payload:
        entries = payload["sources"]
        if isinstance(entries, Mapping):
            sources = [
                _normalize_entry(name, dict(entry, _origin_format=FORMAT_FLAT), payload)
                for name, entry in sorted(entries.items())
            ]
        else:
            sources = [
                _normalize_entry(entry.get("name", "source-%d" % index), dict(entry, _origin_format=FORMAT_FLAT), payload)
                for index, entry in enumerate(entries)
            ]
        fmt = FORMAT_FLAT
    else:
        raise MissingInput(
            "unknown_source_lock_shape",
            "来源锁既没有 sources 也没有 repos：无法确定格式版本",
            keys=sorted(payload),
        )
    if not sources:
        raise MissingInput("empty_source_lock", "来源锁为空")
    return {
        "format": fmt,
        "source_lock_sha256": sha256_json(payload),
        "sources": sources,
        "generated_by": payload.get("generated_by"),
    }


def validate_sources(sources: list[Mapping]) -> list[str]:
    """校验归一后的来源：commit 必须 40 位 hex（不得猜 SHA）、许可必须 approved。

    🔴-A：D0 批准块自身的 problems 也逐条并入（revision 不符、字段缺失、
    copyleft/restrictive 冲突、independent_review.rejected），一条都不吞。
    """
    problems: list[str] = []
    for source in sources:
        name = source.get("name", "?")
        for field in ("name", "repo_url", "commit", "license_spdx", "authorization_scope"):
            if not source.get(field):
                problems.append("%s 缺少 %s" % (name, field))
        commit = str(source.get("commit", ""))
        if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
            problems.append("%s commit 不是 40 位 hex（不得猜 SHA）：%r" % (name, commit[:16]))
        if str(source.get("authorization_scope", "")).lower() not in ("approved", "train_allowed"):
            problems.append(
                "%s 许可未 approved（authorization_scope=%s，decision_source=%s）"
                % (
                    name,
                    source.get("authorization_scope"),
                    source.get("license_decision_source"),
                )
            )
            for detail in source.get("license_review_problems") or []:
                problems.append("%s：%s" % (name, detail))
    return problems


#: split_role 的合法取值（D0 台账 / Mika 2026-10-05 裁决的三值口径）。
SPLIT_ROLES = ("train", "dev", "sealed")


def assert_valid_split_roles(sources: list[Mapping]) -> None:
    """`split_role` 的**形状**校验：必需字段 + 白名单取值（不看是否是 train）。

    与 `assert_train_only` 拆开，是因为全量 source-lock 里**本来就允许** dev/sealed
    记录存在；"这份清单合规"与"这份清单只含 train"是两件事。
    """
    missing = [str(source.get("name", "?")) for source in sources if not source.get("split_role")]
    if missing:
        raise MissingInput(
            "source_record_missing_split_role",
            "来源记录缺少必需字段 split_role，不得默认当作 train 放行",
            repos=sorted(missing),
        )
    unknown = sorted(
        {
            "%s=%s" % (source.get("name", "?"), source["split_role"])
            for source in sources
            if str(source["split_role"]).strip().lower() not in SPLIT_ROLES
        }
    )
    if unknown:
        raise PolicyViolation(
            "source_record_bad_split_role",
            "split_role 不在冻结 split 口径 %s 内" % (list(SPLIT_ROLES),),
            records=unknown,
        )


def assert_train_only(sources: list[Mapping], allowed_roles=("train",)) -> None:
    """D0 的 split_role 若声明为封存/dev，不得进入训练侧 ingest。

    Q0 Y7：`split_role` 由"可选字段"改为**必需字段**。旧实现只在字段存在且非 train
    时才阻断，缺字段的记录静默通过 —— 与全仓"未填字段即 fail-closed"的原则相反。
    另外对取值做白名单校验：出现 `SPLIT_ROLES` 之外的值（例如历史别名 `test`）
    也是 fail-closed，而不是当作 train 放行。

    形状校验见 `assert_valid_split_roles`（可单独复用）。
    """
    assert_valid_split_roles(sources)
    offenders = [
        str(source.get("name", "?"))
        for source in sources
        if str(source["split_role"]).strip().lower() not in allowed_roles
    ]
    if offenders:
        raise PolicyViolation(
            "sealed_source_in_train_ingest",
            "来源锁里出现了非 train 角色的仓库，训练侧不得 ingest",
            repos=offenders,
        )


def ingest_manifest(payload: Mapping, *, train_only: bool = False) -> dict:
    """ingest 阶段的完整处理：归一 → 校验（不通过即抛异常）→ 生成 manifest 正文。

    `train_only=True` 生成**训练侧视图**：全量记录都做许可与 revision 判定
    （因此"9 条批准"这一事实仍然可见），但只有 `split_role="train"` 的记录进入
    `sources`；dev/sealed 记录列在 `excluded_non_train` 里，**没有**被 ingest。
    过滤而不是整体拒绝，是因为 D0 的 source-lock 本来就是**全量**清单，
    dev/sealed 是它的合法成员；把它们挡在训练侧视图外才是 `--train-only` 的语义。
    """
    normalized = adapt(payload)
    all_sources = normalized["sources"]
    problems = validate_sources(all_sources)
    if problems:
        raise PolicyViolation(
            "source_lock_invalid",
            "来源锁不合规（%d 项）" % len(problems),
            problems=problems,
            origin_format=normalized["format"],
        )

    excluded: list[str] = []
    sources = all_sources
    if train_only:
        # 先做 split_role 形状校验（缺字段或别名一律 fail-closed），再按 train 过滤；
        # 过滤后剩下的必然是 train，`assert_train_only` 是恒等复核（保留调用以便未来
        # allowed_roles 改动时仍然生效）。
        assert_valid_split_roles(all_sources)
        sources = [
            source
            for source in all_sources
            if str(source.get("split_role", "")).strip().lower() == "train"
        ]
        assert_train_only(sources)
        excluded = sorted(
            str(source.get("name", "?"))
            for source in all_sources
            if str(source.get("split_role", "")).strip().lower() != "train"
        )

    locked = sorted(
        str(source.get("name", "?"))
        for source in all_sources
        if str(source.get("role_class") or "locked_candidate").strip().lower() == "locked_candidate"
    )
    alternatives = sorted(
        str(source.get("name", "?"))
        for source in all_sources
        if str(source.get("role_class") or "").strip().lower() == "alternative_candidate"
    )
    decisions = {"approved": 0, "rejected": 0, "unverified": 0}
    for source in all_sources:
        scope = str(source.get("authorization_scope", "")).strip().lower()
        decisions["approved" if scope in ("approved", "train_allowed") else "unverified"] += 1

    return {
        "stage": "ingest",
        "origin_format": normalized["format"],
        "generated_by": normalized["generated_by"],
        "train_only_view": bool(train_only),
        "source_count": len(sources),
        "sources": [
            {
                key: source.get(key)
                for key in (
                    "name",
                    "repo_url",
                    "commit",
                    "pinned_tag",
                    "tree_sha",
                    "split_role",
                    "license_spdx",
                    "authorization_scope",
                    "acquired_at",
                    "role_class",
                    "license_decision",
                    "license_decision_source",
                    "license_approved_spdx",
                    "approval_revision",
                    "approval_revision_matches_pin",
                    "independent_review_status",
                )
            }
            for source in sources
        ],
        # 全量口径（与是否 train_only 无关）：这是"许可元数据被识别"的事实，不是放行结论。
        "license_assessment": {
            "records_assessed": len(all_sources),
            "decision_counts": decisions,
            "approval_revision_matches_pin": sum(
                1 for source in all_sources if source.get("approval_revision_matches_pin")
            ),
            "independent_review_pending": sum(
                1
                for source in all_sources
                if str(source.get("independent_review_status") or "").lower() == "pending"
            ),
            "independent_review_countersigned": False,
            "source": "repos.<name>.%s.decision（D0 license_review_schema）" % LICENSE_REVIEW_KEY,
        },
        # 候选池与"导入 ≠ 批准"的显式事实（Mika：候选来源被导入不等于已批准替换或发布）。
        "candidate_pools": {"locked_candidates": locked, "alternative_candidates": alternatives},
        "alternative_candidate_status": (
            "imported_only_not_approved_as_replacement_and_not_released"
        ),
        "excluded_non_train": excluded,
        "training_released": False,
        "released_splits": [],
        "license_metadata_imported_is_not_independent_approval": True,
        "source_lock_sha256": normalized["source_lock_sha256"],
        "note": (
            "本步只登记与校验，不下载权重/受限数据；未获批的来源不会被静默放行。"
            "许可元数据被导入**不等于**独立批准（independent_review 仍是 pending），"
            "也**不等于**任何 split 被 released；替代候选被导入不等于已批准替换或发布。"
        ),
    }
