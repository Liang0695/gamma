"""来源锁适配器：让 D0（KAGGLE-23）的版本化 manifest 能被 ingest 直接消费。

动机：设计稿要求"与数据生产者通过版本化 manifest 交接"，但 D0 的
`d0/out/source-lock.json` 与代码最初假设的形状不同：

- 本仓库最初假设：`{"sources": [{"name", "repo_url", "commit", "license_spdx",
  "authorization_scope", "acquired_at"}]}`；
- D0 实际产出：`{"repos": {"<name>": {"upstream_slug", "mirror_url", "pinned_tag",
  "pinned_commit", "tree_sha", "split_role", "design_license_expectation",
  "spdx_headers_found", "packaging_metadata"}}}`。

本模块把两种形状都归一成同一份内部结构，并且**不猜许可结论**：
D0 的 manifest 里没有逐仓库的人工批准标志，因此一律标 `authorization_scope="unverified"`，
`ingest` 由此 fail-closed —— 这正是"许可未闭合不得进入 released"的要求。
如果 D0 后续加入显式批准字段，只要在 `APPROVAL_KEYS` 里出现即被采纳。
"""

from __future__ import annotations

from typing import Mapping

from ..common.canonical import sha256_json
from ..common.errors import MissingInput, PolicyViolation

#: D0 manifest 里可以用来声明"许可已批准"的键（任一层出现即可）。
APPROVAL_KEYS = ("authorization_scope", "license_approved", "approved", "license_status")
APPROVED_VALUES = ("approved", "train_allowed", "cleared", True)
#: 明确的**拒绝/未决**信号：任何一个出现即整体判 `unverified`。
_DENY_WORDS = ("denied", "pending", "unverified", "rejected", "forbidden", "unlicensed", "none")


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
    """
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
        "authorization_scope": _approval_of(entry, top_level),
        "origin_format": entry.get("_origin_format", FORMAT_FLAT),
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
    """校验归一后的来源：commit 必须 40 位 hex（不得猜 SHA）、许可必须 approved。"""
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
                "%s 许可未 approved（authorization_scope=%s）"
                % (name, source.get("authorization_scope"))
            )
    return problems


#: split_role 的合法取值（D0 台账 / Mika 2026-10-05 裁决的三值口径）。
SPLIT_ROLES = ("train", "dev", "sealed")


def assert_train_only(sources: list[Mapping], allowed_roles=("train",)) -> None:
    """D0 的 split_role 若声明为封存/dev，不得进入训练侧 ingest。

    Q0 Y7：`split_role` 由"可选字段"改为**必需字段**。旧实现只在字段存在且非 train
    时才阻断，缺字段的记录静默通过 —— 与全仓"未填字段即 fail-closed"的原则相反。
    另外对取值做白名单校验：出现 `SPLIT_ROLES` 之外的值（例如历史别名 `test`）
    也是 fail-closed，而不是当作 train 放行。
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


def ingest_manifest(payload: Mapping) -> dict:
    """ingest 阶段的完整处理：归一 → 校验（不通过即抛异常）→ 生成 manifest 正文。"""
    normalized = adapt(payload)
    problems = validate_sources(normalized["sources"])
    if problems:
        raise PolicyViolation(
            "source_lock_invalid",
            "来源锁不合规（%d 项）" % len(problems),
            problems=problems,
            origin_format=normalized["format"],
        )
    return {
        "stage": "ingest",
        "origin_format": normalized["format"],
        "generated_by": normalized["generated_by"],
        "source_count": len(normalized["sources"]),
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
                )
            }
            for source in normalized["sources"]
        ],
        "source_lock_sha256": normalized["source_lock_sha256"],
        "note": "本步只登记与校验，不下载权重/受限数据；未获批的来源不会被静默放行。",
    }
