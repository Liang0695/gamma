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

FORMAT_D0 = "d0-source-lock/1"
FORMAT_FLAT = "v3-source-lock/1"


def _approval_of(entry: Mapping, top_level: Mapping | None = None) -> str:
    for source in (entry, top_level or {}):
        for key in APPROVAL_KEYS:
            if key in source:
                value = source[key]
                if value in APPROVED_VALUES or str(value).lower() in (
                    "approved",
                    "train_allowed",
                    "cleared",
                ):
                    return "approved"
                if value is False or str(value).lower() in ("pending", "unverified", "denied"):
                    return "unverified"
    return "unverified"


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


def assert_train_only(sources: list[Mapping], allowed_roles=("train",)) -> None:
    """D0 的 split_role 若声明为封存/dev，不得进入训练侧 ingest。"""
    offenders = [
        source["name"]
        for source in sources
        if source.get("split_role") and source["split_role"] not in allowed_roles
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
