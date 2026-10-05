"""去重与家族连通分量（KAGGLE-21 §8「去重」与「问题家族」）。

- statement：NFC + LF + 空白归一的精确 SHA；
- patch/source：token 归一（去注释、路径替换）后取 5-token shingles 的 MinHash，
  Jaccard ≥0.80 或 statement 近似相似度 ≥0.90 触发人工审查（阈值预注册）；
- 问题家族：联合 issue/PR/backport/共同 parent/同根因等关系建无向图取连通分量，
  保守合并（排除不了的疑点就合并）；
- `problem_family_id = SHA256(canonical(sorted origin IDs))`。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import Iterable, Mapping, Sequence

from ..common.canonical import problem_family_id, sha256_text
from ..common.errors import PolicyViolation

SHINGLE_TOKENS = 5
JACCARD_TRIGGER = 0.80
STATEMENT_SIMILARITY_TRIGGER = 0.90
MINHASH_PERMUTATIONS = 64

_WS = re.compile(r"\s+")
_COMMENT = re.compile(r"(?m)^\s*#.*$")
_PATHLIKE = re.compile(r"[\w./-]*/[\w./-]+")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def normalize_statement(text: str) -> str:
    """NFC、LF、空白归一。"""
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    normalized = "\n".join(_WS.sub(" ", line).strip() for line in normalized.split("\n"))
    return normalized.strip()


def statement_hash(text: str) -> str:
    return sha256_text(normalize_statement(text))


def normalize_code(text: str) -> str:
    """token 归一：去注释、路径替换成占位符、标识符小写化。"""
    without_comments = _COMMENT.sub("", text)
    without_paths = _PATHLIKE.sub("<path>", without_comments)
    return _WS.sub(" ", without_paths).strip().lower()


def shingles(tokens: Sequence[str], width: int = SHINGLE_TOKENS) -> set[str]:
    if len(tokens) < width:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + width]) for i in range(len(tokens) - width + 1)}


def _hash_int(value: str, salt: int) -> int:
    digest = hashlib.blake2b(("%d|%s" % (salt, value)).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def minhash_signature(shingle_set: Iterable[str], permutations: int = MINHASH_PERMUTATIONS) -> tuple[int, ...]:
    values = sorted(shingle_set)
    if not values:
        return tuple([0] * permutations)
    return tuple(min(_hash_int(value, salt) for value in values) for salt in range(permutations))


def minhash_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    return sum(1 for a, b in zip(left, right) if a == b) / float(len(left))


def code_similarity(left_text: str, right_text: str) -> float:
    left = minhash_signature(shingles(_IDENT.findall(normalize_code(left_text))))
    right = minhash_signature(shingles(_IDENT.findall(normalize_code(right_text))))
    return minhash_similarity(left, right)


def statement_similarity(left: str, right: str) -> float:
    """词袋 Jaccard 近似（statement 近似相似度口径）。"""
    left_tokens = set(_IDENT.findall(normalize_statement(left).lower()))
    right_tokens = set(_IDENT.findall(normalize_statement(right).lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / float(len(left_tokens | right_tokens))


def dedup_report(records: Sequence[Mapping]) -> dict:
    """扫描重复：精确 statement hash、近似 statement、代码相似度。"""
    exact: dict[str, list[str]] = {}
    for record in records:
        key = statement_hash(record["problem_statement"])
        exact.setdefault(key, []).append(record["task_id"])
    duplicates = {key: ids for key, ids in exact.items() if len(ids) > 1}

    near: list[dict] = []
    for i, left in enumerate(records):
        for right in records[i + 1 :]:
            similarity = statement_similarity(left["problem_statement"], right["problem_statement"])
            if similarity >= STATEMENT_SIMILARITY_TRIGGER:
                near.append(
                    {
                        "left": left["task_id"],
                        "right": right["task_id"],
                        "statement_similarity": round(similarity, 4),
                        "trigger": "statement_similarity>=%.2f" % STATEMENT_SIMILARITY_TRIGGER,
                    }
                )
            if "patch_text" in left and "patch_text" in right:
                code_sim = code_similarity(left["patch_text"], right["patch_text"])
                if code_sim >= JACCARD_TRIGGER:
                    near.append(
                        {
                            "left": left["task_id"],
                            "right": right["task_id"],
                            "code_similarity": round(code_sim, 4),
                            "trigger": "minhash_jaccard>=%.2f" % JACCARD_TRIGGER,
                        }
                    )
    return {
        "exact_duplicates": duplicates,
        "near_duplicates": near,
        "thresholds": {
            "shingle_tokens": SHINGLE_TOKENS,
            "minhash_jaccard": JACCARD_TRIGGER,
            "statement_similarity": STATEMENT_SIMILARITY_TRIGGER,
            "permutations": MINHASH_PERMUTATIONS,
        },
        "status": "review_required" if (duplicates or near) else "clean",
        "note": "低于阈值不证明无污染；可疑跨 split 匹配应整族隔离。",
    }


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, item: str) -> str:
        self.parent.setdefault(item, item)
        root = item
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[item] != root:
            self.parent[item], item = root, self.parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        lroot, rroot = self.find(left), self.find(right)
        if lroot != rroot:
            self.parent[rroot] = lroot


def family_components(edges: Sequence[Sequence[str]]) -> dict[str, str]:
    """由关系边求连通分量，返回 origin_id → problem_family_id。

    孤立节点（只出现在单元素边里）也要有自己的家族，因此先登记全部节点。
    """
    union = _UnionFind()
    for edge in edges:
        for node in edge:
            union.find(node)
        if len(edge) < 2:
            continue
        first = edge[0]
        for other in edge[1:]:
            union.union(first, other)
    groups: dict[str, list[str]] = {}
    for node in union.parent:
        groups.setdefault(union.find(node), []).append(node)
    mapping: dict[str, str] = {}
    for members in groups.values():
        family = problem_family_id(members)
        for member in members:
            mapping[member] = family
    return mapping


def assert_no_split_leak(records: Sequence[Mapping], denylist: Iterable[str]) -> None:
    """V2 D/H 家族与官方四仓库不得进入 V3 train/dev。"""
    denied = {str(item) for item in denylist}
    problems = []
    for record in records:
        if record.get("split") in ("train", "dev"):
            if str(record.get("problem_family_id")) in denied:
                problems.append(record["task_id"])
            if str(record.get("repo_family")) in denied:
                problems.append(record["task_id"])
    if problems:
        raise PolicyViolation(
            "v2_denylist_intersection",
            "V3 train/dev 与 V2 排除清单相交，必须整族隔离",
            tasks=sorted(set(problems)),
        )


def deterministically_select(records: Sequence[Mapping], quota: Mapping[str, int], salt: str = "KAGGLE-21-V3-data-v1") -> list[Mapping]:
    """按 SHA256(salt|repo_family|problem_family_id) 排序取首额定数，每家族 1 题。"""
    scored = []
    for record in records:
        key = "%s|%s|%s" % (salt, record.get("repo_family"), record.get("problem_family_id"))
        scored.append((hashlib.sha256(key.encode("utf-8")).hexdigest(), record))
    scored.sort(key=lambda item: item[0])
    selected: list[Mapping] = []
    used_families: set[str] = set()
    counts: dict[str, int] = {}
    for _, record in scored:
        repo = str(record.get("repo_family"))
        limit = int(quota.get(repo, quota.get("*", 0)))
        family = str(record.get("problem_family_id"))
        if family in used_families:
            continue
        if counts.get(repo, 0) >= limit:
            continue
        used_families.add(family)
        counts[repo] = counts.get(repo, 0) + 1
        selected.append(record)
    return selected


def selection_manifest(selected: Sequence[Mapping], salt: str = "KAGGLE-21-V3-data-v1") -> dict:
    return {
        "salt": salt,
        "selected": [record["task_id"] for record in selected],
        "by_repo": {
            repo: sum(1 for record in selected if record.get("repo_family") == repo)
            for repo in sorted({str(record.get("repo_family")) for record in selected})
        },
        "family_count": len({record.get("problem_family_id") for record in selected}),
        "note": "实际配额不够时发布 shortfall，不跨 split 补齐。",
    }
