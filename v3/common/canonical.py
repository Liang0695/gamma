"""canonical JSON、SHA256 与 manifest 生成（KAGGLE-21 §8「版本」条款的实现）。

规则（逐条对应设计稿）：
- UTF-8 无 BOM、LF 换行；
- canonical JSON：键排序、紧凑分隔符、ensure_ascii=false、禁止 NaN；
- 文本末尾恰好一个真实 LF；
- 输入二进制按原字节 SHA256；
- 源码 tree hash 对排序的 (relative_path, mode, file_sha256) canonical manifest 计算；
- 拒绝 symlink 与越界路径。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from typing import Iterable, Mapping

from .errors import IntegrityError, PolicyViolation

CANONICAL_SEPARATORS = (",", ":")


def _reject_nonfinite(obj) -> None:
    """拒绝 NaN / Infinity（canonical JSON 不允许）。"""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            raise IntegrityError("nonfinite_number", "canonical JSON 禁止 NaN/Infinity")
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            _reject_nonfinite(key)
            _reject_nonfinite(value)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _reject_nonfinite(value)


def canonical_json_bytes(obj) -> bytes:
    """返回 canonical JSON 的 UTF-8 字节（末尾带一个真实 LF，无 BOM）。"""
    _reject_nonfinite(obj)
    text = json.dumps(
        obj,
        sort_keys=True,
        separators=CANONICAL_SEPARATORS,
        ensure_ascii=False,
        allow_nan=False,
    )
    return (text + "\n").encode("utf-8")


def canonical_text_bytes(text: str) -> bytes:
    """规范化文本：LF 换行、UTF-8 无 BOM、末尾恰好一个 LF。"""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = normalized.rstrip("\n") + "\n"
    return normalized.encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(canonical_text_bytes(text))


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


def _safe_relative_path(root: str, path: str) -> str:
    """把绝对路径转成 root 下的相对路径，拒绝越界与 symlink。"""
    real_root = os.path.realpath(root)
    real_path = os.path.realpath(path)
    if os.path.islink(path):
        raise PolicyViolation("symlink_rejected", "源码树禁止 symlink：%s" % path)
    rel = os.path.relpath(real_path, real_root)
    if rel.startswith("..") or os.path.isabs(rel):
        raise PolicyViolation("path_escape", "路径越出源码树根：%s" % path)
    return rel.replace(os.sep, "/")


def tree_manifest(root: str, suffixes: Iterable[str] | None = None) -> list[dict]:
    """生成排序后的 (relative_path, mode, file_sha256) 清单。"""
    entries: list[dict] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in (".git", "__pycache__"))
        for name in sorted(filenames):
            full = os.path.join(dirpath, name)
            if os.path.islink(full):
                raise PolicyViolation("symlink_rejected", "源码树禁止 symlink：%s" % full)
            if suffixes is not None and not name.endswith(tuple(suffixes)):
                continue
            rel = _safe_relative_path(root, full)
            mode = "100755" if os.access(full, os.X_OK) else "100644"
            entries.append(
                {"relative_path": rel, "mode": mode, "file_sha256": sha256_file(full)}
            )
    entries.sort(key=lambda item: item["relative_path"].encode("utf-8"))
    return entries


def tree_hash(root: str, suffixes: Iterable[str] | None = None) -> str:
    """对 tree manifest 求 canonical hash —— 本地微索引的失效绑定依据。"""
    manifest = tree_manifest(root, suffixes=suffixes)
    return sha256_json({"entries": manifest})


def problem_family_id(origin_ids: Iterable[str]) -> str:
    """problem_family_id = SHA256(canonical(sorted origin IDs))（KAGGLE-21 §8）。"""
    canonical = sorted({str(item) for item in origin_ids})
    return sha256_bytes(canonical_json_bytes(canonical))


def sorted_by_key(items: Iterable[dict], key: str) -> list[dict]:
    """数组排序规则：按给定键的 UTF-8 字节序（task_id / trace_id / test IDs）。"""
    return sorted(items, key=lambda item: str(item[key]).encode("utf-8"))


def write_canonical(path: str, obj) -> str:
    """写 canonical JSON 文件并返回其 SHA256（用于各类 manifest）。"""
    payload = canonical_json_bytes(obj)
    with open(path, "wb") as handle:
        handle.write(payload)
    return sha256_bytes(payload)
