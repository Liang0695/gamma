"""G13：训练 CLI 的 `--dest-dir` 解析与校验（fail-closed）。

## 缺陷出处

KAGGLE-32 v6 `GAPS-AND-OWNERS.md` §G13：

    entry.py:541  dest_dir = dest_dir or os.path.join(HERE, "_run_adapter")
    main() 的 add_argument 列表里没有 --dest-dir（entry.py:554-573）

`HERE` 是仓库内的 `v3/train/`，所以训练产物（adapter 副本、evidence JSON）只能落进
工作树 —— 而 107 作业的守卫会在提交前跑 `git status --porcelain`，写脏工作树即自拒。

## 本模块强制什么

1. `--dest-dir` 必须**显式给出**：CLI 不再有"悄悄写进仓库"的默认值；
2. 必须是**绝对路径**（相对路径的解析基准随 cwd 漂移，不能作为产物位置契约）；
3. 解析后必须落在**仓库之外**，且不得是仓库的祖先目录；
4. 必须是目录（不能是已存在的文件），父级必须存在且可写；
5. 可写性是**探针实测**（真的建目录 + 写一个探针文件再删），不是 `os.access` 的推测。

`resolve_dest_dir()` 返回的 `resolved` 是**唯一**后续使用的路径：evidence 输出、
adapter 保存、adapter 重载都必须用它，见 :func:`verify_paths_consistent`。

不做的事：不创建仓库内目录、不把相对路径"帮忙"补成绝对路径、不在校验失败时回退默认值。
"""

from __future__ import annotations

import os
import tempfile
from typing import Iterable, Mapping

from ..common.errors import MissingInput, PolicyViolation

#: 探针文件名（写入后立即删除；失败即视为不可写）。
PROBE_FILENAME = ".v3_dest_dir_probe"

#: 默认产物目录名（仅在调用方显式给出父目录时用于拼接，**不**作为仓库内默认值）。
DEFAULT_RUN_DIRNAME = "_run_adapter"


def repo_root() -> str:
    """本仓库根目录（`v3/train/destdir.py` 上溯三级）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", ".."))


def _norm(path: str) -> str:
    """统一成 realpath + 去尾部分隔符，便于做"是否在仓库内"的判定。"""
    return os.path.realpath(os.path.abspath(str(path)))


def _is_within(child: str, parent: str) -> bool:
    """`child` 是否等于 `parent` 或位于其下（大小写与分隔符差异由 _norm 归一）。"""
    child_n = _norm(child).rstrip("\\/")
    parent_n = _norm(parent).rstrip("\\/")
    if not parent_n:
        return True
    if child_n == parent_n:
        return True
    return child_n.startswith(parent_n + os.sep)


def _probe_writable(directory: str) -> None:
    """真的写一个探针文件来判定可写性；不可写即抛 `PolicyViolation`。

    不用 `os.access`：在 Windows 上它对只读目录会给出乐观答案，而"能不能写"
    是这条契约的全部意义。
    """
    probe = os.path.join(directory, PROBE_FILENAME)
    try:
        with open(probe, "wb") as handle:
            handle.write(b"v3-dest-dir-probe")
        os.remove(probe)
    except OSError as exc:
        raise PolicyViolation(
            "dest_dir_not_writable",
            "目标目录不可写（探针写入失败）：%s" % directory,
            directory=directory,
            error=str(exc),
        ) from exc


def resolve_dest_dir(
    requested: str | None,
    *,
    repo_root_path: str | None = None,
    must_be_outside_repo: bool = True,
    create: bool = True,
) -> dict:
    """解析并校验 `--dest-dir`；任何不合规都 fail-closed。

    返回::

        {"requested": str, "resolved": str, "created": bool,
         "outside_repo": bool, "repo_root": str, "parent_existing": str,
         "writable_probe": "ok", "checks": [str, ...]}
    """
    if requested is None or not str(requested).strip():
        raise MissingInput(
            "dest_dir_missing",
            "必须显式给出 --dest-dir（绝对路径、仓库外）：入口不再默认把产物写进仓库",
        )
    text = str(requested)
    if "\0" in text:
        raise PolicyViolation("dest_dir_invalid_chars", "路径含 NUL 字符：拒绝", requested=text)
    if not os.path.isabs(text):
        raise PolicyViolation(
            "dest_dir_not_absolute",
            "dest_dir 必须是绝对路径（相对路径会随 cwd 漂移，不能作为产物位置契约）：%r" % text,
            requested=text,
        )

    checks: list[str] = ["absolute"]
    root = _norm(repo_root_path or repo_root())
    resolved = _norm(text)

    # 文件系统根（如 L:\ 或 /）不接受：产物会被摊到盘根，且极易成为仓库祖先。
    if resolved.rstrip("\\/") == os.path.splitdrive(resolved)[0].rstrip("\\/"):
        raise PolicyViolation(
            "dest_dir_is_filesystem_root",
            "dest_dir 不能是文件系统根目录：%s" % resolved,
            requested=text,
        )

    if os.path.exists(resolved) and not os.path.isdir(resolved):
        raise PolicyViolation(
            "dest_dir_is_file", "dest_dir 已存在且不是目录：%s" % resolved, requested=text
        )
    checks.append("not_a_file")

    inside_repo = _is_within(resolved, root)
    contains_repo = _is_within(root, resolved) and not inside_repo
    if must_be_outside_repo:
        if inside_repo:
            raise PolicyViolation(
                "dest_dir_inside_repo",
                "dest_dir 落在仓库内（会写脏工作树，提交前守卫会自拒）：%s（仓库 %s）"
                % (resolved, root),
                requested=text,
                resolved=resolved,
                repo_root=root,
            )
        if contains_repo:
            raise PolicyViolation(
                "dest_dir_contains_repo",
                "dest_dir 是仓库的祖先目录，拒绝把产物与仓库混在同一层：%s" % resolved,
                requested=text,
                resolved=resolved,
                repo_root=root,
            )
        checks.append("outside_repo")

    # 找到最深的已存在祖先，作为父级存在性证据。
    cursor = resolved
    existing_parent = None
    while True:
        parent = os.path.dirname(cursor.rstrip("\\/"))
        if parent == cursor:  # 到根
            break
        if os.path.isdir(parent):
            existing_parent = parent
            break
        cursor = parent
    if existing_parent is None:
        raise MissingInput(
            "dest_dir_parent_missing",
            "dest_dir 的父级不存在，且无法定位任何已存在祖先：%s" % resolved,
            requested=text,
        )
    checks.append("parent_exists")

    created = False
    if not os.path.isdir(resolved):
        if not create:
            raise MissingInput(
                "dest_dir_missing", "dest_dir 不存在且未允许创建：%s" % resolved, requested=text
            )
        try:
            os.makedirs(resolved, exist_ok=True)
        except OSError as exc:
            raise PolicyViolation(
                "dest_dir_create_failed",
                "无法创建 dest_dir：%s（%s）" % (resolved, exc),
                requested=text,
            ) from exc
        created = True
    checks.append("directory_present")

    _probe_writable(resolved)
    checks.append("writable_probe_ok")

    return {
        "requested": text,
        "resolved": resolved,
        "created": created,
        "outside_repo": not inside_repo,
        "contains_repo": contains_repo,
        "repo_root": root,
        "parent_existing": existing_parent,
        "writable_probe": "ok",
        "checks": checks,
    }


def verify_paths_consistent(
    resolution: Mapping,
    paths: Iterable[tuple[str, str | None]],
) -> dict:
    """核对"输出 / 保存 / 重载"三类路径都落在同一个已解析的 dest_dir 之下。

    `paths` 是 `(label, path)` 序列。任一缺失或越界即 `PolicyViolation`
    （`dest_dir_path_inconsistent`），不使用"取第一个存在的"这类含糊回退。
    """
    resolved = _norm(str(resolution.get("resolved") or ""))
    if not resolved:
        raise MissingInput("dest_dir_resolution_missing", "缺少 dest_dir 解析结果")
    checked: dict[str, str] = {}
    outside: dict[str, str] = {}
    for label, path in paths:
        if not path:
            outside[str(label)] = "<缺失>"
            continue
        full = _norm(str(path))
        if _is_within(full, resolved):
            checked[str(label)] = full
        else:
            outside[str(label)] = full
    if outside:
        raise PolicyViolation(
            "dest_dir_path_inconsistent",
            "以下路径不在已解析的 dest_dir 之下（输出/保存/重载必须同源）：%s" % sorted(outside),
            dest_dir=resolved,
            outside=outside,
        )
    return {
        "dest_dir": resolved,
        "checked": checked,
        "path_count": len(checked),
        "consistent": True,
    }


def make_probe_dir(*, prefix: str = "v3-destdir-") -> str:
    """测试/预检用的合规仓库外临时目录。

    优先用**仓库的父目录**（在 107 上就是 `$HOME/...`，在本地就是工作区根），
    这样既满足"仓库外"约束，又不依赖 `%TEMP%` 是否可写；两者都失败时退回系统临时目录。
    """
    parent = os.path.dirname(repo_root())
    if os.path.isdir(parent):
        try:
            return tempfile.mkdtemp(prefix=prefix, dir=parent)
        except OSError:
            pass
    return tempfile.mkdtemp(prefix=prefix)
