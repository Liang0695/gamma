"""任务内微型索引：绑定 tree hash、失效重建、禁止写进补丁目录。

KAGGLE-19 §4 与 KAGGLE-22 §5：
- 官方预建图只描述 base snapshot；**本地微索引绑定 tree hash，源码修改后失效重建**；
- 旧图只作线索，编辑前重新读源码；
- 索引中间产物一律写 `/tmp`，写 `/workspace` 会污染 `submit_patch()` 的 `git diff`；
- 每题清空，跨任务全部失效。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..common.canonical import sha256_json, tree_hash
from ..common.errors import IntegrityError, PolicyViolation
from .outline import outline_from_source

#: 一切中间产物只允许写这里（官方容器每任务清空）。
SAFE_INDEX_DIR = "/tmp"


@dataclass
class IndexEntry:
    symbol: str
    path: str
    start_line: int
    end_line: int
    kind: str

    def render(self) -> str:
        return "%s\t%s:%d-%d\t%s" % (self.symbol, self.path, self.start_line, self.end_line, self.kind)


@dataclass
class MicroIndex:
    """绑定 (repo, tree_sha) 的符号索引。"""

    repo: str
    tree_sha256: str
    entries: list[IndexEntry] = field(default_factory=list)

    @classmethod
    def build(cls, repo: str, tree_sha256: str, files: Mapping[str, str]) -> "MicroIndex":
        entries: list[IndexEntry] = []
        for path in sorted(files, key=lambda p: p.encode("utf-8")):
            if not path.endswith((".py", ".pyi")):
                continue
            for symbol in outline_from_source(files[path], module=path):
                entries.append(
                    IndexEntry(
                        symbol=symbol.qualname,
                        path=path,
                        start_line=symbol.start_line,
                        end_line=symbol.end_line,
                        kind=symbol.kind,
                    )
                )
        return cls(repo=repo, tree_sha256=tree_sha256, entries=entries)

    @property
    def key(self) -> str:
        return sha256_json({"repo": self.repo, "tree_sha256": self.tree_sha256})

    def is_valid_for(self, current_tree_sha256: str) -> bool:
        """源码修改后 tree hash 变化 → 索引失效，必须重建。"""
        return self.tree_sha256 == current_tree_sha256

    def assert_valid_for(self, current_tree_sha256: str) -> None:
        if not self.is_valid_for(current_tree_sha256):
            raise IntegrityError(
                "index_stale",
                "微索引绑定的是 %s，当前 tree 是 %s：必须重建后再用于编辑决策"
                % (self.tree_sha256[:12], current_tree_sha256[:12]),
            )

    def lookup(self, symbol: str) -> list[IndexEntry]:
        """精确限定名优先，其次短名后缀匹配（短名可能多义）。"""
        exact = [e for e in self.entries if e.symbol == symbol]
        if exact:
            return exact
        return [e for e in self.entries if e.symbol.endswith("." + symbol) or e.symbol == symbol]

    def render(self, limit: int = 2000) -> str:
        lines = [e.render() for e in self.entries[:limit]]
        if len(self.entries) > limit:
            lines.append("[truncated: %d/%d]" % (limit, len(self.entries)))
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "repo": self.repo,
            "tree_sha256": self.tree_sha256,
            "entry_count": len(self.entries),
            "index_key": self.key,
        }


def assert_safe_index_target(target_path: str, workspace_root: str = "/workspace") -> None:
    """索引文件不得写进 workspace（会污染 submit_patch 的 diff）。"""
    normalized = os.path.normpath(target_path).replace("\\", "/")
    root = os.path.normpath(workspace_root).replace("\\", "/").rstrip("/")
    if normalized == root or normalized.startswith(root + "/"):
        raise PolicyViolation(
            "index_in_workspace",
            "索引/临时产物禁止写入 %s（必须写 %s）" % (workspace_root, SAFE_INDEX_DIR),
            target=normalized,
        )
    if not normalized.startswith(SAFE_INDEX_DIR.rstrip("/") + "/"):
        # 非 workspace 的其它路径（例如本机测试用的临时目录）允许，但必须显式声明。
        return


def build_index_command(snapshot_dir: str, out_path: str = "/tmp/idx.txt") -> str:
    """可复制的一行命令：AST 遍历输出 `symbol\\tpath:line` 到 /tmp。"""
    assert_safe_index_target(out_path)
    # 这里刻意不用 % 格式化：内嵌脚本本身含有 % 占位符。
    script = (
        "python3 - \"__ROOT__\" \"__OUT__\" <<'PYEOF'\n"
        "import ast, os, sys\n"
        "root, out = sys.argv[1], sys.argv[2]\n"
        "rows = []\n"
        "for dirpath, dirnames, filenames in os.walk(root):\n"
        "    dirnames[:] = [d for d in dirnames if d not in ('.git', '__pycache__')]\n"
        "    for name in sorted(filenames):\n"
        "        if not name.endswith(('.py', '.pyi')):\n"
        "            continue\n"
        "        full = os.path.join(dirpath, name)\n"
        "        rel = os.path.relpath(full, root).replace(os.sep, '/')\n"
        "        try:\n"
        "            with open(full, 'r', encoding='utf-8', errors='replace') as handle:\n"
        "                tree = ast.parse(handle.read())\n"
        "        except SyntaxError:\n"
        "            continue\n"
        "        stack = [('', tree)]\n"
        "        while stack:\n"
        "            prefix, node = stack.pop()\n"
        "            for child in ast.iter_child_nodes(node):\n"
        "                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):\n"
        "                    qual = (prefix + '.' + child.name) if prefix else child.name\n"
        "                    rows.append('{}\\t{}:{}\\t{}'.format(qual, rel, child.lineno, type(child).__name__))\n"
        "                    stack.append((qual, child))\n"
        "with open(out, 'w', encoding='utf-8') as handle:\n"
        "    handle.write('\\n'.join(rows))\n"
        "PYEOF"
    )
    return script.replace("__ROOT__", snapshot_dir).replace("__OUT__", out_path)


class IndexRegistry:
    """按 (repo, tree_sha) 缓存微索引；tree 变化即视为未命中并要求重建。"""

    def __init__(self) -> None:
        self._store: dict[str, MicroIndex] = {}

    def get(self, repo: str, tree_sha256: str) -> MicroIndex | None:
        index = self._store.get(repo)
        if index is None:
            return None
        if not index.is_valid_for(tree_sha256):
            # 失效即丢弃，绝不返回过期索引。
            self._store.pop(repo, None)
            return None
        return index

    def put(self, index: MicroIndex) -> None:
        self._store[index.repo] = index

    def get_or_build(self, repo: str, tree_sha256: str, files: Mapping[str, str]) -> MicroIndex:
        cached = self.get(repo, tree_sha256)
        if cached is not None:
            return cached
        built = MicroIndex.build(repo, tree_sha256, files)
        self.put(built)
        return built

    def invalidate(self, repo: str) -> None:
        self._store.pop(repo, None)

    def stats(self) -> dict:
        return {repo: index.to_dict() for repo, index in sorted(self._store.items())}


def tree_sha_for_files(files: Mapping[str, str]) -> str:
    """对内存文件集合求 tree hash（与磁盘 tree_hash 同构，便于测试）。"""
    entries = [
        {"relative_path": path, "file_sha256": sha256_json({"content": text})}
        for path, text in sorted(files.items(), key=lambda kv: kv[0].encode("utf-8"))
    ]
    return sha256_json({"entries": entries})


def disk_tree_sha(root: str, suffixes: Iterable[str] | None = (".py", ".pyi")) -> str:
    return tree_hash(root, suffixes=suffixes)
