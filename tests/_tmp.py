"""测试用的临时目录：**必须落在工作区内**。

两个本机约束：
1. 沙箱禁止写系统临时目录（例如用户 Temp 目录）；
2. `tempfile.mkdtemp` 创建的目录带受限 ACL，子进程读不到 → 写入报 Permission denied。

所以这里自己用 `os.makedirs` + 递增编号建目录，退出时显式清理。
"""

from __future__ import annotations

import contextlib
import itertools
import os
import shutil

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TMP_ROOT = os.path.join(REPO_ROOT, "_tmp")

_counter = itertools.count(1)


@contextlib.contextmanager
def temp_dir(prefix: str = "case_"):
    os.makedirs(TMP_ROOT, exist_ok=True)
    path = os.path.join(TMP_ROOT, "%s%d" % (prefix, next(_counter)))
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


#: 仓库的**父目录**：G13 要求训练产物目录必须在仓库外，测试需要一条合规路径。
#: 在 107 上这就是 `$HOME`（仓库是 `$HOME/gamma`），本地是工作区根。
OUTSIDE_ROOT = os.path.dirname(REPO_ROOT)


@contextlib.contextmanager
def temp_dir_outside_repo(prefix: str = "outside_"):
    """仓库**外**的临时目录（G13 契约要求：`--dest-dir` 不得落在仓库内）。

    仍然不用 `tempfile.mkdtemp`：它的受限 ACL 会让子进程读不到（见模块开头说明）。
    """
    os.makedirs(OUTSIDE_ROOT, exist_ok=True)
    path = os.path.join(OUTSIDE_ROOT, "%s%d" % (prefix, next(_counter)))
    os.makedirs(path, exist_ok=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)
