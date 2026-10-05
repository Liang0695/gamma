"""测试用的临时目录：**必须落在工作区内**。

两个本机约束：
1. 沙箱禁止写系统临时目录（`C:\\Users\\...\\Temp`）；
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
