"""T0：精确依赖锁（fail-closed）。

对应 KAGGLE-20 §5「依赖」与 KAGGLE-19 §范围「T0 精确依赖锁」：

- 训练软件与比赛推理软件**分别锁定**，不得为了训练升级正式评分器；
- 候选起点（PyTorch 2.10.0 / Transformers 5.17.0 / PEFT 0.21.0）本轮**未安装验证**，
  因此其 `verified` 必须是 false，runner 必须因此阻断；
- 完整 freeze / wheel SHA / GPU 驱动 / 官方包实际版本是开训前产物；
- **未填字段使 runner fail-closed，不能自动安装 latest。**

本模块只读写锁文件本身，不联网、不安装任何东西。
"""

from __future__ import annotations

import json
import os
from typing import Mapping

from ..common.canonical import sha256_file, sha256_json
from ..common.errors import MissingInput, PolicyViolation, UnverifiedLock, reject_unpinned

#: 两个互相隔离的软件面：训练面 vs 比赛推理面。严禁共用同一把锁。
TRAIN_CHANNEL = "train"
SERVING_CHANNEL = "serving"

REQUIRED_PACKAGE_FIELDS = ("version", "wheel_sha256", "source")
REQUIRED_CHANNEL_FIELDS = ("python", "packages", "verified")


class DependencyLock:
    """一把精确依赖锁。`verify()` 不通过时任何训练入口都必须停止。"""

    def __init__(self, channel: str, payload: Mapping, path: str | None = None) -> None:
        if channel not in (TRAIN_CHANNEL, SERVING_CHANNEL):
            raise PolicyViolation("bad_channel", "未知软件通道：%r" % (channel,))
        self.channel = channel
        self.payload = dict(payload)
        #: 锁文件来源路径（内存构造时为 None）。隔离判定需要它识别"两份锁是同一文件"。
        self.path = os.path.abspath(path) if path else None

    # ---- 读取 ----

    @classmethod
    def from_file(cls, path: str) -> "DependencyLock":
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        for field in ("channel", "python", "packages", "verified", "notes"):
            if field not in raw:
                raise MissingInput("lock_missing_field", "锁文件缺少字段 %r：%s" % (field, path))
        return cls(raw["channel"], raw, path=path)

    def lock_sha256(self) -> str:
        """锁内容自身的哈希，写进各类 manifest。"""
        return sha256_json(self.payload)

    def source_sha256(self) -> str | None:
        """锁文件**原始字节**的 SHA-256；拿不到来源路径时为 None。"""
        if not self.path or not os.path.exists(self.path):
            return None
        return sha256_file(self.path)

    def normalized_path(self) -> str | None:
        """归一化绝对路径（realpath 解析相对路径与软链）。"""
        if not self.path:
            return None
        return os.path.normcase(os.path.realpath(self.path))

    def package_versions(self) -> dict:
        """包名 → 版本 的扁平映射（缺失版本记为空串）。"""
        packages = self.payload.get("packages") or {}
        versions = {}
        for name in sorted(packages):
            spec = packages[name]
            version = spec.get("version") if isinstance(spec, Mapping) else None
            versions[str(name)] = "" if version is None else str(version)
        return versions

    # ---- 校验 ----

    def validate(self) -> list[str]:
        """静态校验：拒绝浮动版本、空 SHA、范围约束。返回收集到的问题列表。"""
        problems: list[str] = []
        try:
            reject_unpinned(self.payload.get("python"), "%s.python" % self.channel)
        except UnverifiedLock as exc:
            problems.append(exc.message)
        packages = self.payload.get("packages") or {}
        if not isinstance(packages, dict) or not packages:
            problems.append("%s.packages 为空" % self.channel)
            return problems
        for name in sorted(packages):
            spec = packages[name]
            if not isinstance(spec, dict):
                problems.append("%s.packages.%s 不是对象" % (self.channel, name))
                continue
            for field in REQUIRED_PACKAGE_FIELDS:
                if field not in spec:
                    problems.append("%s.packages.%s 缺少 %s" % (self.channel, name, field))
            for field in ("version", "wheel_sha256"):
                if field in spec:
                    try:
                        reject_unpinned(spec[field], "%s.packages.%s.%s" % (self.channel, name, field))
                    except UnverifiedLock as exc:
                        problems.append(exc.message)
            digest = str(spec.get("wheel_sha256", ""))
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                problems.append(
                    "%s.packages.%s.wheel_sha256 未取得或不是 64 位小写 hex（%r）"
                    % (self.channel, name, digest)
                )
        return problems

    def verify(self) -> dict:
        """完整校验；任何一项不过 → 抛 UnverifiedLock（阻断），不返回部分结果。"""
        problems = self.validate()
        if problems:
            raise UnverifiedLock(
                "dependency_lock_invalid",
                "%s 依赖锁不合法（%d 项）" % (self.channel, len(problems)),
                problems=problems,
            )
        if not self.payload.get("verified"):
            raise UnverifiedLock(
                "dependency_lock_unverified",
                "%s 依赖锁尚未安装验证（verified=false），禁止自动取 latest 继续"
                % self.channel,
                channel=self.channel,
            )
        return self.summary()

    def summary(self) -> dict:
        packages = self.payload.get("packages") or {}
        summary = {
            "channel": self.channel,
            "python": self.payload.get("python"),
            "package_count": len(packages),
            "packages": {
                name: packages[name].get("version") for name in sorted(packages)
            },
            "verified": bool(self.payload.get("verified")),
            "lock_sha256": self.lock_sha256(),
        }
        # KAGGLE-27 整改②③：把"版本作用域"与"证据等级"带进 preflight 报告，让
        # **物料版本 / 评分端实际版本 / 三种证据等级**在报告里就是分开的字段，
        # 而不是被压成一个 `verified` 布尔值。
        if self.payload.get("python_requirement"):
            summary["python_requirement"] = self.payload["python_requirement"]
        for key in ("python_evidence", "version_scopes", "evidence_levels"):
            if self.payload.get(key):
                summary[key] = self.payload[key]
        return summary


def assert_channels_isolated(train: DependencyLock, serving: DependencyLock) -> None:
    """训练锁与推理锁必须确实是两套；同一份 payload 视为配置错误。"""
    if train.lock_sha256() == serving.lock_sha256():
        raise PolicyViolation(
            "channels_not_isolated",
            "训练锁与推理锁内容完全相同：不得为了训练升级正式评分器（需分面锁定）",
        )


def assert_distinct_lock_channels(train: DependencyLock, serving: DependencyLock) -> dict:
    """判定训练锁与推理锁**确实不是同一份**（fail-closed，KAGGLE-20 §5「分面锁定」）。

    这是 preflight 可以直接接线的判定入口（不再只被 tests 调用）。比较三个维度：

    1. **归一化绝对路径**：`normcase(realpath(...))`，同一个文件即同一份；
    2. **内容 SHA**：`lock_sha256()`（payload 规范化 JSON）与 `source_sha256()`（文件原始字节，
       两者都可比较时同时比较）；
    3. **通道与包集合**：`channel` 必须一侧 train、一侧 serving；包名 → 版本映射**完全一致**
       时也视为未分面锁定（仅包名重叠不算，因为训练面与推理面可以合法地锁定同名包的不同版本）。

    任一维度命中 → `PolicyViolation("training_serving_lock_aliased", ...)`。

    返回（未抛异常时）::

        {"distinct": True,
         "train_path": str|None, "serving_path": str|None,
         "train_lock_sha256": str, "serving_lock_sha256": str,
         "train_source_sha256": str|None, "serving_source_sha256": str|None,
         "train_channel": "train", "serving_channel": "serving",
         "package_overlap": [共同包名…],
         "package_versions_identical": bool,
         "package_count": {"train": int, "serving": int}}
    """
    train_path, serving_path = train.normalized_path(), serving.normalized_path()
    reasons: list[str] = []
    if train_path is not None and train_path == serving_path:
        reasons.append("两份锁指向同一个文件：%s" % train.path)
    if train.lock_sha256() == serving.lock_sha256():
        reasons.append("两份锁 payload 规范化 JSON 哈希相同：%s" % train.lock_sha256())
    train_source, serving_source = train.source_sha256(), serving.source_sha256()
    if train_source is not None and train_source == serving_source:
        reasons.append("两份锁文件原始字节 SHA-256 相同：%s" % train_source)
    if train.channel != TRAIN_CHANNEL or serving.channel != SERVING_CHANNEL:
        reasons.append(
            "通道错配：train.channel=%r, serving.channel=%r" % (train.channel, serving.channel)
        )
    train_packages, serving_packages = train.package_versions(), serving.package_versions()
    packages_identical = bool(train_packages) and train_packages == serving_packages
    if packages_identical:
        reasons.append("包名→版本映射完全一致（%d 个包）：两侧未分面锁定" % len(train_packages))

    if reasons:
        raise PolicyViolation(
            "training_serving_lock_aliased",
            "训练锁与推理锁未隔离：%s" % "；".join(reasons),
            problems=reasons,
            train_path=train.path,
            serving_path=serving.path,
        )
    return {
        "distinct": True,
        "train_path": train.path,
        "serving_path": serving.path,
        "train_lock_sha256": train.lock_sha256(),
        "serving_lock_sha256": serving.lock_sha256(),
        "train_source_sha256": train_source,
        "serving_source_sha256": serving_source,
        "train_channel": train.channel,
        "serving_channel": serving.channel,
        "package_overlap": sorted(set(train_packages) & set(serving_packages)),
        "package_versions_identical": packages_identical,
        "package_count": {"train": len(train_packages), "serving": len(serving_packages)},
    }


def load_pair(train_path: str, serving_path: str) -> tuple[DependencyLock, DependencyLock]:
    """加载训练/推理两把锁并强制校验通道隔离（fail-closed）。

    两层校验都跑：`assert_channels_isolated`（payload 内容级）+
    `assert_distinct_lock_channels`（路径 / 内容 SHA / 包集合级，可返回结构化结果）。
    任一不过即抛 `FailClosed`，调用方不得继续。
    """
    train = DependencyLock.from_file(train_path)
    serving = DependencyLock.from_file(serving_path)
    assert_channels_isolated(train, serving)
    assert_distinct_lock_channels(train, serving)
    return train, serving


def file_digest(path: str) -> str:
    return sha256_file(path)
