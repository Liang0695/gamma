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

    def __init__(self, channel: str, payload: Mapping) -> None:
        if channel not in (TRAIN_CHANNEL, SERVING_CHANNEL):
            raise PolicyViolation("bad_channel", "未知软件通道：%r" % (channel,))
        self.channel = channel
        self.payload = dict(payload)

    # ---- 读取 ----

    @classmethod
    def from_file(cls, path: str) -> "DependencyLock":
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        for field in ("channel", "python", "packages", "verified", "notes"):
            if field not in raw:
                raise MissingInput("lock_missing_field", "锁文件缺少字段 %r：%s" % (field, path))
        return cls(raw["channel"], raw)

    def lock_sha256(self) -> str:
        """锁内容自身的哈希，写进各类 manifest。"""
        return sha256_json(self.payload)

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
        return {
            "channel": self.channel,
            "python": self.payload.get("python"),
            "package_count": len(packages),
            "packages": {
                name: packages[name].get("version") for name in sorted(packages)
            },
            "verified": bool(self.payload.get("verified")),
            "lock_sha256": self.lock_sha256(),
        }


def assert_channels_isolated(train: DependencyLock, serving: DependencyLock) -> None:
    """训练锁与推理锁必须确实是两套；同一份 payload 视为配置错误。"""
    if train.lock_sha256() == serving.lock_sha256():
        raise PolicyViolation(
            "channels_not_isolated",
            "训练锁与推理锁内容完全相同：不得为了训练升级正式评分器（需分面锁定）",
        )


def load_pair(train_path: str, serving_path: str) -> tuple[DependencyLock, DependencyLock]:
    train = DependencyLock.from_file(train_path)
    serving = DependencyLock.from_file(serving_path)
    assert_channels_isolated(train, serving)
    return train, serving


def file_digest(path: str) -> str:
    return sha256_file(path)
