"""流式加载与内存账（KAGGLE-20 §5，KAGGLE-19 §3）。

硬规则：
- **禁止在 CPU 构造 58GiB 完整 state_dict**：逐 tensor / 逐 shard 直接做 device placement；
- 进程 RSS 目标 ≤12GiB，且必须在**实测 cgroup 字节**上再扣 ≥3GiB 余量，取更严格值；
- 主机 16GB 限制按实测扣减，不按标称值乐观估计；
- 磁盘先核查权重/环境/最大快照/日志 + 15% 余量后再下载。

本模块只做**规划与账目**，不真正加载权重（那需要 GPU 与锁定软件）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..common.errors import MissingInput, PolicyViolation

GiB = 1 << 30

#: 设计稿给定的目标与余量。
TARGET_RSS_GIB = 12.0
HOST_RESERVE_GIB = 3.0
DISK_HEADROOM_RATIO = 0.15


@dataclass(frozen=True)
class WeightShard:
    name: str
    bytes: int
    path: str | None = None

    def to_dict(self) -> dict:
        return {"name": self.name, "bytes": self.bytes, "path": self.path}


@dataclass
class StreamingPlan:
    """逐 shard 流式加载计划。"""

    shards: list[WeightShard]
    working_buffer_bytes: int = 0
    host_cgroup_bytes: int | None = None
    host_total_bytes: int | None = None

    @classmethod
    def from_manifest(cls, manifest: Mapping, working_buffer_bytes: int = 0, **kwargs) -> "StreamingPlan":
        entries = manifest.get("shards")
        if not entries:
            raise MissingInput("empty_weight_manifest", "权重 manifest 没有 shards")
        shards = []
        for entry in entries:
            for required in ("name", "bytes"):
                if required not in entry:
                    raise MissingInput(
                        "shard_missing_field", "shard 缺少 %s" % required, shard=entry
                    )
            shards.append(WeightShard(entry["name"], int(entry["bytes"]), entry.get("path")))
        return cls(shards=shards, working_buffer_bytes=working_buffer_bytes, **kwargs)

    # ---- 账目 ----

    @property
    def total_bytes(self) -> int:
        return sum(shard.bytes for shard in self.shards)

    @property
    def largest_shard_bytes(self) -> int:
        return max((shard.bytes for shard in self.shards), default=0)

    def effective_rss_limit_bytes(self) -> int:
        """取 min(目标 12GiB, 实测 cgroup − 3GiB)。"""
        limit = int(TARGET_RSS_GIB * GiB)
        if self.host_cgroup_bytes:
            limit = min(limit, max(0, self.host_cgroup_bytes - int(HOST_RESERVE_GIB * GiB)))
        return limit

    def peak_rss_bytes(self) -> int:
        """流式峰值 = 最大的单个 shard/tensor + 工作缓冲（不是全部权重）。"""
        return self.largest_shard_bytes + self.working_buffer_bytes

    def evaluate(self) -> dict:
        limit = self.effective_rss_limit_bytes()
        peak = self.peak_rss_bytes()
        problems = []
        if self.total_bytes > limit:
            # 这只说明"不能整体驻留"，正是流式加载存在的理由。
            problems.append(
                "权重总量 %.2fGiB 超过 RSS 上限 %.2fGiB：必须流式，不得整份常驻"
                % (self.total_bytes / GiB, limit / GiB)
            )
        streaming_required = bool(problems)
        if peak > limit:
            return {
                "status": "stop",
                "reason": "单 shard + 工作缓冲已超 RSS 上限：需要更细粒度分片或换 NF4 后备",
                "peak_rss_bytes": peak,
                "limit_bytes": limit,
                "shards": len(self.shards),
            }
        return {
            "status": "pass",
            "streaming_required": streaming_required,
            "peak_rss_bytes": peak,
            "limit_bytes": limit,
            "total_weight_bytes": self.total_bytes,
            "shards": len(self.shards),
            "largest_shard_bytes": self.largest_shard_bytes,
            "note": "流式峰值只计单 shard + 缓冲；真实峰值须在作业里实测并按 cgroup 复核。",
        }

    def assert_no_full_state_dict(self) -> None:
        """显式禁止在 CPU 构造完整 state_dict。"""
        limit = self.effective_rss_limit_bytes()
        if self.total_bytes <= limit and self.host_cgroup_bytes:
            raise PolicyViolation(
                "full_state_dict_looks_feasible_but_forbidden",
                "即使理论上装得下，也禁止 CPU 构造完整 state_dict（内存账按最严格口径）",
            )

    def iter_shards(self) -> Iterable[WeightShard]:
        """按 manifest 顺序逐个产出 shard（调用方负责释放上一个）。"""
        for shard in self.shards:
            yield shard


@dataclass
class SimulatedAllocator:
    """模拟逐 shard 加载时的峰值占用（用于离线验证账目口径）。"""

    limit_bytes: int
    held: int = 0
    peak: int = 0
    events: list[dict] = field(default_factory=list)

    def load(self, shard: WeightShard) -> None:
        if self.held:
            raise PolicyViolation(
                "shard_not_released",
                "上一个 shard 未释放：流式加载必须逐个释放",
                held=self.held,
            )
        self.held = shard.bytes
        self.peak = max(self.peak, self.held)
        if self.peak > self.limit_bytes:
            raise PolicyViolation(
                "rss_budget_exceeded",
                "流式峰值 %d 字节超过上限 %d" % (self.peak, self.limit_bytes),
            )
        self.events.append({"load": shard.name, "bytes": shard.bytes, "peak": self.peak})

    def release(self, shard: WeightShard) -> None:
        self.held = 0
        self.events.append({"release": shard.name})


def simulate_streaming(plan: StreamingPlan) -> dict:
    """跑一遍模拟：验证峰值与释放纪律，不触发真实加载。"""
    allocator = SimulatedAllocator(limit_bytes=plan.effective_rss_limit_bytes())
    for shard in plan.iter_shards():
        allocator.load(shard)
        allocator.release(shard)
    return {
        "shards": len(plan.shards),
        "peak_bytes": allocator.peak,
        "limit_bytes": allocator.limit_bytes,
        "events": allocator.events,
        "status": "pass",
    }


def disk_plan(
    weight_bytes: int,
    env_bytes: int = 0,
    largest_snapshot_bytes: int = 0,
    log_bytes: int = 0,
    extra_bytes: int = 0,
) -> dict:
    """磁盘规划：总和 + 15% 余量，且不低于 80GiB 的模型流程起点。"""
    subtotal = weight_bytes + env_bytes + largest_snapshot_bytes + log_bytes + extra_bytes
    with_headroom = int(subtotal * (1 + DISK_HEADROOM_RATIO))
    minimum = max(with_headroom, 80 * GiB)
    return {
        "subtotal_bytes": subtotal,
        "with_headroom_bytes": with_headroom,
        "required_bytes": minimum,
        "headroom_ratio": DISK_HEADROOM_RATIO,
        "note": "80GiB 只是模型流程规划起点，不是全流水线保证。",
    }


def assert_disk_available(plan: Mapping, available_bytes: int) -> None:
    if available_bytes < plan["required_bytes"]:
        raise PolicyViolation(
            "disk_insufficient",
            "可用磁盘 %.2fGiB < 需求 %.2fGiB，不得自动下载"
            % (available_bytes / GiB, plan["required_bytes"] / GiB),
        )


def bf16_intermediate_disk_estimate(params: int = 31_273_088_876) -> dict:
    """NF4 后备需要先写 BF16 中间分片：约 +63GB。"""
    return {
        "bf16_intermediate_bytes": params * 2,
        "note": "NF4 后备路线会增加约 63GB 中间权重磁盘，且每次登记 source_revision/dequantized_artifact_sha/nf4_config_sha。",
    }


def summarize_shards(plan: StreamingPlan, top: int = 5) -> list[dict]:
    ordered = sorted(plan.shards, key=lambda shard: shard.bytes, reverse=True)
    return [shard.to_dict() for shard in ordered[:top]]
