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

    def assert_no_full_state_dict(
        self,
        peak_full_state_dict_bytes: int | None = None,
        allowed_bytes: int = 0,
        evidence: Mapping | None = None,
    ) -> dict:
        """对**行为**的 fail-closed 断言：禁止在 CPU 把完整 state_dict 整体驻留。

        旧实现是"装得下才抛"（方向与规则相反、拦不住真实违规），已彻底删除。
        现在的判定只看调用方提交的**驻留证据**：

        - `peak_full_state_dict_bytes is None`（未提供证据）→
          `PolicyViolation("full_state_dict_resident_unevidenced", ...)`：没有证据不得视为通过；
        - 计数为负 / 非整数 → `PolicyViolation("full_state_dict_counter_invalid", ...)`；
        - `peak_full_state_dict_bytes > allowed_bytes` →
          `PolicyViolation("full_state_dict_resident", ...)`：确实整体驻留过。

        参数
        ----
        - `peak_full_state_dict_bytes`：**曾经同时驻留**的完整 state_dict 峰值字节数，
          一次都没整体驻留过就是 `0`；`None` = 调用方没有提供证据。
        - `allowed_bytes`：允许的整份驻留上限，恒为 `0`（只有 0 才等于"从不允许整体驻留"）。
        - `evidence`：可选旁证（探针名 / 作业号等），原样回带，便于报告引用。

        返回（未抛异常时）::

            {"checked": True, "full_state_dict_resident": False,
             "peak_full_state_dict_bytes": int, "allowed_bytes": int,
             "rss_limit_bytes": int, "total_weight_bytes": int, "evidence": dict}
        """
        if peak_full_state_dict_bytes is None:
            raise PolicyViolation(
                "full_state_dict_resident_unevidenced",
                "没有提供『从未整体驻留完整 state_dict』的证据：不得视为通过（fail-closed）",
            )
        try:
            peak = int(peak_full_state_dict_bytes)
            allowed = int(allowed_bytes)
        except (TypeError, ValueError):
            raise PolicyViolation(
                "full_state_dict_counter_invalid",
                "驻留计数必须是整数：peak=%r, allowed=%r"
                % (peak_full_state_dict_bytes, allowed_bytes),
            )
        if peak < 0 or allowed < 0:
            raise PolicyViolation(
                "full_state_dict_counter_invalid",
                "驻留计数不得为负：peak=%d, allowed=%d" % (peak, allowed),
            )
        if peak > allowed:
            raise PolicyViolation(
                "full_state_dict_resident",
                "检出完整 state_dict 整体驻留 %d 字节 > 允许 %d 字节："
                "必须逐 tensor / 逐 shard 直接做 device placement" % (peak, allowed),
                peak_full_state_dict_bytes=peak,
                allowed_bytes=allowed,
                rss_limit_bytes=self.effective_rss_limit_bytes(),
                evidence=dict(evidence or {}),
            )
        return {
            "checked": True,
            "full_state_dict_resident": False,
            "peak_full_state_dict_bytes": peak,
            "allowed_bytes": allowed,
            "rss_limit_bytes": self.effective_rss_limit_bytes(),
            "total_weight_bytes": self.total_bytes,
            "evidence": dict(evidence or {}),
        }

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


def assert_full_state_dict_guard(
    observation: Mapping | None,
    *,
    allowed_bytes: int = 0,
    weight_manifest: Mapping | None = None,
) -> dict:
    """**生产路径入口**（🟡-8）：把一次真实观测送进 `assert_no_full_state_dict`。

    KAGGLE-26 复核指出的问题不是判据写错，而是**没有生产调用点**：
    `assert_no_full_state_dict` 当时只有 4 处 tests 调用，`runner.run_training`
    在保存 adapter 之前从未经过它。本函数就是那条接线。

    输入必须是**观测值**（`peak_full_state_dict_bytes` + `declared_weight_bytes`
    + `probe` + `method`），缺失即 fail-closed —— 不接受自述结论。
    观测由后端自己产出（见 `runner.TrainBackend.observe_full_state_dict`）。
    """
    if observation is None:
        raise PolicyViolation(
            "full_state_dict_observation_missing",
            "没有提供『从未整体驻留完整 state_dict』的实测观测："
            "保存 adapter 前不得跳过该门槛（fail-closed，不接受自述结论）",
        )
    if not isinstance(observation, Mapping):
        raise MissingInput(
            "full_state_dict_observation_not_mapping",
            "观测必须是映射（含 peak_full_state_dict_bytes 与 declared_weight_bytes）：%r"
            % (observation,),
        )
    payload = dict(observation)
    for key in ("peak_full_state_dict_bytes", "declared_weight_bytes"):
        if key not in payload:
            raise MissingInput(
                "full_state_dict_observation_incomplete",
                "观测缺少必需字段 %s：%r" % (key, sorted(payload)),
                missing=key,
            )
    declared = payload.get("declared_weight_bytes")
    try:
        declared_bytes = int(declared)
    except (TypeError, ValueError):
        raise PolicyViolation(
            "full_state_dict_observation_invalid",
            "declared_weight_bytes 必须是整数：%r" % (declared,),
        )
    shards = (weight_manifest or {}).get("shards") if isinstance(weight_manifest, Mapping) else None
    if not shards:
        shards = [{"name": "declared-total-weight-budget", "bytes": max(0, declared_bytes)}]
    plan = StreamingPlan.from_manifest({"shards": list(shards)})
    result = plan.assert_no_full_state_dict(
        peak_full_state_dict_bytes=payload["peak_full_state_dict_bytes"],
        allowed_bytes=allowed_bytes,
        evidence={
            "probe": str(payload.get("probe") or "unspecified"),
            "method": str(payload.get("method") or "unspecified"),
        },
    )
    result["observation"] = {
        "probe": str(payload.get("probe") or "unspecified"),
        "method": str(payload.get("method") or "unspecified"),
        "peak_full_state_dict_bytes": result["peak_full_state_dict_bytes"],
        "declared_weight_bytes": declared_bytes,
    }
    return result
