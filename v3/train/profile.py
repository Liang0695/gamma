"""G7b：受控的**真实内存测量入口**（前向 + 反向 + 优化器步进 + 主机峰值）。

## 缺陷出处

KAGGLE-32 v6 `GAPS-AND-OWNERS.md` §G7b：

- 消费侧已确定：`entry.py:581-584` 只从 `--memory-profile <json>` 读文件；
  `MemoryPlan.evaluate_or_block` 只接受调用方给的 mapping；
- 全仓 grep `max_memory_allocated|max_memory_reserved|ru_maxrss` **零命中**
  → 固定版 E0 的已检入文件里找不到 profile **生产者**入口；
- 入口必须覆盖**前向+反向+优化器步进**下的主机 RSS 与显存峰值：只测加载
  会系统性低估（GAPS §3.2）；
- **不得在 KAGGLE-32 手工置 true 绕过**。

## 本模块产出什么

:func:`measure_memory_profile` 驱动后端跑一遍真实工作负载（prepare → 前向 → 反向 →
优化器步进），逐阶段记录**主机峰值 RSS** 与**显存峰值**，产出：

    {"peak_gpu_gib": float,          # 仅当显存真的测到时存在
     "host_rss_gib": float,          # 主机峰值 RSS（真实测量）
     "seq_len": int, "source": str, "probe": str, "method": str,
     "is_real_measurement": True, "gpu_measurement_status": "measured"|"unavailable",
     "gpu_provenance": str, "phases": [...], "residual_risk": str, ...}

## 测量方法（都是真的调用，不是自述）

- 主机：`/proc/self/status:VmHWM`（Linux，高水位，KiB），非 Linux 退回
  `resource.getrusage(RUSAGE_SELF).ru_maxrss`（Windows/macOS 为字节）；
  逐阶段取 HWM 差值，HWM 单调递增，因此差值即该阶段的**新增峰值**；
- 显存：`torch.cuda.reset_peak_memory_stats()` → 工作负载 →
  `torch.cuda.max_memory_allocated()` / `max_memory_reserved()`；
- 未装 torch 或没有 CUDA 时 `gpu_measurement_status="unavailable"`，
  **不写** `peak_gpu_gib`，于是 `MemoryPlan.evaluate_or_block()` 直接 fail-closed ——
  本机（无 GPU）永远不可能在本地产出"内存门槛通过"。

## 诚实边界（写进返回值，供审查者核对）

`residual_risk` 字段照实写：provenance/结构校验能挡掉"手打一段 JSON 就说测过"，
但**挡不住**有意伪造 provenance 的人。真正的保证来自获批 GPU 作业上产出的证据包
（`engineering_check.py` 的 `evidence_sha256` 链）+ 独立审查，不是本模块的自证。
"""

from __future__ import annotations

import os
import platform
import sys
import time
from typing import Mapping, Sequence

from ..common.errors import FailClosed, MissingInput, PolicyViolation
from .targets import FIRST_ROUND_SEQ_LEN

#: 必须覆盖的阶段名（顺序即报告顺序）。
REQUIRED_PHASES = ("prepare", "forward", "backward", "optimizer_step")

#: 真实显存 API 的 provenance 白名单。
REAL_GPU_PROVENANCE = "torch.cuda.max_memory_allocated+max_memory_reserved"
#: 没有可用显存测量时的状态值。
GPU_UNAVAILABLE = "unavailable"
GPU_MEASURED = "measured"

#: 合成 fixture profile 的标记（**不是**测量结果，生产门槛必须拒绝）。
SYNTHETIC_PROVENANCE = "synthetic-fixture-not-a-measurement"

_RESIDUAL_RISK = (
    "provenance 与结构校验挡得住「手打一段 JSON 自称测过」，挡不住有意伪造 provenance 的人；"
    "最终保证来自获批 GPU 作业的证据包与独立审查，不是本模块的自证。"
)


def _windows_peak_working_set_bytes() -> dict | None:
    """Windows 上的真实峰值工作集（`psapi.GetProcessMemoryInfo`）。

    `PeakWorkingSetSize` 是内核维护的高水位，语义等价于 Linux 的 `VmHWM`；
    取不到就返回 `None`（由调用方决定 fail-closed），绝不用估计值顶替。
    """
    if os.name != "nt":
        return None
    try:
        import ctypes  # noqa: PLC0415
        from ctypes import wintypes  # noqa: PLC0415
    except Exception:  # noqa: BLE001 - 受限环境下 ctypes 不可用
        return None

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        handle = kernel32.GetCurrentProcess()
        ok = psapi.GetProcessMemoryInfo(
            wintypes.HANDLE(handle), ctypes.byref(counters), counters.cb
        )
        if not ok:
            return None
    except Exception:  # noqa: BLE001
        return None
    return {
        "bytes": int(counters.PeakWorkingSetSize),
        "probe": "psapi.GetProcessMemoryInfo:PeakWorkingSetSize",
        "raw": int(counters.PeakWorkingSetSize),
        "working_set_bytes": int(counters.WorkingSetSize),
    }


def host_peak_rss_bytes() -> dict:
    """当前进程的主机峰值常驻内存（真实高水位）。

    探测顺序（命中即返回，取不到就继续，最后才 fail-closed）：

    1. Linux：`/proc/self/status:VmHWM`；
    2. Windows：`psapi.GetProcessMemoryInfo:PeakWorkingSetSize`；
    3. POSIX：`resource.getrusage(RUSAGE_SELF).ru_maxrss`。

    返回 `{"bytes": int, "probe": str, "raw": ...}`。**任何情况下都不返回估计值**。
    """
    status_path = "/proc/self/status"
    if os.path.exists(status_path):
        try:
            with open(status_path, "r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith("VmHWM:"):
                        kib = float(line.split()[1])
                        return {
                            "bytes": int(kib * 1024),
                            "probe": "proc/self/status:VmHWM",
                            "raw": line.strip(),
                        }
        except OSError:
            pass

    windows = _windows_peak_working_set_bytes()
    if windows is not None:
        return windows

    try:
        import resource  # 只在 POSIX / 部分 Windows 构建里可用
    except ImportError as exc:
        raise FailClosed(
            "host_memory_probe_unavailable",
            "本机没有任何可用的主机峰值内存探针"
            "（/proc/self/status、psapi.GetProcessMemoryInfo、resource 都不可用）",
            error=str(exc),
        ) from exc
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux 上 ru_maxrss 单位是 KiB，Windows/macOS 是字节。
    scale = 1024 if sys.platform.startswith("linux") else 1
    return {
        "bytes": int(usage) * scale,
        "probe": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
        "raw": usage,
        "unit_scale": scale,
        "platform": platform.system(),
    }


class CudaMemoryReader:
    """真实显存读数封装（只有真的 import 到 torch 且 CUDA 可用时才可用）。

    测试用替身必须声明不同的 `provenance`，因此**不会**被生产门槛认作真实测量。
    """

    provenance = REAL_GPU_PROVENANCE

    def __init__(self) -> None:
        self.torch = None
        self.available = False
        self.reason = "torch 未安装"
        try:
            import torch  # noqa: PLC0415 - 延迟导入：无 GPU 环境不得因此崩掉
        except Exception as exc:  # noqa: BLE001
            self.reason = "import torch 失败：%s" % exc
            return
        self.torch = torch
        try:
            self.available = bool(torch.cuda.is_available())
        except Exception as exc:  # noqa: BLE001
            self.reason = "torch.cuda.is_available() 抛错：%s" % exc
            return
        if not self.available:
            self.reason = "torch 已装但 torch.cuda.is_available() 为 False"

    def reset(self) -> None:
        if self.available:
            self.torch.cuda.reset_peak_memory_stats()

    def peaks_gib(self) -> dict:
        if not self.available:
            return {}
        gib = 1024.0 ** 3
        return {
            "peak_gpu_gib": self.torch.cuda.max_memory_allocated() / gib,
            "peak_gpu_reserved_gib": self.torch.cuda.max_memory_reserved() / gib,
            "gpu_device_name": self.torch.cuda.get_device_name(0),
        }

    def free_gib(self) -> float | None:
        if not self.available or not hasattr(self.torch.cuda, "mem_get_info"):
            return None
        try:
            free_bytes, _total = self.torch.cuda.mem_get_info()
        except Exception:  # noqa: BLE001 - 读数失败不猜
            return None
        return free_bytes / (1024.0 ** 3)


class PhaseProbe:
    """逐阶段记录主机峰值 RSS 增量与显存峰值（真实读数，不是估计）。"""

    def __init__(self, *, gpu_reader=None, monotonic=time.monotonic) -> None:
        self.gpu_reader = gpu_reader
        self._monotonic = monotonic
        self.phases: list[dict] = []
        self._open: dict | None = None
        self.start_bytes = host_peak_rss_bytes()
        self._last_hwm = int(self.start_bytes["bytes"])
        if self.gpu_reader is not None:
            self.gpu_reader.reset()

    # -- 上下文管理 --
    def begin(self, name: str) -> None:
        if self._open is not None:
            raise PolicyViolation(
                "memory_probe_nested", "阶段 %s 未结束就开始了 %s" % (self._open["phase"], name)
            )
        self._open = {
            "phase": str(name),
            "host_hwm_bytes_before": self._last_hwm,
            "t0": self._monotonic(),
        }

    def end(self) -> dict:
        if self._open is None:
            raise PolicyViolation("memory_probe_unbalanced", "没有正在进行的阶段")
        opened = self._open
        self._open = None
        after = int(host_peak_rss_bytes()["bytes"])
        delta = max(0, after - int(opened["host_hwm_bytes_before"]))
        self._last_hwm = max(self._last_hwm, after)
        entry = {
            "phase": opened["phase"],
            "host_peak_rss_gib": after / (1024.0 ** 3),
            "host_peak_rss_delta_gib": delta / (1024.0 ** 3),
            "wall_seconds": round(self._monotonic() - opened["t0"], 4),
        }
        if self.gpu_reader is not None and getattr(self.gpu_reader, "available", False):
            entry.update(self.gpu_reader.peaks_gib())
        self.phases.append(entry)
        return entry

    class _Phase:  # pragma: no cover - 薄包装
        def __init__(self, probe: "PhaseProbe", name: str) -> None:
            self.probe = probe
            self.name = name

        def __enter__(self):
            self.probe.begin(self.name)
            return self.probe

        def __exit__(self, exc_type, exc, tb) -> bool:
            self.probe.end()
            return False

    def phase(self, name: str) -> "PhaseProbe._Phase":
        return PhaseProbe._Phase(self, name)

    def phases_seen(self) -> list[str]:
        return [item["phase"] for item in self.phases]


def _backend_phase_runner(backend) -> str:
    """后端是否提供逐阶段入口：决定了测量粒度是真是虚。"""
    if hasattr(backend, "measure_phases") and callable(getattr(backend, "measure_phases")):
        return "backend.measure_phases"
    return "backend.train_steps（整体，无法拆分阶段）"


def measure_memory_profile(
    backend,
    plan,
    *,
    gpu_reader=None,
    seq_len: int | None = None,
    available_gpu_gib: float | None = None,
    downgrade_used: bool = False,
    note: str | None = None,
    prepare: bool = True,
    prepared_report: Mapping | None = None,
) -> dict:
    """跑一遍工作负载并产出内存 profile（主机必测、显存能测才写）。

    `gpu_reader` 省略时构造 :class:`CudaMemoryReader`。测试替身必须声明不同的
    `provenance`，否则本函数会在返回值里把它标成非真实测量来源。

    `prepare=False`（KAGGLE-38 r3 缺陷 B）：调用方**已经** prepare 过这个后端实例，
    这里不得再调一次 `backend.prepare(plan)` —— 真实后端第二次 prepare 会把 31B 基座
    再加载一遍，模型/optimizer 实例错配、驻留翻倍。此时必须给 `prepared_report`
    （前一次的 prepared 结果）；本函数测的是**同一组模型/optimizer 参数**。
    `prepare` 阶段仍会出现在 phases 里，但显式标注 `reused_prepared_backend=true`，
    不假装又测了一遍加载。
    """
    plan.assert_runnable()
    reader = gpu_reader if gpu_reader is not None else CudaMemoryReader()
    probe = PhaseProbe(gpu_reader=reader)

    if prepare:
        with probe.phase("prepare"):
            prepared = backend.prepare(plan)
    else:
        prepared = dict(prepared_report or {})
        if not prepared:
            raise MissingInput(
                "prepared_report_missing",
                "prepare=False 时必须给出上一次的 prepared 结果：复用必须可审计",
            )
        with probe.phase("prepare"):
            # 只做"复用确认"，**不**重复 prepare（不重复加载基座）。
            checker = getattr(backend, "assert_prepared", None)
            if callable(checker):
                checker()
        probe.phases[-1]["reused_prepared_backend"] = True
        probe.phases[-1]["note"] = "复用已加载实例：本阶段没有再次加载模型"
    profile_reused_prepared_backend = not prepare

    phase_runner = _backend_phase_runner(backend)
    if hasattr(backend, "measure_phases") and callable(getattr(backend, "measure_phases")):
        backend.measure_phases(plan, probe)
    else:
        with probe.phase("train_all"):
            backend.train_steps(plan)

    host_peak = host_peak_rss_bytes()
    seen = probe.phases_seen()
    missing_phases = [name for name in REQUIRED_PHASES if name not in seen]

    reader_available = bool(getattr(reader, "available", False))
    provenance = str(getattr(reader, "provenance", "unknown"))
    # 关键的一条：**工作负载本身必须真的用了 GPU**。
    # 否则在一台有 CUDA 的机器上跑 CPU 玩具模型，会读出 max_memory_allocated()==0，
    # 那个 0 不是"显存够用"的证据，只是"根本没上卡"。
    uses_gpu = bool(getattr(backend, "requires_gpu", False))
    is_real = bool(reader_available and provenance == REAL_GPU_PROVENANCE and uses_gpu)

    profile: dict = {
        "seq_len": int(seq_len if seq_len is not None else getattr(plan, "seq_len", FIRST_ROUND_SEQ_LEN)),
        "host_rss_gib": host_peak["bytes"] / (1024.0 ** 3),
        "host_measurement": {
            "probe": host_peak["probe"],
            "raw": host_peak.get("raw"),
            "is_real_measurement": True,
        },
        "phases": probe.phases,
        "phase_measurement": phase_runner,
        "required_phases": list(REQUIRED_PHASES),
        "missing_phases": missing_phases,
        "workload": {
            "steps": int(getattr(plan, "steps", 0)),
            "seq_len": int(getattr(plan, "seq_len", 0)),
            "lora_rank": int(getattr(plan, "lora_rank", 0)),
            "adapter_name": getattr(plan, "adapter_name", None),
            "backend": getattr(backend, "name", type(backend).__name__),
            "requires_gpu": uses_gpu,
            "uses_gpu": uses_gpu,
        },
        "backend_prepared": prepared,
        "reused_prepared_backend": bool(profile_reused_prepared_backend),
        "backend_load_count": int(getattr(backend, "load_count", 0) or 0),
        "backend_parameter_ownership": getattr(backend, "parameter_ownership", None),
        "gpu_provenance": provenance,
        "gpu_measurement_status": GPU_MEASURED if reader_available else GPU_UNAVAILABLE,
        "gpu_unavailable_reason": (
            None
            if reader_available
            else str(getattr(reader, "reason", ""))
        )
        or (None if uses_gpu else "工作负载没有使用 GPU（后端 requires_gpu=False）：不构成显存证据"),
        "is_real_measurement": is_real,
        "probe": "phase-hwm+cuda-peak",
        "method": (
            "主机峰值 = %s 逐阶段高水位差；显存峰值 = %s"
            % (host_peak["probe"], provenance)
        ),
        "source": "measured-%s" % provenance,
        "residual_risk": _RESIDUAL_RISK,
        "note": note
        or (
            "本 profile 只描述**本机这次工作负载**的内存占用；"
            "未测到显存或不构成显存证据时不写 peak_gpu_gib，内存门槛因此 fail-closed。"
        ),
    }
    if reader_available and uses_gpu:
        profile.update(reader.peaks_gib())
    if available_gpu_gib is not None:
        profile["available_gpu_gib"] = float(available_gpu_gib)
    elif reader_available and reader.free_gib() is not None:
        profile["available_gpu_gib"] = reader.free_gib()
    profile["downgrade_used"] = bool(downgrade_used)
    return profile


def profile_to_memory_plan_payload(profile: Mapping | None) -> dict | None:
    """把 profile 转成 `MemoryPlan` 能消费的 mapping。

    显存没测到（`gpu_measurement_status != "measured"`）或不是真实测量时返回 `None` ——
    调用方必须据此 fail-closed，**不得**退回去用估计值。
    """
    if not isinstance(profile, Mapping):
        return None
    if str(profile.get("gpu_measurement_status")) != GPU_MEASURED:
        return None
    if profile.get("is_real_measurement") is not True:
        return None
    if profile.get("peak_gpu_gib") is None or profile.get("host_rss_gib") is None:
        return None
    payload = {
        "peak_gpu_gib": float(profile["peak_gpu_gib"]),
        "host_rss_gib": float(profile["host_rss_gib"]),
        "seq_len": int(profile.get("seq_len", FIRST_ROUND_SEQ_LEN)),
        "source": str(profile.get("source") or "measured"),
    }
    if profile.get("available_gpu_gib") is not None:
        payload["available_gpu_gib"] = float(profile["available_gpu_gib"])
    payload["downgrade_used"] = bool(profile.get("downgrade_used", False))
    return payload


def check_profile_provenance(profile: Mapping | None) -> dict:
    """闸门用的来源校验：返回 `{"ok": bool, "problems": [...], "residual_risk": str}`。

    检查项（缺一即不通过）：

    1. `is_real_measurement is True`；
    2. `gpu_measurement_status == "measured"`；
    3. `gpu_provenance` 在真实 API 白名单里；
    4. `workload.uses_gpu is True`（工作负载真的上卡了 —— 否则 CUDA 可见也说明不了什么）；
    5. `host_measurement.is_real_measurement is True` 且带真实 probe 名；
    6. `missing_phases` 为空（前向/反向/优化器步进都被覆盖过）。
    """
    problems: list[str] = []
    if not isinstance(profile, Mapping) or not profile:
        return {
            "ok": False,
            "problems": ["没有提供内存 profile：未实测不得当作通过"],
            "residual_risk": _RESIDUAL_RISK,
        }
    if profile.get("is_real_measurement") is not True:
        problems.append("profile 未声明为真实测量（is_real_measurement != true）")
    if str(profile.get("gpu_measurement_status")) != GPU_MEASURED:
        problems.append(
            "显存未被测量（gpu_measurement_status=%r）：%s"
            % (profile.get("gpu_measurement_status"), profile.get("gpu_unavailable_reason"))
        )
    if str(profile.get("gpu_provenance")) not in (REAL_GPU_PROVENANCE,):
        problems.append(
            "显存 provenance %r 不在真实 API 白名单里（%s）"
            % (profile.get("gpu_provenance"), REAL_GPU_PROVENANCE)
        )
    workload = profile.get("workload")
    if not isinstance(workload, Mapping) or workload.get("uses_gpu") is not True:
        problems.append(
            "profile 记录的工作负载没有使用 GPU（workload.uses_gpu != true）："
            "CPU 上跑出来的显存数字不构成显存证据"
        )
    host = profile.get("host_measurement")
    if not isinstance(host, Mapping) or host.get("is_real_measurement") is not True:
        problems.append("主机峰值 RSS 缺少真实测量记录（host_measurement）")
    elif not str(host.get("probe") or "").strip():
        problems.append("主机峰值 RSS 记录里没有 probe 名")
    missing = list(profile.get("missing_phases") or [])
    if missing:
        problems.append("以下必需阶段没有被覆盖：%s" % missing)
    return {"ok": not problems, "problems": problems, "residual_risk": _RESIDUAL_RISK}


def assert_profile_is_real_measurement(profile: Mapping | None) -> dict:
    """`check_profile_provenance` 的抛错版本。"""
    report = check_profile_provenance(profile)
    if not report["ok"]:
        raise PolicyViolation(
            "memory_profile_not_a_measurement",
            "内存 profile 不满足真实测量要求：%s" % report["problems"][0],
            problems=report["problems"],
            residual_risk=report["residual_risk"],
        )
    return report


def synthetic_fixture_profile(
    *,
    peak_gpu_gib: float = 1.0,
    host_rss_gib: float = 1.0,
    seq_len: int = FIRST_ROUND_SEQ_LEN,
) -> dict:
    """**仅供单测/负例使用**的合成 profile。

    刻意带 `is_real_measurement=False` 与 `gpu_provenance=synthetic-fixture...`，
    因此 `check_profile_provenance()` 必然拒绝它 —— 这个函数存在是为了让
    "伪造 profile 过门" 变成一条**有测试守着的失败路径**，而不是一个隐患。
    """
    return {
        "seq_len": int(seq_len),
        "peak_gpu_gib": float(peak_gpu_gib),
        "host_rss_gib": float(host_rss_gib),
        "gpu_measurement_status": GPU_MEASURED,
        "gpu_provenance": SYNTHETIC_PROVENANCE,
        "is_real_measurement": False,
        "source": "synthetic-fixture",
        "host_measurement": {"probe": "synthetic", "is_real_measurement": False},
        "phases": [],
        "missing_phases": list(REQUIRED_PHASES),
        "residual_risk": _RESIDUAL_RISK,
        "note": "合成 fixture：不是测量结果，任何生产门槛都必须拒绝它。",
    }


def assert_phases_covered(profile: Mapping, *, required: Sequence[str] = REQUIRED_PHASES) -> None:
    """断言 profile 覆盖了必需阶段（供工程检查自检）。"""
    seen = {str(item.get("phase")) for item in (profile.get("phases") or [])}
    missing = [name for name in required if name not in seen]
    if missing:
        raise MissingInput(
            "memory_profile_phases_missing",
            "内存 profile 未覆盖阶段：%s（实际 %s）" % (missing, sorted(seen)),
            missing=missing,
        )


def probe_cuda_reader(*, mib: int = 64, reader=None) -> dict:
    """**受控的极小显存探针**：验证显存读数路径真的能返回非零峰值。

    只分配一个 `mib` 大小的张量、做一次加 1、就释放 —— 不加载模型、不做训练、
    不产生任何产物。存在的意义是：把"显存读数代码从没在真卡上跑过"变成一条**可复现**的
    证据（或者一条明确的 `unavailable`）。默认**开关关闭**，需显式 `--probe-cuda`。

    返回里带 `is_real_gpu_probe=True`，但 `is_training=False`：
    这条证据**不能**用来证明任何训练内存结论。
    """
    active = reader if reader is not None else CudaMemoryReader()
    if not getattr(active, "available", False):
        return {
            "probe": "cuda-reader",
            "available": False,
            "reason": str(getattr(active, "reason", "")),
            "is_real_gpu_probe": False,
            "is_training": False,
        }
    torch = active.torch
    bytes_per_mib = 1 << 20
    started = time.monotonic()
    active.reset()
    device = "cuda:0"
    tensor = torch.ones(int(mib) * bytes_per_mib // 4, dtype=torch.float32, device=device)
    tensor = tensor + 1.0
    torch.cuda.synchronize()
    peaks = active.peaks_gib()
    allocated_bytes = int(tensor.numel() * tensor.element_size())
    del tensor
    torch.cuda.empty_cache()
    return {
        "probe": "cuda-reader",
        "available": True,
        "device": peaks.get("gpu_device_name"),
        "requested_mib": int(mib),
        "allocated_bytes": allocated_bytes,
        "peak_gpu_gib": peaks.get("peak_gpu_gib"),
        "peak_gpu_reserved_gib": peaks.get("peak_gpu_reserved_gib"),
        "free_gpu_gib": active.free_gib(),
        "provenance": str(getattr(active, "provenance", "")),
        "wall_seconds": round(time.monotonic() - started, 4),
        "is_real_gpu_probe": True,
        "is_training": False,
        "note": (
            "受控极小探针：只验证显存读数路径可用（分配 → 加一 → 释放）；"
            "不是训练、不是模型加载、不构成内存门槛证据。"
        ),
    }
