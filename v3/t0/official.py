"""T0：官方 schema / template / parser 接口锁定。

来源：KAGGLE-22 §1.1–§1.4（一手核对官方 HARNESS_README 与图/嵌入载荷）、
KAGGLE-20 §3–§4（模型 revision、模板 SHA、parser 约束）、KAGGLE-19 §4/§6。

两条硬规则：
1. **不猜**：任何未取得的值写 `unverified`，取值时抛 `UnverifiedLock`，禁止用示例补齐。
2. **不装**：本模块只描述官方注册表，不实现、不替换任何工具。
"""

from __future__ import annotations

import json
from typing import Mapping

from ..common.canonical import sha256_json
from ..common.errors import MissingInput, PolicyViolation, UnverifiedLock

#: 官方注册的 9 个工具。提交侧只能声明子集，不能自定义实现。
OFFICIAL_TOOLS = (
    "run_command",
    "read_file",
    "get_code_neighbors",
    "search_similar_code",
    "get_code_subgraph",
    "edit_file",
    "write_file",
    "submit_patch",
    "get_status",
)

#: 计入 tool_calls 的工具（submit_patch / get_status 不计）。
BILLED_TOOLS = tuple(t for t in OFFICIAL_TOOLS if t not in ("submit_patch", "get_status"))


class OfficialInterface:
    """官方接口常量 + 工具参数 schema（fail-closed 校验）。"""

    def __init__(self, payload: Mapping) -> None:
        self.payload = dict(payload)

    @classmethod
    def from_file(cls, path: str) -> "OfficialInterface":
        with open(path, "r", encoding="utf-8") as handle:
            return cls(json.load(handle))

    def interface_sha256(self) -> str:
        return sha256_json(self.payload)

    # ---- 硬上限 ----

    def limit(self, name: str) -> int:
        limits = self.payload.get("limits") or {}
        if name not in limits:
            raise MissingInput("unknown_limit", "未登记的官方上限：%s" % name)
        return int(limits[name])

    @property
    def eval_budget(self) -> dict:
        return dict(self.payload.get("evaluation_budget") or {})

    # ---- 工具 schema ----

    def tools(self) -> dict:
        return dict(self.payload.get("tools") or {})

    def tool_names(self) -> tuple[str, ...]:
        return tuple(sorted(self.tools()))

    def assert_registry_complete(self) -> None:
        """注册表必须与官方 9 工具集合一致，多一个少一个都算阻断。"""
        declared = set(self.tools())
        expected = set(OFFICIAL_TOOLS)
        if declared != expected:
            raise PolicyViolation(
                "tool_registry_mismatch",
                "工具注册表与官方不一致",
                missing=sorted(expected - declared),
                extra=sorted(declared - expected),
            )

    def validate_call(self, tool_name: str, arguments: Mapping) -> dict:
        """校验一次工具调用：未知工具、未知字段、缺必需字段一律拒绝。

        「工具 schema 来自实际官方注册表，未知字段不得靠示例补齐。」
        """
        tools = self.tools()
        if tool_name not in tools:
            raise PolicyViolation(
                "unknown_tool", "未知工具 %r（不得用示例补齐）" % (tool_name,)
            )
        spec = tools[tool_name]
        allowed = set(spec.get("args") or {})
        required = set(spec.get("required") or ())
        given = set(arguments or {})
        unknown = sorted(given - allowed)
        if unknown:
            raise PolicyViolation(
                "unknown_tool_field",
                "%s 收到未知字段 %s" % (tool_name, unknown),
            )
        missing = sorted(required - given)
        if missing:
            raise MissingInput(
                "missing_tool_field", "%s 缺少必需字段 %s" % (tool_name, missing)
            )
        return dict(arguments)

    def bills_tool_calls(self, tool_name: str) -> bool:
        return tool_name in BILLED_TOOLS

    # ---- 版本/模板 pin ----

    def pin(self, key: str) -> str:
        pins = self.payload.get("pins") or {}
        if key not in pins:
            raise MissingInput("unknown_pin", "未登记的 pin：%s" % key)
        entry = pins[key]
        if not isinstance(entry, dict):
            # Q0 🔵-3：裸字符串 pin 旧实现被当作 `verified=True`，与模块开头"不猜"的规则冲突。
            # 现在一律 UnverifiedLock —— 没有 verified 标志就不允许当作已确认事实。
            raise UnverifiedLock(
                "pin_entry_not_structured",
                "pin %s 是裸值而非带 verified 标志的结构，无法确认已核实" % key,
                pin=entry,
            )
        if not entry.get("verified"):
            raise UnverifiedLock(
                "pin_unverified",
                "pin %s 未验证，禁止当作已确认事实使用" % key,
                pin=entry,
            )
        return str(entry.get("value"))

    def pin_entry(self, key: str) -> dict:
        pins = self.payload.get("pins") or {}
        if key not in pins:
            raise MissingInput("unknown_pin", "未登记的 pin：%s" % key)
        entry = pins[key]
        if not isinstance(entry, dict):
            raise UnverifiedLock(
                "pin_entry_not_structured",
                "pin %s 是裸值而非带 verified 标志的结构，无法确认已核实" % key,
                pin=entry,
            )
        return dict(entry)

    def verified_pins(self) -> dict:
        pins = self.payload.get("pins") or {}
        out = {}
        for key, entry in pins.items():
            if isinstance(entry, dict):
                out[key] = bool(entry.get("verified"))
            else:
                out[key] = True
        return out
