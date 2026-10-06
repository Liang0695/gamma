"""统一的 fail-closed 异常与断言助手。

设计原则（来自 KAGGLE-19 整合稿 §3/§6 与 KAGGLE-21 §5）：
- 缺字段、缺锁定、哈希不一致、许可未闭合、无效 adapter —— 一律抛异常并停止，
  不允许"猜一个默认值继续跑"。
- 任何配置里出现 latest / * / 空 SHA，都必须在这里被拒绝。
"""

from __future__ import annotations


class FailClosed(Exception):
    """所有 fail-closed 停止条件的基类。

    抛出即代表：本步骤没有合法输入，不得继续、不得自动降级、不得自动扩大预算。
    """

    exit_code = 3

    def __init__(self, code: str, message: str, **context: object) -> None:
        self.code = code
        self.message = message
        self.context = dict(context)
        super().__init__("[%s] %s" % (code, message))

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "context": self.context}


class MissingInput(FailClosed):
    """必需输入缺失（字段、文件、锁定值、来源）。"""

    exit_code = 4


class UnverifiedLock(FailClosed):
    """依赖/版本/模板锁定值尚未验证 —— 禁止自动取 latest 代替。"""

    exit_code = 5


class IntegrityError(FailClosed):
    """哈希 / 树 / manifest 不一致。"""

    exit_code = 6


class PolicyViolation(FailClosed):
    """越权或违反隔离策略（gold 泄漏、写补丁目录、GPU 自动开工等）。"""

    exit_code = 7


class Blocked(FailClosed):
    """环境缺依赖或外部条件不满足，明确阻断而不是编造结果。"""

    exit_code = 8


def require(condition: bool, code: str, message: str, **context: object) -> None:
    """最常用的 fail-closed 断言。"""
    if not condition:
        raise FailClosed(code, message, **context)


def require_field(obj: dict, field: str, where: str):
    """取必需字段，缺失即阻断（等价于 schema 里的"省略表示非法"）。"""
    if not isinstance(obj, dict) or field not in obj:
        raise MissingInput("missing_field", "缺少必需字段 %r（位于 %s）" % (field, where))
    return obj[field]


def reject_unpinned(value, where: str) -> str:
    """拒绝 latest / * / 空值 / 未固定版本号。

    这条规则同时覆盖 §5.3 "T0 不能通过时允许一次记录明确的版本修复" 的前提：
    必须先有人写出确定版本号，runner 才允许继续。
    """
    if value is None:
        raise UnverifiedLock("unpinned_value", "%s 未固定（None）" % where)
    text = str(value).strip()
    if text == "":
        raise UnverifiedLock("unpinned_value", "%s 为空字符串，视为未固定" % where)
    lowered = text.lower()
    for bad in ("latest", "head", "master", "main", "*", "any", ">=latest", "unpinned"):
        if lowered == bad:
            raise UnverifiedLock(
                "unpinned_value",
                "%s=%r 属于浮动版本，配置必须 fail-closed" % (where, text),
                value=text,
            )
    if text.startswith(">=") or text.startswith("~=") or text.startswith(">"):
        raise UnverifiedLock(
            "unpinned_range", "%s=%r 是范围约束，不是精确锁定" % (where, text), value=text
        )
    return text
