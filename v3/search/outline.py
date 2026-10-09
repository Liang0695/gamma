"""符号轮廓抽取：支持缩进方法、`async def`、嵌套定义与 decorator。

KAGGLE-22 §3.4 阶段 3 与 KAGGLE-19 §4 明确要求：
「函数检索必须支持缩进方法、async def、嵌套和 decorator；22 示例 `^(class|def)` 会漏方法，
执行实现须用合适模式或任务内 AST。」

本模块用 `ast` 为主、正则为辅：
- AST 给出准确的 start/end 行与限定名（含嵌套与 decorator 起始行）；
- 正则兜底用于语法不完整或超长文件被裁剪的情况，此时标记 `source="regex"`。
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Sequence

#: 兜底正则：允许前置空白，覆盖 async def / def / class，并单独抓 decorator 行。
_REGEX_DEF = re.compile(
    r"^(?P<indent>[ \t]*)(?:async[ \t]+)?(?P<kind>def|class)[ \t]+(?P<name>[A-Za-z_][\w]*)",
)
_REGEX_DECORATOR = re.compile(r"^[ \t]*@")


@dataclass
class Symbol:
    name: str
    qualname: str
    kind: str  # function | async_function | class | method | async_method | nested_function
    start_line: int  # 1-indexed，含 decorator 的第一行
    end_line: int  # 1-indexed，闭区间
    decorators: list[str] = field(default_factory=list)
    is_async: bool = False
    parent: str | None = None
    source: str = "ast"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "qualname": self.qualname,
            "kind": self.kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "decorators": list(self.decorators),
            "is_async": self.is_async,
            "parent": self.parent,
            "source": self.source,
        }


def _decorator_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for dec in getattr(node, "decorator_list", []) or []:
        try:
            names.append(ast.unparse(dec))
        except Exception:  # pragma: no cover - 极端语法
            names.append("<unparsable>")
    return names


def _kind_for(node: ast.AST, is_method: bool, nested: bool) -> str:
    is_async = isinstance(node, ast.AsyncFunctionDef)
    if isinstance(node, ast.ClassDef):
        return "class"
    if is_method:
        return "async_method" if is_async else "method"
    if nested:
        return "nested_async_function" if is_async else "nested_function"
    return "async_function" if is_async else "function"


def outline_from_source(text: str, module: str = "<module>") -> list[Symbol]:
    """用 AST 抽取全部符号（含嵌套）。语法错误时退回正则。"""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return outline_regex(text, module=module)

    lines = text.splitlines()
    symbols: list[Symbol] = []

    def visit(node: ast.AST, prefix: str, in_class: bool, depth: int) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                decorators = _decorator_names(child)
                start = child.lineno
                if decorators and child.decorator_list:
                    start = min(start, min(d.lineno for d in child.decorator_list))
                end = getattr(child, "end_lineno", start) or start
                name = child.name
                qualname = "%s.%s" % (prefix, name) if prefix else name
                is_method = in_class and isinstance(
                    child, (ast.FunctionDef, ast.AsyncFunctionDef)
                )
                nested = depth > 0 and not in_class
                symbols.append(
                    Symbol(
                        name=name,
                        qualname=qualname,
                        kind=_kind_for(child, is_method, nested),
                        start_line=start,
                        end_line=end,
                        decorators=decorators,
                        is_async=isinstance(child, ast.AsyncFunctionDef),
                        parent=prefix or None,
                    )
                )
                visit(
                    child,
                    qualname,
                    in_class=isinstance(child, ast.ClassDef),
                    depth=depth + 1,
                )

    visit(tree, "", in_class=False, depth=0)
    if not symbols and lines:
        # ast 解析成功但没有符号：仍然是合法结果（例如纯赋值模块）
        return []
    symbols.sort(key=lambda s: (s.start_line, s.name))
    return symbols


def outline_regex(text: str, module: str = "<module>") -> list[Symbol]:
    """正则兜底。会正确捕获缩进方法与 `async def`，但嵌套层级需要按缩进推断。"""
    lines = text.splitlines()
    symbols: list[Symbol] = []
    pending_decorators: list[tuple[int, str]] = []
    stack: list[tuple[int, Symbol]] = []

    for index, line in enumerate(lines, start=1):
        if _REGEX_DECORATOR.match(line):
            pending_decorators.append((index, line.strip()))
            continue
        match = _REGEX_DEF.match(line)
        if not match:
            if line.strip():
                pending_decorators = []
            continue
        indent = len(match.group("indent").expandtabs(4))
        name = match.group("name")
        kind_token = match.group("kind")
        is_async = "async" in line.split("def")[0] and kind_token == "def"

        while stack and stack[-1][0] >= indent:
            _, parent_symbol = stack.pop()
            parent_symbol.end_line = max(parent_symbol.start_line, index - 1)

        parent = stack[-1][1] if stack else None
        start = pending_decorators[0][0] if pending_decorators else index
        decorators = [text for _, text in pending_decorators]
        if kind_token == "class":
            kind = "class"
        elif parent is not None and parent.kind == "class":
            kind = "async_method" if is_async else "method"
        elif parent is not None:
            kind = "nested_async_function" if is_async else "nested_function"
        else:
            kind = "async_function" if is_async else "function"
        qualname = "%s.%s" % (parent.qualname, name) if parent else name
        symbol = Symbol(
            name=name,
            qualname=qualname,
            kind=kind,
            start_line=start,
            end_line=len(lines),
            decorators=decorators,
            is_async=is_async,
            parent=parent.qualname if parent else None,
            source="regex",
        )
        symbols.append(symbol)
        stack.append((indent, symbol))
        pending_decorators = []

    while stack:
        _, symbol = stack.pop()
        symbol.end_line = max(symbol.end_line, len(lines))
    symbols.sort(key=lambda s: (s.start_line, s.name))
    return symbols


def find_symbol(symbols: Sequence[Symbol], name: str) -> list[Symbol]:
    """按短名或限定名查找；短名可能多义（官方 resolve_node_name 第 4 档是 substring 匹配）。"""
    exact = [s for s in symbols if s.qualname == name]
    if exact:
        return exact
    return [s for s in symbols if s.name == name or s.qualname.endswith("." + name)]


def render_outline(symbols: Sequence[Symbol], limit_lines: int = 40) -> list[str]:
    """紧凑渲染成 `start-end kind qualname` 行；用于两段式收窄输出。"""
    lines = [
        "%d-%d %s %s" % (s.start_line, s.end_line, s.kind, s.qualname)
        for s in symbols
    ]
    return lines[:limit_lines]
