"""KAGGLE-38 r3 · 官方 Gemma 4 chat template 的**真实 tokenizer 渲染回归**。

## 为什么需要这个回归

Mika 2026-10-08 裁定（KAGGLE-28）：V3 训练与首次工程检查必须以模型 revision
`52f3f65bc7a02d555763bc923bd1d9094898219d` 的**官方原件**作为配置、tokenizer 与
template 输入。现场实测的坑是：`~/gemma4/prep/model/` 里躺着一份被改过的**工作副本** ——
官方 `chat_template.jinja` 在 `function['arguments']` 是字符串时调用 `raise_exception(...)`
直接拒绝，而改写版（224 行重写）把原始字符串静默打印出去。

只核 SHA 说服不了评审，因为「锁里核的是官方 SHA、生产加载的是改过的文件」两件事可以
各说各话。本回归因此做两件事：

1. 用 `v3.train.input_binding` 把**官方原件**绑定到锁里的 pin，把字节**复制**进一个受控
   加载视图，并对副本重算 SHA-256 —— 证明渲染器读到的字节就是被逐字节核对过的那批；
2. 用**真实 Jinja 环境**渲染官方模板本身，跑十个行为用例。其中
   `string_arguments_rejected` 正是"官方模板 vs 改写版"的行为分叉点。

## 渲染器选择（诚实边界）

优先级：

1. `transformers.PreTrainedTokenizerBase.apply_chat_template`（可导入时优先）——
   这是**生产加载路径**的真实入口；
2. `jinja2` + 本文件里的 **transformers 兼容 shim**。shim 复刻
   `transformers/utils/chat_template_utils.py::_compile_jinja_template` 提供的东西：
   `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
   extensions=["jinja2.ext.loopcontrols"])`，加上 `raise_exception` / `tojson` /
   `strftime_now` 三个 global，以及 transformers 用的异常类
   `jinja2.exceptions.TemplateError`。

**shim 是本地替身（local stand-in），不是生产加载路径。** 它证明"官方模板文本在这些输入
下的行为"，**不证明**生产加载器实际读到哪份模板，也**不**等价于 `apply_chat_template`
的全部语义（special token 处理、多模态 content 归一化等）。报告里
`renderer.local_stand_in=true` 会把它显式标红，结论必须在目标解释器
`/home/scc/pb24511961/v3/envs/py3119/bin/python`（锁定栈：transformers + jinja2）上复跑确认。

两者都不可用时**不假装通过**：输出 `status="blocked"` 的 JSON 报告并返回退出码 8
（`v3.common.errors.Blocked.exit_code`），报告里写明缺哪个模块与目标解释器命令。

## 本机装 jinja2 的方式（目录级、非系统安装）

本机（Windows 工作区）没有 jinja2 / transformers，而且**不允许**把依赖装进包内
（那会进交付 ZIP）。用一个包目录之外的 vendor 目录：

    python -m pip install --target "<工作区>\\_vendor" jinja2
    $env:PYTHONPATH = "<工作区>\\_vendor"

`<工作区>` 是包根 `gamma-e0-kaggle-38/` 的**父目录**（本机实际落在它的上一级
`workdir/_vendor`，两者都在包根之外）。本机实测 pip 直装会撞沙箱权限
（`PermissionError` on `pip-unpack-*/...whl`），实际落地方式是下载 wheel 后用
`zipfile` 解包到该 vendor 目录；两种方式得到的是同一个 jinja2 3.1.6 + markupsafe 3.0.3。

没有 PYTHONPATH 时，本文件会主动探测包根**父目录**及其**上一级**下的 `_vendor`
（`--vendor-dir` / `--no-vendor-probe` 可覆盖），并把探测结果记进报告的 `vendor_probe`。

## 用法

    python tools/render_chat_template_regression.py --out <report.json>

退出码：`0` 全部用例符合预期；`1` 有用例不符（报告 `failed_cases` 列出）；
`8` blocked（缺渲染器，报告说明缺什么）。
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from v3.common.errors import Blocked, FailClosed, MissingInput, PolicyViolation  # noqa: E402
from v3.train import input_binding as ib  # noqa: E402

#: 官方原件所在的证据目录（本回归的默认输入根）。
DEFAULT_INPUTS_ROOT = os.path.join(
    REPO_ROOT, "docs", "v3", "evidence", "kaggle-38-g9-evidence"
)
#: 接口锁：`chat_template_sha256` 必须**单列**，不得用 `tokenizer_config_sha256` 顶替。
DEFAULT_INTERFACE = os.path.join(REPO_ROOT, "v3", "locks", "official-interface.json")
#: 现场快照是一层**平铺**的证据目录，与 `input_binding.DEFAULT_SNAPSHOT_LAYOUT`
#: （official/ + shared/ 分组）不同，所以这里必须显式给布局。
FLAT_LAYOUT = {
    "tokenizer_config.json": "tokenizer_config.json",
    "tokenizer.json": "tokenizer.json",
    "chat_template.jinja": "chat_template.jinja",
    # r4 · 缺陷 3：`config.json` 从"无 pin 伴随文件"升为受核对输入，
    # 所以布局里必须显式带上它（证据目录里有这份原件）。
    "config.json": "config.json",
}
#: 官方模板的 SHA-256（等于锁里的 `chat_template_sha256`，两处必须一致）。
OFFICIAL_TEMPLATE_SHA256 = (
    "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
)
#: 目标解释器：锁定栈里 transformers 与 jinja2 都在，shim 结果必须在它上面复跑确认。
TARGET_INTERPRETER = "/home/scc/pb24511961/v3/envs/py3119/bin/python"
TARGET_COMMAND = (
    "%s tools/render_chat_template_regression.py --out <report.json>" % TARGET_INTERPRETER
)
#: 包根的父目录（工作区根）。vendor 目录只允许放在包根**之外**，绝不放进包内。
WORKSPACE_ROOT = os.path.dirname(REPO_ROOT)
#: jinja2 目录级安装的候选位置：包根父目录下的 `_vendor`，以及它上一级下的 `_vendor`
#: （本机实际用的是后者，即任务指定的 `...\workdir\_vendor`）。
VENDOR_DIR_CANDIDATES = (
    os.path.join(WORKSPACE_ROOT, "_vendor"),
    os.path.join(os.path.dirname(WORKSPACE_ROOT), "_vendor"),
)
DEFAULT_VENDOR_DIR = VENDOR_DIR_CANDIDATES[0]
DEFAULT_VIEW_DIR = os.path.join(WORKSPACE_ROOT, "_tmp", "chat_template_load_view")
DEFAULT_WORK_DIR = os.path.join(WORKSPACE_ROOT, "_tmp", "chat_template_render_work")

RENDERER_TRANSFORMERS = "transformers"
RENDERER_SHIM = "jinja2-shim"
RENDERER_NONE = "none"

#: 官方模板在 tokenizer_config.json 里**没有** `chat_template` 键（本机实测），
#: 模板是独立文件，所以 `bos_token` 等只能从 config 读、模板必须按自己的 pin 绑定。
BOS_TOKEN_FALLBACK = "<bos>"
SPECIAL_TOKEN_KEYS = ("bos_token", "eos_token", "pad_token", "unk_token")

#: thinking 相关的两个**不同**标记，必须分开断言（见 `thinking_channel_contents`）：
#:   `<|think|>`            —— 模板顶部的「开启思考」开关标记（enable_thinking=True 才有）
#:   `<|channel>thought…<channel|>` —— 思考通道；`enable_thinking=False` 时
#:                                     `add_generation_prompt` 会补一个**空的**已闭合通道
THINK_ENABLE_MARKER = "<|think|>"
THINK_CHANNEL_OPEN = "<|channel>thought\n"
THINK_CHANNEL_CLOSE = "<channel|>"

CAVEAT_SHIM = (
    "jinja2-shim 是**本地替身**：它复刻 transformers 的 Jinja 环境与三个 global，"
    "但不是生产加载路径，也不覆盖 apply_chat_template 的全部语义。"
    "本报告的渲染结论必须在目标解释器上用锁定栈复跑确认。"
)


# --------------------------------------------------------------------- 小工具


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


#: 导入探测结果缓存：`None` 表示可用，否则是 (异常类型, 消息)。
_IMPORT_ERRORS: dict = {}


def import_error(name: str):
    """**真的导入一次**，返回 `None`（可用）或 `(异常类型, 消息)`。结果缓存。

    为什么不用 `importlib.util.find_spec`：`find_spec` 能找到 ≠ 能导入。本机实测
    `transformers` 就在 `AppData\\Roaming\\Python\\Python313\\site-packages` 里，
    `find_spec` 成功，但 `import transformers` 直接
    `ModuleNotFoundError: No module named 'httpx'` —— 只看 `find_spec` 会把这种
    "半装的依赖"误判成生产渲染路径可用。
    """
    if name not in _IMPORT_ERRORS:
        try:
            __import__(name)
            _IMPORT_ERRORS[name] = None
        except Exception as exc:
            _IMPORT_ERRORS[name] = (type(exc).__name__, str(exc)[:300])
    return _IMPORT_ERRORS[name]


def _module_available(name: str) -> bool:
    return import_error(name) is None


def _import_probe() -> dict:
    """把两个候选渲染器模块的导入探测结果记进报告（含失败原因）。"""
    probe = {}
    for name in ("transformers", "jinja2"):
        error = import_error(name)
        probe[name] = (
            None
            if error is None
            else {"type": error[0], "message": error[1], "usable": False}
        )
    return probe


def _load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def thinking_channel_contents(text: str) -> list:
    """抽出所有思考通道的**内容**（`<|channel>thought` 与 `<channel|>` 之间的部分）。

    `enable_thinking=False` 时模板会输出一个**空的**已闭合通道 —— 那是"推理已被消费"
    的占位，不是"开启了思考"。分开断言才不会把两者混为一谈。
    """
    parts: list = []
    cursor = 0
    while True:
        start = text.find(THINK_CHANNEL_OPEN, cursor)
        if start < 0:
            return parts
        body_start = start + len(THINK_CHANNEL_OPEN)
        close = text.find(THINK_CHANNEL_CLOSE, body_start)
        if close < 0:
            parts.append(text[body_start:])
            return parts
        parts.append(text[body_start:close])
        cursor = close + len(THINK_CHANNEL_CLOSE)


def check(record: dict, name: str, condition: bool, detail: str = "") -> bool:
    """用例内的结构断言：通过记进 `checks`，不通过记进 `check_failures`。"""
    entry = {"check": name, "passed": bool(condition)}
    if detail and not condition:
        entry["detail"] = detail
    if condition:
        record["checks"].append(entry)
    else:
        record["check_failures"].append(entry)
    return bool(condition)


def _render(record: dict, ctx: dict, renderer, messages, **kwargs):
    """统一的渲染入口：顺手记下输入，供"合成改写版"对照使用。"""
    record["_stand_in_call"] = {"messages": messages, "kwargs": dict(kwargs)}
    return renderer.render(ctx["template_text"], messages, **kwargs)


# --------------------------------------------------------------------- 渲染器


def _raise_exception(message):
    """transformers 注册的 `raise_exception` global：直接抛模板错误。

    异常类沿用 transformers 口径 —— `jinja2.exceptions.TemplateError`
    （`_compile_jinja_template` 里注册的就是它）。
    """
    from jinja2.exceptions import TemplateError

    raise TemplateError(str(message))


def _tojson(value, indent=None):
    """transformers 注册的 `tojson` 过滤器（普通 `json.dumps`，不做 HTML 转义）。

    诚实说明：官方模板**一次都没用到 `tojson`**（全文件只有 `raise_exception` 一处
    global 调用），注册它只是为了让 shim 的环境与 transformers 对齐。
    """
    return json.dumps(value, ensure_ascii=False, indent=indent)


def _strftime_now(fmt: str = "%Y-%m-%d") -> str:
    """transformers 注册的 `strftime_now` global（官方模板当前未使用）。"""
    return datetime.datetime.now().strftime(fmt)


class TransformersRenderer:
    """**生产入口**：`PreTrainedTokenizerBase.apply_chat_template`。"""

    kind = RENDERER_TRANSFORMERS
    local_stand_in = False
    shim_globals: tuple = ()

    def __init__(self, view_dir: str, template_text: str) -> None:
        import transformers
        from transformers import AutoTokenizer

        self.transformers_version = getattr(transformers, "__version__", "unknown")
        self.exception_class = "jinja2.exceptions.TemplateError (由 transformers 转抛)"
        self._tok = AutoTokenizer.from_pretrained(view_dir, local_files_only=True)
        self._tok.chat_template = template_text
        self.tokenizer_class = type(self._tok).__name__

    def render(self, template_text: str, messages, **kwargs) -> str:
        # 每次都用**被绑定模板的文本**覆盖，确保渲染的就是被核过的那份字节。
        self._tok.chat_template = template_text
        return self._tok.apply_chat_template(messages, tokenize=False, **kwargs)

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "local_stand_in": self.local_stand_in,
            "implementation": "transformers.PreTrainedTokenizerBase.apply_chat_template",
            "transformers_version": self.transformers_version,
            "tokenizer_class": self.tokenizer_class,
            "exception_class": self.exception_class,
        }


class Jinja2ShimRenderer:
    """**本地替身**：`jinja2` + transformers 兼容 shim（不是生产加载路径）。"""

    kind = RENDERER_SHIM
    local_stand_in = True
    shim_globals = ("raise_exception", "tojson", "strftime_now")

    def __init__(self, *, tokens=None) -> None:
        import jinja2
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        self.jinja2_version = getattr(jinja2, "__version__", "unknown")
        self.exception_class = "jinja2.exceptions.TemplateError"
        self._env = ImmutableSandboxedEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
            extensions=["jinja2.ext.loopcontrols"],
        )
        self._env.filters["tojson"] = _tojson
        self._env.globals["raise_exception"] = _raise_exception
        self._env.globals["strftime_now"] = _strftime_now
        tokens = dict(tokens or {})
        self._tokens = {key: tokens.get(key) for key in SPECIAL_TOKEN_KEYS}
        if not self._tokens.get("bos_token"):
            self._tokens["bos_token"] = BOS_TOKEN_FALLBACK

    def render(self, template_text: str, messages, **kwargs) -> str:
        template = self._env.from_string(template_text)
        template_kwargs = {
            "messages": messages,
            "add_generation_prompt": False,
        }
        template_kwargs.update(self._tokens)
        # transformers 的 `_render_jinja_template` 也是这么合并的：显式 kwargs 覆盖默认。
        template_kwargs.update(kwargs)
        return template.render(**template_kwargs)

    def describe(self) -> dict:
        return {
            "kind": self.kind,
            "local_stand_in": self.local_stand_in,
            "implementation": "%s::Jinja2ShimRenderer" % os.path.relpath(__file__, REPO_ROOT),
            "jinja2_version": self.jinja2_version,
            "shim_globals": list(self.shim_globals),
            "shim_filters": ["tojson"],
            "exception_class": self.exception_class,
            "environment": (
                "ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, "
                "extensions=['jinja2.ext.loopcontrols'])"
            ),
            "bos_token": self._tokens.get("bos_token"),
            "caveat": CAVEAT_SHIM,
        }


_VENDOR_USED = None


def probe_vendor_dir(vendor_dir: str | None = None):
    """没装 jinja2 时探测包根之外的 `_vendor`（目录级安装），返回实际用的目录。"""
    global _VENDOR_USED
    if _module_available("jinja2"):
        return None
    candidates = ([vendor_dir] if vendor_dir else []) + list(VENDOR_DIR_CANDIDATES)
    for candidate in candidates:
        if not os.path.isdir(candidate):
            continue
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        # 目录刚进 sys.path：必须清掉导入探测缓存，否则会拿着"上一次失败"的结论下判断。
        _IMPORT_ERRORS.pop("jinja2", None)
        if not _module_available("jinja2"):
            continue
        _VENDOR_USED = candidate
        return candidate
    return None


def renderer_kind() -> str:
    """本地可用的渲染器种类（不构造实例，供上层 `skipUnless` 使用）。"""
    if _module_available("transformers"):
        return RENDERER_TRANSFORMERS
    if _module_available("jinja2"):
        return RENDERER_SHIM
    return RENDERER_NONE


def build_renderer(ctx: dict, *, allow_fallback: bool = True):
    """构造渲染器，返回 `(renderer_or_None, notes)`。"""
    notes: list = []
    if _module_available("transformers"):
        try:
            return TransformersRenderer(ctx["view_dir"], ctx["template_text"]), notes
        except Exception as exc:  # pragma: no cover - 目标解释器上才会走到成功分支
            notes.append(
                "transformers 可导入，但从加载视图构造 tokenizer 失败：%s: %s"
                % (type(exc).__name__, exc)
            )
            if not allow_fallback:
                raise
    if _module_available("jinja2"):
        if not notes and not _module_available("transformers"):
            notes.append("本机没有 transformers，改用 jinja2 + transformers 兼容 shim（本地替身）")
        return Jinja2ShimRenderer(tokens=ctx["special_tokens"]), notes
    notes.append("transformers 与 jinja2 都不可导入：无法渲染，本回归 blocked（不假装通过）")
    return None, notes


# --------------------------------------------------------------------- 上下文


def build_context(
    *,
    inputs_root: str | None = None,
    interface_path: str | None = None,
    view_dir: str | None = None,
    work_dir: str | None = None,
) -> dict:
    """绑定官方输入 → 装配加载视图 → 读回被绑定的模板文本。"""
    inputs_root = os.path.abspath(inputs_root or DEFAULT_INPUTS_ROOT)
    interface_path = os.path.abspath(interface_path or DEFAULT_INTERFACE)
    view_dir = os.path.abspath(view_dir or DEFAULT_VIEW_DIR)
    work_dir = os.path.abspath(work_dir or DEFAULT_WORK_DIR)
    os.makedirs(work_dir, exist_ok=True)

    binding = ib.bind_model_inputs(
        root=inputs_root, interface_path=interface_path, layout=FLAT_LAYOUT
    )
    view = ib.assemble_load_view(binding, view_dir)
    template_text = ib.read_chat_template(view["chat_template_path"])
    tokenizer_config = _load_json(view["tokenizer_config_path"])
    special_tokens = {key: tokenizer_config.get(key) for key in SPECIAL_TOKEN_KEYS}
    if not special_tokens.get("bos_token"):
        special_tokens["bos_token"] = BOS_TOKEN_FALLBACK
    return {
        "inputs_root": inputs_root,
        "interface_path": interface_path,
        "binding": binding,
        "view": view,
        "view_dir": view_dir,
        "work_dir": work_dir,
        "template_text": template_text,
        "template_sha256": ib.sha256_file(view["chat_template_path"]),
        "template_bytes": os.path.getsize(view["chat_template_path"]),
        "special_tokens": special_tokens,
        "tokenizer_config": tokenizer_config,
    }


def synthesise_modified_template(official_text: str) -> str:
    """由官方模板**就地**合成一个"改写版"替身：把 `raise_exception` 换成直接打印原始字符串。

    边界（必须说清）：仓库里**没有**现场那份 224 行工作副本，本回归拿到的是官方原件。
    所以本函数合成的只是一份**替身（stand-in）**，它只复刻评审点名的那一处行为分叉
    （字符串 `arguments` 由"拒绝"改成"静默打印"），**不是**现场文件本身，也不代表
    改写版的其它 200 多行差异。

    它有两个用途：证明本回归对这两份模板**能分辨**；以及在
    `reject_official_verified_but_modified_loaded` 里充当"被改过的 template"。
    """
    start_marker = "{{- raise_exception("
    end_marker = ") -}}"
    if start_marker not in official_text:
        raise MissingInput(
            "chat_template_raise_exception_missing",
            "官方模板里找不到 raise_exception 分支：无法合成改写版替身（模板可能已被换掉）",
        )
    start = official_text.index(start_marker)
    end = official_text.index(end_marker, start) + len(end_marker)
    return official_text[:start] + "{{- function['arguments'] -}}" + official_text[end:]


def _link_or_copy(src: str, dst: str) -> str:
    """大文件优先硬链接（32MB 的 tokenizer.json 不必真复制）。

    硬链接不是 symlink：`resolve_input_file` 解出的 realpath 仍在根内，不会绕过
    目录逃逸检查，也不会让 pin 核对失真（核的是同一份字节）。

    先删掉目标：上一次运行留下的硬链接会让 `os.link` 抛 `SameFileError`（src 与 dst
    已经是同一个文件），使重复运行不幂等。
    """
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copyfile(src, dst)
        return "copy"


def build_modified_root(ctx: dict, target_dir: str, *, template_text: str | None = None) -> dict:
    """造一个「官方 config/tokenizer + 被改过的 chat_template」的根，供拒绝用例使用。"""
    os.makedirs(target_dir, exist_ok=True)
    methods = {}
    # r4 · 缺陷 3：`config.json` 也是受核对输入了，拒绝用例必须连它一起摆好，
    # 否则会先撞 `input_file_missing`，测不到「模板被改」这条真正的判据。
    for name in ("tokenizer_config.json", "tokenizer.json", "config.json"):
        methods[name] = _link_or_copy(
            os.path.join(ctx["view_dir"], name), os.path.join(target_dir, name)
        )
    text = (
        template_text
        if template_text is not None
        else synthesise_modified_template(ctx["template_text"])
    )
    template_path = os.path.join(target_dir, "chat_template.jinja")
    with open(template_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return {
        "root": os.path.abspath(target_dir),
        "link_methods": methods,
        "template_path": template_path,
        "template_sha256": sha256_text(text),
        "official_template_sha256": ctx["template_sha256"],
        "differs_from_official": sha256_text(text) != ctx["template_sha256"],
        "note": (
            "被改过的 template 是**合成替身**（只复刻字符串 arguments 一处分叉），"
            "不是现场 224 行工作副本。"
        ),
    }


# --------------------------------------------------------------------- 用例输入


TOOL_DECLARATION = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name."}},
            "required": ["city"],
        },
    },
}

STRING_ARGUMENTS = '{"city": "Paris"}'
MAPPING_ARGUMENTS = {"city": "Paris"}
TOOL_CALL_ID = "call_weather_1"


def _weather_tool_call(arguments) -> dict:
    return {
        "id": TOOL_CALL_ID,
        "type": "function",
        "function": {"name": "get_weather", "arguments": arguments},
    }


def _tool_call_messages(arguments) -> list:
    return [
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "tool_calls": [_weather_tool_call(arguments)]},
        {
            "role": "tool",
            "tool_call_id": TOOL_CALL_ID,
            "name": "get_weather",
            "content": "21 C and sunny",
        },
    ]


# --------------------------------------------------------------------- 十个用例


def _case_mapping_arguments_positive(ctx, renderer, record):
    out = _render(
        record,
        ctx,
        renderer,
        _tool_call_messages(MAPPING_ARGUMENTS),
        tools=[TOOL_DECLARATION],
        add_generation_prompt=True,
    )
    check(
        record,
        "tool_call_rendered",
        '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>' in out,
    )
    check(record, "tool_response_rendered", "<|tool_response>response:get_weather{" in out)
    check(record, "raw_json_string_absent", '{"city"' not in out)
    check(record, "tool_declaration_rendered", "<|tool>declaration:get_weather" in out)
    return out


def _case_string_arguments_rejected(ctx, renderer, record):
    """官方模板必须拒绝字符串 arguments；合成改写版会静默打印 —— 两者的分叉点。"""
    stand_in_text = synthesise_modified_template(ctx["template_text"])
    stand_in_info = {
        "template_sha256": sha256_text(stand_in_text),
        "official_template_sha256": ctx["template_sha256"],
        "differs_from_official": sha256_text(stand_in_text) != ctx["template_sha256"],
        "note": "合成替身：把 raise_exception 分支换成直接打印原始字符串（不是现场工作副本）",
    }
    stand_in_out = renderer.render(
        stand_in_text,
        _tool_call_messages(STRING_ARGUMENTS),
        tools=[TOOL_DECLARATION],
        add_generation_prompt=True,
    )
    stand_in_info["outcome"] = "rendered"
    stand_in_info["same_as_official"] = False
    stand_in_info["rendered_preview"] = stand_in_out[:600]
    record["stand_in"] = stand_in_info
    check(record, "stand_in_does_not_reject", True)
    check(
        record,
        "stand_in_prints_raw_string",
        '{"city": "Paris"}' in stand_in_out,
        "改写版替身应当把原始 JSON 字符串直接打印出来",
    )
    check(
        record,
        "stand_in_lacks_rejection_message",
        "must be a JSON object" not in stand_in_out,
    )
    # 官方模板：必须抛（模板内的 raise_exception）
    return renderer.render(
        ctx["template_text"],
        _tool_call_messages(STRING_ARGUMENTS),
        tools=[TOOL_DECLARATION],
        add_generation_prompt=True,
    )


def _case_bad_json_rejected(ctx, renderer, record):
    record["raw_arguments"] = '{"city": "Paris"'
    try:
        ib.normalize_tool_arguments('{"city": "Paris"')
    except PolicyViolation as exc:
        record["normalize_code"] = exc.code
        raise
    record["normalize_code"] = None
    raise AssertionError("畸形 JSON 字符串没有被 normalize_tool_arguments 拒绝")


def _case_json_string_normalized(ctx, renderer, record):
    normalized = ib.normalize_tool_arguments(json.dumps(MAPPING_ARGUMENTS))
    record["normalized"] = {
        "source": normalized["source"],
        "value": normalized["value"],
    }
    check(record, "source_is_json_string", normalized["source"] == "json-string")
    check(record, "normalized_value_is_mapping", normalized["value"] == MAPPING_ARGUMENTS)
    out = _render(
        record,
        ctx,
        renderer,
        _tool_call_messages(normalized["value"]),
        tools=[TOOL_DECLARATION],
        add_generation_prompt=True,
    )
    check(
        record,
        "normalized_mapping_renders",
        '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>' in out,
    )
    check(record, "raw_json_string_absent", '{"city"' not in out)
    return out


def segment_between(text: str, start_marker: str, end_marker: str):
    """取 `start_marker` 之后、`end_marker` 之前的片段（任一标记缺失则返回 None）。"""
    if start_marker not in text or end_marker not in text:
        return None
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker, start)
    return text[start:end]


def _case_multi_turn_tool_continuation(ctx, renderer, record):
    tool_content = '{"temp_c": 21, "sky": "sunny"}'
    messages = [
        {"role": "user", "content": "What's the weather in Paris?"},
        {"role": "assistant", "tool_calls": [_weather_tool_call(MAPPING_ARGUMENTS)]},
        {
            "role": "tool",
            "tool_call_id": TOOL_CALL_ID,
            "name": "get_weather",
            "content": tool_content,
        },
        {"role": "user", "content": "And in Tokyo?"},
    ]
    out = _render(record, ctx, renderer, messages, add_generation_prompt=True)
    tool_call_marker = '<|tool_call>call:get_weather{city:<|"|>Paris<|"|>}<tool_call|>'
    check(record, "tool_call_marker_present", tool_call_marker in out)
    check(record, "tool_response_marker_present", "<|tool_response>response:get_weather{" in out)
    check(
        record,
        "tool_response_body_is_the_string_content",
        '<|tool_response>response:get_weather{value:<|"|>' in out
        and '"sky": "sunny"' in out,
    )
    check(record, "tool_response_block_closed", "<tool_response|>" in out)

    gap = segment_between(out, "<tool_response|>", "<|turn>user\nAnd in Tokyo?")
    check(
        record,
        "model_turn_closed_before_next_user_turn",
        gap is not None and gap.count("<turn|>") == 1 and gap.replace("<turn|>", "").strip() == "",
        "工具回合必须在下一个 user 回合开始前用 <turn|> 收口（实际间隔=%r）" % gap,
    )
    expected_tail = "<|turn>model\n" + THINK_CHANNEL_OPEN + THINK_CHANNEL_CLOSE
    check(
        record,
        "generation_prompt_opens_a_model_turn_with_consumed_thinking_placeholder",
        out.endswith(expected_tail),
        repr(out[-80:]),
    )
    check(record, "no_dangling_tool_response_wait", out.count("<|tool_response>") == 1)

    # 附带记录一条**真实模板行为**（不是本用例的判定条件）：
    # OpenAI 风格的 tool 响应若给 mapping，模板的 `is sequence` 判定会把 dict 当成序列，
    # 于是对**键**调用 .get() → UndefinedError。所以调用边界必须给字符串 content。
    probe_messages = list(messages)
    probe_messages[2] = dict(messages[2], content={"temp_c": 21, "sky": "sunny"})
    try:
        renderer.render(ctx["template_text"], probe_messages, add_generation_prompt=True)
        record["mapping_tool_content"] = {"outcome": "rendered"}
    except Exception as exc:
        record["mapping_tool_content"] = {
            "outcome": "rejected",
            "exception_type": type(exc).__name__,
            "message": str(exc)[:200],
            "note": (
                "官方模板前向扫描分支把 mapping 当 sequence、对键调用 .get() 而报错；"
                "OpenAI 风格 tool 响应必须传字符串 content（本用例用的就是字符串）。"
            ),
        }
    return out


def _case_turn_closure_forward_scan(ctx, renderer, record):
    continued = [
        {"role": "user", "content": "Summarise the repo."},
        {"role": "assistant", "content": "Part one."},
        {"role": "assistant", "content": "Part two."},
        {"role": "user", "content": "Now do it in Chinese."},
    ]
    ended = [
        {"role": "user", "content": "Summarise the repo."},
        {"role": "assistant", "content": "Done."},
        {"role": "user", "content": "Now do it in Chinese."},
    ]
    out_continued = renderer.render(
        ctx["template_text"], continued, add_generation_prompt=False
    )
    out_ended = renderer.render(ctx["template_text"], ended, add_generation_prompt=False)
    record["continued_output"] = out_continued
    record["ended_output"] = out_ended
    # 用"连续两个 assistant"这条更细的输入做改写版对照（它才是走 continuation 分支的那条）。
    record["_stand_in_call"] = {"messages": continued, "kwargs": {"add_generation_prompt": False}}
    record["stand_in_scope"] = "continued_scenario"
    check(
        record,
        "continued_scenario_keeps_one_model_turn",
        out_continued.count("<|turn>model") == 1,
        "连续两条 assistant 属于**同一个** model 回合，不得各开一次 <|turn>model",
    )
    between_parts = segment_between(out_continued, "Part one.", "Part two.")
    check(
        record,
        "continued_scenario_does_not_close_in_between",
        between_parts is not None and "<turn|>" not in between_parts,
        "回合未结束前不得在两条 assistant 之间插入 <turn|>（实际间隔=%r）" % between_parts,
    )
    check(
        record,
        "continued_scenario_content_is_contiguous",
        "Part one.Part two." in out_continued,
        out_continued,
    )
    check(
        record,
        "ended_scenario_opens_exactly_one_model_turn",
        out_ended.count("<|turn>model") == 1 and out_ended.count("<|turn>user") == 2,
        out_ended,
    )
    between_turns = segment_between(out_ended, "Done.", "<|turn>user\nNow do it in Chinese.")
    check(
        record,
        "ended_scenario_closes_turn_after_assistant_content",
        between_turns is not None
        and between_turns.count("<turn|>") == 1
        and between_turns.replace("<turn|>", "").strip() == "",
        "遇到下一个 user 之前必须恰好收口一次（实际间隔=%r）" % between_turns,
    )
    return out_continued


def _case_thinking_enabled(ctx, renderer, record):
    messages = [{"role": "user", "content": "Name three primes."}]
    out = _render(
        record, ctx, renderer, messages, add_generation_prompt=True, enable_thinking=True
    )
    check(record, "think_enable_marker_present", THINK_ENABLE_MARKER in out)
    check(record, "model_turn_opened", "<|turn>model\n" in out)
    check(
        record,
        "no_empty_thought_placeholder",
        THINK_CHANNEL_OPEN + THINK_CHANNEL_CLOSE not in out,
        "enable_thinking=True 时不应再补空的已闭合思考通道占位",
    )
    record["thinking_channel_contents"] = thinking_channel_contents(out)
    return out


def _case_thinking_disabled(ctx, renderer, record):
    messages = [{"role": "user", "content": "Name three primes."}]
    out = _render(
        record, ctx, renderer, messages, add_generation_prompt=True, enable_thinking=False
    )
    contents = thinking_channel_contents(out)
    record["thinking_channel_contents"] = contents
    record["empty_thought_placeholder_emitted"] = (
        THINK_CHANNEL_OPEN + THINK_CHANNEL_CLOSE in out
    )
    check(record, "think_enable_marker_absent", THINK_ENABLE_MARKER not in out)
    check(
        record,
        "no_thinking_channel_content",
        all(not item.strip() for item in contents),
        "enable_thinking=False 时思考通道不得带内容（空占位是允许的，见 empty_thought_placeholder_emitted）",
    )
    return out


def _case_reject_official_verified_but_modified_loaded(ctx, renderer, record):
    official = ib.bind_model_inputs(
        root=ctx["inputs_root"], interface_path=ctx["interface_path"], layout=FLAT_LAYOUT
    )
    official_files = {item["filename"]: item for item in official["files"]}
    record["official_binding_sha256"] = official["binding_sha256"]
    record["official_template"] = {
        "pin": official_files["chat_template.jinja"]["pin"],
        "declared_sha256": official_files["chat_template.jinja"]["declared_sha256"],
        "measured_sha256": official_files["chat_template.jinja"]["measured_sha256"],
        "matches_pin": official_files["chat_template.jinja"]["matches_pin"],
    }
    check(
        record,
        "official_template_bound_to_its_own_pin",
        official_files["chat_template.jinja"]["pin"] == "chat_template_sha256"
        and official_files["chat_template.jinja"]["matches_pin"] is True,
    )

    modified_root = os.path.join(ctx["work_dir"], "modified_root")
    info = build_modified_root(ctx, modified_root)
    record["modified_root"] = info
    check(
        record,
        "modified_template_differs_from_official",
        info["differs_from_official"],
        info["note"],
    )
    config_sha = ib.sha256_file(os.path.join(modified_root, "tokenizer_config.json"))
    record["modified_root_tokenizer_config_sha256"] = config_sha
    record["official_tokenizer_config_sha256"] = official_files["tokenizer_config.json"][
        "measured_sha256"
    ]
    check(
        record,
        "tokenizer_config_byte_identical_in_modified_root",
        config_sha == official_files["tokenizer_config.json"]["measured_sha256"],
        "tokenizer_config 与官方逐字节相同 → 只有 template pin 能拦住这次替换",
    )
    return ib.bind_model_inputs(
        root=modified_root, interface_path=ctx["interface_path"], layout=FLAT_LAYOUT
    )


def _case_reject_explicit_template_override(ctx, renderer, record):
    override = "{% for message in messages %}{{ raise_exception('nope') }}{% endfor %}"
    unauthorized = None
    try:
        ib.assert_no_template_override(override_template=override, binding=ctx["binding"])
    except PolicyViolation as exc:
        unauthorized = exc
    record["unauthorized_code"] = getattr(unauthorized, "code", None)
    check(
        record,
        "unauthorized_override_rejected",
        unauthorized is not None and unauthorized.code == "template_override_not_authorized",
    )
    record["no_override_record"] = ib.assert_no_template_override()

    authorized = ib.assert_no_template_override(
        override_template=override,
        override_authorized=True,
        override_reason="本地渲染回归需要一份合成模板做取证对照（不用于生产加载）",
        binding=ctx["binding"],
    )
    record["authorized_record"] = authorized
    check(
        record,
        "authorized_record_has_override_sha256",
        len(str(authorized.get("override_sha256") or "")) == 64,
    )
    check(
        record,
        "authorized_record_names_the_official_template",
        authorized.get("official_template_sha256") == ctx["template_sha256"],
    )
    dumped = json.dumps(authorized, ensure_ascii=False).lower()
    note = str(authorized.get("note") or "")
    check(
        record,
        "authorized_record_does_not_claim_semantic_equivalence",
        "equivalent" not in dumped and "语义等价" in note and "不" in note,
        "授权记录必须显式**否认**语义等价（实际 note=%r）" % note,
    )
    check(
        record,
        "override_sha256_documents_the_inline_template",
        authorized.get("override_sha256") == sha256_text(override),
    )
    if unauthorized is None:
        raise AssertionError("未授权的 template 覆盖没有被拒绝")
    raise unauthorized


class Case:
    """一个回归用例：`expect="render"` 或 `expect="reject"`。

    `requires_renderer=False` 的用例（normalize、pin 拒绝、覆盖拒绝）**不需要**渲染器，
    所以本地没有渲染器时它们照样真跑 —— 报告整体仍是 `blocked`，但不假装通过。
    """

    __slots__ = (
        "case_id",
        "expect",
        "run",
        "description",
        "expected_code",
        "expected_message_contains",
        "requires_renderer",
    )

    def __init__(
        self,
        case_id: str,
        expect: str,
        run,
        description: str,
        *,
        expected_code: str | None = None,
        expected_message_contains: str | None = None,
        requires_renderer: bool = True,
    ) -> None:
        self.case_id = case_id
        self.expect = expect
        self.run = run
        self.description = description
        self.expected_code = expected_code
        self.expected_message_contains = expected_message_contains
        self.requires_renderer = requires_renderer

    def to_dict(self) -> dict:
        return {
            "case": self.case_id,
            "expect": self.expect,
            "description": self.description,
            "expected_code": self.expected_code,
            "expected_message_contains": self.expected_message_contains,
            "requires_renderer": self.requires_renderer,
        }


CASES = (
    Case(
        "mapping_arguments_positive",
        "render",
        _case_mapping_arguments_positive,
        "tool_calls[].function.arguments 是 mapping → 必须渲染出 call:<name>{k:v}",
    ),
    Case(
        "string_arguments_rejected",
        "reject",
        _case_string_arguments_rejected,
        "同一调用但 arguments 是 JSON 字符串 → 官方模板必须经 raise_exception 拒绝",
        expected_message_contains="must be a JSON object (mapping), not a string",
    ),
    Case(
        "bad_json_rejected",
        "reject",
        _case_bad_json_rejected,
        "调用边界收到畸形 JSON 字符串 → normalize_tool_arguments 必须拒绝",
        expected_code="tool_arguments_invalid_json",
        requires_renderer=False,
    ),
    Case(
        "json_string_normalized",
        "render",
        _case_json_string_normalized,
        "合法 JSON 字符串 → 先归一化成 mapping，再由**归一化后的 mapping** 渲染",
    ),
    Case(
        "multi_turn_tool_continuation",
        "render",
        _case_multi_turn_tool_continuation,
        "user → assistant tool_call → tool 响应 → user：回合必须正确收口",
    ),
    Case(
        "turn_closure_forward_scan",
        "render",
        _case_turn_closure_forward_scan,
        "前向扫描：连续 assistant 属同一回合（continued），遇到 user 才收口（ended）",
    ),
    Case(
        "thinking_enabled",
        "render",
        _case_thinking_enabled,
        "enable_thinking=True → 必须出现 <|think|> 思考开关标记",
    ),
    Case(
        "thinking_disabled",
        "render",
        _case_thinking_disabled,
        "enable_thinking=False → 不得出现 <|think|>，思考通道不得带内容",
    ),
    Case(
        "reject_official_verified_but_modified_loaded",
        "reject",
        _case_reject_official_verified_but_modified_loaded,
        "官方 pin 核过之后，再绑一个被改过的 template 根 → 必须 input_pin_mismatch",
        expected_code="input_pin_mismatch",
        requires_renderer=False,
    ),
    Case(
        "reject_explicit_template_override",
        "reject",
        _case_reject_explicit_template_override,
        "显式 template 覆盖未授权 → template_override_not_authorized（授权则留审计记录）",
        expected_code="template_override_not_authorized",
        requires_renderer=False,
    ),
)

CASE_INDEX = {case.case_id: case for case in CASES}


# --------------------------------------------------------------------- 用例执行


def _compare_with_stand_in(ctx: dict, renderer, call: dict, official_output: str) -> dict:
    """把同一份输入喂给**合成改写版**模板，记录两者能否分辨。

    拿不到现场 224 行工作副本时这是替代对照：改写版只改了字符串 arguments 一处，
    所以除该分支外的用例两者应当逐字节一致 —— "一致"本身就是"本用例不能分辨两份模板"
    的诚实结论，报告里如实写出来。
    """
    stand_in_text = synthesise_modified_template(ctx["template_text"])
    info = {
        "template_sha256": sha256_text(stand_in_text),
        "official_template_sha256": ctx["template_sha256"],
        "note": "合成替身（只复刻字符串 arguments 一处分叉），不是现场工作副本",
    }
    try:
        alt = renderer.render(stand_in_text, call["messages"], **call["kwargs"])
    except Exception as exc:
        info.update(
            {
                "outcome": "rejected",
                "exception_type": type(exc).__name__,
                "same_as_official": False,
            }
        )
        return info
    info.update(
        {
            "outcome": "rendered",
            "same_as_official": alt == official_output,
            "rendered_sha256": sha256_text(alt),
        }
    )
    return info


def run_case(case: Case, ctx: dict, renderer) -> dict:
    """跑一个用例，返回可 JSON 序列化的记录（含实际输出/异常与判定）。"""
    record = case.to_dict()
    record["checks"] = []
    record["check_failures"] = []
    record["outcome"] = None
    record["exception"] = None
    record["mismatch_reasons"] = []

    if renderer is None and case.requires_renderer:
        record["outcome"] = "blocked"
        record["mismatch_reasons"].append("本地没有 transformers / jinja2，无法渲染")
        record["matched"] = None
        return record

    exception = None
    outcome = "rendered"
    try:
        rendered = case.run(ctx, renderer, record)
        record["rendered"] = rendered
        record["rendered_bytes"] = len(rendered.encode("utf-8"))
        record["rendered_sha256"] = sha256_text(rendered)
    except FailClosed as exc:
        outcome = "rejected"
        exception = {
            "type": type(exc).__name__,
            "code": exc.code,
            "message": exc.message,
            "text": str(exc),
            "context": exc.context,
        }
    except Exception as exc:  # jinja2 的 TemplateError 等非 fail-closed 异常
        outcome = "rejected"
        exception = {
            "type": type(exc).__name__,
            "code": None,
            "message": str(exc),
            "text": traceback.format_exc(limit=8),
        }
    record["outcome"] = outcome
    record["exception"] = exception

    call = record.pop("_stand_in_call", None)
    if outcome == "rendered" and call is not None:
        record["stand_in"] = _compare_with_stand_in(
            ctx, renderer, call, record.get("rendered", "")
        )

    if case.expect == "render":
        matched = outcome == "rendered"
    else:
        matched = outcome == "rejected"
    if matched and case.expected_code:
        matched = bool(exception) and exception.get("code") == case.expected_code
    if matched and case.expected_message_contains:
        matched = bool(exception) and case.expected_message_contains in (
            exception.get("message") or ""
        )
    if not matched:
        record["mismatch_reasons"].append(
            "期望 %s，实际 %s%s"
            % (
                case.expect,
                outcome,
                ""
                if not exception
                else "（%s: %s）"
                % (
                    exception.get("type"),
                    (exception.get("code") or exception.get("message") or "")[:160],
                ),
            )
        )
    if record["check_failures"]:
        matched = False
        record["mismatch_reasons"].append(
            "结构断言未通过：%s" % [item["check"] for item in record["check_failures"]]
        )
    record["matched"] = matched
    return record


def run_cases(ctx: dict, renderer, only=None) -> list:
    """按需跑全部（或指定的）用例，返回记录列表。"""
    wanted = set(only) if only else None
    records = []
    for case in CASES:
        if wanted is not None and case.case_id not in wanted:
            continue
        records.append(run_case(case, ctx, renderer))
    return records


# --------------------------------------------------------------------- 报告


def build_report(ctx: dict, renderer, notes, case_records, *, vendor_used) -> dict:
    template_files = {item["filename"]: item for item in ctx["binding"]["files"]}
    template_entry = template_files["chat_template.jinja"]
    global_checks: list = []

    def global_check(name, condition, detail=""):
        entry = {"check": name, "passed": bool(condition)}
        if detail and not condition:
            entry["detail"] = detail
        global_checks.append(entry)
        return bool(condition)

    global_check(
        "template_sha256_equals_official_pin",
        template_entry["measured_sha256"] == OFFICIAL_TEMPLATE_SHA256,
        template_entry["measured_sha256"],
    )
    global_check(
        "template_pin_is_chat_template_sha256_not_tokenizer_config",
        template_entry["pin"] == "chat_template_sha256"
        and template_entry["pin"] != template_files["tokenizer_config.json"]["pin"],
    )
    global_check(
        "tokenizer_config_carries_no_inline_chat_template",
        "chat_template" not in ctx["tokenizer_config"],
        "tokenizer_config.json 里有 chat_template 键的话，模板 pin 就可能被 config pin 顶替",
    )
    global_check(
        "load_view_copies_match_binding",
        all(item["copy_matches_source"] for item in ctx["view"]["files"]),
    )

    blocked = renderer is None
    failed = [item["case"] for item in case_records if item.get("matched") is False]
    case_matches = [item.get("matched") for item in case_records]
    if blocked:
        status = "blocked"
    elif failed or any(value is None for value in case_matches):
        status = "failed"
    elif all(item["passed"] for item in global_checks):
        status = "ok"
    else:
        status = "failed"

    renderer_info = (
        renderer.describe()
        if renderer is not None
        else {
            "kind": RENDERER_NONE,
            "local_stand_in": None,
            "caveat": "没有可用的渲染器：本回归不假装通过，状态 blocked。",
        }
    )
    renderer_info["detected_kind"] = renderer_kind()
    renderer_info["notes"] = list(notes)
    renderer_info["import_probe"] = _import_probe()

    report = {
        "kind": "chat-template-render-regression/1",
        "status": status,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "local_interpreter": {
            "executable": sys.executable,
            "version": sys.version.split()[0],
            "platform": sys.platform,
        },
        "target_interpreter": {
            "executable": TARGET_INTERPRETER,
            "command": TARGET_COMMAND,
            "required": True,
            "reason": (
                "shim 只是本地替身：不证明生产加载路径，也不覆盖 apply_chat_template 的全部语义；"
                "本报告的渲染结论必须在目标解释器的锁定栈上复跑。"
            ),
        },
        "vendor_probe": {
            "used": vendor_used,
            "candidate_dirs": list(VENDOR_DIR_CANDIDATES),
            "default_dir": DEFAULT_VENDOR_DIR,
            "in_repo": bool(
                vendor_used
                and os.path.abspath(vendor_used).startswith(REPO_ROOT + os.sep)
            ),
            "note": (
                "jinja2 的目录级安装位于**包根之外**（不会进交付 ZIP）；"
                "没有 PYTHONPATH 时本文件会主动探测候选目录。"
            ),
        },
        "inputs": {
            "root": ctx["inputs_root"],
            "interface_path": ctx["interface_path"],
            "layout": dict(FLAT_LAYOUT),
            "binding_sha256": ctx["binding"]["binding_sha256"],
            "repo_id": ctx["binding"]["repo_id"],
            "revision": ctx["binding"]["revision"],
            "files": [
                {
                    "filename": item["filename"],
                    "pin": item["pin"],
                    "declared_sha256": item["declared_sha256"],
                    "measured_sha256": item["measured_sha256"],
                    "bytes": item["bytes"],
                    "source_path": item["source_path"],
                    "matches_pin": item["matches_pin"],
                }
                for item in ctx["binding"]["files"]
            ],
            "consumed_inputs": ctx["binding"]["consumed_inputs"],
            "weights_sha256": ctx["binding"]["weights_sha256"],
        },
        "load_view": {
            "view_dir": ctx["view"]["view_dir"],
            "manifest_sha256": ctx["view"]["manifest_sha256"],
            "files": ctx["view"]["files"],
        },
        "template": {
            "path": ctx["view"]["chat_template_path"],
            "sha256": ctx["template_sha256"],
            "bytes": ctx["template_bytes"],
            "pin": template_entry["pin"],
            "pin_value": template_entry["declared_sha256"],
            "matches_pin": template_entry["matches_pin"],
            "official_sha256_reference": OFFICIAL_TEMPLATE_SHA256,
            "tokenizer_config_pin": template_files["tokenizer_config.json"]["pin"],
            "template_pin_is_separate_from_tokenizer_config_pin": (
                template_entry["pin"] != template_files["tokenizer_config.json"]["pin"]
            ),
            "tokenizer_config_has_inline_chat_template": "chat_template"
            in ctx["tokenizer_config"],
            "raised_globals_in_template": ["raise_exception"],
        },
        "renderer": renderer_info,
        "global_checks": global_checks,
        "cases": case_records,
        "summary": {
            "cases_run": len(case_records),
            "cases_matched": sum(1 for item in case_records if item.get("matched") is True),
            "cases_mismatched": len(failed),
            "cases_blocked": sum(
                1 for item in case_records if item.get("outcome") == "blocked"
            ),
            "failed_cases": failed,
            "renderer_kind": renderer_info["kind"],
        },
        "not_verified_locally": [
            "本机**没有** transformers / peft：没有在 PreTrainedTokenizerBase.apply_chat_template "
            "（生产入口）上跑过；本报告的渲染结论来自本地 shim。"
            if renderer is not None and renderer.kind == RENDERER_SHIM
            else "本机渲染器=%s：仍未在目标解释器上复跑确认。"
            % (renderer_info["kind"],),
            "仓库内**没有**现场那份 224 行工作副本；本回归用的是**合成替身**，"
            "只复刻「字符串 arguments 由拒绝改成静默打印」这一处分叉，不代表改写版其余差异。",
            "本回归证明「渲染器读到的字节 = 被 pin 核过的官方模板字节」，"
            "不证明生产 `TorchPeftBackend.prepare()` 的 `from_pretrained` 真的指向该加载视图（那属于 runner 接线，另行评审）。",
            "权重 `model.safetensors` 仍是 hash_pending，不属于本回归范围。",
            "未在 GPU / 目标解释器上执行；目标解释器命令见 target_interpreter.command。",
        ],
        "note": (
            "绑定只证明本轮读到的字节等于锁里 pin 的字节；渲染结论只证明"
            "「该模板文本在这些输入下的行为」。两者都不翻 verified，也不证明模型可训练。"
        ),
    }
    if blocked:
        missing = []
        for name in ("transformers", "jinja2"):
            if not _module_available(name):
                missing.append(name)
        report["blocked"] = {
            "status": "blocked",
            "missing_modules": missing,
            "import_probe": _import_probe(),
            "cases_that_ran": [
                item["case"] for item in case_records if item.get("matched") is not None
            ],
            "target_interpreter": TARGET_INTERPRETER,
            "target_command": TARGET_COMMAND,
            "hint": (
                "本地可用目录级安装取得 jinja2："
                'python -m pip install --target "%s" jinja2 然后设置 PYTHONPATH；'
                "候选目录=%s。"
                % (DEFAULT_VENDOR_DIR, list(VENDOR_DIR_CANDIDATES))
            ),
            "note": (
                "本回归不假装通过：渲染类用例一律计 blocked，整体状态 blocked，退出码 8。"
                "不依赖渲染器的用例（normalize / pin 拒绝 / 覆盖拒绝）照样真跑并计入 "
                "`cases_that_ran`。"
            ),
        }
    return report


# --------------------------------------------------------------------- CLI


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="官方 Gemma 4 chat template 的真实 tokenizer 渲染回归"
    )
    parser.add_argument("--inputs-root", default=None, help="官方原件目录（默认证据目录）")
    parser.add_argument("--interface", default=None, help="official-interface.json 路径")
    parser.add_argument("--view-dir", default=None, help="受控加载视图目录（放在包外）")
    parser.add_argument("--work-dir", default=None, help="用例临时目录（放在包外）")
    parser.add_argument("--vendor-dir", default=None, help="jinja2 目录级安装位置")
    parser.add_argument("--no-vendor-probe", action="store_true", help="不探测 _vendor 目录")
    parser.add_argument("--only", default=None, help="只跑指定用例（逗号分隔的 case id）")
    parser.add_argument("--out", default=None, help="JSON 报告输出路径")
    args = parser.parse_args(argv)

    vendor_used = None if args.no_vendor_probe else probe_vendor_dir(args.vendor_dir)

    try:
        ctx = build_context(
            inputs_root=args.inputs_root,
            interface_path=args.interface,
            view_dir=args.view_dir,
            work_dir=args.work_dir,
        )
    except FailClosed as exc:
        report = {
            "kind": "chat-template-render-regression/1",
            "status": "failed",
            "stage": "input_binding",
            "blocking_error": exc.to_dict(),
            "vendor_probe": {"used": vendor_used},
        }
        _emit(report, args.out)
        return 1

    renderer, notes = build_renderer(ctx)
    only = [item.strip() for item in args.only.split(",")] if args.only else None
    records = run_cases(ctx, renderer, only=only)
    report = build_report(ctx, renderer, notes, records, vendor_used=vendor_used)
    _emit(report, args.out)

    if report["status"] == "blocked":
        return Blocked("chat_template_renderer_unavailable", "缺渲染器").exit_code
    return 0 if report["status"] == "ok" else 1


def _emit(report: dict, out: str | None) -> None:
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if out:
        directory = os.path.dirname(os.path.abspath(out))
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
