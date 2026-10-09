"""KAGGLE-38 r3 · C：生产加载器**实际消费**的配置 / tokenizer / template 绑定。

## 缺陷出处

Mika 2026-10-08 裁定（KAGGLE-28 评论 01a11984-419b-783f-b142-b73c1cf629a4 与任务顶部）：

    V3训练及首次工程检查以模型 revision 52f3f65bc7a02d555763bc923bd1d9094898219d
    的官方原件作为配置、tokenizer与template输入。不得一边核验 hf-revision-originals、
    一边让生产加载器隐式读 prep/model 里的修改版。
    官方 hash 匹配但加载修改版 / 显式 template 覆盖 的拒绝反例必须补上；
    输出日志须能确定消费者实际用了哪份 tokenizer 和 template。

现场实测的坑：`~/gemma4/prep/model/` 里躺着一份**被改过**的工作副本 ——
`tokenizer_config.json` 少了 `response_template` 键、`chat_template.jinja` 是 224 行重写版，
而官方原件另存在 `hf-revision-originals/`。旧实现 `runner.TorchPeftBackend.prepare()`
直接 `from_pretrained(model_id, revision=...)`，**从不核对**它实际读到的字节 ——
于是"锁里核的是官方 SHA、生产加载的是改过的文件"两件事可以各说各话。

## 本模块把这件事收成一个可判定的对象

1. `load_interface_pins()`：从 `official-interface.json` 读出每个输入文件的 pin；
2. `bind_model_inputs()`：在给定 root 下按**显式布局**解析实际文件路径
   （拒绝 symlink / 目录逃逸），逐字节重算 SHA-256 并与 pin 比对；
3. `assemble_load_view()`：把被绑定的文件**复制**进一个受控目录（加载视图），
   再对副本重算 SHA-256 —— 证明"消费者读的就是被核过的那批字节"，而不是软链接文字；
4. `assert_no_template_override()`：默认**拒绝**任何显式 template 覆盖；
   确需覆盖时必须显式授权，且授权事实写进证据，不允许静默改模板；
5. 绑定结果里 `consumed_inputs` 逐文件给出 `source_path` / `view_path` / `revision` /
   `bytes` / `sha256` / `pin` —— 日志据此可确定消费者实际用了哪一份。

## 诚实边界

- 本模块**不下载**任何模型权重。权重（`model.safetensors`）的 SHA 本轮明确
  `hash_pending`，它不属于本绑定范围；
- `verified` 一律**不翻**：绑定只证明"这一次读的字节等于锁里 pin 的字节"，
  不证明模型可训练、不证明 serving 兼容；
- 纯标准库；不 import torch / transformers / peft。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from typing import Mapping

from ..common.canonical import sha256_json
from ..common.errors import MissingInput, PolicyViolation, UnverifiedLock

#: 输入文件 → official-interface 锁里对应的 pin 名。
INPUT_PINS: dict[str, str] = {
    "tokenizer_config.json": "tokenizer_config_sha256",
    "tokenizer.json": "tokenizer_vocab_sha256",
    "chat_template.jinja": "chat_template_sha256",
    # r4 · 缺陷 3：`config.json` 原来是"不参与 pin 核对的伴随文件"。独立审查实测：
    # 把它改成 `{"model_type":"changed"}` 仍会被复制进加载视图——即"核验了官方
    # tokenizer/template，却让模型按一份没核过的配置加载"。现在它同样按 pin 核对。
    "config.json": "config_sha256",
}

#: 装配加载视图时**额外**一起放进去的伴随文件。`config.json` 已升为受核对输入，
#: 这里的其余文件（生成/处理器配置）仍属伴随，但会被记进视图清单以供复核。
COMPANION_FILES = ("generation_config.json", "processor_config.json")

#: 现场快照的默认布局：`official/` 与 `shared/` 是**证据分组**，不是可直接
#: `from_pretrained` 的完整根；加载视图由本模块受控装配。
DEFAULT_SNAPSHOT_LAYOUT: dict[str, str] = {
    "tokenizer_config.json": "official/tokenizer_config.json",
    "chat_template.jinja": "official/chat_template.jinja",
    "tokenizer.json": "shared/tokenizer.json",
    "config.json": "shared/config.json",
}

#: 本模块要求的输入 pin 文件名（模板 pin 必须**单列**，不得用 config 顶替）。
REQUIRED_INPUT_FILES = (
    "tokenizer_config.json",
    "tokenizer.json",
    "chat_template.jinja",
    "config.json",
)


@dataclass(frozen=True)
class InputPin:
    filename: str
    pin: str
    declared: str | None

    @property
    def locked(self) -> bool:
        return bool(self.declared)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_interface_pins(interface_path: str) -> dict:
    """读出 official-interface 锁（只读；本模块永不写锁、永不翻 verified）。"""
    if not os.path.exists(interface_path):
        raise MissingInput("official_interface_missing", "接口锁不存在：%s" % interface_path)
    with open(interface_path, "r", encoding="utf-8-sig") as handle:
        payload = json.load(handle)
    pins = payload.get("pins")
    if not isinstance(pins, Mapping):
        raise MissingInput("official_interface_invalid", "接口锁没有 pins 段：%s" % interface_path)
    return dict(payload)


def declared_value(lock: Mapping, pin_name: str) -> str | None:
    entry = (lock.get("pins") or {}).get(pin_name) or {}
    if not isinstance(entry, Mapping):
        raise MissingInput("official_interface_invalid", "pin %s 不是对象" % pin_name)
    value = entry.get("value")
    return None if value in (None, "") else str(value)


def resolve_layout(layout: Mapping | None) -> dict:
    """归一化布局：文件名 → root 下的相对路径。默认用现场快照布局。"""
    resolved = dict(DEFAULT_SNAPSHOT_LAYOUT)
    for filename, relative in (layout or {}).items():
        resolved[str(filename)] = str(relative)
    return resolved


def resolve_input_file(
    root: str,
    filename: str,
    layout: Mapping | None = None,
    *,
    allowed_roots=None,
) -> str:
    """在 root 下解析一个输入文件的**真实路径**，拒绝目录逃逸与 symlink 逃逸。"""
    layout = resolve_layout(layout)
    relative = str(layout.get(filename, filename))
    candidate = os.path.join(root, *relative.split("/"))
    if not os.path.exists(candidate):
        raise MissingInput(
            "input_file_missing",
            "绑定的输入文件不存在：%s（root=%s，layout=%r）" % (filename, root, relative),
            filename=filename,
            root=os.path.abspath(root),
            relative=relative,
        )
    if not os.path.isfile(candidate):
        raise PolicyViolation(
            "input_path_not_a_file", "绑定的输入路径不是普通文件：%s" % candidate
        )
    real = os.path.realpath(candidate)
    allowed = [os.path.realpath(root)] + [os.path.realpath(item) for item in (allowed_roots or ())]
    if not any(real == base or real.startswith(base + os.sep) for base in allowed):
        raise PolicyViolation(
            "input_path_escapes_root",
            "输入文件解析后逃出了允许的根（symlink 或 ../ 逃逸）：%s → %s" % (candidate, real),
            candidate=candidate,
            realpath=real,
            allowed_roots=allowed,
        )
    return real


def bind_model_inputs(
    *,
    root: str,
    interface_path: str,
    layout: Mapping | None = None,
    allowed_roots=None,
    decoy_roots=None,
    require_locked: bool = True,
) -> dict:
    """把 root 下的实际输入文件绑定到锁里的 pin，逐字节核对。

    fail-closed：

    - `input_pin_not_locked`：锁里该 pin 仍是 `null`（未锁定）——**不得**用未锁定值放行；
    - `input_file_missing` / `input_path_not_a_file` / `input_path_escapes_root`；
    - `input_pin_mismatch`：实测 SHA-256 与 pin 不一致 —— 这正是"核官方、加载修改版"的
      拒绝点，也**不允许**降级为警告。

    `decoy_roots` 只做**取证**：如果别的目录（例如现场被改过的 `prep/model/`）里存在同名
    文件但哈希不同，如实记进 `shadowed_files`，让日志能说清"消费者没用那一份"。
    """
    lock = load_interface_pins(interface_path)
    revision = declared_value(lock, "model_revision")
    repo_id = declared_value(lock, "model_repo_id")
    layout = resolve_layout(layout)
    decoy_roots = [str(item) for item in (decoy_roots or ())]

    files: list[dict] = []
    mismatches: list[dict] = []
    shadowed: list[dict] = []
    for filename in REQUIRED_INPUT_FILES:
        pin_name = INPUT_PINS[filename]
        declared = declared_value(lock, pin_name)
        if require_locked and not declared:
            raise UnverifiedLock(
                "input_pin_not_locked",
                "输入 pin %s 仍是 null：未锁定的值不得用作加载绑定" % pin_name,
                filename=filename,
                pin=pin_name,
            )
        path = resolve_input_file(root, filename, layout, allowed_roots=allowed_roots)
        measured = sha256_file(path)
        entry = {
            "filename": filename,
            "pin": pin_name,
            "declared_sha256": declared,
            "measured_sha256": measured,
            "bytes": os.path.getsize(path),
            "source_path": path,
            "layout_relative_path": layout.get(filename, filename),
            "matches_pin": bool(declared) and measured == declared,
        }
        files.append(entry)
        if declared and measured != declared:
            mismatches.append(entry)
        for decoy_root in decoy_roots:
            decoy = os.path.join(decoy_root, *layout.get(filename, filename).split("/"))
            if not os.path.exists(decoy) or not os.path.isfile(decoy):
                continue
            decoy_sha = sha256_file(decoy)
            shadowed.append(
                {
                    "filename": filename,
                    "decoy_root": os.path.abspath(decoy_root),
                    "decoy_path": os.path.abspath(decoy),
                    "decoy_sha256": decoy_sha,
                    "agrees_with_pin": bool(declared) and decoy_sha == declared,
                    "used": False,
                }
            )

    if mismatches:
        first = mismatches[0]
        raise PolicyViolation(
            "input_pin_mismatch",
            "绑定的输入文件与被核验的官方 pin 不一致（官方 hash 匹配但加载了修改版）："
            "%s 实测 %s，锁里 %s"
            % (
                first["filename"],
                first["measured_sha256"][:16],
                str(first["declared_sha256"])[:16],
            ),
            mismatches=[
                {
                    "filename": item["filename"],
                    "pin": item["pin"],
                    "declared": item["declared_sha256"],
                    "measured": item["measured_sha256"],
                }
                for item in mismatches
            ],
            source_root=os.path.abspath(root),
        )

    binding = {
        "kind": "bound-model-inputs",
        "root": os.path.abspath(root),
        "repo_id": repo_id,
        "revision": revision,
        "layout": layout,
        "files": files,
        "consumed_inputs": {
            item["filename"]: {
                "path": item["source_path"],
                "sha256": item["measured_sha256"],
                "bytes": item["bytes"],
                "pin": item["pin"],
            }
            for item in files
        },
        "shadowed_files": shadowed,
        "weights_sha256": "hash_pending",
        "verified_flipped": False,
        "lock_modified": False,
        "note": (
            "绑定只证明本轮读到的字节等于锁里 pin 的字节；不证明模型可训练，"
            "也不证明 serving 兼容。权重 SHA 明确 hash_pending。"
        ),
    }
    binding["binding_sha256"] = sha256_json(
        {key: value for key, value in binding.items() if key != "binding_sha256"}
    )
    return binding


def assemble_load_view(
    binding: Mapping,
    view_dir: str,
    *,
    companions: bool = True,
    decoy_roots=None,
) -> dict:
    """把被绑定的文件**复制**进受控目录，形成可交给 `from_pretrained` 的加载视图。

    复制后对副本**重算** SHA-256 并与绑定值比对：这样"消费者实际读的字节"
    是可证明的，而不是一句"它应该会读那个路径"。
    只有被绑定 + 明确列入伴随清单的文件会进视图；权重不在此范围。
    """
    if not isinstance(binding, Mapping) or not binding.get("files"):
        raise MissingInput("input_binding_missing", "装配加载视图需要先完成 bind_model_inputs()")
    os.makedirs(view_dir, exist_ok=True)
    copied: list[dict] = []
    problems: list[str] = []
    entries = list(binding["files"])
    if companions:
        root = str(binding.get("root"))
        layout = binding.get("layout") or {}
        for name in COMPANION_FILES:
            relative = layout.get(name, name)
            candidate = os.path.join(root, *str(relative).split("/"))
            if os.path.isfile(candidate):
                entries.append(
                    {
                        "filename": name,
                        "pin": None,
                        "declared_sha256": None,
                        "measured_sha256": sha256_file(candidate),
                        "bytes": os.path.getsize(candidate),
                        "source_path": candidate,
                        "layout_relative_path": relative,
                        "matches_pin": None,
                        "companion": True,
                    }
                )
    for item in entries:
        target = os.path.join(view_dir, str(item["filename"]))
        shutil.copyfile(str(item["source_path"]), target)
        view_sha = sha256_file(target)
        agrees = view_sha == item["measured_sha256"]
        if not agrees:
            problems.append(str(item["filename"]))
        copied.append(
            {
                "filename": item["filename"],
                "path": target,
                "bytes": os.path.getsize(target),
                "sha256": view_sha,
                "source_path": item["source_path"],
                "copy_matches_source": agrees,
                "pin": item.get("pin"),
                "companion": bool(item.get("companion")),
            }
        )
    if problems:
        raise PolicyViolation(
            "load_view_byte_mismatch",
            "加载视图里的副本与绑定字节不一致：%s" % problems,
            files=problems,
        )
    manifest = {
        "kind": "v3-load-view/1",
        "view_dir": os.path.abspath(view_dir),
        "repo_id": binding.get("repo_id"),
        "revision": binding.get("revision"),
        "binding_sha256": binding.get("binding_sha256"),
        "files": copied,
        "tokenizer_config_path": os.path.join(view_dir, "tokenizer_config.json"),
        "tokenizer_data_path": os.path.join(view_dir, "tokenizer.json"),
        "chat_template_path": os.path.join(view_dir, "chat_template.jinja"),
        "weights_in_view": False,
        "weights_sha256": "hash_pending",
        "note": (
            "视图由受控装配产生：只含被绑定的输入文件与伴随配置。"
            "`from_pretrained` 指向它时，消费者读到的字节就是上面逐个核对过的那批。"
        ),
    }
    manifest["manifest_sha256"] = sha256_json(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    return manifest


def read_chat_template(path: str) -> str:
    """读回被绑定的 chat template 文本（供渲染回归使用）。"""
    if not os.path.isfile(path):
        raise MissingInput("chat_template_missing", "chat template 不存在：%s" % path)
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def normalize_tool_arguments(value):
    """调用边界收到 JSON 字符串时，**先显式解析并验证成 mapping** 再交给模板渲染。

    Mika 的原话：「若调用边界接收JSON字符串，应显式解析验证为mapping再渲染，
    不能绕过官方模板拒绝或静默改模板。」

    三种结果，第三种必须拒绝：

    - 已经是 mapping → 原样返回（`source="mapping"`）；
    - JSON 字符串且解析出来是 mapping → 返回解析结果（`source="json-string"`）；
    - 其它（JSON 解析失败、解析出来不是 mapping、类型不认识）→ `PolicyViolation`。
    """
    if isinstance(value, Mapping):
        return {"value": dict(value), "source": "mapping"}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError) as exc:
            raise PolicyViolation(
                "tool_arguments_invalid_json",
                "tool_call.arguments 是字符串但不是合法 JSON：%s" % exc,
                arguments=value[:200],
            ) from exc
        if not isinstance(parsed, Mapping):
            raise PolicyViolation(
                "tool_arguments_not_a_mapping",
                "tool_call.arguments 解析后不是 mapping（得到 %s）：不得直接交给模板"
                % type(parsed).__name__,
                arguments=value[:200],
            )
        return {"value": dict(parsed), "source": "json-string"}
    raise PolicyViolation(
        "tool_arguments_unsupported_type",
        "tool_call.arguments 类型不支持：%s" % type(value).__name__,
    )


def assert_no_template_override(
    *,
    override_template=None,
    override_authorized: bool = False,
    override_reason: str | None = None,
    binding: Mapping | None = None,
) -> dict:
    """默认**拒绝**显式 template 覆盖；授权覆盖也必须留下可审计记录。

    Mika 的原话：不能绕过官方模板拒绝或静默改模板。
    """
    if override_template in (None, ""):
        return {"override_used": False, "authorized": False}
    if not override_authorized:
        raise PolicyViolation(
            "template_override_not_authorized",
            "提供了显式 chat template 覆盖但未授权：不得绕过官方模板（fail-closed）",
            override=(
                str(override_template)
                if isinstance(override_template, str) and len(str(override_template)) < 200
                else "<inline-template>"
            ),
            binding_revision=(binding or {}).get("revision"),
        )
    if not str(override_reason or "").strip():
        raise PolicyViolation(
            "template_override_reason_missing",
            "授权覆盖 chat template 必须写明理由（用于事后审计）",
        )
    inline = str(override_template)
    digest = hashlib.sha256(inline.encode("utf-8")).hexdigest()
    return {
        "override_used": True,
        "authorized": True,
        "reason": str(override_reason),
        "override_sha256": digest,
        "override_bytes": len(inline.encode("utf-8")),
        "official_template_sha256": next(
            (
                item["measured_sha256"]
                for item in ((binding or {}).get("files") or [])
                if item.get("filename") == "chat_template.jinja"
            ),
            None,
        ),
        "note": (
            "覆盖已授权并记录；本记录**不**表示覆盖后的模板与官方语义等价，"
            "需另行行为回归与独审。"
        ),
    }
