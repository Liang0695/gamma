"""模板锁定与渲染器抽象（KAGGLE-20 §4，KAGGLE-19 §3）。

事实边界（必须照实标注）：
- canonical Gemma4 template SHA 与 CT tokenizer_config SHA 来自 KAGGLE-20 §4；
- **完整 tokenizer 词表本轮未下载**，`tokenizer_vocab_sha256` 仍为未知；
- 因此本仓库提供的 `ShimRenderer` 只是**离线测试用的确定性占位**，
  它的 `official_template_verified` 永远为 False；
- 任何"渲染与 serving 一致"的结论都必须由 `OfficialTemplateRenderer` 在真实
  HF tokenizer + canonical chat_template 上跑出逐字节相等，才允许写下。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from ..common.errors import Blocked, IntegrityError, UnverifiedLock

#: KAGGLE-20 §4 给出的模板 SHA（原 IT 与 CT 模板相同）。
CANONICAL_TEMPLATE_SHA256 = "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
#: CT tokenizer_config SHA；词表 SHA 未取得。
CT_TOKENIZER_CONFIG_SHA256 = "b8045a4576903e86903291d5cbdd4adfc8859e9ce3c98621bdbd957f73ed394b"

PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 2
EOS_TOKEN_IDS = (1, 106)

#: 训练 format 采用 Gemma4 原生 turn 标记，不新增 token、不改 embedding。
TURN_MARKERS = {
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "tool": "<|tool|>",
    "system": "<|system|>",
    "end": "<end_of_turn>",
}

#: shim 专用的 reasoning 通道标记。**只在 ShimRenderer 里出现**，不是官方 token；
#: 真机渲染必须由 official template 决定 reasoning 与 action 的分界，不得沿用此标记。
REASONING_MARKER = "<|reasoning|>"


@dataclass
class Span:
    """渲染后 token 区间（半开区间 [start, end)）。

    `channel` 区分同一条 assistant 消息内的两条通道（KAGGLE-26 / Q0 报告 🔴-1 的显式口径）：

    - ``"action"``：被监督的目标通道 —— 工具名、结构化参数、完整编辑片段、
      以及 thinking 关闭时的短计划/定位（`masks.py` 契约第 6-7 行的监督项清单）；
    - ``"reasoning"``：thinking 打开时的可见 reasoning，**只作 context，不施加 loss**；
    - ``"bos"`` / 其它 role 的通道：一律 context-only。

    `text` 是该 span 实际渲染进 input_ids 的可见正文（用于"正文是否真的进了
    input_ids"这类可断言的检查，而不是靠字符串搜索猜）。`arguments_text` 只在
    action 通道携带工具调用时给出，是 `_stable_args()` 产出的**逐字节原文**，
    供 `assert_roundtrip_arguments()` 做真正的渲染-还原比对。
    """

    role: str
    start: int
    end: int
    step: int | None = None
    supervised: bool = False
    phase: str | None = None
    reason: str = ""
    channel: str = "action"
    text: str = ""
    arguments_text: str | None = None

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "start": self.start,
            "end": self.end,
            "step": self.step,
            "supervised": self.supervised,
            "phase": self.phase,
            "reason": self.reason,
            "channel": self.channel,
            "text": self.text,
            "arguments_text": self.arguments_text,
        }


@dataclass
class RenderedSample:
    input_ids: list[int]
    spans: list[Span] = field(default_factory=list)
    bos_count: int = 0
    renderer: str = "shim"
    official_template_verified: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.input_ids)

    def spans_for(self, role: str) -> list[Span]:
        return [s for s in self.spans if s.role == role]


class Renderer:
    """渲染器接口。实现必须自己声明 `official_template_verified`。"""

    name = "base"
    official_template_verified = False

    def encode(self, text: str) -> list[int]:
        raise NotImplementedError

    def render(self, messages: Sequence[Mapping], thinking: bool = False) -> RenderedSample:
        raise NotImplementedError

    def verify_template_bytes(self, template_bytes: bytes) -> None:
        raise NotImplementedError


class ShimRenderer(Renderer):
    """离线确定性渲染器：**不是**官方模板，只用于 mask 工程回归。"""

    name = "shim-deterministic"
    official_template_verified = False

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        """词表无关的确定性编码：按空白切分 + 追加 EOS 之外的稳定 id。"""
        ids: list[int] = []
        for token in text.split(" "):
            if not token:
                continue
            if token in self._vocab:
                ids.append(self._vocab[token])
            else:
                # 稳定 id：offset 1000 起，按首次出现顺序分配。
                self._vocab[token] = 1000 + len(self._vocab)
                ids.append(self._vocab[token])
        return ids

    def render(self, messages: Sequence[Mapping], thinking: bool = False) -> RenderedSample:
        """确定性渲染。

        🔴-1（KAGGLE-26 / Q0 报告）修复后的**显式**口径：

        1. assistant 的 `content` **永不丢弃**，且必须能在 `input_ids` 里按 token
           子序列定位到；
        2. `thinking=True` 时，assistant 的 `content` 渲染成**独立 reasoning 通道**
           （`channel="reasoning"`、`supervised=False`），只作 context；
           `thinking=False` 时不产生 reasoning 通道，`content` 并入动作通道并随
           `loss_eligible` 参与监督；
        3. 结构化 `arguments` 走 action 通道，`content` 在前、`tool_name + 参数` 在后，
           两者都不再出现"带 `arguments` 就把 `content` 吃掉"的分支；
        4. reasoning 通道与 action 通道的 token 区间**严格不重叠**。

        这不是"两种解读都行"的含糊处理：监督目标清单见 `masks.py` 模块 docstring。
        """
        ids: list[int] = [BOS_TOKEN_ID]
        spans: list[Span] = [Span(role="bos", start=0, end=1, reason="bos", channel="bos")]
        for message in messages:
            role = message["role"]
            marker = TURN_MARKERS.get(role, "<|%s|>" % role)
            content = str(message.get("content", ""))
            arguments = message.get("arguments")
            tool_name = str(message.get("tool_name") or "")
            supervised = bool(message.get("loss_eligible")) and role == "assistant"

            turn_start = len(ids)
            ids.append(self.encode(marker)[0])

            action_start = turn_start
            if role == "assistant" and thinking and content:
                reasoning_start = len(ids)
                ids.append(self.encode(REASONING_MARKER)[0])
                ids.extend(self.encode(content))
                spans.append(
                    Span(
                        role="assistant",
                        start=reasoning_start,
                        end=len(ids),
                        step=message.get("step"),
                        supervised=False,
                        phase=message.get("phase"),
                        reason="reasoning_context_only",
                        channel="reasoning",
                        text=content,
                    )
                )
                action_start = len(ids)
                content_for_action = ""
            else:
                content_for_action = content

            parts: list[str] = []
            if content_for_action:
                parts.append(content_for_action)
            arguments_text: str | None = None
            if arguments is not None:
                arguments_text = _stable_args(arguments)
                if tool_name:
                    parts.append(tool_name)
                parts.append(arguments_text)
            body = " ".join(parts)
            if body:
                ids.extend(self.encode(body))
            ids.append(self.encode(TURN_MARKERS["end"])[0])
            spans.append(
                Span(
                    role=role,
                    start=action_start,
                    end=len(ids),
                    step=message.get("step"),
                    supervised=supervised,
                    phase=message.get("phase"),
                    reason=str(message.get("reason", "")),
                    channel="action",
                    text=body,
                    arguments_text=arguments_text,
                )
            )
        return RenderedSample(
            input_ids=ids,
            spans=spans,
            bos_count=1,
            renderer=self.name,
            official_template_verified=False,
            notes=["shim 渲染器：结果不能用于声称与 serving 渲染一致"],
        )

    def verify_template_bytes(self, template_bytes: bytes) -> None:
        raise UnverifiedLock(
            "shim_cannot_verify_template",
            "ShimRenderer 无法验证官方模板字节；需要真实 tokenizer 的 chat_template",
        )


def _stable_args(arguments: Mapping) -> str:
    import json

    return json.dumps(arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


class OfficialTemplateRenderer(Renderer):
    """真实 HF tokenizer + canonical chat_template 渲染器（离线不可用时 fail-closed）。

    依赖锁定版本（transformers 5.17.0 等）与真实 tokenizer 文件；缺一即阻断，
    **不允许退回 shim 冒充官方模板**。
    """

    name = "hf-official"

    def __init__(self, tokenizer, template_sha256: str) -> None:
        self.tokenizer = tokenizer
        if template_sha256 != CANONICAL_TEMPLATE_SHA256:
            raise IntegrityError(
                "template_sha_mismatch",
                "chat_template SHA 与 KAGGLE-20 §4 记录的 canonical 值不一致",
                expected=CANONICAL_TEMPLATE_SHA256,
                actual=template_sha256,
            )
        self.official_template_verified = True

    @classmethod
    def from_pretrained(cls, model_id: str, revision: str, allow_download: bool = False):
        if not allow_download:
            raise Blocked(
                "model_download_not_authorized",
                "未授权下载模型/tokenizer：本轮为 CPU 工程实现，不联网取权重",
            )
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as exc:  # pragma: no cover - 环境相关
            raise Blocked(
                "transformers_unavailable",
                "缺少锁定版本 transformers，禁止用未锁定版本顶替：%s" % exc,
            ) from exc
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        template = getattr(tokenizer, "chat_template", "") or ""
        digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
        return cls(tokenizer, digest)

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def render(self, messages: Sequence[Mapping], thinking: bool = False) -> RenderedSample:
        rendered_text = self.tokenizer.apply_chat_template(
            [dict(m) for m in messages],
            tokenize=False,
            add_generation_prompt=False,
        )
        # 官方模板当前没有 generation 区间：不能靠它推 mask，必须自建偏移。
        ids = list(self.tokenizer.encode(rendered_text, add_special_tokens=False))
        return RenderedSample(
            input_ids=ids,
            spans=[],
            bos_count=ids.count(BOS_TOKEN_ID),
            renderer=self.name,
            official_template_verified=True,
            notes=["span 需由逐段渲染 + 字节偏移映射得到，不得用字符串搜索猜 mask"],
        )

    def verify_template_bytes(self, template_bytes: bytes) -> None:
        digest = hashlib.sha256(template_bytes).hexdigest()
        if digest != CANONICAL_TEMPLATE_SHA256:
            raise IntegrityError(
                "template_sha_mismatch",
                "模板字节 SHA 与 canonical 值不一致",
                expected=CANONICAL_TEMPLATE_SHA256,
                actual=digest,
            )


def assert_text_in_input_ids(renderer: Renderer, rendered: RenderedSample, text: str) -> int:
    """断言 `text` 的 token 序列作为**连续子序列**出现在 `rendered.input_ids` 里。

    这是 🔴-1（KAGGLE-26 / Q0 报告）的机械检查：不搜索字符串、不信 span 声明，
    只看真实 token 序列。返回首次出现的起始下标；`text` 为空时返回 -1。
    """
    needle = renderer.encode(text)
    if not needle:
        return -1
    hay = list(rendered.input_ids)
    width = len(needle)
    for start in range(len(hay) - width + 1):
        if hay[start : start + width] == needle:
            return start
    raise IntegrityError(
        "assistant_content_dropped",
        "assistant content 没有出现在 input_ids 里（渲染层丢弃）",
        text=text,
        needle=needle,
    )


def assert_roundtrip_arguments(
    renderer: Renderer, arguments: Mapping, tool_name: str = "run_command"
) -> dict:
    """工具参数 roundtrip 无损：**必须真的经过渲染器**（KAGGLE-26 / Q0 报告 Y2）。

    旧实现只做 `json.loads(json.dumps(x)) == x`，与渲染器无关，恒真。
    现在：
    1. 用 renderer 渲染一条只含该工具调用的 assistant 消息；
    2. 从渲染结果的 action 通道取回 `arguments_text`（而不是重新序列化一遍）；
    3. 反序列化后与原对象逐字段比较。

    传入一个不实现渲染的假 renderer 时，本函数必然失败 —— 这正是它该有的行为。
    """
    import json

    payload = _stable_args(arguments)
    expected = dict(arguments)
    if json.loads(payload) != expected:
        raise IntegrityError("args_roundtrip_loss", "工具参数 roundtrip 丢信息", args=arguments)

    sample = renderer.render(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_name": tool_name,
                "arguments": expected,
                "step": 1,
                "loss_eligible": True,
                "phase": "localize",
            }
        ]
    )
    carriers = [s for s in sample.spans if s.role == "assistant" and s.arguments_text is not None]
    if not carriers:
        raise IntegrityError(
            "args_roundtrip_not_rendered",
            "渲染器没有输出 action 通道的参数原文，roundtrip 不成立",
            args=arguments,
        )
    recovered = json.loads(carriers[0].arguments_text)
    if recovered != expected:
        raise IntegrityError(
            "args_roundtrip_loss",
            "渲染后还原的参数与原对象不一致",
            expected=expected,
            actual=recovered,
        )
    return {
        "payload": payload,
        "recovered": recovered,
        "channel": carriers[0].channel,
        "span": [carriers[0].start, carriers[0].end],
    }
