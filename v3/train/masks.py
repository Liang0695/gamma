"""显式 labels 构造（KAGGLE-20 §4「显式构造 labels 为主」）。

规则：
- system / user / tool / padding **全部** `-100`；
- assistant 历史与**失败动作** `-100`（context-only）；
- 只监督：本窗口的目标后继 assistant 动作（工具名、参数、完整编辑片段、短计划/定位）
  与合法结束标记；
- **按 token 偏移映射决定 mask，不用"搜索某段字符串"**；
- 不依赖 `assistant_only_loss=True`：官方模板当前没有 generation 区间。

通道口径（KAGGLE-26 / Q0 报告 🔴-1、Y3 的显式规定，与 `template.Span.channel` 对齐）：

- `channel="action"` 的 assistant span：`loss_eligible` 为真时参与监督。**短计划/定位**
  属于这一通道（thinking 关闭时它是 `content`，thinking 打开时它是工具名 + 参数）；
- `channel="reasoning"` 的 assistant span：thinking 打开时的可见 reasoning，
  **永远 `supervised=False`**，只作 context；
- 因此"assistant 的 content 被静默丢弃"这种状态在本实现里不存在：要么它以
  reasoning 通道进 context，要么它以 action 通道进监督，二者必居其一。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ..common.errors import PolicyViolation
from .template import BOS_TOKEN_ID, PAD_TOKEN_ID, RenderedSample, Renderer, Span

IGNORE_INDEX = -100


@dataclass
class LabelResult:
    input_ids: list[int]
    labels: list[int]
    span_report: list[dict] = field(default_factory=list)
    padding: int = 0

    @property
    def supervised_tokens(self) -> int:
        return sum(1 for value in self.labels if value != IGNORE_INDEX)

    def loss_counts(self) -> dict:
        """按 span_report 真正分桶统计（KAGGLE-26 / Q0 报告 Y13）。

        旧实现只累加 `supervised`，其余 5 个桶恒为 0，任一样本都返回同一个错的分桶。
        现在的口径（写清以免再次变成"恒零桩"）：

        - `supervised`：真的参与 loss 的 token 数（`labels != IGNORE_INDEX`）；
        - `system` / `tool` / `user`：这些 role 的 **context token 数**。
          它们按设计**永不**参与 loss，所以旧口径下必然恒为 0；
          这里给出的是"这些 role 贡献了多少 context token"，逐样本不同；
        - `assistant_context` / `reasoning_context`：未参与 loss 的 assistant span
          长度，按 `channel` 分流；
        - `padding`：取自 `padding` 字段。

        各 role 是否真的没进 loss，由 `assert_mask_invariants()` 断言，不靠本函数。
        """
        counts = {
            "supervised": 0,
            "system": 0,
            "tool": 0,
            "user": 0,
            "assistant_context": 0,
            "reasoning_context": 0,
            "padding": 0,
        }
        counts["supervised"] = self.supervised_tokens
        counts["padding"] = int(self.padding)
        for entry in self.span_report:
            role = entry.get("role")
            span_len = max(0, int(entry.get("end", 0)) - int(entry.get("start", 0)))
            loss_tokens = int(entry.get("loss_tokens") or 0)
            if role in ("system", "tool", "user"):
                counts[role] += span_len
            elif role == "assistant":
                if loss_tokens:
                    # 已计入 supervised，不重复累加
                    continue
                if entry.get("channel") == "reasoning":
                    counts["reasoning_context"] += span_len
                else:
                    counts["assistant_context"] += span_len
        return counts

    def to_dict(self) -> dict:
        return {
            "length": len(self.input_ids),
            "supervised_tokens": self.supervised_tokens,
            "padding": self.padding,
            "spans": self.span_report,
        }


def build_labels(rendered: RenderedSample, pad_to: int | None = None) -> LabelResult:
    """把 span 映射成 labels。只有 `supervised=True` 的 assistant span 参与 loss。

    本函数**不修改入参** `rendered`（KAGGLE-26 / Q0 报告 🔵-1）：旧实现就地 append 到
    `rendered.input_ids`，同一个 `RenderedSample` 二次调用会因长度变长而抛
    `pad_shorter_than_sample`。现在返回新的 list。
    """
    labels = [IGNORE_INDEX] * rendered.length
    report: list[dict] = []
    for span in rendered.spans:
        entry = span.to_dict()
        if span.supervised and span.role == "assistant":
            if span.end <= span.start:
                raise PolicyViolation("empty_supervised_span", "被监督 span 长度为 0")
            for index in range(span.start, span.end):
                labels[index] = rendered.input_ids[index]
            entry["loss_tokens"] = span.end - span.start
        else:
            entry["loss_tokens"] = 0
        report.append(entry)

    input_ids = list(rendered.input_ids)
    padding = 0
    if pad_to is not None:
        if pad_to < len(input_ids):
            raise PolicyViolation(
                "pad_shorter_than_sample", "padding 长度小于样本长度"
            )
        padding = pad_to - len(input_ids)
        input_ids = input_ids + [PAD_TOKEN_ID] * padding
        labels = labels + [IGNORE_INDEX] * padding
    return LabelResult(
        input_ids=input_ids,
        labels=labels,
        span_report=report,
        padding=padding,
    )


def assert_mask_invariants(rendered: RenderedSample, labels: LabelResult) -> dict:
    """20 个 fixture 的公共断言集合，返回统计供报告使用。"""
    problems: list[str] = []

    def loss_in(spans: Sequence[Span]) -> int:
        total = 0
        for span in spans:
            if span.role in ("system", "user", "tool", "padding", "bos"):
                continue
            total += sum(
                1 for i in range(span.start, span.end) if labels.labels[i] != IGNORE_INDEX
            )
        return total

    system_spans = [s for s in rendered.spans if s.role == "system"]
    tool_spans = [s for s in rendered.spans if s.role == "tool"]
    user_spans = [s for s in rendered.spans if s.role == "user"]

    for role, spans in (("system", system_spans), ("tool", tool_spans), ("user", user_spans)):
        for span in spans:
            if any(labels.labels[i] != IGNORE_INDEX for i in range(span.start, span.end)):
                problems.append("%s span 参与了 loss（%d-%d）" % (role, span.start, span.end))

    bad_actions = [s for s in rendered.spans if s.role == "assistant" and not s.supervised]
    for span in bad_actions:
        if any(labels.labels[i] != IGNORE_INDEX for i in range(span.start, span.end)):
            problems.append("失败/历史 assistant 动作参与了 loss（step=%s）" % span.step)

    # reasoning 通道永远 context-only（🔴-1 / Y3 的显式口径）
    reasoning_spans = [s for s in rendered.spans if s.channel == "reasoning"]
    for span in reasoning_spans:
        if span.supervised:
            problems.append("reasoning 通道被标记为 supervised（step=%s）" % span.step)
    reasoning_tokens = sum(span.end - span.start for span in reasoning_spans)
    assistant_context_tokens = sum(
        span.end - span.start
        for span in rendered.spans
        if span.role == "assistant" and not span.supervised and span.channel != "reasoning"
    )

    target_tokens = sum(
        1
        for span in rendered.spans
        if span.role == "assistant" and span.supervised
        for i in range(span.start, span.end)
        if labels.labels[i] != IGNORE_INDEX
    )
    if target_tokens <= 0:
        problems.append("没有任何目标动作进入 loss")

    if labels.padding:
        tail = labels.labels[len(labels.input_ids) - labels.padding :]
        if any(value != IGNORE_INDEX for value in tail):
            problems.append("padding 未全部 mask 为 -100")

    bos_positions = [i for i, token in enumerate(labels.input_ids) if token == BOS_TOKEN_ID]
    if len(bos_positions) > 1:
        problems.append("BOS 出现 %d 次（重复）" % len(bos_positions))

    if problems:
        raise PolicyViolation("mask_invariants_failed", "; ".join(problems), problems=problems)
    return {
        "system_loss_tokens": 0,
        "tool_loss_tokens": 0,
        "user_loss_tokens": 0,
        "bad_action_loss_tokens": 0,
        "target_loss_tokens": target_tokens,
        "bos_count": len(bos_positions),
        "padding": labels.padding,
        "supervised_tokens": labels.supervised_tokens,
        "reasoning_span_count": len(reasoning_spans),
        "reasoning_context_tokens": reasoning_tokens,
        "assistant_context_tokens": assistant_context_tokens,
        "loss_counts": labels.loss_counts(),
    }


def render_and_mask(
    renderer: Renderer,
    messages: Sequence[Mapping],
    thinking: bool = False,
    pad_to: int | None = None,
) -> tuple[RenderedSample, LabelResult, dict]:
    rendered = renderer.render(messages, thinking=thinking)
    labels = build_labels(rendered, pad_to=pad_to)
    stats = assert_mask_invariants(rendered, labels)
    stats["renderer"] = rendered.renderer
    stats["official_template_verified"] = rendered.official_template_verified
    return rendered, labels, stats
