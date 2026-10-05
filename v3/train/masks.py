"""显式 labels 构造（KAGGLE-20 §4「显式构造 labels 为主」）。

规则：
- system / user / tool / padding **全部** `-100`；
- assistant 历史与**失败动作** `-100`（context-only）；
- 只监督：本窗口的目标后继 assistant 动作（工具名、参数、完整编辑片段、短计划/定位）
  与合法结束标记；
- **按 token 偏移映射决定 mask，不用"搜索某段字符串"**；
- 不依赖 `assistant_only_loss=True`：官方模板当前没有 generation 区间。
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
        counts = {"supervised": 0, "system": 0, "tool": 0, "user": 0, "assistant_context": 0, "padding": 0}
        for index, value in enumerate(self.labels):
            if value == IGNORE_INDEX:
                continue
            counts["supervised"] += 1
        return counts

    def to_dict(self) -> dict:
        return {
            "length": len(self.input_ids),
            "supervised_tokens": self.supervised_tokens,
            "padding": self.padding,
            "spans": self.span_report,
        }


def build_labels(rendered: RenderedSample, pad_to: int | None = None) -> LabelResult:
    """把 span 映射成 labels。只有 `supervised=True` 的 assistant span 参与 loss。"""
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

    padding = 0
    if pad_to is not None:
        if pad_to < rendered.length:
            raise PolicyViolation(
                "pad_shorter_than_sample", "padding 长度小于样本长度"
            )
        padding = pad_to - rendered.length
        rendered.input_ids = rendered.input_ids + [PAD_TOKEN_ID] * padding
        labels = labels + [IGNORE_INDEX] * padding
    return LabelResult(
        input_ids=list(rendered.input_ids),
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
