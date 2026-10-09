"""LOCATE 统一状态块（KAGGLE-19 §4 / KAGGLE-22 §3.1、§8.4）。

统一字段：schema_version、task_id、base_commit、tree_sha、anchors、queries、
candidates[{path,symbol,start,end,source,observation_ref,evidence,confidence}]、
chosen 或 null、next_action、calls_used、truncation_flags。

硬规则：
- 路径必须是**当前源码里真实存在**的文件，行号必须来自**真实观察**；
- 训练窗口不得带未来 chosen、gold 或未发生的测试结果；
- 文本形态必须 ≤250 词，且**禁止回贴文件内容**（最多逐字引用 ≤2 行 `path:line: 片段`）。
"""

from __future__ import annotations

import re
from typing import Callable, Mapping, Sequence

from ..common.errors import MissingInput, PolicyViolation

SCHEMA_VERSION = "locate/1"
CONFIDENCE = ("high", "med", "low")
CANDIDATE_SOURCES = ("lexical", "path_token", "test_bridge", "graph", "outline", "analyzer")
NEXT_ACTIONS = (
    "read",
    "escalate_l1",
    "escalate_l2",
    "escalate_l3",
    "edit",
    "fail",
)
EVIDENCE_CLASSES = ("literal", "behaviour", "api", "symbol", "none")

REQUIRED_FIELDS = (
    "schema_version",
    "task_id",
    "base_commit",
    "tree_sha",
    "anchors",
    "queries",
    "candidates",
    "chosen",
    "next_action",
    "calls_used",
    "truncation_flags",
)

CANDIDATE_FIELDS = (
    "path",
    "symbol",
    "start",
    "end",
    "source",
    "observation_ref",
    "evidence",
    "confidence",
)

#: 250 词硬要求（KAGGLE-22 §3.1）。
MAX_WORDS = 250
MAX_CANDIDATES = 5
MAX_QUOTE_LINES = 2
MAX_LOCATE_CHARS = 4000

_FORBIDDEN_KEYS = ("gold", "gold_files", "gold_hunks", "test_patch", "future_commit", "f2p")


class LocateState:
    """LOCATE 块的结构化表示。"""

    def __init__(self, payload: Mapping) -> None:
        self.payload = dict(payload)

    # ---- 构造 ----

    @classmethod
    def create(
        cls,
        task_id: str,
        base_commit: str,
        tree_sha: str,
        anchors: Mapping,
        queries: Sequence[str],
    ) -> "LocateState":
        return cls(
            {
                "schema_version": SCHEMA_VERSION,
                "task_id": task_id,
                "base_commit": base_commit,
                "tree_sha": tree_sha,
                "anchors": dict(anchors),
                "queries": list(queries)[:3],
                "candidates": [],
                "chosen": None,
                "next_action": "read",
                "calls_used": 0,
                "truncation_flags": [],
            }
        )

    # ---- 校验 ----

    def validate(self, path_exists: Callable[[str], bool] | None = None) -> None:
        missing = [f for f in REQUIRED_FIELDS if f not in self.payload]
        if missing:
            raise MissingInput("locate_missing_fields", "LOCATE 缺少字段", missing=missing)
        if self.payload["schema_version"] != SCHEMA_VERSION:
            raise PolicyViolation(
                "locate_bad_version", "LOCATE schema_version 必须是 %s" % SCHEMA_VERSION
            )
        if self.payload["next_action"] not in NEXT_ACTIONS:
            raise PolicyViolation("locate_bad_action", "非法 next_action：%r" % (self.payload["next_action"],))
        if len(self.payload["candidates"]) > MAX_CANDIDATES:
            raise PolicyViolation(
                "locate_too_many_candidates",
                "candidates 最多 %d 条，实际 %d" % (MAX_CANDIDATES, len(self.payload["candidates"])),
            )
        for candidate in self.payload["candidates"]:
            self._validate_candidate(candidate, path_exists)
        chosen = self.payload["chosen"]
        if chosen is not None:
            self._validate_candidate(chosen, path_exists, chosen=True)
            paths = {c["path"] for c in self.payload["candidates"]}
            if chosen["path"] not in paths:
                raise PolicyViolation(
                    "locate_chosen_not_in_candidates", "chosen 必须来自 candidates"
                )
        if not isinstance(self.payload["calls_used"], int) or self.payload["calls_used"] < 0:
            raise PolicyViolation("locate_bad_calls", "calls_used 必须是非负整数")

    def _validate_candidate(
        self, candidate: Mapping, path_exists: Callable[[str], bool] | None, chosen: bool = False
    ) -> None:
        missing = [f for f in CANDIDATE_FIELDS if f not in candidate]
        if missing:
            raise MissingInput(
                "candidate_missing_fields",
                "候选%s缺少字段" % ("(chosen)" if chosen else ""),
                missing=missing,
            )
        path = str(candidate["path"]).replace("\\", "/")
        if not path or path.startswith("/") or ".." in path.split("/"):
            raise PolicyViolation("candidate_bad_path", "候选路径必须是仓库内相对路径：%r" % path)
        if path_exists is not None and not path_exists(path):
            raise PolicyViolation(
                "candidate_path_missing", "候选路径在当前源码中不存在：%s" % path
            )
        if candidate["source"] not in CANDIDATE_SOURCES:
            raise PolicyViolation("candidate_bad_source", "非法 source：%r" % (candidate["source"],))
        if candidate["confidence"] not in CONFIDENCE:
            raise PolicyViolation("candidate_bad_confidence", "非法 confidence：%r" % (candidate["confidence"],))
        if candidate["evidence"] not in EVIDENCE_CLASSES:
            raise PolicyViolation("candidate_bad_evidence", "非法 evidence：%r" % (candidate["evidence"],))
        start, end = int(candidate["start"]), int(candidate["end"])
        if start <= 0 or end < start:
            raise PolicyViolation(
                "candidate_bad_lines", "行号必须来自真实观察且 start<=end：%r-%r" % (start, end)
            )
        if not str(candidate["observation_ref"]).strip():
            raise PolicyViolation(
                "candidate_no_observation",
                "候选缺少 observation_ref：行号必须有真实读取证据",
            )
        if candidate["source"] in ("lexical", "path_token", "test_bridge") and candidate["evidence"] == "none":
            raise PolicyViolation(
                "candidate_evidence_required",
                "检索来源的候选必须给出 literal/behaviour/api/symbol 证据",
            )

    def assert_no_future_information(self) -> None:
        """训练窗口不得带未来 chosen、gold 或未发生的测试结果。"""
        found: list[str] = []

        def walk(node, path: str) -> None:
            if isinstance(node, Mapping):
                for key, value in node.items():
                    if any(bad in str(key).lower() for bad in _FORBIDDEN_KEYS):
                        found.append("%s.%s" % (path, key))
                    walk(value, "%s.%s" % (path, key))
            elif isinstance(node, (list, tuple)):
                for index, value in enumerate(node):
                    walk(value, "%s[%d]" % (path, index))

        walk(self.payload, "locate")
        if found:
            raise PolicyViolation("locate_leak", "LOCATE 含未来/gold 信息：%s" % sorted(found))

    # ---- 文本形态 ----

    def to_block(self) -> str:
        """渲染成 LOCATE v1 文本块（≤250 词，禁止回贴文件内容）。"""
        anchors = self.payload["anchors"]
        lines = ["LOCATE v1"]
        anchor_bits = []
        for key in ("literal", "behaviour", "api"):
            for token in (anchors.get(key) or [])[:3]:
                anchor_bits.append(token)
        lines.append("anchor: %s" % (" | ".join(anchor_bits[:3]) or "none"))
        lines.append("candidates:")
        for candidate in self.payload["candidates"][:MAX_CANDIDATES]:
            why = str(candidate.get("why", candidate.get("evidence", ""))).replace("\n", " ")[:80]
            lines.append(
                "  - %s:%d-%d %s conf=%s why=%s"
                % (
                    candidate["path"],
                    int(candidate["start"]),
                    int(candidate["end"]),
                    candidate.get("symbol", ""),
                    candidate["confidence"],
                    why,
                )
            )
        chosen = self.payload["chosen"]
        if chosen:
            lines.append(
                "chosen: %s:%d-%d %s"
                % (chosen["path"], int(chosen["start"]), int(chosen["end"]), chosen.get("symbol", ""))
            )
        else:
            lines.append("chosen: none")
        lines.append(
            "search: {queries: %d, tool_calls: %d, escalate: %s}"
            % (
                self.payload.get("queries_used", len(self.payload["queries"])),
                self.payload["calls_used"],
                self.payload["next_action"].replace("escalate_", "") if self.payload["next_action"].startswith("escalate_") else "none",
            )
        )
        lines.append("unknown: %s" % (", ".join(self.payload.get("unknown") or ["none"])))
        return "\n".join(lines)

    def validate_block(self, block: str) -> None:
        """文本契约检查：词数、字符数、禁止回贴正文。"""
        words = block.split()
        if len(words) > MAX_WORDS:
            raise PolicyViolation(
                "locate_block_too_long", "LOCATE 块 %d 词，超过 %d 词硬要求" % (len(words), MAX_WORDS)
            )
        if len(block) > MAX_LOCATE_CHARS:
            raise PolicyViolation("locate_block_too_many_chars", "LOCATE 块字符数超限")
        quoted = 0
        for line in block.splitlines():
            if re.match(r"^\s*[-*]?\s*[\w./-]+\.py:\d+: ", line):
                quoted += 1
        if quoted > MAX_QUOTE_LINES:
            raise PolicyViolation(
                "locate_regurgitation", "LOCATE 只允许逐字引用 ≤%d 行" % MAX_QUOTE_LINES
            )

    def to_training_target(self) -> dict:
        """训练目标：只输出块 + 结构化字段；不监督长篇内部推理。"""
        self.assert_no_future_information()
        return {"block": self.to_block(), "state": self.payload}

    def to_dict(self) -> dict:
        return dict(self.payload)

    def add_candidate(self, candidate: Mapping) -> None:
        self.payload["candidates"].append(dict(candidate))
        self.payload["candidates"] = self.payload["candidates"][:MAX_CANDIDATES]

    def mark_truncated(self, flag: str) -> None:
        flags = self.payload.setdefault("truncation_flags", [])
        if flag not in flags:
            flags.append(flag)


def parse_block(block: str) -> dict:
    """解析 LOCATE v1 文本块（父 agent 插值前的最小解析器）。"""
    state: dict = {"candidates": [], "chosen": None, "queries": [], "truncation_flags": []}
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("anchor:"):
            state["anchor_line"] = stripped[len("anchor:") :].strip()
        elif stripped.startswith("- ") and ".py:" in stripped:
            match = re.match(r"-\s+(\S+):(\d+)-(\d+)\s+(\S*)\s+conf=(\w+)", stripped)
            if match:
                state["candidates"].append(
                    {
                        "path": match.group(1),
                        "start": int(match.group(2)),
                        "end": int(match.group(3)),
                        "symbol": match.group(4),
                        "confidence": match.group(5),
                    }
                )
        elif stripped.startswith("chosen:"):
            value = stripped[len("chosen:") :].strip()
            state["chosen"] = None if value.startswith("none") else value
        elif stripped.startswith("search:"):
            numbers = re.findall(r"tool_calls:\s*(\d+)", stripped)
            if numbers:
                state["calls_used"] = int(numbers[0])
        elif stripped.startswith("unknown:"):
            state["unknown"] = stripped[len("unknown:") :].strip()
    return state
