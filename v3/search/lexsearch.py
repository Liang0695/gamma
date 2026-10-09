"""词法召回与输出预算（EXP-1 的规则侧实现）。

对应 KAGGLE-22 §3.4（四步五阶段）、§4.1（输出预算）、附录 A（EXP-1 协议）。
实现四个变体：

- **V0** 锚点直推：用路径样 token 直接匹配文件路径；
- **V1** 词法：按 idf 加权的"命中不同 token 数"排序；
- **V2** V1 + 测试桥（把 top-5 里第一个测试文件 import 的模块补进候选）；
- **V3** V1 的候选里剔除测试文件（测试只作桥）。

排序键（KAGGLE-19 §4）：① 命中不同锚点覆盖数降序；② 局部命中集中度；
③ 非测试优先；④ 路径字节序打破平局。**不得使用 gold。**
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from ..common.canonical import sha256_json
from ..common.errors import PolicyViolation

#: 输出预算（官方不可配置，见 locks/official-interface.json）。
HARD_OUTPUT_CHARS = 5000
TARGET_OUTPUT_CHARS = 1500

IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
DOTTED = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+")
PATHLIKE = re.compile(r"\b[\w./-]+\.(?:py|pyi|json|yaml|yml|toml|cfg|txt|md)\b")
FLAG = re.compile(r"(?<![\w-])--[A-Za-z][\w-]*")
ERRORNAME = re.compile(r"\b[A-Z][A-Za-z0-9_]*(?:Error|Exception|Warning)\b")
BACKTICK = re.compile(r"`([^`]+)`")
QUOTED = re.compile(r"[\"'“”‘’]([^\"'“”‘’]{3,120})[\"'“”‘’]")
CODEBLOCK = re.compile(r"```[a-zA-Z0-9_+-]*\n(.*?)```", re.DOTALL)

STOPWORDS = frozenset(
    """the a an and or of to in is are was were be been it this that these those for with
    from on at by as if then else when should would could can not no but do does did have has
    had i you we they he she my your our their use used using get got make made need needs
    want wants please help issue bug error expected actual instead rather than about into over
    after before while during while here there what which who whom whose why how all any some
    more most other others such only own same so too very just also""".split()
)


def is_test_path(path: str) -> bool:
    """测试路径判定（KAGGLE-22 §6.3：tests?/ 目录、test_*.py、*_test.py）。"""
    parts = path.replace("\\", "/").split("/")
    if any(part == "tests" or part == "test" for part in parts[:-1]):
        return True
    name = parts[-1]
    return name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py"


def src_side_priority(path: str) -> int:
    """③ 非测试优先：src 侧 0，docs_src/examples 1，tests/benchmarks 2（只作桥）。"""
    lowered = path.replace("\\", "/")
    if is_test_path(lowered):
        return 2
    if lowered.startswith("benchmarks/") or "/benchmarks/" in lowered:
        return 2
    if lowered.startswith("docs_src/") or "/docs_src/" in lowered:
        return 1
    if lowered.startswith("examples/") or "/examples/" in lowered:
        return 1
    return 0


@dataclass
class Anchors:
    literal: list[str] = field(default_factory=list)
    behaviour: list[str] = field(default_factory=list)
    api: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)

    def queries(self, max_queries: int = 3) -> list[str]:
        """LOCATE 的 anchor 行最多逐字引用 3 条原始查询串。"""
        ordered = list(self.literal) + list(self.behaviour) + list(self.api)
        return ordered[:max_queries]

    def all_tokens(self) -> list[str]:
        return list(dict.fromkeys(self.literal + self.behaviour + self.api))

    def empty(self) -> bool:
        return not (self.literal or self.behaviour or self.api or self.paths)

    def to_dict(self) -> dict:
        return {
            "literal": list(self.literal),
            "behaviour": list(self.behaviour),
            "api": list(self.api),
            "paths": list(self.paths),
        }


def extract_anchors(problem_statement: str, max_tokens: int = 24) -> Anchors:
    """阶段 0 的机械锚点抽取（不调用任何模型）。"""
    anchors = Anchors()
    text = problem_statement or ""

    for raw in BACKTICK.findall(text):
        token = raw.strip()
        if not token:
            continue
        if PATHLIKE.fullmatch(token):
            anchors.paths.append(token)
        elif DOTTED.fullmatch(token):
            anchors.literal.append(token)
        elif FLAG.fullmatch(token):
            anchors.literal.append(token)
        elif ERRORNAME.fullmatch(token):
            anchors.literal.append(token)
        elif IDENTIFIER.fullmatch(token):
            anchors.literal.append(token)
        else:
            anchors.behaviour.append(token)

    anchors.paths.extend(m for m in PATHLIKE.findall(text) if m not in anchors.paths)
    anchors.literal.extend(m for m in ERRORNAME.findall(text) if m not in anchors.literal)
    anchors.literal.extend(m for m in FLAG.findall(text) if m not in anchors.literal)

    for phrase in QUOTED.findall(text):
        cleaned = phrase.strip()
        if cleaned and cleaned not in anchors.behaviour and not PATHLIKE.fullmatch(cleaned):
            anchors.behaviour.append(cleaned)

    for block in CODEBLOCK.findall(text):
        for match in DOTTED.findall(block):
            if match not in anchors.api:
                anchors.api.append(match)

    for match in DOTTED.findall(text):
        if match not in anchors.literal and match not in anchors.api:
            anchors.literal.append(match)

    words = [
        w
        for w in IDENTIFIER.findall(text)
        if len(w) >= 4 and w.lower() not in STOPWORDS and not w.isdigit()
    ]
    for word in words:
        if word not in anchors.api and (len(word) >= 6):
            anchors.api.append(word)

    anchors.literal = anchors.literal[:max_tokens]
    anchors.behaviour = anchors.behaviour[:max_tokens]
    anchors.api = anchors.api[:max_tokens]
    anchors.paths = anchors.paths[:max_tokens]
    return anchors


def file_match_counts(files: Mapping[str, str], token: str) -> list[tuple[str, int]]:
    """C2：先出「文件:计数」表（只给计数，不给正文）。"""
    counts: list[tuple[str, int]] = []
    for path, text in files.items():
        hits = text.count(token)
        if hits:
            counts.append((path, hits))
    counts.sort(key=lambda item: (-item[1], item[0].encode("utf-8")))
    return counts


def document_frequencies(files: Mapping[str, str], tokens: Sequence[str]) -> dict[str, int]:
    df = {}
    for token in tokens:
        df[token] = sum(1 for text in files.values() if token in text)
    return df


def idf(df: int, total_docs: int) -> float:
    return math.log((1 + total_docs) / (1 + df)) + 1.0


@dataclass
class Candidate:
    path: str
    score: float
    distinct_queries: int
    hits: int
    concentration: float
    src_priority: int
    origin: str = "lexical"
    #: 候选**自带**的行区间（1 基、闭区间）——Q0 Y10 要求 RegionHit 用真实预测区间，
    #: 而不是在评测层写死 1..1。无可用行信息时保持 None，评测记 NA（不猜、不回填）。
    start_line: int | None = None
    end_line: int | None = None
    #: 行区间的来源说明（例如 "hit_lines" / "symbol_region"），供报告标注口径。
    line_span_source: str | None = None

    def rank_key(self):
        # ① 不同锚点覆盖降序 ② 局部命中集中度降序 ③ 非测试优先 ④ 路径字节序
        return (
            -self.distinct_queries,
            -self.concentration,
            self.src_priority,
            self.path.encode("utf-8"),
        )

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "score": round(self.score, 6),
            "distinct_queries": self.distinct_queries,
            "hits": self.hits,
            "concentration": round(self.concentration, 6),
            "src_priority": self.src_priority,
            "origin": self.origin,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "line_span_source": self.line_span_source,
        }


def _concentration(text: str, token: str) -> float:
    """局部命中集中度：命中行里最长连续命中块的占比（越集中越像"定义处"）。"""
    lines = text.splitlines()
    hit_indexes = [i for i, line in enumerate(lines) if token in line]
    if not hit_indexes:
        return 0.0
    best = current = 1
    for prev, cur in zip(hit_indexes, hit_indexes[1:]):
        current = current + 1 if cur - prev <= 2 else 1
        best = max(best, current)
    return best / float(len(hit_indexes))


def _hit_line_numbers(text: str, token: str) -> list[int]:
    """token 在 text 里出现的**行号**（1 基）。"""
    return [index + 1 for index, line in enumerate(text.splitlines()) if token in line]


def longest_hit_block(lines: Sequence[int]) -> tuple[int | None, int | None]:
    """命中行里最长的连续块（允许相邻行间隔 ≤1）→ `(start, end)` 闭区间。

    这是候选**自带**的行区间来源：它由真实命中行推出，不是评测层假定。
    没有命中行时返回 `(None, None)`，即"无行信息"，评测必须记 NA。
    """
    ordered = sorted({int(line) for line in lines})
    if not ordered:
        return (None, None)
    best_start = best_end = ordered[0]
    cur_start = prev = ordered[0]
    for line in ordered[1:]:
        if line - prev <= 2:
            prev = line
        else:
            if prev - cur_start > best_end - best_start:
                best_start, best_end = cur_start, prev
            cur_start = prev = line
    if prev - cur_start > best_end - best_start:
        best_start, best_end = cur_start, prev
    return (best_start, best_end)


def lexical_rank(files: Mapping[str, str], tokens: Sequence[str]) -> list[Candidate]:
    """V1：idf 加权 + 命中不同 token 数。"""
    total = max(1, len(files))
    df = document_frequencies(files, tokens)
    per_file: dict[str, dict] = defaultdict(
        lambda: {"tokens": set(), "hits": 0, "score": 0.0, "conc": 0.0, "lines": []}
    )
    for token in tokens:
        if df[token] == 0 or df[token] == total:
            continue
        weight = idf(df[token], total)
        for path, text in files.items():
            hits = text.count(token)
            if not hits:
                continue
            entry = per_file[path]
            entry["tokens"].add(token)
            entry["hits"] += hits
            entry["score"] += weight * math.log(1 + hits)
            entry["conc"] = max(entry["conc"], _concentration(text, token))
            # Y10：同时记录命中行号，供候选自带真实行区间（而不是评测层写死 1..1）。
            entry["lines"].extend(_hit_line_numbers(text, token))
    candidates = []
    for path, data in per_file.items():
        start_line, end_line = longest_hit_block(data["lines"])
        candidates.append(
            Candidate(
                path=path,
                score=data["score"],
                distinct_queries=len(data["tokens"]),
                hits=data["hits"],
                concentration=data["conc"],
                src_priority=src_side_priority(path),
                start_line=start_line,
                end_line=end_line,
                line_span_source="hit_lines_longest_block" if start_line is not None else None,
            )
        )
    candidates.sort(key=lambda c: c.rank_key())
    return candidates


def anchor_direct(files: Mapping[str, str], paths: Sequence[str]) -> list[Candidate]:
    """V0：用路径样 token 直接匹配文件路径。"""
    out: list[Candidate] = []
    for candidate_path in sorted(files, key=lambda p: p.encode("utf-8")):
        for token in paths:
            token_norm = token.replace("\\", "/").lstrip("./")
            if candidate_path.endswith(token_norm) or candidate_path == token_norm:
                out.append(
                    Candidate(
                        path=candidate_path,
                        score=1.0,
                        distinct_queries=1,
                        hits=1,
                        concentration=1.0,
                        src_priority=src_side_priority(candidate_path),
                        origin="path_token",
                    )
                )
                break
    out.sort(key=lambda c: c.rank_key())
    return out


def imported_modules(test_text: str) -> list[str]:
    """C4 测试当桥：取出测试文件 import 的模块名。"""
    modules: list[str] = []
    for line in test_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("from ") and " import " in stripped:
            module = stripped[5:].split(" import ")[0].strip()
        elif stripped.startswith("import "):
            module = stripped[7:].split(" as ")[0].split(",")[0].strip()
        else:
            continue
        module = module.split(".")[0]
        if module and module not in modules:
            modules.append(module)
    return modules


def bridge_from_tests(
    files: Mapping[str, str], ranked: Sequence[Candidate], top_n: int = 5
) -> list[Candidate]:
    """V2：把 top-N 里第一个测试文件所 import 的模块追加进候选。"""
    bridged: list[Candidate] = []
    for candidate in ranked[:top_n]:
        if not is_test_path(candidate.path):
            continue
        for module in imported_modules(files.get(candidate.path, "")):
            for path in files:
                if path.replace("\\", "/").startswith(module + "/") and not is_test_path(path):
                    bridged.append(
                        Candidate(
                            path=path,
                            score=candidate.score * 0.5,
                            distinct_queries=1,
                            hits=1,
                            concentration=candidate.concentration,
                            src_priority=src_side_priority(path),
                            origin="test_bridge",
                        )
                    )
        break
    bridged.sort(key=lambda c: c.rank_key())
    return bridged


def exclude_tests(ranked: Sequence[Candidate]) -> list[Candidate]:
    """V3：候选里剔除测试文件（测试只作桥，不作候选）。"""
    out = [c for c in ranked if not is_test_path(c.path)]
    out.sort(key=lambda c: c.rank_key())
    return out


def recall_at_k(candidates: Sequence[Candidate], gold_files: Iterable[str], k: int) -> bool:
    """Hit@k 的原始定义：至少命中一个 gold 文件（原稿误名为 FileRecall）。"""
    top = {c.path for c in candidates[:k]}
    return bool(top & {g.replace("\\", "/") for g in gold_files})


def true_recall_at_k(candidates: Sequence[Candidate], gold_files: Iterable[str], k: int) -> float:
    """真正的 Recall@k = 命中 gold 文件数 / 全部 gold 文件数。"""
    gold = {g.replace("\\", "/") for g in gold_files}
    if not gold:
        return 0.0
    top = {c.path for c in candidates[:k]}
    return len(top & gold) / float(len(gold))


def region_hit(chosen: Mapping | None, gold_hunks: Sequence[Mapping]) -> bool | None:
    """RegionHit@1：chosen 的 (file, 行区间) 与 gold hunk 相交。

    返回值三态（Q0 Y10）：

    - `True` / `False`：chosen 带**真实**行区间时给出的判定；
    - `None`：**NA —— 未评估**。没有候选、候选没有行区间、或 chosen 里
      `start`/`end` 显式为 `None` 时返回 NA。

    旧实现把缺失区间折成 `0`（等价第 1 行），于是"命中"退化成"gold hunk 是否覆盖
    第 1 行"，报出 0.125/0.25 这类看似合理的假数字。现在缺失就是缺失，
    由调用方从分母里剔除，**不允许**用假区间造指标。
    """
    if not chosen:
        return None
    raw_start = chosen.get("start", chosen.get("start_line"))
    raw_end = chosen.get("end", chosen.get("end_line"))
    if raw_start is None or raw_end is None:
        return None
    path = str(chosen.get("path", "")).replace("\\", "/")
    try:
        start = int(raw_start)
        end = int(raw_end)
    except (TypeError, ValueError):
        return None
    for hunk in gold_hunks:
        if str(hunk.get("file", "")).replace("\\", "/") != path:
            continue
        hunk_start = int(hunk.get("start_line", 0))
        hunk_end = hunk_start + int(hunk.get("lines", 1)) - 1
        if start <= hunk_end and end >= hunk_start:
            return True
    return False


def narrow_output(
    header: str,
    rows: Sequence[str],
    target_chars: int = TARGET_OUTPUT_CHARS,
    hard_chars: int = HARD_OUTPUT_CHARS,
) -> dict:
    """两段式收窄：先头部+计数，再按预算逐行填，直到接近目标字符数。

    返回 `truncated` 标记 —— 「不能静默丢关键证据」。
    """
    if target_chars > hard_chars:
        raise PolicyViolation("budget_inverted", "目标字符数不得大于官方硬上限")
    out = [header]
    size = len(header) + 1
    for row in rows:
        projected = size + len(row) + 1
        if projected > target_chars:
            break
        out.append(row)
        size = projected
    truncated = len(out) - 1 < len(rows)
    if truncated:
        out.append("[truncated: %d/%d rows shown]" % (len(out) - 1, len(rows)))
    text = "\n".join(out)
    return {
        "text": text[:hard_chars],
        "chars": len(text),
        "truncated": truncated,
        "truncation_flags": ["row_budget"] if truncated else [],
        "hard_limit_hit": len(text) >= hard_chars,
    }


def candidate_table(candidates: Sequence[Candidate], limit: int = 10) -> list[str]:
    """candidates → 一行一条 `path  hits=.. q=.. conf=..`（不回贴文件内容）。"""
    rows = []
    for candidate in candidates[:limit]:
        confidence = "high" if candidate.distinct_queries >= 2 and candidate.concentration >= 0.5 else (
            "med" if candidate.distinct_queries >= 1 else "low"
        )
        rows.append(
            "%s  hits=%d queries=%d conf=%s"
            % (candidate.path, candidate.hits, candidate.distinct_queries, confidence)
        )
    return rows


def run_variants(files: Mapping[str, str], anchors: Anchors, top_k: int = 10) -> dict:
    """一次算出 V0/V1/V2/V3 的候选表（EXP-1 用）。"""
    tokens = anchors.all_tokens()
    v0 = anchor_direct(files, anchors.paths)
    v1 = lexical_rank(files, tokens)
    v2 = v1 + [c for c in bridge_from_tests(files, v1) if c.path not in {x.path for x in v1}]
    v2.sort(key=lambda c: c.rank_key())
    v3 = exclude_tests(v1)
    return {
        "tokens": tokens,
        "V0": v0[:top_k],
        "V1": v1[:top_k],
        "V2": v2[:top_k],
        "V3": v3[:top_k],
        "candidate_table_sha256": sha256_json(
            {name: [c.to_dict() for c in variants] for name, variants in
             (("V0", v0[:top_k]), ("V1", v1[:top_k]), ("V2", v2[:top_k]), ("V3", v3[:top_k]))}
        ),
    }
