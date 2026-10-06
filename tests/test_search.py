"""搜索侧测试：符号轮廓、词法召回、LOCATE、预算协议、微索引失效。"""

from __future__ import annotations

import unittest

from v3.common.errors import IntegrityError, MissingInput, PolicyViolation
from v3.search.index import (
    IndexRegistry,
    MicroIndex,
    assert_safe_index_target,
    build_index_command,
    tree_sha_for_files,
)
from v3.search.lexsearch import (
    HARD_OUTPUT_CHARS,
    TARGET_OUTPUT_CHARS,
    anchor_direct,
    candidate_table,
    exclude_tests,
    extract_anchors,
    is_test_path,
    lexical_rank,
    narrow_output,
    recall_at_k,
    region_hit,
    src_side_priority,
    true_recall_at_k,
)
from v3.search.locate import LocateState, parse_block
from v3.search.outline import find_symbol, outline_from_source, outline_regex
from v3.search.protocol import (
    SEARCH_CALL_BUDGET,
    S0Policy,
    S1Policy,
    SearchBudget,
    SearchSession,
    assert_same_skeleton,
    assert_single_writer,
    build_count_command,
    build_grep_command,
    build_read_command,
    escape_shell_arg,
    four_arm_matrix,
    policies,
)

SOURCE = '''\
"""module docstring"""

import functools


def top_level(a=1, b=2):
    def nested(x):
        return x + a

    return nested(b)


class Client:
    """A client."""

    @property
    def name(self):
        return "c"

    @functools.lru_cache(maxsize=8)
    async def fetch(self, url):
        return url

    class Inner:
        def deep(self):
            return 1


async def top_async():
    return None
'''

BROKEN_SOURCE = "def broken(:\n    pass\n"


class OutlineTests(unittest.TestCase):
    def test_ast_outline_covers_methods_async_nested_and_decorators(self) -> None:
        symbols = outline_from_source(SOURCE)
        qualnames = {s.qualname for s in symbols}
        self.assertIn("top_level", qualnames)
        self.assertIn("top_level.nested", qualnames)
        self.assertIn("Client", qualnames)
        self.assertIn("Client.name", qualnames)
        self.assertIn("Client.fetch", qualnames)
        self.assertIn("Client.Inner", qualnames)
        self.assertIn("Client.Inner.deep", qualnames)
        self.assertIn("top_async", qualnames)

        fetch = [s for s in symbols if s.qualname == "Client.fetch"][0]
        self.assertTrue(fetch.is_async)
        self.assertEqual(fetch.kind, "async_method")
        self.assertEqual(len(fetch.decorators), 1)
        # decorator 行必须包含在区间内
        self.assertLess(fetch.start_line, fetch.end_line)
        self.assertIn("@functools.lru_cache", SOURCE.splitlines()[fetch.start_line - 1])

        name = [s for s in symbols if s.qualname == "Client.name"][0]
        self.assertEqual(name.kind, "method")
        self.assertFalse(name.is_async)
        self.assertIn("property", name.decorators)

        nested = [s for s in symbols if s.qualname == "top_level.nested"][0]
        self.assertEqual(nested.kind, "nested_function")

    def test_regex_fallback_matches_ast_on_indented_methods_and_async(self) -> None:
        symbols = outline_regex(SOURCE)
        qualnames = {s.qualname for s in symbols}
        self.assertIn("Client.fetch", qualnames)
        self.assertIn("top_async", qualnames)
        fetch = [s for s in symbols if s.qualname == "Client.fetch"][0]
        self.assertTrue(fetch.is_async)
        self.assertEqual(fetch.kind, "async_method")
        self.assertTrue(all(s.source == "regex" for s in symbols))

    def test_regex_path_is_used_for_unparsable_source(self) -> None:
        symbols = outline_from_source(BROKEN_SOURCE)
        self.assertEqual([s.name for s in symbols], ["broken"])
        self.assertEqual(symbols[0].source, "regex")

    def test_find_symbol_reports_ambiguity(self) -> None:
        symbols = outline_from_source(SOURCE)
        self.assertEqual(len(find_symbol(symbols, "Client.fetch")), 1)
        matches = find_symbol(symbols, "deep")
        self.assertEqual([m.qualname for m in matches], ["Client.Inner.deep"])


class LexicalTests(unittest.TestCase):
    FILES = {
        "alpha/routing.py": "def handle(request):\n    return dispatch(request)\n\ndef dispatch(request):\n    return request.payload\n",
        "alpha/util.py": "def helper(x):\n    return x\n",
        "alpha/tests/test_routing.py": "from alpha.routing import handle\n\ndef test_timeout():\n    assert handle(None) is None\n",
        "docs_src/examples/tutorial.py": "def dispatch_hint():\n    return 'dispatch'\n",
    }

    def test_test_path_detection_and_src_priority(self) -> None:
        self.assertTrue(is_test_path("alpha/tests/test_routing.py"))
        self.assertTrue(is_test_path("pkg/test_x.py"))
        self.assertTrue(is_test_path("pkg/x_test.py"))
        self.assertFalse(is_test_path("alpha/routing.py"))
        self.assertEqual(src_side_priority("alpha/routing.py"), 0)
        self.assertEqual(src_side_priority("docs_src/examples/tutorial.py"), 1)
        self.assertEqual(src_side_priority("alpha/tests/test_routing.py"), 2)

    def test_anchor_extraction_buckets(self) -> None:
        anchors = extract_anchors(
            "In `alpha/routing.py` the flag `--strict` raises `KeyError`; "
            "expected \"returns None\" and `dispatch` is called."
        )
        self.assertIn("--strict", anchors.literal)
        self.assertIn("KeyError", anchors.literal)
        self.assertIn("dispatch", anchors.literal)
        self.assertIn("alpha/routing.py", anchors.paths)
        self.assertTrue(any("returns None" in item for item in anchors.behaviour))
        self.assertLessEqual(len(anchors.queries(3)), 3)

    def test_anchor_extraction_on_double_backtick_escapes(self) -> None:
        """markdown 转义 `` `x` `` 仍应抽出 x（回归用）。

        注意：分桶可能从 literal 落到 api —— 检索用的是 all_tokens()，分桶只影响
        LOCATE 的 anchor 行展示，属已知的小偏差。
        """
        anchors = extract_anchors("在 `` `strict` `` 模式下失败")
        self.assertIn("strict", anchors.all_tokens())

    def test_lexical_rank_orders_non_test_sources_first_on_tie(self) -> None:
        anchors = extract_anchors("`dispatch` payload is wrong")
        ranked = lexical_rank(self.FILES, anchors.all_tokens())
        self.assertTrue(ranked)
        self.assertEqual(ranked[0].path, "alpha/routing.py")

    def test_exclude_tests_removes_test_candidates_only(self) -> None:
        anchors = extract_anchors("`handle` and `dispatch`")
        ranked = lexical_rank(self.FILES, anchors.all_tokens())
        filtered = exclude_tests(ranked)
        self.assertTrue(filtered)
        self.assertTrue(all(not is_test_path(c.path) for c in filtered))

    def test_anchor_direct_only_matches_path_tokens(self) -> None:
        anchors = extract_anchors("see `alpha/util.py` for details")
        direct = anchor_direct(self.FILES, anchors.paths)
        self.assertEqual([c.path for c in direct], ["alpha/util.py"])
        self.assertEqual(direct[0].origin, "path_token")

    def test_hit_at_k_versus_true_recall(self) -> None:
        anchors = extract_anchors("`dispatch` and `helper`")
        ranked = lexical_rank(self.FILES, anchors.all_tokens())
        gold = ["alpha/routing.py", "alpha/util.py"]
        self.assertTrue(recall_at_k(ranked, gold, 5))
        self.assertLessEqual(true_recall_at_k(ranked, gold, 1), 1.0)
        self.assertEqual(true_recall_at_k(ranked, [], 5), 0.0)
        # 单文件 gold 且命中 → recall=1.0
        self.assertEqual(true_recall_at_k(ranked, ["alpha/routing.py"], 5), 1.0)

    def test_region_hit_requires_overlap(self) -> None:
        gold = [{"file": "alpha/routing.py", "start_line": 4, "lines": 2}]
        self.assertTrue(region_hit({"path": "alpha/routing.py", "start": 5, "end": 6}, gold))
        self.assertFalse(region_hit({"path": "alpha/routing.py", "start": 1, "end": 3}, gold))
        self.assertFalse(region_hit({"path": "alpha/util.py", "start": 4, "end": 5}, gold))
        self.assertFalse(region_hit(None, gold))

    def test_narrow_output_truncates_and_flags(self) -> None:
        rows = ["row %d %s" % (i, "x" * 40) for i in range(200)]
        out = narrow_output("grep -rc:", rows)
        self.assertTrue(out["truncated"])
        self.assertIn("row_budget", out["truncation_flags"])
        self.assertLessEqual(out["chars"], HARD_OUTPUT_CHARS)
        self.assertLessEqual(out["chars"], TARGET_OUTPUT_CHARS + 100)
        short = narrow_output("grep -rc:", rows[:2])
        self.assertFalse(short["truncated"])

    def test_narrow_output_rejects_inverted_budget(self) -> None:
        with self.assertRaises(PolicyViolation):
            narrow_output("h", ["x"], target_chars=6000, hard_chars=5000)

    def test_candidate_table_is_compact(self) -> None:
        anchors = extract_anchors("`dispatch`")
        ranked = lexical_rank(self.FILES, anchors.all_tokens())
        rows = candidate_table(ranked, limit=2)
        self.assertLessEqual(len(rows), 2)
        self.assertTrue(all("conf=" in row for row in rows))


class LocateTests(unittest.TestCase):
    def _state(self) -> LocateState:
        return LocateState.create(
            task_id="syn-1",
            base_commit="0" * 40,
            tree_sha="a" * 64,
            anchors={"literal": ["dispatch"], "behaviour": [], "api": []},
            queries=["dispatch"],
        )

    def _candidate(self, **overrides) -> dict:
        base = {
            "path": "alpha/routing.py",
            "symbol": "alpha.routing.dispatch",
            "start": 4,
            "end": 5,
            "source": "lexical",
            "observation_ref": "obs/7",
            "evidence": "literal",
            "confidence": "high",
        }
        base.update(overrides)
        return base

    def test_valid_state_and_block_contract(self) -> None:
        state = self._state()
        state.add_candidate(self._candidate())
        state.payload["chosen"] = self._candidate()
        state.validate(path_exists=lambda path: path == "alpha/routing.py")
        block = state.to_block()
        state.validate_block(block)
        self.assertLessEqual(len(block.split()), 250)
        parsed = parse_block(block)
        self.assertEqual(parsed["candidates"][0]["path"], "alpha/routing.py")
        self.assertIsNotNone(parsed["chosen"])

    def test_missing_path_is_rejected(self) -> None:
        state = self._state()
        state.add_candidate(self._candidate(path="alpha/ghost.py"))
        with self.assertRaises(PolicyViolation):
            state.validate(path_exists=lambda path: path == "alpha/routing.py")

    def test_evidence_none_is_rejected_for_retrieval_sources(self) -> None:
        state = self._state()
        state.add_candidate(self._candidate(evidence="none"))
        with self.assertRaises(PolicyViolation):
            state.validate()

    def test_missing_observation_ref_is_rejected(self) -> None:
        state = self._state()
        state.add_candidate(self._candidate(observation_ref=""))
        with self.assertRaises(PolicyViolation):
            state.validate()

    def test_absolute_and_escaping_paths_are_rejected(self) -> None:
        state = self._state()
        state.add_candidate(self._candidate(path="/etc/passwd"))
        with self.assertRaises(PolicyViolation):
            state.validate()
        state2 = self._state()
        state2.add_candidate(self._candidate(path="../outside.py"))
        with self.assertRaises(PolicyViolation):
            state2.validate()

    def test_missing_field_is_rejected(self) -> None:
        state = self._state()
        candidate = self._candidate()
        candidate.pop("confidence")
        state.add_candidate(candidate)
        with self.assertRaises(MissingInput):
            state.validate()

    def test_chosen_must_come_from_candidates(self) -> None:
        state = self._state()
        state.add_candidate(self._candidate())
        state.payload["chosen"] = self._candidate(path="alpha/util.py")
        with self.assertRaises(PolicyViolation):
            state.validate()

    def test_candidate_count_is_capped_at_five(self) -> None:
        state = self._state()
        for index in range(7):
            state.add_candidate(self._candidate(symbol="s%d" % index))
        self.assertEqual(len(state.payload["candidates"]), 5)

    def test_block_rejects_regurgitation_and_overlong(self) -> None:
        state = self._state()
        state.validate_block("LOCATE v1\nchosen: none")
        with self.assertRaises(PolicyViolation):
            state.validate_block(" ".join("w" for _ in range(300)))
        with self.assertRaises(PolicyViolation):
            state.validate_block(
                "a.py:1: line one\nb.py:2: line two\nc.py:3: line three"
            )

    def test_future_or_gold_information_is_rejected(self) -> None:
        state = self._state()
        state.payload["gold_files"] = ["alpha/routing.py"]
        with self.assertRaises(PolicyViolation):
            state.assert_no_future_information()
        with self.assertRaises(PolicyViolation):
            state.to_training_target()

    def test_bad_next_action_and_lines_are_rejected(self) -> None:
        state = self._state()
        state.payload["next_action"] = "teleport"
        with self.assertRaises(PolicyViolation):
            state.validate()
        state2 = self._state()
        state2.add_candidate(self._candidate(start=9, end=3))
        with self.assertRaises(PolicyViolation):
            state2.validate()


class ProtocolTests(unittest.TestCase):
    def test_search_budget_is_enforced(self) -> None:
        session = SearchSession(task_id="syn-1", budget=SearchBudget(total_tool_calls=100))
        for _ in range(SEARCH_CALL_BUDGET):
            session.note_call("run_command", output_chars=100)
        with self.assertRaises(PolicyViolation):
            session.note_call("run_command", output_chars=100)

    def test_non_search_calls_do_not_consume_search_budget(self) -> None:
        session = SearchSession(task_id="syn-1")
        session.note_call("submit_patch", bills=False)
        session.note_call("edit_file", output_chars=10)
        self.assertEqual(session.search_calls_used, 0)
        self.assertEqual(session.tool_calls_used, 1)
        self.assertEqual(session.snapshot()["search_share"], 0.0)

    def test_output_flags_record_hard_truncation(self) -> None:
        session = SearchSession(task_id="syn-1")
        session.note_call("run_command", output_chars=HARD_OUTPUT_CHARS + 1)
        session.note_call("run_command", output_chars=TARGET_OUTPUT_CHARS + 1)
        self.assertTrue(any(flag.startswith("hard_truncation_") for flag in session.output_flags))
        self.assertTrue(any(flag.startswith("over_target_") for flag in session.output_flags))

    def test_escalation_requires_a_trigger_and_is_capped(self) -> None:
        session = SearchSession(task_id="syn-1")
        with self.assertRaises(PolicyViolation):
            session.escalate({})
        for _ in range(8):
            session.note_call("run_command", output_chars=10)
        self.assertIn("E-d", session.signals({}))
        self.assertEqual(session.escalate({}), "L1")
        self.assertEqual(session.escalate({}), "L2")
        with self.assertRaises(PolicyViolation):
            session.escalate({})

    def test_stop_conditions(self) -> None:
        session = SearchSession(task_id="syn-1", budget=SearchBudget(total_tool_calls=100))
        self.assertFalse(session.must_edit())
        session.elapsed_ratio = 0.56
        self.assertTrue(session.must_edit())
        session2 = SearchSession(task_id="syn-1", budget=SearchBudget(total_tool_calls=40))
        session2.note_call("run_command", output_chars=10)
        self.assertFalse(session2.must_edit())  # 剩余 tool_calls = 39 > 25
        session3 = SearchSession(task_id="syn-1", budget=SearchBudget(total_tool_calls=26))
        session3.note_call("run_command", output_chars=10)
        self.assertTrue(session3.must_edit())  # 剩余 tool_calls = 25 → 必须收敛

    def test_fail_closed_when_evidence_insufficient(self) -> None:
        session = SearchSession(task_id="syn-1", budget=SearchBudget(total_tool_calls=26))
        session.note_call("run_command", output_chars=10)
        triggers = session.signals({"rounds_without_src_hit": 2})
        self.assertIn("E-a", triggers)
        self.assertEqual(session.escalate({"rounds_without_src_hit": 2}), "L1")
        self.assertTrue(session.must_edit())  # 剩余 tool_calls = 25
        self.assertIsNone(session.chosen)
        self.assertIsNotNone(session.next_rung())  # 还剩一轮升级额度
        session.escalate({"rounds_without_src_hit": 2})
        self.assertIsNone(session.next_rung())
        self.assertEqual(session.next_action(), "fail")

    def test_fail_closed_never_forces_a_blind_edit(self) -> None:
        session = SearchSession(task_id="syn-1", budget=SearchBudget(total_tool_calls=26))
        session.note_call("run_command", output_chars=10)
        self.assertFalse(session.must_fail_closed())  # 还有升级额度
        session.escalate({"rounds_without_src_hit": 2})
        session.escalate({"rounds_without_src_hit": 2})
        self.assertIsNone(session.next_rung())
        self.assertTrue(session.must_fail_closed())
        self.assertEqual(session.next_action(), "fail")
        state = parse_block("LOCATE v1\nchosen: none")
        self.assertIsNone(state["chosen"])

    def test_s0_and_s1_share_one_skeleton(self) -> None:
        left, right = policies()["S0"], policies()["S1"]
        assert_same_skeleton(left, right)
        self.assertIsInstance(left, S0Policy)
        self.assertIsInstance(right, S1Policy)

    def test_s1_excludes_tests_in_ranking_but_s0_does_not(self) -> None:
        session = SearchSession(task_id="syn-1")
        session.note_candidate({"path": "alpha/tests/test_routing.py", "hits": 9, "distinct_queries": 3, "concentration": 1.0})
        session.note_candidate({"path": "alpha/routing.py", "hits": 2, "distinct_queries": 1, "concentration": 0.5})
        s0 = S0Policy().rank_candidates(session)
        s1 = S1Policy().rank_candidates(session)
        self.assertEqual(s0[0]["path"], "alpha/tests/test_routing.py")
        self.assertEqual([c["path"] for c in s1], ["alpha/routing.py"])

    def test_shell_arguments_are_quoted(self) -> None:
        dangerous = "x; rm -rf /workspace #"
        quoted = escape_shell_arg(dangerous)
        self.assertNotEqual(quoted, dangerous)
        self.assertIn("'", quoted)
        command = build_grep_command("a b'c", root="/workspace")
        self.assertIn("grep -rln", command)
        self.assertIn("| head -20", command)
        count = build_count_command("dispatch")
        self.assertIn("grep -rc", count)
        self.assertIn("head -10", count)

    def test_read_command_enforces_line_cap(self) -> None:
        self.assertIn("start_line=1", build_read_command("a.py", 1, 150))
        with self.assertRaises(PolicyViolation):
            build_read_command("a.py", 1, 200)

    def test_four_arms_are_single_writer(self) -> None:
        arms = four_arm_matrix()
        self.assertEqual([arm["arm"] for arm in arms], ["A", "B", "C", "D"])
        self.assertEqual({arm["prompt"] for arm in arms}, {"W0"})
        self.assertEqual({arm["adapter"] for arm in arms if arm["arm"] in ("C", "D")}, {"L1"})
        assert_single_writer(arms)
        with self.assertRaises(PolicyViolation):
            assert_single_writer([{"arm": "A", "forced_analyzer": True}])

    def test_policy_plan_drives_next_action(self) -> None:
        session = SearchSession(task_id="syn-1")
        self.assertEqual(S1Policy().plan(session, {})["channel"], "C2")
        self.assertEqual(S0Policy().plan(session, {})["channel"], "C1")
        for _ in range(8):
            session.note_call("run_command", output_chars=10)
        plan = S1Policy().plan(session, {})
        self.assertEqual(plan["action"], "escalate")
        self.assertEqual(plan["rung"], "L1")


class IndexTests(unittest.TestCase):
    FILES = {"alpha/routing.py": SOURCE}

    def test_index_builds_and_binds_tree_hash(self) -> None:
        tree = tree_sha_for_files(self.FILES)
        index = MicroIndex.build("alpha", tree, self.FILES)
        self.assertGreater(len(index.entries), 5)
        self.assertTrue(index.is_valid_for(tree))
        self.assertIn("Client.fetch", index.render())
        with self.assertRaises(IntegrityError):
            index.assert_valid_for("f" * 64)

    def test_registry_drops_stale_index_after_source_edit(self) -> None:
        registry = IndexRegistry()
        tree = tree_sha_for_files(self.FILES)
        index = registry.get_or_build("alpha", tree, self.FILES)
        self.assertIs(registry.get("alpha", tree), index)
        edited = {"alpha/routing.py": SOURCE + "\ndef extra():\n    return 2\n"}
        self.assertIsNone(registry.get("alpha", tree_sha_for_files(edited)))

    def test_index_target_must_not_be_inside_workspace(self) -> None:
        assert_safe_index_target("/tmp/idx.txt")
        with self.assertRaises(PolicyViolation):
            assert_safe_index_target("/workspace/idx.txt")
        with self.assertRaises(PolicyViolation):
            assert_safe_index_target("/workspace/sub/idx.txt")

    def test_build_index_command_writes_to_tmp(self) -> None:
        command = build_index_command("/workspace", "/tmp/idx.txt")
        self.assertIn("ast.parse", command)
        self.assertIn("/tmp/idx.txt", command)
        with self.assertRaises(PolicyViolation):
            build_index_command("/workspace", "/workspace/idx.txt")

    def test_lookup_prefers_exact_qualname(self) -> None:
        tree = tree_sha_for_files(self.FILES)
        index = MicroIndex.build("alpha", tree, self.FILES)
        exact = index.lookup("Client.fetch")
        self.assertEqual(len(exact), 1)
        self.assertEqual(exact[0].path, "alpha/routing.py")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
