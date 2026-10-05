#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""KAGGLE-16：机械核对 B0 与 A 的官方编译对象树，证明「除两处提示外无行为/成员变更」。

输入：两份 ``E1-b0-compile-report.json``（由官方 ``adk_submission.compile_submission``
跑出来，脚本 ``compile_variant.py`` / ``e1_compile_b0.py`` 只负责喂参数与 dump）。

判定项（全部必须为真，否则退出码 3）：
1. 两份报告 status 均为 compiled；
2. agent_count 相同；
3. 对象树里除 root agent 的 instruction 之外，**逐字段完全一致**
   （工具名与顺序、子 agent instruction 哈希、generate_content_config、
   model、description、嵌套结构）；
4. 差异恰好只有 root agent 的 instruction.chars / utf8_bytes / sha256 / head；
5. ``code_analyzer`` 仍以 AgentTool 形式挂在 root 工具表里（条件调用，不是删除）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def strip_instruction(node: dict) -> dict:
    """把 instruction 换成占位符，用于「除 instruction 外全等」比较。"""
    out = {k: v for k, v in node.items() if k != "instruction"}
    out["instruction"] = "<stripped>"
    out["sub_agents"] = [strip_instruction(s) for s in node.get("sub_agents", [])]
    out["tools"] = list(node.get("tools", []))
    return out


def collect_instructions(node: dict, acc: dict) -> dict:
    inst = node.get("instruction")
    if isinstance(inst, dict):
        acc[node["name"]] = {
            "chars": inst.get("chars"),
            "utf8_bytes": inst.get("utf8_bytes"),
            "sha256": inst.get("sha256"),
        }
    for sub in node.get("sub_agents", []):
        collect_instructions(sub, acc)
    return acc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b0-report", required=True)
    ap.add_argument("--a-report", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    b0, a = load(Path(args.b0_report)), load(Path(args.a_report))
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    check("both_status_compiled",
          b0.get("status") == "compiled" and a.get("status") == "compiled",
          f"B0={b0.get('status')} A={a.get('status')}")
    check("agent_count_equal",
          b0.get("agent_count") == a.get("agent_count"),
          f"B0={b0.get('agent_count')} A={a.get('agent_count')}")
    check("effective_sampling_equal",
          b0.get("effective_sampling") == a.get("effective_sampling"),
          "")
    check("limits_equal", b0.get("limits") == a.get("limits"),
          f"source B0={b0.get('limits_source')} A={a.get('limits_source')}")
    check("tool_registry_equal", b0.get("tool_registry_names") == a.get("tool_registry_names"),
          f"mode B0={b0.get('tool_registry_mode')} A={a.get('tool_registry_mode')}")
    check("declared_models_equal", b0.get("declared_models") == a.get("declared_models"), "")
    check("adapters_discovered_equal",
          [x["name"] for x in b0.get("adapters_discovered", [])]
          == [x["name"] for x in a.get("adapters_discovered", [])],
          "U 未引入：adapters 仍为 2 个")

    tb, ta = b0["object_tree"], a["object_tree"]
    check("tree_except_instruction_equal",
          strip_instruction(tb) == strip_instruction(ta),
          "对象树除 instruction 外逐字段一致")

    ib, ia = collect_instructions(tb, {}), collect_instructions(ta, {})
    check("instruction_node_names_equal", sorted(ib) == sorted(ia), f"{sorted(ib)}")
    diff_nodes = sorted(n for n in ib if ib[n] != ia[n])
    check("only_root_instruction_differs", diff_nodes == ["root_coder_agent"], f"{diff_nodes}")
    check("sub_agent_instruction_bytes_identical",
          ib.get("code_analyzer", {}).get("sha256") == ia.get("code_analyzer", {}).get("sha256"),
          f"code_analyzer sha256={ia.get('code_analyzer', {}).get('sha256')}")

    root_tools_a = [t.get("name") for t in ta.get("tools", [])]
    check("code_analyzer_still_declared",
          "code_analyzer" in root_tools_a,
          "条件调用（保留 AgentTool），非删除")
    wrapped = [t for t in ta.get("tools", []) if t.get("wrapped_agent_name") == "code_analyzer"]
    check("code_analyzer_wrapped_as_agenttool", len(wrapped) == 1,
          json.dumps(wrapped, ensure_ascii=False) if wrapped else "缺失")

    ok = all(c["pass"] for c in checks)
    report = {
        "task": "KAGGLE-16 / V2 A —— B0/A 官方编译对象树对照",
        "b0_report": str(args.b0_report),
        "a_report": str(args.a_report),
        "b0_instruction": ib,
        "a_instruction": ia,
        "root_instruction_chars_delta": (ia.get("root_coder_agent", {}).get("chars", 0)
                                         - ib.get("root_coder_agent", {}).get("chars", 0)),
        "checks": checks,
        "all_pass": ok,
    }
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    for c in checks:
        print(f"[{'PASS' if c['pass'] else 'FAIL'}] {c['check']}  {c['detail']}")
    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
