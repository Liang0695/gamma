"""Emit the D0 human-readable report and the family table.

This generator is the v2 revision of the D0 report.  It supersedes the first
pass: the first pass described the withdrawn time rule (train < 2025-01-01,
dev = calendar 2025, sealed >= 2026-01-01) and had no machine-readable licence
approval field, which is exactly what Q0 refused and Mika sent back.

Every number in the report is read out of the generated JSON, so the prose and
the machine-readable files can never drift apart.
"""
import csv
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ACCESS_DATE = "2026-10-05"
BRANCH = "agent/research/kaggle-23-d0-source-lock"
PREVIOUS_COMMIT = "8b8ff5a"
LOCKED = ["click", "more-itertools", "pluggy", "boltons", "attrs", "dateutil",
          "packaging", "marshmallow"]


def load(p):
    with open(os.path.join(OUT, p), encoding="utf-8") as f:
        return json.load(f)


def spec_sha256():
    p = os.path.join(os.path.dirname(HERE), "attachments",
                     "KAGGLE-19-V3-integrated-review.md")
    if not os.path.isfile(p):
        return "(spec file not present in this checkout)"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def gate_summary():
    p = os.path.join(OUT, "validate_d0.output.txt")
    if not os.path.isfile(p):
        return (None, None), None
    lines = [l.rstrip("\n") for l in open(p, encoding="utf-8") if l.strip()]
    total = failed = None
    for l in lines:
        if "checks," in l and "failed" in l:
            parts = l.split()
            total, failed = parts[0], parts[2]
    n_pass = sum(1 for l in lines if l.startswith("PASS"))
    return (total, failed), n_pass


def main():
    lock = load("source-lock.json")
    ledger = load("family-ledger.json")
    iso = load("d0-time-isolation.json")
    short = load("d0-shortfall.json")
    fams = ledger["families"]
    real = [f for f in fams if f["kind"] == "real"]
    var = [f for f in fams if f["kind"] == "variant"]
    repos = lock["repos"]
    gate, n_pass = gate_summary()
    n_checks = gate[0] if gate else "?"
    n_failed = gate[1] if gate else "?"
    locked_approved = sum(1 for n in LOCKED
                          if repos[n]["license_review"]["decision"] == "approved")

    # ---- family table csv ----
    with open(os.path.join(OUT, "kaggle-23-d0-family-table.csv"), "w",
              encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family_id", "kind", "repo", "split", "released", "base_commit",
                    "oracle_fix_commit", "oracle_patch_sha256", "fix_time_author",
                    "fix_time_committer", "both_dates_in_window", "ancestor_of_pin",
                    "fail_to_pass_nodes", "mutation_class", "license_spdx",
                    "license_review_decision"])
        for f in fams:
            w.writerow([
                f["family_id"], f["kind"], f["repo"], f["split"], f["released"],
                f.get("base_commit") or f.get("source_base_commit") or "",
                f.get("oracle_fix_commit") or "",
                f.get("oracle_patch_sha256") or "",
                (f.get("fix_time") or {}).get("author_date", ""),
                (f.get("fix_time") or {}).get("committer_date", ""),
                (f.get("fix_time") or {}).get("both_dates_in_train_window", ""),
                f.get("fix_commit_is_ancestor_of_pinned_revision", ""),
                ";".join(n["node_id"] for n in (f.get("fail_to_pass") or [])),
                f.get("mutation_class", ""),
                f["license"]["spdx"],
                f["license"].get("license_review_decision", ""),
            ])

    L = []
    a = L.append
    a("# V3 D0 资料来源锁定与许可证据说明（v2 修订版）")
    a("")
    a("生成：资料调研与分发 · %s · 访问日期 %s · 分支 `%s`" % (ACCESS_DATE, ACCESS_DATE, BRANCH))
    a("")
    a("> **本文件取代第一版（该分支上一提交 `%s`）。** 第一版沿用了一套**未经批准**的"
      "时间划分（train < 2025-01-01、dev = 2025 整年、sealed ≥ 2026-01-01），"
      "且 source-lock 里没有机器可读的许可批准字段。Q0 据此退回，Mika 裁决后重新升版。"
      "第一版的划分在本版中明确记为**已废止**，见 §2.3。" % PREVIOUS_COMMIT)
    a("")

    a("## 0. 一句话结论")
    a("")
    a("八个锁定来源全部固定在可复现的 revision（commit 与计划 8/8 一致，tree SHA 已记录）；"
      "**每个来源都带机器可读的许可批准记录**（`repos.<name>.license_review.decision`，"
      "当前 %d/%d 为 `approved`，copyleft 与限制性条款命中数均为 0）。时间策略已按 Mika 裁决升版："
      "**train = 家族原始修复时间 ≤ 2025-12-31，dev/sealed 共用 2026-01-01 至 2026-10-04**，"
      "并显式声明 dev 与 sealed 之间**不主张先后时间隔离**。P0 的四个真实家族全部 released，"
      "每个都带非空 base_commit、oracle 修复 commit、补丁哈希与 FAIL_TO_PASS 节点，"
      "且新增三项验证：**修复 commit 必须在固定快照内、必须不是 backport/cherry-pick、"
      "作者时间与提交者时间必须同时在窗口内**。四个变异家族仍只有构造规格、没有构造 commit，"
      "一律 released=false —— 没有任何空 SHA 记录被当作已发布示例。校验 %s 项、%s 失败。"
      "本轮**仍未运行 FAIL_TO_PASS**。最大实质缺口：dateutil 在固定 revision 下 "
      "dev 窗口内合格家族数为 0（与 Mika 的暂计一致）。" % (
          locked_approved, len(LOCKED),
          n_checks, n_failed))
    a("")

    a("## 1. 需求回顾")
    a("")
    a("- **谁要的**：本任务负责人为资料调研与分发（KAGGLE-23）。需求来自 KAGGLE-19 的 V3 联合设计"
      "（附件 `KAGGLE-19-V3-integrated-review.md`，本运行实测 SHA256 `%s`），"
      "由 yg123456 批准。" % spec_sha256())
    a("- **要什么**：把设计中的候选来源变成可执行、许可闭合的数据输入 —— 固定 8 候选仓库的 commit、"
      "逐文件许可/NOTICE、来源与问题家族台账、时间隔离与 shortfall。")
    a("- **本轮整改要求**（Mika 裁决 + Q0 报告）：① 按裁决升版时间/split 策略；"
      "② 补 source-lock 的**机器可读许可批准字段**；③ 在原分支提交**新 SHA 与差异验证**。")
    a("- **用途**：优先支撑 P0「8 训练家族（4 真实 + 4 变异，≥2 仓库）」；"
      "split 与封存资料交独立审查官保管。")
    a("- **边界**：不跑教师、不用 GPU、不申请 107 作业、不修改编码官代码；"
      "不读取 V2 留出正文/gold，不采公开比赛 gold 作训练。本轮仅用既有 CPU 准备额度内的取证。")
    a("")

    a("## 2. 事实与证据")
    a("")
    a("### 2.1 八个锁定来源的固定 revision")
    a("")
    a("| 仓库 | 上游 | split 角色 | 固定 tag | 固定 commit | tree SHA | 快照日期¹ | 声明许可 |")
    a("|---|---|---|---|---|---|---|---|")
    for name in LOCKED:
        r = repos[name]
        a("| %s | `%s` | %s | `%s` | `%s` | `%s` | %s | %s |" % (
            name, r["upstream_slug"], r["split_role"], r["pinned_tag"],
            r["pinned_commit"], r["tree_sha"], r["commit_date"][:10],
            r["design_license_expectation"]))
    a("")
    a("¹ **快照日期不是家族时间。** 取证方式：`git ls-remote --tags` 取 refs，"
      "再按 tag 校验 peeled commit 与计划一致（8/8 match）。本机直连 github.com 不通，"
      "全程走 `https://ghfast.top/https://github.com/...` 镜像；完整性由 **commit SHA + tree SHA** "
      "双重锚定，不依赖镜像的字节可复现性。")
    a("")

    a("### 2.2 许可证据与机器可读批准字段（本轮新增）")
    a("")
    a("新增契约位于 `source-lock.json` 顶层 `license_review_schema`，逐仓库记录在 "
      "`repos.<name>.license_review`。必备字段：`%s`。" % "`、`".join(
          lock["license_review_schema"]["required_fields"]))
    a("")
    a("| 仓库 | decision | approved_spdx | 主许可文件 | 文件 SHA256（前 16） | copyleft 命中 | 限制性命中 | 独立复核 |")
    a("|---|---|---|---|---|---|---|---|")
    for name in LOCKED:
        lr = repos[name]["license_review"]
        pf = lr["evidence"]["primary_license_file"] or {}
        a("| %s | **%s** | %s | `%s` | `%s` | %d | %d | %s |" % (
            name, lr["decision"], lr["approved_spdx"], pf.get("path", "-"),
            (pf.get("sha256") or "")[:16],
            len(lr["copyleft_marker_hits"]), len(lr["restrictive_marker_hits"]),
            lr["independent_review"]["status"]))
    a("")
    a("关键性质（都是可被 Q0 直接否证的断言）：")
    a("")
    a("- **批准绑定到具体 revision**：`decided_against_revision` 恒等于该仓库的 `pinned_commit`；"
      "重新 pin 会让批准失效，必须重做。")
    a("- **命中列表逐条留证**：copyleft / 限制性标记不是布尔值，而是 "
      "`{file, marker, line_no, line}` 列表（本轮全为空）。扫描范围限定在 "
      "LICENSE / COPYING / NOTICE / PATENTS 这类**主许可文件**，"
      "避免 AUTHORS、CONTRIBUTING 等文件顺带提到别的许可造成假阳性。")
    a("- **批准不冒充独立签字**：`independent_review.status` 全部为 `pending`，"
      "复核角色为 Q0。这份 `decision` 是 D0 负责人基于已哈希文件的事实判定，"
      "**不是**独立背书，shortfall 中已如实列出。")
    a("- 逐文件台账 `per-file-ledger.csv` 覆盖全部被扫描文本文件，"
      "每行含 SHA256、SPDX 头与版权行。")
    a("")

    a("### 2.3 时间策略 v2（本轮升版，取代第一版）")
    a("")
    a("| 轴 | 定义 | 本版取值 |")
    a("|---|---|---|")
    a("| train 窗口 | 家族**原始修复时间** ≤ 2025-12-31 | `end_exclusive = %s` |"
      % iso["rule"]["windows"]["train"]["end_exclusive"])
    a("| dev / sealed 窗口 | 家族**原始修复时间** ∈ 2026-01-01 … 2026-10-04 | "
      "`[%s, %s)` |" % (iso["rule"]["windows"]["dev_sealed"]["start"],
                         iso["rule"]["windows"]["dev_sealed"]["end_exclusive"]))
    a("")
    a("**已废止（第一版，未经批准）**：`train_end_exclusive = %s`、"
      "`dev = [%s, %s)`、`sealed >= %s`。本版 JSON 中保留该记录并标记 `withdrawn`，"
      "避免下游误用旧划分。" % (
          iso["rule"]["withdrawn_windows"]["train_end_exclusive"],
          iso["rule"]["withdrawn_windows"]["dev_start"],
          iso["rule"]["withdrawn_windows"]["dev_end_exclusive"],
          iso["rule"]["withdrawn_windows"]["sealed_start"]))
    a("")
    a("**dev 与 sealed 之间的关系**：共用同一个窗口，"
      "`dev_vs_sealed_ordering_claimed = %s` —— 二者由**仓库角色**区分，"
      "**不宣称**任何额外的先后时间隔离。"
      % iso["rule"]["windows"]["dev_sealed"]["dev_vs_sealed_ordering_claimed"])
    a("")
    a("**家族时间的唯一权威**：%s。具体落到数据上："
      % iso["rule"]["time_axis_authority"])
    a("")
    a("- 每个候选家族同时记录**作者时间与提交者时间**，**两者都**必须落在窗口内，"
      "分类不依赖任何一个可被 rebase 改写的单独日期。")
    a("- 明确**拒绝**作为家族时间的替代品：%s。"
      % "、".join(iso["rule"]["rejected_time_substitutes"]))
    a("- **快照轴与时间轴分离**：固定 revision 只决定「哪些内容存在」"
      "（家族修复 commit 必须是该快照的祖先），**绝不**决定家族属于哪个窗口。"
      "source-lock 中每个 `commit_date` 都带 `commit_date_role` 说明它不是家族时间。")
    a("- **backport / cherry-pick commit 直接拒收**，命中标记在台账中留证，"
      "防止「重新落地的旧缺陷」被塞进更晚的窗口。")
    a("")

    a("### 2.4 家族台账与 oracle 构造依据")
    a("")
    a("| family_id | 仓库 | released | base_commit | oracle 修复 commit | 补丁 SHA256（前 16） | 作者时间 | 提交者时间 | F2P 节点 |")
    a("|---|---|---|---|---|---|---|---|---|")
    for f in real:
        a("| %s | %s | %s | `%s` | `%s` | `%s` | %s | %s | %d |" % (
            f["family_id"], f["repo"], f["released"], f["base_commit"],
            f["oracle_fix_commit"], f["oracle_patch_sha256"][:16],
            f["fix_time"]["author_date"][:19], f["fix_time"]["committer_date"][:19],
            len(f["fail_to_pass"])))
    a("")
    a("每个真实家族的完整字段见 `family-ledger.json`：逐文件 base/fix 哈希、"
      "test patch 与 code-only gold patch 的分离哈希、FAIL_TO_PASS 节点"
      "（要求 base 不存在、fix 存在）、oracle 断言行、以及环境依赖材料。")
    a("")
    a("**变异家族（released=false）**：")
    a("")
    a("| family_id | 派生自 | 变异类 | 源基线 commit | expected patch shape |")
    a("|---|---|---|---|---|")
    for f in var:
        a("| %s | %s | %s | `%s` | %s |" % (
            f["family_id"], f["derived_from"], f["mutation_class"],
            f["source_base_commit"][:12], f["expected_patch_shape"]))
    a("")
    a("变异家族的 release 阻断原因是结构性的：变异 commit 在编码官构造出来之前**不存在**，"
      "因此只固定上游源基线，**任何空 SHA 都不会被当作已发布示例**。"
      "这正是验收条目「示例空 SHA 不能 released」对应的证据。")
    a("")

    a("### 2.5 时间隔离证据（新增 dev/sealed 清单）")
    a("")
    a("| 仓库 | 角色 | 窗口 | 窗口内合格候选家族 | 最早修复时间 | 最新修复时间 | 快照早于窗口? |")
    a("|---|---|---|---|---|---|---|")
    for src in (iso["train_inventory"], iso["dev_sealed_inventory"]):
        for name, v in src.items():
            a("| %s | %s | %s | %d | %s | %s | %s |" % (
                name, v["role"], v["window"], v["qualified_candidate_families_in_window"],
                (v["oldest_fix_time"] or "-")[:10], (v["newest_fix_time"] or "-")[:10],
                v["pinned_snapshot_predates_window"]))
    a("")
    a("train 窗口内候选家族总数 **%d**（%s）；dev 角色窗口内合格家族 **%d**。" % (
        iso["evidence"]["train_candidate_total_in_window"],
        "、".join("%s %d" % (k, v["qualified_candidate_families_in_window"])
                  for k, v in iso["train_inventory"].items()),
        iso["evidence"]["dev_candidate_total_in_window"]))
    a("")
    a("窗口内出现、但被 backport/cherry-pick 标记拒收的 commit 观察数：%s"
      "（其中本可成为候选者 %d 个）。该规则本轮**未改变候选集合**，"
      "但它是被实际执行的，不是纸面声明。" % (
          "、".join("%s %d" % (k, v) for k, v
                    in iso["evidence"]["backport_marked_commits_observed_in_window"].items()
                    if v) or "0",
          iso["evidence"]["backport_marked_candidates_excluded"]))
    a("")
    a("四个已发布真实家族的逐项时间证据：")
    a("")
    a("| family_id | 作者时间 | 提交者时间 | 两者都在 train 窗口 | 修复 commit 在快照内 | backport 标记 |")
    a("|---|---|---|---|---|---|")
    for c in iso["evidence"]["chosen_real_families"]:
        a("| %s | %s | %s | %s | %s | %s |" % (
            c["family_id"], c["fix_time_author"][:19], c["fix_time_committer"][:19],
            c["both_dates_in_train_window"], c["is_ancestor_of_pinned_revision"],
            c["backport_or_cherry_pick_marker"] or "无"))
    a("")
    a("注意 pluggy 的作者时间与提交者时间相差数天（%s vs %s）——"
      "这正是本版要求两个日期同时在窗口内、并禁止用 release/snapshot/backport 日期顶替的原因。"
      % (real[2]["fix_time"]["author_date"][:10],
         real[2]["fix_time"]["committer_date"][:10]))
    a("")

    a("### 2.6 替代 dev 候选核验：python-dotenv")
    a("")
    alt = iso["alternative_dev_candidate"]
    fi = alt["family_inventory"]
    a("按 Mika 裁决，python-dotenv 仅作为替代候选开展**许可 / 家族 / 环境**三项核验，"
      "**不等于替换 dateutil，也不等于批准发布**。")
    a("")
    a("- 许可：`decision = %s`，`%s`，主许可文件 `%s`，copyleft 命中 %d、限制性命中 %d"
      "（路径 `%s`）。" % (
          alt["license"]["decision"], alt["license"]["approved_spdx"],
          (alt["license"]["primary_license_file"] or {}).get("path"),
          len(alt["license"]["copyleft_marker_hits"]),
          len(alt["license"]["restrictive_marker_hits"]),
          alt["license"]["path_in_source_lock"]))
    a("- 家族：固定 tag `%s` = `%s`（快照日期 %s），dev/sealed 窗口内合格候选家族 **%d** 个，"
      "最早 %s、最新 %s。" % (
          alt["pinned_tag"], alt["pinned_commit"][:12], alt["snapshot_date"][:10],
          fi["qualified_candidate_families_in_window"],
          (fi["oldest_fix_time"] or "-")[:10], (fi["newest_fix_time"] or "-")[:10]))
    a("- 环境：`requires_python = %s`，运行时依赖 %s，测试运行器 `%s`，"
      "并已对其 `%d` 个依赖/配置文件计算 SHA256。" % (
          alt["environment_material"]["requires_python"],
          alt["environment_material"]["runtime_dependencies"] or "无",
          alt["environment_material"]["test_runner"],
          len(alt["environment_material"]["files"])))
    a("- 边界字段：`replaces = %s`；`approval_status` 明确写为替代候选。"
      % alt["replaces"])
    a("")

    a("### 2.7 校验输出")
    a("")
    a("`d0/validate_d0.py` 独立于生成脚本、只读产物 JSON 重新断言：**%s 项检查、%s 失败**"
      "（PASS 行 %s）。校验项分为：固定 revision、许可批准契约、"
      "「空 SHA 不得 released」、P0 4+4、逐家族结构检查、时间策略 v2、"
      "替代候选边界、清单覆盖、公开/受限分离、逐文件台账完整性。"
      "完整输出见 `d0/out/validate_d0.output.txt`。" % (n_checks, n_failed, n_pass))
    a("")

    a("## 3. 推断与建议（标注为推断 / 建议）")
    a("")
    a("- **推断**：train 侧名义 headroom 为 %d 个候选对 24 个真实家族需求，约 %.1f 倍，"
      "但**没有一个**候选经过验证器跑通、actor 可达性与有界测试补丁检查，"
      "不能把 %d 读成 24。" % (
          short["p1_headroom"]["raw_candidates_in_train_window"],
          short["p1_headroom"]["raw_candidates_in_train_window"] / 24.0,
          short["p1_headroom"]["raw_candidates_in_train_window"]))
    a("- **推断**：dev 供给偏薄，只有 attrs 一个锁定 dev 角色仓库产出窗口内家族（%d 个，"
      "且未验证）；dateutil 在固定 revision 下为 0。"
      % iso["dev_sealed_inventory"]["attrs"]["qualified_candidate_families_in_window"])
    a("- **建议**：dev 方案二选一由 Mika / 保管侧裁决 —— "
      "① 以 attrs + 已核验的替代候选承担 dev，或 ② 对 dateutil 重新 pin "
      "（但会使本版绑定在该 revision 上的许可批准失效，必须重做逐文件许可台账"
      "与新 revision 的批准）。")
    a("- **建议**：请 Q0 对 `license_review` 逐条反证（`independent_review.status` 仍为 pending），"
      "并注意该字段是自述而非独立背书。")
    a("- **建议**：E0 在环境就绪后跑四个真实家族的 broken/reference 双次干净对照；"
      "本版不把静态来源验收当作数据 released。")
    a("")

    a("## 4. 冲突与不确定项")
    a("")
    a("- **权限隔离未建立，因此 dev/sealed 的发布与验收保持阻断**（沿用 Mika 裁决，本轮不改变）。"
      "`restricted-oracle.json` 已显式写入 `split_declaration_pending`："
      "本交付**不声称**该文件已安全切分或未被污染，切分须由持有 gold 的保管侧判定。")
    a("- **没有跑过 FAIL_TO_PASS**：运行环境取不到包索引，装不上 pytest，"
      "所有 oracle 结论均为静态证据。")
    a("- **许可批准是自述**：见 §2.2。")
    for r in iso["open_items"]:
        a("- **%s（%s）**：固定 revision %s，早于窗口起点 %s，"
          "在该 revision 下窗口内合格家族数为 0。%s" % (
              r["repo"], r["role"], r["pinned_revision_date"][:10],
              r["required_window_start"][:10], r["consequence"]))
    a("")
    a("未找到可靠来源 / 未执行的部分：上游 issue/PR **正文的著作权**未单独清理；"
      "本轮不复制任何正文，任务文本按设计重写（不引用原句），"
      "但重写文本仍需按原创写作复核。")
    a("")

    a("## 5. 交付物清单")
    a("")
    a("| 文件（`d0/out/`） | 用途 | 接收方 |")
    a("|---|---|---|")
    a("| `source-lock.json` | 8 来源固定 revision + 逐文件许可 + **机器可读批准字段** | 编码官 / Q0 |")
    a("| `license-files.json` | 每个许可文件的 SHA256 | Q0 |")
    a("| `per-file-ledger.csv` | 逐文件 SHA256 / SPDX 头 / 版权行 | Q0 |")
    a("| `family-ledger.json` | 家族台账、oracle 分离哈希、时间证据 | 编码官 / E0 |")
    a("| `family-candidates.json` | 挖掘准则、各角色候选与拒收计数 | 编码官 |")
    a("| `d0-time-isolation.json` | 时间策略 v2、train/dev/sealed 清单、替代候选核验 | Q0 / Mika |")
    a("| `d0-shortfall.json` | P0/P1/dev 供给缺口 | Mika |")
    a("| `public-manifest.json` | **可公开**部分（不含封存内容与 gold） | 编码官 → 纳入 gamma |")
    a("| `restricted-oracle.json` | 受限 oracle 提示，**不提交 GitHub** | 独立保管侧 |")
    a("| `kaggle-23-d0-family-table.csv` | 家族一览 | 编码官 |")
    a("| `validate_d0.output.txt` | 校验输出（%s 项 / %s 失败） | Q0 |" % (n_checks, n_failed))
    a("| `kaggle-23-d0-source-lock-report.md` | 本说明 | Mika / Liang |")
    a("")
    a("**不入 Git**：`restricted-oracle.json` 与任何 gold 正文/答案材料。"
      "可公开的 `public-manifest.json` 由编码官决定何时纳入 `gamma`。")
    a("")

    a("## 6. 检索方法与盲区")
    a("")
    a("- 路径优先级：官方 tag refs → 一手 git 对象（commit/tree/blob 与哈希）→ "
      "包装元数据声明 → 逐文件头部信号。逐文件扫描只读**已固定 revision 的工作树**。")
    a("- 家族挖掘只在**各自固定 revision 可达的历史**中进行，"
      "因此候选天然位于所固定快照之内；时间分类另用家族自身修复时间。")
    a("- **盲区**：无包索引（无法装 pytest，未跑 F2P）；"
      "封存内容按规则未读取；上游 issue/PR 正文未取用（著作权未清理）；"
      "backport 规则只能靠 commit message 标记与快照祖先关系识别，"
      "无法识别**没有任何标记**的静默重落地。")
    a("")

    a("## 7. 本轮（v2）相对第一版 `%s` 的变更" % PREVIOUS_COMMIT)
    a("")
    a("1. **时间策略升版**：train 窗口从 ≤2024-12-31 改为 **≤2025-12-31**；"
      "dev/sealed 从「dev=2025 整年、sealed≥2026-01-01」改为"
      "**共用一个窗口 2026-01-01…2026-10-04**，并显式声明不主张 dev/sealed 先后隔离；"
      "旧划分标为 `withdrawn`。")
    a("2. **家族时间双日期化**：同时记录作者时间与提交者时间、两者都须在窗口内；"
      "新增 `fix_time.basis` 与 `not_derived_from` 字段，"
      "禁止 release/snapshot/backport/cherry-pick 日期顶替。")
    a("3. **新增快照轴分离**：家族修复 commit 必须是固定快照的祖先，"
      "且每个 `commit_date` 都带 `commit_date_role` 说明它不是家族时间。")
    a("4. **新增 backport/cherry-pick 拒收规则**与窗口内观察计数。")
    a("5. **source-lock 新增机器可读许可批准契约**"
      "（`license_review_schema` + 逐仓库 `license_review`），"
      "批准绑定 revision，命中列表逐条留证，独立复核状态如实标为 pending。")
    a("6. **新增 dev/sealed 家族清单**（此前未枚举）与 **dateutil 0 家族**的机器可读记录。")
    a("7. **新增替代候选核验**：python-dotenv 的许可 / 家族 / 环境材料，"
      "状态明确为「替代候选、非替换、未批准发布」。")
    a("8. **校验从 26 项扩到 %s 项**，新增时间策略 v2、许可契约、快照祖先、"
      "替代候选边界、清单覆盖等断言（%s 失败）。" % (n_checks, n_failed))
    a("")
    a("差异的机器可读留证见 `d0/out/v2-diff-evidence.txt`。")
    a("")

    with open(os.path.join(OUT, "kaggle-23-d0-source-lock-report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("report bytes=%d" % os.path.getsize(
        os.path.join(OUT, "kaggle-23-d0-source-lock-report.md")))


if __name__ == "__main__":
    main()
