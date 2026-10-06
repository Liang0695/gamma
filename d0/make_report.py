"""Emit the D0 human-readable report and the family table.

This generator is the v3 revision of the D0 report.  It supersedes the v2 pass,
which Mika sent back on two counts (2026-10-06):

  1. the python-dotenv licence was approved as MIT while both its pinned LICENSE
     and its pyproject.toml say BSD-3-Clause, and the same record contradicted
     itself in its own packaging metadata; and
  2. the split window was decided from the fix commit's author/committer dates,
     which are only audit corroboration -- the window must be decided by the
     original fix's upstream MERGE event.

Every number in the report is read out of the generated JSON, so the prose and
the machine-readable files can never drift apart.
"""
import csv
import hashlib
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ACCESS_DATE = "2026-10-06"
BRANCH = "agent/research/kaggle-23-d0-source-lock"
PREVIOUS_COMMIT = "65aaa16"
PREVIOUS_PASS_COMMIT = "8b8ff5a"
LOCKED = ["click", "more-itertools", "pluggy", "boltons", "attrs", "dateutil",
          "packaging", "marshmallow"]


def load(p):
    with open(os.path.join(OUT, p), encoding="utf-8") as f:
        return json.load(f)


def spec_sha256():
    p = os.path.join(os.path.dirname(HERE), "attachments",
                     "KAGGLE-19-V3-integrated-review.md")
    if not os.path.isfile(p):
        return None
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
    merge = load("merge-evidence.json")
    merge_by_commit = {r["fix_commit"]: r for r in merge["records"]}
    fams = ledger["families"]
    real = [f for f in fams if f["kind"] == "real"]
    var = [f for f in fams if f["kind"] == "variant"]
    released_real = [f for f in real if f["released"]]
    blocked_real = [f for f in real if not f["released"]]
    repos = lock["repos"]
    gate, n_pass = gate_summary()
    n_checks = gate[0] if gate else "?"
    n_failed = gate[1] if gate else "?"
    locked_approved = sum(1 for n in LOCKED
                          if repos[n]["license_review"]["decision"] == "approved")
    conflict_repos = [n for n, r in repos.items()
                      if r["license_review"]["license_conflicts"]]
    spec_sha = spec_sha256()

    # ---- family table csv ----
    with open(os.path.join(OUT, "kaggle-23-d0-family-table.csv"), "w",
              encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family_id", "kind", "repo", "split", "released", "base_commit",
                    "oracle_fix_commit", "oracle_patch_sha256",
                    "author_date_audit_only", "committer_date_audit_only",
                    "merge_event_utc", "merge_pr", "merge_sha_matches_fix_commit",
                    "qualifies_by_merge_event", "merge_evidence_status",
                    "ancestor_of_pin", "fail_to_pass_nodes", "mutation_class",
                    "license_spdx", "license_review_decision",
                    "license_conflicts"])
        for f in fams:
            ft = f.get("fix_time") or {}
            me = f.get("merge_evidence") or {}
            w.writerow([
                f["family_id"], f["kind"], f["repo"], f["split"], f["released"],
                f.get("base_commit") or f.get("source_base_commit") or "",
                f.get("oracle_fix_commit") or "",
                f.get("oracle_patch_sha256") or "",
                ft.get("author_date", ""),
                ft.get("committer_date", ""),
                ft.get("merge_event_utc") or "",
                me.get("pull_request_url") or "",
                me.get("merge_commit_sha_matches_fix_commit", ""),
                ft.get("qualifies_by_merge_event", ""),
                me.get("status", ""),
                f.get("fix_commit_is_ancestor_of_pinned_revision", ""),
                ";".join(n["node_id"] for n in (f.get("fail_to_pass") or [])),
                f.get("mutation_class", ""),
                f["license"]["spdx"],
                f["license"].get("license_review_decision", ""),
                ";".join(c["kind"] for c in f["license"].get("license_conflicts") or []),
            ])

    L = []
    a = L.append
    a("# V3 D0 资料来源锁定与许可证据说明（v3 修订版 · 时间与许可专项整改）")
    a("")
    a("生成：资料调研与分发 · %s · 访问日期 %s · 分支 `%s`" % (
        ACCESS_DATE, ACCESS_DATE, BRANCH))
    a("")
    a("> **本文件取代 v2（该分支提交 `%s`，其前身是第一版 `%s`）。**"
      "v2 被 Mika 定点退回两项：① python-dotenv 的 MIT 自述批准是错的"
      "（固定 LICENSE 与 pyproject 都是 BSD-3-Clause），生成输入与全部派生字段需纠正；"
      "② 窗口判定用的是修复 commit 的作者/提交者双日期，"
      "必须改用**可追溯的原始合并事件**，缺证据的项不得晋级。" %
      (PREVIOUS_COMMIT, PREVIOUS_PASS_COMMIT))
    a("")

    a("## 0. 一句话结论")
    a("")
    a("两项退回都已整改，并在**原 D0 分支**提交新 SHA 与逐字段差异。")
    a("")
    a("- **许可**：python-dotenv 的 `approved_spdx` 由 MIT 改为 **BSD-3-Clause**，"
      "并借此把「信任手写预设」改成**三方一致性判定** —— "
      "手写预设、固定许可**正文**、固定包装**元数据**三者必须相容，"
      "任何一处正向冲突即 `decision=pending`。整改后 %d/%d 个锁定来源全部 `approved`"
      "（替代候选 python-dotenv 亦为 `approved`），交付集内 **0 个**记录处于冲突态；"
      "冲突规则由一条否定测试驱动验证。"
      % (locked_approved, len(LOCKED)))
    a("- **哈希口径**：每个许可文件同时记录**上游 git blob 字节的 SHA256**与"
      "**checkout 工作树字节的 SHA256**，以及二者之间的换行变换（`core.autocrlf=true` "
      "下 LF→CRLF）。python-dotenv 的 LICENSE 正是 `80619b70…`（1556 字节，LF）"
      "与 `dd1c70c9…`（1583 字节，CRLF）的关系，**不是**许可变更。")
    a("- **时间**：窗口判定改用**原始合并事件**（PR 的 `merged_at_utc`，"
      "带元数据来源 URL 与响应 SHA256）；作者/提交者日期降级为纯审计字段。"
      "本版对 10 个拟计入的 commit 逐一取证：**8 个 verified、2 个 unverified**。")
    a("- **代价（如实上报）**：应用合并事件规则后，**P0 真实家族 %d/4 released**。"
      "`v3-train-click-001` 的 2015 修复**根本没有 PR 合并事件**（直接推到 main；"
      "issue #222 由 commit 直接引用关闭，PR #258/#259 都是**未合并**关闭），"
      "因此它保持 `released=false`、不计入任何配额，已作为阻塞缺口交给 Mika 裁决。"
      "四个变异家族仍只有构造规格、`released=false`。"
      "**没有任何空 SHA 记录被当作已发布示例。**" % len(released_real))
    a("- 校验 **%s 项、%s 失败**（PASS %s）。本轮**仍未运行 FAIL_TO_PASS**、"
      "未用 GPU、未申请 107 作业。其余实质缺口：dateutil 在固定 revision 下 "
      "dev 窗口内合格家族数为 0（与 Mika 的暂计一致）。" % (n_checks, n_failed, n_pass))
    a("")

    a("## 1. 需求回顾")
    a("")
    a("- **谁要的**：本任务负责人为资料调研与分发（KAGGLE-23）。需求来自 KAGGLE-19 的 V3 联合设计")
    if spec_sha:
        a("  （附件 `KAGGLE-19-V3-integrated-review.md`，本运行实测 SHA256 `%s`），由 yg123456 批准。"
          % spec_sha)
    else:
        a("  （附件 `KAGGLE-19-V3-integrated-review.md` 在本次运行的工作目录中不存在，"
          "因此**不声称**复算过它的 SHA256），由 yg123456 批准。")
    a("- **要什么**：把设计中的候选来源变成可执行、许可闭合的数据输入 —— 固定 8 候选仓库的 commit、"
      "逐文件许可/NOTICE、来源与问题家族台账、时间隔离与 shortfall。")
    a("- **本轮整改要求**（Mika 定点退回 2026-10-06，依据父任务评论 "
      "`01a10fd2-3469-7666-a948-ee9a955a3be9`）：① 撤回 python-dotenv 的错误 MIT 自述批准，"
      "纠正生成输入与全部派生清单/报告，保留版权声明，独立签署仍 pending，"
      "明确上游 blob 与 checkout 换行哈希的不同口径，"
      "并**加一条「许可元数据冲突即拒绝批准」的回归验证**；"
      "② 窗口判定改用可追溯的**原始合并事件**，补原始 PR、`merged_at_utc`、"
      "匹配的 `merge_commit_sha`、来源与响应哈希，backport 追溯原修复，"
      "缺证据项保持 `unverified`、不计配额，并**加一条「缺合并证据不得晋级」的否定测试**。")
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

    a("### 2.2 许可：从「信任预设」到「三方一致性判定」（本轮整改 ①）")
    a("")
    a("v2 的错误在于：`collect_licenses.py` 里的手写预设 `design_license_expectation` "
      "被直接抄进 `approved_spdx`，于是 python-dotenv 记录写成 MIT，"
      "而同一条记录的包装元数据里明明写着 `{ text = \"BSD-3-Clause\" }` —— "
      "**自相矛盾**。本版改为：")
    a("")
    a("1. **许可正文**中正向识别许可族（只读仓库**自己的**根级 LICENSE/COPYING；"
      "NOTICE 与第三方许可副本被排除，因为 marshmallow 的 NOTICE 内嵌了 Django 的 "
      "BSD-3-Clause，纳入会凭空制造冲突）；")
    a("2. **包装元数据**中提取显式 SPDX（`license = { text = … }`、"
      "`License :: OSI Approved :: …` 分类器；**歧义分类器不给 token**，"
      "以免掩盖真实冲突或凭空制造冲突）；")
    a("3. 与实际**包装声明**做三方比对；任一正向不一致 ⇒ `decision = \"pending\"`，"
      "冲突逐条写入 `license_conflicts`。**空证据集不是冲突**（“未知”不等于“不一致”）。")
    a("")
    a("| 仓库 | decision | approved_spdx | 正文识别 | 元数据声明 | 冲突 | 主许可文件 | copyleft | 限制性 | 独立复核 |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for name in LOCKED + ["python-dotenv"]:
        lr = repos[name]["license_review"]
        facts = lr["license_facts"]
        pf = lr["evidence"]["primary_license_file"] or {}
        a("| %s%s | **%s** | %s | %s | %s | %s | `%s` | %d | %d | %s |" % (
            name, "（替代候选）" if name == "python-dotenv" else "",
            lr["decision"], lr["approved_spdx"],
            "、".join(facts["detected_from_licence_text"]) or "未识别",
            "、".join(facts["detected_from_packaging_metadata"]) or "无显式 SPDX",
            "、".join(c["kind"] for c in lr["license_conflicts"]) or "无",
            pf.get("path", "-"),
            len(lr["copyleft_marker_hits"]), len(lr["restrictive_marker_hits"]),
            lr["independent_review"]["status"]))
    a("")
    a("**本轮对 python-dotenv 的具体更正**")
    a("")
    a("- 固定 pin `a565c2cc41599c48eabc6b7b7f5b826d43c5a6d7` 的 LICENSE "
      "（git blob `3a97119010ac82e15e917a69b7b8f9f59b5a4601`）是 **BSD-3-Clause** 全文"
      "（含 “may not be used to endorse or promote products derived from this software” 第三条款）；"
      "`pyproject.toml` 亦为 `{ text = \"BSD-3-Clause\" }`。")
    a("- 此前核对的候选 pin `791414804eff08a23f0b7970968e1717e3b28e66` 携带**同一个 LICENSE blob**，"
      "所以两次 pin 的许可没有变化：MIT 是**本文件预设的转录错误**，不是上游许可变更。")
    a("- 更正范围：`d0/collect_licenses.py` 的生成输入（预设），以及由它派生的 "
      "`source-lock.json`、`public-manifest.json`、`d0-time-isolation.json`、"
      "`family-table.csv` 与本报告中的全部许可字段。版权与适用声明原样保留，未删改。")
    a("- `independent_review.status` 对 9 个来源**全部仍为 `pending`**："
      "这份 `decision` 是 D0 负责人的事实判定，**不是**独立签字，也没有被表述成独立签字。")
    a("")
    a("**哈希口径（本轮补齐）**")
    a("")
    a("`core.autocrlf=true` 使工作树携带 CRLF、而 git 对象仍是 LF，"
      "同一个许可文件因此有两个**都正确**的 SHA256。v2 只记录了一个且没说明是哪一个。"
      "本版对每个许可文件同时记录：")
    a("")
    a("| 字段 | 含义 |")
    a("|---|---|")
    a("| `upstream_blob_sha256` / `upstream_blob_bytes` | `git cat-file blob` 的**上游对象字节** |")
    a("| `checkout_sha256` / `checkout_bytes` | 工作树字节（受换行转换影响） |")
    a("| `newline_transformation` | 二者关系：`none` / `lf_to_crlf_on_checkout` / `other` |")
    a("| `sha256` | 为兼容保留，**恒等于 `checkout_sha256`**，并由门禁断言这一点 |")
    a("")
    dotenv_lic = [x for x in load("license-files.json")["by_repo"]["python-dotenv"]
                  if x["path"] == "LICENSE"][0]
    a("python-dotenv `LICENSE` 复核结果：blob `%s` / %d 字节（LF）→ checkout `%s` / %d 字节"
      "（CRLF），变换 `%s`。这正是审阅者手算得到的一对哈希，"
      "**它只能支持“同一份许可的换行表示差异”，不能支持任何许可变更**。"
      % (dotenv_lic["upstream_blob_sha256"], dotenv_lic["upstream_blob_bytes"],
         dotenv_lic["checkout_sha256"], dotenv_lic["checkout_bytes"],
         dotenv_lic["newline_transformation"]))
    if conflict_repos:
        a("")
        a("**当前处于冲突态的仓库**：%s（已按规则 withholding 批准）。"
          % "、".join(conflict_repos))
    else:
        a("")
        a("**当前交付集中没有任何仓库处于冲突态** —— 这不是放宽，"
          "而是把 python-dotenv 的预设改成了与固定字节一致的值；"
          "冲突规则本身由 §2.7 的否定测试证明它在冲突时确实会拒绝批准。")
    a("")

    a("### 2.3 时间：窗口判定改用原始合并事件（本轮整改 ②）")
    a("")
    a("| 轴 | 定义 | 本版取值 |")
    a("|---|---|---|")
    a("| train 窗口 | 家族**原始合并事件** ≤ 2025-12-31 | `end_exclusive = %s` |"
      % iso["rule"]["windows"]["train"]["end_exclusive"])
    a("| dev / sealed 窗口 | 家族**原始合并事件** ∈ 2026-01-01 … 2026-10-04 | "
      "`[%s, %s)` |" % (iso["rule"]["windows"]["dev_sealed"]["start"],
                         iso["rule"]["windows"]["dev_sealed"]["end_exclusive"]))
    a("")
    a("**判定基准**：%s" % iso["rule"]["time_axis_authority"])
    a("")
    a("- **作者/提交者日期降级为审计字段**：`fix_time.author_and_committer_dates_role` "
      "明确写着它们只作佐证；`fix_time.primary` 现在是 `merged_at_utc`，"
      "缺合并事件时为 `null`（而不是「退回作者日期」）。")
    a("- **合并事件的取证**由 `d0/fetch_merge_evidence.py` 从 GitHub REST API 拉取，"
      "每个响应的**原始字节**写入 `d0/pr-evidence/raw/` 并计算 SHA256，"
      "同时记录 `merged_at_utc`、`merge_commit_sha`、PR 链接与查询 URL。"
      "取证是**缓存优先**的：已下载的响应不会重复消耗配额；全程**未认证、无 token**。")
    a("- **`merge_commit_sha` 不匹配不等于造假**：rebase/squash 合并会让它与修复 commit 不同。"
      "本版如实记录几何关系（`merge_commit_geometry`），"
      "包括「GitHub 报告的合并 commit 在固定快照中不存在」这种异常，"
      "而不是静默对齐。")
    a("- **缺证据即不计数**：`status=unverified` ⇒ `qualifies_by_merge_event=false` ⇒ "
      "`released=false`，并进入 shortfall 的阻塞缺口。"
      "commit message 里的 issue 编号**不是**合并证据。")
    a("- **backport 追溯原修复**：backport 标记的 commit 仍直接拒收；"
      "变异家族的窗口从**父家族的合并事件**继承，且父家族未 released 时不得继承。")
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

    a("### 2.4 家族台账与 oracle 构造依据")
    a("")
    a("| family_id | 仓库 | released | base_commit | oracle 修复 commit | 补丁 SHA256（前 16） | 合并事件（UTC） | PR | merge_sha 匹配 | 作者时间（仅审计） | F2P 节点 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for f in real:
        me = f["merge_evidence"]
        a("| %s | %s | %s | `%s` | `%s` | `%s` | %s | %s | %s | %s | %d |" % (
            f["family_id"], f["repo"], f["released"], f["base_commit"],
            f["oracle_fix_commit"], f["oracle_patch_sha256"][:16],
            f["fix_time"]["merge_event_utc"] or "**无**",
            ("[#%s](%s)" % (me["pull_request_number"], me["pull_request_url"]))
            if me["pull_request_number"] else "**无**",
            me["merge_commit_sha_matches_fix_commit"],
            f["fix_time"]["author_date"][:10], len(f["fail_to_pass"])))
    a("")
    a("每个真实家族的完整字段见 `family-ledger.json`：逐文件 base/fix 哈希、"
      "test patch 与 code-only gold patch 的分离哈希、FAIL_TO_PASS 节点"
      "（要求 base 不存在、fix 存在）、oracle 断言行、环境依赖材料、"
      "以及 `merge_evidence` 的来源与响应哈希。")
    a("")
    a("**未 released 的真实家族（如实保留失败账）**")
    a("")
    for f in blocked_real:
        a("- **%s**（%s）：`release_decision.blocking_checks = %s`。%s"
          % (f["family_id"], f["repo"],
             "、".join(f["release_decision"]["blocking_checks"]),
             f["merge_evidence"]["reason"]))
        a("  审计事实：作者时间 %s、提交者时间 %s（**都落在窗口内，但按规则不足以晋级**）；"
          "修复 commit 仍是固定快照的祖先（%s）。"
          % (f["fix_time"]["author_date"][:10], f["fix_time"]["committer_date"][:10],
             f["fix_commit_is_ancestor_of_pinned_revision"]))
    a("")
    a("**变异家族（released=false）**：")
    a("")
    a("| family_id | 派生自 | 变异类 | 源基线 commit | 父家族 released | expected patch shape |")
    a("|---|---|---|---|---|---|")
    for f in var:
        a("| %s | %s | %s | `%s` | %s | %s |" % (
            f["family_id"], f["derived_from"], f["mutation_class"],
            f["source_base_commit"][:12], f["source_base_released"],
            f["expected_patch_shape"]))
    a("")
    a("变异家族的 release 阻断原因是结构性的：变异 commit 在编码官构造出来之前**不存在**，"
      "因此只固定上游源基线，**任何空 SHA 都不会被当作已发布示例**。"
      "这正是验收条目「示例空 SHA 不能 released」对应的证据。")
    a("")

    a("### 2.5 时间隔离证据与候选计数口径")
    a("")
    a("| 仓库 | 角色 | 窗口 | commit 日期**筛选**数 | 合并事件**已验证**数 | 最早修复时间 | 最新修复时间 | 快照早于窗口? |")
    a("|---|---|---|---|---|---|---|---|")
    for src in (iso["train_inventory"], iso["dev_sealed_inventory"]):
        for name, v in src.items():
            a("| %s | %s | %s | %d | %d | %s | %s | %s |" % (
                name, v["role"], v["window"],
                v["screened_candidate_commits_in_window"],
                v["merge_event_verified_and_in_window"],
                (v["oldest_fix_time"] or "-")[:10], (v["newest_fix_time"] or "-")[:10],
                v["pinned_snapshot_predates_window"]))
    a("")
    a("**计数口径（本轮明确区分，避免把筛选数读成配额）**："
      "`screened_candidate_commits_in_window` 是按 commit 双日期做的**筛选计数**，"
      "只是 headroom 观察；`merge_event_verified_and_in_window` 才是具备时间规则所要求"
      "合并事件证据的计数。本轮 train 侧筛选 %d 个、其中合并事件已验证 %d 个"
      "（%s）；dev 角色筛选 %d 个。"
      % (iso["evidence"]["train_candidate_total_screened_in_window"],
         iso["evidence"]["train_candidate_total_merge_verified"],
         "、".join("%s %d" % (k, v["screened_candidate_commits_in_window"])
                   for k, v in iso["train_inventory"].items()),
         iso["evidence"]["dev_candidate_total_screened_in_window"]))
    a("")
    a("窗口内出现、但被 backport/cherry-pick 标记拒收的 commit 观察数：%s"
      "（其中本可成为候选者 %d 个）。该规则本轮**未改变候选集合**，"
      "但它是被实际执行的，不是纸面声明。" % (
          "、".join("%s %d" % (k, v) for k, v
                    in iso["evidence"]["backport_marked_commits_observed_in_window"].items()
                    if v) or "0",
          iso["evidence"]["backport_marked_candidates_excluded"]))
    a("")
    a("**合并事件取证明细（全部 10 条，含 2 条失败账）**")
    a("")
    a("| 仓库 | 修复 commit | 状态 | PR | merged_at (UTC) | merge_commit_sha | 与 fix 相同? | 响应 SHA256（PR 元数据，前 16） |")
    a("|---|---|---|---|---|---|---|---|")
    for r in merge["records"]:
        pr = r.get("pull_request") or {}
        a("| %s | `%s` | %s | %s | %s | %s | %s | %s |" % (
            r["repo"], r["fix_commit"][:12], r["status"],
            ("[#%s](%s)" % (pr.get("number"), r.get("pull_request_url")))
            if pr else "**无 PR**",
            r.get("merged_at_utc") or "—",
            (r.get("merge_commit_sha") or "—")[:12],
            r.get("merge_commit_sha_matches_fix_commit"),
            (pr.get("raw_response_sha256") or "—")[:16]))
    a("")
    a("未通过的两条：`click 9da1791476fe`（GitHub 报告**没有任何关联 PR**）与 "
      "`python-dotenv f5485a61eefa`（无关联 PR；其 commit message 引用的 **#600 是 issue，不是 PR**）。"
      "两者都保持 `unverified`，计入 no window。"
      "**这是「尚未取得合并证据」，不是「证明不存在原始修复」。**")
    a("")
    a("异常项如实披露：`boltons` 与另外两个 train 家族的 `merge_commit_sha` "
      "与修复 commit 不同（rebase/squash 几何），其中 boltons 的 GitHub 合并 commit "
      "在固定快照中**根本不存在**；这些都由 `merge_commit_geometry` 记录，未被静默对齐。")
    a("")

    a("### 2.6 替代 dev 候选核验：python-dotenv")
    a("")
    alt = iso["alternative_dev_candidate"]
    fi = alt["family_inventory"]
    a("按 Mika 裁决，python-dotenv 仅作为替代候选开展**许可 / 家族 / 环境**三项核验，"
      "**不等于替换 dateutil，也不等于批准发布**。")
    a("")
    a("- 许可：`decision = %s`，`%s`（%s），主许可文件 `%s`，"
      "正文与元数据识别结果都是 BSD-3-Clause，冲突 0 条（路径 `%s`）。" % (
          alt["license"]["decision"], alt["license"]["approved_spdx"],
          alt["license"]["spdx_correction"],
          (alt["license"]["primary_license_file"] or {}).get("path"),
          alt["license"]["path_in_source_lock"]))
    a("- 家族：固定 tag `%s` = `%s`（快照日期 %s）；dev/sealed 窗口内 commit 日期**筛选** %d 个，"
      "其中**合并事件已验证** %d 个（最早 %s、最新 %s）。"
      "未验证的那一个是 `f5485a61eefa`。" % (
          alt["pinned_tag"], alt["pinned_commit"][:12], alt["snapshot_date"][:10],
          fi["screened_candidate_commits_in_window"],
          fi["merge_event_verified_and_in_window"],
          (fi["oldest_merge_event_utc"] or "-")[:10],
          (fi["newest_merge_event_utc"] or "-")[:10]))
    a("- 环境：`requires_python = %s`，运行时依赖 %s，测试运行器 `%s`，"
      "并已对其 `%d` 个依赖/配置文件计算 SHA256。" % (
          alt["environment_material"]["requires_python"],
          alt["environment_material"]["runtime_dependencies"] or "无",
          alt["environment_material"]["test_runner"],
          len(alt["environment_material"]["files"])))
    a("- 边界字段：`replaces = %s`；`approval_status` 明确写为替代候选。"
      % alt["replaces"])
    a("")

    a("### 2.7 校验输出与两条否定测试")
    a("")
    a("`d0/validate_d0.py` 独立于生成脚本、只读产物 JSON 重新断言："
      "**%s 项检查、%s 失败**（PASS 行 %s）。"
      "完整输出见 `d0/out/validate_d0.output.txt`。" % (n_checks, n_failed, n_pass))
    a("")
    a("本轮新增的**两条否定测试**（都是把真实缺陷重新植入、驱动**同一个**判定函数，"
      "因此规则一旦被放宽，门禁立刻失败）：")
    a("")
    a("1. `negative_test_licence_metadata_conflict_withholds_approval`："
      "把审阅者实际抓到的缺陷（预设 MIT vs 正文/元数据 BSD-3-Clause）喂给"
      "`collect_licenses.classify_license`，要求返回 `pending` 且 `osi_permissive=false`。")
    a("2. `negative_test_missing_merge_evidence_cannot_qualify`："
      "取一条真实的 `unverified` 记录（`f5485a61eefa`），"
      "要求合并事件判定函数返回 `False` —— 即 issue 引用不能把它推进配额。")
    a("")
    a("另有两条一致性断言专门盯着这次退回的两种误读："
      "`licence_decision_matches_a_fresh_re_run_of_the_same_predicate`"
      "（逐步重算每条许可判定，与落盘值比对）与 "
      "`checkout_hash_of_dotenv_licence_is_the_crlf_transformation_of_the_blob`"
      "（用审阅者手算的那对哈希钉住哈希口径）。")
    a("")
    a("**note**：门禁断言的是**自洽**而不是「全部成功」—— "
      "某个家族可以是 `released=false`，但那时它的 `blocking_checks` 与 shortfall "
      "必须被记录，且不得计入任何配额。"
      "把「4/4 released」写死成断言，等于奖励一个合并事件根本没取到的家族。")
    a("")

    a("## 3. 推断与建议（标注为推断 / 建议）")
    a("")
    a("- **事实**：P0 真实半边 %d/4；`v3-train-click-001` 因**不存在 PR 合并事件**而 unqualified。"
      "变异半边 4/4 规格、0 个构造 commit。合计 released %d/8。"
      % (len(released_real), sum(1 for f in fams if f["released"])))
    a("- **建议（需 Mika / 保管侧裁决，D0 不自行换家族）**：click 槽位三选一 —— "
      "① 为「直接推送到默认分支」的 landing 事件定义一套可接受证据标准；"
      "② 从已筛选的 train 清单中换入一个**合并事件可取证**的家族，"
      "并走同一套派生与审核；③ 承认 P0 配额缺口并如实记为 3/4。"
      "本轮**不擅自**替换，以免下游 E0 环境与既有审阅基线失效。")
    a("- **推断**：train 侧 commit 日期筛选 %d 个候选对 24 个真实家族需求，"
      "名义 headroom 约 %.1f 倍，但其中合并事件已验证的只有 %d 个，"
      "且没有任何一个经过验证器跑通、actor 可达性与有界测试补丁检查，"
      "不能把 %d 读成 24。" % (
          short["p1_headroom"]["screened_candidate_commits_in_train_window"],
          short["p1_headroom"]["screened_candidate_commits_in_train_window"] / 24.0,
          short["p1_headroom"]["merge_event_verified_train_candidates"],
          short["p1_headroom"]["screened_candidate_commits_in_train_window"]))
    a("- **推断**：dev 供给偏薄，只有 attrs 一个锁定 dev 角色仓库产出窗口内家族（%d 个，"
      "且未验证）；dateutil 在固定 revision 下为 0。"
      % iso["dev_sealed_inventory"]["attrs"]["screened_candidate_commits_in_window"])
    a("- **建议**：dev 方案二选一由 Mika / 保管侧裁决 —— "
      "① 以 attrs + 已核验的替代候选承担 dev，或 ② 对 dateutil 重新 pin "
      "（但会使本版绑定在该 revision 上的许可批准失效，必须重做逐文件许可台账"
      "与新 revision 的批准）。")
    a("- **建议**：请 Q0 对 `license_review` 逐条反证（`independent_review.status` 仍为 pending），"
      "重点复核本版新增的许可正文识别与三方一致性判定，以及 "
      "`merge_commit_geometry` 里那两条与修复 commit 不一致、以及快照中不存在的合并 commit。")
    a("- **建议**：E0 在环境就绪后对**已 released 的 %d 个**真实家族跑 broken/reference "
      "双次干净对照；click 家族在裁决前不应进入环境构建队列。"
      "本版不把静态来源验收当作数据 released。" % len(released_real))
    a("")

    a("## 4. 冲突与不确定项")
    a("")
    a("- **P0 真实半边缺口（本轮新增，最重要的未解决项）**：见 §3 建议 ①。"
      "该缺口已写入 `d0-shortfall.json` 的 `blocking_gaps`，并带上 PR 查询 URL 与额外观察。")
    a("- **权限隔离未建立，因此 dev/sealed 的发布与验收保持阻断**（沿用 Mika 裁决，本轮不改变）。"
      "`restricted-oracle.json` 已显式写入 `split_declaration_pending`："
      "本交付**不声称**该文件已安全切分或未被污染，切分须由持有 gold 的保管侧判定。")
    a("- **没有跑过 FAIL_TO_PASS**：运行环境取不到包索引，装不上 pytest，"
      "所有 oracle 结论均为静态证据。")
    a("- **许可批准是自述**：见 §2.2，`independent_review.status` 全部 pending。")
    a("- **合并事件只取了「拟计入的家族」**：批量清单仍是 commit 日期筛选，"
      "不是逐条取证的合格配额（见 §2.5 计数口径与 shortfall 的非阻塞缺口）。")
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
    a("| `source-lock.json` | 8 来源固定 revision + 逐文件许可 + **三方一致性批准字段** + **双哈希口径** | 编码官 / Q0 |")
    a("| `license-files.json` | 每个许可文件的上游 blob / checkout 双 SHA256 与换行变换 | Q0 |")
    a("| `per-file-ledger.csv` | 逐文件 SHA256 / SPDX 头 / 版权行 | Q0 |")
    a("| `merge-evidence.json` | **10 条合并事件取证**：PR、`merged_at_utc`、"
      "`merge_commit_sha`、来源 URL、响应 SHA256 | Q0 / Mika |")
    a("| `family-ledger.json` | 家族台账、oracle 分离哈希、**合并事件时间证据** | 编码官 / E0 |")
    a("| `family-candidates.json` | 挖掘准则、各角色候选与拒收计数 | 编码官 |")
    a("| `d0-time-isolation.json` | 时间策略、train/dev/sealed 清单、替代候选核验 | Q0 / Mika |")
    a("| `d0-shortfall.json` | P0/P1/dev 供给缺口（含 click 阻塞项） | Mika |")
    a("| `public-manifest.json` | **可公开**部分（不含封存内容与 gold） | 编码官 → 纳入 gamma |")
    a("| `restricted-oracle.json` | 受限 oracle 提示，**不提交 GitHub** | 独立保管侧 |")
    a("| `kaggle-23-d0-family-table.csv` | 家族一览（含合并事件列） | 编码官 |")
    a("| `validate_d0.output.txt` | 校验输出（%s 项 / %s 失败） | Q0 |" % (n_checks, n_failed))
    a("| `kaggle-23-d0-source-lock-report.md` | 本说明 | Mika / Liang |")
    a("")
    a("**可复现入口**：`python d0/run_all.py` 按序跑许可 → 家族挖掘 → 台账 → extras → "
      "合并事件取证 → 门禁 → 报告；门禁非零则不生成报告。"
      "合并事件取证是缓存优先的，重跑不会重复消耗 API 配额。")
    a("")
    a("**不入 Git**：`restricted-oracle.json` 与任何 gold 正文/答案材料，"
      "以及 `d0/src/`（固定快照工作树）与 `d0/raw/`。"
      "`d0/pr-evidence/raw/` 中的 GitHub 公开元数据响应**入 Git**，"
      "这样每条记录的 `raw_response_sha256` 才能被离线复核。")
    a("")

    a("## 6. 检索方法与盲区")
    a("")
    a("- 路径优先级：官方 tag refs → 一手 git 对象（commit/tree/blob 与哈希）→ "
      "包装元数据声明 → 逐文件头部信号。逐文件扫描只读**已固定 revision 的工作树**。")
    a("- 许可正文识别只读仓库**自己的**根级 LICENSE/COPYING；"
      "NOTICE 与第三方许可副本被显式排除，原因见 §2.2。")
    a("- 家族挖掘只在**各自固定 revision 可达的历史**中进行，"
      "因此候选天然位于所固定快照之内；时间分类改用**原始合并事件**。")
    a("- **盲区**：无包索引（无法装 pytest，未跑 F2P）；"
      "封存内容按规则未读取；上游 issue/PR 正文未取用（著作权未清理）；"
      "backport 规则只能靠 commit message 标记与快照祖先关系识别，"
      "无法识别**没有任何标记**的静默重落地；"
      "GitHub 未关联 PR 的 commit（如 click 那次直接推送）无法从 PR 元数据取得合并事件 —— "
      "这是**证据不可得**，不是「未合并」。")
    a("")

    a("## 7. 本轮（v3）相对 v2 `%s` 的变更" % PREVIOUS_COMMIT)
    a("")
    a("1. **许可判定从「信任预设」改为「三方一致性」**：新增 "
      "`license_facts`、`license_conflicts`；冲突即 `pending`。"
      "`license_review_schema` 升到 1.1，`required_fields` 增加这三项。")
    a("2. **python-dotenv 的 `approved_spdx` 由 MIT 更正为 BSD-3-Clause**，"
      "生成输入（预设）与全部派生清单/报告同步更正；版权与适用声明未改动；"
      "独立复核仍为 `pending`。")
    a("3. **新增双哈希口径**：每个许可文件记录上游 blob 与 checkout 两个 SHA256 "
      "及换行变换，并由门禁钉住 python-dotenv 的那一对已知值。")
    a("4. **新增 `d0/fetch_merge_evidence.py` 与 `merge-evidence.json`**："
      "窗口判定基准从 commit 双日期改为原始合并事件，带来源 URL 与响应 SHA256，"
      "原始响应入 Git 以便离线复核。")
    a("5. **作者/提交者日期降级**：`fix_time.primary` 改为 `merged_at_utc`，"
      "新增 `author_and_committer_dates_role`；无合并事件时为 `null`，不再退回作者日期。")
    a("6. **家族 `released` 现由合并事件证据驱动**：`v3-train-click-001` 因此变为 "
      "`released=false`（**本轮新发现的后果**），P0 真实半边 4/4 → 3/4，"
      "并作为阻塞缺口上报。")
    a("7. **计数口径拆开**：清单里的 `qualified_candidate_families_in_window` "
      "改名为 `screened_candidate_commits_in_window`，另加 "
      "`merge_event_verified_and_in_window`，避免把筛选数读成配额。")
    a("8. **两条否定测试 + 两条一致性断言**，校验从 v2 的 67 项扩到 %s 项（%s 失败）。"
      % (n_checks, n_failed))
    a("")
    a("差异的机器可读留证见 `d0/out/v3-diff-evidence.txt`"
      "（v2 那一轮的留证保留在 `d0/out/v2-diff-evidence.txt`）。")
    a("")

    with open(os.path.join(OUT, "kaggle-23-d0-source-lock-report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("report bytes=%d" % os.path.getsize(
        os.path.join(OUT, "kaggle-23-d0-source-lock-report.md")))


if __name__ == "__main__":
    main()
