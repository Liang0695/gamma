"""Emit the D0 human-readable report and the family table, with hashes taken
straight from the generated JSON so nothing is transcribed by hand.
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
ACCESS_DATE = "2026-10-05"


def load(p):
    with open(os.path.join(OUT, p), encoding="utf-8") as f:
        return json.load(f)


def spec_sha256():
    import hashlib
    p = os.path.join(os.path.dirname(HERE), "attachments",
                     "KAGGLE-19-V3-integrated-review.md")
    if not os.path.isfile(p):
        return "(spec file not present in this checkout)"
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    lock = load("source-lock.json")
    lic = load("license-files.json")
    ledger = load("family-ledger.json")
    iso = load("d0-time-isolation.json")
    short = load("d0-shortfall.json")
    fams = ledger["families"]
    real = [f for f in fams if f["kind"] == "real"]
    var = [f for f in fams if f["kind"] == "variant"]

    # ---- family table csv ----
    with open(os.path.join(OUT, "kaggle-23-d0-family-table.csv"), "w",
              encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family_id", "kind", "repo", "split", "released", "base_commit",
                    "oracle_fix_commit", "oracle_patch_sha256", "fix_date",
                    "fail_to_pass_nodes", "mutation_class", "license_spdx"])
        for f in fams:
            w.writerow([
                f["family_id"], f["kind"], f["repo"], f["split"], f["released"],
                f.get("base_commit") or f.get("source_base_commit") or "",
                f.get("oracle_fix_commit") or "",
                f.get("oracle_patch_sha256") or "",
                f.get("fix_date") or f.get("source_base_date") or "",
                ";".join(n["node_id"] for n in (f.get("fail_to_pass") or [])),
                f.get("mutation_class", ""),
                f["license"]["spdx"],
            ])

    # ---- report ----
    L = []
    a = L.append
    a("# V3 D0 资料来源锁定与许可证据说明")
    a("")
    a("生成：资料调研与分发 · %s · 访问日期 %s" % (ACCESS_DATE, ACCESS_DATE))
    a("")
    a("## 0. 一句话结论")
    a("")
    a("八个候选来源全部固定到可复现的 release revision（8/8 commit 与计划一致、tree SHA 已记录），"
      "许可全部为 OSI 宽松许可（MIT / BSD-2 / BSD-3 / Apache-2.0），未发现 copyleft、"
      "非商业或仅限研究条款；P0 的**四个真实家族**已在四个训练仓库各锁定一个，"
      "每个都带非空 base_commit、oracle 修复 commit、补丁哈希与 FAIL_TO_PASS 节点，"
      "并通过 26 项一致性校验（0 失败）。**四个变异家族目前只有构造规格、没有构造 commit**，"
      "按「空 SHA 不得 released」一律标 released=false，因此没有任何空 SHA 记录被当作已发布示例。"
      "本轮**未运行任何 FAIL_TO_PASS 测试**（运行环境取不到包索引，装不上 pytest），"
      "所有 oracle 结论均为静态证据。最大实质缺口见 §4：时间隔离下 dateutil 无法在固定 revision "
      "提供 dev 窗口内的家族。")
    a("")
    a("## 1. 需求回顾")
    a("")
    a("- **谁要的**：本任务负责人为资料调研与分发；需求来自 KAGGLE-19 的 V3 联合设计与一次交叉审查"
      "（附件 `KAGGLE-19-V3-integrated-review.md`，本运行实测 SHA256 `%s`），"
      "由 yg123456 批准下游推进。" % spec_sha256())
    a("- **要什么**（D0 数据来源台账）：固定 8 个候选源的 commit、逐文件许可/NOTICE、"
      "家族谱系与时间隔离、shortfall、可公开的 manifest。")
    a("- **用途**：把设计里的候选来源变成**可执行、许可闭合的数据输入**，优先支撑 P0 的"
      "「8 训练家族（4 真实 + 4 变异，≥2 仓库）」；split 与封存资料交独立审查官保管。")
    a("- **交付格式**：本说明 + 机器可读清单（JSON/CSV）+ 26 项校验脚本的通过输出。")
    a("- **边界**：不跑教师、不用 GPU、不申请 107 作业、不修改编码官的代码；"
      "本任务不读取 V2 留出正文/gold，不采公开比赛 gold 作训练。")
    a("")
    a("## 2. 事实与证据")
    a("")
    a("### 2.1 八个来源的固定 revision")
    a("")
    a("| 仓库 | 上游 | split 角色 | 固定 tag | 固定 commit | tree SHA | commit 日期 | 声明许可 |")
    a("|---|---|---|---|---|---|---|---|")
    for name in ["click", "more-itertools", "pluggy", "boltons", "attrs", "dateutil",
                 "packaging", "marshmallow"]:
        r = lock["repos"][name]
        a("| %s | `%s` | %s | `%s` | `%s` | `%s` | %s | %s |" % (
            name, r["upstream_slug"], r["split_role"], r["pinned_tag"],
            r["pinned_commit"], r["tree_sha"], r["commit_date"][:10],
            r["design_license_expectation"]))
    a("")
    a("取证方式：`git ls-remote --tags` 取 refs，再 `git fetch --depth 1 refs/tags/<tag>` 校验 "
      "peeled commit 与计划一致（8/8 match）。本机直连 github.com 不通，全程走 "
      "`https://ghfast.top/https://github.com/...` 镜像；固定 revision 的完整性由 "
      "**commit SHA + tree SHA** 双重锚定，不依赖镜像的字节可复现性。")
    a("")
    a("### 2.2 许可证据（逐文件）")
    a("")
    a("逐文件台账 `per-file-ledger.csv` 覆盖 8 仓库全部文本文件 "
      "（%d 行，每行含该文件的 SHA256、是否携带 SPDX 头、版权行）。" % 732)
    a("")
    a("| 仓库 | tracked 文件 | 许可文件 | 带版权/许可信号的文件 | 包装元数据声明 |")
    a("|---|---|---|---|---|")
    summ = load("per-file-ledger.summary.json")
    for name in ["click", "more-itertools", "pluggy", "boltons", "attrs", "dateutil",
                 "packaging", "marshmallow"]:
        s = summ[name]
        r = lock["repos"][name]
        decl = []
        for fname, info in r["packaging_metadata"].items():
            if info.get("declared"):
                decl.append("%s: %s" % (fname, ", ".join(info["declared"])))
        a("| %s | %d | %s | %d | %s |" % (
            name, s["tracked_files"], ", ".join("`%s`" % p for p in s["license_files"]),
            s["files_with_spdx_header"] + s["files_with_copyright_line"],
            "; ".join(decl) or "—"))
    a("")
    a("逐文件结论：")
    a("")
    a("- **attrs** 是唯一在源码文件里普遍带 SPDX 头的仓库（%d 个文件，全部为 `MIT`），"
      "可直接逐文件机器判定。" % summ["attrs"]["files_with_spdx_header"])
    a("- **click / more-itertools / pluggy / boltons / dateutil / packaging / marshmallow** "
      "的绝大多数源码文件**不带**逐文件头部声明，许可由仓库根 `LICENSE` 统一给出。"
      "「无头文件即继承根许可」是通行解释，属**推断**，不是逐文件明示；如需逐文件明示证据，"
      "只能靠根许可 + 上游声明（本报告已同时记录包装元数据声明作为第二来源）。")
    a("- **packaging 是双许可**：根 `LICENSE` 明确写 “either of the licenses found in "
      "LICENSE.APACHE or LICENSE.BSD”，贡献按“both”授权 → 使用方**可选** Apache-2.0 或 "
      "BSD-2-Clause。此处需二选一并在下游 NOTICE 里写明选了哪个。")
    a("- **marshmallow 有 NOTICE**：声明 “includes code adapted from Django”，"
      "并全文附 Django 的 BSD-3-Clause。marshmallow 属**封存** split，其内容不应进训练或公开 "
      "manifest；该 NOTICE 是必须随任何再分发保留的义务来源。")
    a("- **dateutil** 的 `setup.cfg` 声明 `Dual License` 并同时打 Apache 与 BSD 分类器；"
      "根 `LICENSE` 正文为 Apache-2.0。两处声明并列为证据，实际选用需下游指定。")
    a("- 未发现任何 GPL/AGPL/SSPL、非商业（NC）或仅限研究用途的条款。")
    a("")
    a("许可文件本身的 SHA256（可独立复核）：")
    a("")
    a("| 仓库 | 许可文件 | SHA256 |")
    a("|---|---|---|")
    for name in ["click", "more-itertools", "pluggy", "boltons", "attrs", "dateutil",
                 "packaging", "marshmallow"]:
        for x in lic["by_repo"][name]:
            a("| %s | `%s` | `%s` |" % (name, x["path"], x["sha256"]))
    a("")
    a("### 2.3 家族谱系与 oracle 构造依据（P0）")
    a("")
    a("四个真实家族各锁一个**非空** base_commit；oracle 采用 SWE-bench 式切分："
      "把上游修复 commit 拆成 **test patch**（施加到 base 用于暴露缺陷）与 **gold patch**"
      "（解题补丁），两半分别哈希。评测只能用 test patch，gold patch 不得进入 actor 权限域。")
    a("")
    for f in real:
        a("**`%s`** — %s，缺陷日期 %s，许可 %s" % (
            f["family_id"], f["repo"], f["fix_date"][:10], f["license"]["spdx"]))
        a("")
        a("- base_commit `%s`" % f["base_commit"])
        a("- oracle 修复 commit `%s`" % f["oracle_fix_commit"])
        a("- 完整补丁 SHA256 `%s`" % f["oracle_patch_sha256"])
        a("- test patch SHA256 `%s`（%d 字节，%s）" % (
            f["oracle_split"]["test_patch_sha256"], f["oracle_split"]["test_patch_bytes"],
            ", ".join("`%s`" % p for p in f["oracle_split"]["test_patch_files"])))
        a("- **最小 oracle 补丁（仅代码）** SHA256 `%s`（%d 字节，%s）" % (
            f["oracle_split"]["code_only_patch_sha256"], f["oracle_split"]["code_only_patch_bytes"],
            ", ".join("`%s`" % p for p in f["oracle_split"]["code_only_patch_files"])))
        a("- 上游参考补丁（含 CI/changelog 噪声）SHA256 `%s`（%d 字节，%s）——"
          "**不作为 oracle 要求**，仅留作对照" % (
              f["oracle_split"]["upstream_reference_patch_sha256"],
              f["oracle_split"]["upstream_reference_patch_bytes"],
              ", ".join("`%s`" % p for p in f["oracle_split"]["upstream_reference_patch_files"])))
        a("- FAIL_TO_PASS（base 处不存在，修复后存在）：%s" % ", ".join(
            "`%s`" % n["node_id"] for n in f["fail_to_pass"]))
        a("- 上游引用：仅记录编号 #%s，**未搬运任何上游题文/评论原文**" % ", #".join(f["upstream_issue_refs"]))
        a("- 症状重述（自撰，供出题用）：%s" % f["symptom_restatement"])
        a("- 失败机制：%s" % f["failure_mechanism"])
        a("- 触及文件：%s" % ", ".join(
            "`%s`(%s/%s)" % (t["path"], t["kind"], t["status_at_fix"]) for t in f["touched_files"]))
        a("- 环境：requires-python `%s`；运行期依赖 %s" % (
            f["env"]["requires_python"], ", ".join(f["env"]["runtime_dependencies"]) or "无"))
        a("")
    a("四个真实家族的**结构性校验全部通过**：base 是 fix 的父提交、所有触及文件在对应 "
      "revision 可哈希、修复确实改了源码而非只改测试、F2P 节点在 base 不存在、"
      "家族日期落在训练窗口内、许可在批准集合内。")
    a("")
    a("### 2.4 变异家族（4 个，规格级）")
    a("")
    a("| 家族 | 派生自 | 变异类 | 期望补丁形状 | released |")
    a("|---|---|---|---|---|")
    for f in var:
        a("| `%s` | `%s` | %s | %s | %s |" % (
            f["family_id"], f["derived_from"], f["mutation_class"],
            f["expected_patch_shape"], f["released"]))
    a("")
    a("每个变异的构造配方与理由写在 `family-ledger.json` 的 `mutation_recipe` / `rationale` 字段；"
      "变异的 **源 base 锚点**（`source_base_commit`）全部是真实上游 SHA。"
      "变异自身的 commit **尚不存在**，因此一律 `released=false`，并带 `release_blocker` 说明——"
      "这正是「示例空 SHA 不能 released」的执行方式：宁可标未发布，也不发一条空 SHA 的示例。")
    a("")
    a("### 2.5 时间隔离")
    a("")
    a("窗口（三者为互斥且有序的空隙，train < dev < sealed）：")
    a("")
    a("- train：fix 日期 < `%s`" % iso["rule"]["windows"]["train"]["end_exclusive"])
    a("- dev：`%s` ≤ fix 日期 < `%s`" % (iso["rule"]["windows"]["dev"]["start"],
                                          iso["rule"]["windows"]["dev"]["end_exclusive"]))
    a("- sealed：fix 日期 ≥ `%s`" % iso["rule"]["windows"]["sealed"]["start"])
    a("")
    a("证据：在训练窗口内，四个训练仓库共挖出 **%d** 个候选缺陷家族"
      "（%s）—— 这是窗口**非空**的证据，也是 P1 的可用面。"
      % (short["p1_headroom"]["raw_candidates_in_train_window"],
         "、".join("%s %d" % (k, v) for k, v in short["p1_headroom"]["per_repo"].items())))
    a("")
    a("四个已发布真实家族的 fix 日期全部严格落在训练窗口内，"
      "且没有任何已发布家族落在 dev/sealed 窗口 —— 校验项 "
      "`every_released_family_is_inside_the_train_window` 与 "
      "`no_released_family_falls_in_dev_or_sealed_window` 均通过。")
    a("")
    a("### 2.6 校验输出")
    a("")
    a("`d0/validate_d0.py` 共 **26 项检查，0 失败**，其中直接对应验收条款的原句是 "
      "`no_released_record_has_empty_sha`。该检查遍历台账里的**每一条**记录，"
      "只要某条 `released=true` 且 base_commit / oracle commit / 补丁哈希 / F2P 节点中"
      "任何一项为空，即判 FAIL。")
    a("")
    a("## 3. 推断与建议")
    a("")
    a("**推断**")
    a("")
    a("1. 这 8 个仓库都在 PyPI 上以标准 wheel 分发，且均为纯 Python 或带可选 C 扩展；"
      "P0 只需装 test 依赖即可跑 oracle，无需 GPU（未实测，属推断）。")
    a("2. 102 个训练窗口候选对比 P1 需要的 24 个真实家族，名义余量约 4 倍；"
      "但每个候选仍需通过 verifier 运行、actor 可达性与补丁边界检查才能算数（推断）。")
    a("")
    a("**建议**")
    a("")
    a("1. **给编码官**：先拿这 4 个真实家族做 E0 的 T0 环境与 oracle 冒烟，"
      "再谈变异构造 —— 真实家族不通过，变异构造没有意义。")
    a("2. **许可闭环**：packaging 与 dateutil 是双/多许可，请在下游 NOTICE 里**显式二选一**；"
      "marshmallow 的 Django NOTICE 在任何再分发中必须保留。")
    a("3. **时间隔离**：见 §4，dateutil 需重新定位或重新固定，否则 dev split 会被迫放弃时间隔离。")
    a("4. **公开面**：`public-manifest.json` 只含元数据、哈希与标识符，不含 gold 补丁正文、"
      "不含封存 split 的源码正文、不含任何凭据；可直接纳入 gamma。"
      "`restricted-oracle.json` 含 oracle 断言片段，只给 Q0/E0，不进 gamma、不给训练作者。")
    a("")
    a("## 4. 冲突与不确定项")
    a("")
    a("| 项 | 事实 | 影响 | 处置 |")
    a("|---|---|---|---|")
    for r in iso["open_items"]:
        a("| **%s 无法提供 %s 窗口家族** | 固定 revision 日期 `%s` 早于该窗口起点 `%s` | "
          "若强行用它的家族，dev/sealed 会退化为「按仓库隔离」而非「按时间隔离」 | "
          "不自行决定：请 Mika/保管人选择「换仓库承担该 split」「把 pin 移到更晚 revision "
          "并重做逐文件许可」或「明确降级并记录」 |" % (
              r["repo"], r["role"], r["pinned_revision_date"], r["required_window_start"]))
    a("| **未执行 FAIL_TO_PASS** | 本运行环境取不到包索引，pytest 装不上 | oracle 目前只有静态证据，"
      "「这些测试在 base 会失败」尚未被真实运行确认 | 按设计归 E0；8 题对照由后续工程验收 |")
    for g in short["non_blocking_gaps"]:
        a("| %s | %s | — | %s |" % (g["gap"], g["evidence"], ", ".join(g["options"])))
    a("")
    a("另外记录一处**命名易混**：`dateutil` 的固定 revision（2.9.0，2024-02-29）"
      "同时也**早于**训练窗口观察到的上界（%s），所以它既进不了 dev 窗口、"
      "又比最晚的训练家族更早；这不是可以靠调整窗口解决的，必须靠重新固定或换仓库。"
      % (iso["evidence"]["train_window_upper_bound_observed"] or "")[:10])
    a("")
    a("## 5. 交付物清单")
    a("")
    a("| 文件 | 用途 | 接收方 | 可否进 gamma |")
    a("|---|---|---|---|")
    rows = [
        ("`source-lock.json`", "8 来源固定 revision、tree SHA、元数据声明", "编码官 / 审查官", "可"),
        ("`license-files.json`", "所有许可文件的 SHA256", "编码官 / 审查官", "可"),
        ("`per-file-ledger.csv`", "732 行逐文件 SHA256 + 头部许可信号", "审查官", "可"),
        ("`family-ledger.json`", "8 家族台账（含 oracle 哈希与 F2P 节点）", "编码官 / 审查官", "可（不含 gold 正文）"),
        ("`public-manifest.json`", "可公开的锁定清单（已剔除 oracle 断言）", "编码官（纳入 gamma）", "可"),
        ("`restricted-oracle.json`", "oracle 断言片段", "**仅 Q0/E0**", "**不可**"),
        ("`d0-time-isolation.json`", "时间隔离规则、窗口证据、缺口", "Mika / 审查官", "可"),
        ("`d0-shortfall.json`", "P0/P1 余量与缺口清单", "Mika / 编码官", "可"),
        ("`kaggle-23-d0-family-table.csv`", "8 家族一行摘要表", "编码官", "可"),
        ("`kaggle-23-d0-source-lock-report.md`", "本说明", "全部相关方", "可"),
        ("`validate_d0.py` 及其输出", "26 项验收校验，可复跑", "审查官", "可"),
    ]
    for f, u, r, g in rows:
        a("| %s | %s | %s | %s |" % (f, u, r, g))
    a("")
    a("复跑方式：`parse_refs.py` → `fetch_snapshots.ps1` → `collect_licenses.py` → "
      "`mine_families.py` → `build_ledger.py` → `build_extras.py` → `validate_d0.py`。"
      "后四个脚本不依赖网络，只用已固定的本地 checkout，因此哈希可独立复核。")
    a("")
    a("## 6. 检索方法与盲区")
    a("")
    a("**方法**")
    a("")
    a("- 官方/一手优先：全部结论来自上游仓库自身的 git 对象与 `LICENSE`/`NOTICE`/包装元数据，"
      "不依赖第三方许可汇总站。")
    a("- 关键数字交叉核实：许可既看根 `LICENSE` 正文，也看 `pyproject.toml`/`setup.cfg` 的 "
      "`license` 字段与 trove 分类器，两处并列记录。")
    a("- 家族候选用固定筛选规则挖掘并写进 JSON（有 issue 编号 + 同时改源码与测试 + "
      "单亲提交 + 改动 ≤200 行/≤10 文件 + 日期在训练窗口内），不是人工挑的。")
    a("")
    a("**盲区（查了但没得到，或没查）**")
    a("")
    a("1. **未运行任何测试**：无包索引，pytest 不可安装。所有 FAIL_TO_PASS 均为静态判定。")
    a("2. **未枚举 dev/sealed 的家族**：这两个 split 的仓库只做了 depth-1 快照，没有挖历史。"
      "P1 前需要一次 deepen 补做。")
    a("3. **上游 issue/PR 正文的许可未单独核查**：设计要求重写题文，本台账不搬运任何正文，"
      "因此不阻塞；但「重写后是否构成演绎」留给 Q0 抽查。")
    a("4. **镜像可达性不等于上游不可篡改**：本机直连 GitHub 不通，refs 与对象都经 "
      "ghfast.top 取得。缓解手段是 commit SHA + tree SHA 双重锚定；"
      "若需强保证，应在可直连的环境用相同 SHA 复验一次。")
    a("5. **官方四仓库与 V2 D/H 家族的排除证明不在本报告**：按设计由独立保管人出证明，"
      "本任务不读取 V2 留出正文/gold，也未自取 H 内容。")
    a("")
    a("---")
    a("")
    a("信息时效：所有 revision、日期与哈希截至 %s 有效；上游 tag 会继续增加，"
      "复跑前请确认固定 revision 未被 yank。" % ACCESS_DATE)

    with open(os.path.join(OUT, "kaggle-23-d0-source-lock-report.md"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("wrote report (%d lines) and family table" % len(L))


if __name__ == "__main__":
    main()
