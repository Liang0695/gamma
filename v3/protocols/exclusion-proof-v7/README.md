# KAGGLE-27 协议 v7.3 定点修订包

## 这份代码做什么

在 v7.2 的审计流完整性基础上，增加预期进程角色/父子关系核验和有限执行期限。缺少审计 endpoint 时 bootstrap 在执行 payload 前以退出码 78 拒绝。

## 文件清单

- `verify_manifest.py`：协议 v7 checker；Git blob、固定 commit:path 或换行映射错误会在执行快照和消费者启动前阻断。
- `run_synthetic_cases.py`：合成用例 runner、逐 case/round 预期角色计划、父进程 TCP 收集器、期限控制、进程回收及审计反例。
- `external_audit_bootstrap.py`：Python 进程包装入口，发送带连续序号的启动、打开和结束事件；无 endpoint 或传输失败时拒绝/失败。
- `make_synthetic_cases.py`：基于固定 E0 版本生成合成 fixtures，合成来源显式标为 `not_git`。
- `exclusion-proof-protocol-v7.json`、`exclusion-proof-execution-manifest.template.v6.json`：冻结协议和 manifest 模板。
- `P7-synthetic-cases.json`：47 个合成用例、角色集合与父子关系核验、传输及 timeout 回归的实测报告。
- `P1-validator-contract-v7.json`、`p1_validator_contract_v3.py`：E0 接口与退出码检查器及结果。
- `frozen-denominator-baseline.json`、`synthetic/`：合成基线与运行时生成的合成数据。
- `preserved/`：v1 至 v6 对照文件。
- `CHANGELOG-v7.1.md`、`CHANGELOG-v7.2.md`、`CHANGELOG-v7.3.md`：版本差异、反例和覆盖边界。
- `E0-alignment-v7.md`：固定 E0 执行闭包、字段和 receipt 对齐说明。
- `MANIFEST-sha256.txt`：逐文件 SHA256 清单，不含清单文件自身。

fixture generator 会生成 `fixtures.json` 和 `synthetic/`；这些运行文件不随源码包分发，以免嵌入机器绝对路径。

## 环境与依赖

Python 3.10 或更新版本；仅标准库。另需 Git，以及可解析固定提交 `128d9b98b05ddf128c2e77b599e078de65675b8a` 的只读 gamma worktree。

## 怎么运行

在包目录执行以下命令。示例假设只读 worktree 位于 `../gamma/gamma-fixed-src`；按实际位置调整该路径：

```powershell
python -m py_compile verify_manifest.py run_synthetic_cases.py external_audit_bootstrap.py make_synthetic_cases.py
python -X utf8 make_synthetic_cases.py --e0src ../gamma/gamma-fixed-src
python -X utf8 run_synthetic_cases.py --e0src ../gamma/gamma-fixed-src
```

输入是合成 fixture；固定 E0 仅用于版本对象和执行闭包对齐。生成器写入 `fixtures.json` 和 `synthetic/`，runner 写入 `P7-synthetic-cases.json` 并清理临时工作目录。

## 输出说明

- `fixtures.json` 列出 47 个合成正反例、目标 reason code 和验证器闭包。
- `P7-synthetic-cases.json` 保存各例的 verdict、reason code、consumer 计数、父级观察到的进程启动和目标文件打开事件，以及每条 TCP 事件流的完整性。
- 父级不读取 checker 的 `all_open_events`。每个进程流从序号 0 开始，必须有连续序号、`process_end` 和 EOF。断连、坏帧、序号缺口或缺结束确认都会拒绝该 case，不能作为“零打开已证明”。
- runner 在启动前冻结各 case/round 的角色数，再校验实际角色、PID 唯一性和 PID/PPID 链。缺流、额外流或错角色会拒绝 case；前置拒绝 case 使用仅 checker 的计划，并单独要求 compute/consumer 均为 0。
- deadline：case 45 秒、每个包装进程/连接 30 秒、连接握手 3 秒、case 结束后有限排空 3 秒。超时会记录原因和退出码，并回收本 case 启动的进程树。
- `MANIFEST-sha256.txt` 校验源码包内文件。修改包文件后需重算清单。

## 实测结果

2026-10-07，在固定 E0 `128d9b98b05ddf128c2e77b599e078de65675b8a` 的只读 worktree 上，以 CPU 合成数据实测：

- `py_compile`：checker、runner、bootstrap、fixture generator 成功。
- fixture generator：47 cases；执行闭包 28 个文件；固定 E0 对象身份 28/28 匹配，其中 7 个字节相同、21 个 CRLF→LF 映射。
- runner：47/47 通过，0 失败；合法合成消费者退出码为 0/7/4。健康 case 记录 checker + acceptance/diagnostic/bad_input 三个 consumer，带正确 PID/PPID。
- Git 身份错误的 3 个反例均命中 `GIT_BLOB_INVALID` 与 `COMPUTE_NOT_STARTED`，checker consumer 与父级观察 consumer 启动均为 0。
- 健康审计探针清空模拟 checker 日志后，父级仍收到合成 sentinel `open`；进程事件序号连续，结束码为 0 且收到 EOF。
- RST 回归：父级收到 `process_start` 后复位连接；子进程退出码 1，父级明确记录 `missing_process_end`，case 被拒绝。
- 缺结束确认回归：子进程退出码虽为 0，父级报告 `injected_missing_end_confirmation` 和 `missing_process_end`，case 仍被拒绝；两种故障都不接受“零打开已证明”。
- 缺子进程回归：完整 checker 流无法满足预期 consumer；case 以 `MISSING_EXPECTED_ROLE` 拒绝。额外角色和错角色分别以 `UNEXPECTED_ROLE` 及缺失角色拒绝。
- endpoint 缺失回归：退出码 78，`AUDIT_ENDPOINT_REQUIRED`，payload 哨兵未创建。
- process deadline 回归：挂起 wrapper 在约 0.7 秒被终止，记录 `PROCESS_TIMEOUT`、非零退出码和缺结束流；case deadline 回归在约 0.5 秒被终止并记录 `CASE_TIMEOUT`。本 case 进程均已回收。
- 坏帧和截断帧回归仍拒绝：分别记录 `invalid_frame` 和 `truncated_frame`。

## 假设与未验证项

- 审计覆盖限于 runner 通过 bootstrap 启动且建立 TCP 连接的 Python 进程及 Python `open` audit hook。它不是 OS 级穷尽审计；原生扩展、直接系统调用、Git 原生内部读取和未包装进程不在范围内。
- 进程启动事件由 runner 包装路径生成，不能证明拦截任意恶意原生执行。独立进程流完整性不等同于操作系统级文件访问证明。
- 45 秒 case deadline、30 秒进程/连接 deadline、3 秒握手/排空界限是这套合成 runner 的上限配置，不代表评分端或外部生产作业的期限。
- 真实 trust provider、receipt、访问隔离及 G1–G4 未验证。无 provider 时生产流程继续 fail-closed。
- 本轮只运行合成 CPU 用例；未读取真实受限数据、使用 GPU、修改 E0、正式提交 Kaggle 或推送仓库。
