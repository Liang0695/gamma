# v7 → v7.1 定点修订

基线为 v7 ZIP SHA256 `42cace396c762a67df52632fa6ab8a28ae5ff13edd2e11bb43d3843a9357799f`。协议 schema 仍为 v7；本次只修复 Mika 指定的一个阻断和一个应修项。

| 审查项 | v7 问题 | v7.1 修订 | 实测证据 |
|---|---|---|---|
| 固定 Git 对象身份计算前 gate | `GIT_BLOB_INVALID` 未计入 `pre_compute_blocking`；manifest 可报错后仍运行 3 个 consumer。 | 将 `GIT_BLOB_INVALID`、执行快照摘要差异和输入快照字节差异纳入计算前阻断集合；失败返回 `COMPUTE_NOT_STARTED`。 | `neg_git_blob_zero_placeholder`、`neg_git_commit_identity_mismatch`、`neg_git_newline_mapping_mismatch` 分别保留目标错误；3 例均 checker consumer `0`、父级 consumer `process_start` `0`。 |
| 外层打开审计独立性 | runner 从 checker 自报 `all_open_events` 统计所谓外部打开数。 | runner 启动父进程 TCP 收集器；bootstrap 在 checker、package probe 与三类 Python consumer 启动时发送 `process_start`，并将 Python `open` 路径事件发送至父进程。外层统计不读取 checker 报告。 | `external_audit_self_test` 清空模拟 checker 日志后，由独立子进程打开合成 sentinel；父进程仍收到对应 open 事件。P7 报告保存相关原始事件和进程关系。 |

## 覆盖边界

父级收集的是 Python audit hook 事件，不是 ETW、Procmon 或系统调用级 trace。它覆盖这套 runner 明确包装的 Python 进程，包括主 checker、包来源 probe 和 consumer 子进程；不声称覆盖任意原生扩展、直接系统调用、未包装子进程，或 Git 原生进程读取其对象数据库。报告将现有 checker 内部 hook 作为内部证据保留；父级计数独立于它。

## 固定输入及保留结果

- 固定 E0 commit：`128d9b98b05ddf128c2e77b599e078de65675b8a`；28 个 commit:path/blob 均匹配，7 个字节相同、21 个 CRLF→LF 映射。
- 固定 legacy 负向控制仍为 `ffc37b4f3ecde6579117b0012ab4f69b4cff16ab`。
- 合成 CPU 回归：47/47 通过；合法合成 consumer 退出码为 0/7/4。
- 无真实 trust provider 时生产 fail-closed 结果保持。没有修改 E0、共享基础、权限、GPU 资源或正式提交路径。
