# v7.1 → v7.2 审计传输完整性修订

基线为 v7.1 ZIP SHA256 `87f00239b837b5f4b447520f74e727d3925ba1b51191cee495f7d8500ed2804e`。本次只处理审查指出的传输丢失/结束确认缺失，不重做已关闭的 Git 身份阻断、固定 28 对象核对或 47 个既有用例。

## 改动

- bootstrap 为每个包装 Python 进程建立独立 TCP 流，发送连续序号的 `process_start`、`open` 与 `process_end`。每条流绑定 PID/role，发送失败会让包装进程非零退出。
- 父级按连接独立校验首事件、身份一致性、连续序号、帧格式、结束事件及 EOF。任一错误使 case 的 `audit_stream_complete` 为 false，case 失败，且不把观测空列表解释为零打开证明。
- runner 新增故障注入探针：在 `process_start` 后对连接发 RST；另丢弃结束确认。健康流仍要求 checker 日志清空后父级观察到 sentinel。
- `P7-synthetic-cases.json` 保存两类故障的原始父级状态和退出码，及健康流完整性摘要。

## 验证

固定 E0 为 `128d9b98b05ddf128c2e77b599e078de65675b8a`；仅使用合成 CPU 数据。47/47 用例通过，原合法退出码 0/7/4 保持。RST 用例子进程退出码 1 且流缺结束确认；缺结束确认用例子进程退出码 0，但父级完整性失败且 runner 拒绝 case。健康 sentinel 流首尾完整、无序号缺口。

## 覆盖边界

本机制证明 runner 包装的 Python 进程审计流在应用层完整，不是 OS 级穷尽审计。原生扩展、直接系统调用、Git 原生读取和未包装进程不在覆盖范围。真实 trust provider/receipt、权限隔离和 G1–G4 仍未验证。
