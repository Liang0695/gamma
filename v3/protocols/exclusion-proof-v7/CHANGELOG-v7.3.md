# v7.2 → v7.3：预期进程集合与有限期限

本次只修 v7.2 独审指出的两项缺口，保留已通过的 RST、坏帧、截断帧、缺结束确认、Git 身份计算前阻断与合法 0/7/4 回归。

## 预期进程集合

- Runner 在启动前为固定的 47-case 合成集生成 case/round 计划：每例一个 checker；需要 package origin probe 的例子预期一个 probe 和三个消费轮；两个合成消费者例子预期三个消费轮。
- 收集完成后精确核角色数、PID 唯一性、root checker PID/runner PPID，以及子进程 PID/checker PPID。缺失、额外、错角色、错父子关系或不完整流都拒绝 case。
- 带 `AUTH_BEFORE_ACCESS` 或 `COMPUTE_NOT_STARTED` 的前置拒绝仍用较小的 checker-only 计划，并额外断言 checker 计算数与外层 consumer 数均为 0，不应用健康 4-stream 计划。
- Bootstrap 缺少 `K27_EXTERNAL_AUDIT_ENDPOINT` 时在执行目标前返回 `AUDIT_ENDPOINT_REQUIRED`/78。

## Deadline 与回收

- 常规 case 45 秒；每个已连接包装进程/流 30 秒；握手 3 秒；case 返回后的有限排空 3 秒、静默确认 250ms。
- 超时会记录 `CASE_TIMEOUT` 或 `PROCESS_TIMEOUT`、进程退出码及耗时。Runner 只终止本次 Popen PID 与该 collector 观察到的进程，随后有限等待并关闭管道/连接。
- Timeout probes 覆盖单进程流期限和整 case 期限；missing-child probe 证明一个完整 checker 流不能冒充期望的 consumer 流。另测额外角色、错角色和缺 endpoint。

## 实测

固定 E0 `128d9b98b05ddf128c2e77b599e078de65675b8a`，合成 CPU 数据：47/47 cases 通过；健康 checker + package probe/consumer 角色计划符合 PID/PPID；合法 consumer 退出码 0/7/4。缺角色、额外角色、错角色、RST、坏帧、截断帧、缺结束确认、缺 endpoint、PROCESS_TIMEOUT 和 CASE_TIMEOUT 反例均明确拒绝；两个 timeout probe 子进程在各自期限内退出并回收。

审计仍只覆盖 runner bootstrap 包装的 Python 进程和 Python `open` hook，不声称 OS 级穷尽审计。真实 trust provider/receipt、隔离与 G1–G4 未验证。
