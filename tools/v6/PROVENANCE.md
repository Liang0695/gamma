# `tools/v6/` —— KAGGLE-32 v6 监督器原件（只读复用，逐字节核对）

这个目录**不是**本任务写的代码，是 KAGGLE-32 v6 交付包的原件副本，随包放置的目的是让
`python -m v3.train.supervised_check` 与 `tests/test_g38_supervised_check.py`
在没有外部目录时也能跑起来（外加让复核者能对着同一份字节核对复用关系）。

## 出处

- 来源附件：KAGGLE-32 v6 预检包 `01a118ad-75ee-7f37-9363-c8dd4bdbe9bb`
  （整包 SHA-256 `0151b35ad92c36985d1632d37b9b06e9513c64e3afd4c4c8d26f889dfd1676c9`）
- 复制时间：2026-10-08（KAGGLE-38 r5 交付）

## 文件与 SHA-256（复制后逐一实算，与来源一致）

| 文件 | SHA-256 | 必需 |
|---|---|---|
| `shard_supervisor.py` | `a3f96572a14cf98965c739d85bd5ecee318ace912fd0339453036a877e6420e5` | 是 |
| `shard_job.py` | `63ee2531757e3a34d70660ca257171dcd3b23f893cbddd8e0137c68215f9cd33` | 是 |
| `shard_guard.py` | `f459321fde54e0fa660eba7d63d3f8a1fc031afacffe4fc58fc7512f4bb46845` | 是 |
| `stage_scripts.json` | `7c1a7b6009ee2e82f61820060d53a92fc91cbf1e33e2f6013c63eb85e9e1bfd2` | 是 |
| `deadline_wrapper.py` | `5c515706ad2e1f52bafc6e7d10c1b39343c56a908f72b09940b8355139726f2c` | 否（存在即核对） |

## 使用规则

1. `v3/train/supervised_check.py::verify_v6_package()` 在**每次**启动监督器前逐字节核对上表；
   缺文件或哈希不符 → `v6_package_tampered` / `v6_supervisor_missing`，**不降级、不回退到本地改写版**。
2. 本目录**只读**：本任务没有、也不会修改其中任何一个字节（改了会立刻被上一条抓住）。
3. v6 自身的已知边界照原样继承，不因放进本包而"升级"为已验证：
   POSIX 进程组回收已实现但**未在 Windows 上验证**。

## 为什么不直接拷进 `v3/` 包内

复用关系需要**可核对**：留在 `tools/v6/` 并带出处与哈希，复核者一眼能看出
「这是 v6 原件」还是「这是长得像 v6 的本地实现」；混进 `v3/train/` 会让这条界线变模糊。
