## KAGGLE-24 G2 字段对齐表

复核输入：KAGGLE-27 附件 `01a11242-bdc5-78ff-a522-4464369cd80f`，文件名 `KAGGLE-27-protocol-v4-and-verifier.zip`，SHA-256 `79d329f5dd50deb4d799d90659061b2717064cd285ed9c2180f90d9cb12ead72`。协议候选版本为 `v3-v2-exclusion-proof-protocol-4`，执行 manifest 候选版本为 `v3-v2-exclusion-proof-execution-manifest-3`。

**版本边界：** KAGGLE-27 v4 包仍是待独立复核候选，本表不把它称为已审结论或通行凭据。KAGGLE-27 上一个已交付复核包是 v3（ZIP SHA-256 `e8459db305c395b85244fa2f4979436f5ea47b30a4fa8ac73dce1ac423055113`）。v3 中 `counts_declared` 可选；v4 候选将其改为三数组必需。本 issue 最新整改要求补独立分母并阻止同步缩小，因此本实现采用 v4 候选的必需计数契约，但 real-input 仍完全 fail-closed，等待协议独立复核和保管侧可信完整性基线。

| 主题 | v4 候选字段/规则 | E0 本地修改 | 证据与限制 |
|---|---|---|---|
| Denylist schema | `denylist_schema_version=v2-dh-exclusion-denylist-1`；必需 `d_family_ids/h_family_ids/exposed_family_ids/counts_declared`；只接受指定 optional 字段；未知键整批拒绝 | `v3/data/dedup.py::validate_denylist_payload` 严格执行；`bool` 不作为整数；`None` 与合法字面量 `none` 分开 | 固定公开 fixture 覆盖 null/类型/空白/版本/未知键/空并集 |
| Count fields | 每数组 raw/effective/duplicate；union raw = effective + duplicate；跨数组重复可见；source map 多值 | 每 token 输出所有来源数组；为避免传播 family ID，只输出 token SHA-256 与 source-array 列表；跨数组 duplicate 独立计数 | 公开合成数据验证；真实来源计数无权读取 |
| 独立分母 | v4 `denominator_baseline.baseline_ref` 指向 `v2-dh-custody-frozen-baseline-1`，当前 `file_bytes_sha256=null/status=PENDING_CUSTODIAN`；公开期望口径含 V2 task 129、family 126、D8 8、H24 24、exposed union 17，另有 V3 D8/H16 8/16 | 仅给 checked-in 合成 fixture 使用独立 SHA/计数基线；真实候选与 denylist 在 baseline 就绪前拒绝运行 | 不把协议列出的公共声明数误当成保管侧实物 SHA 或真实输入授权 |
| 授权顺序 | 先读取可信授权/非敏感协议，再验证清单之外的信任锚、授权主体和对象 scope；授权失败时受保护输入零读取 | 当前没有 trusted approval provider；真实模式在解析 candidate/denylist/授权文件之前返回 `AUTH_BEFORE_ACCESS`，且不打开调用者给的授权或证据路径 | 自写 `authorized` + 自算摘要反例实测 exit 7；不存在的 candidate/denylist 哨兵未被访问。fixture 入口仅允许固定 case 名，拒绝路径/摘要/授权 override |
| Execution manifest | schema `v3-v2-exclusion-proof-execution-manifest-3`；绑定实际 `python -m v3.cli split` 与完整本地导入闭包，逐文件 bytes SHA / length / newline / Git blob / commit | 将 `__main__` 映射为真实 `v3/cli.py`；绑定所有 CLI 导入的 `v3/` Python 文件，并在代码 SHA 偏离固定 map 时 fail closed | 当前包未处于包含这些文件的 Git commit：每项明确记 `git_blob_sha1=null` / Git binding incomplete；不能作为 Git SHA 级 G2 验收 |
| Same-buffer | v4 要求单次读取，摘要与解析使用同一 buffer，并记录 `bytes_hashed == bytes_parsed` | baseline、candidate、denylist 分别一次读入 bytes；使用同一 bytes 做 SHA 与 JSON decode；manifest 记录字节长度与两项处理长度 | fixture 输出逐项相等；执行模块在导入后读取哈希，真实模式仍拒绝，不形成授权结论 |
| Exit semantics | 生产 exit 0 只表示机器 ACCEPTED；测试模式需有明确预期，不可从被检清单 overall 自取期望 | production CLI 无可信授权环境时始终 exit 7；仅 `--fixture-case {clean,missing,intersection,invalid}` 执行固定哈希 fixture，clean=0、输入缺失/无效=4、真交集=7 | 所有 fixture 输出均标 `COMPUTATION_ONLY / NOT_ACCEPTED`，不等同真实生产验收 |

### 仍 pending

v4 独立评审；保管侧 real-input baseline 的受控字节与分母；可信授权主体/对象范围/证据内容校验提供器；真实 Git commit/blob 对应新 E0 固定 SHA；G1 命名空间、G3 snapshot、G4 OS 隔离。没有这些条件，代码不会读取真实候选或 denylist，也不会产出 `ACCEPTED`。
