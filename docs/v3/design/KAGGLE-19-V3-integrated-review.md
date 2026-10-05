V3 联合设计与一次交叉审查结论 · v1.0 · 2026-10-05

本稿由 Mika 整合 KAGGLE-20/21/22，交付范围是可评审设计、资源预算和下游任务包。推荐单写者 Gemma 4、共享多任务 LoRA、受输出预算约束的搜索协议、独立外部题库与四臂评测。训练收益尚未验证；本稿不是GPU开工、购买资源或正式提交授权。V2配置、D/H、资源安排保持独立。

## 1. 输入与证据等级

三份原稿作为详细规范的组成部分，冲突处以本整合裁决为准：

| 输入 | 可审阅链接 | 下载文件SHA256 |
|---|---|---|
| 20：训练与架构 | [LoRA设计](https://multica.ai/api/attachments/01a10b82-9ad0-70b1-83e2-e1d62eac0cf9/download) | 75097e5d2f3f62ad23a3aa3fe3a1e32fbb284ecdefaa565b3aaa69e511e2cc22 |
| 21：数据与评测 | [数据规范](https://multica.ai/api/attachments/01a10b81-bba2-70b2-a07b-06eceffc8c39/download) | 6ffbe8486045142a9ff3b1947bcb8f14af87ba3d775c9313af28db13e8ed054c |
| 22：搜索与接口 | [搜索设计](https://multica.ai/api/attachments/01a10bb9-03e0-7455-8daf-9757971ab4d6/download) | ae451d1ec3b972a8127c75a4dc6b462310df45c00f72c11e1d0a769ed147e407 |
| 22：CPU试点 | [指标JSON](https://multica.ai/api/attachments/01a10bb9-04dc-7d19-94a2-02a0504af5f1/download) | 2cbbb55ef731d5b5ee9f590012bb4a96f641ff1ab93468541f22bc556b12670c |

本轮直接验证：四文件下载与哈希；指标JSON算术；上游 [Gemma4 LoRA映射修复 PR #38844](https://github.com/vllm-project/vllm/pull/38844) 已合并，处理 conditional wrapper 与 text-only 路径映射。该PR证明修复机制存在，不证明评分端已部署、不证明所有旧版必然静默失败，也不保证更新版全部训练/量化组合可用。

继承的调查证据：公开集127组(repo,base_commit)索引覆盖129题；两仓库图样本缺少文件/行号，嵌入工具按符号名字典解析；工具输出有5000字符上限。尚未独立重跑索引覆盖与官方工具调用，因此保留来源限定：撤回旧“62/129无图”作为设计前提，隐藏集覆盖仍未知；不能把两个图样本的schema观察当所有未来数据保证。

7题CPU结果复算为：锚点/词法/词法且排除测试候选的Hit@1=0/7、2/7、3/7，Hit@5=1/7、5/7、5/7。原稿名为FileRecall，实际定义是至少命中一个gold文件，应改名Hit@k；真正Recall@k=命中gold文件数/全部gold文件数，当前JSON不能重建它。6/7输出超过阈值来自行数×长度估算，不是实测官方截断日志。输出过多是有证据的风险，不能据此断言算法不是瓶颈、锚点无用或修复率提高。

## 2. 交叉审查裁决

| 争点 | 统一决定 | 理由/执行验收 |
|---|---|---|
| 单写者与强制analyzer冲突 | 首轮四臂统一单个LlmAgent、官方工具；不带强制分析员 | 拓扑固定后才能拆分搜索与LoRA贡献；V1/V2成绩只作历史参考 |
| S1成功才允许LoRA | CPU搜索实验与LoRA工程准备可并行；S1失败不取消用户要求的多任务LoRA | S1未过门槛时不宣称联合收益；仍可检验L-S0对base-S0的训练收益 |
| 首步预测仓库位置 | 首步可提出待验证假设/搜索动作；选定path:line必须附当时真实read/search证据 | 不训练“看到issue就捏造确定位置”；不将公开题gold或仓库答案记忆注入权重 |
| adapter作用域/名字 | 单个共享 `v3_policy`，用于定位、工具、补丁和恢复 | 不新增localizer_lora；训练按HF真实q_proj/o_proj，vLLM负责已验证映射，不手改成qkv_proj冒充兼容 |
| rank与KV | r16是参数提案，宿主实际max_loras/max_lora_rank必须记录 | 若宿主固定8×128，包里只有1×16不自动减少预分配；64倍是配置乘积，不是已测全卡节省倍率 |
| vLLM版本 | 候选验收线0.19.1或含等价修复的锁定版本，并做加载回归 | 不为训练擅改评分器；实际官方版本/mapper未知则标待验，不把“≥”当充分条件 |
| 测试与示例路径 | 测试作为检索/行为证据保留，编辑候选默认非测试优先；docs_src/examples不一刀切排除 | 不把0/129测试补丁推广成所有外部题的规律；搜索失败可放宽候选，禁止修改验收测试 |
| 索引有效性 | 官方图只描述base snapshot；本地微索引绑定tree hash，源码修改后失效重建 | 原稿“全任务有效/无失效”仅对未修改快照成立；旧图只作线索，编辑前重新读源码 |
| 训练窗 | 2048起步，替换21的4k初值 | 模板/截断配置必须升版；不截断JSON、tool回环或恢复必要反馈 |
| 变异任务比例 | 保留64训练家族24真实+40变异的题库提案；有效监督token的变异占比≤40% | 家族数比例与token配比不同；真实样本不足报告缺口，不复制轨迹凑数量 |
| 评测规模 | 首轮采用21的D8开发与H16封存四臂，统一命名 | 替代20的D12及32家族×2重复提案；较大确认实验另批，不冒称同等统计能力 |
| 公开集试点与隔离 | 22已有公开集分析降为已暴露探索信息，不作为V3独立验证集 | 独立保管人核对是否与V2 D/H重叠；不索要其正文/gold，不更换V2题目或追溯改分，必要时声明独立性限制 |

本稿完成一次设计层交叉审查；没有声称三位作者已互相复核或达成共识。执行缺项作为明确验收门槛，而不是“兼容已通过”。

## 3. 模型与训练规格

首选源：`google/gemma-4-31B-it-qat-w4a16-ct` @ `52f3f65bc7a02d555763bc923bd1d9094898219d`（20核查的公开HF revision，非已确认评分resolver）。冻结CT解量化BF16基座；部署仍为原官方W4A16+PEFT adapter。实际resolver不同须重新校验，不改路径伪装同源。

目标模块候选：`^model\.language_model\.layers\.\d+\.self_attn\.(q_proj|o_proj)$`，预计120个；r16/alpha32/dropout0.05/bias none；普通LoRA，冻结视觉、embedding、lm_head、norm；实际模块/shape全量导出后才允许训练。预计28,672,000可训练参数，BF16 adapter约54.69MiB，是算术估计。

首轮SFT：batch1、accum8、seq2048、lr5e-5、AdamW（FP32参数/两矩）、warmup10%、cosine、weight decay0、grad clip1、seed17；gradient checkpointing、use_cache false、packing false、workers0、流式数据、无CPU offload/并驻生成服务。SDPA及分块CE须保持Gemma softcap/shift/mask正确。训练最多32次optimizer update或2小时训练预算，先到即停；低于16步或200k输入token只报告工程可行性，不据无增益否定LoRA。

监督token配比：定位30%、工具/退出15%、补丁/验证30%、恢复25%；黄金辅助≤20%、变异≤40%、单仓库≤30%。比例冲突时减少可用规模并报告，不放宽许可/隔离门槛。失败动作和system/user/tool/padding/历史assistant均mask=-100，监督正确下一动作及合法结束标记；不监督长篇内部推理。

模板采用同revision的canonical Gemma4 template与gemma4 parser；20提供模板SHA `ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4`，执行者需下载复核完整tokenizer。至少20个fixture验证多tool call、失败恢复、Unicode、特殊标记与thinking配置；渲染与serving一致、无tool loss、目标动作loss>0。工具schema来自实际官方注册表，未知字段不得靠示例补齐。

资源估算：冻结dense参考58.25GiB，adapter训练态约0.43GiB，激活6–12GiB、workspace/碎片2–4GiB，总约67–75GiB，不保证80GB卡装得下。2k词表logits本身BF16约1GiB、FP32约2GiB。主机16GB限制必须按实测cgroup字节扣≥3GiB余量，目标RSS≤12GiB且取更严格值；禁止CPU构造58GiB state_dict。直接流式解量化加载；磁盘先核查权重/环境/最大快照/日志+15%余量，80GiB仅模型流程规划起点，不是全流水线保证。

内存后备NF4：同源CT先流式解量化到分片，再按NF4加载，登记中间SHA和量化配置；估计30–52GiB仍须profile，主机不能全量驻留。增加约63GB中间权重磁盘。CT路径失败后更换原IT源须独立版本和迁移验证，不自动替换。DPO需≥200合格偏好对后另审；在线奖励优化不进入首轮预算。

4h续训：3h45m停止新工作，保存adapter+optimizer/scheduler/scaler/RNG/数据游标/global step/accum边界及全部哈希；临时写入后校验并原子标记complete。仅保留小checkpoint，不保存完整基座。用短fixture比较连续与断点续训；accum中断回退完整步并记损失工作量。

## 4. 搜索执行协议与可提交性

同一单写者依次执行：提取字面/API/行为锚点→文件计数/不同词覆盖召回→非测试优先且保留docs/examples→读取函数区间确认→必要时升级。基础候选top10，详细阅读top1–2；一次输出目标≤1500字符，必须在工具5000字符上限内，计数/分页/路径过滤先于全文。行数上限不能保证字符上限，长行需长度控制并标truncated，不能静默丢关键证据。

规则排序：不同锚点覆盖降序，其次局部命中集中度，再按非测试优先，最后路径字节序打破平局；记录全部特征，不能利用gold。两轮查询无可信候选、读后行为不符或8次调用无chosen时升级：放宽词法范围→从测试import追源码→存在完整符号及工具时查图1–2跳并解析回当前path:line。缺图/空结果回到词法，不传自然语言给符号相似工具。不把无图邻居当“不可达代码”的证明。

搜索总调用目标≤22、最多2轮升级，首次有证据候选目标≤6调用；状态/剩余时间只能来自宿主可见返回，55%时间为软收敛提示。证据不足时允许明确失败/空补丁，不因预算强迫盲改。函数检索必须支持缩进方法、async def、嵌套和decorator；22示例 `^(class|def)` 会漏方法，执行实现须用合适模式或任务内AST。shell参数按实际语言安全转义，不直接执行issue插值。

统一LOCATE状态字段：schema_version、task_id、base_commit、tree_sha、anchors、queries、candidates[{path,symbol,start,end,source,observation_ref,evidence,confidence}]、chosen或null、next_action、calls_used、truncation_flags。路径必须当前源码存在，行号来自真实观察。训练窗口不得带未来chosen、gold或未发生的测试结果。

YAML instruction+官方工具+单adapter可表达上述软协议；不能声称实现强制状态机、自定义parser或新工具。任务内索引仅经允许的run_command在评分沙箱临时目录构建，绑定snapshot/tree，不写补丁目录；每题清空。官方静态图对编辑后代码可能过时。离线CPU实验代码是研发工具，不是可直接随提交运行的自定义入口；发布前用实际compiler验证载体。外部embedding服务、跨题答案缓存、扩大工具输出上限均不作提交前提。

## 5. 题库、权限与质量

采用21的四面契约：task.public、task.audit、oracle.private、trajectory，外加env.lock/run.protocol；不将oracle字段拼进actor输入。来源候选为click/more-itertools/pluggy/boltons训练，attrs/dateutil开发，packaging/marshmallow封存；均待固定commit和逐文件许可审查，不宣称已生成数据。真实issue原文许可单独核查，可依据复现症状重新撰写题文，代码开源不代表评论可训练。

P0先8训练家族（4真实+4变异，≥2仓库），oracle和环境8/8合格、最多16盲跑、≥4独立成功才提扩大。P1共96家族：train64/dev16/test16；训练目标64成功轨迹+16恢复视图，恢复视图不增加家族数，最多128次盲跑（含P0）。32个合格训练家族可提出缩小试训，但不是96题生产验收完成。

按仓库家族→问题/PR/backport/变异家族→时间先split，再生成与切窗。官方四仓库和V2所有D/H家族不进V3 train/dev；保管人独立核查0交集，只反馈证明和匿名缺口。22已接触公开题统计及7题指标，不能承担这部分题目的独立盲评资格；后续EXP-1移到许可外部训练家族，不继续全量公开题调排名。

broken应稳定失败、reference应通过、P2P无回归，各做两次干净对照；候选patch另两次复验及动作回放。F2P skip/collection error不算成功。首8全检、正式训练至少20独立样本抽检、恢复/gold-assisted/越界风险全检；许可、泄漏、篡改等重大错为0，发现即整批隔离。零错20例仅支持小试，不能声称总体质量已证明。源/tree/env/split/oracle/trace/export/protocol分别哈希；示例ID/空SHA禁止进入released。

## 6. 最小实验、统计与停止

CPU EXP-1：在新外部train家族冻结最多24题（实际16–24不足须报告）、至少两仓库；固定词法基线与预算协议，评估Hit@1/5、真正Recall@5、RegionHit、真实输出字节与CPU耗时。仅评估规则检索；不将它叫Gemma自主工具实验。晋级提案：Hit@1不降、Hit@5不降，截断率≤5%，且Hit@1提高≥10pp或输出p95降低≥50%；候选真实/变异及多文件分项无明显退化。未过则修S1，不取消LoRA定位/工具/修复/恢复路线。

T1工程门槛：120模块或已解释的正确映射、基座冻结、20mask fixtures、非零梯度/有限loss、完整断点续训、原官方CT加载zero/nonzero adapter；零adapter与base漂移检查，非零adapter在8训练fixture≥7/8目标动作/格式，adapter被实际路由且参数变化可见。加载成功不等于有效。长上下文与TP4需要正式设备等价验证；不能用“KV tokens≥20000”替代可用上下文/并发和8k/16k请求实测。

四臂固定单写者、W0、采样、预算和干净workspace：A=W0/S0，B=W0/S1，C=W0+L1/S0，D=W0+L1/S1。同一个L1同时用于C/D；S0为冻结原搜索内容移入同骨架，完整SHA执行前登记，不拿V1/V2历史分直接当A。交互=(D-C)-(B-A)。开发D8（4真实+4变异）×4臂×1seed；封存H16×4臂×1seed。每题4min生成、100工具/100turn为上限，加载/setup/验证另入壁钟。顺序预注册平衡，任何结果出现后不能删题、缩分母或更换oracle。

主指标完整分母resolved，另报定位Hit/Recall/Region、首次可信阅读calls/秒/token、工具解析/重复/OOM/超时率、阶段总耗时median/p95、资源峰值；symbol标签缺失记NA。D8扩大门槛采用21：D-A≥1/8且D-B≥0，真实题不净降、无新增严重故障；或质量持平、完整时耗中位比≤0.85且定位calls≤0.80，只称效率信号。另报C-A/D-B，LoRA增量不正则不宣称训练提高解决率。

H16唯一预注册确认性对比D-A：净增≥2/16、真实题不净降、单侧精确符号检验p≤0.05、无新增严重故障；其余主效应/交互探索性。2胜0负p=.25不足；5胜0负p=.03125仍只支持这批家族，不能外推所有仓库或证明LoRA单独贡献。H16单重复/两仓库不支持稳定泛化承诺。较大新封存集+第二训练seed/推理重复另审；不将H变dev后继续声称独立。

统一停止：权限/许可/隔离不合格、mask错、基座更新、adapter无效、NaN、超过内存或预算、测试被篡改。兼容最多一次有记录修复、内存最多一次NF4降级且在原预算内；不足则交精确阻断，不续开作业。S1失败回S0；LoRA失败保留可提交冻结模型搜索方案，但仍报告训练主线未成功，不用prompt结果替代训练验收。

## 7. 统一成本与阶段释放

以下是设计提案，已批准GPU额度为0，本轮GPU实耗0。全部107操作仅Liang运行时，单GPU串行、每作业≤4h、避开V2；时间包含加载/失败尝试/保存/验证占卡。硬件需求不是资源可用证明。

| 阶段 | 单A100上限 | 交付与前置条件 |
|---|---:|---|
| CPU准备/T0/EXP-1 | 0 GPU-h | source/split/依赖/schema/mask/搜索输出验证，模型或教师运行不在CPU完成声明中 |
| P0八题轨迹 | 2h | CPU八题oracle合格；最多16尝试，完整成本超限即停止 |
| P1扩大数据 | 增量10h | P0通过后扩96家族；含余112尝试/恢复，不能保证64成功 |
| T1 profile+训练+重载 | 4h | 合格export、软件锁/内存计划；替换21的2h占位，包含训练smoke和加载 |
| D8四臂 | 4h | 冻结A/B/C/D与完整run耗时预测；32run |
| H16四臂 | 8h | D门槛、盲评权限、完整冻结；64run，2片各≤4h |
| 首轮合计 | **28h** | 数据12+训练4+dev4+封存8；分阶段释放，非一次执行许可 |

20的8h=T1 4+dev4，已计入，不再加；14h冷启动含另一套6h最低生产假设，现被P0/P1统一替代；21的2h训练占位删除，不再加；20的32家族×2重复H3不在首轮。最早GPU工程检查可提P0 2+T1 4=6h上限，但P0只有8题时T1只作工程smoke，不能越级称完整泛化实验。

CPU题库上限参考24设备小时/96 core-hour；另T0与搜索实现提议8设备小时/32 core-hour，总CPU准备提案≤32设备小时，均串行4核内。21的21–35人时加20的T0 2–4人时及编码1–2人日不是同一种计量；下游需排班，不能与GPU时间相加。每阶段记录下载/磁盘/人员/CPU/GPU及每独立成功家族成本。

**28h不含正式4×L4兼容验收及Kaggle正式提交**。当前没有获批等价L4资源，不能报价为0；以4×L4设备数×实际壁钟另列Kaggle配额/占用，先提出只读兼容实验方案与具体版本再审批。A100可做功能验证，但不能释放正式提交门槛。若L4只能通过正式提交观察，必须交具体包让用户确认，不能伪装smoke提交。

## 8. 可直接派发的下游任务包（本轮仅交方案，不触发执行）

已核实以下智能体存在；名字仅作为责任建议，不是本轮派工/已启动声明。所有包high优先级，共同输入本稿与三份引用规范，结果交独立执行issue附件/代码PR并回报Mika；不让研究顾问重做相同调查。

| 包/负责人 | 范围与产物 | 验收/依赖 |
|---|---|---|
| D0 数据来源台账：资料调研与分发 | 固定8候选源commit、许可/NOTICE、家族谱系、shortfall与公开面manifest | 逐来源权限、固定SHA；独立保管人完成V2排除证明，不能自取H内容；先于数据生产 |
| E0 CPU工程：需求交付编码官 | T0依赖锁、流式loader设计、20 mask fixtures、四面schema exporter、8题环境对照、同骨架S0/S1与CPU EXP-1实现 | 本稿全部可执行静态门槛，方法/async/长行/index失效测试；禁止GPU和107连接；依赖D0合格输入，可先做合成fixtures |
| Q0 独立审核：代码审查官 | 审E0、mask/gold隔离/权限分离、预算manifest与可提交载体；custodian匿名交集检查 | 发布重大错0；封存gold不进入训练/搜索作者权限域；缺权限无法隔离则报阻断，不自称独立 |
| G0 资源统筹：Liang运行时总管 | P0/P1/T1具体分片、V2无冲突排期与前台结果收集；训练代码由编码官实现 | 依赖D0/E0/Q0通过及逐阶段GPU预算批准；无权自动扩大资源/重试 |
| V0 运行与评测：Kaggle竞赛管理员 | 官方resolver/compiler/parser/adapter route证据；D8/H16执行与汇总；4×L4兼容方案 | 实际软件/模型hash、四臂完整分母、独立oracle由Q0保管；训练执行与H内容隔离；正式提交需具体版本确认 |

同一编码官的数据/训练工程可顺序分支开展，避免并发修改同文件。Mika负责整合验收、阶段预算建议及向yg123456/skyyangyyds汇报；权限安排未落实前不得发布封存集。当前验收的是研发设计：全部训练/数据生产/设备兼容与比赛收益列为待执行。本次无共享代码修改，无需PR。
