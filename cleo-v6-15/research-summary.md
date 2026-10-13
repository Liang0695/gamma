# Cleo 关闭身份与固定版本资料说明

## 0. 一句话结论

截至2026-10-13，本次三批补查30个此前仅索引列出的issue的关闭事件；12个绑定11个不同同库PR，10个PR新增固定父/tree/LICENSE证据，PR247复用原证据；完整题包与训练资格增量均0。

## 1. 需求回顾

接收方为49任务的既有评审/62汇总链及Mika。依据49当前“62新滚动窗口”授权，前台最多三批、每批目标10、单请求不超过30秒，方向为未核Cleo元数据。普通新增100目标累计15，至少缺85。本次交付一次性不可变资料附件，不修改共享表、不执行候选。

## 2. 事实与证据

**事实：** 使用已交v6.12 `index-response.json`中尚未请求关闭事件的身份，未重发索引请求。三批分别核10个不同对象；不是30个新行为题，也不是新题文授权。来源输入成员SHA/长度见`input-bindings.json`，与原包SHA清单核对见`verification.json`。访问日期2026-10-13。

**事实：** 实际9次GitHub GraphQL请求，全部CLI退出0且无GraphQL errors；每次30秒超时上限、HTTP前独占attempt占位，自动重试0。请求/响应元数据和实测wall见`requests.json`与各批文件。API未请求title/body、评论、PR正文、patch/diff、commit message、源码快照或外库内容。来源为GitHub GraphQL实际响应；本轮未访问网页正文。访问日期2026-10-13。

| 完全限定身份及来源 | 本次同库PR关闭关联 | 证据状态 |
|---|---|---|
| [python-poetry/cleo#26](https://github.com/python-poetry/cleo/issues/26) | 未取得同库PR closer | 不自动重查空closer |
| [python-poetry/cleo#372](https://github.com/python-poetry/cleo/issues/372) | 未取得同库PR closer | 不自动重查空closer |
| [python-poetry/cleo#363](https://github.com/python-poetry/cleo/issues/363) | 未取得同库PR closer | 不自动重查空closer |
| [python-poetry/cleo#362](https://github.com/python-poetry/cleo/issues/362) | 未取得同库PR closer | 不自动重查空closer |
| [python-poetry/cleo#104](https://github.com/python-poetry/cleo/issues/104) | [PR313](https://github.com/python-poetry/cleo/pull/313) | 已绑定父/tree/LICENSE；broken未知 |
| [python-poetry/cleo#325](https://github.com/python-poetry/cleo/issues/325) | [PR327](https://github.com/python-poetry/cleo/pull/327) | 已绑定父/tree/LICENSE；broken未知 |
| [python-poetry/cleo#314](https://github.com/python-poetry/cleo/issues/314) | [PR321](https://github.com/python-poetry/cleo/pull/321) | 已绑定父/tree/LICENSE；broken未知 |
| [python-poetry/cleo#295](https://github.com/python-poetry/cleo/issues/295) | 未取得同库PR closer | 不自动重查空closer |
| [python-poetry/cleo#289](https://github.com/python-poetry/cleo/issues/289) | [PR291](https://github.com/python-poetry/cleo/pull/291) | 已绑定父/tree/LICENSE；broken未知 |
| [python-poetry/cleo#248](https://github.com/python-poetry/cleo/issues/248) | 未取得同库PR closer | 不自动重查空closer |
| [python-poetry/cleo#284](https://github.com/python-poetry/cleo/issues/284) | [PR285](https://github.com/python-poetry/cleo/pull/285) | 已绑定父/tree/LICENSE；broken未知 |
| [python-poetry/cleo#162](https://github.com/python-poetry/cleo/issues/162) | [PR179](https://github.com/python-poetry/cleo/pull/179) | 已绑定父/tree/LICENSE；broken未知 |
| [python-poetry/cleo#279](https://github.com/python-poetry/cleo/issues/279) | 未取得同库PR closer | 不自动重查空closer |
| [python-poetry/cleo#74](https://github.com/python-poetry/cleo/issues/74) | [PR269](https://github.com/python-poetry/cleo/pull/269) | 已绑定父/tree/LICENSE；broken未知 |
| [python-poetry/cleo#69](https://github.com/python-poetry/cleo/issues/69) | 未取得同库PR closer | 不自动重查空closer |
| [python-poetry/cleo#232](https://github.com/python-poetry/cleo/issues/232) | [PR245](https://github.com/python-poetry/cleo/pull/245) | 已绑定父/tree/LICENSE；broken未知 |
| [python-poetry/cleo#255](https://github.com/python-poetry/cleo/issues/255) | 未取得同库PR closer | 不自动重查空closer |
| [python-poetry/cleo#256](https://github.com/python-poetry/cleo/issues/256) | 未取得同库PR closer | 不自动重查空closer |
| [python-poetry/cleo#127](https://github.com/python-poetry/cleo/issues/127) | 未取得同库PR closer | 不自动重查空closer |
| [python-poetry/cleo#63](https://github.com/python-poetry/cleo/issues/63) | 未取得同库PR closer | 不自动重查空closer |
| [python-poetry/cleo#133](https://github.com/python-poetry/cleo/issues/133) | [PR247](https://github.com/python-poetry/cleo/pull/247) | 已绑定父/tree/LICENSE；broken未知 |
| [python-poetry/cleo#54](https://github.com/python-poetry/cleo/issues/54) | 未取得同库PR closer | 不自动重查空closer |
| [python-poetry/cleo#46](https://github.com/python-poetry/cleo/issues/46) | 未取得同库PR closer | 不自动重查空closer |
| [python-poetry/cleo#108](https://github.com/python-poetry/cleo/issues/108) | 未取得同库PR closer | 不自动重查空closer |
| [python-poetry/cleo#182](https://github.com/python-poetry/cleo/issues/182) | 未取得同库PR closer | 不自动重查空closer |
| [python-poetry/cleo#64](https://github.com/python-poetry/cleo/issues/64) | [PR239](https://github.com/python-poetry/cleo/pull/239) | 已绑定父/tree/LICENSE；broken未知 |
| [python-poetry/cleo#65](https://github.com/python-poetry/cleo/issues/65) | [PR239](https://github.com/python-poetry/cleo/pull/239) | 已绑定父/tree/LICENSE；broken未知 |
| [python-poetry/cleo#87](https://github.com/python-poetry/cleo/issues/87) | 未取得同库PR closer | 不自动重查空closer |
| [python-poetry/cleo#246](https://github.com/python-poetry/cleo/issues/246) | 未取得同库PR closer | 不自动重查空closer |
| [python-poetry/cleo#56](https://github.com/python-poetry/cleo/issues/56) | [PR236](https://github.com/python-poetry/cleo/pull/236) | 已绑定父/tree/LICENSE；broken未知 |

**事实：** #64与#65共同关联PR239；#133关联PR247，与既有#437早期关闭记录共用同一落地/第一父/tree。#437后续空closer及最终修复仍不确定，不重新查询，不据共享事件签同root或多个独立root。来源：[PR239](https://github.com/python-poetry/cleo/pull/239)、[PR247](https://github.com/python-poetry/cleo/pull/247)的API元数据及v6.12绑定，访问日期2026-10-13。`family-relations.json`保留各身份。

**事实：** 合格事件的mergedAt早于2026-01-01，第一父/tree是版本身份证据，未证broken。每个绑定的LICENSE按UTF-8字节长度/SHA256与Git blob OID核对；PR247许可复用原包字节，其余10个按固定第一父API采集。逐pin许可URL见`license-bindings.json`；访问日期2026-10-13。源码许可证据不代替issue文本训练用途授权。

## 3. 推断与建议

**推断：** 有同库关闭PR可作为下一中性研究用途评审线索，但不证明修复了该问题，不证明行为合同完整或可执行。共享事件应先保留关系而非按成员数计root。

**建议：** 由既有Mika/62链评审这12个具名issue→PR关联的有限研究范围；获得具体范围后才可初始说明预筛/中性摘要，行为充分后再按批准固定父采源码。原49负责人不从metadata权限自动激活这些步骤。18个无同库PR closer本次收口，不反复轮询。

## 4. 冲突与不确定项

**不确定：** 行为合同、题文用途、broken、可信验收、完整家族/root与隐藏/dev/sealed冲突覆盖均未知。空closer不等于无修复。仅所取v15有限字段/旧详细身份并集检查，空匹配不签全局NO_MATCH。

五v6.14 marker停止341/84/231/366/359及454保持排除，446合同缺项保持，不重取、不改写、不材料化PR358组。旧dotenv待审/隔离、510历史UNKNOWN与NOT_RUN保持。

关键身份只有GitHub这一上游来源；缓存与本次响应不是两个独立来源，Git对象哈希校验也是完整性证据而非独立来源交叉核实。受本次范围限制，未开展外部独立源或题文核验。

## 5. 交付物清单

- `manifest.json` → 30身份/关闭关联/固定pin/具体缺项 → Mika与62既有链。
- `family-relations.json`、`license-bindings.json`及许可字节 → 共事件与逐固定父许可证据 → 同上。
- `requests.json`、批次请求/响应/attempt、`batches.json` → 9次实测请求账与批次边界 → 同上。
- `input-bindings.json`、选定input、`coverage.json`、`verification.json` → 有限gate覆盖/完整性 → 同上。
- `counts.json`、`next-queue.json` → 零题增量及下一未查关闭身份9个 → 原49负责人；不含新的正文/源码授权。
- `collect-cleo-metadata.py` → 本次实际资料采集脚本，只作审计，不授权重复请求 → 同上。

## 6. 检索方法与盲区

执行前检索计划：①复用Cleo固定截止issue索引，剔除已详细核查与停止身份；②仅CLOSED_EVENT查同库closer；③只查具名PR的mergedAt与mergeCommit第一父/tree；④只取这些第一父的LICENSE并核字节；⑤与实际v15及v6.12元数据身份/版本并集登记关系。

三批每批都有新增关闭资格证据，但均零材料包；后两批转向不同未查issue的关闭身份，仍属同一元数据资料范围，没有开启题文/源码的新方向。本轮到三批即收口，未翻下一索引页或自动发第四批。下一缓存未查身份9个：241/75/28/61/70/105/81/60/101；仅是未查关闭元数据，不算材料。实际训练、环境、GPU/107、CPU业务执行均0；未新增预算、agent或产品代码，无产品PR。
