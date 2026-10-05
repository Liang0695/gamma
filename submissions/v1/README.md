# v1 submission（冻结版）

本目录保存 v1 提交物的**冻结副本**，来源可核验、内容不可改。

## 文件

- `submission.zip` —— Kaggle 官方提交包，**root 布局**（不是 `submission_wrapped.zip`）

## 校验值

| 项 | 值 |
|---|---|
| sha256 | `93cb223e79ecb610ba281e507f7833811b879718708b749feff77d05247256bb` |
| 压缩前 | 180,732 B |
| entry 数 | 9 |

包内结构：

```text
adapters/main_lora/adapter_config.json
adapters/main_lora/adapter_model.safetensors
adapters/tool_lora/adapter_config.json
adapters/tool_lora/adapter_model.safetensors
agent.yaml
configs/sampling.yaml
eval_config.yaml
prompts/system.md
sub_agents/code_analyzer.yaml
```

## Kaggle 提交结果

- 竞赛：`gemma-4-developer-agent`
- 状态：`COMPLETE`，publicScore = **0.06**
- 提交描述：`v1: LlmAgent baseline + Agentless two-phase localize prompt + Anthropic context discipline`

## 来源与审查链

| 环节 | 载体 |
|---|---|
| harness 交付 | `KAGGLE-6` |
| 代码审查（结论「放行」，无 🔴、无未解决 🟡） | `KAGGLE-7` |
| Kaggle 提交执行 | `KAGGLE-12` |

本仓库副本由 Multica agent 于 2026-10-05 推送，**推送前已复核 sha256 与冻结值一致**。

## 约束

- 本包是评测提交物：**不要**改动其格式、列名、目录结构与内容。
- 需要新版本时请新建 `submissions/v2/`，不要原地覆盖 v1。
