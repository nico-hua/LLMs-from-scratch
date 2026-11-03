# Agent 轨迹 SFT：全量训练说明

对应代码：[sft-full-training.ipynb](sft-full-training.ipynb)
导航：[目录说明](README.md) · [先期验证说明](sft-pre-verify-guide.md) · [模型评估说明](sft-model-evaluation-guide.md)

本文说明全量 Notebook 的数据处理、训练配置、运行步骤与输出含义。SFT、Loss Masking 和 LoRA 的基础原理见先期验证说明。本次文档维护只读取代码和已有产物，未启动或恢复训练。

## 1. 全量版与先期验证版的关系

| 项目 | 先期验证版 | 全量版 |
| --- | --- | --- |
| 数据范围 | 单个含工具调用的 Session | 本地全部 JSONL Session |
| 分支处理 | 检查所选 Session，按事件顺序转换 | 优先恢复最后一条消息的祖先路径 |
| 数据划分 | 只有训练集 | 先按 Session 划分，再生成样本 |
| 样本构建 | 保存展开的样本用于教学检查 | 逐 Session 生成 Token 数据并写入 JSONL |
| 工具定义 | 所选 Session 实际出现的工具 | 固定八种 Agent 工具 |
| LoRA | 语言模型 `q_proj/v_proj` | 语言模型 Attention 和 MLP 的七类投影 |
| 训练评估 | 手动验证、冒烟、训练集拟合对比 | 按 Epoch 验证与保存 Checkpoint |

全量版继续使用 **2048 Token 预算、Completion-only Loss、单卡 BF16 和 LoRA**，重点增加跨 Session 验证与完整训练记录。

## 2. Notebook 章节与代码入口

| 章节 | 关键代码与作用 |
| --- | --- |
| 1. 配置与下载 | `TRAIN_CONFIG`、`download_dataset()`，设置路径、长度、划分和训练参数。 |
| 2. 事件与分支 | `load_session()`、`restore_message_branch()`，读取事件并选择一条消息路径。 |
| 3. 消息转换 | 文本、工具调用与 Tool Result 转为 `user/assistant/tool` 标准消息。 |
| 4. 监督样本 | `merge_consecutive_assistants()`、`iter_sft_examples()`，逐条产生历史与目标。 |
| 5. 模板与裁剪 | `render_aligned_pair()`、`build_tokenized_sft_example()`，对齐、裁剪并构造标签。 |
| 6. 划分与 Dataset | 按 Session 划分，`build_full_token_files()` 写缓存，`load_dataset()` 加载，Collator 动态填充。 |
| 7. 模型与 LoRA | 单卡检查、BF16 加载、匹配语言模型投影层并注入 Adapter。 |
| 8. Trainer | 配置优化器、梯度累积、显存优化、验证和 Checkpoint 策略。 |
| 9. 训练与导出 | `trainer.train()`、`trainer.evaluate()`，处理 Notebook 进度回调状态，再保存 Adapter、Tokenizer、日志与实验配置。 |

## 3. 数据流与样本质量检查

### 3.1 下载和恢复分支

原始数据保存在 `sft-data/raw/`。全量版通过 `rglob("*.jsonl")` 递归查找；没有找到任何 JSONL 时才调用 Hugging Face Hub 下载。已有部分 JSONL 时会复用，不会自动核实远端快照是否完整。

`restore_message_branch()` 的处理为：

1. 取出消息事件并建立事件 ID 映射。
2. 父链接足够可靠时，从文件中最后一条有 ID 的消息沿 `parentId` 向上追溯。
3. 反转祖先链恢复消息顺序，遇到缺失父节点或重复 ID 时结束。
4. 可用链接不足时，回退到文件中的消息顺序。

每个 Session 只选一条最终路径，没有把所有备选分支展开为额外训练轨迹。

### 3.2 消息转换与训练目标

保留 `user`、`assistant`、`tool`，Assistant 的 `tool_calls` 与 Tool 的 `tool_call_id` 表示调用关系。Thinking 不作为监督文本，图片仅保留占位符。

合并连续 Assistant 消息后，`iter_sft_examples()` 以已有 User 的历史作为 Prompt，当前 Assistant 消息作为 Completion。模型学习的是下一步回复或工具调用；Notebook 不执行生成出来的工具，也不构成完整 Agent 运行环境。

### 3.3 先分 Session，再展开样本

代码先排序原始文件，用种子 42 打乱，取约 10% 为验证 Session，其余用于训练，再分别生成样本。这样，同一 Session 中高度重叠的历史不会同时进入两个分区。

当前实现按文件划分，没有额外对跨 Session 重复内容去重，也未单独导出划分清单。复现划分需要保持原始文件集合与配置一致；仅保留相同随机种子不足以抵消文件集合变化。

### 3.4 模板、工具与裁剪

固定工具集为 `bash/read/edit/write/grep/find/ls/todo`。模板使用 `enable_thinking=False`；Prompt 使用生成前缀，完整序列包含当前目标。先校验文本前缀，再校验 Token ID 前缀，仅对已知的末尾 `<turn|>\n` 差异做受限对齐。

`build_tokenized_sft_example()` 从最近的完整 User 轮次向前尝试扩展历史。历史中的工具调用必须与结果配对；不能放下最新完整轮次时，丢弃该目标。当前实现没有压缩工具输出，也没有按工具步骤拆开过长的 User 轮次。

目标 Assistant 使用未包含在八种 Schema 中的工具时会被过滤。该检查针对当前目标，并非对全部历史工具名称逐一建立白名单。

### 3.5 Token 数据、Loss 和过滤原因

每条 JSONL 保存 `input_ids`、`attention_mask`、`labels`；Prompt 标签为 `-100`，只有 Completion 参与 Loss。Collator 使用 PAD/EOS 动态右侧填充，并将填充位置的 Attention Mask 设为 0、标签设为 `-100`。

| 统计后缀或错误原因 | 含义 |
| --- | --- |
| `overlength_latest_turn` | 最新完整 User 轮次与目标超过总长度预算。 |
| `unpaired_tool_result` | 保留历史中的工具调用与结果不完整配对。 |
| `unsupported_target_tool` | 当前目标调用了未支持的工具。 |
| `chat_template_prefix_mismatch` / `token_prefix_mismatch` | 文本或 Token 前缀无法安全对齐。 |
| `missing_user_or_target` / `empty_completion` | 缺少合法上下文、目标或有效监督 Token。 |
| `invalid_message_or_template` | 样本处理捕获到类型、字段或模板相关异常。 |

这些原因以 `dropped_*` 计数写入转换统计。Session 读取失败单独记为 `failed_sessions`，候选样本数由成功进入样本生成流程的轨迹决定。

`build_full_token_files()` 每次读取一个 Session，写入 `.jsonl.tmp`，正常完成后依次替换正式 JSONL。它减少了全部展开历史同时驻留内存的开销；后续 `load_dataset("json", ...)` 仍是普通 Dataset 加载，没有启用 Hugging Face 的 Streaming 模式。

## 4. 模型、LoRA 与训练配置

模型使用 `AutoModelForMultimodalLM`，BF16 权重、SDPA 注意力、`device_map={"": 0}`。实际训练使用文本及工具调用消息。

LoRA 先通过正则完整匹配真实模块名称，再把匹配结果传给 PEFT，仅选择语言模型中的：

- Attention：`q_proj`、`k_proj`、`v_proj`、`o_proj`。
- MLP：`gate_proj`、`up_proj`、`down_proj`。

当前保存记录为 **245 个目标模块、12,668,928 个可训练参数，占总参数约 0.2467%**；视觉和音频分支未添加 Adapter。

| 配置 | 当前值 |
| --- | --- |
| 模型 / 数据集 | `google/gemma-4-E2B-it` / `badlogicgames/pi-mono` |
| 最大长度 / Seed / 验证比例 | 2048 / 42 / 0.10 |
| 每卡 Train / Eval Batch Size | 1 / 1 |
| 梯度累积 / 训练轮数 | 4 / 3 Epoch，单卡有效 Batch Size 为 4 |
| 学习率 / 权重衰减 | `2e-4` / `0.01` |
| 优化器 / 调度 | `adamw_torch` / Cosine，Warmup 比例 `0.03` |
| 梯度裁剪 | `max_grad_norm=1.0` |
| LoRA | `r=8`、`alpha=16`、`dropout=0.05`、`bias="none"` |
| 显存设置 | BF16，Gradient Checkpointing，`use_reentrant=False`，`use_cache=False` |
| 日志 | 每 10 Step，`report_to="none"` |
| 验证与保存 | 每个 Epoch，`save_total_limit=2` |
| 验证输出 | `prediction_loss_only=True`，不保留整套预测 Logits |
| 输出目录 | `sft-output/gemma4-e2b-lora-full/` |

`trainer.model_accepts_loss_kwargs=False` 配合模型返回的平均 Loss 使用，避免额外传入样本计数参与损失处理。`prediction_loss_only=True` 减少验证时保留预测结果的开销，但不能保证模型 Forward 不计算完整词表 Logits。

Notebook 未设置 `load_best_model_at_end`，最后导出的 Adapter 是当前训练结束后的模型，不会自动选择验证 Loss 最低的 Checkpoint。

## 5. 运行步骤

### 5.1 启动与准备

从仓库根目录执行：

```bash
cd agent_post_training
CUDA_VISIBLE_DEVICES=0 jupyter lab
```

环境记录见 [README.md](README.md)。设置必须早于 Kernel 启动；Notebook 会要求 `torch.cuda.device_count() == 1`。

按第 1～8 节执行配置、数据与 Trainer。Tokenizer 优先使用 `sft-data/gemma4-tokenizer/`；缺少本地配置时使用模型 ID 加载。模型和数据需要能通过 Hugging Face Hub 访问。

### 5.2 首次生成与复用缓存

当前第 6 节的转换调用和统计保存被注释，后续会直接读取现有 JSONL。首次运行或重新处理数据时，在定义 `build_full_token_files()` 后启用：

```python
conversion_stats = build_full_token_files(
    raw_files, train_files, tokenizer, TOOLS
)
(OUTPUT_DIR / "conversion_stats.json").write_text(
    json.dumps(conversion_stats, indent=2, ensure_ascii=False),
    encoding="utf-8",
)
```

原始数据集合、Tokenizer、工具 Schema、裁剪规则、`MAX_LENGTH`、`SEED` 或 `EVAL_RATIO` 变化后，应重新生成数据和统计。缓存仅按长度命名，不记录其他配置的指纹。

复用已有缓存并重启 Kernel 后，在最后导出实验配置前恢复同一次转换的统计：

```python
conversion_stats = json.loads(
    (OUTPUT_DIR / "conversion_stats.json").read_text(encoding="utf-8")
)
```

只加载训练/验证 JSONL 不会定义 `conversion_stats`，而最终 `experiment_config` 会引用它。JSONL、统计文件和当前参数应对应同一次数据处理。

### 5.3 训练与继续已有 Checkpoint

当前代码使用 `trainer.train()` 开始训练，返回后再执行 `trainer.evaluate()`。它没有显式指定恢复 Checkpoint。

如需接续已有训练，可将该训练调用改为指定现存路径，例如：

```python
train_result = trainer.train(
    resume_from_checkpoint=str(OUTPUT_DIR / "checkpoint-2386")
)
```

路径以当时实际保留的 Checkpoint 为准，并保持基础模型、LoRA 目标、数据与训练配置一致。恢复训练使用 Checkpoint 中的训练状态；仅加载 `final_adapter/` 用于推理，不能代替完整训练恢复。

### 5.4 训练后验证回调与最终导出

当前 Notebook 的 `trainer.train()` 已完成 3 Epoch。随后第一次单独调用 `trainer.evaluate()` 时，保存的输出包含：

```text
RuntimeError: on_train_begin must be called before on_evaluate
```

这是 Notebook 进度展示回调的状态异常。后续单元格已移除 `NotebookProgressCallback` 并重新评估成功，无需重复训练：

```python
from transformers.utils.notebook import NotebookProgressCallback

if any(
    isinstance(cb, NotebookProgressCallback)
    for cb in trainer.callback_handler.callbacks
):
    trainer.remove_callback(NotebookProgressCallback)

eval_metrics = trainer.evaluate()
```

Notebook 保留了首次异常和后续成功输出，阅读时应结合两个单元格判断状态。最终验证返回后执行保存单元格，导出 Adapter、Tokenizer、完整 `log_history`、数据规模、训练参数和最终指标；这些文件目前均已生成。

### 5.5 加载与独立评估

后续使用 [sft-model-evaluation.ipynb](sft-model-evaluation.ipynb) 重新加载基础模型与已保存 Adapter，通过直接模型 Forward 比较 Base/SFT Loss，再比较真实验证轨迹和自定义提示的生成结果。

训练阶段的 Trainer `eval_loss`、重新加载后的样本平均 Loss、按 Completion Token 数加权的 Loss 来自不同评估路径或聚合口径，应分别报告。详细计算与已保存结果见 [模型评估说明](sft-model-evaluation-guide.md)。

## 6. 数据与训练产物

以下路径均相对于 `agent_post_training/`，`sft-data/` 与 `sft-output/` 已被根目录 `.gitignore` 排除。

| 路径 | 内容与生成时机 |
| --- | --- |
| `sft-data/raw/` | 下载或已有的原始 Session JSONL。 |
| `sft-data/gemma4-tokenizer/` | 可复用的本地 Tokenizer；不存在时从模型 ID 加载。 |
| `sft-data/train_tokens_2048.jsonl` | 训练 Token 数据，转换流程生成。 |
| `sft-data/validation_tokens_2048.jsonl` | 验证 Token 数据，转换流程生成。 |
| `sft-output/gemma4-e2b-lora-full/conversion_stats.json` | Session、候选、有效样本与过滤原因统计。 |
| `sft-output/gemma4-e2b-lora-full/checkpoint-*/` | 按 Epoch 保存的训练状态，供检查和恢复。 |
| `sft-output/gemma4-e2b-lora-full/final_adapter/` | 最后保存单元格导出的 Adapter 与 Tokenizer。 |
| `sft-output/gemma4-e2b-lora-full/train_logs.csv` | 最后导出的训练与验证日志。 |
| `sft-output/gemma4-e2b-lora-full/experiment_config.json` | 模型、数据、长度、Seed、配置、规模、转换统计及最终指标。 |
| `sft-output/gemma4-e2b-lora-full/evaluation/` | 独立评估 Notebook 导出的 Loss CSV、汇总 JSON 与两类生成对比。 |

先期验证的输出保存在 `gemma4-e2b-lora-sft/`，冒烟实验保存在 `gemma4-e2b-lora-smoke/`；查看结果时应匹配实验目录。

## 7. 完成记录与结果

以下为 **2026-10-22 文档维护时**核对的训练与产物记录：

| 项目 | 保存记录 |
| --- | ---: |
| Session / 原始事件 | 634 / 38,779 |
| Train / Validation Session | 571 / 63 |
| 候选样本 | 15,528 |
| 有效训练 / 验证样本 | 4,770 / 534 |
| 最新轮次超长 | 10,096 |
| 工具配对失败 | 109 |
| 不支持的目标工具 | 11 |
| Token 前缀不一致 | 8 |
| 过滤总数 | 10,224 |

### 7.1 训练与按 Epoch 验证

`checkpoint-3579/trainer_state.json` 确认训练完成 **3 Epoch、3,579 次 Optimizer Step**；`experiment_config.json` 记录训练耗时约 **9,795.56 秒（2 小时 43 分钟）**，训练过程平均 Loss 为 **1.1086**。

| Epoch | Optimizer Step | Trainer 验证 Loss |
| ---: | ---: | ---: |
| 1 | 1,193 | 1.181981 |
| 2 | 2,386 | 1.028047 |
| 3 | 3,579 | 1.026830 |

根据保留策略，目前保存 `checkpoint-2386/` 和 `checkpoint-3579/`。第一轮指标仍保留在日志中，第一轮 Checkpoint 已不在保留目录中。

### 7.2 最终产物与模型加载

`final_adapter/`、`train_logs.csv` 和 `experiment_config.json` 已生成。独立评估 Notebook 已用 `PeftModel.from_pretrained(..., is_trainable=False)` 加载最终 Adapter，记录的可训练参数为 0，并完成全部 534 条验证样本的 Base/SFT 对比。

重新加载后的 SFT 样本平均 Loss 为 **1.0261**，Completion Token 加权 Loss 为 **1.6523**；评估 CSV、汇总指标和生成对比保存在 `evaluation/`。这两个值与训练阶段最终 Trainer Loss **1.0268** 分别保留，具体口径见 [模型评估说明](sft-model-evaluation-guide.md#3-评估方法与指标口径)。

当前最主要的数据限制仍是大量目标因最新 User 轮次超长而被过滤。后续可围绕保留任务指令与最近完整工具交互、压缩工具输出及独立任务评测继续学习；目前的评估尚未执行真实工具或测量完整任务成功率。
