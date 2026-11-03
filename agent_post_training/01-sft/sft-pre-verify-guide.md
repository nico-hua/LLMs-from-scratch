# Agent 轨迹 SFT：先期技术验证说明

对应代码：[sft-pre-verify.ipynb](sft-pre-verify.ipynb)
导航：[目录说明](README.md) · [全量训练说明](sft-full-training-guide.md) · [模型评估说明](sft-model-evaluation-guide.md)

本文结合当前 Notebook 的函数、执行顺序和已保存输出，说明如何用单个 Agent Session 验证 SFT 流程。文中的实验数字来自历史执行记录；本次文档维护未重新运行训练。

## 1. 实验目标与范围

基础模型为 `google/gemma-4-E2B-it`，轨迹来自 `badlogicgames/pi-mono`。训练任务是：给定用户请求、历史 Assistant 行为和工具返回结果，预测下一条 Assistant 文本或工具调用。

本 Notebook 依次验证消息转换、样本构建、Chat Template、Completion-only Loss、LoRA 参数更新、Trainer、Adapter 保存与加载，以及 Base/SFT 对比。它只使用一个 Session，没有独立验证集；全量训练及跨 Session 验证由另一份 Notebook 接续。

## 2. SFT 原理与代码中的监督边界

### 2.1 一条轨迹如何产生多条训练样本

`build_sft_examples()` 遍历标准化消息，先合并连续 Assistant 消息，再将每条合法 Assistant 消息作为目标：

| 样本 | Prompt | Completion |
| --- | --- | --- |
| 1 | 用户请求 | Assistant 发起第一个工具调用 |
| 2 | 用户请求、工具调用和结果 | Assistant 决定下一步操作 |
| 3 | 此前全部交互 | Assistant 给出最终回复 |

历史 Assistant 和工具结果是当前决策的条件；每个样本只监督当前目标。原始 Thinking 被忽略，图片转换为 `[image omitted]`，本流程没有训练视觉输入。

### 2.2 Teacher Forcing 与 Completion-only Loss

训练时模型读取真实前序 Token，预测下一个 Token。对 Prompt $x$ 和 Completion $y$，目标为：

$$
\mathcal L_{\mathrm{SFT}}
= -\frac{1}{T}\sum_{t=1}^{T}
\log p_\theta(y_t\mid x,y_{<t})
$$

`tokenize_sft_example()` 构造三个同长度字段，`SFTDataCollator` 再为批次动态右侧 Padding：

```text
input_ids:      [ Prompt Tokens ][ Completion Tokens ][ Padding ]
attention_mask: [       1       ][         1         ][    0    ]
labels:         [     -100      ][ Completion IDs   ][  -100   ]
```

Prompt 仍然参与注意力计算，只有对应标签不参与交叉熵；Padding 同时通过 Attention Mask 和 `labels=-100` 处理。下一词元预测所需的标签移位由因果语言模型的 Loss 实现完成。

这里使用 Hugging Face `Trainer` 和手工标签，不依赖 TRL 的角色级 `assistant_only_loss`。当前做法按 Prompt/Completion 的位置边界屏蔽历史，不能与“监督历史中所有 Assistant 消息”混为一谈。

### 2.3 LoRA 更新哪些参数

LoRA 冻结基础权重 $W$，训练低秩分支：

$$
h = Wx + \frac{\alpha}{r}BAx
$$

代码使用 `r=8`、`lora_alpha=16`、`lora_dropout=0.05`，正则仅匹配语言模型 Attention 的 `q_proj/v_proj`。已保存输出为 **1,634,304 个可训练参数，占总参数约 0.0319%**。

这减少了梯度和优化器状态的开销；长序列激活与词表 Logits 的显存开销仍然存在。

## 3. Notebook 章节与函数对应关系

| 章节 | 关键代码与作用 |
| --- | --- |
| 1. 下载数据 | `download_dataset()` 下载 JSONL，读取本地原始文件。 |
| 2. 理解事件流 | `find_tool_session()` 找到含工具调用的 Session；统计事件、角色、内容类型并检查分支。 |
| 3. 消息转换 | `extract_text_content()`、`convert_tool_call()`、`convert_event_to_message()` 生成标准消息，并检查调用 ID。 |
| 4. 监督样本 | `merge_consecutive_assistants()`、`build_sft_examples()`；探索工具分组与按 User 轮次裁剪。 |
| 5. 模板渲染 | `render_aligned_pair()`、`build_rendered_sft_example()` 按真实模板长度裁剪并校验前缀。 |
| 6. Dataset | `tokenize_sft_example()`、`Dataset.from_list()`、`SFTDataCollator`；检查 Padding 与监督位置。 |
| 7. 模型与 LoRA | BF16、SDPA 加载模型，注入语言模型 `q_proj/v_proj` Adapter，检查可训练参数。 |
| 8. 参数更新验证 | 手动执行 Forward、Backward、AdamW 更新，再使用 8 条短样本做 Trainer 冒烟训练。 |
| 9. 正式单 Session 训练 | 保留显存测试函数，使用 43 条样本训练 3 Epoch，保存 Adapter 与日志。 |
| 10. 加载与对比 | 重新加载基础模型和 Adapter，比较生成输出及训练集 Loss。 |
| 11. 总结 | 记录流程已验证的部分及单 Session 实验的局限。 |

先期验证按文件顺序转换所选 Session；虽然检查了事件树，当前转换流程没有执行全量版的 `restore_message_branch()`。已保存的所选 Session 分支节点数为 0。

## 4. 数据、工具与上下文裁剪

### 4.1 数据记录

历史下载记录为 627 个 Session 文件，实际选用其中一个含工具调用的 Session：

- 325 个事件，320 条原始消息。
- 转换后 319 条消息：19 条 User、156 条 Assistant、144 条 Tool。
- 一个内容为空的 Assistant 被过滤，得到 156 个候选监督目标。
- 2048 Token 预算下保留 43 条，实际序列长度为 772～2011，平均约 1380.65。

627 是先期快照的文件数；全量版保存的 634 来自另一份数据快照，复现时以本地文件与转换统计为准。

### 4.2 工具定义的实际来源

`KNOWN_TOOL_SCHEMAS` 包含八种工具的已知 Schema，但先期验证实际调用：

```python
tools = make_tool_definitions(set(tool_counts))
```

所选 Session 使用的是 `bash/read/edit/write` 四种工具，因此本次渲染并未固定加入全部八种定义。全量版才固定使用八种工具；工具定义变化也会改变 Prompt 长度及样本保留情况。

### 4.3 探索性裁剪与最终裁剪不同

第 4 节的 `trim_sft_context()`、`trim_context_by_turns()` 会按工具分组处理历史，尝试压缩长文本，并用 JSON 序列化后的 Token 数粗估预算。

后续真正构建数据的 `build_rendered_sft_example()` 使用模型 Chat Template 计算完整长度，从最近的完整 User 轮次向前添加历史，**不再调用上述文本压缩或组级回退逻辑**。最新轮次放不下时直接丢弃样本，不拆分轮次，也不截断目标 Completion。

`group_history()` 和工具 ID 检查用于前期结构验证；虽然定义了 `validate_tool_pairs()`，当前最终渲染器没有显式调用它。全量版将该检查接入了样本构建流程。

| Token 预算 | 保留 / 候选 | 保留率 |
| ---: | ---: | ---: |
| 2048 | 43 / 156 | 27.56% |
| 4096 | 60 / 156 | 38.46% |
| 8192 | 78 / 156 | 50.00% |
| 16384 | 101 / 156 | 64.74% |

预算包含工具定义、全部 Prompt 和 Completion。保留率低主要与长历史及完整轮次裁剪有关。

### 4.4 模板前缀对齐

Prompt 使用 `add_generation_prompt=True`，完整对话使用 `False`，两者均设置 `enable_thinking=False`。先检查文本前缀，再检查 Token ID 前缀，才能用 Prompt 长度确定 Loss 边界。

`render_aligned_pair()` 只处理一种已知差异：Prompt 末尾多出 `<turn|>\n`，且仅移除该后缀后 Full 才以 Prompt 开头。其他差异仍抛出异常；不能用任意字符串截断绕过检查。

## 5. 运行与有效配置

从仓库根目录执行：

```bash
cd agent_post_training
CUDA_VISIBLE_DEVICES=0 jupyter lab
```

依赖与先期环境记录见 [README.md](README.md)。启动 Kernel 前限制为单张 GPU，Kernel 工作目录应为本目录。

### 5.1 本地数据与 Tokenizer

先期版的 `download_dataset()` 发现 `sft-data/raw/` 存在就返回，不会补齐空目录；之后仅使用 `glob("*.jsonl")` 读取该目录顶层文件。

Tokenizer 下载块当前被注释，实际加载为 `local_files_only=True`。首次运行需准备 `sft-data/gemma4-tokenizer/`；可启用 Notebook 中的下载块，再加载本地 Tokenizer：

```python
tokenizer_dir = snapshot_download(
    repo_id="google/gemma-4-E2B-it",
    local_dir="./sft-data/gemma4-tokenizer",
    allow_patterns=[
        "tokenizer.json", "tokenizer_config.json",
        "chat_template.jinja", "config.json",
    ],
    max_workers=1,
)
```

### 5.2 以实际构建 Dataset 的配置为准

| 项目 | 当前实际设置 |
| --- | --- |
| 训练集长度预算 | 构建 `train_dataset` 的单元格设置 `MAX_LENGTH=2048`。 |
| 训练 Batch / 梯度累积 | 1 / 4，单卡有效 Batch Size 为 4。 |
| 正式训练 | 3 Epoch，学习率 `2e-4`，权重衰减 `0.01`，Cosine，Warmup 比例 `0.03`。 |
| 精度与显存 | BF16、SDPA、Gradient Checkpointing，`use_cache=False`。 |
| 验证与 Checkpoint | 正式单 Session Trainer 未配置验证集，`eval_strategy="no"`、`save_strategy="no"`。 |
| 训练损失处理 | `trainer.model_accepts_loss_kwargs=False`，使用模型返回的平均 Loss。 |

演示样本和 `TRAIN_CONFIG` 中仍有 `8192`；它们不会自动重建已经生成的 2048 长度训练集。修改长度时，应重新执行数据构建并检查实际序列长度。

### 5.3 实验执行顺序会影响 Adapter

手动单步验证调用了 `optimizer.step()`，冒烟测试也执行了训练；后续正式训练继续使用同一个 `model`。按当前顺序执行时，正式 3 Epoch 会接续这些 Adapter 更新。

如果需要从相同初始状态比较独立实验，应在正式训练前重新加载基础模型并重新注入 LoRA，再创建对应 Trainer。显存测试函数只做 Forward/Backward，不调用 `optimizer.step()`；其三个调用单元格目前均被注释。

显存函数仅从传入 Dataset 中选择“不超过阈值的最长样本”。已有 Dataset 全部不超过 2048 时，把阈值改为 4096 或 8192 也不会产生更长样本；需要先构建对应长度的数据集。

## 6. 显存记录与常见问题

以下显存结果来自 Notebook 保留的历史文字记录，未在本次维护中复测：

| 长度预算 | 当时实际长度 | Forward + Backward 峰值 | 结果 |
| ---: | ---: | ---: | --- |
| 2048 | 1997 | 19.45 GiB | 成功 |
| 4096 | 4045 | 28.79 GiB | 成功，余量较小 |
| 8192 | 接近 8192 | OOM | Loss 阶段额外申请约 8 GiB 失败 |

`labels=-100` 屏蔽监督位置，并不保证模型跳过这些位置的词表投影。完整 Logits 的规模约为 `[Batch, Length, Vocabulary]`，因此 LoRA 也不能单独解决长序列显存问题。

| 现象 | 原因与处理 |
| --- | --- |
| Prompt / Full 前缀不同 | 先检查 Chat Template 的轮次语义，只使用已实现的受限后缀对齐，再验证 Token 前缀。 |
| Trainer 意外使用多张卡 | `device_map={"": 0}` 不能限制 Trainer 的可见设备；在启动 Kernel 前设置 `CUDA_VISIBLE_DEVICES=0`。 |
| 推理时报 `BatchEncoding` 没有 `shape` | 从模板返回值中提取 `encoded["input_ids"]` 和 `attention_mask`，再传给 `generate()`。 |
| Loss 下降但输出相同 | 贪心解码只取最高概率 Token，概率变化未必改变输出；结合 Loss 与多任务评估观察。 |
| 单条样本 Loss 变差 | 整体优化不保证每条样本改善，应报告样本分布和整体指标。 |

先期保存的冒烟输出为 8 条样本、1 Epoch、2 次 Optimizer Step，峰值显存 **13.72 GiB**。它使用较短样本，与上面的长序列边界测试不同。

## 7. 保存、推理与结果解释

训练输出位于 `sft-output/gemma4-e2b-lora-sft/`：

- `final_adapter/`：LoRA Adapter 与 Tokenizer，由训练后的保存单元格导出。
- `train_logs.csv`：从 `trainer.state.log_history` 中筛选包含 `loss` 的记录。
- 没有按 Epoch 自动保存的训练 Checkpoint；目录中的 Adapter 也不包含完整基础模型权重。

第 10 节重新加载基础模型，通过 `PeftModel.from_pretrained(..., is_trainable=False)` 加载 Adapter。生成时使用相同工具定义及模板，`do_sample=False`；通过 `disable_adapter()` 比较基础模型。

已保存的正式训练记录为 **43 条样本、3 Epoch、33 次 Optimizer Step**。在这 43 条训练样本上的评估为：

| 指标 | Base | SFT |
| --- | ---: | ---: |
| 样本平均 Loss | 2.9912 | 2.9586 |
| 按目标 Token 数加权的 Loss | 2.7200 | 2.7084 |

29 条样本改善、14 条变差，加权 Loss 降低约 0.0116。代码先逐样本计算 Loss，再以 `labels != -100` 的数量加权；这与训练过程日志的统计条件不同。

同一个代码查找提示下，Base 和 SFT 都生成了相同的 `find` 工具调用。这些记录说明流程可以运行、训练集拟合略有变化；跨 Session 的泛化与完整 Agent 任务完成率需要独立评估。

全量训练、Adapter 加载与独立 Session 验证集评估现已完成，见 [全量训练说明](sft-full-training-guide.md) 和 [模型评估说明](sft-model-evaluation-guide.md)。本页保留单 Session 先期实验的历史数据，不能与后续全量验证指标直接比较。
