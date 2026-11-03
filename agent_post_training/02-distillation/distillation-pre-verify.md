
# Teacher–Student 知识蒸馏：先期技术验证与 GKD 实验说明

对应代码：[distillation-pre-verify.ipynb](disttillation-pre-verify.ipynb)
参考风格：[Agent 轨迹 SFT：先期技术验证说明](sft-pre-verify-guide.md)

本文按照当前 Notebook 的代码、执行顺序及其**已保存的输出**，记录从手写 Teacher–Student Forward KL，到使用 TRL `GKDTrainer` 完成 Off-policy / On-policy 训练，再到 Base / Off-policy / On-policy 对照评估的完整过程。本文没有重新运行训练。**本页中的数字以所附 Notebook 的历史输出为准**；如果重新执行时结果不同，应以新运行环境、随机种子、采样及实际日志为准。

## 1. 实验目标与范围

Teacher 为 `Qwen/Qwen3-4B`，Student 为 `Qwen/Qwen3-0.6B`，数据集为 `sergiopaniego/pi-mono-chat`。目标不是让 Student 直接复制一份标准答案，而是让 Student 在指定上下文上学习 Teacher 对**整个下一个 Token 词表**的概率分布。

本 Notebook 分三个层次验证：

1. **数学与手动实现**：Softmax、Forward KL、Completion-only Mask、因果 Shift、双 GPU Forward / Backward、AdamW 参数更新。
2. **框架实现**：TRL 1.14.2 的 `GKDConfig`、`DataCollatorForChatML`、自定义双 GPU `GKDTrainer`，以及完整训练数据上的 Off-policy / On-policy GKD。
3. **效果检查**：固定参考回答上的 Token-weighted KL，以及三个 Student 的实际生成对比。

训练的是 **Student 的全部参数**，不是 SFT 指南中的 LoRA Adapter。Teacher 始终冻结。当前验证了分布拟合与少量文本生成示例；**未进行代码单元测试、Agent 工具调用成功率或真实任务完成率评测**。

## 2. 知识蒸馏原理与监督边界

### 2.1 与 SFT 的区别

对相同的 Prompt $x$ 与回答 $y$，SFT 使用回答中的真实 Token 作为硬标签：

$$
\mathcal L_{\mathrm{SFT}}=-\frac{1}{T}\sum_{t=1}^T\log P_S(y_t\mid x,y_{<t}).
$$

本实验的白盒 Logits 蒸馏使用 Teacher 的完整概率分布作为软目标。设
$P_T(\cdot\mid x,y_{<t})$ 是 Teacher 的分布、$P_S(\cdot\mid x,y_{<t})$ 是 Student 的分布，则手写 Off-policy Forward KL 为：

$$
\mathcal L_{\mathrm{KD}}=\frac{1}{T}\sum_{t=1}^T
D_{\mathrm{KL}}\!\left(P_T(\cdot\mid x,y_{<t})\,\|\,P_S(\cdot\mid x,y_{<t})\right),
$$

其中

$$
D_{\mathrm{KL}}(P_T\|P_S)=\sum_{v\in V}P_T(v)\log\frac{P_T(v)}{P_S(v)}.
$$

数据集中的 Completion 决定 Teacher / Student 在哪些**前序 Token 上下文**计算分布；KL 目标不是只把数据集中的正确 Token 概率推高。因此即使原始回答不够好，也可以计算 KL，但 KL 下降不能自动等价于任务正确率提高。

### 2.2 Off-policy、On-policy 与 `beta` / `lmbda`

| 配置维度      | 值         | 本实验中的含义                                                                |
| ------------- | ---------- | ----------------------------------------------------------------------------- |
| `lmbda=0.0` | Off-policy | 在数据集已有 Completion 上计算蒸馏 Loss。                                     |
| `lmbda=1.0` | On-policy  | 每个训练步骤先让当前 Student 生成 Completion，再在生成轨迹上蒸馏。            |
| `0<lmbda<1` | 混合       | 按步骤概率决定是否采用 Student 生成；**不是**把两种 Loss 直接线性相加。 |
| `beta=0.0`  | Forward KL | $D_{\mathrm{KL}}(P_T\|P_S)$。                                               |
| `beta=0.5`  | JSD 类散度 | 使用 Teacher / Student 的混合分布作为参照。                                   |
| `beta=1.0`  | Reverse KL | $D_{\mathrm{KL}}(P_S\|P_T)$。                                               |

`lmbda` 决定**在哪些轨迹上学习**，`beta` 决定**如何比较概率分布**。本次完整对照训练固定 `beta=0.0`，只把 `lmbda` 从 `0.0` 改为 `1.0`；**没有进行完整的 Reverse KL 或 JSD 训练对比**。GKD 中另有 `seq_kd`（教师生成文本的序列级蒸馏），本实验设置为 `False`。

### 2.3 Token Shift 与 Completion-only Loss

Qwen3 的 Prompt 使用 `add_generation_prompt=True`，完整对话使用 `False`，两者均显式设置 `enable_thinking=False`。只有确认 `full_ids[:prompt_len] == prompt_ids` 后，才用 `prompt_len` 决定监督边界。

```text
input_ids:      [ Prompt Tokens ][ Completion Tokens ][ Padding ]
attention_mask: [       1       ][         1         ][    0    ]
labels:         [     -100      ][ Completion IDs   ][  -100   ]
```

Prompt 仍然参加注意力计算，只是其标签不参与蒸馏损失；Padding 也不参与损失。对于因果语言模型，`logits[t-1]` 预测 `labels[t]`：本次第一条样本的 Prompt 长度为 15，完整长度为 38，因此预测第一个 Completion Token 使用 Logits 位置 14，有效目标位置为 15～37，共 23 个。

手动实现通过 `start_pos = prompt_len - 1`、`end_pos = len(input_ids) - 1` 截取 Logits；GKD 实现采用 `student_logits[:, :-1, :]`、`teacher_logits[:, :-1, :]` 和 `labels[:, 1:]`，两者在监督边界对齐时是同一类计算。

## 3. Notebook 章节与关键代码

| Notebook 章节      | 关键变量、函数及验证内容                                                                                                 |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| 1. 原理            | Teacher–Student 目标、硬标签与软标签的区别。                                                                            |
| 2. Softmax 与 KL   | 三 Token 的 Softmax、手动 KL 与 `F.kl_div()` 对齐、简单梯度下降。                                                      |
| 3. 数据            | `load_dataset()` / `load_from_disk()`、`train_ds` / `eval_ds`，统计消息结构。                                    |
| 4. Chat Template   | `apply_chat_template()`、`enable_thinking=False`、Token 前缀对齐、`labels=-100`、Shift。                           |
| 5. 模型            | BF16 + SDPA 加载 Teacher / Student，分配到 `cuda:1` / `cuda:0`，比较 Logits，提取 Completion KL。                    |
| 6. 手动 Off-policy | `distillation_train_step()`、AdamW、8 条样本训练、`evaluate_distillation()`、Base 对比和保存。                       |
| 7. GKDTrainer      | `GKDConfig`、`DataCollatorForChatML`、`completion_mask`、`DualGPUGKDTrainer`、1 Step 冒烟、721 Step Off-policy。 |
| 8. On-policy       | `OnPolicyDualGPUGKDTrainer`、`lmbda=1`、生成续写后蒸馏、721 Step 并保存。                                            |
| 9. 评估            | `full_eval_examples`、84 条有效 Test 样本的 Token-weighted KL、3 条生成对照和参考回答核查。                            |

Notebook 中的若干调试输出保留了错误修复后的状态，不意味着直接跳到最后一个 Cell 就能在空 Kernel 中运行。复现应从模型、数据和 Trainer 的定义处按依赖顺序执行。

## 4. 数据与 Tokenizer：实际保留了什么

### 4.1 数据来源与结构

`DATASET_ID = "sergiopaniego/pi-mono-chat"`，本地缓存目录为 `./disttillation-data/pi-mono-chat`。如果目录存在，Notebook 调用 `load_from_disk()`；否则下载后通过 `save_to_disk()` 缓存。

| Split     | 原始记录数 | 结构                                          |
| --------- | ---------: | --------------------------------------------- |
| `train` |        797 | 每条均为一条 User 消息加一条 Assistant 消息。 |
| `test`  |         89 | 相同的 `messages` 组织方式。                |

训练集角色统计为 User 797、Assistant 797，每条样本恰好 2 条消息。**这不是完整的多轮 Agent 工具轨迹**：虽然数据来自 pi-mono 场景，但当前字段里没有完整历史工具调用与更早的讨论上下文。后文测试样本中出现的 `setSessionName` / PR 背景缺失，就与这一限制有关。

### 4.2 Tokenizer 对齐

Teacher 与 Student 的模型配置都报告 `vocab_size=151936`，Tokenizer 长度均为 `151669`；`get_vocab()` 的 Token→ID 映射逐项相同。两者词表大小与映射一致，使本实验可以直接在同一个词表维度上比较 Logits，而无需额外做 Token 对齐。

第一条样本 `User: do 1` 的实际长度为：

| 项目       | Token 数 |
| ---------- | -------: |
| Prompt     |       15 |
| 完整对话   |       38 |
| Completion |       23 |

`151936` 是模型 Logits 的最后一维，`151669` 是 Tokenizer 的实际词条数量，两者在这里并不矛盾，不应混为同一个指标。

### 4.3 512 Token 长度筛选

数据处理对 Prompt 和完整对话分别渲染、Tokenize 后检查：完整序列不超过 `512`、Prompt 是完整序列的 Token 前缀，且 Completion 非空。**超长样本被整体丢弃，没有截断后继续训练**。

| 阶段                   |         原始数 |        有效数 |                      过滤数 |
| ---------------------- | -------------: | ------------: | --------------------------: |
| 8 条手工冒烟数据的搜集 | 顺序扫描训练集 |             8 | 扫描过程中跳过 2 条超长记录 |
| 完整 Train             |            797 | **721** |                **76** |
| 完整 Test              |             89 |  **84** |                 **5** |

“完整训练”指遍历**全部符合当前过滤规则的训练记录**，不是对原始 797 条不加筛选地训练。训练和测试采用同样的长度门槛；没有记录 76 条和 5 条中各类过滤原因的单独计数。

## 5. 模型加载与手动蒸馏验证

### 5.1 运行环境及设备分配

| 项目         | Notebook 保存的实际环境                                                                  |
| ------------ | ---------------------------------------------------------------------------------------- |
| PyTorch      | `2.10.0+cu128`（CUDA 12.8）                                                            |
| Transformers | `5.5.0`                                                                                |
| TRL          | `1.14.2`，接口位于 `trl.experimental.gkd`                                            |
| Accelerate   | `1.15.0`                                                                               |
| GPU          | 2 × NVIDIA GeForce RTX 5090，单卡报告约 31.35 GiB                                       |
| Teacher      | Qwen3-4B，约 4.022B 参数；`cuda:1`；BF16；SDPA；`requires_grad_(False)` + `eval()` |
| Student      | Qwen3-0.6B，约 0.596B 参数；`cuda:0`；BF16；SDPA；**全参数训练**                 |

加载完成后，GPU 1 上 Teacher 权重约占 7.49 GiB，GPU 0 上 Student 权重约占 1.11 GiB。这里是模型加载时的已分配显存，**不是正式训练的峰值显存**；训练还需要梯度、优化器状态、激活与 `[Batch, Length, Vocabulary]` 规模的 Logits。

### 5.2 先验证 Logits，再验证参数更新

同一条长 38 Token 的样本，Teacher 与 Student 输出形状均为 `[1, 38, 151936]`。Teacher 在 GPU 1 计算，Student 在 GPU 0 计算；将 Teacher 的 Log 概率移到 GPU 0 后，用：

```python
F.kl_div(
    input=student_log_probs,
    target=teacher_log_probs,
    reduction="none",
    log_target=True,
).sum(dim=-1).mean()
```

计算 Completion 上的 Forward KL。Notebook 已保存结果：

- 第一个有效 Completion 位置的 KL：**5.151323**。
- 完整 23 个 Completion Token 的平均 KL：**1.637961**。
- 手动与 PyTorch 内置 KL 数值一致；逐词表张量形状为 `[1, 23, 151936]`，逐 Token 结果为 `[1, 23]`。

随后使用 `AdamW(lr=1e-5)` 更新 Student。首次更新时，310 个 Student 参数张量均检测到非零梯度，Teacher 没有梯度；同一条样本的评估模式 KL 从 **1.637961 降到 1.020303**。第二次更新后降到 **0.652552**，第一层 Attention `q_proj` 有 198474 个权重元素发生变化。上述数值只是单样本拟合检查，**不能据此证明泛化**。

### 5.3 八条样本的手写训练与早期对照

`distillation_train_step(example)` 封装双 GPU 前向、Completion 位置 KL、`loss.backward()` 和 `optimizer.step()`。执行 8 条样本训练前，Notebook **重新加载了原始 Student 并重建优化器**，避免把前面几次教学单步的更新混入该实验。

8 条有效样本的序列长度为 `38、40、299、171、49、360、496、117`；1 Epoch 执行 8 次更新。Notebook 历史输出的**样本平均训练 KL 为 1.092310**，注意这不是一个对固定样本反复测量的单调下降曲线。

之后在 Test Split 前 2 条符合规则的样本上，采用同一 Teacher 与同一 Prompt/Completion 上下文比较：

| 早期验证（2 条 / 365 个 Completion Token） |  Token-weighted KL |
| ------------------------------------------ | -----------------: |
| 原始 Base Student                          |           0.884294 |
| 手写 8-step Student                        | **0.829324** |
| 相对下降                                   |    **6.22%** |

这组数是**上传 Notebook 当前保存的输出**；它与后来 721 Step、84 条 Test 样本的最终结果**不是同一次评估，不能横向比较其绝对 KL**。原先曾只取 2 条 Test 样本作为调试验证，因此正式评估前对测试集已有一定程度的观察。

## 6. 从手写实现迁移到 TRL GKDTrainer

### 6.1 有效 GKD 配置

Notebook 实际使用 `GKDConfig`（并非 `SFTConfig`）：

| 配置                                  | Off-policy 完整训练             | On-policy 完整训练     |
| ------------------------------------- | ------------------------------- | ---------------------- |
| `lmbda`                             | `0.0`                         | `1.0`                |
| `beta`                              | `0.0`（Forward KL）           | `0.0`（Forward KL）  |
| `seq_kd`                            | `False`                       | `False`              |
| `max_length`                        | `512`                         | `512`                |
| `max_new_tokens`                    | `64`（Off-policy 不触发生成） | `64`                 |
| `temperature`                       | `0.9`（生成配置）             | `0.9`（生成采样）    |
| `per_device_train_batch_size`       | `1`                           | `1`                  |
| `gradient_accumulation_steps`       | `1`                           | `1`                  |
| `learning_rate`                     | `1e-5`                        | `1e-5`               |
| `bf16`                              | `True`                        | `True`               |
| `max_steps`                         | `721`                         | `721`                |
| `logging_steps`                     | `20`                          | `20`                 |
| `eval_strategy` / `save_strategy` | `no` / `no`                 | `no` / `no`        |
| `seed`                              | `42`                          | 从 Off-policy 配置复制 |

最初 `max_steps=1` 只用于冒烟训练，**正式训练前修改为 `len(gkd_full_train_ds)`**。两次正式训练都从 `Qwen/Qwen3-0.6B` 原始权重重新加载，没有在前一次实验的 Student 上接续训练。`DualGPUGKDTrainer` 在计算 Loss 时使用 `temperature=1.0`；这与生成配置的 `temperature=0.9` 是两回事。

### 6.2 默认 Collator 的 Completion 边界不一致

`DataCollatorForChatML` 可生成 `input_ids`、`attention_mask`、`labels`、`prompts`、`prompt_attention_mask` 和 `byte_offsets`。Notebook 最初直接将 `train_ds` 中的 `messages` 交给默认 Collator，得到：

| 样本 | 手动 Completion Tokens | 默认 Collator 有效标签 | 差异 |
| ---- | ---------------------: | ---------------------: | ---: |
| 0    |                     23 |                     27 | 多 4 |
| 1    |                     17 |                     21 | 多 4 |

虽然样本 0 的**完整 Token IDs 完全相同**，但默认 Collator 把前缀中的空 `<think>\n\n</think>\n\n` 部分也纳入了监督标签。其 Prompt 边界与之前显式 `enable_thinking=False` 的渲染方式不一致。

最终使用预先准备好的 Token 序列并显式提供：

```python
completion_mask = [0] * prompt_len + [1] * completion_len
```

再交给 `gkd_collator`。复测后，样本 0 / 1 的有效标签分别恢复为 **23 / 17**，Token IDs 和 Loss Mask 都通过断言。**只有对齐后，手写 KD 与 GKDTrainer 的损失才是在相同监督位置上比较。**

### 6.3 为什么自定义双 GPU Trainer

原生 Trainer 的单进程、多可见 GPU 行为不等价于“Teacher 在 GPU 1，Student 在 GPU 0”的手动流水线。Notebook 定义：

```python
class DualGPUGKDTrainer(GKDTrainer):
    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        # Student：使用 GPU 0 上的 batch 和可训练计算图
        # Teacher：输入显式搬到 GPU 1，在 no_grad 下前向
        # Teacher Logits：搬回 GPU 0，Shift 并按 labels 计算原生 GKD Loss
        ...
```

完整实现见 Notebook。这里重用 TRL 的训练循环与 `generalized_jsd_loss()`，但**没有直接使用原生 `compute_loss()` 的设备逻辑**。

在 1 条真实样本的 Eval 模式下，对照结果为：

- `GKDTrainer` 计算的 Forward KL：`1.6379611492`。
- 手动 Forward KL：`1.6379613876`。

两者通过 `torch.allclose(..., atol=1e-4, rtol=1e-4)`。这验证了当前样本与计算路径下的数值对齐，但不是对所有数据或全部 GKD 分支的通用等价证明。

### 6.4 DataParallel 设备冲突及修复

首次调用 `gkd_trainer.train()` 时，Trainer 自动启用 `torch.nn.DataParallel` 复制 Student 到两张 GPU；Student 部分权重留在 GPU 0，复制到 GPU 1 的输入与权重发生跨设备冲突，报错中包含：

```text
RuntimeError: Expected all tensors to be on the same device
index is on cuda:1, different from other tensors on cuda:0
```

当前 Notebook 的修复是在**重新创建 Trainer 前**设置：

```python
gkd_config._n_gpu = 1
```

从而阻止 Trainer 自动 DataParallel，但仍保留 GPU 1 给 Teacher。这是依赖当前 Transformers 内部属性的**环境特定兼容性处理**，并非官方通用的两卡分布式方案；升级版本后需要重新验证。修复后的 1 Step 冒烟结果：`Global Step=1`、`Training Loss=0.627588`。

另有 `TRLExperimentalWarning`（实验接口可能变化）和 Tokenizer PAD/BOS/EOS 自动同步提示，它们不是上述跨 GPU 运行错误的原因。

## 7. 完整训练、模型保存与运行顺序

### 7.1 Off-policy GKD

构建 `gkd_full_train_ds` 得到 **721 条**有效训练数据。正式训练前删除冒烟 Trainer，重新加载原始 Student，并将 `max_steps=721`、`logging_steps=20`、`save_strategy="no"`。由于 Batch Size=1、梯度累积=1、单进程，本次 721 Steps 对应遍历有效训练集一轮。

Notebook 已保存的终值：

| 指标          |                                   Off-policy |
| ------------- | -------------------------------------------: |
| 有效训练样本  |                                          721 |
| Global Step   |                                **721** |
| Training Loss |                           **0.851554** |
| 模型目录      | `./distillation-output/gkd-offpolicy-full` |

日志每 20 Steps 记录一次训练 Loss，存在明显上下波动：例如 Step 20 为 `1.074581`，Step 400 为 `0.680196`，Step 720 为 `0.783464`。不同 Batch 的内容和长度不同，**不能把这些数当成固定验证集上的连续下降曲线**。

### 7.2 On-policy GKD

On-policy 使用相同的 721 条数据的 Prompt，设置 `lmbda=1.0`，由 Student 在训练中生成续写，生成上限 `max_new_tokens=64`。为了独立比较，再次从原始 Student 权重加载，而不是接续 Off-policy 权重。

Notebook 定义 `OnPolicyDualGPUGKDTrainer(DualGPUGKDTrainer)`，覆盖 `compute_loss()` 并传入 `num_items_in_batch=None`，使 Loss 使用当前生成序列的有效标签进行归约。与 Off-policy 不同，每步需要额外进行生成，因此 **721 Steps 不代表两组实验具有相同 GPU 时间或相同监督 Token 数**。

| 指标          |                                   On-policy |
| ------------- | ------------------------------------------: |
| 训练步骤      |                               **721** |
| Training Loss |                          **0.748038** |
| 模型目录      | `./distillation-output/gkd-onpolicy-full` |

**On-policy 训练日志中的 Loss 较低，不表示模型一定更好**：两种训练使用不同 Completion 轨迹、有效监督位置与归约条件，训练 Loss 不能直接当作跨方法排名。

### 7.3 文件与 Checkpoint

Notebook 保存了三份全参数 Student 模型：

| 目录                                                | 内容与用途                                    |
| --------------------------------------------------- | --------------------------------------------- |
| `distillation-output/qwen3-0.6b-offpolicy-8step/` | 手写 8-step 蒸馏的完整 Student 与 Tokenizer。 |
| `distillation-output/gkd-offpolicy-full/`         | TRL 完整 Off-policy Student 与 Tokenizer。    |
| `distillation-output/gkd-onpolicy-full/`          | TRL 完整 On-policy Student 与 Tokenizer。     |

手写 8-step 使用 `student_model.save_pretrained(..., safe_serialization=True)`；两次完整训练使用 `trainer.save_model()` 保存，随后保存 Tokenizer。它们是**完整 Student 权重，不是 LoRA Adapter**，无需通过 `PeftModel` 加载。Notebook 的 `save_strategy="no"` 表示未配置训练过程中的周期性 Checkpoint；这些最终模型目录也不等于包含 AdamW 状态的完整续训 Checkpoint。

推荐执行顺序：先确保数据与 Tokenizer 完成对齐 → 加载双 GPU Teacher / Student → 完成可选手动实验 → 定义 GKD Collator 与 `DualGPUGKDTrainer` → 运行完整 Off-policy 并**确认保存** → 释放 Off-policy Student、重新从 Base 初始化 On-policy → 保存 → 逐个加载三个版本评估。工作目录应是包含 Notebook 的项目目录，否则所有 `./distillation-output/` 相对路径会指向别的位置。

## 8. 评估：分布接近程度与实际回答

### 8.1 统一的 Token-weighted KL

`evaluate_distillation(examples, student)` 在 Eval / inference 模式下，以 Teacher 为固定参照，逐条计算 Completion 上的 Forward KL；记录每条的 `kl_sum` 和有效 Token 数，最终统计：

$$
\operatorname{KL}_{\mathrm{token\text{-}weighted}}
=\frac{\sum_i\sum_{t=1}^{T_i}D_{\mathrm{KL}}(P_{T,i,t}\|P_{S,i,t})}
{\sum_i T_i}.
$$

它不是简单地对样本平均 KL 再取平均。完整 Test Split 有 89 条记录，筛选后有效 **84 条**，三个 Student 都在相同的参考 Completion 前缀上评估：

| 模型                         | 训练步骤 | Test Token-weighted KL | 相比 Base 相对下降 |
| ---------------------------- | -------: | ---------------------: | -----------------: |
| Base Student                 |        0 |     **1.012856** |                 — |
| **Off-policy Student** |      721 |     **0.817501** |   **19.29%** |
| On-policy Student            |      721 |     **0.964253** |    **4.80%** |

在本次**固定参考回答的 Off-policy 式评估**上，Off-policy Student 最接近 Teacher；On-policy Student 也优于 Base。但这种指标与 Off-policy 的训练上下文一致，未必能充分反映 Student 在**自己生成的轨迹**上的表现。Teacher–Student KL 只衡量分布差距，也不衡量事实正确性、功能通过率或用户满意度。

### 8.2 三条生成示例的定性检查

三个模型使用相同 Prompt、`enable_thinking=False`、`do_sample=False`、`max_new_tokens=128` 进行贪心生成。Notebook 对照了模型输出与数据集参考回答：

| 测试示例                           | 参考回答重点                                                     | 观察到的模型表现                                                                                    |
| ---------------------------------- | ---------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| `/resume` TUI 界面输出           | 窗口宽度、自动换行以及不会触发崩溃的条件。                       | Base 误判为 `grep`；两种蒸馏模型更能识别界面文本，但都没有解释参考回答指出的关键换行 / 宽度原因。 |
| `this all sucks`                 | 针对前文 `setSessionName` / 事件可重入 / PR 设计提出具体方案。 | Base 请求澄清；两种蒸馏模型泛泛谈情绪，未涉及参考回答中的技术决策。                                 |
| `think about birds again please` | 具体描述雨燕的空气动力学与飞行。                                 | Base 作不必要的拒绝式回应；两种蒸馏模型愿意谈鸟，但更多是在追问话题，而非直接完成该请求。           |

第二条样本的 User 输入只有 `this all sucks`，不含此前的代码讨论与 PR 语境。**这是数据上下文缺失的直接证据**：如果输入里没有先前对话，模型无法合理推断参考回答中的具体技术背景。因而不能把这种错位全归咎于模型是否蒸馏成功。

该生成对照仅 3 条，而且有回答被 128 Token 上限截断；只支持定性观察，**不支持统计显著的胜率或能力结论**。

## 9. 常见问题与复现注意事项

| 现象                                                                    | 本次原因与处理                                                                                                                                                                                  |
| ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `input_ids` 相同，GKD 有效监督 Token 却多 4 个                        | 默认 ChatML Prompt 边界与手工 `enable_thinking=False` 处理不一致；显式创建 `completion_mask`，验证 `labels != -100` 后再训练。                                                            |
| 训练时报 `DataParallel` replica 1 / `cuda:1` 设备冲突               | Trainer 自动复制 Student；当前代码在重建 Trainer 前设 `gkd_config._n_gpu=1`，并手动保持 Teacher 在 GPU 1。内部属性方案需要随版本复核。                                                        |
| `BatchEncoding` 没有 `.shape`                                       | `apply_chat_template(..., return_tensors="pt")` 在当前调用方式下返回包装对象；显式 `return_dict=True`，使用 `encoded["input_ids"]`、`encoded["attention_mask"]` 再调用 `generate()`。 |
| `temperature`、`top_p`、`top_k` 与 `do_sample=False` 的生成警告 | 生成配置含采样字段，而评估选择确定性解码；该提示不是 `BatchEncoding` 报错原因。                                                                                                               |
| Tokenizer PAD/BOS/EOS 自动同步提示                                      | Trainer 对模型配置与 Tokenizer 特殊 Token ID 做对齐；本次与跨卡错误无关。                                                                                                                       |
| 训练 Loss 小于另一方法，但 Test KL 较高                                 | 两种策略使用不同续写；训练 Loss 不具跨策略可比性，本次固定参考回答 Test KL 反而是 Off-policy 更低。                                                                                             |
| 增大 `MAX_LENGTH` 但有效样本数不变                                    | 需**重新执行数据筛选和 Dataset 构建**；修改配置本身不会自动重建已经生成的 `gkd_full_train_ds`。                                                                                         |
| 代码生成评估中出现“无法回答”或泛泛追问                                | 可能涉及模型本身、参考回答依赖缺失上下文、生成长度限制；应逐条核查完整 Prompt 和参考语境。                                                                                                      |

另外需要保持三个实验的起点可比：手写 8-step 训练前、完整 Off-policy 训练前和完整 On-policy 训练前，代码都重新从基础 Student 权重加载。若在 Notebook 中跳跃执行单元格，很容易复用已经训练过的 `student_model` 或覆盖 `gkd_trainer` 变量，导致对照不再独立。

当前使用 BF16 对 Student 做全参数 AdamW 训练，这是教学验证设定；长期或大规模实验需进一步检查优化器状态精度、训练稳定性、显存、混合精度策略及断点恢复。双 RTX 5090 的实现是**单进程、Teacher/Student 手动分卡**，不是 DDP / FSDP / DeepSpeed 的通用分布式方案。

## 10. 已验证结论与未覆盖范围

**已验证：**

- Teacher / Student 词表映射一致，Prompt / Completion Token 前缀可对齐，完成了双 GPU Forward KL 与 Student 全参数更新。
- 通过显式 `completion_mask` 修复 GKD Collator 的监督边界；在一条真实样本上，手写与 TRL GKD Forward KL 数值对齐。
- 处理 797 条 Train 中的 721 条有效样本，完成 721-step Off-policy 与 721-step On-policy 训练，各自保存独立的 Student 权重。
- 在 Test 的 84 条有效样本上，Base / Off-policy / On-policy 的 KL 分别为 **1.012856 / 0.817501 / 0.964253**；固定参考回答分布下 Off-policy 更接近 Teacher。
- 三条生成对照证明蒸馏会改变模型回答，但仍存在未解释关键原因、泛化式回应与上下文缺失等问题。

**尚不能据此断言：**

- Off-policy 在所有任务上优于 On-policy，或两者都提升了代码功能正确率。
- 学生已经学会了 Teacher 的真实 Agent 决策能力、工具调用能力或长程上下文处理能力。
- 当前 84 条 Test 与 3 条生成示例足以提供完全无偏、统计显著的能力排名；部分测试样本曾在前期调试中查看。
- Reverse KL、JSD、混合 `lmbda`、不同温度或大于 512 Token 的训练配置已经通过完整对照实验。

如果继续实验，优先级应是**恢复完整上下文或构造任务级、可自动评分的独立评估集**，例如对生成代码运行单元测试，并补充 Student 自生成轨迹上的评估。这样才能回答比“KL 是否下降”更重要的问题：蒸馏是否提升了真实任务完成能力。
