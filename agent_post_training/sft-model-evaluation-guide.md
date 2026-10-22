# Agent 轨迹 SFT：模型加载与评估说明

对应代码：[sft-model-evaluation.ipynb](sft-model-evaluation.ipynb)
导航：[目录说明](README.md) · [先期验证说明](sft-pre-verify-guide.md) · [全量训练说明](sft-full-training-guide.md)

本 Notebook 加载全量训练保存的 LoRA Adapter，在独立 Session 验证集上比较 Base/SFT 的预测 Loss，并观察下一条回复或工具调用的生成结果。本文依据当前代码和已保存产物整理；文档维护未重新执行训练或 GPU 评估。

## 1. 输入、环境与运行顺序

需要先完成全量训练并准备以下本地产物：

| 输入 | 默认路径（相对于本目录） |
| --- | --- |
| 基础模型 | `google/gemma-4-E2B-it`，通过 Hugging Face Hub 加载。 |
| 已训练 Adapter | `sft-output/gemma4-e2b-lora-full/final_adapter/`。 |
| 验证数据 | `sft-data/validation_tokens_2048.jsonl`。 |
| Tokenizer | 优先使用 Adapter 目录，缺少 `tokenizer_config.json` 时回退到基础模型。 |

从仓库根目录启动：

```bash
cd agent_post_training
CUDA_VISIBLE_DEVICES=0 jupyter lab
```

Kernel 工作目录应为 `agent_post_training/`。依赖与先期环境记录见 [README.md](README.md)。第 1 节检查 CUDA、Adapter 配置和验证文件；模型放在可见的 GPU 0。

按 Notebook 第 1～7 节顺序执行。评估无需 Optimizer、Trainer、Gradient Checkpointing，也不需要重新读取原始 Session 或构建训练数据。

| 配置 | 当前值与作用 |
| --- | --- |
| `MAX_LENGTH` | 2048，用于选择已有 Token JSONL 文件，不会重新裁剪样本。 |
| `EVAL_LIMIT` | `None`，评估全部验证样本；设为整数时仅使用前若干条。 |
| `GENERATION_COUNT` | 3，最多选择三条满足筛选条件的真实验证样本生成。 |
| `MAX_NEW_TOKENS` | 128，限制每次新生成的 Token 数。 |
| `SEED` | 42，初始化 Python、NumPy 和 PyTorch 随机种子。 |
| 输出目录 | `sft-output/gemma4-e2b-lora-full/evaluation/`。 |

## 2. Notebook 章节与函数对应关系

| 章节 | 关键实现 |
| --- | --- |
| 1. 配置与依赖 | 设置模型、Adapter、验证缓存、输出目录和评估规模，检查输入文件。 |
| 2. Tokenizer 与数据 | `load_dataset("json", ..., split="train")` 加载已划分的验证 JSONL。 |
| 3. 模型加载 | BF16 + SDPA 加载基础模型，通过 `PeftModel.from_pretrained()` 恢复已训练 Adapter。 |
| 4. 验证 Loss | `compare_validation_loss()` 逐样本直接 Forward，切换 Adapter 比较两种模型。 |
| 4.1～4.2 汇总与图表 | 计算两种平均口径、改善样本数，绘制加权 Loss 柱状图及样本改善量直方图。 |
| 5. 真实轨迹生成 | `split_eval_prompt_target()` 恢复 Prompt，`generate_from_token_ids()` 比较下一步生成。 |
| 6. 自定义提示 | `generate_from_messages()` 使用训练时八种工具 Schema 和 Chat Template。 |
| 7. 保存结果 | 导出逐样本 CSV、汇总 JSON 和两类生成对比 JSON。 |

`split="train"` 只是 Datasets 对单个输入文件返回的 Split 名称；输入仍是训练阶段已经按 Session 隔离的验证数据，没有在评估阶段重新划分。

## 3. 评估方法与指标口径

### 3.1 同一基础权重上切换 Adapter

只加载一份基础模型，再加载：

```python
model = PeftModel.from_pretrained(
    base_model,
    str(ADAPTER_DIR),
    is_trainable=False,
)
model.eval()
```

Base 分支在 `with model.disable_adapter():` 中运行，SFT 分支启用已加载的 Adapter。两者共用基础权重、验证 Token IDs、Attention Mask 和 Completion-only 标签。

评估阶段没有调用 `get_peft_model()` 初始化新的 LoRA；保存的输出确认可训练参数为 **0**。Loss 评估使用 `torch.inference_mode()`、`model.eval()` 和 `use_cache=False`，不更新权重。

### 3.2 逐样本 Forward 与有效监督 Token

`compare_validation_loss()` 将每条样本的 `input_ids/attention_mask/labels` 转为 Batch Size 1 的 Tensor，不重复分词或填充，直接读取模型返回的 `loss`。

Prompt 标签为 `-100`，不参与监督。因果预测需要移位，计数使用：

```python
target_tokens = int((
    (batch["labels"][:, 1:] != -100)
    & (batch["attention_mask"][:, 1:] == 1)
).sum().item())
```

没有有效目标 Token 的样本跳过，非有限 Loss 会抛出异常。当前记录中全部 534 条验证样本进入汇总。

这里不依赖 `trainer.evaluate()`，也不依赖 Jupyter 的 `NotebookProgressCallback`。全量训练曾遇到的训练后回调状态异常及处理见 [训练说明](sft-full-training-guide.md#54-训练后验证回调与最终导出)。

### 3.3 样本平均、Token 加权与相对改善

设每条样本模型返回的 Loss 为 $L_i$，有效监督 Token 数为 $T_i$：

$$
L_{\mathrm{sample}} = \frac{1}{N}\sum_i L_i,
\qquad
L_{\mathrm{weighted}} = \frac{\sum_i T_i L_i}{\sum_i T_i}
$$

样本平均给每条轨迹决策相同权重；Token 加权给较长 Completion 更大权重。两种结果应同时报告。

代码中的相对改善只针对 Token 加权 Loss：

$$
\mathrm{Improvement}
= \frac{L_{\mathrm{Base,weighted}}-L_{\mathrm{SFT,weighted}}}
{L_{\mathrm{Base,weighted}}}\times 100\%
$$

逐样本 `improvement = base_loss - sft_loss`：大于 0 表示 Loss 改善，小于 0 表示变差。它没有评判生成文本是否正确。

训练阶段 Trainer 的最终验证 Loss 与重新加载模型后的逐样本 Loss 来自不同评估路径。本次两个保存值分别为 1.0268303156 和 1.0260701163；文档分别记录，不将其混写为同一次测量。

## 4. 已保存评估结果

以下为 **2026-10-22** 核对的 `evaluation_summary.json` 记录：

| 指标 | Base | SFT |
| --- | ---: | ---: |
| 验证样本数 | 534 | 534 |
| 样本平均 Loss | 14.4432 | 1.0261 |
| Completion Token 加权 Loss | 14.0866 | 1.6523 |

Token 加权 Loss 相对下降 **88.27%**；534 条样本的 Loss 均改善，变差和持平样本数均为 0。

这表示本次固定轨迹验证数据上的条件预测损失下降。**88.27% 不是工具调用准确率或 Agent 任务成功率**。该比较仍采用真实历史和真实前序目标 Token，开放生成时模型需要自行决定后续 Token，两者的评估条件不同。

## 5. 生成对比与结果边界

### 5.1 真实验证轨迹

`split_eval_prompt_target()` 在第一处 `labels != -100` 处分开输入，直接恢复训练时的 Prompt Token IDs 和参考 Completion。这样无需重新套用 Chat Template，避免引入新的前缀差异。

生成样本先筛选完整序列不超过 1400 Token、Completion 长度为 20～256 Token，再按长度排序，覆盖短、中、长位置，最多取三条；它不是对全部验证任务的随机抽样。

保存记录选中了索引 **168、189、335**。Base 和 SFT 均使用贪心解码（`do_sample=False`）、相同 Prompt 和 128 个新 Token 上限；只解码新增部分，保留特殊 Token。

观察保存的输出可见：

- 样本 168：两者都生成一般性的后续协助回复，与参考措辞不同。
- 样本 189：参考回复指出同一流式增量可能同时包含两种内容；两种生成均未复现这一关键反例。
- 样本 335：SFT 回复提到了参考目标之外的文件名，Notebook 没有执行工具来确认这些陈述。

部分回复达到生成长度上限后可能被截断。三条定性样例有助于发现问题，不能替代系统性的生成质量评估。

### 5.2 自定义提示与工具调用

自定义消息使用与全量训练相同的八种工具：`bash/read/edit/write/grep/find/ls/todo`，并设置 `add_generation_prompt=True`、`enable_thinking=False`。

已保存的两个用例为：

| 用例 | Base / SFT 观察 |
| --- | --- |
| 定位代码位置 | 都提出 `find` 调用，搜索词分别为 `load session data` 和 `loadSessionData`。 |
| 查找配置文件 | 都提出 `ls` 调用，路径为 `.`。 |

这些输出只是模型提出的下一步动作。Notebook 没有执行工具、检查仓库文件或继续多轮任务，因此不能仅凭调用格式判断任务完成或参数正确。

## 6. 输出文件与复现记录

结果保存在 `sft-output/gemma4-e2b-lora-full/evaluation/`，由第 7 节统一写出：

| 文件 | 内容 |
| --- | --- |
| `validation_loss_comparison.csv` | `index/seq_len/target_tokens/base_loss/sft_loss/improvement`，每条有效验证样本一行。 |
| `evaluation_summary.json` | 模型、Adapter 路径、样本数、两种平均 Loss、相对改善及改善/变差/持平数量。 |
| `validation_generation_comparison.json` | 选中验证样本的索引、Prompt 长度、参考目标与 Base/SFT 生成。 |
| `manual_generation_comparison.json` | 自定义提示与 Base/SFT 工具调用输出。 |

柱状图和直方图显示在 Notebook 中，代码没有单独导出图像文件。同一路径再次执行保存单元格会覆盖同名结果；比较不同实验时，应保留对应 Adapter、验证缓存、训练配置和评估输出。

上述目录遵循仓库的 `.gitignore`，本地权重、Token 缓存与评估产物不随代码推送。本文与 Notebook 的已保存输出记录了本次结果；复现时需要准备对应训练产物。
