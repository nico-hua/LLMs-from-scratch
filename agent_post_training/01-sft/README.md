# Agent 后训练学习：SFT

[返回项目说明](../README.md)

基于 `badlogicgames/pi-mono` 的 Agent 执行轨迹，对 `google/gemma-4-E2B-it` 进行监督微调，学习预测下一条 Assistant 文本或工具调用。目前已完成先期验证、全量训练、Adapter 加载及独立 Session 验证集评估。

## 文件与学习顺序

| Notebook | 配套说明 |
| --- | --- |
| [sft-pre-verify.ipynb](sft-pre-verify.ipynb)：单 Session 技术验证。 | [先期验证说明](sft-pre-verify-guide.md)：SFT 原理、模板、损失掩码、LoRA 与实验记录。 |
| [sft-full-training.ipynb](sft-full-training.ipynb)：全量数据处理、训练、验证和保存。 | [全量训练说明](sft-full-training-guide.md)：分支恢复、Session 划分、缓存、训练恢复与完成结果。 |
| [sft-model-evaluation.ipynb](sft-model-evaluation.ipynb)：加载已训练 Adapter，比较 Base/SFT。 | [模型评估说明](sft-model-evaluation-guide.md)：指标口径、轨迹生成、自定义工具调用及结果解释。 |

按表格顺序阅读。先期验证包含显存边界测试，按实验目的选择执行单元格；已有训练产物时可直接运行评估 Notebook。

## 运行方式

从仓库根目录进入本目录，在启动 Jupyter/Kernel 前限制为单张 GPU：

```bash
cd agent_post_training
CUDA_VISIBLE_DEVICES=0 jupyter lab
```

Kernel 工作目录应为本目录，`./sft-data` 和 `./sft-output` 均相对于此。全量训练会检查只可见一张 GPU；评估使用可见的 GPU 0。

依赖包括 PyTorch、Transformers、PEFT、Accelerate、Datasets、Hugging Face Hub、Pandas、NumPy、Matplotlib、tqdm 和 JupyterLab。先期 Notebook 保存的环境为 Python 3.10.20、PyTorch 2.10.0+cu128、Transformers 5.5.0、PEFT 0.21.2、Accelerate 1.15.0，单张 RTX 5090，支持 BF16；仓库尚未提供锁定依赖文件。

## 全量训练与结果

| 项目 | 当前记录 |
| --- | --- |
| 数据 | 634 个 Session，训练 / 验证为 571 / 63，生成 4,770 / 534 条有效样本。 |
| 训练 | 长度 2048，单卡 BF16，Batch Size 1，梯度累积 4，3 Epoch，学习率 `2e-4`。 |
| LoRA | `r=8`、`alpha=16`；语言模型 Attention 的 `q/k/v/o_proj` 和 MLP 的 `gate/up/down_proj`，共 12,668,928 个可训练参数。 |
| 完成状态 | 3,579 次 Optimizer Step；已保存最终 Adapter、Tokenizer、训练日志、实验配置及评估结果。 |

截至 2026-10-22，训练阶段最终 Trainer 验证 Loss 为 **1.0268**。重新加载 Adapter 后，在全部 534 条验证样本上逐样本评估，SFT 样本平均 Loss 为 **1.0261**，按 Completion Token 数加权的 Loss 为 **1.6523**。指标口径及 Base/SFT 对比见[评估说明](sft-model-evaluation-guide.md#4-已保存评估结果)。

当前全量版复用已有 Token JSONL，数据转换调用已注释；首次生成数据或重启 Kernel 后恢复统计的步骤见[缓存说明](sft-full-training-guide.md#52-首次生成与复用缓存)。

## 本地数据与输出

以下目录已由根目录 `.gitignore` 排除：

- `sft-data/`：原始 Session、Tokenizer 缓存与训练/验证 Token JSONL。
- `sft-output/gemma4-e2b-lora-sft/`、`gemma4-e2b-lora-smoke/`：先期验证和冒烟测试输出。
- `sft-output/gemma4-e2b-lora-full/`：转换统计、最近两个 Checkpoint、`final_adapter/`、训练日志、实验配置，以及 `evaluation/` 下的逐样本指标、汇总和生成对比。

当前评估生成下一条回复或工具调用文本，不执行真实工具；验证 Loss 改善与完整 Agent 任务成功率属于不同评估目标。
