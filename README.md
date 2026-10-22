# LLMs from Scratch 学习记录

大语言模型与 Agent 后训练的代码和中文学习笔记。原有 GPT 学习内容集中在 `llm_pre_training/`，`agent_post_training/` 用于学习 Agent 后训练，目前已完成基于 Agent 轨迹的 SFT（监督微调）、模型加载与验证集评估。

## 学习内容

| 目录 | 主要内容 |
| --- | --- |
| [llm_pre_training/](llm_pre_training/README.md) | 学习《Build a Large Language Model (From Scratch)》：文本处理、注意力机制、从头实现 GPT、预训练、分类与指令微调，以及 LoRA。 |
| [agent_post_training/](agent_post_training/README.md) | 基于 Agent 执行轨迹进行 SFT：工具调用样本构建、Chat Template、Completion-only Loss、LoRA 训练、模型加载及 Base/SFT 评估。 |

## Notebook 入口

- **GPT 基础与预训练**：[第 1 章](llm_pre_training/chapter01.ipynb)至[第 7 章](llm_pre_training/chapter07.ipynb)，以及 [LoRA 附录](llm_pre_training/appendix.ipynb)；章节说明见 [llm_pre_training/README.md](llm_pre_training/README.md)。
- **Agent SFT 先期验证**：[sft-pre-verify.ipynb](agent_post_training/sft-pre-verify.ipynb)；原理、实验结果与问题记录见[配套说明](agent_post_training/sft-pre-verify-guide.md)。
- **Agent SFT 全量训练**：[sft-full-training.ipynb](agent_post_training/sft-full-training.ipynb)；按 Session 划分训练/验证集，进行全量数据处理、训练与 Checkpoint 保存；参见[全量训练说明](agent_post_training/sft-full-training-guide.md)。
- **模型加载与评估**：[sft-model-evaluation.ipynb](agent_post_training/sft-model-evaluation.ipynb)；加载已训练 Adapter，对比验证 Loss 与生成结果；参见[评估说明](agent_post_training/sft-model-evaluation-guide.md)。

## 运行方式

Notebook 使用各自目录下的相对路径，请从仓库根目录进入对应目录后启动 JupyterLab：

```bash
# GPT 基础与预训练
cd llm_pre_training
jupyter lab
```

```bash
# Agent SFT（从仓库根目录执行）
cd agent_post_training
CUDA_VISIBLE_DEVICES=0 jupyter lab
```

依赖、权重和数据准备见各目录 README。模型权重、下载缓存及 SFT 训练产物由 `.gitignore` 排除，运行时按需准备。

学习参考：[Sebastian Raschka / LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)。
