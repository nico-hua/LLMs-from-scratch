# LLMs from Scratch 学习记录

学习《Build a Large Language Model (From Scratch)》时整理的代码与中文笔记，使用 PyTorch 逐步实现 GPT，涵盖文本处理、注意力机制、预训练、分类与指令微调，以及 LoRA 参数高效微调。

## 章节内容

| 章节 | 主要内容 |
| --- | --- |
| [第 1 章：理解大语言模型](chapter01.ipynb) | 概念与图解笔记，梳理 Transformer 架构，以及模型构建、预训练和微调的整体流程。 |
| [第 2 章：处理文本数据](chapter02.ipynb) | 实现简单分词器与未知词处理，使用 tiktoken 的 GPT-2 BPE 分词器；通过滑动窗口构建训练样本，组合词元嵌入与位置嵌入。 |
| [第 3 章：编码注意力机制](chapter03.ipynb) | 从简单自注意力出发，实现可训练的 Q/K/V 投影、因果掩码、Dropout 和多头注意力。 |
| [第 4 章：从头实现 GPT](chapter04.ipynb) | 实现 LayerNorm、GELU、前馈网络和残差连接，组装 Transformer 块与 GPT 模型，并用贪心解码演示逐词元生成。 |
| [第 5 章：预训练](chapter05.ipynb) | 使用短篇小说 The Verdict 演示下一词元预测训练，计算交叉熵损失并绘制曲线；实现温度缩放与 Top-k 采样、模型保存与恢复，以及预训练 GPT-2 权重加载。 |
| [第 6 章：分类微调](chapter06.ipynb) | 使用 SMS Spam Collection 数据集，完成类别平衡与数据划分；为 GPT-2 124M 替换分类头，微调最后一个 Transformer 块等参数，评估垃圾短信分类效果。 |
| [第 7 章：指令微调](chapter07.ipynb) | 构造 Alpaca 风格提示词，实现变长批次填充与填充位置的损失屏蔽；微调 GPT-2 355M，生成测试集回复，并通过本地 Ollama 模型评分。 |
| [附录：LoRA 参数高效微调](appendix.ipynb) | 手动实现 LoRA 层与线性层封装，冻结原模型参数，为 GPT-2 124M 的线性层添加低秩分支；复用第 6 章的垃圾短信分类任务，统计可训练参数、绘制损失与准确率曲线，并保存与加载模型。 |

## 文件说明

- [gpt2.py](gpt2.py)：复用的模型组件、数据加载、训练、文本生成及权重映射函数。
- [gpt_download.py](gpt_download.py)：下载 GPT-2 权重，或从本地 TensorFlow checkpoint 读取参数。
- [data/](data/)：小说语料、短信分类数据划分、指令数据及生成的回复示例。

## 运行方式

在仓库根目录安装依赖并启动 JupyterLab，按章节顺序阅读，各 Notebook 内从上到下执行：

```bash
python -m pip install jupyterlab torch tiktoken numpy matplotlib pandas requests tqdm
jupyter lab
```

加载预训练 GPT-2 权重时，还需安装 `tensorflow`：

```bash
python -m pip install tensorflow
```

- **权重准备**：第 5、6 章及附录使用 124M，第 7 章使用 355M。首次运行可在 Notebook 中调用下列函数下载；后续章节的 `load_gpt2` 只读取本地文件。

  ```python
  from gpt_download import download_and_load_gpt2

  download_and_load_gpt2(model_size="124M", models_dir="gpt2")
  # 第 7 章将 model_size 改为 "355M"
  ```

- **模型保存**：提前创建 `model/` 目录；训练时可按硬件情况调整 `batch_size` 和上下文长度。
- **附录数据**：LoRA 实验复用 `data/` 中的 `train.csv`、`validation.csv` 和 `test.csv`；如需重新生成，可先运行第 6 章的数据准备部分。
- **演示报错**：第 2 章用词表外文本演示 `SimpleTokenizerV1` 的局限，会触发 `KeyError`，阅读后继续执行后续单元格即可。
- **回复评分**：第 7 章评分部分需要启动本地 Ollama 服务，并将 `query_model` 和评分调用中的模型名改为本机已安装的模型。

学习参考：[Sebastian Raschka / LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)。
