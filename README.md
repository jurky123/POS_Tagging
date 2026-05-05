# POS Tagging

基于 Transformer Encoder（RoPE 位置编码）的词性标注模型，从头训练。

## 环境配置

```bash
conda create -n tag python=3.10
conda activate tag
pip install -r requirements.txt
```

requirements.txt 内容：`pandas`、`pyarrow`、`torch`（需 CUDA 版本，参考 [pytorch.org](https://pytorch.org) 安装）。

## 数据集

数据位于 `data/` 目录，包含三个数据集的 train/test 划分：

| 数据集 | 词性类别数 |
|--------|-----------|
| english | 37 |
| english_perturbed | 38 |
| alien | 18 |

## 训练

单卡训练：

```bash
python src/train.py
```

多卡训练（DDP）：

```bash
torchrun --nproc_per_node=N src/train.py
```

模型权重保存至 `checkpoints/`，每个数据集一个文件：`model_english.pt`、`model_english_perturbed.pt`、`model_alien.pt`。

## 评估

```python
from src.eval import evaluate, evaluate_baseline

# Majority Voting 基线
acc = evaluate_baseline("english")

# 训练好的模型
acc = evaluate("checkpoints/model_english.pt", "english")
```

## 项目结构

```
├── data/                  # 数据集（.parquet）
├── checkpoints/           # 模型权重
├── src/
│   ├── dataloader.py      # 数据加载与词表构建
│   ├── model.py           # Transformer + RoPE 模型
│   ├── train.py           # 训练流程
│   └── eval.py            # 评估与 Majority Voting 基线
├── requirements.txt
└── README.md
```

## 主要方法

- **RoPE 位置编码**：在注意力计算中对 q/k 做旋转变换，无需可学习参数
- **Word Dropout**：训练时以 10% 概率将 token 替换为 [UNK]，提升对未见词的泛化能力
- **Transformer Encoder**：8 层，d=768，4 头，Pre-LN 架构，GELU 激活

## 结果

| 数据集 | Majority Voting | 本文方法 |
|--------|----------------|---------|
| english | 91.40% | 96.15% |
| english_perturbed | 88.42% | 88.84% |
| alien | 82.93% | 85.69% |
