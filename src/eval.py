"""
加载训练好的模型，使用test集进行评估。
返回accuracy。
"""
from collections import Counter

import torch

from dataloader import DataLoader
from model import BERT_POS_Tagging_Model


def evaluate_baseline(dataset_type: str) -> float:
    """Majority Voting 基线：为每个单词分配训练集中出现频率最高的词性。"""
    train_loader = DataLoader(dataset_type, "train")
    test_loader = DataLoader(dataset_type, "test",
                             max_len=train_loader.max_len,
                             token2id=train_loader.token2id)

    # 统计训练集中每个单词的词性频率
    word_tags = {}
    for sent_ids, sent_tags in zip(train_loader.ids, train_loader.labels):
        for tid, tag in zip(sent_ids, sent_tags):
            word = tid  # 用 token id 作为 key
            if word not in word_tags:
                word_tags[word] = Counter()
            word_tags[word][tag] += 1

    # 每个单词的最常见词性
    word_best_tag = {w: c.most_common(1)[0][0] for w, c in word_tags.items()}

    # 全局最常见的词性，作为未知词的回退
    all_tags = [t for tags in train_loader.labels for t in tags]
    default_tag = Counter(all_tags).most_common(1)[0][0]

    # 在测试集上评估
    correct = 0
    total = 0
    for sent_ids, sent_tags in zip(test_loader.ids, test_loader.labels):
        for tid, gold in zip(sent_ids, sent_tags):
            pred = word_best_tag.get(tid, default_tag)
            if pred == gold:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0


def evaluate(model_path: str, dataset_type: str,
             d_model: int = 512, nhead: int = 8,
             num_layers: int = 6, dim_feedforward: int = 2048,
             dropout: float = 0.1) -> float:
    # 加载数据（vocab_size、max_len、num_labels 从数据集自动获取）
    train_loader = DataLoader(dataset_type, "train")
    test_loader = DataLoader(dataset_type, "test",
                             max_len=train_loader.max_len,
                             token2id=train_loader.token2id)

    # 加载模型
    model = BERT_POS_Tagging_Model(
        vocab_size=train_loader.vocab_size,
        num_labels=train_loader.num_labels,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=train_loader.max_len,
    )
    model.load_model(model_path)
    model.eval()

    # 执行评估。由于模型输出的是概率，所以要选择最大的作为预测标签。
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_ids, batch_tags in test_loader:
            for sent_ids, sent_tags in zip(batch_ids, batch_tags):
                if len(sent_ids) == 0:
                    continue
                input_ids = torch.tensor(sent_ids, dtype=torch.long).unsqueeze(0)
                logits = model(input_ids)
                preds = torch.argmax(logits, dim=-1).squeeze(0)

                valid_len = len(sent_tags)
                preds = preds[:valid_len]
                gold = torch.tensor(sent_tags, dtype=torch.long)

                correct += (preds == gold).sum().item()
                total += valid_len

    accuracy = correct / total if total > 0 else 0.0
    return accuracy
