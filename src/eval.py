"""
加载训练好的模型，使用test集进行评估。
返回accuracy。
"""
import torch

from dataloader import DataLoader
from model import BERT_POS_Tagging_Model


def evaluate(model_path: str, dataset_type: str, vocab_size: int = 30522,
             num_labels: int = 37, d_model: int = 512, nhead: int = 8,
             num_layers: int = 6, dim_feedforward: int = 2048,
             dropout: float = 0.1, max_len: int = 128) -> float:
    # 加载数据
    train_loader = DataLoader(dataset_type, "train", vocab_size=vocab_size, max_len=max_len)
    test_loader = DataLoader(dataset_type, "test", vocab_size=vocab_size, max_len=max_len,
                             token2id=train_loader.token2id)

    # 加载模型
    model = BERT_POS_Tagging_Model(
        vocab_size=vocab_size,
        num_labels=num_labels,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_len=max_len,
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
                probs = model(input_ids)
                preds = torch.argmax(probs, dim=-1).squeeze(0)

                valid_len = len(sent_tags)
                preds = preds[:valid_len]
                gold = torch.tensor(sent_tags, dtype=torch.long)

                correct += (preds == gold).sum().item()
                total += valid_len

    accuracy = correct / total if total > 0 else 0.0
    return accuracy
