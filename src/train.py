"""
定义训练流程
"""
import torch
import torch.nn as nn

from dataloader import DataLoader
from model import BERT_POS_Tagging_Model

# 先把超参数都定义在最外面。
VOCAB_SIZE = 30522
NUM_LABELS = 37
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
MAX_LEN = 512

BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 10
MODEL_SAVE_PATH = "model.pt"


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据
    train_loader = DataLoader("english", "train", vocab_size=VOCAB_SIZE,
                              batch_size=BATCH_SIZE, max_len=MAX_LEN)
    test_loader = DataLoader("english", "test", vocab_size=VOCAB_SIZE,
                             batch_size=BATCH_SIZE, max_len=MAX_LEN,
                             token2id=train_loader.token2id)

    # 初始化模型
    model = BERT_POS_Tagging_Model(
        vocab_size=VOCAB_SIZE,
        num_labels=NUM_LABELS,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=MAX_LEN,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch_ids, batch_tags in train_loader:
            # 填充到批次内最长
            input_ids = nn.utils.rnn.pad_sequence(
                [torch.tensor(ids, dtype=torch.long) for ids in batch_ids],
                batch_first=True, padding_value=0
            ).to(device)
            attention_mask = (input_ids != 0).long()
            labels = nn.utils.rnn.pad_sequence(
                [torch.tensor(tags, dtype=torch.long) for tags in batch_tags],
                batch_first=True, padding_value=-100
            ).to(device)

            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits.view(-1, NUM_LABELS), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # 每个 epoch 在 test 集上评估 accuracy
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_ids, batch_tags in test_loader:
                input_ids = nn.utils.rnn.pad_sequence(
                    [torch.tensor(ids, dtype=torch.long) for ids in batch_ids],
                    batch_first=True, padding_value=0
                ).to(device)
                attention_mask = (input_ids != 0).long()
                labels = nn.utils.rnn.pad_sequence(
                    [torch.tensor(tags, dtype=torch.long) for tags in batch_tags],
                    batch_first=True, padding_value=-100
                ).to(device)

                logits = model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=-1)

                mask = labels != -100
                correct += (preds[mask] == labels[mask]).sum().item()
                total += mask.sum().item()

        accuracy = correct / total if total > 0 else 0.0
        print(f"epoch {epoch + 1:2d}  loss={total_loss:.4f}  test accuracy={accuracy:.4f}")

    model.save_model(MODEL_SAVE_PATH)
    print(f"model saved to {MODEL_SAVE_PATH}")
