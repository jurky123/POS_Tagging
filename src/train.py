"""
定义训练流程
"""
import torch
import torch.nn as nn

from dataloader import DataLoader
from model import BERT_POS_Tagging_Model

# 先把超参数都定义在最外面。
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1

BATCH_SIZE = 16
LR = 1e-4
EPOCHS = 10


def train(dataset_type: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 加载数据（vocab_size、max_len、num_labels 从数据集自动获取）
    train_loader = DataLoader(dataset_type, "train", batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset_type, "test", batch_size=BATCH_SIZE,
                             max_len=train_loader.max_len,
                             token2id=train_loader.token2id)
    num_labels = train_loader.num_labels

    # 初始化模型
    model = BERT_POS_Tagging_Model(
        vocab_size=train_loader.vocab_size,
        num_labels=num_labels,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=train_loader.max_len,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        train_correct = 0
        train_total = 0
        for batch_ids, batch_tags in train_loader:
            input_ids = nn.utils.rnn.pad_sequence(
                [torch.tensor(ids, dtype=torch.long) for ids in batch_ids],
                batch_first=True, padding_value=0
            ).to(device)
            labels = nn.utils.rnn.pad_sequence(
                [torch.tensor(tags, dtype=torch.long) for tags in batch_tags],
                batch_first=True, padding_value=-100
            ).to(device)

            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, num_labels), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            valid = labels != -100
            train_correct += (preds[valid] == labels[valid]).sum().item()
            train_total += valid.sum().item()

        train_acc = train_correct / train_total if train_total > 0 else 0.0

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
                labels = nn.utils.rnn.pad_sequence(
                    [torch.tensor(tags, dtype=torch.long) for tags in batch_tags],
                    batch_first=True, padding_value=-100
                ).to(device)

                logits = model(input_ids)
                preds = torch.argmax(logits, dim=-1)

                valid = labels != -100
                correct += (preds[valid] == labels[valid]).sum().item()
                total += valid.sum().item()

        test_acc = correct / total if total > 0 else 0.0
        print(f"[{dataset_type}] epoch {epoch + 1:2d}  loss={total_loss:.2f}  "
              f"train acc={train_acc:.4f}  test acc={test_acc:.4f}")

    save_path = f"model_{dataset_type}.pt"
    model.save_model(save_path)
    print(f"model saved to {save_path}")


if __name__ == "__main__":
    for dataset in ["english", "english_perturbed", "alien"]:
        train(dataset)
