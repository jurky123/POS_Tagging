"""
定义训练流程

单卡运行:  python src/train.py
多卡运行:  torchrun --nproc_per_node=N src/train.py
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist

from dataloader import DataLoader
from model import BERT_POS_Tagging_Model

# 超参数
D_MODEL = 512
NHEAD = 8
NUM_LAYERS = 6
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1

BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 10
UNK_MASK_RATE = 0.1

_device = None
_is_main = True
_local_rank = 0
_world_size = 1


def setup_ddp():
    global _device, _is_main, _local_rank, _world_size
    _local_rank = int(os.environ.get("LOCAL_RANK", 0))
    _world_size = int(os.environ.get("WORLD_SIZE", 1))
    if _world_size > 1:
        torch.cuda.set_device(_local_rank)
        dist.init_process_group(backend="nccl")
        _device = torch.device(f"cuda:{_local_rank}")
        _is_main = (_local_rank == 0)
    else:
        _device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        _is_main = True


def cleanup_ddp():
    if _world_size > 1:
        dist.destroy_process_group()


def log(msg: str):
    print(f"[rank {_local_rank}] {msg}", flush=True)


def train(dataset_type: str):
    log(f"[{dataset_type}] loading data...")
    train_loader = DataLoader(dataset_type, "train", batch_size=BATCH_SIZE,
                              shuffle=True, unk_mask_rate=UNK_MASK_RATE,
                              rank=_local_rank, world_size=_world_size)
    test_loader = DataLoader(dataset_type, "test", batch_size=1,
                             max_len=train_loader.max_len,
                             token2id=train_loader.token2id)
    num_labels = train_loader.num_labels
    log(f"[{dataset_type}] vocab_size={train_loader.vocab_size}, max_len={train_loader.max_len}, "
        f"batches={len(train_loader)}, num_labels={num_labels}")

    log(f"[{dataset_type}] init model...")
    model = BERT_POS_Tagging_Model(
        vocab_size=train_loader.vocab_size,
        num_labels=num_labels,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT,
        max_len=train_loader.max_len,
    ).to(_device)
    if _world_size > 1:
        log(f"[{dataset_type}] wrapping DDP...")
        model = nn.parallel.DistributedDataParallel(model, device_ids=[_local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = torch.tensor(0.0, device=_device)
        train_correct = torch.tensor(0, device=_device)
        train_total = torch.tensor(0, device=_device)
        for batch_ids, batch_tags in train_loader:
            input_ids = nn.utils.rnn.pad_sequence(
                [torch.tensor(ids, dtype=torch.long) for ids in batch_ids],
                batch_first=True, padding_value=0
            ).to(_device)
            labels = nn.utils.rnn.pad_sequence(
                [torch.tensor(tags, dtype=torch.long) for tags in batch_tags],
                batch_first=True, padding_value=-100
            ).to(_device)

            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, num_labels), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            valid = labels != -100
            train_correct += (preds[valid] == labels[valid]).sum()
            train_total += valid.sum()

        if _world_size > 1:
            dist.all_reduce(total_loss)
            dist.all_reduce(train_correct)
            dist.all_reduce(train_total)
        train_acc = (train_correct / train_total).item() if train_total > 0 else 0.0
        total_loss = total_loss.item()

        if _is_main:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_ids, batch_tags in test_loader:
                    input_ids = nn.utils.rnn.pad_sequence(
                        [torch.tensor(ids, dtype=torch.long) for ids in batch_ids],
                        batch_first=True, padding_value=0
                    ).to(_device)
                    labels = nn.utils.rnn.pad_sequence(
                        [torch.tensor(tags, dtype=torch.long) for tags in batch_tags],
                        batch_first=True, padding_value=-100
                    ).to(_device)

                    logits = model(input_ids)
                    preds = torch.argmax(logits, dim=-1)

                    valid = labels != -100
                    correct += (preds[valid] == labels[valid]).sum().item()
                    total += valid.sum().item()

            test_acc = correct / total if total > 0 else 0.0
            log(f"[{dataset_type}] epoch {epoch + 1:2d}  loss={total_loss:.2f}  "
                f"train acc={train_acc:.4f}  test acc={test_acc:.4f}")

    if _is_main:
        save_path = f"checkpoints/model_{dataset_type}.pt"
        model_to_save = model.module if _world_size > 1 else model
        model_to_save.save_model(save_path)
        log(f"model saved to {save_path}")


if __name__ == "__main__":
    setup_ddp()
    log(f"gpus: {_world_size}")
    for dataset in ["english", "english_perturbed", "alien"]:
        train(dataset)
    cleanup_ddp()
