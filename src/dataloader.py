"""
从data中加载指定的数据集，有三类，一类是english，一类是alien，一类是english_perturbed。每类分为train和test。
得到的是tokens和tags列表。
"""
import os
import random
from typing import Tuple, Iterator, Optional

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

DATASET_CONFIG = {
    "english": {
        "train": "english_train.parquet",
        "test": "english_test.parquet",
        "num_labels": 37,
    },
    "english_perturbed": {
        "train": "english_perturbed_train.parquet",
        "test": "english_perturbed_test.parquet",
        "num_labels": 38,
    },
    "alien": {
        "train": "alien_train.parquet",
        "test": "alien_test.parquet",
        "num_labels": 18,
    },
}


class DataLoader:
    """数据加载器，加载数据并构建词表，迭代时返回 token_id 和 tag 对。"""

    def __init__(self, dataset_type: str, split: str,
                 vocab_size: Optional[int] = None,
                 batch_size: int = 1, max_len: Optional[int] = None,
                 token2id: Optional[dict] = None, shuffle: bool = False,
                 unk_mask_rate: float = 0.0,
                 rank: int = 0, world_size: int = 1):
        self.dataset_type = dataset_type
        raw_tokens, raw_tags = self._load_dataset(dataset_type, split)
        self.num_labels = DATASET_CONFIG[dataset_type]["num_labels"]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.unk_mask_rate = unk_mask_rate
        self.rank = rank
        self.world_size = world_size

        if max_len is not None:
            self.max_len = max_len
        else:
            self.max_len = max(len(s) for s in raw_tokens)

        if token2id is not None:
            self.token2id = token2id
        else:
            self.token2id = self._build_vocab(raw_tokens, vocab_size)

        self.ids = [self._tokens_to_ids(t[:self.max_len]) for t in raw_tokens]
        self.labels = [t[:self.max_len] for t in raw_tags]

    @property
    def vocab_size(self) -> int:
        return len(self.token2id)

    @staticmethod
    def _load_dataset(dataset_type: str, split: str) -> Tuple[list, list]:
        cfg = DATASET_CONFIG[dataset_type]
        path = os.path.join(DATA_DIR, cfg[split])
        df = pd.read_parquet(path)
        return df["tokens"].tolist(), df["pos_tags"].tolist()

    def _build_vocab(self, tokens: list, vocab_size: Optional[int]) -> dict:
        freq = {}
        for sent in tokens:
            for t in sent:
                freq[t] = freq.get(t, 0) + 1
        sorted_tokens = sorted(freq, key=freq.get, reverse=True)
        n = len(sorted_tokens)
        token2id = {t: i + 2 for i, t in enumerate(sorted_tokens[:n])}
        token2id["[PAD]"] = 0
        token2id["[UNK]"] = 1
        return token2id

    def _tokens_to_ids(self, tokens: list) -> list:
        return [self.token2id.get(t, 1) for t in tokens]

    def __len__(self) -> int:
        total = (len(self.ids) + self.batch_size - 1) // self.batch_size
        return total // self.world_size

    def __iter__(self) -> Iterator[Tuple[list, list]]:
        indices = list(range(len(self.ids)))
        if self.shuffle:
            random.shuffle(indices)
        all_batches = []
        for i in range(0, len(indices), self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            batch_ids = [self.ids[j].copy() for j in batch_idx]
            if self.unk_mask_rate > 0:
                for ids in batch_ids:
                    for k in range(len(ids)):
                        if ids[k] != 0 and random.random() < self.unk_mask_rate:
                            ids[k] = 1
            all_batches.append((batch_ids, [self.labels[j] for j in batch_idx]))
        batches_per_rank = len(all_batches) // self.world_size
        start = self.rank * batches_per_rank
        for b in all_batches[start:start + batches_per_rank]:
            yield b
