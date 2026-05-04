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
                 token2id: Optional[dict] = None, shuffle: bool = False):
        self.dataset_type = dataset_type
        raw_tokens, raw_tags = self._load_dataset(dataset_type, split)
        self.num_labels = DATASET_CONFIG[dataset_type]["num_labels"]
        self.batch_size = batch_size
        self.shuffle = shuffle

        # max_len：未指定则取数据中最长句子的长度
        if max_len is not None:
            self.max_len = max_len
        else:
            self.max_len = max(len(s) for s in raw_tokens)

        # 词表：如果传入则使用，否则基于当前数据构建
        if token2id is not None:
            self.token2id = token2id
        else:
            self.token2id = self._build_vocab(raw_tokens, vocab_size)

        # 预先将所有 token 转换为 id，只做一次
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
        """基于当前加载的 tokens 构建词表，高频词优先。"""
        freq = {}
        for sent in tokens:
            for t in sent:
                freq[t] = freq.get(t, 0) + 1
        sorted_tokens = sorted(freq, key=freq.get, reverse=True)
        # 未指定 vocab_size 则使用全部 token
        n = len(sorted_tokens) if vocab_size is None else vocab_size - 2
        # 预留 0 给 [PAD]，1 给 [UNK]
        token2id = {t: i + 2 for i, t in enumerate(sorted_tokens[:n])}
        token2id["[PAD]"] = 0
        token2id["[UNK]"] = 1
        return token2id

    def _tokens_to_ids(self, tokens: list) -> list:
        """将 token 列表转为 id 列表。"""
        return [self.token2id.get(t, 1) for t in tokens]  # 1 = [UNK]

    def __len__(self) -> int:
        return (len(self.ids) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[list, list]]:
        indices = list(range(len(self.ids)))
        if self.shuffle:
            random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_idx = indices[i:i + self.batch_size]
            yield [self.ids[j] for j in batch_idx], [self.labels[j] for j in batch_idx]
