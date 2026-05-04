"""
从data中加载指定的数据集，有三类，一类是english，一类是alien，一类是english_perturbed。每类分为train和test。
得到的是tokens和tags列表。
"""
import os
from typing import Tuple, Iterator

import pandas as pd


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

DATASET_CONFIG = {
    "english": {
        "train": "english_train.parquet",
        "test": "english_test.parquet",
    },
    "english_perturbed": {
        "train": "english_perturbed_train.parquet",
        "test": "english_perturbed_test.parquet",
    },
    "alien": {
        "train": "alien_train.parquet",
        "test": "alien_test.parquet",
    },
}


def load_dataset(dataset_type: str, split: str) -> Tuple[list, list]:
    """Load tokens and tags from a parquet file.

    Args:
        dataset_type: "english", "english_perturbed", or "alien"
        split: "train" or "test"

    Returns:
        tokens: list[list[str]], each inner list is a sentence's tokens
        tags: list[list[int]], each inner list is the corresponding POS tag ids
    """
    cfg = DATASET_CONFIG[dataset_type]
    path = os.path.join(DATA_DIR, cfg[split])
    df = pd.read_parquet(path)
    return df["tokens"].tolist(), df["pos_tags"].tolist()


class DataLoader:
    """Simple data loader that yields (tokens, tags) pairs."""

    def __init__(self, dataset_type: str, split: str, batch_size: int = 1):
        self.tokens, self.tags = load_dataset(dataset_type, split)
        self.batch_size = batch_size

    def __len__(self) -> int:
        return (len(self.tokens) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Tuple[list, list]]:
        for i in range(0, len(self.tokens), self.batch_size):
            yield self.tokens[i:i + self.batch_size], self.tags[i:i + self.batch_size]
