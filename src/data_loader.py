
from pathlib import Path

import numpy as np
import torch

from configs import config

DEVICE = torch.device(
    config.DEVICE if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = config.BATCH_SIZE
BLOCK_SIZE = config.BLOCK_SIZE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

train_data = np.memmap(
    DATA_DIR / "train.bin",
    dtype=np.uint16,
    mode="r"
)

val_data = np.memmap(
    DATA_DIR / "val.bin",
    dtype=np.uint16,
    mode="r"
)


def get_batch(split):
    if split == "train":
        data = train_data
    elif split == "val":
        data = val_data
    else:
        raise ValueError("split must be 'train' or 'val'")

    ix = torch.randint(
        len(data) - BLOCK_SIZE - 1,
        (BATCH_SIZE,)
    )

    x = torch.stack([
        torch.from_numpy(
            data[i:i + BLOCK_SIZE].astype(np.int64)
        )
        for i in ix
    ])

    y = torch.stack([
        torch.from_numpy(
            data[i + 1:i + BLOCK_SIZE + 1].astype(np.int64)
        )
        for i in ix
    ])

    return x.to(DEVICE), y.to(DEVICE)
