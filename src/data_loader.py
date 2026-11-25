
import numpy as np
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Training configuration
BATCH_SIZE = 16
BLOCK_SIZE = 256

# Load binary token files
train_data = np.memmap(
    "train.bin",
    dtype=np.uint16,
    mode="r"
)

val_data = np.memmap(
    "val.bin",
    dtype=np.uint16,
    mode="r"
)

def get_batch(split):
    data = train_data if split == "train" else val_data

    # Random starting positions
    ix = torch.randint(
        len(data) - BLOCK_SIZE - 1,
        (BATCH_SIZE,)
    )

    # Extract input and shifted target sequences
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
    x = x.to(DEVICE)
    y = y.to(DEVICE)



    return x, y


# # Test the batch generator
# x, y = get_batch("train")

# print("Input shape:", x.shape)
# print("Target shape:", y.shape)

# print("\nFirst input sequence:")
# print(x[0][:10])

# print("\nFirst target sequence:")
# print(y[0][:10])

# print("\nData type:", x.dtype)
