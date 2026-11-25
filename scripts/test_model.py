
import torch

import config
from model import GPTLanguageModel

# Check CUDA
device = torch.device(
    config.DEVICE if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

# Create model
model = GPTLanguageModel().to(device)

# Count parameters
total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print("\nTotal parameters:", f"{total_params:,}")
print("Trainable parameters:", f"{trainable_params:,}")

# Create dummy input and target
x = torch.randint(
    0,
    config.VOCAB_SIZE,
    (config.BATCH_SIZE, config.BLOCK_SIZE),
    device=device
)

y = torch.randint(
    0,
    config.VOCAB_SIZE,
    (config.BATCH_SIZE, config.BLOCK_SIZE),
    device=device
)

# Forward pass
logits, loss = model(x, y)

print("\nInput shape:", x.shape)
print("Logits shape:", logits.shape)
print("Loss:", loss.item())

# Backward pass test
loss.backward()

print("\nBackward pass successful!")

# GPU memory
if device.type == "cuda":
    print(
        "GPU memory allocated:",
        round(torch.cuda.memory_allocated() / 1024**3, 2),
        "GB"
    )
