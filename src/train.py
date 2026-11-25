
import time
import torch
import torch.nn as nn

from configs import config
from src.model import GPTLanguageModel
from src.data_loader import get_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)


# ==========================================
# 1. Setup
# ==========================================

device = torch.device(
    config.DEVICE if torch.cuda.is_available() else "cpu"
)

torch.manual_seed(42)

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

model = GPTLanguageModel().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY
)

use_bf16 = (
    device.type == "cuda"
    and torch.cuda.is_bf16_supported()
)

print("Device:", device)
print("BF16 enabled:", use_bf16)

total_params = sum(
    p.numel() for p in model.parameters()
)

print("Total parameters:", f"{total_params:,}")


# ==========================================
# 2. Evaluation function
# ==========================================

@torch.no_grad()
def estimate_loss():

    model.eval()

    results = {}

    for split in ["train", "val"]:

        losses = []

        for _ in range(config.EVAL_BATCHES):

            x, y = get_batch(split)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16
            ):

                _, loss = model(x, y)

            losses.append(loss.item())

        results[split] = sum(losses) / len(losses)

    model.train()

    return results


# ==========================================
# 3. Training configuration
# ==========================================

MAX_ITERS = 500
EVAL_INTERVAL = 50

model.train()

optimizer.zero_grad(set_to_none=True)

# Start measuring training time
start_time = time.perf_counter()


# ==========================================
# 4. Training loop
# ==========================================

for iteration in range(MAX_ITERS):

    accumulated_loss = 0.0

    for micro_step in range(config.GRAD_ACCUMULATION_STEPS):

        x, y = get_batch("train")

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16
        ):

            _, loss = model(x, y)

            loss = loss / config.GRAD_ACCUMULATION_STEPS

        # Backpropagation
        loss.backward()

        accumulated_loss += loss.item()

    # Gradient clipping
    nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=1.0
    )

    # Update model parameters
    optimizer.step()

    optimizer.zero_grad(set_to_none=True)

    # Evaluate periodically
    if iteration % EVAL_INTERVAL == 0:

        losses = estimate_loss()

        print(
            f"Step {iteration:4d} | "
            f"Train Loss: {losses['train']:.4f} | "
            f"Val Loss: {losses['val']:.4f} | "
            f"Batch Loss: {accumulated_loss:.4f}"
        )


# ==========================================
# 5. Save model checkpoint
# ==========================================

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),

        "config": {
            "vocab_size": config.VOCAB_SIZE,
            "d_model": config.D_MODEL,
            "n_layers": config.N_LAYERS,
            "n_heads": config.N_HEADS,
            "block_size": config.BLOCK_SIZE
        }
    },
    CHECKPOINT_DIR / "gpt_checkpoint.pt"
)


# ==========================================
# 6. Training statistics
# ==========================================

elapsed = time.perf_counter() - start_time

print("\n========== TRAINING SUMMARY ==========")

print("Training iterations:", MAX_ITERS)
print("Checkpoint saved:", CHECKPOINT_DIR / "gpt_checkpoint.pt")

print(
    f"Total training time: {elapsed:.2f} seconds"
)

print(
    f"Average time per iteration: "
    f"{elapsed / MAX_ITERS:.3f} seconds"
)

if device.type == "cuda":

    peak_memory = (
        torch.cuda.max_memory_allocated() / 1024**3
    )

    print(
        f"Peak GPU memory: {peak_memory:.2f} GB"
    )

print("\nTraining experiment completed successfully!")
