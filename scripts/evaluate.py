
import math
from pathlib import Path

import torch

from configs import config
from src.model import GPTLanguageModel
from src.data_loader import get_batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "gpt_checkpoint.pt"

device = torch.device(
    config.DEVICE if torch.cuda.is_available() else "cpu"
)

use_bf16 = (
    device.type == "cuda"
    and torch.cuda.is_bf16_supported()
)


def load_model():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True
    )

    model = GPTLanguageModel().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


@torch.no_grad()
def evaluate(model, split, batches=100):
    losses = []

    for _ in range(batches):
        x, y = get_batch(split)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16
        ):
            _, loss = model(x, y)

        losses.append(loss.item())

    average_loss = sum(losses) / len(losses)
    perplexity = math.exp(average_loss)

    return average_loss, perplexity


def main():
    print("Device:", device)
    print("BF16 enabled:", use_bf16)
    print("Loading checkpoint...")

    model = load_model()

    total_params = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("Model loaded successfully!")
    print("Parameters:", f"{total_params:,}")

    print("\nEvaluating model...")

    train_loss, train_ppl = evaluate(model, "train")
    val_loss, val_ppl = evaluate(model, "val")

    print("\n========== EVALUATION RESULTS ==========")
    print(f"Train Loss:       {train_loss:.4f}")
    print(f"Train Perplexity: {train_ppl:.2f}")
    print("----------------------------------------")
    print(f"Val Loss:         {val_loss:.4f}")
    print(f"Val Perplexity:   {val_ppl:.2f}")
    print("========================================")


if __name__ == "__main__":
    main()
