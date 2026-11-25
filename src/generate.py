
import torch
from tokenizers import Tokenizer
from configs import config
from src.model import GPTLanguageModel

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# ==========================================
# 1. Setup
# ==========================================

device = torch.device(
    config.DEVICE if torch.cuda.is_available() else "cpu"
)

tokenizer = Tokenizer.from_file(
    str(PROJECT_ROOT / "tokenizer" / "tokenizer.json")
)
checkpoint = torch.load(
    PROJECT_ROOT / "checkpoints" / "gpt_checkpoint.pt",
    map_location=device,
    weights_only=True
)

model = GPTLanguageModel().to(device)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully!")


# ==========================================
# 2. Text generation
# ==========================================

@torch.no_grad()
def generate_text(
    prompt,
    max_new_tokens=150,
    temperature=0.8,
    top_k=40
):

    encoded = tokenizer.encode(prompt)

    input_ids = torch.tensor(
        [encoded.ids],
        dtype=torch.long,
        device=device
    )

    for _ in range(max_new_tokens):

        # Keep context within model limit
        idx = input_ids[:, -config.BLOCK_SIZE:]

        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=(
                device.type == "cuda"
                and torch.cuda.is_bf16_supported()
            )
        ):

            logits, _ = model(idx)

        # Get logits for the last position
        logits = logits[:, -1, :]

        # Temperature
        logits = logits / temperature

        # Top-k sampling
        values, _ = torch.topk(logits, top_k)

        logits[logits < values[:, [-1]]] = float("-inf")

        probabilities = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probabilities,
            num_samples=1
        )

        input_ids = torch.cat(
            (input_ids, next_token),
            dim=1
        )

        # Stop at EOS
        if next_token.item() == tokenizer.token_to_id("[EOS]"):
            break

    return tokenizer.decode(
        input_ids[0].tolist()
    )


# ==========================================
# 3. Generate samples
# ==========================================

prompts = [
    "Artificial intelligence is",
    "The history of science",
    "Once upon a time"
]

for prompt in prompts:

    print("\n" + "=" * 60)
    print("PROMPT:", prompt)
    print("-" * 60)

    output = generate_text(prompt)

    print(output)
