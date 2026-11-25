
import json

with open("tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer_data = json.load(f)

bpe_model = tokenizer_data["model"]

print("Model type:", bpe_model["type"])

print("\nFirst 20 merge rules:")

for rank, merge in enumerate(bpe_model["merges"][:20]):
    print(f"Rank {rank}: {merge}")

print("\nTotal merge rules:", len(bpe_model["merges"]))
