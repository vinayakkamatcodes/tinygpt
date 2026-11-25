
from tokenizers import Tokenizer
import json
from tqdm import tqdm

tokenizer = Tokenizer.from_file("tokenizer.json")

def count_tokens(filename):
    total_tokens = 0
    total_documents = 0

    with open(filename, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc=filename):
            document = json.loads(line)

            encoded = tokenizer.encode(document["text"])

            total_tokens += len(encoded.ids)
            total_documents += 1

    return total_documents, total_tokens


train_docs, train_tokens = count_tokens("train.jsonl")
val_docs, val_tokens = count_tokens("validation.jsonl")

print("\n========== TOKEN STATISTICS ==========")
print("Training documents:", train_docs)
print("Training tokens:", f"{train_tokens:,}")

print("Validation documents:", val_docs)
print("Validation tokens:", f"{val_tokens:,}")

print("Total tokens:", f"{train_tokens + val_tokens:,}")
