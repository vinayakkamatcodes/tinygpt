
from datasets import load_dataset
from tqdm import tqdm
import json
import hashlib

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True
)

MAX_DOCUMENTS = 10_000
MIN_TOKENS = 100
MIN_LANGUAGE_SCORE = 0.80

seen_hashes = set()
clean_documents = []

for document in tqdm(dataset, total=MAX_DOCUMENTS):

    text = document["text"]

    # Basic validation
    if not text or not text.strip():
        continue

    # Language filtering
    if document["language"] != "en":
        continue

    if document["language_score"] < MIN_LANGUAGE_SCORE:
        continue

    # Length filtering
    if document["token_count"] < MIN_TOKENS:
        continue

    # Normalize whitespace
    text = " ".join(text.split())

    # Duplicate detection
    content_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    if content_hash in seen_hashes:
        continue

    seen_hashes.add(content_hash)
    clean_documents.append({
        "text": text,
        "id": document["id"],
        "url": document["url"],
        "token_count": document["token_count"],
        "score": document["score"]
    })

    if len(clean_documents) >= MAX_DOCUMENTS:
        break

print("\nClean documents:", len(clean_documents))

# Document-level split
split_index = int(len(clean_documents) * 0.95)

train_documents = clean_documents[:split_index]
validation_documents = clean_documents[split_index:]

# Save files
with open("train.jsonl", "w", encoding="utf-8") as f:
    for document in train_documents:
        f.write(json.dumps(document, ensure_ascii=False) + "\n")

with open("validation.jsonl", "w", encoding="utf-8") as f:
    for document in validation_documents:
        f.write(json.dumps(document, ensure_ascii=False) + "\n")

print("Training documents:", len(train_documents))
print("Validation documents:", len(validation_documents))
