
import json

def inspect_file(filename):
    documents = 0
    recorded_tokens = 0
    total_characters = 0

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            document = json.loads(line)

            documents += 1
            recorded_tokens += document["token_count"]
            total_characters += len(document["text"])

    print(f"\nFile: {filename}")
    print("Documents:", documents)
    print("Recorded tokens:", f"{recorded_tokens:,}")
    print("Characters:", f"{total_characters:,}")

    return recorded_tokens

train_tokens = inspect_file("train.jsonl")
val_tokens = inspect_file("validation.jsonl")

print("\nTotal recorded tokens:", f"{train_tokens + val_tokens:,}")
