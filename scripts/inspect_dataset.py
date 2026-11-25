
from datasets import load_dataset

# Load a small streaming sample
dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True
)

# Inspect the first 3 documents
for i, document in enumerate(dataset):
    print(f"\nDOCUMENT {i + 1}")
    print("-" * 60)

    print("Available fields:", document.keys())
    print("Text:", document["text"][:500])

    if i == 2:
        break
