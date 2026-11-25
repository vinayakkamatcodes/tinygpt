
from datasets import load_dataset

dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
    streaming=True
)

for i, document in enumerate(dataset):
    print(f"\nDocument {i + 1}")
    print("Language:", repr(document["language"]))
    print("Language score:", repr(document["language_score"]))
    print("Token count:", repr(document["token_count"]))
    print("Quality score:", repr(document["score"]))

    if i == 9:
        break
