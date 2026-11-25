
import json
import numpy as np
from tokenizers import Tokenizer
from tqdm import tqdm

tokenizer = Tokenizer.from_file("tokenizer.json")

eos_id = tokenizer.token_to_id("[EOS]")

if eos_id is None:
    raise ValueError("[EOS] token not found!")

def convert(input_file, output_file):
    total_tokens = 0

    with open(input_file, "r", encoding="utf-8") as source:
        with open(output_file, "wb") as destination:

            for line in tqdm(source, desc=output_file):
                document = json.loads(line)

                ids = tokenizer.encode(document["text"]).ids

                # Add document separator
                ids.append(eos_id)

                # Save compact 16-bit token IDs
                token_array = np.asarray(ids, dtype=np.uint16)
                token_array.tofile(destination)

                total_tokens += len(ids)

    print(f"\nSaved: {output_file}")
    print(f"Total tokens including EOS: {total_tokens:,}")

convert("train.jsonl", "train.bin")
convert("validation.jsonl", "val.bin")
