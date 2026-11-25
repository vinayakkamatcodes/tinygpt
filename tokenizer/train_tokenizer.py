
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, normalizers
import json

# Special tokens
special_tokens = [
    "[PAD]",
    "[UNK]",
    "[BOS]",
    "[EOS]"
]

# Create BPE tokenizer
tokenizer = Tokenizer(
    models.BPE(unk_token="[UNK]")
)

# Normalize Unicode text
tokenizer.normalizer = normalizers.NFKC()

# Byte-level processing
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(
    add_prefix_space=False
)

# Configure trainer
trainer = trainers.BpeTrainer(
    vocab_size=8000,
    min_frequency=2,
    special_tokens=special_tokens,
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
)

# Stream training text from JSONL
def text_iterator(filename):
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            document = json.loads(line)
            yield document["text"]

# Train tokenizer using training documents only
tokenizer.train_from_iterator(
    text_iterator("train.jsonl"),
    trainer=trainer
)

# Configure decoder
tokenizer.decoder = decoders.ByteLevel()

# Save tokenizer
tokenizer.save("tokenizer.json")

print("Tokenizer training completed!")
print("Vocabulary size:", tokenizer.get_vocab_size())

# Test encoding
sample = "Transformers learn language through attention mechanisms."

encoded = tokenizer.encode(sample)

print("\nOriginal text:", sample)
print("Token IDs:", encoded.ids)
print("Tokens:", encoded.tokens)
print("Decoded:", tokenizer.decode(encoded.ids))
