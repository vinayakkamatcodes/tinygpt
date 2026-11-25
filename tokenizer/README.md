# Tokenizer

This directory contains the custom Byte-Level Byte Pair Encoding (BPE) tokenizer used by the General English GPT model.

## Implementation

The tokenizer is trained using the Hugging Face `tokenizers` library.

## Configuration

| Setting | Value |
|---|---|
| Tokenization Algorithm | BPE |
| Pre-tokenization | ByteLevel |
| Decoder | ByteLevel |
| Normalization | NFKC |
| Vocabulary Size | 8,000 |
| Minimum Frequency | 2 |

## Special Tokens

- `[PAD]`: Padding token
- `[UNK]`: Unknown token
- `[BOS]`: Beginning-of-sequence token
- `[EOS]`: End-of-sequence token

## Training

Run the tokenizer training script from the repository root:

```bash
python tokenizer/train_tokenizer.py
```

The script uses the prepared training corpus.

## Output

`tokenizer.json` contains:

- Learned vocabulary.
- BPE merge rules.
- Token IDs.
- Normalization configuration.
- Pre-tokenizer configuration.
- Decoder configuration.

## Important

The tokenizer and language model serve different purposes:

- The tokenizer converts text into token IDs.
- The GPT model learns patterns from sequences of token IDs and predicts the next token.