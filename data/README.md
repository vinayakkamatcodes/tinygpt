# Dataset

This directory contains documentation related to the training and validation datasets.

## Dataset Source

- Name: FineWeb-Edu
- Provider: Hugging Face
- Repository: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- Configuration: `sample-10BT`

## Dataset Processing

The initial corpus preparation pipeline performs:

1. Streaming dataset access.
2. English-language filtering.
3. Language-score filtering.
4. Minimum document-length filtering.
5. Whitespace normalization.
6. Exact duplicate removal using SHA-256 hashes.
7. Document-level training and validation splitting.

## Initial Dataset Statistics

| Split | Documents |
|---|---:|
| Training | 9,500 |
| Validation | 500 |
| Total | 10,000 |

The initial corpus contains approximately 12.1 million tokens according to the custom BPE tokenizer before the addition of document-ending tokens.

## Data Files

The generated files are intentionally excluded from Git:

- `train.jsonl`
- `validation.jsonl`
- `train.bin`
- `val.bin`

Run the preprocessing scripts to recreate them.

## Future Improvements

- Randomized document sampling.
- Reproducible train/validation splitting.
- Improved document-quality filtering.
- Near-duplicate detection.
- Better document-boundary handling.

## Attribution

FineWeb-Edu is maintained by Hugging Face. Review the dataset card and applicable source-document rights before redistributing data or trained model weights.