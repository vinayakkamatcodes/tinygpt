# Data Preparation Scripts

This directory contains utilities used to prepare the training corpus and tokenized datasets.

## Scripts

### `prepare_dataset.py`

Downloads and filters the FineWeb-Edu dataset.

Responsibilities:

- Read dataset documents.
- Filter English-language content.
- Apply document-quality and length filters.
- Remove exact duplicates.
- Create training and validation JSONL files.

### `prepare_tokens.py`

Converts prepared text documents into token IDs using the trained BPE tokenizer.

Responsibilities:

- Load the tokenizer.
- Encode documents.
- Append document-ending tokens.
- Save token IDs in binary format.

### `measure_tokens.py`

Measures token counts for the prepared corpus using the custom tokenizer.

## Generated Files

The scripts generate:

- `train.jsonl`
- `validation.jsonl`
- `train.bin`
- `val.bin`

These generated datasets are excluded from version control.

## Execution

Run scripts from the repository root:

```bash
python scripts/prepare_dataset.py
python tokenizer/train_tokenizer.py
python scripts/measure_tokens.py
python scripts/prepare_tokens.py
```

Check the individual script configurations before execution. Some preprocessing steps may take time and download data from Hugging Face.