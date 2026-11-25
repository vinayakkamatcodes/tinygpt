# General English GPT

A GPT-style decoder-only Transformer language model built from scratch using PyTorch.

This project focuses on understanding and implementing the complete language-model development pipeline, including tokenizer training, dataset preparation, Transformer architecture, pretraining, and text generation.

## Project Overview

The goal is to develop a 50-million-parameter general English language model from random initialization.

Unlike projects that simply fine-tune an existing pretrained model, this project implements and trains the Transformer architecture from scratch.

## Model Architecture

| Component | Configuration |
|---|---|
| Architecture | Decoder-only Transformer |
| Parameters | 49.8 Million |
| Framework | PyTorch |
| Tokenizer | Byte-Level BPE |
| Vocabulary Size | 8,000 |
| Transformer Layers | 8 |
| Attention Heads | 10 |
| Embedding Dimension | 640 |
| Head Dimension | 64 |
| Feed-Forward Dimension | 2,560 |
| Context Length | 256 |
| Activation | GELU |
| Normalization | Pre-LayerNorm |
| Attention | Causal Scaled Dot-Product Attention |

The input token embeddings and output language-model head are currently untied.

## Dataset

The initial training corpus is derived from:

- **Dataset:** FineWeb-Edu
- **Source:** Hugging Face
- **Configuration:** sample-10BT
- **Initial filtered documents:** Approximately 10,000
- **Training documents:** 9,500
- **Validation documents:** 500
- **Tokenizer:** Custom Byte-Level BPE tokenizer

The current corpus is an initial experimental subset. Dataset sampling, filtering, and reproducibility will be improved in subsequent experiments.

## Tokenizer

A custom Byte-Level BPE tokenizer is trained using the Hugging Face `tokenizers` library.

Configuration:

- Vocabulary size: 8,000
- Special tokens: `[PAD]`, `[UNK]`, `[BOS]`, `[EOS]`
- Normalization: NFKC
- Pre-tokenizer: ByteLevel
- Decoder: ByteLevel

The trained tokenizer is saved as `tokenizer/tokenizer.json`.

## Initial Training Experiment

The first training experiment was completed successfully.

| Metric | Result |
|---|---:|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Training Updates | 500 |
| Batch Size | 8 |
| Gradient Accumulation | 4 |
| Effective Batch Size | 32 |
| Context Length | 256 |
| Precision | BF16 |
| Last Logged Training Loss | 5.7201 |
| Last Logged Validation Loss | 5.7582 |
| Training Duration | Approximately 190 seconds |

Note: Loss values were last evaluated at update 450. The model completed 500 optimizer updates.

## Hardware

Initial experiments were performed on:

- NVIDIA GeForce RTX 4060 Laptop GPU
- 8 GB GPU memory
- 16 GB system RAM
- PyTorch with CUDA support

## Project Structure

```text
general-english-gpt/
├── configs/          # Model and training configuration
├── src/              # Model, training, generation and data loading
├── tokenizer/        # BPE tokenizer and documentation
├── data/             # Dataset instructions and documentation
├── scripts/          # Dataset and token preprocessing utilities
├── experiments/      # Experiment configurations and results
├── checkpoints/      # Local model checkpoints
├── README.md
├── requirements.txt
└── .gitignore
```

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/general-english-gpt.git
cd general-english-gpt
```

### 2. Create and activate an environment

```bash
conda create -n tinygpt python=3.11 -y
conda activate tinygpt
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the dataset

Follow the instructions in `data/README.md`.

### 5. Train the tokenizer

Follow the instructions in `tokenizer/README.md`.

### 6. Prepare tokenized data

Run the dataset preparation scripts described in `scripts/README.md`.

### 7. Train the model

```bash
python src/train.py
```

### 8. Generate text

```bash
python src/generate.py
```

Run commands from the repository root. If the implementation uses different paths or import conventions, the commands will be updated to match the final code structure.

## Current Status

**Initial baseline training completed.**

The model successfully generates text from prompts, but generation quality is currently limited by the small training corpus and short training run.

This is an ongoing educational and engineering project.

## Roadmap

- [x] Implement decoder-only Transformer architecture
- [x] Implement causal multi-head self-attention
- [x] Train custom BPE tokenizer
- [x] Prepare initial FineWeb-Edu corpus
- [x] Implement binary tokenization and data loading
- [x] Train initial 49.8M-parameter model
- [x] Generate text from trained checkpoint
- [ ] Implement formal evaluation and perplexity
- [ ] Improve dataset sampling and quality
- [ ] Implement learning-rate scheduling
- [ ] Extend model pretraining
- [ ] Compare training experiments
- [ ] Publish model weights and training documentation

## Disclaimer

This is an educational language-model research project. The initial model is not intended to compete with production-grade or frontier language models.

## License

License and dataset attribution details will be finalized before public model-weight distribution.