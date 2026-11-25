# Baseline Evaluation — 500 Training Steps

## Model Architecture
- Parameters: 49,801,280 (~49.8M)
- Architecture: Decoder-only Transformer
- Vocabulary size: 8,000
- Transformer layers: 8
- Attention heads: 10
- Embedding dimension: 640
- Context length: 256

## Dataset
- Source: FineWeb-Edu (`sample-10BT`)
- Training documents: 9,500
- Validation documents: 500
- Training tokens: 11,551,383
- Validation tokens: 568,228
- Tokenizer: ByteLevel BPE
- Tokenizer vocabulary: 8,000

## Training Configuration
- Optimizer: AdamW
- Learning rate: 3e-4
- Training steps: 500
- Batch size: 8
- Gradient accumulation steps: 4
- Effective batch size: 32
- Precision: BF16
- Hardware: NVIDIA RTX 4060 Laptop GPU (8 GB VRAM)

## Evaluation Results

| Metric | Result |
|---|---:|
| Training loss | 5.6412 |
| Validation loss | 5.6948 |
| Training perplexity | 281.81 |
| Validation perplexity | 297.32 |
| Loss gap | 0.0536 |

## Observations
- The model successfully completed 500 training steps.
- Training and validation losses decreased throughout the run.
- Training and validation losses are relatively close.
- Validation perplexity remains high.
- Generated text is not yet coherent.
- The model requires substantially more pretraining.

## Next Steps
1. Improve corpus sampling and document splitting.
2. Establish a reliable evaluation protocol.
3. Implement checkpoint-resume training.
4. Introduce learning-rate scheduling.
5. Continue pretraining and monitor validation loss.