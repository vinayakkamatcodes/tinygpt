# Baseline Experiment: 500 Training Updates

## Overview

This experiment represents the first successful training run of the 49.8M-parameter General English GPT model.

## Configuration

- Model parameters: 49,801,280
- Dataset: Initial FineWeb-Edu subset
- Vocabulary size: 8,000
- Context length: 256
- Batch size: 8
- Gradient accumulation steps: 4
- Effective batch size: 32
- Optimizer: AdamW
- Learning rate: 3e-4
- Weight decay: 0.1
- Precision: BF16
- Gradient clipping: 1.0
- Training updates: 500

## Training Results

| Update | Training Loss | Validation Loss |
|---:|---:|---:|
| 0 | 8.4904 | 8.5084 |
| 50 | 7.1110 | 7.1081 |
| 100 | 6.5785 | 6.5826 |
| 150 | 6.3570 | 6.3709 |
| 200 | 6.1998 | 6.2263 |
| 250 | 6.0869 | 6.1108 |
| 300 | 5.9945 | 6.0214 |
| 350 | 5.8971 | 5.9241 |
| 400 | 5.8058 | 5.8625 |
| 450 | 5.7201 | 5.7582 |

The model completed 500 optimizer updates. The final logged evaluation was at update 450.

## Performance

- Total duration: Approximately 190 seconds.
- Average reported time per iteration: Approximately 0.38 seconds.
- Peak GPU memory: Approximately 1.99 GB.

The reported average includes periodic evaluation overhead.

## Generation Observations

The trained model can generate text from prompts.

Observed limitations:

- Repetitive phrases.
- Inconsistent grammar.
- Weak long-range coherence.
- Occasional malformed words.

These results are expected for an early training experiment.

## Conclusion

The experiment confirms that the Transformer implementation, tokenizer, data loader, GPU training pipeline, checkpoint saving, and generation pipeline work together.

Further experiments will focus on data quality, evaluation, and extended pretraining.