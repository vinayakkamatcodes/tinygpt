
# Model configuration

VOCAB_SIZE = 8000

D_MODEL = 640
N_LAYERS = 8
N_HEADS = 10
BLOCK_SIZE = 256
DROPOUT = 0.1

# Feed-forward expansion
FFN_DIM = 4 * D_MODEL

# Training configuration (initial values)
BATCH_SIZE = 8
GRAD_ACCUMULATION_STEPS = 4

LEARNING_RATE = 3e-4
WEIGHT_DECAY = 0.1

MAX_ITERS = 10000
EVAL_INTERVAL = 5
EVAL_BATCHES = 50

DEVICE = "cuda"
