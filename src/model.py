
import torch
import torch.nn as nn
import torch.nn.functional as F

from configs import config


# ==========================================
# 1. Multi-Head Causal Self-Attention
# ==========================================

class CausalSelfAttention(nn.Module):

    def __init__(self):
        super().__init__()

        assert config.D_MODEL % config.N_HEADS == 0

        self.head_dim = config.D_MODEL // config.N_HEADS

        # Generate Query, Key and Value together
        self.qkv = nn.Linear(
            config.D_MODEL,
            3 * config.D_MODEL
        )

        # Combine attention heads
        self.proj = nn.Linear(
            config.D_MODEL,
            config.D_MODEL
        )

        self.attn_dropout = config.DROPOUT
        self.resid_dropout = nn.Dropout(config.DROPOUT)

    def forward(self, x):

        B, T, C = x.shape

        # Generate Q, K, V
        q, k, v = self.qkv(x).split(C, dim=2)

        # Reshape into multiple attention heads
        q = q.view(B, T, config.N_HEADS, self.head_dim)
        k = k.view(B, T, config.N_HEADS, self.head_dim)
        v = v.view(B, T, config.N_HEADS, self.head_dim)

        # Move heads before sequence dimension
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product causal attention
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            is_causal=True,
            dropout_p=(
                self.attn_dropout if self.training else 0.0
            )
        )

        # Combine attention heads
        y = y.transpose(1, 2).contiguous()
        y = y.view(B, T, C)

        y = self.proj(y)
        y = self.resid_dropout(y)

        return y


# ==========================================
# 2. Feed-Forward Network
# ==========================================

class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(config.D_MODEL, config.FFN_DIM),
            nn.GELU(),
            nn.Linear(config.FFN_DIM, config.D_MODEL),
            nn.Dropout(config.DROPOUT)
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 3. Transformer Block
# ==========================================

class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.D_MODEL)
        self.ln2 = nn.LayerNorm(config.D_MODEL)

        self.attention = CausalSelfAttention()
        self.feed_forward = FeedForward()

    def forward(self, x):

        # Pre-LayerNorm + residual connection
        x = x + self.attention(self.ln1(x))

        x = x + self.feed_forward(self.ln2(x))

        return x


# ==========================================
# 4. Complete GPT Model
# ==========================================

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # Token embeddings
        self.token_embedding = nn.Embedding(
            config.VOCAB_SIZE,
            config.D_MODEL
        )

        # Positional embeddings
        self.position_embedding = nn.Embedding(
            config.BLOCK_SIZE,
            config.D_MODEL
        )

        self.embedding_dropout = nn.Dropout(config.DROPOUT)

        # Stack transformer blocks
        self.blocks = nn.Sequential(
            *[
                TransformerBlock()
                for _ in range(config.N_LAYERS)
            ]
        )

        # Final normalization
        self.final_ln = nn.LayerNorm(config.D_MODEL)

        # Language-model output head
        self.lm_head = nn.Linear(
            config.D_MODEL,
            config.VOCAB_SIZE,
            bias=True
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):

        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):

        B, T = idx.shape

        if T > config.BLOCK_SIZE:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"BLOCK_SIZE {config.BLOCK_SIZE}"
            )

        # Position indices
        positions = torch.arange(
            T,
            device=idx.device
        )

        # Token + positional embeddings
        token_emb = self.token_embedding(idx)
        position_emb = self.position_embedding(positions)

        x = token_emb + position_emb
        x = self.embedding_dropout(x)

        # Transformer layers
        x = self.blocks(x)

        # Final normalization
        x = self.final_ln(x)

        # Vocabulary logits
        logits = self.lm_head(x)

        loss = None

        # Next-token prediction loss
        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(-1, config.VOCAB_SIZE),
                targets.reshape(-1)
            )

        return logits, loss
