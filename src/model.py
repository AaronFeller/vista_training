import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from typing import Mapping
from transformers.tokenization_utils_base import BatchEncoding
from transformers import PretrainedConfig

class model_config(PretrainedConfig):
    model_type = "MLM_model"
    def __init__(
        self,
        ffn_hidden_dim = 2048,
        embed_dim = 1024,
        num_heads = 16,
        num_blocks = 32,
        vocab_size = 405,
        output_dim = 405,
        max_seq_len = 2048,
        size = "large",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.ffn_hidden_dim = ffn_hidden_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.vocab_size = vocab_size
        self.output_dim = output_dim
        self.max_seq_len = max_seq_len
        self.size = size


class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim * 2, bias=True)
        self.linear2 = nn.Linear(hidden_dim, input_dim, bias=True)
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization
    def forward(self, x):
        # x: (N, input_dim)
        x1, x2 = self.linear1(x).chunk(2, dim=-1)
        output = self.linear2(F.silu(x1) * x2)
        return self.dropout(output)

class RotaryPositionalEmbeddingsCustom(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        assert head_dim % 2 == 0, "Head dimension must be even for RoPE pairs"
        self.head_dim = head_dim
        self.base = base
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (self.base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq)

        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, head_dim//2)

        cos = freqs.cos().unsqueeze(0).unsqueeze(2)  # ⇒ (1, seq_len, 1, head_dim//2)
        sin = freqs.sin().unsqueeze(0).unsqueeze(2)

        # Register buffers
        self.register_buffer("cos_cached", cos)  # (1, max_seq_len, 1, head_dim//2)
        self.register_buffer("sin_cached", sin)

        self.max_seq_len = seq_len

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor = None) -> torch.Tensor:
        # x: (B, seq_len, num_heads, head_dim)
        B, seq_len, num_heads, head_dim = x.shape
        assert head_dim == self.head_dim, f"Head dim mismatch {head_dim} vs {self.head_dim}"

        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        cos = self.cos_cached[:, :seq_len]   # (1, seq_len, 1, head_dim//2)
        sin = self.sin_cached[:, :seq_len]

        # split pairs
        x = x.view(B, seq_len, num_heads, head_dim//2, 2)
        x1, x2 = x[..., 0], x[..., 1]  # both (B, seq_len, num_heads, head_dim//2)

        # apply rotation
        x_rotated = torch.stack([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)  # (B, seq_len, num_heads, head_dim//2, 2)

        return x_rotated.flatten(-2)  # (B, seq_len, num_heads, head_dim)


class UnifiedTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, max_seq_len):
        super().__init__()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, max_seq_len)
        self.ffn_norm = nn.LayerNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, ffn_hidden_dim)

    def forward(self, x, input_pos=None, mask=None):
        x = x + self.attn(self.attn_norm(x), input_pos=input_pos, mask=mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x

class TransformerStack(nn.Module):
    def __init__(self, num_blocks, embed_dim, num_heads, ffn_hidden_dim, max_seq_len):
        super().__init__()
        self.blocks = nn.ModuleList([
            UnifiedTransformerBlock(embed_dim, num_heads, ffn_hidden_dim, max_seq_len)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x, input_pos=None, mask=None):
        for block in self.blocks:
            x = block(x, input_pos=input_pos, mask=mask)
        return self.norm(x)


class MLM_model(PreTrainedModel): # HF-facing class name
    config_class = model_config

    def __init__(self, config):
        super().__init__(config)
        self.model = MLM_core(
            vocab_size=config.vocab_size,
            embed_dim=config.embed_dim,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            ffn_hidden_dim=config.ffn_hidden_dim,
            output_dim=config.output_dim,
            max_seq_len=config.max_seq_len,
        )
        self.post_init()  # Initialize weights and apply final processing
    
    # if inputs are dictionary
    def forward(self, x=None, **kwargs):
        if isinstance(x, (BatchEncoding, Mapping)):
            return self.model(x.get("input_ids"), mask=x.get("attention_mask"))
        
        if "input_ids" in kwargs or "attention_mask" in kwargs:
            return self.model(kwargs.get("input_ids"), mask=kwargs.get("attention_mask"))
        
        return self.model(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, max_seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.rotary = RotaryPositionalEmbeddingsCustom(head_dim=self.head_dim, max_seq_len=max_seq_len)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.dropout = nn.Dropout(0.1)  # Add dropout for regularization

    def forward(self, x, input_pos=None, mask=None):
        B, T, C = x.shape  # Batch, sequence, embedding dim
        
        # project into queries, keys, and values
        q, k, v = self.qkv_proj(x).view(B, T, 3, self.num_heads, self.head_dim).unbind(2)  # (B, T, num_heads, head_dim)

        # Apply rotary positional embeddings to queries and keys
        q, k = self.rotary(q, input_pos=input_pos), self.rotary(k, input_pos=input_pos)

        # Reshape to (B, num_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if mask is not None:
            # set padding positions to -inf
            mask = mask.to(dtype=torch.float32)  # Ensure mask is float
            mask = (1.0 - mask) * -1e9  # Convert to -inf for padding positions
            
            # mask: (B, T) -> (B, 1, 1, T)
            mask = mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
            mask = mask.expand(B, 1, T, T)  # expands to (batch, 1, seqlen, seqlen)
            
        # Scaled dot-product attention
        attn_output = F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        attn_output = self.out_proj(attn_output)
        return self.dropout(attn_output)


class MLM_core(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_blocks: int,
        num_heads: int,
        ffn_hidden_dim: int,
        output_dim: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.transformer = TransformerStack(
            num_blocks, embed_dim, num_heads, ffn_hidden_dim, max_seq_len
        )
        self.sequence_head = nn.Linear(embed_dim, output_dim, bias=True)


    def forward(self, ids, mask=None, pad_token_id=0, input_pos=None):
        x = self.embed(ids)
        x = self.transformer(x, mask=mask, input_pos=input_pos)
        # generate logits for MLM
        # print(f"x shape: {x.shape}")  # Debugging line to check the shape of x
        logits = self.sequence_head(x)
        # print(f"logits shape: {logits.shape}")  # Debugging line to check the shape of logits

        # mean pool but remove positions that have pad tokens
        # mean_pool = x.masked_fill(ids.unsqueeze(-1) == pad_token_id, 0).mean(dim=1)

        # outputs = {
        #     'logits': logits,
        #     'last_layer': x,
        #     'mean_pool': mean_pool
        # }

        return logits

