import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum
import math
from typing import Optional

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device="cpu", dtype=None):
        super().__init__()
        self.device = device if device is not None else "cpu"
        self.W = nn.Parameter(torch.randn((out_features, in_features), dtype=dtype, device=device))
        sigma = math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(self.W, mean=0, std=sigma, a=-3 * sigma, b=3 * sigma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = einsum(x, self.W, "... i, o i -> ... o")
        return out
    
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.emb = nn.Parameter(torch.randn((num_embeddings, embedding_dim), device=device, dtype=dtype))
        nn.init.trunc_normal_(self.emb, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.emb[token_ids]
    
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.gamma = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = (x * self.gamma) / rms
        return out.to(in_dtype)
    
class SwiFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()        
        ls = int((8 * d_model / 3) // 64)
        d_ff1 = ls * 64
        d_ff = d_ff1 if d_ff is None else d_ff
        self.fc1 = Linear(in_features=d_model, out_features=d_ff)
        self.fc2 = Linear(in_features=d_ff, out_features=d_model)
        self.fc_gate = Linear(in_features=d_model, out_features=d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.fc1(x)
        act = x1 * F.sigmoid(x1)
        x2 = self.fc_gate(x)
        gated = act * x2
        out = self.fc2(gated)
        return out
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        theta_k = theta ** (-torch.arange(0, d_k, 2) / d_k)
        pos = torch.arange(max_seq_len)
        thetaik = pos[:, None] * theta_k[None, :]
        self.register_buffer("cos", torch.cos(thetaik), persistent=False)
        self.register_buffer("sin", torch.sin(thetaik), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        rot_even = x_even * cos - x_odd * sin
        rot_odd = x_even * sin + x_odd * cos
        out = torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)
        return out
    
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    x1 = x - torch.max(x, dim=dim, keepdim=True)[0]
    expx = torch.exp(x1)
    out = expx / torch.sum(expx, dim=dim, keepdim=True)
    return out

def scaled_dot_product_attention(queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: Optional[torch.Tensor]):
    d_k = keys.shape[-1]
    attn_scores = einsum(queries, keys, "bs ... tq dk, bs ... tk dk -> bs ... tq tk") / math.sqrt(d_k)
    if mask is not None:
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
    attn_probs = softmax(attn_scores, dim=-1)
    out = einsum(attn_probs, values, "bs ... tq tk, bs ... tk dv -> bs ... tq dv")
    return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, d_model, theta, max_seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = self.d_v = d_model // num_heads
        self.q_proj = Linear(d_model, num_heads * self.d_k)
        self.k_proj = Linear(d_model, num_heads * self.d_k)
        self.v_proj = Linear(d_model, num_heads * self.d_v)
        self.o_proj = Linear(num_heads * self.d_v, d_model)
        self.rotary_emb = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len)
    
    def forward(self, x: torch.Tensor, token_positions) -> torch.Tensor:
        bs, seq_len, _ = x.shape
        Q = self.q_proj(x).reshape(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        Q = self.rotary_emb(Q, token_positions)
        K = self.k_proj(x).reshape(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.rotary_emb(K, token_positions)
        V = self.v_proj(x).reshape(bs, -1, self.num_heads, self.d_k).transpose(1, 2)
        mask = torch.tril(torch.ones((seq_len, seq_len), device=x.device)).bool()
        mask = mask[None, None, :, :]
        attn_values = scaled_dot_product_attention(Q, K, V, mask)
        attn_values = attn_values.transpose(1, 2).reshape(bs, seq_len, self.num_heads * self.d_k)
        out = self.o_proj(attn_values)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len, device = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.norm1 = RMSNorm(d_model, device=device)
        self.norm2 = RMSNorm(d_model, device=device)
        self.mhsa = MultiHeadSelfAttention(num_heads, d_model, theta, max_seq_len)
        self.ffn = SwiFFN(d_model, d_ff)
        
    def forward(self, x, token_positions) -> torch.Tensor:
        x_norm1 = self.norm1(x)
        x1 = x + self.mhsa(x_norm1, token_positions)
        x_norm2 = self.norm2(x1)
        x2 = self.ffn(x_norm2)
        out = x1 + x2
        return out

class TransformerLM(nn.Module):
    def __init__(self, vocab_size, context_length, num_layers, d_model, d_ff, num_heads, theta):
        super().__init__()
        self.context_length = context_length
        self.embed = Embedding(vocab_size, d_model)
        self.layers = [TransformerBlock(d_model, num_heads, d_ff, theta, context_length) for _ in range(num_layers)]
        self.layers = nn.ModuleList(self.layers)
        self.norm = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(token_ids)
        token_positions = torch.arange(x.shape[-2], device=x.device)
        for layer in self.layers:
            x = layer(x, token_positions)
        norm_x = self.norm(x)
        logits = self.lm_head(norm_x)
        return logits

