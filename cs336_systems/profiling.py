import torch.cuda.nvtx as  nvtx
from typing import Optional
import math
from cs336_basics.cs336_basics.model import softmax
import torch
from einops import einsum

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(Q, K, V, mask: Optional[torch.Tensor]):
    d_k = K.shape[-1]
    with nvtx.range("computing attention scores"):
        attn_scores = einsum(Q, K, "bs ... tq dk, bs ... tk dk -> bs ... tq tk") / math.sqrt(d_k)
    if mask is not None:
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))
    with nvtx.range("applying softmax"):
        attn_probs = softmax(attn_scores, dim=-1)
    with nvtx.range("final matmul"):
        out = einsum(attn_probs, V, "bs ... tq tk, bs ... tk dv -> bs ... tq dv")
    return out