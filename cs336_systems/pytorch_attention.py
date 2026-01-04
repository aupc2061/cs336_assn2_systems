import math
import timeit
import torch
import torch.nn as nn
import torch.cuda.nvtx as nvtx

from cs336_basics.cs336_basics.model import Linear, scaled_dot_product_attention


class SelfAttention(nn.Module):
    """Single-head self-attention without an explicit head dimension.

    Expects Q, K, V of shape (batch, seq_len, d_model) and applies a
    causal mask before calling scaled_dot_product_attention.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_k = self.d_v = d_model
        self.o_proj = Linear(self.d_v, d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        _, seq_len, _ = Q.shape
        mask = torch.tril(torch.ones((seq_len, seq_len), device=Q.device)).bool()
        mask = mask[None, :, :]
        attn_values = scaled_dot_product_attention(Q, K, V, mask)
        out = self.o_proj(attn_values)
        return out


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark.")

    device = torch.device("cuda")
    batch_size = 8
    num_warmup = 10
    num_iters = 100

    d_models = [16, 32, 64, 128]
    seq_lens = [256, 1024, 4096, 8192, 16384]

    for d_model in d_models:
        for seq_len in seq_lens:
            print(f"\nBenchmarking d_model={d_model}, seq_len={seq_len}")

            sa = SelfAttention(d_model=d_model).to(device)
            sa.train()

            Q = torch.randn(batch_size, seq_len, d_model, device=device)
            K = torch.randn(batch_size, seq_len, d_model, device=device)
            V = torch.randn(batch_size, seq_len, d_model, device=device)

            # Warmup forwards
            for _ in range(num_warmup):
                with nvtx.range("forward_warmup"):
                    _ = sa(Q, K, V)
                torch.cuda.synchronize()

            # Timed forwards
            start = timeit.default_timer()
            for _ in range(num_iters):
                with nvtx.range("forward"):
                    _ = sa(Q, K, V)
                torch.cuda.synchronize()
            end = timeit.default_timer()
            forward_time = end - start

            # Backward benchmark
            # Warmup backward passes.
            for _ in range(num_warmup):
                Q_bwd = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
                K_bwd = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
                V_bwd = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

                with nvtx.range("backward_warmup_forward"):
                    out = sa(Q_bwd, K_bwd, V_bwd)
                    loss = out.sum()
                torch.cuda.synchronize()

                with nvtx.range("backward_warmup_backward"):
                    loss.backward()
                torch.cuda.synchronize()

                sa.zero_grad(set_to_none=True)

            # Timed backward passes with memory history recording for profiling.
            torch.cuda.memory._record_memory_history(max_entries=1000000)

            start = timeit.default_timer()
            for i in range(num_iters):
                Q_bwd = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
                K_bwd = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)
                V_bwd = torch.randn(batch_size, seq_len, d_model, device=device, requires_grad=True)

                with nvtx.range("backward_forward"):
                    out = sa(Q_bwd, K_bwd, V_bwd)
                    loss = out.sum()
                torch.cuda.synchronize()

                with nvtx.range("backward"):
                    loss.backward()
                torch.cuda.synchronize()

                sa.zero_grad(set_to_none=True)

            end = timeit.default_timer()
            backward_time = end - start

            # Save a pickle snapshot that can be inspected with PyTorch's tool.
            snapshot_name = f"memory_snapshot_d{d_model}_L{seq_len}.pickle"
            torch.cuda.memory._dump_snapshot(snapshot_name)
            torch.cuda.memory._record_memory_history(enabled=None)

            print(
                f"Forward: total={forward_time:.4f}s, avg={forward_time/num_iters:.6f}s; "
                f"Backward: total={backward_time:.4f}s, avg={backward_time/num_iters:.6f}s"
            )

if __name__ == "__main__":
    main()

