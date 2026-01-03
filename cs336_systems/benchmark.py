import sys
import timeit
import argparse
from contextlib import nullcontext
import torch
import torch.cuda.nvtx as nvtx
import numpy as np

sys.path.append("cs336_assn2_systems")

from cs336_basics.cs336_basics.model import BasicsTransformerLM
from cs336_basics.cs336_basics.optimizer import AdamW
from cs336_basics.cs336_basics.data import get_batch
from cs336_basics.cs336_basics.nn_utils import cross_entropy
from cs336_systems.profiling import annotated_scaled_dot_product_attention


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--context_length", type=int, required=True)
    parser.add_argument("--num_layers", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--num_heads", type=int, required=True)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--data_size", type=int, default=2048)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--backward", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    args = parser.parse_args()

    device = args.device
    amp_ctx = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if args.mixed_precision and "cuda" in device
        else nullcontext()
    )

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        d_ff=args.d_ff,
        num_heads=args.num_heads,
        rope_theta=args.rope_theta,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=3e-4)

    import cs336_basics.cs336_basics.model as model_module
    model_module.scaled_dot_product_attention = annotated_scaled_dot_product_attention

    data_size = max(args.data_size, args.context_length + 1)
    data = np.random.randint(0, args.vocab_size, size=data_size)

    def run_step():
        x, y = get_batch(
            data,
            batch_size=args.batch_size,
            context_length=args.context_length,
            device=device,
        )

        with amp_ctx:
            with nvtx.range("forward"):
                logits = model(x)

        if args.backward:
            with nvtx.range("loss"):
                loss = cross_entropy(logits, y)

            with nvtx.range("backward"):
                optimizer.zero_grad()
                loss.backward()

            with nvtx.range("optimizer"):
                optimizer.step()

        torch.cuda.synchronize()

    with nvtx.range("warmup"):
        for _ in range(args.warmup_steps):
            run_step()

    with nvtx.range("benchmark"):
        start = timeit.default_timer()
        for _ in range(args.num_steps):
            run_step()
        end = timeit.default_timer()

    total_time = end - start
    print(f"Time for {args.num_steps} steps: {total_time:.4f} seconds")
    print(f"Avg time per step: {total_time / args.num_steps:.4f}s")


if __name__ == "__main__":
    main()
