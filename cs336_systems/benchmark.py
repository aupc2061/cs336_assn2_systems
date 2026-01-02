import sys
import torch.cuda.nvtx as nvtx
sys.path.append(r"D:\CS336\assignment2-systems")

from cs336_basics.cs336_basics.model import BasicsTransformerLM
from cs336_basics.cs336_basics.optimizer import AdamW, get_cosine_lr
from cs336_basics.cs336_basics.data import get_batch
from cs336_basics.cs336_basics.nn_utils import *
from cs336_systems.profiling import annotated_scaled_dot_product_attention
import timeit
import argparse
import torch
import torch.nn.functional as F
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab_size", type=int, default=512)
    parser.add_argument("--context_length", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=1024)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--data_size", type=int, default=1024)
    parser.add_argument("--num_steps", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--backward", action='store_true', help="Include backward pass in timing")
    args = parser.parse_args()
    device = args.device
    model = BasicsTransformerLM(
            vocab_size=args.vocab_size, 
            context_length=args.context_length, 
            num_layers=args.num_layers,
            d_model=args.d_model,
            d_ff=args.d_ff,
            num_heads=args.num_heads,
            rope_theta=args.rope_theta).to(device=device)
    data = np.random.randint(0, args.vocab_size, size=args.data_size)
    
    # Swap to annotated version for profiling
    import cs336_basics.cs336_basics.model as model_module
    model_module.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    
    def run_step():
        x, y = get_batch(data, batch_size=args.batch_size, context_length=args.context_length, device=args.device)
        logits = model(x)
        if args.backward:
            loss = cross_entropy(logits, y)
            loss.backward()
        if 'cuda' in args.device:
            torch.cuda.synchronize()
    
    # Warm-up
    with nvtx.range("warmup"):
        for _   in range(args.warmup_steps):
            run_step()
    
    # Timing
    with nvtx.range("benchmark"):
        start_time = timeit.default_timer()
        for _ in range(args.num_steps):
            run_step()
        end_time = timeit.default_timer()
    
    total_time = end_time - start_time
    print(f"Time for {args.num_steps} steps: {total_time:.4f} seconds")
    print(f"Average time per step: {total_time / args.num_steps:.4f} seconds")

if __name__ == "__main__":
    main()
