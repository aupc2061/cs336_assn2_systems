import torch
import math
import os
import typing
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm
import logging
import time
import wandb
from cs336_basics.layers import TransformerLM, softmax
from cs336_basics.optimizer import AdamW, lr_cosine_schedule
from  cs336_basics.utils import gradient_clipping, load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.losses import cross_entropy_loss
from cs336_basics.tokenizer import Tokenizer


def evaluate(model, tokens, batch_size, context_length, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    with torch.no_grad():
        for _ in range(10):  # evaluate on 10 batches
            x, y = get_batch(tokens, batch_size, context_length, device)
            pred = model(x)
            loss = cross_entropy_loss(pred, y)
            total_loss += loss.item()
            num_batches += 1
    model.train()
    return total_loss / num_batches


def train(data_path: str, val_data_path: str, vocab_path, merges_path, d_model, d_ff, num_heads, num_layers, context_length, theta, lr = 1e-3, num_steps: int = 10, device: str = "cpu", batch_size:int = 32, experiment_name: str = "baseline"):
    wandb.init(project="cs336-basics", name=experiment_name, config={
        "d_model": d_model,
        "d_ff": d_ff,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "context_length": context_length,
        "theta": theta,
        "lr": lr,
        "num_steps": num_steps,
        "batch_size": batch_size,
    })
    
    with open(data_path, "r", encoding="utf-8") as fp:
        text = fp.read()
    tokenizer = Tokenizer.from_files(vocab_filepath=vocab_path, merges_filepath= merges_path)
    tokens = tokenizer.encode(text)
    
    with open(val_data_path, "r", encoding="utf-8") as fp:
        val_text = fp.read()
    val_tokens = tokenizer.encode(val_text)
    
    vocab_size = tokenizer.vocab_size
    model = TransformerLM(
        vocab_size=vocab_size, 
        context_length=context_length, 
        num_layers=num_layers,
        d_model=d_model,
        d_ff=d_ff,
        num_heads=num_heads,
        theta=theta).to(device=device)
    model = torch.compile(model)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)
    
    start_time = time.time()
    
    for step in range(num_steps):
        curr_lr = lr_cosine_schedule(
            t=step,
            alpha_max=lr,
            alpha_min=0.0,
            Tw=int(0.1 * num_steps),
            Tc=num_steps
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = curr_lr
        x, y = get_batch(tokens, batch_size, context_length, device)
        pred = model(x)
        loss = cross_entropy_loss(pred, y)
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), max_l2_norm=1.0)
        optimizer.step()
        
        elapsed_time = time.time() - start_time
        train_ppl = math.exp(min(loss.item(), 20))
        wandb.log({
            "train/loss": loss.item(),
            "train/ppl": train_ppl,
            "lr": curr_lr,
            "step": step,
            "wallclock_time": elapsed_time
        })  
        if step > 0 and step % 100 == 0:
            val_loss = evaluate(model, val_tokens, batch_size, context_length, device)
            wandb.log({"val/loss": val_loss, "val/ppl": math.exp(min(val_loss, 20)), "step": step, "wallclock_time": elapsed_time})
            save_checkpoint(model, optimizer, step, f"ckpt_{step}.pt")
            print(f"Step {step + 1}/{num_steps}: train_loss = {loss.item():.4f}, val_loss = {val_loss:.4f}, lr = {curr_lr:.2e}")
        else:
            print(f"Step {step + 1}/{num_steps}: loss = {loss.item():.4f}, lr = {curr_lr:.2e}")
    
    save_checkpoint(model, optimizer, num_steps, "ckpt_final.pt")
    print("Final checkpoint saved.")
    wandb.finish()

def generate(model: TransformerLM, tokenizer: Tokenizer, prompt: str, max_length: int, temperature: float = 0.5, top_p: float = 0.9, device: str = "cpu") -> str:
    model.eval()
    tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device)
    eos_tok = tokenizer.tok2id[bytes("<|endoftext|>", "utf-8")]
    context_length = model.context_length  
    while len(tokens) < max_length:
        # Truncate to context length
        if len(tokens) > context_length:
            tokens = tokens[-context_length:]
        with torch.no_grad():
            logits = model(tokens.unsqueeze(0))[0, -1]
            logits = logits / temperature
            probs = softmax(logits, dim=-1)

        # top-p sampling
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cum_probs <= top_p
        mask[0] = True
        filtered_probs = sorted_probs * mask
        filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)

        # sample a token
        idx = torch.multinomial(filtered_probs, 1)
        next_tok = sorted_indices[idx]
        tokens = torch.cat([tokens, next_tok])
        if next_tok.item() == eos_tok:
            break
    return tokenizer.decode(tokens.tolist())


def main():
    train(
        data_path=r"D:\CS336\assignment1-basics\data\TinyStoriesV2-GPT4-train.txt",
        val_data_path=r"D:\CS336\assignment1-basics\data\TinyStoriesV2-GPT4-valid.txt",
        vocab_path="cs336_basics/bpe_vocab.pkl",
        merges_path="cs336_basics/bpe_merges.pkl",
        d_model=256,
        d_ff=1024,
        num_heads=8,
        num_layers=6,
        context_length=128,
        theta=10000.0,
        lr=3e-4,
        num_steps=200,
        batch_size=32,
        device="cuda" if torch.cuda.is_available() else "cpu",
        experiment_name="baseline"
    )

if __name__ == "__main__":
    main()
