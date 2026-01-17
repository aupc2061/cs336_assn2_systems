import torch

def get_batch(x, batch_size, context_length, device):
    preds = torch.tensor(x, dtype=torch.long, device=device)
    starts = torch.randint(low = 0, high= len(x) - context_length, size=(batch_size, ), device=device)
    x = torch.stack([preds[s: s + context_length] for s in starts])
    y = torch.stack([preds[s + 1: s + context_length + 1] for s in starts])
    return x, y