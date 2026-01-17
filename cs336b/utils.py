from collections.abc import Callable, Iterable
from typing import Dict, Optional
import torch
import math
import os
import typing

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps = 1e-6):
    global_norm_sq = 0.0
    for p in parameters:
        if p.grad is not None:
            global_norm_sq += p.grad.data.norm(2).item() ** 2
    global_norm = math.sqrt(global_norm_sq)
    clip_factor = max_l2_norm / (global_norm + eps)
    if clip_factor < 1.0:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_factor)

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    d = {}
    d["model"] = model.state_dict()
    d["optimizer"] = optimizer.state_dict()
    d["iteration"] = iteration
    torch.save(d, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    d = torch.load(src)
    model.load_state_dict(d["model"])
    optimizer.load_state_dict(d["optimizer"])
    return d["iteration"]