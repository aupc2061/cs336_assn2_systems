import torch
import time
import os
from typing import List, Iterable, Dict, Any, Tuple, Type
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch._utils as utils
import torch.nn as nn
import torch._utils as utils
import torch.optim as optim
from cs336_systems.mixed_precision_dtypes import ToyModel

batch_size = 8
x = torch.randn((batch_size, 4))
y = torch.randn((batch_size, 1))

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class ShardedOptimizer(optim.Optimizer):
    def __init__(self, params, optimizer_cls: Type[optim.Optimizer], **kwargs: Any):
        defaults = kwargs  
        params_list = list(params)
        super().__init__(params_list, defaults)
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_params = [p for i, p in enumerate(params_list) if i % self.world_size == self.rank]
        self.optimizer = optimizer_cls(self.local_params, **kwargs)

    def step(self, **kwargs):
        self.optimizer.step(**kwargs)
        for i, p in enumerate(self.param_groups[0]['params']):
            src = i % self.world_size
            dist.broadcast(p.data, src=src)

class DDP(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.module = model
        self.handles = []
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        def _grad_hook(param):
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)

        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(_grad_hook)

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        world_size = dist.get_world_size()
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad /= world_size
                
        self.handles.clear()

def get_bucketed_params(params: List[torch.nn.Parameter], bucket_size_mb: float):
    buckets = []
    bucket = []
    bs_bytes = int(bucket_size_mb * 1024 * 1024)
    cs = 0
    for p in params:
        if not p.requires_grad:
            continue
        p_size = p.numel() * p.element_size()
        if cs + p_size > bs_bytes:
            buckets.append(bucket)
            bucket = []
            cs = 0

        cs += p_size
        bucket.append(p)
    if bucket:
        buckets.append(bucket)
    return buckets

class DDPBucketed(nn.Module):
    def __init__(self, model: nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = model
        self.handles = []
        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        params = list(reversed(list(self.module.parameters())))
        self.bucketed_params = get_bucketed_params(params, bucket_size_mb)
        self.param_to_bucket_idx = {p:i for i, b in enumerate(self.bucketed_params) for p in b}
        self.cnt = [0] * len(self.bucketed_params)
        self.bgrads = []
        self.flattened_bgrads = []

        def _grad_hook(param: torch.nn.Parameter) -> None:
            bucket_idx = self.param_to_bucket_idx[param]
            cur_bucket = self.bucketed_params[bucket_idx]
            self.cnt[bucket_idx] += 1
            if self.cnt[bucket_idx] == len(cur_bucket):
                grads = [p.grad for p in cur_bucket]
                self.bgrads.append(grads)
                flattened_grads = utils._flatten_dense_tensors(grads)
                self.flattened_bgrads.append(flattened_grads)
                h = dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM, async_op=True)
                self.handles.append(h)

        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(_grad_hook)

    def forward(self, *inputs, **kwargs):
        return self.module.forward(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()

        world_size = dist.get_world_size()
        self.flattened_bgrads = [fg / world_size for fg in self.flattened_bgrads]
        self.unflattened_bgrads = [utils._unflatten_dense_tensors(fg, g) for fg, g in zip(self.flattened_bgrads, self.bgrads)]
        grads = [g for bucket in self.unflattened_bgrads for g in bucket]
        j = len(grads) - 1
        for p in self.module.parameters():
            if p.grad is not None:
                p.grad.copy_(grads[j])
                j -= 1
        self.handles.clear()
        self.cnt = [0] * len(self.bucketed_params)
        self.bgrads = []
        self.flattened_bgrads = []

def train(rank:int, world_size: int, device: str = "cpu"):
    setup(rank, world_size)
    torch.manual_seed(rank)
    model = ToyModel(in_features=4, out_features=1).to(device=device)
    ddp_model = DDP(model)
    optimizer = optim.SGD(params=model.parameters(), lr=0.01)

    xr = x[rank::world_size]
    yr = y[rank::world_size]

    optimizer.zero_grad()
    out = model(xr)
    loss = F.mse_loss(out, yr)
    loss.backward()
    ddp_model.finish_gradient_synchronization()
    optimizer.step()

    cleanup()

def normal_train():
    torch.manual_seed(0)
    model = ToyModel(in_features=4, out_features=1)
    optimizer = optim.SGD(params=model.parameters(), lr=0.01)
    optimizer.zero_grad()
    out = model(x)
    loss = F.mse_loss(out, y)
    loss.backward()
    optimizer.step()
    return [p.detach().clone() for p in model.parameters()]

def ddp_train(rank:int, world_size: int, return_dict):
    setup(rank, world_size)
    torch.manual_seed(rank)
    model = ToyModel(in_features=4, out_features=1)
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    optimizer = optim.SGD(params=model.parameters(), lr=0.01)

    xr = x[rank::world_size]
    yr = y[rank::world_size]

    optimizer.zero_grad()
    out = model(xr)
    loss = F.mse_loss(out, yr)
    loss.backward()

    grads = [p.grad for p in model.parameters()]
    flattened_grads = utils._flatten_dense_tensors(grads)
    dist.all_reduce(flattened_grads, op=dist.ReduceOp.SUM)
    flattened_grads /= world_size
    unflattend_grads = utils._unflatten_dense_tensors(flattened_grads, grads)

    for param, grad in zip(model.parameters(), unflattend_grads):
        if param.grad is not None:
            param.grad.copy_(grad)

    # # Naive all-reduce for each parameter
    # for param, grad in model.parameters():
    #     if param.grad is not None:
    #         dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
    #         param.grad /= world_size

    optimizer.step()
    if rank == 0:
        return_dict["ddp_params"] = [
            p.detach().cpu().clone() for p in model.parameters()
        ]
    cleanup()

if __name__ == "__main__":
    world_size = 2
    manager = mp.Manager()
    return_dict = manager.dict()
    ref_params = normal_train()
    mp.spawn(ddp_train, args = (world_size, return_dict), nprocs=world_size, join=True)
    ddp_params = return_dict["ddp_params"]

    for i, (p_ref, p_ddp) in enumerate(zip(ref_params, ddp_params)):
        torch.testing.assert_close(p_ref, p_ddp, rtol=1e-5, atol=1e-6)
        print(f"Parameter {i}: OK")



