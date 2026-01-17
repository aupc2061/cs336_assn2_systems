import torch

def cross_entropy_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    logsumexp = torch.logsumexp(logits, dim=-1)
    o_xi = torch.gather(logits, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    log_probs = -o_xi + logsumexp
    loss = log_probs.mean()
    return loss