import torch
import numpy as np
from torch import nn
import os


def hr(topk, k, pos_items):
    return torch.sum(topk[:, :k] == pos_items[:, None], dim=1, dtype=torch.float32)


def ndcg(topk, k, pos_items):
    coef = torch.log2(torch.arange(k) + 2)[None, :] * torch.log2(torch.tensor([2]))
    dcg = (topk[:, :k] == pos_items[:, None]) / coef.to(topk.device)
    return torch.sum(dcg, dim=1)


def grad_norm(model: nn.Module):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.detach().data.norm(2)
        total_norm += param_norm.item() ** 2
    return total_norm**0.5


def fix_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_printoptions(precision=10)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    np.random.seed(seed)


@torch.jit.script
def arsinh(x: torch.Tensor):
    return (x + torch.sqrt(1 + x.pow(2))).clamp_min(1e-15).log().to(x.dtype)
