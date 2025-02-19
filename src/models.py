import numpy as np
from scipy.sparse.linalg import svds
from scipy.sparse import coo_array
from src.utils import arsinh
from src.frechetmean import frechet_mean, Poincare, Lorentz
from geoopt import PoincareBall

import torch
import geoopt
from geoopt.manifolds.lorentz.math import lorentz_to_poincare
import pandas as pd
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


class PureSVD:
    def __init__(self, rank=10):
        self.rank = rank
        self.item_factors = None

    def fit(self, vt):
        self.item_factors = torch.tensor(vt.T, dtype=torch.float64)
        return self

    def __call__(self, batch):
        factors = self.item_factors
        return (batch @ factors) @ factors.T

    def eval(self):
        pass

class Easer:
    def __init__(self):
        self.item_factors = None

    def fit(self, gram_matrix, lam):
        num_items = gram_matrix.shape[0]
        diag_indices = np.diag_indices(num_items)
        gram_matrix[diag_indices] += lam
        P = np.linalg.inv(gram_matrix)
        self.item_factors = P / (-np.diag(P))
        self.item_factors[diag_indices] = 0
        self.item_factors = torch.tensor(self.item_factors, dtype=torch.float64)
        return self

    def __call__(self, batch):
        return (batch @ self.item_factors) @ self.item_factors.T

    def eval(self):
        pass


class PopularityModel:
    def fit(self, dataloader):
        freqs = torch.zeros(next(iter(dataloader)).shape[1])
        for batch in dataloader:
            freqs += torch.sum(batch, dim=0)

        self.ranks = torch.zeros(freqs.shape[0], dtype=torch.float32)
        self.ranks[torch.argsort(freqs)] = torch.arange(
            freqs.shape[0], dtype=torch.float32
        )
        return self

    def __call__(self, batch):
        return torch.tile(self.ranks, (batch.shape[0],)).reshape(batch.shape[0], -1)

    def eval(self):
        pass


class MobiusLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        manifold: geoopt.Manifold,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.manifold = manifold
        self.weight_v = nn.Parameter(
            torch.empty((out_features, in_features), dtype=dtype)
        )
        self.bias = None
        if bias:
            self.bias = geoopt.ManifoldParameter(
                torch.zeros(out_features, dtype=dtype), manifold=self.manifold
            )
        self.reset_parameters()

    def forward(self, x: torch.Tensor):
        output = self.manifold.mobius_matvec(self.weight_v, x)
        if self.bias is not None:
            output = self.manifold.mobius_add(output, self.bias)
        return self.manifold.projx(output)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_v, a=np.sqrt(5))
        # if self.bias is not None:
        #     self.bias.zero_()


class HypLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        manifold: geoopt.Manifold,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super(HypLinear, self).__init__()
        self.manifold = manifold
        self.weight_v = nn.Parameter(
            torch.empty((out_features, in_features), dtype=dtype)
        )
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight_v, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight_v)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor):
        output = self.manifold.mobius_matvec(self.weight_v, x)
        if self.bias is not None:
            bias = self.manifold.expmap0(self.bias)
            output = self.manifold.mobius_add(output, bias)
        return self.manifold.projx(output)


class ExpMap0(nn.Module):
    def __init__(self, manifold: geoopt.Manifold):
        super().__init__()
        self.manifold = manifold

    def forward(self, x: torch.Tensor):
        return self.manifold.expmap0(x)


class LogMap0(nn.Module):
    def __init__(self, manifold: geoopt.Manifold):
        super().__init__()
        self.manifold = manifold

    def forward(self, x: torch.Tensor):
        return self.manifold.logmap0(x)


class UnidirectionalPoincareMLR(nn.Module):
    __constants__ = ["in_dim", "out_dim"]

    def __init__(
        self,
        in_features: int,
        out_features: int,
        manifold: geoopt.Manifold,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super(UnidirectionalPoincareMLR, self).__init__()
        self.manifold = manifold
        self.feat_dim = in_features
        self.num_outcome = out_features
        weight = torch.empty(in_features, out_features, dtype=dtype).normal_(
            mean=0, std=(self.feat_dim) ** -0.5 / self.manifold.c.data.sqrt()
        )
        self.weight_v = nn.Parameter(weight)
        self.weight_g = nn.Parameter(weight.norm(dim=0))
        self.bias = nn.Parameter(
            torch.empty(out_features, dtype=dtype), requires_grad=bias
        )
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor):
        w_norm = self.weight_v.norm(dim=0).clamp_min(1e-15)
        output = unidirectional_poincare_mlr(
            x, self.weight_g, self.weight_v / w_norm, self.bias, self.manifold.c
        )
        assert torch.all(~torch.isnan(output))
        return output

    def extra_repr(self):
        return "feat_dim={}, num_outcome={}, bias={}".format(
            self.feat_dim, self.num_outcome, self.bias.requires_grad
        )


@torch.jit.script
def unidirectional_poincare_mlr(x, z_norm, z_unit, r, c):
    # parameters
    rc = c.sqrt()
    drcr = 2.0 * rc * r

    # input
    rcx = rc * x
    cx2 = rcx.pow(2).sum(dim=-1, keepdim=True)

    return (2 * z_norm / rc * arsinh(
            (2.0 * torch.matmul(rcx, z_unit) * drcr.cosh() - (1.0 + cx2) * drcr.sinh())
            / torch.clamp_min(1.0 - cx2, 1e-15)
        )
    )


class PoincareLinear(nn.Module):
    def __init__(self, in_features, out_features, manifold=None, bias=True, dtype=torch.float64):
        super(PoincareLinear, self).__init__()
        self.manifold = manifold
        self.in_features = in_features
        self.out_features = out_features
        weight = torch.empty(in_features, out_features, dtype=dtype).normal_( 
            mean=0, std=(2 * self.in_features * self.out_features) ** -0.5)
        self.weight_v = nn.Parameter(weight)
        self.weight_g = nn.Parameter(weight.norm(dim=0))
        self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype), requires_grad=bias)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.zeros_(self.bias)
    
    def forward(self, x):
        x = poincare_linear(
            x, 
            self.weight_g,
            self.weight_v / self.weight_v.norm(dim=0).clamp_min(1e-15), 
            self.bias, 
            self.manifold.c
        )
        return x
    

def poincare_linear(x, weight_g, weight_v, bias, c):
    rc = c.sqrt()
    x = unidirectional_poincare_mlr(x, weight_g, weight_v, bias, c)
    x = (rc * x).sinh() / rc

    return _project(x / (1 + (1 + c * x.pow(2).sum(dim=-1, keepdim=True)).sqrt()), -c, dim=-1)


def _project(x, k, dim: int = -1, eps: float = -1.0):
    if eps < 0:
        if x.dtype == torch.float32:
            eps = 4e-3
        else:
            eps = 1e-5
    maxnorm = (1 - eps) / (sabs(k) ** 0.5)
    maxnorm = torch.where(k.lt(0), maxnorm, k.new_full((), 1e15))
    norm = x.norm(dim=dim, keepdim=True, p=2).clamp_min(1e-15)
    cond = norm > maxnorm
    projected = x / norm * maxnorm
    return torch.where(cond, projected, x)

def sabs(x, eps: float = 1e-15):
    return x.abs().clamp_min(eps)


class PoincareEmbedding(nn.Module):
    def __init__(self, num_items: int, emb_dim: int, manifold: geoopt.Manifold, dtype: torch.dtype):
        super().__init__()
        init_tensor = torch.rand(num_items, emb_dim, dtype=dtype) * 0.002 - 0.001
        if isinstance(manifold, geoopt.Lorentz):
            init_tensor = torch.cat((torch.sqrt(torch.sum(init_tensor**2, dim=1, keepdim=True) + manifold.k), init_tensor), dim=1)
        self.matrix = geoopt.ManifoldParameter(init_tensor, manifold=manifold, requires_grad=True)
        self.manifold = manifold
    
    def forward(self, x, mode="inference"):
        if mode == "inference":
            # output = frechet_mean(self.matrix[None], Poincare() if isinstance(self.manifold, geoopt.PoincareBall) else Lorentz(), w=x)
            # if isinstance(self.manifold, geoopt.Lorentz):
            #     output = lorentz_to_poincare(output, self.manifold.k, dim=-1)
            x, manifold = x
            matrix = self.matrix
            if isinstance(self.manifold, geoopt.Lorentz):
                matrix = lorentz_to_poincare(matrix, self.manifold.k, dim=-1)
            mean = frechet_mean(matrix[None], Poincare(K=-torch.abs(self.manifold.k)), w=x)
            mapped_mean = PoincareBall(torch.abs(self.manifold.k)).logmap0(mean)
            output = manifold.expmap0(mapped_mean)
            assert torch.all(~torch.isnan(output))
            return output
        items, origin_item = x
        return -self.manifold.dist(self.matrix[items], self.matrix[origin_item].unsqueeze(dim=1))
    

class AutoEncoder(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_manifolds: int,
        emb_dim: int,
        bias: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        # self.manifolds = nn.ModuleList([PoincareBall(i + 0.5, learnable=True) for i in range(num_manifolds)])
        self.manifolds = nn.ModuleList([PoincareBall(1e-3, learnable=True), PoincareBall(1e-6, learnable=True)])
        self.encoders = nn.ModuleList([HypLinear(num_items, emb_dim, manifold, bias, dtype) for manifold in self.manifolds])
        self.decoders = nn.ModuleList([PoincareLinear(emb_dim, num_items, manifold, bias, dtype) for manifold in self.manifolds])
        self.exp_maps = nn.ModuleList([ExpMap0(manifold) for manifold in self.manifolds])
        # self.routers = nn.ModuleList([UnidirectionalPoincareMLR(emb_dim, num_items, manifold, bias, dtype) for manifold in self.manifolds])

        self.emb_dim = emb_dim

    def forward(self, x: torch.Tensor):
        logits = torch.stack([decoder(encoder(exp_map(x))) for exp_map, encoder, decoder in zip(self.exp_maps, self.encoders, self.decoders)], dim=-1)
        # weights = F.softmax(torch.stack([router(encoder(exp_map(x))) for exp_map, encoder, router in zip(self.exp_maps, self.encoders, self.routers)], dim=-1), dim=-1)
        # return torch.sum(logits * weights, dim=-1)
        return torch.sum(logits, dim=-1)