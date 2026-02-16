from torch import nn
from src.models import HypLinear, MobiusLinear, UnidirectionalPoincareMLR, PoincareLinear

str2layer = {
    "Euc": nn.Linear,
    "HypLinear": HypLinear,
    "Mobius": MobiusLinear,
    "HypMLR": UnidirectionalPoincareMLR,
    "PoincareLinear": PoincareLinear
}

def tuple_to_device(tensors, device):
    if len(tensors) == 0:
        return tensors
    return tuple(t.to(device) for t in tensors)