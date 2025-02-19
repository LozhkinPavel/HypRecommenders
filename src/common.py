from torch import nn
from src.models import HypLinear, MobiusLinear, UnidirectionalPoincareMLR, PoincareLinear

str2layer = {
    "Euc": nn.Linear,
    "HypLinear": HypLinear,
    "Mobius": MobiusLinear,
    "HypMLR": UnidirectionalPoincareMLR,
    "PoincareLinear": PoincareLinear
}