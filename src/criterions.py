import torch
from torch import nn

class CrossEntropyWrapper(nn.CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits, batch, _):
        return super().forward(batch[0], logits)
    

class TruePairwiseLTRLoss(nn.BCEWithLogitsLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, logits, batch, _):
        target = torch.ones_like(logits)
        return super().forward(logits, target)
