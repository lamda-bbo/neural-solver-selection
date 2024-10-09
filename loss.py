import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class RankingLoss(nn.Module):
    def __init__(self, num_solvers, top_k):
        super().__init__()
        self.num_solvers = num_solvers
        self.top_k = top_k

    def forward(self, logits, costs):
        loss = 0
        for i in range(self.top_k):
            cur_cost, ind = costs.topk(self.num_solvers - i, largest=True)
            cur_label = cur_cost.min(dim=1)[1]
            cur_logits = torch.take_along_dim(logits, ind, 1)
            loss += F.nll_loss(F.log_softmax(cur_logits, 1), cur_label)
        return loss