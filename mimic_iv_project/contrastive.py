from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SymmetricInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        logits = torch.matmul(anchor, positive.t()) / self.temperature
        labels = torch.arange(anchor.size(0), device=anchor.device)
        loss = 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

        with torch.no_grad():
            retrieval = (logits.argmax(dim=1) == labels).float().mean().item()
            positive_cosine = (anchor * positive).sum(dim=-1).mean().item()

        return loss, {
            "retrieval_at_1": float(retrieval),
            "positive_cosine": float(positive_cosine),
        }
