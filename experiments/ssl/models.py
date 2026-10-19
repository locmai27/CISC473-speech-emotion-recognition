"""Frame-wise Dense heads with masked mean or learned attention pooling."""
import torch
from torch import nn


class FrameDenseHead(nn.Module):
    def __init__(self, pooling='mean', input_size=768, hidden_size=128, classes=8):
        super().__init__()
        self.frame_network = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Dropout(0.2),
        )
        self.attention = nn.Linear(hidden_size, 1, bias=False) if pooling == 'attention' else None
        self.classifier = nn.Linear(hidden_size, classes)

    def forward(self, features, mask):
        hidden = self.frame_network(features)
        if self.attention is None:
            weights = mask.to(hidden.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True)
        else:
            scores = self.attention(hidden).squeeze(-1).masked_fill(~mask, float('-inf'))
            weights = scores.softmax(dim=1)
        pooled = (hidden * weights.unsqueeze(-1)).sum(dim=1)
        return self.classifier(pooled)

