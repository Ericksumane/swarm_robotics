"""Masked, permutation-invariant encoding of received neighbor features."""

import torch
from torch import nn


class NeighborEncoder(nn.Module):
    """Encode a variable-size neighbor set into a fixed-size vector.

    Inputs:
        features: [batch, neighbors, feature_dim]
        mask: [batch, neighbors], containing zeros or ones

    Output:
        [batch, embedding_dim + 2]

    The final two entries are the valid-neighbor fraction and an
    indicator that at least one valid neighbor is available.
    """

    def __init__(self, feature_dim=7, embedding_dim=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim

        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, embedding_dim),
            nn.Tanh(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.Tanh(),
        )

    def forward(self, features, mask):
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError("Invalid neighbor feature shape.")
        if mask.shape != features.shape[:2]:
            raise ValueError("Mask shape does not match features.")
        if not torch.isfinite(features).all():
            raise ValueError("Features must be finite.")
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError("Mask entries must be zero or one.")

        weights = mask.to(dtype=features.dtype).unsqueeze(-1)

        # Invalid rows cannot affect the representation or gradients.
        clean_features = torch.where(
            weights.bool(), features, torch.zeros_like(features)
        )
        encoded = self.encoder(clean_features) * weights

        valid_count = weights.sum(dim=1)
        pooled = encoded.sum(dim=1) / valid_count.clamp_min(1.0)

        available = (valid_count > 0).to(features.dtype)
        total_neighbors = max(features.shape[1], 1)
        fraction = valid_count / total_neighbors

        return torch.cat([pooled, fraction, available], dim=-1)