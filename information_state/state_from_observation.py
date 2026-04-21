"""State-from-Observation model components."""

from __future__ import annotations

import math

import torch
from torch import nn


class ObservationTripletEncoder(nn.Module):
    """Encode each [value, mask, delta] observation triplet into a local state."""

    def __init__(self, num_variables: int, num_time_bins: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_variables = num_variables
        self.num_time_bins = num_time_bins
        self.triplet_mlp = nn.Sequential(
            nn.Linear(3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        self.variable_embedding = nn.Embedding(num_variables, d_model)
        self.time_embedding = nn.Embedding(num_time_bins, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_time_bins, num_variables, _ = x.shape
        if num_time_bins != self.num_time_bins or num_variables != self.num_variables:
            raise ValueError(
                f"Expected input shape [B, {self.num_time_bins}, {self.num_variables}, 3], "
                f"received [B, {num_time_bins}, {num_variables}, 3]."
            )

        local = self.triplet_mlp(x)
        variable_ids = torch.arange(num_variables, device=x.device)
        time_ids = torch.arange(num_time_bins, device=x.device)
        local = local + self.variable_embedding(variable_ids).view(1, 1, num_variables, -1)
        local = local + self.time_embedding(time_ids).view(1, num_time_bins, 1, -1)
        return self.dropout(local)


class ClinicalStateFormationOperator(nn.Module):
    """Jointly mix observation content, variable relations, time, and observation state."""

    def __init__(
        self,
        num_variables: int,
        num_time_bins: int,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        observation_dim: int = 16,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.num_variables = num_variables
        self.num_time_bins = num_time_bins
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.observation_dim = observation_dim
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.observation_scale = 1.0 / math.sqrt(observation_dim)

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.obs_q_proj = nn.Linear(2, num_heads * observation_dim)
        self.obs_k_proj = nn.Linear(2, num_heads * observation_dim)
        self.variable_bias = nn.Parameter(torch.zeros(num_heads, num_variables, num_variables))
        self.relative_time_bias = nn.Parameter(torch.zeros(num_heads, 2 * num_time_bins - 1))
        self.attn_dropout = nn.Dropout(dropout)

        flat_var_ids = torch.arange(num_variables).repeat(num_time_bins)
        flat_time_ids = torch.arange(num_time_bins).repeat_interleave(num_variables)
        rel_index = flat_time_ids[:, None] - flat_time_ids[None, :] + (num_time_bins - 1)
        self.register_buffer("flat_var_ids", flat_var_ids.long(), persistent=False)
        self.register_buffer("relative_time_index", rel_index.long(), persistent=False)

    def forward(self, h: torch.Tensor, observation_state: torch.Tensor) -> torch.Tensor:
        batch_size, num_time_bins, num_variables, d_model = h.shape
        if num_time_bins != self.num_time_bins or num_variables != self.num_variables:
            raise ValueError(
                f"Expected hidden shape [B, {self.num_time_bins}, {self.num_variables}, H], "
                f"received [B, {num_time_bins}, {num_variables}, H]."
            )

        num_positions = num_time_bins * num_variables
        flat_hidden = h.view(batch_size, num_positions, d_model)
        flat_obs = observation_state.view(batch_size, num_positions, 2)

        q = self.q_proj(flat_hidden).view(batch_size, num_positions, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(flat_hidden).view(batch_size, num_positions, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(flat_hidden).view(batch_size, num_positions, self.num_heads, self.head_dim).transpose(1, 2)
        obs_q = self.obs_q_proj(flat_obs).view(batch_size, num_positions, self.num_heads, self.observation_dim).transpose(1, 2)
        obs_k = self.obs_k_proj(flat_obs).view(batch_size, num_positions, self.num_heads, self.observation_dim).transpose(1, 2)

        content_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        observation_scores = torch.matmul(obs_q, obs_k.transpose(-1, -2)) * self.observation_scale
        variable_scores = self.variable_bias[:, self.flat_var_ids[:, None], self.flat_var_ids[None, :]]
        time_scores = self.relative_time_bias[:, self.relative_time_index]

        scores = content_scores + observation_scores
        scores = scores + variable_scores.unsqueeze(0) + time_scores.unsqueeze(0)

        attention = torch.softmax(scores, dim=-1)
        attention = self.attn_dropout(attention)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, num_positions, d_model)
        output = self.out_proj(output)
        return output.view(batch_size, num_time_bins, num_variables, d_model)


class StateFormationBlock(nn.Module):
    """Residual state-formation block with joint operator and feed-forward layer."""

    def __init__(
        self,
        num_variables: int,
        num_time_bins: int,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.operator = ClinicalStateFormationOperator(
            num_variables=num_variables,
            num_time_bins=num_time_bins,
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, h: torch.Tensor, observation_state: torch.Tensor) -> torch.Tensor:
        h = h + self.operator(self.norm1(h), observation_state)
        h = h + self.ffn(self.norm2(h))
        return h


class StateFromObservationEncoder(nn.Module):
    """Map an observation tensor to a pooled latent clinical state."""

    def __init__(
        self,
        num_variables: int,
        num_time_bins: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.triplet_encoder = ObservationTripletEncoder(
            num_variables=num_variables,
            num_time_bins=num_time_bins,
            d_model=d_model,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [
                StateFormationBlock(
                    num_variables=num_variables,
                    num_time_bins=num_time_bins,
                    d_model=d_model,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        observation_state = x[..., 1:]
        h = self.triplet_encoder(x)
        for block in self.blocks:
            h = block(h, observation_state)
        h = self.output_norm(h)
        return h.view(h.size(0), -1, h.size(-1)).mean(dim=1)


class StateFromObservationModel(nn.Module):
    """Encoder plus projection head for contrastive training and downstream encoding."""

    def __init__(
        self,
        num_variables: int,
        num_time_bins: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        projection_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.encoder = StateFromObservationEncoder(
            num_variables=num_variables,
            num_time_bins=num_time_bins,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(projection_dim, projection_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection_head(self.encode(x))

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        anchor_projection = self.project(anchor)
        if positive is None:
            return anchor_projection, None
        positive_projection = self.project(positive)
        return anchor_projection, positive_projection
