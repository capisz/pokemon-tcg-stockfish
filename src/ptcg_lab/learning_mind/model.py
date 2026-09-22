from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .encoding import ACTION_TYPES, CARD_BUCKETS, OPTION_DIM, RAW_DIM, TOKEN_TYPES


class StrategyTransformerV1(nn.Module):
    """Actor-visible policy/value model; there is intentionally no oracle head."""

    def __init__(self, *, width: int = 192, heads: int = 6, ff_width: int = 384,
                 layers: int = 3, enforce_parameters: bool = True):
        super().__init__()
        if (width, heads, ff_width, layers) != (192, 6, 384, 3):
            raise ValueError("StrategyTransformerV1 architecture is frozen")
        self.card_embedding = nn.Embedding(CARD_BUCKETS, 48, padding_idx=0)
        self.state_type_embedding = nn.Embedding(max(TOKEN_TYPES.values()) + 1, 16, padding_idx=0)
        self.state_projection = nn.Linear(RAW_DIM + 48 + 16, width)
        self.global_broadcast = nn.Linear(width, width, bias=False)
        block = nn.TransformerEncoderLayer(d_model=width, nhead=heads,
                                           dim_feedforward=ff_width, dropout=0.0,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, norm=nn.LayerNorm(width),
                                             enable_nested_tensor=False)
        self.option_type_embedding = nn.Embedding(max(ACTION_TYPES.values()) + 1, 32, padding_idx=0)
        self.option_projection = nn.Linear(OPTION_DIM + 32, width)
        self.source_projection = nn.Linear(width, width, bias=False)
        self.target_projection = nn.Linear(width, width, bias=False)
        self.selected_projection = nn.Linear(1, width, bias=False)
        self.policy_query = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                          nn.Linear(width, width, bias=False))
        self.option_key = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width, bias=False))
        self.option_bias = nn.Linear(width, 1)
        self.value_head = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.GELU(),
                                        nn.Linear(width, 1))
        self._reset_padding()
        count = self.parameter_count()
        if enforce_parameters and not 1_400_000 <= count <= 1_900_000:
            raise ValueError(f"StrategyTransformerV1 has {count:,} parameters; expected 1.4M-1.9M")

    def _reset_padding(self) -> None:
        with torch.no_grad():
            self.card_embedding.weight[0].zero_()
            self.state_type_embedding.weight[0].zero_()
            self.option_type_embedding.weight[0].zero_()

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _encode(self, state_card_ids, state_features, state_type_ids, state_mask):
        raw = torch.cat((state_features, self.card_embedding(state_card_ids),
                         self.state_type_embedding(state_type_ids)), dim=-1)
        tokens = self.state_projection(raw)
        global_context = tokens[:, :1]
        tokens = tokens + self.global_broadcast(global_context)
        return self.encoder(tokens, src_key_padding_mask=~state_mask), state_mask

    @staticmethod
    def _gather(tokens, indices):
        safe = indices.clamp(min=0, max=tokens.shape[1] - 1)
        return torch.gather(tokens, 1, safe.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]))

    def policy_forward(self, *, state_card_ids, state_features, state_type_ids, state_mask,
                       option_features, option_type_ids, source_indices, target_indices,
                       option_mask, selected_mask=None):
        encoded, _ = self._encode(state_card_ids, state_features, state_type_ids, state_mask)
        options = self.option_projection(torch.cat((option_features, self.option_type_embedding(option_type_ids)), dim=-1))
        options = options + self.source_projection(self._gather(encoded, source_indices))
        options = options + self.target_projection(self._gather(encoded, target_indices))
        if selected_mask is not None:
            options = options + self.selected_projection(selected_mask.unsqueeze(-1).to(options.dtype))
        query = self.policy_query(encoded[:, 0])
        logits = (self.option_key(options) * query.unsqueeze(1)).sum(-1) / math.sqrt(query.shape[-1])
        logits = logits + self.option_bias(options).squeeze(-1)
        return logits.masked_fill(~option_mask, torch.finfo(logits.dtype).min)

    def evaluation_forward(self, *, state_card_ids, state_features, state_type_ids, state_mask):
        encoded, mask = self._encode(state_card_ids, state_features, state_type_ids, state_mask)
        # The head is blind because the encoder received only actor-visible tensors.
        pooled = (encoded * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.value_head(pooled).squeeze(-1)

    def forward(self, *args, **kwargs):  # pragma: no cover - misuse guard
        raise RuntimeError("Use policy_forward or evaluation_forward explicitly")


@dataclass(frozen=True)
class SelectionResult:
    indices: tuple[int, ...]
    stopped: bool


def autoregressive_select(logit_steps: Callable[[tuple[int, ...]], torch.Tensor], *,
                          action_count: int, minimum: int, maximum: int,
                          legality: Callable[[tuple[int, ...], int], bool] | None = None) -> SelectionResult:
    """Deterministic greedy decoder. STOP is index ``action_count``."""
    if not 0 <= minimum <= maximum <= action_count:
        raise ValueError("invalid selection bounds")
    chosen: tuple[int, ...] = ()
    for _ in range(maximum + 1):
        logits = logit_steps(chosen).detach().clone().flatten()
        if logits.numel() != action_count + 1:
            raise ValueError("decoder logits must include exactly one STOP")
        for index in range(action_count):
            if index in chosen or (legality and not legality(chosen, index)):
                logits[index] = -torch.inf
        if len(chosen) < minimum:
            logits[action_count] = -torch.inf
        if len(chosen) >= maximum:
            logits[:action_count] = -torch.inf
        index = int(torch.argmax(logits).item())
        if index == action_count:
            return SelectionResult(chosen, True)
        if not torch.isfinite(logits[index]):
            raise RuntimeError("no legal autoregressive choice")
        chosen = (*chosen, index)
    raise RuntimeError("autoregressive decoder did not terminate")
