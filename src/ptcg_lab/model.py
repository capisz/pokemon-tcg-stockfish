"""Small additive resource/value model plus a legal-action policy head.

Requires the optional training extra. Resource terms are learned, not causal card
values; the interaction term includes the public-card embedding context.
"""
from __future__ import annotations

import torch
from torch import nn

from .features import ACTION_DIM, CARD_BUCKETS, FEATURE_NAMES


class PolicyResourceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.card_embedding = nn.Embedding(CARD_BUCKETS, 16, padding_idx=0)
        self.resource_terms = nn.ModuleList([nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1, bias=False)) for _ in FEATURE_NAMES])
        self.baseline = nn.Parameter(torch.zeros(1))
        self.context = nn.Sequential(nn.Linear(len(FEATURE_NAMES) + 16, 64), nn.Tanh(), nn.Linear(64, 32), nn.Tanh())
        self.interaction = nn.Linear(32, 1, bias=False)
        self.outcomes = nn.Linear(32, 3)
        self.action_encoder = nn.Sequential(nn.Linear(ACTION_DIM, 32), nn.Tanh())

    def forward(self, resources, cards, actions=None):
        embedding = self.card_embedding(cards)
        mask = (cards != 0).unsqueeze(-1)
        pooled = (embedding * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        context = self.context(torch.cat([resources, pooled], dim=-1))
        components = torch.cat([term(resources[:, i:i + 1]) for i, term in enumerate(self.resource_terms)], dim=-1)
        interaction = self.interaction(context).squeeze(-1)
        logit = components.sum(dim=-1) + interaction + self.baseline
        policy = None
        if actions is not None:
            policy = (self.action_encoder(actions) * context.unsqueeze(1)).sum(dim=-1) / (32 ** .5)
        return {"logit": logit, "components": components, "interaction": interaction,
                "baseline": self.baseline, "outcome_logits": self.outcomes(context), "policy": policy}
