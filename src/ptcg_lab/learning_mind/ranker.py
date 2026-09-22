from __future__ import annotations

import importlib.util
from dataclasses import dataclass

import numpy as np

from .schema import identity_hash


def _xgboost():
    if importlib.util.find_spec("xgboost") is None:
        raise RuntimeError("Macro ranking requires the optional 'mind' extra with XGBoost; no substitute model is permitted")
    import xgboost
    return xgboost


@dataclass
class FrozenIteration:
    iteration: int
    teacher_hash: str
    opponent_policy_hash: str
    input_hash: str
    position_hashes: tuple[str, ...]

    def __post_init__(self):
        if not 1 <= self.iteration <= 6:
            raise ValueError("macro ranker permits at most six refit iterations")
        if len(set(self.position_hashes)) != len(self.position_hashes):
            raise ValueError("training positions must be unique")


class XGBoostMacroRanker:
    def __init__(self, **overrides):
        self.parameters = {"objective": "rank:pairwise", "n_estimators": 200, "max_depth": 6,
                           "learning_rate": .05, "subsample": .8, "colsample_bytree": .8,
                           "random_state": 7543298, "n_jobs": 1, **overrides}
        self.model = None

    def fit(self, features, labels, groups, weights=None):
        xgb = _xgboost()
        self.model = xgb.XGBRanker(**self.parameters)
        self.model.fit(np.asarray(features), np.asarray(labels), group=np.asarray(groups),
                       sample_weight=None if weights is None else np.asarray(weights))
        return self

    def predict(self, features):
        if self.model is None:
            raise RuntimeError("ranker has not been fit")
        return self.model.predict(np.asarray(features))

    def manifest(self, iteration: FrozenIteration) -> dict:
        return {"kind": "xgboost-macro-ranker-v1", "parameters": self.parameters,
                "iteration": iteration.__dict__, "manifestHash": identity_hash({"parameters": self.parameters,
                                                                                  "iteration": iteration.__dict__})}


def holdout_splits(rows: list[dict]) -> list[dict]:
    archetypes = sorted({row["opponentArchetype"] for row in rows})
    families = sorted({row["opponentPolicyFamily"] for row in rows})
    return ([{"kind": "leave-one-opponent-archetype-out", "heldOut": value,
              "train": [i for i, row in enumerate(rows) if row["opponentArchetype"] != value],
              "test": [i for i, row in enumerate(rows) if row["opponentArchetype"] == value]} for value in archetypes]
            + [{"kind": "frozen-policy-family", "heldOut": value,
                "train": [i for i, row in enumerate(rows) if row["opponentPolicyFamily"] != value],
                "test": [i for i, row in enumerate(rows) if row["opponentPolicyFamily"] == value]} for value in families])
