"""Version-bound inference for portable ranker v2 artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset_v1 import file_sha256
from .ranker_features import MACRO_FEATURE_SCHEMA, MACRO_FEATURE_SCHEMA_HASH


def inference_implementation_sha256() -> str:
    return file_sha256(Path(__file__))


def _tree_score(node: dict, row: np.ndarray) -> float:
    if "leaf" in node:
        return float(node["leaf"])
    split = node.get("split")
    if not isinstance(split, str) or not split.startswith("f") or not split[1:].isdigit():
        raise ValueError("portable ranker tree contains an invalid feature split")
    feature = int(split[1:])
    if not 0 <= feature < row.shape[0]:
        raise ValueError("portable ranker tree split exceeds the feature dimension")
    value = float(row[feature])
    if not np.isfinite(value):
        child_id = node.get("missing")
    else:
        child_id = node.get("yes") if value < float(node["split_condition"]) else node.get("no")
    children = node.get("children")
    if not isinstance(children, list):
        raise ValueError("portable ranker tree split has no children")
    child = next((item for item in children if item.get("nodeid") == child_id), None)
    if child is None:
        raise ValueError("portable ranker tree points to a missing child")
    return _tree_score(child, row)


def predict_macro_ranker_v2(artifact: dict, features) -> np.ndarray:
    """Score candidate vectors using the portable tree dump (ranking margins)."""
    if (not isinstance(artifact, dict) or artifact.get("kind") != "xgboost-macro-ranker-v2-portable"
            or artifact.get("featureSchemaHash") != MACRO_FEATURE_SCHEMA_HASH
            or artifact.get("featureCount") != MACRO_FEATURE_SCHEMA["dimension"]
            or artifact.get("inferenceImplementationSha256") != inference_implementation_sha256()
            or artifact.get("featureImplementationSha256") != file_sha256(
                Path(__file__).with_name("ranker_features.py"))):
        raise ValueError("portable macro ranker inference identity mismatch")
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != MACRO_FEATURE_SCHEMA["dimension"]:
        raise ValueError("portable macro ranker expects a [batch, 640] feature matrix")
    if not isinstance(artifact.get("trees"), list) or not artifact["trees"]:
        raise ValueError("portable macro ranker has no trees")
    return np.asarray([sum(_tree_score(tree, row) for tree in artifact["trees"])
                       for row in matrix], dtype=np.float32)
