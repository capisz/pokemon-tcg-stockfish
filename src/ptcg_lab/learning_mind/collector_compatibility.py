"""Audited compatibility rules for immutable macro-label collector sources.

The source-file hash remains recorded per run. This registry only permits
combination when different file hashes are known to share the exact
collection-code surface; it never rewrites a run's rollout identity.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from .schema import identity_hash


COLLECTOR_EQUIVALENCE_REGISTRY = {
    "schemaVersion": 1,
    "kind": "macro-label-collector-equivalence-registry-v1",
    "equivalences": [
        {
            "labelCollectorVersion": "macro-rollout-labeler-v6",
            "collectionSurfaceSha256": "a3f0225ffb11a69b87b6ee786dcaaab4e5ebe484b333424fc88bdee61e0811f9",
            "sourceModules": [
                {
                    "sha256": "1b7e8df4a69cba6dcbb74d23ba310a9ed18c8ca5859ae5b9c6efbf2f8e4e646a",
                    "gitCommit": "cc3802e38c9e8b41e49196309bf03976eac7251b",
                },
                {
                    "sha256": "822500b4113aea6c69abeb7fe069ce2647d561d742977f5453097c0bae6882a4",
                    "gitCommit": "45620268a1f085184b58d834c0a547164c507649",
                },
                {
                    "sha256": "44e205ed8c8f494950019c1fb7d00fbf79f699317cd784575c98162e5f1c121b",
                    "worktreeBasedOn": "7846ef70aee19c1b0317c1f5e6c11f04b5a5d848",
                    "worktreeStatus": "uncommitted research worktree source; exact file hash pinned",
                },
            ],
            "scope": "transition_generator_identity through collect_macro_labels source; exact bytes",
            "excludedDifference": "The only other experiment.py changes are ranker-input/model-fit code and its tempfile import.",
        }
    ],
}


def collection_surface_sha256(path: Path) -> str:
    """Hash the exact audited collector function block in a source module."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    start = next((index for index, node in enumerate(functions)
                  if node.name == "transition_generator_identity"), None)
    end = next((index for index, node in enumerate(functions)
                if node.name == "collect_macro_labels"), None)
    if start is None or end is None or end < start:
        raise ValueError("collector source is missing the audited collection surface")
    last_line = functions[end].end_lineno
    next_lines = [node.lineno for node in functions[end + 1:]]
    if next_lines:
        last_line = min(next_lines) - 1
    lines = source.splitlines(keepends=True)
    block = "".join(lines[functions[start].lineno - 1:last_line])
    return hashlib.sha256(block.encode("utf-8")).hexdigest()


def registered_collection_surface(*, version: str, source_hash: str) -> str | None:
    """Return the surface only when this exact source hash is allowlisted."""
    for entry in COLLECTOR_EQUIVALENCE_REGISTRY["equivalences"]:
        if entry["labelCollectorVersion"] == version and any(
                item["sha256"] == source_hash for item in entry["sourceModules"]):
            return entry["collectionSurfaceSha256"]
    return None


def collector_compatibility(*, versions: list[str], source_hashes: list[str]) -> dict:
    """Return a provenance-preserving compatibility certificate or reject.

    Identical module hashes use exact-source compatibility. A differing hash
    is accepted only when every source is named by one audited registry entry
    with the same collection-surface hash and collector version.
    """
    if (not versions or any(not isinstance(value, str) or not value for value in versions)
            or len(set(versions)) != 1 or not source_hashes
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for value in source_hashes)):
        raise ValueError("macro-label collector identity is malformed or version-mixed")

    unique_hashes = sorted(set(source_hashes))
    version = versions[0]
    if len(unique_hashes) == 1:
        return {
            "schemaVersion": 1,
            "kind": "exact-source",
            "labelCollectorVersion": version,
            "sourceModuleSha256s": unique_hashes,
            "collectionSurfaceSha256": None,
            "registrySha256": identity_hash(COLLECTOR_EQUIVALENCE_REGISTRY),
        }

    matching = [entry for entry in COLLECTOR_EQUIVALENCE_REGISTRY["equivalences"]
                if entry["labelCollectorVersion"] == version]
    for source_hash in unique_hashes:
        if not any(source_hash == source["sha256"]
                   for entry in matching for source in entry["sourceModules"]):
            raise ValueError("macro-label runs use unaudited collector source hashes")

    surfaces = {
        entry["collectionSurfaceSha256"]
        for entry in matching
        if all(source_hash in {source["sha256"] for source in entry["sourceModules"]}
               for source_hash in unique_hashes)
    }
    if len(surfaces) != 1:
        raise ValueError("macro-label collector sources do not share one audited collection surface")
    surface_hash = next(iter(surfaces))
    commits = sorted({source["gitCommit"] for entry in matching
                      if entry["collectionSurfaceSha256"] == surface_hash
                      for source in entry["sourceModules"]
                      if source["sha256"] in unique_hashes and "gitCommit" in source})
    worktree_sources = [
        {"sha256": source["sha256"], "basedOn": source["worktreeBasedOn"],
         "status": source["worktreeStatus"]}
        for entry in matching if entry["collectionSurfaceSha256"] == surface_hash
        for source in entry["sourceModules"]
        if source["sha256"] in unique_hashes and "worktreeBasedOn" in source
    ]
    return {
        "schemaVersion": 1,
        "kind": "audited-same-collection-surface",
        "labelCollectorVersion": version,
        "sourceModuleSha256s": unique_hashes,
        "collectionSurfaceSha256": surface_hash,
        "sourceCommits": commits,
        "sourceWorktrees": worktree_sources,
        "registrySha256": identity_hash(COLLECTOR_EQUIVALENCE_REGISTRY),
    }
