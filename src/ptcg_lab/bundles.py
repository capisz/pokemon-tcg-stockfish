"""Checksummed directory bundles; atomically published, never merge live SQLite."""
from __future__ import annotations

import json
import os
import shutil
import uuid
from pathlib import Path, PurePosixPath

from .resources import ResourceLimit, check_storage, volume_free
from .storage import Store, digest, file_digest
from .features import FEATURE_VERSION

BUNDLE_VERSION = 1
CATEGORIES = {"replays", "replay-index", "models", "experiments", "selfplay", "cycles", "partitions", "population", "evaluation-protocols", "evaluation-models", "teaching", "teaching-families", "fixture-receipts", "teaching-reviews", "teaching-audits", "datasets", "dataset-rows", "teaching-snapshots", "experimental-partitions", "search-targets", "position-bank", "learning-runs", "learning-games", "comparisons", "incumbents", "comparison-protocols", "comparison-seeds", "learning-recoveries", "private-models", "private-streams"}


def compatibility(root: Path) -> dict:
    worker = root / "packages/engine/dist/worker.cjs"
    manifests = sorted((root / "decks").glob("*.json"))
    return {"engineBuildHash": file_digest(worker) if worker.exists() else None,
            "deckManifestHash": digest({p.name: file_digest(p) for p in manifests}),
            "featureVersion": FEATURE_VERSION}


def _relative(value: str) -> Path:
    pure = PurePosixPath(value)
    if not value or not pure.parts or pure.is_absolute() or ".." in pure.parts or "\\" in value or ":" in value:
        raise ValueError("Bundle contains an unsafe relative path")
    parts = pure.parts[1:] if pure.parts[0] == "experimental" else pure.parts
    if len(parts) != 2 or parts[0] not in CATEGORIES or pure.parts[-1].startswith("."):
        raise ValueError("Bundle contains a category or nesting level that is not portable research data")
    if pure.suffix not in ({".jsonl"} if parts[0] == "private-streams" else {".json", ".pt", ".parquet"}):
        raise ValueError("Bundle artifact extension is not allowed")
    return Path(*pure.parts)


def _manifest(source: Path, expected: dict | None, max_bytes: int) -> dict:
    path = source / "manifest.json"
    if path.is_symlink() or path.stat().st_size > 16 * 1024**2:
        raise ValueError("Invalid bundle manifest")
    manifest = json.loads(path.read_text())
    if manifest.get("schemaVersion") != BUNDLE_VERSION:
        raise ValueError("Unsupported bundle schema")
    if digest({k: v for k, v in manifest.items() if k != "id"}) != manifest.get("id"):
        raise ValueError("Bundle manifest checksum mismatch")
    if expected is not None and manifest["compatibility"] != expected:
        raise ValueError("Bundle engine/deck/feature identities are incompatible")
    total, seen = 0, set()
    for entry in manifest["files"]:
        relative = _relative(entry["path"])
        if str(relative).casefold() in seen:
            raise ValueError("Duplicate bundle path")
        seen.add(str(relative).casefold())
        artifact = source / relative
        if any((source / Path(*relative.parts[:count])).is_symlink() for count in range(1, len(relative.parts) + 1)):
            raise ValueError("Symlink artifacts cannot be imported")
        if not artifact.is_file() or artifact.stat().st_size != entry["bytes"]:
            raise ValueError("Bundle artifact missing or size mismatch")
        total += entry["bytes"]
        if total > max_bytes:
            raise ResourceLimit("Bundle exceeds the destination data cap")
        if file_digest(artifact) != entry["sha256"]:
            raise ValueError("Bundle artifact checksum mismatch")
    # Evaluation and reserved-list games may be carried for evidence, but their
    # permanent exclusion travels with the bundle and is checked on ingestion.
    from .dataset import family_key, family_partition
    assignments = {}
    for entry in manifest["files"]:
        relative = _relative(entry["path"])
        if relative.parts[-2] != "replays":
            continue
        game = json.loads((source / relative).read_text())
        key, partition = family_key(game), family_partition(game)
        if relative.parts[0] == "experimental":
            key = "experimental:" + key
        if key in assignments and assignments[key] != partition:
            raise ValueError("Conflicting dataset partitions in the same game family")
        assignments[key] = partition
    if assignments != manifest["partitions"]:
        raise ValueError("Bundle partition manifest does not match its replays")
    return manifest


def export_bundle(store: Store, destination: Path, identities: dict, *, min_free: int = 20 * 1024**3) -> dict:
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("Immutable bundle destination already exists")
    if destination.is_relative_to(store.path.resolve()):
        raise ValueError("Export outside active storage to prevent recursive archives")
    entries, assignments = [], {}
    from .dataset import family_key, family_partition
    for base in (store.path, store.path / "experimental"):
        if base.is_symlink():
            raise ValueError("Bundle source directory cannot be a symlink")
        for category in sorted(CATEGORIES):
            directory = base / category
            if directory.is_symlink():
                raise ValueError("Bundle source directory cannot be a symlink")
            for path in sorted(directory.glob("*")):
                if not path.is_file() or path.suffix not in ({".jsonl"} if category == "private-streams" else {".json", ".pt", ".parquet"}):
                    continue
                relative = path.relative_to(store.path).as_posix()
                _relative(relative)
                if path.is_symlink():
                    raise ValueError("Bundle source cannot be a symlink")
                entries.append({"path": relative, "bytes": path.stat().st_size, "sha256": file_digest(path)})
                if category == "replays":
                    game = json.loads(path.read_text())
                    key, partition = family_key(game), family_partition(game)
                    if base != store.path:
                        key = "experimental:" + key
                    if key in assignments and assignments[key] != partition:
                        raise ValueError("Conflicting dataset partitions in source game family")
                    assignments[key] = partition
    size = sum(entry["bytes"] for entry in entries)
    if volume_free(destination.parent) < size + min_free:
        raise ResourceLimit("Export would violate the destination free-space reserve")
    if not destination.parent.is_dir():
        raise ValueError("Export parent must already exist; reconnect removable storage before exporting")
    staging = destination.with_name(f".{destination.name}-{uuid.uuid4().hex}.partial")
    manifest = {"schemaVersion": BUNDLE_VERSION, "compatibility": identities, "files": entries,
                "partitions": assignments, "provenance": {"sourceDataName": store.path.name},
                "trackers": "Local SQLite databases and private guide sources are intentionally excluded"}
    manifest["id"] = digest(manifest)
    try:
        staging.mkdir()
        for entry in entries:
            target = staging / _relative(entry["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(store.path / entry["path"], target)
            with target.open("rb") as file:
                os.fsync(file.fileno())
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
        _manifest(staging, identities, store.max_bytes)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"id": manifest["id"], "path": str(destination), "files": len(entries), "bytes": size}


def import_bundle(store: Store, source: Path, identities: dict, *, min_free: int = 20 * 1024**3) -> dict:
    manifest = _manifest(source.resolve(), identities, store.max_bytes)
    # Publish an entire independent data root in one rename. Mixing an import
    # piecemeal into an active run could leak partitions or corrupt a checkpoint.
    target = store.path / "imports" / manifest["id"]
    if target.exists():
        _manifest(target, identities, store.max_bytes)
        return {"id": manifest["id"], "dataRoot": str(target), "alreadyImported": True}
    required = sum(entry["bytes"] for entry in manifest["files"])
    check_storage(store.path, store.max_bytes, min_free, required)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(f".{target.name}-{uuid.uuid4().hex}.partial")
    try:
        staging.mkdir()
        for entry in manifest["files"]:
            relative = _relative(entry["path"])
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination)
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
        _manifest(staging, identities, store.max_bytes)
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"id": manifest["id"], "dataRoot": str(target), "alreadyImported": False,
            "continuation": "Use --data with this independent root; training --linked-resume creates a new child experiment"}
