"""Explicit offline fork of a paused experimental comparison after code repair.

No identity bypass and no job startup: the successor begins a fresh comparison
under current identities while its original data, models, and journals remain
immutable source evidence.
"""
from __future__ import annotations

import copy
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .storage import Store, digest, file_digest

CATEGORY = "learning-recoveries"
MAX_RECEIPT_BYTES = 8 * 1024**2


def _hash(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read(store, category, identifier, limit=MAX_RECEIPT_BYTES):
    path = store.location(category, identifier)
    if path.is_symlink() or path.parent.is_symlink() or path.stat().st_size > limit:
        raise ValueError("Recovery input is linked or exceeds its bounded size")
    return store.get(category, identifier)


def _validate_sources(store, parent, decks):
    from .training import load_model
    if (parent.get("schemaVersion") != 1 or parent.get("phase") != "comparing"
            or parent.get("status") != "paused" or parent.get("desired") != "paused"
            or parent.get("dataTier", "experimental") != "experimental"
            or not _hash(parent.get("engine", {}).get("engineBuildHash"))
            or not parent.get("engine", {}).get("engineVersion") or not _hash(parent.get("implementationHash"))):
        raise ValueError("Recovery only forks a paused experimental comparison")
    if (digest(parent.get("decks")) != parent.get("deckHash") or digest(decks) != parent["deckHash"]
            or len(decks) != 10 or len({deck["archetype"] for deck in decks}) != 5
            or any(deck.get("role") not in {"main", "training-variant"} for deck in decks)):
        raise ValueError("Frozen deck registry differs; this recovery cannot change deck identities")
    manifest = _read(store, "datasets", parent["datasetId"], 16 * 1024**2)
    if (manifest.get("id") != parent["datasetId"] or manifest.get("dataTier") != "experimental"
            or digest({k: v for k, v in manifest.items() if k != "hash"}) != manifest.get("hash")):
        raise ValueError("Pinned recovery dataset manifest checksum differs")
    rows = _read(store, "dataset-rows", parent["datasetId"], 64 * 1024**2)
    if digest(rows) != manifest.get("rowsHash") or len(rows) != manifest.get("rows"):
        raise ValueError("Pinned recovery dataset rows changed")
    snapshot = _read(store, "teaching-snapshots", parent["teachingSnapshotId"], 64 * 1024**2)
    if (digest({k: v for k, v in snapshot.items() if k != "hash"}) != snapshot.get("hash")
            or snapshot.get("hash") != manifest.get("teachingSnapshotHash")):
        raise ValueError("Pinned recovery teaching snapshot changed")
    history = parent.get("history", [])
    if not isinstance(history, list) or len(history) > 1000:
        raise ValueError("Historical opponent list exceeds the bounded recovery scope")
    models = {}
    policies = [(name, parent.get(name)) for name in ("candidate", "incumbent", "guidePolicy")]
    policies.extend((f"history:{index}", policy) for index, policy in enumerate(history))
    for name, policy in policies:
        if not isinstance(policy, dict):
            raise ValueError("Recovery requires frozen candidate, incumbent, and guide policy")
        if (name == "incumbent" or name.startswith("history:")) and policy.get("path") == policy.get("hash") == "heuristic":
            models[name] = "heuristic"
            continue
        path = Path(policy.get("path", ""))
        if (path.is_symlink() or path.parent.is_symlink() or path.parent.resolve() != (store.path / "private-models").resolve()
                or path.name != f"{policy.get('hash')}.pt" or path.stat().st_size > 64 * 1024**2
                or file_digest(path) != policy.get("hash")):
            raise ValueError("Frozen recovery model path or bytes differ")
        _, checkpoint = load_model(path, allow_experimental=True)
        if checkpoint.get("dataTier") != "experimental":
            raise ValueError("Recovery cannot import trusted models into a quarantined lineage")
        if name == "candidate" and (checkpoint.get("experimentId") != parent.get("trainingId")
                                    or checkpoint.get("config", {}).get("datasetHash") != digest(manifest)):
            raise ValueError("Candidate is not the model trained on the pinned recovery dataset")
        if name == "guidePolicy" and checkpoint.get("modelKind") != "policy-only":
            raise ValueError("Recovery guide checkpoint is not a reviewed policy-only model")
        if file_digest(path) != policy["hash"]:
            raise ValueError("Frozen model changed during recovery validation")
        models[name] = policy["hash"]
    # Existing collections remain historical inputs. New comparison seeds are
    # namespaced by the successor ID; no old comparison game becomes training.
    for identifier in parent.get("collectedReplayIds", []):
        path = store.location("replays", identifier)
        if not path.is_file() or path.is_symlink():
            raise ValueError("An inherited collection replay is unavailable")
    return {"datasetHash": digest(manifest), "rowsHash": manifest["rowsHash"],
            "teachingSnapshotHash": snapshot["hash"], "models": models}


def fork_paused_comparison(store: Store, *, parent_id: str, expected_revision: int,
                           expected_parent_hash: str, current_engine: dict,
                           implementation_hash: str, decks: list[dict]) -> dict:
    """Prepare an idempotent paused successor. Caller performs explicit API resume.

    Call with the API stopped or otherwise idle. Original parent and pending-game
    snapshots live only in the private recovery receipt, outside public run fields.
    """
    store.location("learning-runs", parent_id)
    if (store.path.name != "experimental" or "imports" in store.path.parts or store.path.is_symlink()
            or (store.path.parent / "manifest.json").exists()
            or not _hash(expected_parent_hash) or type(expected_revision) is not int or expected_revision < 0
            or not _hash(implementation_hash) or not isinstance(current_engine, dict)
            or not _hash(current_engine.get("engineBuildHash")) or not current_engine.get("engineVersion")
            or current_engine.get("protocolVersion") != 1):
        raise ValueError("Recovery requires a local experimental store and exact current identities")
    identity = {"schemaVersion": 1, "kind": "linked-comparison-recovery", "parentId": parent_id,
                "parentRevision": expected_revision, "parentHash": expected_parent_hash,
                "engine": current_engine, "implementationHash": implementation_hash, "deckHash": digest(decks)}
    recovery_id = digest(identity)
    successor_id = recovery_id[:32]
    receipt_path = store.location(CATEGORY, recovery_id)
    for path in (store.path / "learning-runs").glob("*.json"):
        if path.stem == parent_id:
            continue
        existing = _read(store, "learning-runs", path.stem)
        if path.stem == successor_id and receipt_path.exists():
            if existing.get("status") != "paused" or existing.get("desired") != "paused":
                raise ValueError("Prepared successor is no longer paused; do not repair an active run")
        elif existing.get("status") != "stopped":
            raise ValueError("Another learning run is active or paused; resolve it before recovery")
    current_parent = _read(store, "learning-runs", parent_id)
    if receipt_path.exists():
        receipt = _read(store, CATEGORY, recovery_id)
        if (receipt.get("identity") != identity or receipt.get("id") != recovery_id
                or digest({k: v for k, v in receipt.items() if k != "hash"}) != receipt.get("hash")):
            raise ValueError("Immutable recovery receipt changed")
        parent = receipt["parentSnapshot"]
        if digest(parent) != expected_parent_hash or parent.get("revision") != expected_revision:
            raise ValueError("Recovery snapshot does not match its immutable original identity")
        if digest(current_parent) not in {expected_parent_hash, digest(receipt["archivedParent"])}:
            raise ValueError("Parent changed after the recovery was prepared")
    else:
        parent = current_parent
        if digest(parent) != expected_parent_hash or parent.get("revision") != expected_revision:
            raise ValueError("Parent identity or revision changed; read it again before recovery")
        evidence = _validate_sources(store, parent, decks)
        pending = {identifier: _read(store, "learning-games", identifier) for identifier in parent.get("pendingGames", [])}
        for identifier, game in pending.items():
            if game.get("id") != identifier or game.get("runId") != parent_id or game.get("purpose") != "comparison":
                raise ValueError("Pending journal does not belong to the original comparison")
        created = datetime.now(timezone.utc).isoformat()
        successor = copy.deepcopy(parent)
        successor.update(id=successor_id, dataTier="experimental", revision=0, status="paused", desired="paused",
                         createdAt=created, updatedAt=created, engine=copy.deepcopy(current_engine),
                         implementationHash=implementation_hash, controls={}, pendingGames=[],
                         comparisonCursor=0, comparisonRecords=[], error=None,
                         pauseReason="Linked recovery is prepared; explicitly resume to start a fresh comparison.",
                         recovery={"id": recovery_id, "parentId": parent_id, "parentHash": expected_parent_hash,
                                   "comparisonRestarted": True})
        successor["metrics"] = copy.deepcopy(parent["metrics"])
        successor["metrics"]["comparison"] = None
        archived = copy.deepcopy(parent)
        archived.update(status="stopped", desired="stopped", revision=expected_revision+1, updatedAt=created,
                        successorId=successor_id, recoveryId=recovery_id,
                        pauseReason="Archived unchanged work in a linked recovery; comparison restarts with fresh seeds.")
        receipt = {"id": recovery_id, "identity": identity, "createdAt": created, "dataTier": "experimental",
                   "parentSnapshot": parent, "pendingGameSnapshots": pending, "validatedSources": evidence,
                   "successorSnapshot": successor, "archivedParent": archived,
                   "scope": "Original comparison evidence is archived, excluded from the fresh comparison and never relabeled as training."}
        receipt["hash"] = digest(receipt)
        if len(json.dumps(receipt).encode()) > MAX_RECEIPT_BYTES:
            raise ValueError("Recovery receipt exceeds its 8 MiB bound")
        if store.location("learning-runs", successor_id).exists():
            raise ValueError("Deterministic successor exists without its recovery receipt")
        store.put(CATEGORY, recovery_id, receipt)
    # Revalidate source bytes on retries too. The receipt is a preparation marker,
    # not permission to skip checks after a crash or concurrent file replacement.
    if _validate_sources(store, parent, decks) != receipt["validatedSources"]:
        raise ValueError("Recovery source evidence changed")
    for identifier, journal in receipt["pendingGameSnapshots"].items():
        if _read(store, "learning-games", identifier) != journal:
            raise ValueError("Original pending game journal changed during recovery")
    successor_path = store.location("learning-runs", successor_id)
    if successor_path.exists():
        successor = _read(store, "learning-runs", successor_id)
        if successor.get("recovery") != receipt["successorSnapshot"]["recovery"]:
            raise ValueError("Successor provenance differs from its recovery receipt")
    else:
        successor = receipt["successorSnapshot"]
        store.put("learning-runs", successor_id, successor)
    # Parent archival is last; an interrupted preparation cannot silently enable
    # computation. Repeating the call completes this final acknowledgement.
    if digest(current_parent) == expected_parent_hash:
        if digest(_read(store, "learning-runs", parent_id)) != expected_parent_hash:
            raise ValueError("Parent changed before final recovery archival")
        store.put("learning-runs", parent_id, receipt["archivedParent"])
    return {"recoveryId": recovery_id, "parentId": parent_id, "successorId": successor_id,
            "status": successor["status"], "comparisonRestarted": True,
            "instructions": "Explicitly resume the paused successor through the API; importing or preparing recovery never starts work."}
