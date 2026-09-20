from __future__ import annotations

import base64
import copy
import gzip
import json
from pathlib import Path

import pytest

from conftest import make_replay
from ptcg_lab import storage
from ptcg_lab.storage import Store, digest


def experimental_replay(observation):
    replay = make_replay(observation)
    replay.update(dataTier="experimental", trainingEligible=False, experimentalLearning=True)
    replay["frames"] *= 40
    return replay


def test_experimental_roundtrip_is_transparent_deterministic_and_idempotent(tmp_path, observation):
    store = Store(tmp_path)
    replay = experimental_replay(observation)
    original = copy.deepcopy(replay)
    first = store.save_replay(replay)
    path = store.location("replays", first)
    before = path.read_bytes()
    raw = json.loads(before)
    assert "frames" not in raw and raw["compressedFrames"]["codec"] == "gzip+base64"
    assert raw["seed"] == replay["seed"] and raw["deckRoles"] == replay["deckRoles"]
    assert len(before) < len(json.dumps(replay).encode()) // 5
    assert store.get("replays", first) == original and replay == original
    assert list(store.iter_records("replays")) == [original]
    assert store.save_replay(replay) == first and path.read_bytes() == before
    assert len(list((tmp_path / "replays").glob("*.json"))) == 1


def test_legacy_and_trusted_replays_are_unchanged(tmp_path, observation):
    store = Store(tmp_path)
    for index, tier in enumerate((None, "verified")):
        replay = make_replay(observation, index)
        if tier:
            replay["dataTier"] = tier
        store.save_replay(replay)
        raw = json.loads(store.location("replays", replay["id"]).read_text())
        assert raw == replay and "compressedFrames" not in raw
        assert store.get("replays", replay["id"]) == replay


def test_compressed_replay_projects_exactly_the_same_player_information(tmp_path, observation):
    from ptcg_lab.presentation import project_replay
    store = Store(tmp_path)
    replay = experimental_replay(observation)
    store.save_replay(replay)
    for player in (0, 1):
        expected = project_replay(replay, player)
        assert project_replay(store.get("replays", replay["id"]), player) == expected
        assert expected["decks"][1-player] == "opponent"
        assert "compressedFrames" not in expected


@pytest.mark.parametrize("change", ["codec", "checksum", "length", "oversized", "base64", "tail", "truncated", "count", "trusted"])
def test_invalid_compression_fails_closed(tmp_path, observation, change):
    store = Store(tmp_path)
    replay = experimental_replay(observation)
    store.save_replay(replay)
    path = store.location("replays", replay["id"])
    raw = json.loads(path.read_text())
    envelope = raw["compressedFrames"]
    if change == "codec": envelope["codec"] = "pickle"
    if change == "checksum": envelope["sha256"] = "0" * 64
    if change == "length": envelope["decodedBytes"] = 1
    if change == "oversized": envelope["decodedBytes"] = storage.MAX_REPLAY_FRAME_BYTES + 1
    if change == "base64": envelope["data"] = "not base64!"
    if change == "count": envelope["frameCount"] += 1
    if change == "trusted": raw["dataTier"] = "verified"
    if change in {"tail", "truncated"}:
        compressed = base64.b64decode(envelope["data"])
        compressed = compressed + gzip.compress(b"[]") if change == "tail" else compressed[:-8]
        envelope["data"] = base64.b64encode(compressed).decode()
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        store.get("replays", replay["id"])


def test_encoded_and_decoded_size_bounds_apply_before_artifact_publication(tmp_path, observation, monkeypatch):
    store = Store(tmp_path)
    replay = experimental_replay(observation)
    monkeypatch.setattr(storage, "MAX_REPLAY_FRAME_BYTES", 10)
    with pytest.raises(ValueError, match="decoded bound"):
        store.put("replays", replay["id"], replay)
    assert not store.location("replays", replay["id"]).exists()


def test_bundle_preserves_compressed_frames_and_family_manifest(tmp_path, observation):
    from ptcg_lab.bundles import export_bundle, import_bundle
    from ptcg_lab.features import FEATURE_VERSION
    main = Store(tmp_path / "source")
    store = Store(main.path / "experimental")
    replay = experimental_replay(observation)
    store.save_replay(replay)
    identities = {"engineBuildHash": "synthetic", "deckManifestHash": "synthetic", "featureVersion": FEATURE_VERSION}
    exported = export_bundle(main, tmp_path / "bundle", identities, min_free=0)
    imported = import_bundle(Store(tmp_path / "destination"), Path(exported["path"]), identities, min_free=0)
    restored = Store(Path(imported["dataRoot"]) / "experimental")
    assert restored.get("replays", replay["id"]) == replay
    assert digest(restored.get("replays", replay["id"])) == digest(replay)
    assert restored.location("replays", replay["id"]).read_bytes() == store.location("replays", replay["id"]).read_bytes()
