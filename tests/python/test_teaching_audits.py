from __future__ import annotations

import copy
from pathlib import Path

import pytest

from ptcg_lab.dataset import demonstration_records, demonstration_manifest
from ptcg_lab.engine import EngineClient, EngineError
from ptcg_lab.fixtures import from_fixture
from ptcg_lab.storage import Store, digest
from ptcg_lab.teaching import review, eligible
from ptcg_lab.teaching_audits import audit_review, attestation_valid

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def receipt():
    with EngineClient(ROOT) as engine:
        value = engine.request("fixture", {"fixtureId": "crustle-delay-prize", "variationId": "crustle-delay-prize-2"})
    # The engine may audit an older receipt only if the complete information set
    # and revision-scoped action bindings still match the current recipe.
    value["engineVersion"] = "original-engine-version"
    value["fixtureHash"] = "original-fixture-hash"
    return value


def saved_review(tmp_path, receipt):
    store = Store(tmp_path / "data")
    record = from_fixture(store, ROOT, copy.deepcopy(receipt))
    actions = [a["id"] for a in record["observation"]["legalActions"]
               if a["type"] == "attach-energy" and (a.get("cardId") == "POR-86"
                  or a.get("cardId") == "JTG-159" and a.get("targetRef", {}).get("zone") == "bench")]
    record = review(store, record["id"], review_status="reviewed", acceptable_action_ids=actions,
                    reasoning="Synthetic annotation: preserve future energy options.", critical_resources="Spare energy",
                    confidence="likely")
    assert len(actions) == 3 and not eligible(record)
    return store, record


def test_append_only_audit_admits_the_unchanged_review_and_manifest(tmp_path, receipt):
    store, record = saved_review(tmp_path, receipt)
    paths = [store.location("teaching", record["id"]), store.location("teaching-reviews", record["reviewHash"]),
             store.location("fixture-receipts", record["id"])]
    original = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
    assert demonstration_records(store) == []
    attestation = audit_review(store, ROOT, record["id"])
    assert attestation_valid(record, attestation)
    assert attestation["sourceEngineVersion"] != attestation["engineVersion"]
    again = audit_review(store, ROOT, record["id"])
    assert again == attestation and len(store.list("teaching-audits")) == 1
    assert all((path.read_bytes(), path.stat().st_mtime_ns) == expected for path, expected in original.items())
    admitted = demonstration_records(store)
    assert len(admitted) == 1
    assert admitted[0]["reviewHash"] == record["reviewHash"]
    assert admitted[0]["reviewedAt"] == record["reviewedAt"]
    assert admitted[0]["acceptableActionIds"] == record["acceptableActionIds"]
    assert admitted[0]["criticalResources"] == record["criticalResources"]
    assert not admitted[0]["trainingEligible"] and admitted[0]["effectiveTrainingEligible"]
    manifest = demonstration_manifest(admitted)[0]
    assert manifest["recordHash"] == digest(record)
    assert manifest["mechanicsAttestationHash"] == attestation["attestationHash"]
    assert manifest["fixtureReceiptHash"] == record["fixtureReceiptHash"]
    assert demonstration_records(store, partition="test") == []


@pytest.mark.parametrize("category", ["teaching", "teaching-reviews", "fixture-receipts", "teaching-audits"])
def test_changed_source_or_audit_is_not_admitted(tmp_path, receipt, category):
    store, record = saved_review(tmp_path, receipt)
    audit = audit_review(store, ROOT, record["id"])
    identifier = {"teaching": record["id"], "teaching-reviews": record["reviewHash"],
                  "fixture-receipts": record["id"], "teaching-audits": audit["id"]}[category]
    changed = store.get(category, identifier)
    changed["unexpectedMutation"] = True
    store.put(category, identifier, changed)
    assert demonstration_records(store) == []


def test_different_review_cannot_reuse_earlier_attestation(tmp_path, receipt):
    store, record = saved_review(tmp_path, receipt)
    audit_review(store, ROOT, record["id"])
    review(store, record["id"], review_status="reviewed", acceptable_action_ids=record["acceptableActionIds"][:1],
           reasoning="A revised synthetic preference.")
    assert demonstration_records(store) == []


def test_meaningful_position_change_refuses_audit_without_artifacts(tmp_path, receipt):
    changed = copy.deepcopy(receipt)
    changed["observation"]["players"][0]["active"]["damage"] += 10
    store, record = saved_review(tmp_path, changed)
    with pytest.raises(EngineError, match="saved observation"):
        audit_review(store, ROOT, record["id"])
    assert store.list("teaching-audits") == []


def test_rehashed_wrong_action_binding_fails_validation(tmp_path, receipt):
    store, record = saved_review(tmp_path, receipt)
    audit = audit_review(store, ROOT, record["id"])
    audit["acceptedActionBindings"][record["acceptableActionIds"][0]]["targetRef"]["playerId"] = 1
    payload = {key: value for key, value in audit.items() if key not in {"id", "attestationHash"}}
    audit.update(id=digest(payload), attestationHash=digest(payload))
    assert not attestation_valid(record, audit)
