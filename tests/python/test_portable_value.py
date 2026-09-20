"""Numerical parity between PyTorch and the dependency-free simulator evaluator."""
import copy
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest


def test_portable_value_inputs_and_outputs_match_pytorch(observation):
    torch = pytest.importorskip("torch")
    from ptcg_lab.model import PolicyResourceModel
    from ptcg_lab.features import FEATURE_NAMES, FEATURE_VERSION, resource_features, card_tokens
    torch.manual_seed(912)
    model = PolicyResourceModel().eval()
    weights = {key: value.tolist() for key, value in model.state_dict().items()
               if key == "baseline" or key.startswith(("card_embedding.", "resource_terms.", "context.", "interaction."))}
    payload = json.dumps({"featureVersion": FEATURE_VERSION, "featureNames": list(FEATURE_NAMES), "cardBuckets": 2048,
                          "maxVisibleCards": 128, "modelVersion": "numerical-test-only", "checkpointHash": "a" * 64,
                          "valueTrained": True, "weights": weights}, separators=(",", ":"), allow_nan=False)
    envelope = {"schemaVersion": 1, "payload": payload, "hash": hashlib.sha256(payload.encode()).hexdigest()}
    examples = []
    for player_id in (0, 1):
        for energies in (["Grass Energy"], ["Mist Energy", "Grass Energy"], ["Unidentified Energy"], ["Basic [G] Energy", "Spiky Energy"]):
            item = copy.deepcopy(observation)
            item["playerId"] = player_id
            item["players"][0]["active"]["energy"] = list(energies)
            item["players"][1]["active"]["card"]["name"] = "Dragapult ex"
            item["players"][1]["active"]["card"]["prizeValue"] = 2
            item["players"][0]["hand"] = [{"id": "POR-081", "name": "Poké Pad"}, {"id": "TWM-165", "name": "Unfair Stamp"}]
            item["players"][1]["hand"] = [{"id": "SECRET-ONLY", "name": "Opponent private card"}]
            examples.append(item)
    # Token truncation and repeated physical printings have deterministic parity.
    examples[-1]["players"][0]["discard"] = [{"id": f"test-{i}", "name": "Card"} for i in range(135)]
    source = """
import {loadLeafModel,valueFeatures,valueCardTokens} from './packages/engine/src/learned-value.ts';
let text='';for await(const part of process.stdin)text+=part;
const {envelope,examples}=JSON.parse(text),model=loadLeafModel(envelope);
const rows=examples.map(o=>({features:valueFeatures(o),tokens:valueCardTokens(o),value:model.evaluate(o)}));
let rejected=false;try{loadLeafModel({...envelope,hash:'incorrect'});}catch{rejected=true;}
process.stdout.write(JSON.stringify({rows,rejected}));
"""
    completed = subprocess.run(["node", "--import", "tsx", "--input-type=module", "-e", source],
                               input=json.dumps({"envelope": envelope, "examples": examples}), text=True,
                               capture_output=True, cwd=Path(__file__).resolve().parents[2], timeout=30, check=True)
    result = json.loads(completed.stdout)
    assert result["rejected"]
    assert len(json.dumps({"method": "search", "params": {"observation": observation, "leafModel": envelope}})) < 1024**2
    with torch.no_grad():
        for row, example in zip(result["rows"], examples):
            resources = resource_features(example)
            tokens = card_tokens(example)
            np.testing.assert_array_equal(row["features"], resources)
            np.testing.assert_array_equal(row["tokens"], tokens)
            expected = torch.sigmoid(model(torch.tensor(resources[None, :]), torch.tensor(tokens[None, :]))["logit"])[0].item()
            assert row["value"] == pytest.approx(expected, abs=2e-6)
