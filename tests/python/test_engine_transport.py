"""Exercise the actual worker transport without rebuilding the running dist file."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from ptcg_lab.engine import EngineClient, EngineError, MAX_REQUEST_BYTES

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def worker(tmp_path_factory):
    output = tmp_path_factory.mktemp("engine-transport") / "worker.cjs"
    source = """
import {build} from 'esbuild';
await build({entryPoints:['packages/engine/src/worker.ts'],outfile:process.argv[1],
  bundle:true,platform:'node',format:'cjs',target:'node22',tsconfig:'tsconfig.json',
  define:{__ENGINE_BUILD__:JSON.stringify('transport-test-only')}});
"""
    subprocess.run(["node", "--input-type=module", "-e", source, str(output)], cwd=ROOT,
                   capture_output=True, text=True, check=True, timeout=30)
    return output


def test_real_portable_value_search_larger_than_old_limit_reaches_engine(worker, observation):
    torch = pytest.importorskip("torch")
    from ptcg_lab.model import PolicyResourceModel
    from ptcg_lab.features import FEATURE_NAMES, FEATURE_VERSION
    torch.manual_seed(912)
    model = PolicyResourceModel().eval()
    weights = {key: value.tolist() for key, value in model.state_dict().items()
               if key == "baseline" or key.startswith(("card_embedding.", "resource_terms.", "context.", "interaction."))}
    payload = json.dumps({"featureVersion": FEATURE_VERSION, "featureNames": list(FEATURE_NAMES), "cardBuckets": 2048,
                         "maxVisibleCards": 128, "modelVersion": "transport-test-only", "checkpointHash": "a" * 64,
                         "valueTrained": True, "dataTier": "experimental", "weights": weights}, separators=(",", ":"))
    leaf = {"schemaVersion": 1, "payload": payload, "hash": hashlib.sha256(payload.encode()).hexdigest()}
    view = copy.deepcopy(observation)
    view["history"] = ["P1: resolved Choose cards. " * 20] * 1000
    view["searchUnavailableReason"] = "Synthetic long-history transport fixture"
    params = {"observation": view, "leafModel": leaf, "budgetMs": 1, "iterations": 1}
    encoded = json.dumps({"id": "x" * 32, "method": "search", "params": params}, ensure_ascii=False, separators=(",", ":")).encode()
    assert 1024**2 < len(encoded) < MAX_REQUEST_BYTES
    with EngineClient(ROOT, command=["node", str(worker)], timeout=20) as engine:
        result = engine.request("search", params)
        assert result["status"] == "unavailable"
        assert result["warnings"] == [view["searchUnavailableReason"]]
        assert engine.request("health")["ok"]


def test_client_rejects_utf8_oversize_before_start_or_write_and_remains_usable(tmp_path, monkeypatch):
    engine = EngineClient(tmp_path)
    started = []
    monkeypatch.setattr(engine, "start", lambda: started.append(True))
    # Fewer than 32 Mi characters, but more than 32 MiB in UTF-8.
    value = "⚡" * (MAX_REQUEST_BYTES // 3 + 1)
    assert len(value) < MAX_REQUEST_BYTES < len(value.encode("utf-8"))
    with pytest.raises(EngineError, match="32 MiB UTF-8 protocol limit before sending"):
        engine.request("health", {"text": value})
    assert started == [] and engine.process is None


def test_client_compacts_json_and_preserves_unicode(tmp_path):
    source = "import sys,json\nfor line in sys.stdin:\n r=json.loads(line); print(json.dumps({'id':r['id'],'result':{'raw':line.rstrip('\\n')}}),flush=True)"
    with EngineClient(tmp_path, command=[sys.executable, "-u", "-c", source]) as engine:
        result = engine.request("health", {"text": "Pokémon ⚡"})
        line = result["raw"]
        assert "Pokémon ⚡" in line and ": " not in line and ", " not in line


@pytest.mark.parametrize("worker_id,expected", [(None, "Request exceeds prior worker limit"), ("wrong-id", "request id mismatch")])
def test_preparse_error_is_preserved_but_wrong_nonnull_id_still_fails(tmp_path, worker_id, expected):
    reply = {"id": worker_id, "error": {"code": "ENGINE_ERROR", "message": "Request exceeds prior worker limit"}}
    source = "import sys,json\nfor line in sys.stdin:\n print(" + repr(json.dumps(reply)) + ",flush=True)"
    with EngineClient(tmp_path, command=[sys.executable, "-u", "-c", source]) as engine:
        with pytest.raises(EngineError, match=expected):
            engine.request("health")


def test_worker_uses_same_exact_utf8_boundary_and_recovers_after_preparse_errors(worker):
    with EngineClient(ROOT, command=["node", str(worker)], timeout=20) as engine:
        # Deliberately bypass client preflight to test the worker's independent
        # guard and resynchronization. These requests cannot touch a live game.
        prefix = '{"id":"at-limit","method":"health","params":{"padding":"'
        suffix = '"}}'
        line = prefix + "a" * (MAX_REQUEST_BYTES-len(prefix.encode())-len(suffix.encode())) + suffix
        assert len(line.encode()) == MAX_REQUEST_BYTES
        engine.process.stdin.write(line+"\n")
        engine.process.stdin.flush()
        response = json.loads(engine.responses.get(timeout=20))
        assert response["id"] == "at-limit" and response["result"]["ok"]
        # Same character count, two additional bytes. A character-length guard
        # would incorrectly accept this multibyte payload.
        line = prefix + "⚡" + line[len(prefix)+1:]
        assert len(line.encode()) == MAX_REQUEST_BYTES+2
        engine.process.stdin.write(line+"\n")
        engine.process.stdin.flush()
        response = json.loads(engine.responses.get(timeout=20))
        assert response["id"] is None and "32 MiB UTF-8" in response["error"]["message"]
        engine.process.stdin.write("not-json\n")
        engine.process.stdin.flush()
        response = json.loads(engine.responses.get(timeout=20))
        assert response["id"] is None and response["error"]["code"] == "ENGINE_ERROR"
        assert engine.request("health")["ok"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), object()])
def test_non_json_requests_fail_before_process_start(tmp_path, value):
    engine = EngineClient(tmp_path)
    with pytest.raises(EngineError, match="Cannot encode engine request"):
        engine.request("health", {"value": value})
    assert engine.process is None
