from __future__ import annotations

import json

from ptcg_lab.learning_mind import __main__


def test_specialist_verifier_cli_calls_source_verifier_and_reports_diagnostic_only(tmp_path, monkeypatch, capsys):
    captured = {}

    def verify(*, root, registry_path, output):
        captured.update(root=root, registry=registry_path, output=output)
        return {"status": "passed", "reportHash": "a" * 64, "specialistCount": 5}, object()

    monkeypatch.setattr(__main__, "verify_specialist_curriculum", verify)
    __main__.main(["verify-specialist-curriculum", "--root", str(tmp_path),
        "--registry", str(tmp_path / "registry.json"), "--output", str(tmp_path / "report.json")])
    result = json.loads(capsys.readouterr().out)

    assert result["verified"] is True
    assert result["specialistCount"] == 5
    assert result["automaticPromotion"] is False
    assert "not a routing capability" in result["runtimeNote"]
    assert captured == {"root": tmp_path.resolve(), "registry": (tmp_path / "registry.json").resolve(),
        "output": (tmp_path / "report.json").resolve()}
