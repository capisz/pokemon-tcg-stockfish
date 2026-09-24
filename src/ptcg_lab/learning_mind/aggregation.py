from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile

from .dataset_v1 import file_sha256
from .schema import identity_hash

POLICY_FAMILIES = {"python-heuristic", "typescript-heuristic"}


def combine_macro_label_runs(*, inputs: list[Path], output: Path, identity: dict) -> dict:
    """Verify separate policy-family label runs and combine them for ranker evaluation."""
    if not inputs:
        raise ValueError("at least one macro-label run is required")
    output = output.resolve()
    if output.exists():
        raise ValueError("combined macro-label outputs are immutable; choose a new directory")
    loaded = []
    position_hashes = set()
    families = set()
    generator_identity = None
    collector_version = None
    collector_sha = None
    for input_dir in inputs:
        input_dir = input_dir.resolve()
        manifest_path = input_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        recorded_hash = manifest.get("manifestHash")
        if recorded_hash != identity_hash({key: value for key, value in manifest.items()
                                             if key != "manifestHash"}):
            raise ValueError(f"input manifest hash mismatch: {input_dir}")
        if manifest.get("identity") != identity:
            raise ValueError(f"input experiment identity mismatch: {input_dir}")
        if not isinstance(manifest.get("files"), list) or len(manifest["files"]) != manifest.get("positions"):
            raise ValueError(f"input manifest file list mismatch: {input_dir}")
        if generator_identity is None:
            generator_identity = manifest.get("candidateGeneratorIdentity")
            collector_version = manifest.get("labelCollectorVersion")
            collector_sha = manifest.get("labelCollectorSha256")
        elif (manifest.get("candidateGeneratorIdentity") != generator_identity
              or manifest.get("labelCollectorVersion") != collector_version
              or manifest.get("labelCollectorSha256") != collector_sha):
            raise ValueError("macro-label runs use different candidate generators or collectors")
        records = []
        for item in manifest["files"]:
            name = item.get("path")
            if not isinstance(name, str) or Path(name).name != name or not name.endswith(".json") or name == "manifest.json":
                raise ValueError(f"unsafe macro-label record path in {input_dir}")
            source = input_dir / name
            if file_sha256(source) != item.get("sha256"):
                raise ValueError(f"macro-label record hash mismatch: {source}")
            record = json.loads(source.read_text())
            key = record.get("positionHash")
            if not isinstance(key, str) or name != f"{key}.json":
                raise ValueError(f"macro-label record filename/position mismatch: {source}")
            if key in position_hashes:
                raise ValueError(f"duplicate macro-label position across runs: {key}")
            if record.get("identity") != identity or record.get("datasetManifestHash") != manifest.get("datasetManifestHash"):
                raise ValueError(f"macro-label record identity mismatch: {source}")
            if record.get("rolloutIdentity") != manifest.get("rolloutIdentity"):
                raise ValueError(f"macro-label rollout identity mismatch: {source}")
            family = record.get("opponentPolicyFamily")
            if family not in POLICY_FAMILIES:
                raise ValueError(f"macro-label record lacks policy-family provenance: {source}")
            if record.get("status") != "collected":
                raise ValueError(f"macro-label record is not collected: {source}")
            position_hashes.add(key)
            families.add(family)
            records.append((source, item, record, family))
        if len({record[3] for record in records}) != 1:
            raise ValueError(f"each input run must contain exactly one policy family: {input_dir}")
        if set(manifest.get("selectedPositionHashes", [])) != {record[2]["positionHash"] for record in records}:
            raise ValueError(f"selected positions differ from input records: {input_dir}")
        loaded.append({"directory": input_dir, "manifestPath": manifest_path,
                       "manifest": manifest, "records": records})

    if families != POLICY_FAMILIES:
        raise ValueError("combined ranker input must contain both Python and TypeScript policy families")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        files = []
        splits = Counter()
        policy_labels = 0
        for run in loaded:
            for source, item, record, family in run["records"]:
                relative = Path(family) / source.name
                target = temporary_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
                with target.open("rb") as stream:
                    os.fsync(stream.fileno())
                files.append({"path": relative.as_posix(), "sha256": file_sha256(target)})
                splits[record["split"]] += 1
                policy_labels += int(record.get("highConfidencePolicyEligible") is True)
        files.sort(key=lambda item: item["path"])
        combined = {
            "schemaVersion": 1,
            "kind": "combined-macro-label-runs-v1",
            "identity": identity,
            "candidateGeneratorIdentity": generator_identity,
            "labelCollectorVersion": collector_version,
            "labelCollectorSha256": collector_sha,
            "sourceRuns": [{
                "policyFamilies": sorted({record[3] for record in run["records"]}),
                "datasetManifestHash": run["manifest"]["datasetManifestHash"],
                "rolloutIdentity": run["manifest"]["rolloutIdentity"],
                "manifestHash": run["manifest"]["manifestHash"],
                "manifestSha256": file_sha256(run["manifestPath"]),
                "positions": len(run["records"]),
            } for run in loaded],
            "positions": len(files),
            "positionsBySplit": dict(sorted(splits.items())),
            "policyFamilies": sorted(families),
            "highConfidencePolicyLabels": policy_labels,
            "files": files,
            "rawInputsRemainImmutable": True,
        }
        combined["manifestHash"] = identity_hash(combined)
        manifest_path = temporary_root / "manifest.json"
        manifest_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
        with manifest_path.open("rb") as stream:
            os.fsync(stream.fileno())
        temporary_root.replace(output)
        return combined
    except Exception:
        shutil.rmtree(temporary_root)
        raise
