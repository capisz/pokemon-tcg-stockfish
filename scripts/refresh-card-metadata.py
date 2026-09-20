#!/usr/bin/env python3
"""Resolve registered printings to public TCGdex metadata; never download artwork.

This is a metadata cross-check, not official format-legality certification.
Raw responses stay in ignored data/; only verified identities and URLs are published.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import unicodedata
from urllib.request import Request, urlopen

SETS = dict(TEF="sv05", TWM="sv06", SFA="sv06.5", SCR="sv07", SSP="sv08",
            JTG="sv09", DRI="sv10", BLK="sv10.5b", WHT="sv10.5w", SVE="sve",
            MEE="mee", MEG="me01", PFL="me02", ASC="me02.5", POR="me03",
            CRI="me04", PBL="me05")
API = "https://api.tcgdex.net/v2/en"


def normalized(value):
    value = value.replace("[G]", "Grass")
    return "".join(c for c in unicodedata.normalize("NFKD", value).casefold() if c.isalnum())


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=Path("formats/card-registry.json"))
    parser.add_argument("--output", type=Path, default=Path("formats/card-art.json"))
    parser.add_argument("--cache", type=Path, default=Path("data/competitive/card-metadata"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    cards = json.loads(args.registry.read_text())["cards"]

    def fetch(kind, identity):
        path = args.cache / kind / (identity + ".json")
        if path.exists() and not args.refresh:
            return json.loads(path.read_text())
        request = Request(f"{API}/{kind}/{identity}", headers={"User-Agent": "ptcg-lab/0.2 metadata-audit"})
        with urlopen(request, timeout=30) as response:
            raw = response.read(4 * 1024 * 1024)
        value = json.loads(raw)
        atomic_json(path, value)
        return value

    needed = sorted({c["cardId"].split("-")[0] for c in cards})
    with ThreadPoolExecutor(max_workers=4) as pool:
        sets = dict(zip(needed, pool.map(lambda s: fetch("sets", SETS[s]), needed)))

    def resolve(card):
        key = card["cardId"]
        code, number = key.split("-", 1)
        try:
            matches = [c for c in sets[code]["cards"] if str(c["localId"]).lstrip("0") == number.lstrip("0")]
            if len(matches) != 1:
                raise ValueError("printing number did not resolve uniquely")
            data = fetch("cards", matches[0]["id"])
            # Energy naming differs by provider. Accept only explicit Basic prefix equivalence.
            expected = normalized(card["name"])
            actual = normalized(data["name"])
            if expected != actual and "basic" + expected != actual:
                raise ValueError(f"name mismatch: registry={card['name']!r}, provider={data['name']!r}")
            if data.get("regulationMark") and data["regulationMark"] != card.get("regulationMark"):
                raise ValueError(f"regulation mark mismatch: {card.get('regulationMark')} vs {data['regulationMark']}")
            image = data.get("image")
            if not image or not image.startswith("https://assets.tcgdex.net/"):
                raise ValueError("verified image URL unavailable")
            digest = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
            return key, {"name": data["name"], "providerCardId": data["id"],
                         "imageUrl": image + "/high.webp", "sourceUrl": f"{API}/cards/{data['id']}",
                         "metadataHash": digest}, None
        except Exception as error:
            return key, None, str(error)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(resolve, cards))
    verified = {key: value for key, value, error in sorted(results) if value}
    failures = {key: error for key, value, error in sorted(results) if error}
    atomic_json(args.output, {"schemaVersion": 1, "provider": "TCGdex", "cards": verified})
    report = {"checkedAt": datetime.now(timezone.utc).isoformat(), "verified": len(verified),
              "total": len(cards), "failures": failures,
              "limitation": "Provider identity cross-check only; not official release, reprint or rules validation."}
    atomic_json(args.cache / "audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
