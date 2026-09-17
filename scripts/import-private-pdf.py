"""Import a user-supplied PDF as private, page-addressable evidence.

Run with the project interpreter after installing the optional pypdf package.
Source files and passages belong in an ignored data directory, never Git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ptcg_lab.storage import Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/competitive"))
    parser.add_argument("--title", required=True)
    parser.add_argument("--author", required=True)
    parser.add_argument("--matchup", default="Supported five-deck research pool")
    parser.add_argument("--format-date", default="2026-09-17")
    parser.add_argument("--main-list-page", type=int)
    args = parser.parse_args()
    try:
        from pypdf import PdfReader
    except ImportError:
        parser.error("Install the optional pypdf package in the local environment first.")
    source = args.source.resolve(strict=True)
    if source.suffix.lower() != ".pdf" or source.stat().st_size > 64 * 1024**2:
        parser.error("Expected a local PDF no larger than 64 MiB.")
    content_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    identifier = content_hash[:32]
    chunks = []
    reader = PdfReader(source)
    for number, page in enumerate(reader.pages, 1):
        page_text = page.extract_text() or ""
        for part, start in enumerate(range(0, len(page_text), 1600), 1):
            chunks.append({"id": f"{identifier}-p{number}-{part}", "text": page_text[start:start + 1600],
                           "page": number, "sourceHash": content_hash})
    if not chunks:
        parser.error("No text extracted; this document needs explicit OCR and visual review.")
    args.data.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.data).free - source.stat().st_size < 20 * 1024**3:
        parser.error("Import would breach the 20 GiB free-space floor.")
    folder = args.data / "sources"
    folder.mkdir(exist_ok=True)
    target = folder / f"{content_hash}.pdf"
    if not target.exists():
        temporary = target.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            with source.open("rb") as input_file, temporary.open("xb") as output:
                shutil.copyfileobj(input_file, output)
                output.flush()
                os.fsync(output.fileno())
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != content_hash:
                raise ValueError("Source changed during import")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    record = {"id": identifier, "title": args.title, "author": args.author,
              "source": f"User-supplied private PDF: {source.name}",
              "sourceFile": target.relative_to(args.data).as_posix(), "contentHash": content_hash,
              "formatDate": args.format_date, "matchup": args.matchup, "mainListPage": args.main_list_page,
              "reviewStatus": "unreviewed", "trainingEligible": False, "chunks": chunks,
              "use": "Private attributed evidence; not an authoritative rule or training label."}
    Store(args.data, 25 * 1024**3).put("guides", identifier, record)
    print(json.dumps({"id": identifier, "pages": len(reader.pages), "chunks": len(chunks),
                      "contentHash": content_hash, "trainingEligible": False}, indent=2))


if __name__ == "__main__":
    main()
