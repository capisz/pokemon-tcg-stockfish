"""Immutable copies shared by interactive runs; no database or live file sync."""
import os
import shutil
import uuid
from pathlib import Path

from .resources import check_storage
from .storage import file_digest


def freeze_checkpoint(store, source: Path) -> Path:
    fingerprint = file_digest(source)
    destination = store.path / "private-models" / f"{fingerprint}.pt"
    if destination.exists():
        if file_digest(destination) != fingerprint:
            raise ValueError("Frozen checkpoint checksum differs; restore the recorded file")
        return destination
    check_storage(store.path, store.max_bytes, store.min_free_bytes, additional=source.stat().st_size)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
    try:
        with source.open("rb") as incoming, temporary.open("wb") as outgoing:
            shutil.copyfileobj(incoming, outgoing)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if file_digest(temporary) != fingerprint:
            raise ValueError("Checkpoint changed while making its frozen copy")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
