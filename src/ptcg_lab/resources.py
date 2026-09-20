"""Cross-platform, fail-closed local resource accounting. No scheduler."""
from __future__ import annotations

import os
import shutil
import threading
from pathlib import Path


class ResourceLimit(RuntimeError):
    """Pause work; this exception is never a rules outcome."""


def process_memory() -> int:
    try:
        import psutil
    except ImportError as exc:
        raise ResourceLimit("Process accounting requires psutil; install the base project dependencies") from exc
    parent = psutil.Process(os.getpid())
    total = 0
    try:
        processes = [parent, *parent.children(recursive=True)]
    except (psutil.AccessDenied, OSError) as exc:
        raise ResourceLimit("Cannot account for the project process tree; run from a terminal with process inspection access") from exc
    for process in processes:
        try:
            total += process.memory_info().rss
        except psutil.NoSuchProcess:
            pass
        except psutil.AccessDenied as exc:
            raise ResourceLimit("Cannot account for a project child process") from exc
    return total


def directory_bytes(path: Path) -> int:
    # Explicitly prune POSIX links and Windows reparse points (including
    # junctions); pathlib traversal behavior differs between Python versions.
    def linked(item: Path) -> bool:
        metadata = item.lstat()
        return item.is_symlink() or bool(getattr(metadata, "st_file_attributes", 0) & 0x400)

    total = 0
    if not path.exists() or linked(path):
        return 0
    for directory, dirs, files in os.walk(path, followlinks=False):
        dirs[:] = [name for name in dirs if not linked(Path(directory) / name)]
        for name in files:
            item = Path(directory) / name
            if not linked(item) and item.is_file():
                total += item.stat().st_size
    return total


def volume_free(path: Path) -> int:
    existing = path.resolve()
    while not existing.exists():
        if existing == existing.parent:
            raise ResourceLimit("Destination volume is unavailable")
        existing = existing.parent
    return shutil.disk_usage(existing).free


def check_storage(path: Path, max_bytes: int, min_free: int, additional: int = 0) -> dict:
    used, free = directory_bytes(path), volume_free(path)
    if used + additional > max_bytes:
        raise ResourceLimit("Local data disk cap reached; archive data before continuing")
    if free - additional < min_free:
        raise ResourceLimit("Destination free-space reserve reached; work paused without a game result")
    return {"managedBytes": used, "freeBytes": free}


class ResourceGuard:
    """Poll memory every second and storage at safe decision/batch boundaries.

    A target is not an OS hard cap. The worker terminates on the next request
    boundary; explicit bounded worker requests keep an overrun recoverable.
    """

    def __init__(self, settings):
        self.settings = settings
        self.peak = 0
        self.failure: str | None = None
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None

    def check(self, storage: bool = True) -> None:
        if self.failure:
            raise ResourceLimit(self.failure)
        usage = process_memory()
        self.peak = max(self.peak, usage)
        if usage > self.settings.max_memory_bytes:
            raise ResourceLimit("Aggregate project process memory target exceeded; resume with fewer workers")
        if storage:
            check_storage(self.settings.data, self.settings.max_disk_bytes, self.settings.min_free_bytes)

    def __enter__(self):
        self.check()

        def watch():
            while not self.stop.wait(1):
                try:
                    self.check(storage=False)
                except (ResourceLimit, OSError) as exc:
                    self.failure = str(exc)
                    return

        self.thread = threading.Thread(target=watch, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        if self.thread:
            self.thread.join(timeout=2)
