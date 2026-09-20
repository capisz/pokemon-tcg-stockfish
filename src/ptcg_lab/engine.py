from __future__ import annotations

import contextlib
import hashlib
import json
import os
import queue
import subprocess
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Iterator


class EngineError(RuntimeError):
    pass


class EngineClient:
    """One persistent, serial JSON-lines worker, with bounded reply waits.

    A reader thread handles large replay lines without partial-line timeout bugs.
    Never share a worker between simultaneous games. EnginePool leases exclusively.
    """

    def __init__(self, root: Path, timeout: float = 300, command: list[str] | None = None):
        self.root, self.timeout = root, timeout
        self.command = command or ["node", "--max-old-space-size=1024", str(root / "packages/engine/dist/worker.cjs")]
        self.process: subprocess.Popen | None = None
        self.responses: queue.Queue = queue.Queue()
        self.diagnostics: deque[str] = deque(maxlen=20)
        self.lock = threading.Lock()
        self.build_hash: str | None = None

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        self.responses = queue.Queue()
        bundle = self.root / "packages/engine/dist/worker.cjs"
        self.build_hash = hashlib.sha256(bundle.read_bytes()).hexdigest() if bundle.exists() else None
        environment = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2"}
        self.process = subprocess.Popen(self.command, cwd=self.root, env=environment, stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True, encoding="utf-8", bufsize=1)
        responses, process = self.responses, self.process

        def read_stdout() -> None:
            try:
                for line in process.stdout:
                    if len(line) > 128 * 1024**2:
                        responses.put(EngineError("Engine reply exceeded 128 MiB"))
                        return
                    responses.put(line)
            finally:
                responses.put(EngineError("Engine worker exited before replying"))

        def read_stderr() -> None:
            for line in process.stderr:
                self.diagnostics.append(line.rstrip()[:2000])

        threading.Thread(target=read_stdout, daemon=True).start()
        threading.Thread(target=read_stderr, daemon=True).start()

    def request(self, method: str, params: dict | None = None) -> Any:
        with self.lock:
            self.start()
            request_id = uuid.uuid4().hex
            try:
                self.process.stdin.write(json.dumps({"id": request_id, "method": method, "params": params or {}}) + "\n")
                self.process.stdin.flush()
                raw = self.responses.get(timeout=self.timeout)
                if isinstance(raw, Exception):
                    raise raw
                response = json.loads(raw)
                if response.get("id") != request_id:
                    raise EngineError("Engine protocol request id mismatch")
                if "error" in response:
                    raise EngineError(str(response["error"].get("message", response["error"])))
                result = response["result"]
                if isinstance(result, dict) and method in {"health", "run", "replay"} and self.build_hash:
                    result["engineBuildHash"] = self.build_hash
                return result
            except queue.Empty as exc:
                self.close()
                raise EngineError(f"Worker exceeded {self.timeout:g}s; run is interrupted, not a draw") from exc
            except (ValueError, KeyError, BrokenPipeError, OSError) as exc:
                self.close()
                raise EngineError(f"Engine protocol failure: {exc}") from exc

    def close(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=3)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream:
                stream.close()

    def __enter__(self) -> "EngineClient":
        self.start()
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class EnginePool:
    def __init__(self, root: Path, size: int = 2, timeout: float = 300):
        if not 1 <= size <= 8:
            raise ValueError("Local configuration supports one to eight simulation workers; benchmark before raising the default of two")
        self.clients = [EngineClient(root, timeout) for _ in range(size)]
        self.available: queue.Queue[EngineClient] = queue.Queue()
        for client in self.clients:
            self.available.put(client)

    def acquire(self, wait_timeout: float = 310) -> EngineClient:
        try:
            return self.available.get(timeout=wait_timeout)
        except queue.Empty as exc:
            raise EngineError("All local simulation workers are busy") from exc

    def release(self, client: EngineClient) -> None:
        self.available.put(client)

    @contextlib.contextmanager
    def lease(self, wait_timeout: float = 310) -> Iterator[EngineClient]:
        client = self.acquire(wait_timeout)
        try:
            yield client
        except EngineError:
            client.close()
            raise
        finally:
            self.release(client)

    def close(self) -> None:
        for client in self.clients:
            client.close()
