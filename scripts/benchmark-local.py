"""Bounded local worker/model memory and throughput probe, not a strength test."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
import subprocess
import time
from ptcg_lab.config import Settings
from ptcg_lab.engine import EngineClient
from ptcg_lab.model import PolicyResourceModel
import torch

torch.set_num_threads(2)
model = PolicyResourceModel()
settings = Settings.from_env()
workers = [EngineClient(settings.root) for _ in range(2)]
try:
    for worker in workers:
        worker.start()
    ids = [os.getpid()] + [worker.process.pid for worker in workers]
    peaks = {pid: 0 for pid in ids}
    total_peak = 0
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker.request, "run", {"seed": 41 + i, "decks": ["crustle", "mega-lucario"], "policy": "heuristic", "maxDecisions": 500}) for i, worker in enumerate(workers)]
        while True:
            output = subprocess.check_output(["ps", "-o", "pid=,rss=", "-p", ",".join(map(str, ids))], text=True)
            usage = {int(pid): int(rss) * 1024 for pid, rss in (line.split() for line in output.splitlines())}
            total_peak = max(total_peak, sum(usage.values()))
            for pid, value in usage.items():
                peaks[pid] = max(peaks[pid], value)
            if all(future.done() for future in futures):
                break
            time.sleep(.2)
        replays = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    decisions = sum(len(replay["frames"]) - 1 for replay in replays)
    print(json.dumps({"engine": workers[0].request("health"), "workers": 2, "modelParameters": sum(p.numel() for p in model.parameters()), "elapsedSeconds": elapsed, "decisions": decisions, "decisionsPerSecond": decisions / elapsed, "statuses": [r["status"] for r in replays], "sampledPeakTotalRssBytes": total_peak, "pythonPeakRssBytes": peaks[ids[0]], "workerPeakRssBytes": [peaks[pid] for pid in ids[1:]], "limits": "RSS sampled every200ms for two games, including resident PyTorch/model; excludes browser, API, OS, and Ollama. Not a worst-case bound."}, indent=2))
finally:
    for worker in workers:
        worker.close()
