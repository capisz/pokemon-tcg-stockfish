from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

from .config import Settings
from .engine import EngineClient
from .storage import Store


def doctor(settings: Settings, benchmark: bool = False) -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform(), "root": str(settings.root),
              "data": str(settings.data), "engineBuilt": settings.worker.exists(),
              "optional": {name: importlib.util.find_spec(name) is not None for name in ("torch", "sklearn", "pyarrow", "mlflow", "sentence_transformers")},
              "limits": {"simulationWorkers": settings.workers, "maxDecisions": settings.max_decisions,
                         "maxDiskBytes": settings.max_disk_bytes}}
    if settings.worker.exists():
        try:
            with EngineClient(settings.root, timeout=20) as engine:
                result["engine"] = engine.request("health")
        except Exception as exc:
            result["engineError"] = str(exc)
    if result["optional"]["torch"]:
        import torch
        result["torchVersion"] = torch.__version__
        result["mpsAvailable"] = torch.backends.mps.is_available()
        if benchmark:
            from .model import PolicyResourceModel
            from .features import FEATURE_NAMES, MAX_VISIBLE_CARDS, ACTION_DIM
            measurements = {}
            torch.set_num_threads(2)
            for device in (["cpu", "mps"] if result["mpsAvailable"] else ["cpu"]):
                model = PolicyResourceModel().to(device)
                resources = torch.randn(32, len(FEATURE_NAMES), device=device)
                cards = torch.ones((32, MAX_VISIBLE_CARDS), dtype=torch.long, device=device)
                actions = torch.randn(32, 16, ACTION_DIM, device=device)
                for _ in range(3):
                    model(resources, cards, actions)["logit"].sum().backward()
                    model.zero_grad()
                if device == "mps":
                    torch.mps.synchronize()
                started = time.perf_counter()
                for _ in range(20):
                    model(resources, cards, actions)["logit"].sum().backward()
                    model.zero_grad()
                if device == "mps":
                    torch.mps.synchronize()
                measurements[device] = {"stepsPerSecond": 20 / (time.perf_counter() - started),
                                        "parameters": sum(parameter.numel() for parameter in model.parameters())}
            result["deviceBenchmark"] = measurements
            result["recommendedDevice"] = max(measurements, key=lambda device: measurements[device]["stepsPerSecond"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local Pokémon TCG research. All storage stays on this machine.")
    parser.add_argument("--data", type=Path, help="Override local data directory")
    commands = parser.add_subparsers(dest="command", required=True)
    diagnostic = commands.add_parser("doctor", help="Check local engine/dependencies")
    diagnostic.add_argument("--benchmark", action="store_true")
    serve = commands.add_parser("serve", help="Serve replay API on loopback")
    serve.add_argument("--port", type=int, default=8765)
    play = commands.add_parser("selfplay", help="Generate bounded self-play with resumable run manifests")
    play.add_argument("--games", type=int, default=20)
    play.add_argument("--seed", type=int, default=42)
    play.add_argument("--max-decisions", type=int, default=1000)
    play.add_argument("--policy", default="heuristic", help="random, heuristic, or a local model checkpoint path")
    play.add_argument("--resume", help="Run id; repeat the original configuration arguments")
    play.add_argument("--stop-after", type=int, help="Pause after this many new completed/truncated games")
    training = commands.add_parser("train", help="Train only from completed games with whole-game splits")
    training.add_argument("--epochs", type=int, default=3)
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    training.add_argument("--resume", type=Path)
    training.add_argument("--max-positions", type=int, default=10000)
    training.add_argument("--mlflow", action="store_true")
    evaluation = commands.add_parser("evaluate", help="All 25 candidate/opponent deck assignments with paired seeds and swapped seats")
    evaluation.add_argument("--candidate", default="heuristic")
    evaluation.add_argument("--opponent", default="random")
    evaluation.add_argument("--seeds", type=int, default=2, help="Seeds per matchup; each plays both seats")
    evaluation.add_argument("--seed-start", type=int, default=1_000_000_000)
    evaluation.add_argument("--max-decisions", type=int, default=1000)
    evaluation.add_argument("--promote", action="store_true", help="Install learned champion only if all statistical gates pass")
    guide = commands.add_parser("import-guide", help="Import attributed UTF-8 text or Markdown")
    guide.add_argument("path", type=Path)
    for name in ("title", "author", "source", "matchup"):
        guide.add_argument(f"--{name}", required=True)
    guide.add_argument("--format-date", default="2026-09-17")
    retrieval = commands.add_parser("retrieve", help="Retrieve cited passages from local guides")
    retrieval.add_argument("query")
    retrieval.add_argument("--embedding-model", type=Path)
    notes = commands.add_parser("draft-notes", help="Ask local Ollama for cited, explicitly unreviewed strategy notes")
    notes.add_argument("query")
    notes.add_argument("--model", default="qwen3:4b")
    export = commands.add_parser("export-parquet", help="Export completed, game-split training trajectories")
    export.add_argument("path", type=Path)
    arguments = parser.parse_args(argv)
    settings = Settings.from_env()
    if arguments.data:
        from dataclasses import replace
        settings = replace(settings, data=arguments.data.resolve())
    store = Store(settings.data, settings.max_disk_bytes)
    try:
        if arguments.command == "doctor":
            result = doctor(settings, arguments.benchmark)
        elif arguments.command == "serve":
            import uvicorn
            from .api import create_app
            uvicorn.run(create_app(settings), host="127.0.0.1", port=arguments.port)
            return 0
        elif arguments.command == "selfplay":
            from .selfplay import selfplay
            result = selfplay(settings, games=arguments.games, seed=arguments.seed, max_decisions=arguments.max_decisions,
                              policy=arguments.policy, resume=arguments.resume, stop_after=arguments.stop_after)
        elif arguments.command == "train":
            from .training import train
            result = train(store, epochs=arguments.epochs, seed=arguments.seed, device=arguments.device,
                           resume=arguments.resume, max_positions=arguments.max_positions, mlflow=arguments.mlflow)
        elif arguments.command == "evaluate":
            from .evaluation import evaluate
            result = evaluate(settings, candidate=arguments.candidate, opponent=arguments.opponent, seeds=arguments.seeds,
                              seed_start=arguments.seed_start, max_decisions=arguments.max_decisions, promote=arguments.promote)
        elif arguments.command == "import-guide":
            from .guides import import_guide
            result = import_guide(store, arguments.path, title=arguments.title, author=arguments.author,
                                  source=arguments.source, matchup=arguments.matchup, format_date=arguments.format_date)
        elif arguments.command == "retrieve":
            from .guides import retrieve
            result = retrieve(store, arguments.query, embedding_model=arguments.embedding_model)
        elif arguments.command == "draft-notes":
            from .guides import draft_notes
            result = draft_notes(store, arguments.query, model=arguments.model)
        elif arguments.command == "export-parquet":
            from .dataset import export_parquet
            result = export_parquet(store, arguments.path)
        print(json.dumps(result, indent=2, allow_nan=False))
        return 0
    except (RuntimeError, ValueError, FileNotFoundError, ImportError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
