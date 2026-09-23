from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_manifest
from .candidate_support import audit_macro_candidate_support
from .dataset_v1 import build_dataset, build_macro_position_pool
from .experiment import (collect_macro_labels, evaluate_candidate, fit_ranker,
                         runtime_identity, train_candidate)
from .fresh_collection import collect_fresh_positions
from .model import StrategyTransformerV1
from .supervisor import MindSupervisor


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Research-only Autonomous Learning Mind v1")
    sub = parser.add_subparsers(dest="command", required=True)
    audit = sub.add_parser("audit-baseline", help="verify and encode actor-private frozen baseline replays")
    audit.add_argument("--manifest", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--limit", type=int)
    sub.add_parser("model-info")
    initialize = sub.add_parser("initialize-supervisor")
    initialize.add_argument("--state-root", type=Path, required=True)
    initialize.add_argument("--reserve-gb", type=int, default=25)
    initialize.add_argument("--data-cap-gb", type=int, default=100)
    freeze = sub.add_parser("build-supervised-dataset")
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--review-root", type=Path, required=True)
    freeze.add_argument("--experimental-root", type=Path, required=True)
    freeze.add_argument("--source-dataset-manifest", type=Path, required=True)
    pool = sub.add_parser("build-macro-position-pool")
    pool.add_argument("--root", type=Path, required=True)
    pool.add_argument("--output", type=Path, required=True)
    pool.add_argument("--experimental-root", type=Path, required=True)
    pool.add_argument("--source-dataset-manifest", type=Path, required=True)
    pool.add_argument("--target-deck", default="raging-bolt",
                      help="one archetype deck name or 'all' for a generalist pool")
    pool.add_argument("--limit", type=int, default=18)
    support = sub.add_parser("audit-macro-candidate-support",
                             help="audit actor-visible candidate support without rollouts or labels")
    support.add_argument("--root", type=Path, required=True)
    support.add_argument("--dataset", type=Path, required=True)
    support.add_argument("--output", type=Path, required=True)
    support.add_argument("--workers", type=int, default=1)
    support.add_argument("--limit", type=int)
    support.add_argument("--split", choices=("train", "development", "heldout"),
                         help="restrict the audit to one frozen game-disjoint split")
    support.add_argument("--position-hash", action="append", default=[], metavar="HASH",
                         help="audit an exact position from the frozen dataset; may be repeated")
    collect = sub.add_parser("collect-macro-labels")
    collect.add_argument("--root", type=Path, required=True)
    collect.add_argument("--dataset", type=Path, required=True)
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--limit", type=int, default=20)
    collect.add_argument("--initial", type=int, default=16)
    collect.add_argument("--maximum", type=int, default=64)
    collect.add_argument("--extension-batch-size", type=int, default=8)
    collect.add_argument("--horizon", type=int, default=16)
    collect.add_argument("--rollout-budget-ms", type=int, default=1000,
                         help="per-candidate search time cap; cutoffs are recorded as truncated")
    collect.add_argument("--rollout-workers", type=int, default=1,
                         help="parallel engine workers (1-8); use 1 for the serial reference run")
    collect.add_argument("--position-hash", action="append", default=[], metavar="HASH",
                         help="collect an exact position from the frozen dataset; may be repeated")
    collect.add_argument("--split", choices=("train", "development", "heldout"),
                         help="restrict collection to one frozen game-disjoint split")
    fresh = sub.add_parser("collect-fresh-positions", help="collect small resumable current-engine research games")
    fresh.add_argument("--root", type=Path, required=True)
    fresh.add_argument("--output", type=Path, required=True,
                       help="ignored local artifact directory; never point at data/competitive")
    fresh.add_argument("--games-per-matchup", type=int, default=2)
    fresh.add_argument("--policy", choices=("typescript-heuristic", "python-heuristic"),
                       default="typescript-heuristic")
    fresh.add_argument("--collection-namespace", default="main",
                       help="seed/replay-ID epoch slug; default main preserves the original deterministic schedule")
    fresh.add_argument("--max-decisions", type=int, default=1200)
    rank = sub.add_parser("fit-macro-ranker")
    rank.add_argument("--labels", type=Path, required=True)
    rank.add_argument("--output", type=Path, required=True)
    rank.add_argument("--teacher-hash", required=True)
    rank.add_argument("--opponent-policy-hash", required=True)
    rank.add_argument("--iteration", type=int, default=1)
    train = sub.add_parser("train-supervised")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=1)
    evaluate = sub.add_parser("evaluate-supervised")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "audit-baseline":
        result = audit_manifest(args.manifest, limit=args.limit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    elif args.command == "model-info":
        model = StrategyTransformerV1()
        result = {"model": "StrategyTransformerV1", "parameters": model.parameter_count(),
                  "policyInterface": "policy_forward", "evaluationInterface": "evaluation_forward"}
    elif args.command == "initialize-supervisor":
        supervisor = MindSupervisor(args.state_root, reserve_bytes=args.reserve_gb * 1024**3,
                                    data_cap_bytes=args.data_cap_gb * 1024**3)
        supervisor.persist(); result = supervisor.state.__dict__
    elif args.command == "build-supervised-dataset":
        root = args.root.resolve(); identity = runtime_identity(root)
        result = build_dataset(root=root, output=args.output.resolve(), review_root=args.review_root.resolve(),
                               experimental_root=args.experimental_root.resolve(),
                               source_dataset_manifest=args.source_dataset_manifest.resolve(), identity=identity)
    elif args.command == "collect-macro-labels":
        root = args.root.resolve(); identity = runtime_identity(root).record()
        result = collect_macro_labels(root=root, dataset_dir=args.dataset.resolve(), output=args.output.resolve(),
                                      identity=identity, limit=args.limit, initial=args.initial, maximum=args.maximum,
                                      extension_batch_size=args.extension_batch_size,
                                      horizon=args.horizon, rollout_budget_ms=args.rollout_budget_ms,
                                      rollout_workers=args.rollout_workers,
                                      position_hashes=args.position_hash, split=args.split)
    elif args.command == "collect-fresh-positions":
        result = collect_fresh_positions(root=args.root, output=args.output,
            games_per_matchup=args.games_per_matchup, policy=args.policy,
            max_decisions=args.max_decisions, collection_namespace=args.collection_namespace)
    elif args.command == "build-macro-position-pool":
        root = args.root.resolve(); identity = runtime_identity(root)
        result = build_macro_position_pool(output=args.output.resolve(),
            experimental_root=args.experimental_root.resolve(),
            source_dataset_manifest=args.source_dataset_manifest.resolve(), identity=identity,
            target_deck=args.target_deck, limit=args.limit)
    elif args.command == "audit-macro-candidate-support":
        root = args.root.resolve(); identity = runtime_identity(root).record()
        result = audit_macro_candidate_support(root=root, dataset_dir=args.dataset.resolve(),
            output=args.output.resolve(), identity=identity, workers=args.workers,
            limit=args.limit, split=args.split, position_hashes=args.position_hash)
    elif args.command == "fit-macro-ranker":
        result = fit_ranker(args.labels.resolve(), args.output.resolve(), teacher_hash=args.teacher_hash,
                            opponent_policy_hash=args.opponent_policy_hash, iteration=args.iteration)
    elif args.command == "train-supervised":
        result = train_candidate(args.dataset.resolve(), args.output.resolve(), epochs=args.epochs)
    else:
        result = evaluate_candidate(args.dataset.resolve(), args.checkpoint.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
