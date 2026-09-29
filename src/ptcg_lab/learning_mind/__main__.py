from __future__ import annotations

import argparse
import json
from pathlib import Path

from .candidate_safety import audit_candidate_safety
from .audit import audit_manifest
from .aggregation import combine_macro_label_runs
from .candidate_support import audit_macro_candidate_support
from .dataset_v1 import (build_dataset, build_macro_position_pool,
                         build_strategy_probe_dataset, load_dataset, training_records)
from .disagreement_review import (audit_disagreement_review,
    build_disagreement_review_packet, write_disagreement_review_template)
from .experiment import (collect_macro_labels, fit_ranker,
                         runtime_identity, train_candidate)
from .policy_evaluation import evaluate_candidate
from .fresh_collection import collect_fresh_positions
from .model import StrategyTransformerV1
from .macro_fidelity import audit_raging_bolt_macro_fidelity
from .ranker_v2 import fit_macro_ranker_v2, verify_macro_ranker_v2_artifact
from .ranker_distillation import build_macro_ranker_distillation
from .selection import freeze_macro_label_selection
from .supervisor import MindSupervisor
from .supervised_evidence import audit_supervised_evaluation
from .stage_evidence import verify_ppo_stage_evidence
from .specialist_evidence import verify_specialist_curriculum
from .value_targets import (build_value_target_dataset, load_value_target_dataset,
                            training_records as value_training_records)
from .training import train_supervised
from .value_evaluation import evaluate_value_head


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
    initialize.add_argument("--artifact-root", type=Path, action="append", default=[],
                            help="additional local directory included in the frozen data-cap accounting")
    initialize.add_argument("--reserve-gb", type=int, default=25)
    initialize.add_argument("--data-cap-gb", type=int, default=100)
    freeze = sub.add_parser("build-supervised-dataset")
    freeze.add_argument("--root", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument("--review-root", type=Path, required=True)
    freeze.add_argument("--experimental-root", type=Path, required=True)
    freeze.add_argument("--source-dataset-manifest", type=Path, required=True)
    freeze.add_argument("--ranker-distillation-dir", type=Path,
                        help="optional verified train-only macro-ranker distribution dataset")
    value_targets = sub.add_parser("build-value-target-dataset",
        help="freeze actor-visible terminal outcomes as value-only labels from eligible completed games")
    value_targets.add_argument("--root", type=Path, required=True)
    value_targets.add_argument("--experimental-root", type=Path, required=True)
    value_targets.add_argument("--source-dataset-manifest", type=Path, required=True)
    value_targets.add_argument("--output", type=Path, required=True)
    distill = sub.add_parser("build-macro-ranker-distillation",
        help="freeze train-only legal-action distributions from a verified macro ranker")
    distill.add_argument("--root", type=Path, required=True)
    distill.add_argument("--macro-position-pool", type=Path, required=True)
    distill.add_argument("--labels", type=Path, required=True)
    distill.add_argument("--selection", type=Path, required=True)
    distill.add_argument("--model", type=Path, required=True)
    distill.add_argument("--report", type=Path, required=True)
    distill.add_argument("--confidence-audit", type=Path, required=True)
    distill.add_argument("--output", type=Path, required=True)
    distill.add_argument("--temperature", type=float, default=1.0)
    pool = sub.add_parser("build-macro-position-pool")
    pool.add_argument("--root", type=Path, required=True)
    pool.add_argument("--output", type=Path, required=True)
    pool.add_argument("--experimental-root", type=Path, required=True)
    pool.add_argument("--source-dataset-manifest", type=Path, required=True)
    pool.add_argument("--target-deck", default="raging-bolt",
                      help="one archetype deck name or 'all' for a generalist pool")
    pool.add_argument("--limit", type=int, default=18)
    probes = sub.add_parser("build-strategy-probe-dataset",
                            help="freeze every actor-visible decision from held-out source games")
    probes.add_argument("--root", type=Path, required=True)
    probes.add_argument("--experimental-root", type=Path, required=True)
    probes.add_argument("--source-dataset-manifest", type=Path, required=True)
    probes.add_argument("--output", type=Path, required=True)
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
    selection = sub.add_parser("freeze-macro-label-selection",
                               help="freeze supported train/development positions by source game")
    selection.add_argument("--python-dataset", type=Path, required=True)
    selection.add_argument("--typescript-dataset", type=Path, required=True)
    selection.add_argument("--python-support", type=Path, required=True)
    selection.add_argument("--typescript-support", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)
    selection.add_argument("--pinned-train-hash", action="append", default=[], metavar="FAMILY:HASH",
                           help="retain a prior train label position without recollecting it")
    combine = sub.add_parser("combine-macro-label-runs",
                             help="verify and combine separate frozen macro-label runs for ranker evaluation")
    combine.add_argument("--input-dir", type=Path, action="append", required=True,
                         help="frozen run directory; repeat once per policy-family run")
    combine.add_argument("--selection", type=Path, required=True,
                         help="frozen train/development selection manifest that every run must cover exactly")
    combine.add_argument("--output", type=Path, required=True)
    fidelity = sub.add_parser("audit-raging-bolt-macro-fidelity",
        help="audit finalized Raging Bolt plan executions from frozen labels without running games")
    fidelity.add_argument("--root", type=Path, required=True)
    fidelity.add_argument("--selection", type=Path, required=True)
    fidelity.add_argument("--python-dataset", type=Path, required=True)
    fidelity.add_argument("--typescript-dataset", type=Path, required=True)
    fidelity.add_argument("--python-labels", type=Path, required=True)
    fidelity.add_argument("--typescript-labels", type=Path, required=True)
    fidelity.add_argument("--output", type=Path, required=True)
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
    rank.add_argument("--selection", type=Path, required=True,
                      help="same frozen selection manifest used to verify complete macro-label coverage")
    rank.add_argument("--output", type=Path, required=True)
    rank.add_argument("--teacher-hash", required=True)
    rank.add_argument("--opponent-policy-hash", required=True)
    rank.add_argument("--iteration", type=int, default=1)
    rank_v2 = sub.add_parser("fit-macro-ranker-v2",
        help="fit the actor-visible state/plan-feature ranker with a separately versioned model contract")
    rank_v2.add_argument("--labels", type=Path, required=True)
    rank_v2.add_argument("--selection", type=Path, required=True)
    rank_v2.add_argument("--confidence-audit", type=Path, required=True,
                         help="verified final-label confidence report for these exact labels and selection")
    rank_v2.add_argument("--output", type=Path, required=True)
    rank_v2.add_argument("--teacher-hash", required=True)
    rank_v2.add_argument("--opponent-policy-hash", required=True)
    rank_v2.add_argument("--iteration", type=int, default=1)
    verify_ranker_v2 = sub.add_parser("verify-macro-ranker-v2",
        help="verify v2 report/schema and model checksum without loading native XGBoost code")
    verify_ranker_v2.add_argument("--model", type=Path, required=True)
    verify_ranker_v2.add_argument("--report", type=Path, required=True)
    train = sub.add_parser("train-supervised")
    train.add_argument("--dataset", type=Path, required=True)
    train.add_argument("--value-dataset", type=Path,
                       help="optional immutable terminal-outcome dataset for value-only examples")
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--epochs", type=int, default=1)
    evaluate_value = sub.add_parser("evaluate-value-head",
        help="score a frozen value head on development or held-out terminal outcomes")
    evaluate_value.add_argument("--dataset", type=Path, required=True)
    evaluate_value.add_argument("--checkpoint", type=Path, required=True)
    evaluate_value.add_argument("--output", type=Path, required=True)
    evaluate_value.add_argument("--split", choices=("development", "heldout"), default="heldout")
    evaluate = sub.add_parser("evaluate-supervised")
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--probe-dataset", type=Path, required=True,
                          help="complete held-out actor-view corpus; sparse supervised rows are not accepted")
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    audit_supervised = sub.add_parser("audit-supervised-evaluation",
        help="verify frozen held-out predictions and require independent paired evidence")
    audit_supervised.add_argument("--dataset", type=Path, required=True)
    audit_supervised.add_argument("--checkpoint", type=Path, required=True)
    audit_supervised.add_argument("--evaluation", type=Path, required=True)
    audit_supervised.add_argument("--output", type=Path, required=True)
    audit_supervised.add_argument("--minimum-game-sides", type=int, default=20)
    safety = sub.add_parser("audit-candidate-safety",
        help="verify legal-option coverage, cap bounds, and one-action decoding")
    safety.add_argument("--dataset", type=Path, required=True)
    safety.add_argument("--checkpoint", type=Path, required=True)
    safety.add_argument("--evaluation", type=Path, required=True)
    safety.add_argument("--audit", type=Path, required=True)
    safety.add_argument("--output", type=Path, required=True)
    review_packet = sub.add_parser("build-disagreement-review",
        help="build a hash-bound actor-view-only human review packet")
    review_packet.add_argument("--dataset", type=Path, required=True)
    review_packet.add_argument("--checkpoint", type=Path, required=True)
    review_packet.add_argument("--evaluation", type=Path, required=True)
    review_packet.add_argument("--audit", type=Path, required=True)
    review_packet.add_argument("--output", type=Path, required=True)
    review_template = sub.add_parser("make-disagreement-review-form",
        help="make a human-fillable review form bound to a frozen packet")
    review_template.add_argument("--packet", type=Path, required=True)
    review_template.add_argument("--output", type=Path, required=True)
    review_audit = sub.add_parser("audit-disagreement-review",
        help="verify reviewer identity and complete acceptable findings for every disagreement")
    review_audit.add_argument("--packet", type=Path, required=True)
    review_audit.add_argument("--review", type=Path, required=True)
    review_audit.add_argument("--output", type=Path, required=True)
    stage = sub.add_parser("verify-ppo-stage",
        help="recompute frozen PPO prerequisites and emit a hash-bound evidence report")
    for option in ("root", "baseline-manifest", "dataset", "probe-dataset", "checkpoint",
                   "evaluation", "supervised-audit", "safety-report", "macro-selection",
                   "python-dataset", "typescript-dataset", "python-labels", "typescript-labels",
                   "macro-fidelity", "ranker-labels", "confidence-audit",
                   "ranker-model", "ranker-report",
                   "disagreement-packet", "disagreement-review",
                   "disagreement-receipt", "output"):
        stage.add_argument(f"--{option}", type=Path, required=True)
    stage.add_argument("--human-enable-ppo", action="store_true",
                       help="explicit one-time authorization; prerequisites must still pass")
    specialists = sub.add_parser("verify-specialist-curriculum",
        help="reverify exact-deck specialist promotion evidence and write an immutable report")
    specialists.add_argument("--root", type=Path, required=True,
        help="repository root containing the five approved deck manifests")
    specialists.add_argument("--registry", type=Path, required=True,
        help="frozen specialist registry with checkpoint hashes and promotion receipts")
    specialists.add_argument("--output", type=Path, required=True,
        help="new immutable verification-report path")
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
                                    data_cap_bytes=args.data_cap_gb * 1024**3,
                                    artifact_roots=args.artifact_root)
        supervisor.persist(); result = supervisor.state.__dict__
    elif args.command == "build-supervised-dataset":
        root = args.root.resolve(); identity = runtime_identity(root)
        result = build_dataset(root=root, output=args.output.resolve(), review_root=args.review_root.resolve(),
                               experimental_root=args.experimental_root.resolve(),
                               source_dataset_manifest=args.source_dataset_manifest.resolve(), identity=identity,
                               ranker_distillation_dir=(args.ranker_distillation_dir.resolve()
                                   if args.ranker_distillation_dir else None))
    elif args.command == "build-value-target-dataset":
        root = args.root.resolve(); identity = runtime_identity(root).record()
        result = build_value_target_dataset(output=args.output.resolve(),
            experimental_root=args.experimental_root.resolve(),
            source_manifest_path=args.source_dataset_manifest.resolve(), identity=identity)
    elif args.command == "build-macro-ranker-distillation":
        root = args.root.resolve(); identity = runtime_identity(root).record()
        result = build_macro_ranker_distillation(output=args.output.resolve(),
            macro_position_pool=args.macro_position_pool.resolve(), labels_dir=args.labels.resolve(),
            selection_path=args.selection.resolve(), model_path=args.model.resolve(),
            report_path=args.report.resolve(), confidence_audit_path=args.confidence_audit.resolve(),
            identity=identity, temperature=args.temperature)
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
    elif args.command == "build-strategy-probe-dataset":
        root = args.root.resolve(); identity = runtime_identity(root)
        result = build_strategy_probe_dataset(output=args.output.resolve(),
            experimental_root=args.experimental_root.resolve(),
            source_dataset_manifest=args.source_dataset_manifest.resolve(), identity=identity)
    elif args.command == "audit-macro-candidate-support":
        root = args.root.resolve(); identity = runtime_identity(root).record()
        result = audit_macro_candidate_support(root=root, dataset_dir=args.dataset.resolve(),
            output=args.output.resolve(), identity=identity, workers=args.workers,
            limit=args.limit, split=args.split, position_hashes=args.position_hash)
    elif args.command == "freeze-macro-label-selection":
        root = Path.cwd().resolve(); identity = runtime_identity(root).record()
        pinned = {"python-heuristic": [], "typescript-heuristic": []}
        for value in args.pinned_train_hash:
            family, separator, position_hash = value.partition(":")
            if not separator or family not in pinned or not position_hash:
                parser.error("--pinned-train-hash must be FAMILY:HASH for python-heuristic or typescript-heuristic")
            pinned[family].append(position_hash)
        result = freeze_macro_label_selection(
            datasets={"python-heuristic": args.python_dataset, "typescript-heuristic": args.typescript_dataset},
            support_reports={"python-heuristic": args.python_support, "typescript-heuristic": args.typescript_support},
            output=args.output, identity=identity, pinned_train_hashes=pinned)
    elif args.command == "combine-macro-label-runs":
        root = Path.cwd().resolve(); identity = runtime_identity(root).record()
        result = combine_macro_label_runs(inputs=args.input_dir, output=args.output, identity=identity,
                                          selection_path=args.selection.resolve())
    elif args.command == "audit-raging-bolt-macro-fidelity":
        result = audit_raging_bolt_macro_fidelity(root=args.root.resolve(),
            selection_path=args.selection.resolve(), python_dataset=args.python_dataset.resolve(),
            typescript_dataset=args.typescript_dataset.resolve(), python_labels=args.python_labels.resolve(),
            typescript_labels=args.typescript_labels.resolve(), output=args.output.resolve())
    elif args.command == "fit-macro-ranker":
        result = fit_ranker(args.labels.resolve(), args.output.resolve(), selection_path=args.selection.resolve(),
                            teacher_hash=args.teacher_hash,
                            opponent_policy_hash=args.opponent_policy_hash, iteration=args.iteration)
    elif args.command == "fit-macro-ranker-v2":
        result = fit_macro_ranker_v2(args.labels.resolve(), args.output.resolve(),
            selection_path=args.selection.resolve(), confidence_audit_path=args.confidence_audit.resolve(),
            teacher_hash=args.teacher_hash,
            opponent_policy_hash=args.opponent_policy_hash, iteration=args.iteration)
    elif args.command == "verify-macro-ranker-v2":
        verified = verify_macro_ranker_v2_artifact(args.model, args.report)
        report = verified["report"]
        result = {"verified": True, "modelSha256": report["modelSha256"],
            "featureSchemaHash": report["featureSchemaHash"],
            "featureImplementationSha256": report["featureImplementationSha256"],
            "inferenceImplementationSha256": report["inferenceImplementationSha256"]}
    elif args.command == "train-supervised":
        if args.value_dataset is None:
            result = train_candidate(args.dataset.resolve(), args.output.resolve(), epochs=args.epochs)
        else:
            policy_manifest, policy_rows = load_dataset(args.dataset.resolve())
            value_manifest, value_rows = load_value_target_dataset(args.value_dataset.resolve(),
                identity=policy_manifest["identity"])
            records = training_records(policy_rows, "train") + value_training_records(value_rows, "train")
            result = train_supervised(records, args.output.resolve(), policy_manifest["identity"],
                epochs=args.epochs, dataset_manifest_hash=policy_manifest["manifestHash"],
                value_dataset_manifest_hash=value_manifest["manifestHash"])
            result = {**result, "datasetManifestHash": policy_manifest["manifestHash"],
                      "valueDatasetManifestHash": value_manifest["manifestHash"]}
    elif args.command == "evaluate-value-head":
        result = evaluate_value_head(dataset_dir=args.dataset.resolve(),
            checkpoint_path=args.checkpoint.resolve(), output=args.output.resolve(), split=args.split)
    elif args.command == "evaluate-supervised":
        result = evaluate_candidate(args.dataset.resolve(), args.checkpoint.resolve(),
                                    args.probe_dataset.resolve())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    elif args.command == "audit-supervised-evaluation":
        result = audit_supervised_evaluation(dataset_dir=args.dataset.resolve(),
            checkpoint=args.checkpoint.resolve(), evaluation_path=args.evaluation.resolve(),
            output=args.output.resolve(), minimum_game_sides=args.minimum_game_sides)
    elif args.command == "audit-candidate-safety":
        result = audit_candidate_safety(dataset_dir=args.dataset.resolve(), checkpoint=args.checkpoint.resolve(),
            evaluation_path=args.evaluation.resolve(), audit_path=args.audit.resolve(),
            output=args.output.resolve())
    elif args.command == "build-disagreement-review":
        result = build_disagreement_review_packet(dataset_dir=args.dataset.resolve(),
            checkpoint=args.checkpoint.resolve(), evaluation_path=args.evaluation.resolve(),
            audit_path=args.audit.resolve(), output=args.output.resolve())
    elif args.command == "make-disagreement-review-form":
        result = write_disagreement_review_template(packet_path=args.packet.resolve(), output=args.output.resolve())
    elif args.command == "audit-disagreement-review":
        result = audit_disagreement_review(packet_path=args.packet.resolve(),
            review_path=args.review.resolve(), output=args.output.resolve())
    elif args.command == "verify-ppo-stage":
        result, _capability = verify_ppo_stage_evidence(root=args.root,
            baseline_manifest=args.baseline_manifest, dataset_dir=args.dataset,
            probe_dataset_dir=args.probe_dataset, checkpoint=args.checkpoint,
            evaluation_path=args.evaluation, supervised_audit_path=args.supervised_audit,
            safety_report_path=args.safety_report, macro_selection_path=args.macro_selection,
            python_dataset=args.python_dataset, typescript_dataset=args.typescript_dataset,
            python_labels=args.python_labels, typescript_labels=args.typescript_labels,
            macro_fidelity_path=args.macro_fidelity,
            ranker_labels_dir=args.ranker_labels, confidence_audit_path=args.confidence_audit,
            ranker_model_path=args.ranker_model, ranker_report_path=args.ranker_report,
            disagreement_packet_path=args.disagreement_packet,
            disagreement_review_path=args.disagreement_review,
            disagreement_receipt_path=args.disagreement_receipt, output=args.output,
            human_enable_ppo=args.human_enable_ppo)
    elif args.command == "verify-specialist-curriculum":
        report, _capability = verify_specialist_curriculum(root=args.root.resolve(),
            registry_path=args.registry.resolve(), output=args.output.resolve())
        result = {"verified": report["status"] == "passed",
            "reportPath": str(args.output.resolve()), "reportHash": report["reportHash"],
            "specialistCount": report["specialistCount"], "automaticPromotion": False,
            "runtimeNote": "A saved report is not a routing capability; runtime must reverify sources."}
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
