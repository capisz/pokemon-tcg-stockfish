from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audit import audit_manifest
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
    args = parser.parse_args(argv)
    if args.command == "audit-baseline":
        result = audit_manifest(args.manifest, limit=args.limit)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    elif args.command == "model-info":
        model = StrategyTransformerV1()
        result = {"model": "StrategyTransformerV1", "parameters": model.parameter_count(),
                  "policyInterface": "policy_forward", "evaluationInterface": "evaluation_forward"}
    else:
        supervisor = MindSupervisor(args.state_root, reserve_bytes=args.reserve_gb * 1024**3,
                                    data_cap_bytes=args.data_cap_gb * 1024**3)
        supervisor.persist(); result = supervisor.state.__dict__
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
