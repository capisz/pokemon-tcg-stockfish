#!/usr/bin/env python3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ptcg_lab.strategy_contract import validate_strategy_contract  # noqa: E402


if __name__ == "__main__":
    result = validate_strategy_contract(ROOT)
    print(
        "Validated strategy-contract-v1: "
        f"{len(result['playbooks'])} specialists, "
        f"{sum(len(item['principles']) for item in result['playbooks'].values())} principles, "
        f"{sum(len(item['matchups']) for item in result['playbooks'].values())} matchup perspectives."
    )
