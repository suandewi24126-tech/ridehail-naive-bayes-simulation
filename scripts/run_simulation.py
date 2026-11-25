from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ridehail_nb.data import load_reviews
from ridehail_nb.simulation import run_simulations, summarize_simulations


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Naive Bayes simulations.")
    parser.add_argument("--data", default="data/demo_reviews.csv", help="Path CSV dataset")
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    args = parser.parse_args()

    dataset = load_reviews(ROOT / args.data)
    seeds = tuple(range(42, 42 + args.seeds))
    results = run_simulations(dataset, seeds=seeds)
    summary = summarize_simulations(results)

    out_dir = ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "simulation_runs.csv", index=False)
    summary.to_csv(out_dir / "simulation_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nHasil tersimpan di folder reports/")


if __name__ == "__main__":
    main()
