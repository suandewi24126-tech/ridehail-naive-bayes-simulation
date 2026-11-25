from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ridehail_nb.data import load_reviews
from ridehail_nb.modeling import fit_and_evaluate
from ridehail_nb.reporting import save_metrics, save_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TF-IDF + MultinomialNB.")
    parser.add_argument("--data", default="data/demo_reviews.csv", help="Path CSV dataset")
    parser.add_argument("--alpha", type=float, default=1.0, help="Smoothing parameter")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    dataset = load_reviews(ROOT / args.data)
    result = fit_and_evaluate(
        dataset,
        alpha=args.alpha,
        test_size=args.test_size,
        random_state=args.seed,
    )

    save_model(result["model"], ROOT / "models" / "naive_bayes_pipeline.joblib")
    metrics = {
        "alpha": result["alpha"],
        "random_state": result["random_state"],
        "n_train": result["n_train"],
        "n_test": result["n_test"],
        "metrics": result["metrics"],
        "confusion_matrix": result["confusion_matrix"],
        "training_seconds": result["training_seconds"],
        "prediction_seconds": result["prediction_seconds"],
    }
    save_metrics(metrics, ROOT / "reports" / "metrics.json")
    print(json.dumps(metrics, indent=2, ensure_ascii=False, default=lambda value: value.tolist() if hasattr(value, "tolist") else float(value)))
    print("\nModel tersimpan di models/naive_bayes_pipeline.joblib")


if __name__ == "__main__":
    main()
