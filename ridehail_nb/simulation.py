"""Controlled simulation experiments for Multinomial Naive Bayes."""

from __future__ import annotations

from time import perf_counter
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from .modeling import build_pipeline, calculate_metrics

DEFAULT_ALPHAS = (0.1, 0.5, 1.0, 2.0, 5.0)
DEFAULT_TRAIN_RATIOS = (0.6, 0.7, 0.8)
DEFAULT_SEEDS = (42, 43, 44, 45, 46)


def _undersample_training(x: pd.Series, y: pd.Series, seed: int):
    """Randomly undersample only the training partition, never the test set."""
    train = pd.DataFrame({"review": x.values, "label": y.values})
    min_count = int(train["label"].value_counts().min())
    sampled = (
        train.groupby("label", group_keys=False)
        .apply(lambda group: group.sample(n=min_count, random_state=seed))
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )
    return sampled["review"], sampled["label"]


def _fit_once(x_train, y_train, x_test, y_test, alpha: float, meta: dict):
    model = build_pipeline(alpha)
    t0 = perf_counter()
    model.fit(x_train, y_train)
    train_seconds = perf_counter() - t0

    t1 = perf_counter()
    pred = model.predict(x_test)
    predict_seconds = perf_counter() - t1

    row = {
        **meta,
        **calculate_metrics(y_test, pred),
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "training_seconds": float(train_seconds),
        "prediction_seconds": float(predict_seconds),
    }
    return row


def run_simulations(
    df: pd.DataFrame,
    alphas: Iterable[float] = DEFAULT_ALPHAS,
    train_ratios: Iterable[float] = DEFAULT_TRAIN_RATIOS,
    seeds: Iterable[int] = DEFAULT_SEEDS,
) -> pd.DataFrame:
    """Run three one-factor-at-a-time experiments.

    The test split is held constant within each random seed. For training-size
    experiments, smaller samples are drawn from the same 80% training pool.
    Undersampling is applied to training rows only; the natural test distribution
    is retained to make the evaluation more realistic.
    """
    rows: list[dict] = []
    for seed in seeds:
        x_pool, x_test, y_pool, y_test = train_test_split(
            df["review"].astype(str),
            df["label"].astype(int),
            test_size=0.2,
            random_state=int(seed),
            stratify=df["label"],
        )

        # Experiment A: smoothing parameter alpha, training pool fixed at 80%.
        for alpha in alphas:
            rows.append(
                _fit_once(
                    x_pool,
                    y_pool,
                    x_test,
                    y_test,
                    float(alpha),
                    {
                        "experiment": "Variasi alpha",
                        "parameter": float(alpha),
                        "balance": "Distribusi asli",
                        "train_ratio_total": 0.8,
                        "seed": int(seed),
                    },
                )
            )

        # Experiment B: sample 60%, 70%, and 80% of all rows from the same
        # 80% training pool; the 20% test split remains unchanged.
        for ratio in train_ratios:
            if not 0 < float(ratio) <= 0.8:
                raise ValueError("train_ratios harus lebih besar dari 0 dan maksimal 0.8.")
            fraction_of_pool = float(ratio) / 0.8
            subset = (
                pd.DataFrame({"review": x_pool.values, "label": y_pool.values})
                .groupby("label", group_keys=False)
                .apply(
                    lambda group: group.sample(
                        n=max(1, int(round(len(group) * fraction_of_pool))),
                        random_state=int(seed),
                    )
                )
                .sample(frac=1, random_state=int(seed))
                .reset_index(drop=True)
            )
            rows.append(
                _fit_once(
                    subset["review"],
                    subset["label"],
                    x_test,
                    y_test,
                    1.0,
                    {
                        "experiment": "Variasi jumlah data latih",
                        "parameter": float(ratio),
                        "balance": "Distribusi asli",
                        "train_ratio_total": float(ratio),
                        "seed": int(seed),
                    },
                )
            )

        # Experiment C: compare original vs undersampled training data.
        rows.append(
            _fit_once(
                x_pool,
                y_pool,
                x_test,
                y_test,
                1.0,
                {
                    "experiment": "Distribusi kelas training",
                    "parameter": "asli",
                    "balance": "Distribusi asli",
                    "train_ratio_total": 0.8,
                    "seed": int(seed),
                },
            )
        )
        x_under, y_under = _undersample_training(x_pool, y_pool, int(seed))
        rows.append(
            _fit_once(
                x_under,
                y_under,
                x_test,
                y_test,
                1.0,
                {
                    "experiment": "Distribusi kelas training",
                    "parameter": "undersampling",
                    "balance": "Undersampling training",
                    "train_ratio_total": 0.8,
                    "seed": int(seed),
                },
            )
        )

    return pd.DataFrame(rows)


def summarize_simulations(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate repeated runs into mean and standard deviation."""
    grouping = ["experiment", "parameter", "balance", "train_ratio_total"]
    metrics = [
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "training_seconds",
        "prediction_seconds",
        "n_train",
    ]
    summary = (
        results.groupby(grouping, dropna=False)[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in col if part != "").rstrip("_")
        if isinstance(col, tuple)
        else col
        for col in summary.columns
    ]
    return summary
