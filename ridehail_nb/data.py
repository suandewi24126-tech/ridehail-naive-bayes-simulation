"""CSV loading and validation utilities."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO

import pandas as pd

LABEL_MAP = {
    "0": 0,
    "objektif": 0,
    "objective": 0,
    "obj": 0,
    "1": 1,
    "subjektif": 1,
    "subjective": 1,
    "subj": 1,
}


class DatasetValidationError(ValueError):
    """Raised when the uploaded dataset does not follow the expected schema."""


def _read_csv(source: str | Path | BinaryIO | BytesIO) -> pd.DataFrame:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File tidak ditemukan: {path}")
        try:
            return pd.read_csv(path, encoding="utf-8-sig")
        except UnicodeDecodeError:
            return pd.read_csv(path, encoding="latin-1")

    content = source.read()
    if hasattr(source, "seek"):
        source.seek(0)
    if isinstance(content, str):
        content = content.encode("utf-8")
    try:
        return pd.read_csv(BytesIO(content), encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(BytesIO(content), encoding="latin-1")


def normalize_label(value: object) -> int:
    # CSV exported from spreadsheets can encode integer labels as 0.0 / 1.0.
    try:
        numeric = float(value)
        if numeric in (0.0, 1.0):
            return int(numeric)
    except (TypeError, ValueError):
        pass

    key = str(value).strip().lower()
    if key not in LABEL_MAP:
        raise DatasetValidationError(
            "Label harus berupa 0/1 atau teks objektif/subjektif. "
            f"Nilai yang tidak dikenali: {value!r}"
        )
    return LABEL_MAP[key]


def load_reviews(source: str | Path | BinaryIO | BytesIO) -> pd.DataFrame:
    """Load a CSV and return standardized columns: review and label.

    Expected source columns are ``review`` and ``label``. Additional columns
    such as ``score`` and ``app`` are retained for descriptive analysis.
    Label convention follows the group's earlier report: 0=Objektif,
    1=Subjektif.
    """
    df = _read_csv(source)
    df.columns = [str(column).strip().lower() for column in df.columns]

    required = {"review", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise DatasetValidationError(
            "Kolom wajib tidak ditemukan: " + ", ".join(sorted(missing))
            + ". CSV minimal harus memiliki kolom 'review' dan 'label'."
        )

    df = df.copy()
    df["review"] = df["review"].fillna("").astype(str).str.strip()
    df = df[(df["review"] != "") & df["label"].notna()].copy()
    if df.empty:
        raise DatasetValidationError("Tidak ada baris dengan review dan label yang valid.")

    try:
        df["label"] = df["label"].map(normalize_label).astype(int)
    except DatasetValidationError:
        raise

    if df["label"].nunique() < 2:
        raise DatasetValidationError(
            "Dataset harus berisi kedua kelas: 0=Objektif dan 1=Subjektif."
        )

    counts = df["label"].value_counts()
    if counts.min() < 2:
        raise DatasetValidationError(
            "Setiap kelas minimal harus memiliki 2 baris agar dapat dibagi secara stratified."
        )

    return df.reset_index(drop=True)


def class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["label"].value_counts().reindex([0, 1], fill_value=0)
    return pd.DataFrame(
        {
            "kelas": ["Objektif", "Subjektif"],
            "label": [0, 1],
            "jumlah": [int(counts.loc[0]), int(counts.loc[1])],
            "persentase": [
                float(counts.loc[0] / len(df) * 100),
                float(counts.loc[1] / len(df) * 100),
            ],
        }
    )
