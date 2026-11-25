"""Lightweight Indonesian text preprocessing for ride-hailing reviews."""

from __future__ import annotations

import re
from typing import Iterable

_URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
_NON_LETTER_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_DIGIT_RE = re.compile(r"\b\d+\b")
_WHITESPACE_RE = re.compile(r"\s+")

# A small, transparent normalization dictionary. Expand this only when the
# group has reviewed the vocabulary in its own dataset.
NORMALIZATION_MAP = {
    "gak": "tidak",
    "nggak": "tidak",
    "ngga": "tidak",
    "gk": "tidak",
    "ga": "tidak",
    "enggak": "tidak",
    "bgt": "banget",
    "yg": "yang",
    "dgn": "dengan",
    "krn": "karena",
    "aja": "saja",
    "apk": "aplikasi",
}


def clean_text(text: object, normalization_map: dict[str, str] | None = None) -> str:
    """Clean one review while preserving negation words such as ``tidak``.

    The function intentionally avoids stop-word removal and stemming by
    default because those steps can remove or alter cues important to
    subjectivity classification. It is safe to pass as a scikit-learn
    TfidfVectorizer preprocessor.
    """
    if text is None:
        return ""

    value = str(text).lower().strip()
    value = _URL_RE.sub(" ", value)
    value = _EMAIL_RE.sub(" ", value)
    value = value.replace("\n", " ").replace("\r", " ")
    value = _NON_LETTER_RE.sub(" ", value)
    value = _DIGIT_RE.sub(" ", value)
    value = _WHITESPACE_RE.sub(" ", value).strip()

    mapping = normalization_map if normalization_map is not None else NORMALIZATION_MAP
    tokens: Iterable[str] = value.split()
    normalized = [mapping.get(token, token) for token in tokens]
    return " ".join(normalized)
