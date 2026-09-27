"""
Common normalization utilities for business entity resolution.

Design principles:
- Never destroy the original value.
- Normalize conservatively.
- Preserve Unicode.
- Preserve meaningful numbers.
- Do not perform fuzzy matching or typo correction here.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable


# Values that should be treated as missing.
NULL_LIKE_VALUES = {
    "",
    "null",
    "none",
    "nan",
    "n/a",
    "na",
    "nil",
    "-",
}


def clean_raw(value: object) -> str:
    """Convert an input value into a safe raw string."""
    if value is None:
        return ""

    text = str(value).strip()

    if text.casefold() in NULL_LIKE_VALUES:
        return ""

    return text


def unicode_normalize(value: object) -> str:
    """
    Apply Unicode compatibility normalization.

    NFKC helps normalize visually/structurally equivalent Unicode forms
    without deleting non-Latin scripts.
    """
    text = clean_raw(value)

    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)

    return text


def normalize_case(value: object) -> str:
    """Unicode-aware lowercase normalization."""
    text = unicode_normalize(value)

    if not text:
        return ""

    return text.casefold()


def normalize_whitespace(value: object) -> str:
    """Collapse repeated whitespace."""
    text = normalize_case(value)

    if not text:
        return ""

    return re.sub(r"\s+", " ", text).strip()


def normalize_ampersand(value: object) -> str:
    """
    Normalize '&' to the word 'and'.

    This is useful because '&' vs 'and' is a common business-name
    variation.
    """
    text = normalize_whitespace(value)

    if not text:
        return ""

    text = text.replace("&", " and ")

    return re.sub(r"\s+", " ", text).strip()


def tokenize(value: object) -> list[str]:
    """
    Tokenize while preserving Unicode letters and numbers.

    Punctuation becomes a separator.
    """
    text = normalize_ampersand(value)

    if not text:
        return []

    return re.findall(r"[^\W_]+", text, flags=re.UNICODE)


def compact_tokens(tokens: Iterable[str]) -> str:
    """
    Join tokens without separators.

    Unicode characters are intentionally preserved.
    """
    return "".join(tokens)


def compact(value: object) -> str:
    """Tokenize and concatenate tokens."""
    return compact_tokens(tokenize(value))


def extract_numbers(value: object) -> list[str]:
    """
    Extract numeric sequences.

    Numbers are intentionally preserved because they are highly
    informative in addresses.
    """
    text = unicode_normalize(value)

    if not text:
        return []

    return re.findall(r"\d+", text)


def extract_postal_candidates(value: object) -> list[str]:
    """
    Extract plausible postal-code candidates.

    These are candidates, NOT guaranteed postal codes.

    Supports:
    - 4-6 digit numeric sequences
    - common French 5-digit postal codes
    - Indian 6-digit postal codes
    - US ZIP / ZIP+4 style values
    """
    text = unicode_normalize(value)

    if not text:
        return []

    candidates = []

    # ZIP / ZIP+4
    candidates.extend(
        re.findall(r"\b\d{5}(?:-\d{4})?\b", text)
    )

    # 4-6 digit candidates
    candidates.extend(
        re.findall(r"\b\d{4,6}\b", text)
    )

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(candidates))


def normalize_punctuation(value: object) -> str:
    """
    Convert punctuation into spaces while preserving letters,
    numbers and Unicode characters.
    """
    text = normalize_ampersand(value)

    if not text:
        return ""

    # Replace punctuation/symbols with spaces.
    text = re.sub(
        r"[^\w\s]",
        " ",
        text,
        flags=re.UNICODE,
    )

    # '_' is technically \w but is not useful for entity matching.
    text = text.replace("_", " ")

    return re.sub(r"\s+", " ", text).strip()