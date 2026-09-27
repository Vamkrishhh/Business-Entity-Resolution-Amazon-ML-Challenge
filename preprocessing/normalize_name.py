"""
Business-name normalization.

Produces:
- raw_name
- normalized_name
- name_tokens
- compact_name

Normalization is deliberately conservative.
"""

from __future__ import annotations

import re

from .normalize_common import (
    clean_raw,
    normalize_punctuation,
    tokenize,
    compact_tokens,
)


# Observed/common legal-form variants.
#
# These are canonicalized, NOT removed.
LEGAL_SUFFIX_PATTERNS = [
    (
        r"\bprivate\s+limited\b",
        "private limited",
    ),
    (
        r"\bprivate\s+ltd\b",
        "private limited",
    ),
    (
        r"\bpvt\s+ltd\b",
        "private limited",
    ),
    (
        r"\bpvt\s+limited\b",
        "private limited",
    ),
    (
        r"\bpvt\b",
        "private",
    ),
    (
        r"\blimited\b",
        "limited",
    ),
    (
        r"\bltd\b",
        "limited",
    ),
    (
        r"\bpublic\s+limited\b",
        "public limited",
    ),
    (
        r"\bpublic\s+ltd\b",
        "public limited",
    ),
    (
        r"\bllp\b",
        "llp",
    ),
    (
        r"\bincorporated\b",
        "inc",
    ),
    (
        r"\bcorporation\b",
        "corp",
    ),
]


def standardize_legal_terms(text: str) -> str:
    """
    Standardize common legal-form spelling variants.

    This function does NOT remove legal terms.
    """
    for pattern, replacement in LEGAL_SUFFIX_PATTERNS:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    return re.sub(r"\s+", " ", text).strip()


def normalize_name(value: object) -> dict:
    """
    Normalize one business name.

    Returns a dictionary containing all representations.
    """
    raw_name = clean_raw(value)

    if not raw_name:
        return {
            "raw_name": "",
            "normalized_name": "",
            "name_tokens": [],
            "compact_name": "",
        }

    # Punctuation -> spaces, casefolding, Unicode normalization.
    normalized = normalize_punctuation(raw_name)

    # Standardize common legal-form variants.
    normalized = standardize_legal_terms(normalized)

    # Final whitespace cleanup.
    normalized = re.sub(
        r"\s+",
        " ",
        normalized,
    ).strip()

    tokens = tokenize(normalized)

    return {
        "raw_name": raw_name,
        "normalized_name": normalized,
        "name_tokens": tokens,
        "compact_name": compact_tokens(tokens),
    }


def normalize_names(values) -> list[dict]:
    """Normalize an iterable of business names."""
    return [normalize_name(value) for value in values]