"""
Business-address normalization.

Produces:
- raw_address
- normalized_address
- address_tokens
- numbers
- postal_codes

The postal-code field contains candidates only.
"""

from __future__ import annotations

import re

from .normalize_common import (
    clean_raw,
    normalize_punctuation,
    tokenize,
    extract_numbers,
    extract_postal_candidates,
)


# Conservative address abbreviations observed/common in the dataset.
#
# We expand abbreviations rather than deleting them.
ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "dr": "drive",
    "ln": "lane",
    "lane": "lane",
    "blvd": "boulevard",
    "bvd": "boulevard",
    "hwy": "highway",
    "pkwy": "parkway",
    "pl": "place",
    "ct": "court",
    "cir": "circle",
    "trl": "trail",
    "ter": "terrace",
    "sq": "square",
}


def standardize_address_tokens(tokens: list[str]) -> list[str]:
    """Expand conservative address abbreviations."""
    return [
        ADDRESS_ABBREVIATIONS.get(token, token)
        for token in tokens
    ]


def normalize_address(value: object) -> dict:
    """
    Normalize one business address.
    """
    raw_address = clean_raw(value)

    if not raw_address:
        return {
            "raw_address": "",
            "normalized_address": "",
            "address_tokens": [],
            "numbers": [],
            "postal_codes": [],
        }

    # Extract numbers BEFORE punctuation normalization.
    numbers = extract_numbers(raw_address)

    # Postal codes are extracted separately.
    postal_codes = extract_postal_candidates(raw_address)

    # Normalize punctuation and case.
    normalized = normalize_punctuation(raw_address)

    # Tokenize.
    tokens = tokenize(normalized)

    # Expand conservative address abbreviations.
    tokens = standardize_address_tokens(tokens)

    normalized = " ".join(tokens)

    return {
        "raw_address": raw_address,
        "normalized_address": normalized,
        "address_tokens": tokens,
        "numbers": numbers,
        "postal_codes": postal_codes,
    }


def normalize_addresses(values) -> list[dict]:
    """Normalize an iterable of business addresses."""
    return [normalize_address(value) for value in values]