"""
features.py — Pairwise similarity feature engineering.

build_features(left, right) takes two entity records (dicts with keys
``name_normalized``, ``address_normalized``, ``country_normalized``) and
returns a feature dict mapping feature names to float values in [0, 1].

Features are grouped into:
  - Name similarity (token Jaccard, char n-gram Jaccard, length ratio,
    prefix match, exact match)
  - Address similarity (token Jaccard, char n-gram Jaccard, numeric
    token overlap, length ratio, exact match)
  - Country agreement (binary exact match)
  - Combined cross-field score

No external libraries beyond the standard library are required for this
module.  All feature values lie in [0.0, 1.0].
"""

from __future__ import annotations

import re
from typing import Any


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _tokens(text: str) -> set[str]:
    """Return the set of Unicode word tokens from a normalized string."""
    if not text:
        return set()
    return set(re.findall(r"[^\W_]+", text, flags=re.UNICODE))


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity |a ∩ b| / |a ∪ b|."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _char_ngrams(text: str, n: int) -> set[str]:
    """
    Character n-gram multiset (as a set — duplicates discarded).

    Boundary padding with a single space improves prefix/suffix handling.
    """
    padded = f" {text} "
    if len(padded) < n:
        return {padded}
    return {padded[i : i + n] for i in range(len(padded) - n + 1)}


def _ngram_similarity(a: str, b: str, n: int = 3) -> float:
    """Jaccard similarity on character n-grams."""
    sa = _char_ngrams(a, n)
    sb = _char_ngrams(b, n)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _length_ratio(a: str, b: str) -> float:
    """
    Ratio of shorter length to longer length.

    Returns 1.0 when both are empty (equal length), 0.0 when one is empty.
    """
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _number_overlap(a: str, b: str) -> float:
    """
    Fraction of numeric tokens in ``a`` that also appear in ``b``.

    Returns 1.0 when ``a`` contains no numbers (nothing to contradict),
    which avoids penalizing pairs where only one side has address numbers.
    """
    nums_a = set(re.findall(r"\d+", a))
    nums_b = set(re.findall(r"\d+", b))
    if not nums_a:
        return 1.0
    return len(nums_a & nums_b) / len(nums_a)


def _prefix_match(a: str, b: str, length: int = 4) -> float:
    """Binary feature: 1.0 if both strings share the first ``length`` chars."""
    if not a or not b:
        return 0.0
    return 1.0 if a[:length] == b[:length] else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(
    left: dict[str, Any],
    right: dict[str, Any],
) -> dict[str, float]:
    """
    Construct a pairwise feature vector for an entity candidate pair.

    Parameters
    ----------
    left : dict
        Source-1 entity record.  Expected keys:
        ``entity_id``, ``name_normalized``, ``address_normalized``,
        ``country_normalized``.
    right : dict
        Source-2 or Source-3 candidate entity record (same structure).

    Returns
    -------
    dict[str, float]
        Mapping from feature name to float value.  All values lie in
        [0.0, 1.0].  The dict is always the same size and key-order;
        missing fields default to "".

    Feature catalogue
    -----------------
    Name features
      name_token_jaccard     — Jaccard similarity of name word-token sets
      name_char3gram_jaccard — Jaccard of 3-char n-gram sets
      name_char4gram_jaccard — Jaccard of 4-char n-gram sets
      name_length_ratio      — min(len_a, len_b) / max(len_a, len_b)
      name_prefix4_match     — 1.0 if first 4 chars match exactly
      name_exact_match       — 1.0 if names are identical (non-empty)

    Address features
      addr_token_jaccard     — Jaccard of address word-token sets
      addr_char3gram_jaccard — Jaccard of 3-char n-gram sets
      addr_number_overlap    — fraction of left numeric tokens in right
      addr_length_ratio      — length ratio
      addr_exact_match       — 1.0 if addresses are identical (non-empty)

    Country features
      country_exact_match    — 1.0 if ISO codes agree (non-empty)

    Cross features
      name_addr_combined     — 0.6 × name_token_jaccard
                               + 0.4 × addr_token_jaccard
    """
    lname = str(left.get("name_normalized") or "")
    rname = str(right.get("name_normalized") or "")
    laddr = str(left.get("address_normalized") or "")
    raddr = str(right.get("address_normalized") or "")
    lcountry = str(left.get("country_normalized") or "")
    rcountry = str(right.get("country_normalized") or "")

    name_toks_l = _tokens(lname)
    name_toks_r = _tokens(rname)
    addr_toks_l = _tokens(laddr)
    addr_toks_r = _tokens(raddr)

    name_tok_j = _jaccard(name_toks_l, name_toks_r)
    addr_tok_j = _jaccard(addr_toks_l, addr_toks_r)

    return {
        # --- Name ---
        "name_token_jaccard": name_tok_j,
        "name_char3gram_jaccard": _ngram_similarity(lname, rname, n=3),
        "name_char4gram_jaccard": _ngram_similarity(lname, rname, n=4),
        "name_length_ratio": _length_ratio(lname, rname),
        "name_prefix4_match": _prefix_match(lname, rname, length=4),
        "name_exact_match": float(lname != "" and lname == rname),
        # --- Address ---
        "addr_token_jaccard": addr_tok_j,
        "addr_char3gram_jaccard": _ngram_similarity(laddr, raddr, n=3),
        "addr_number_overlap": _number_overlap(laddr, raddr),
        "addr_length_ratio": _length_ratio(laddr, raddr),
        "addr_exact_match": float(laddr != "" and laddr == raddr),
        # --- Country ---
        "country_exact_match": float(lcountry != "" and lcountry == rcountry),
        # --- Cross ---
        "name_addr_combined": 0.6 * name_tok_j + 0.4 * addr_tok_j,
    }
