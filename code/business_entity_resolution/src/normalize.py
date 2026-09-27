"""
normalize.py — Business entity field normalization.

Produces scalar string representations suitable for:
  - DataFrame column storage (used by pipeline.py)
  - Blocking / candidate-generation keys (country_normalized)
  - Feature engineering inputs (name_normalized, address_normalized)

Design principles (consistent with the team preprocessing module):
  - Never destroy the original value.
  - Normalize conservatively; no fuzzy correction.
  - Preserve Unicode letters and numbers.
  - Preserve meaningful numbers in addresses.
  - Do not call external APIs or geocoding services.
"""

from __future__ import annotations

import re
import unicodedata


# ---------------------------------------------------------------------------
# Null-like sentinel values
# ---------------------------------------------------------------------------

_NULL_LIKE: frozenset[str] = frozenset(
    {"", "null", "none", "nan", "n/a", "na", "nil", "-"}
)


# ---------------------------------------------------------------------------
# Legal-form suffix canonicalization table
# ---------------------------------------------------------------------------

_LEGAL_SUFFIX_PATTERNS: list[tuple[str, str]] = [
    (r"\bprivate\s+limited\b", "private limited"),
    (r"\bprivate\s+ltd\b", "private limited"),
    (r"\bpvt\s+ltd\b", "private limited"),
    (r"\bpvt\s+limited\b", "private limited"),
    (r"\bpvt\b", "private"),
    (r"\blimited\b", "limited"),
    (r"\bltd\b", "limited"),
    (r"\bpublic\s+limited\b", "public limited"),
    (r"\bpublic\s+ltd\b", "public limited"),
    (r"\bllp\b", "llp"),
    (r"\bincorporated\b", "inc"),
    (r"\bcorporation\b", "corp"),
]


# ---------------------------------------------------------------------------
# Conservative address abbreviation expansion table
# ---------------------------------------------------------------------------

_ADDRESS_ABBREVIATIONS: dict[str, str] = {
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


# ---------------------------------------------------------------------------
# Country alias table → ISO 3166-1 alpha-2
# ---------------------------------------------------------------------------

_COUNTRY_ALIASES: dict[str, str] = {
    "united states": "us",
    "united states of america": "us",
    "usa": "us",
    "u.s.a.": "us",
    "u.s.": "us",
    "united kingdom": "gb",
    "uk": "gb",
    "great britain": "gb",
    "england": "gb",
    "deutschland": "de",
    "germany": "de",
    "france": "fr",
    "españa": "es",
    "spain": "es",
    "italia": "it",
    "italy": "it",
    "india": "in",
    "china": "cn",
    "people's republic of china": "cn",
    "brasil": "br",
    "brazil": "br",
    "canada": "ca",
    "australia": "au",
    "nederland": "nl",
    "netherlands": "nl",
    "holland": "nl",
    "polska": "pl",
    "poland": "pl",
    "türkiye": "tr",
    "turkey": "tr",
    "россия": "ru",
    "russia": "ru",
    "日本": "jp",
    "japan": "jp",
    "한국": "kr",
    "south korea": "kr",
    "korea": "kr",
    "méxico": "mx",
    "mexico": "mx",
    "österreich": "at",
    "austria": "at",
    "schweiz": "ch",
    "switzerland": "ch",
    "suisse": "ch",
    "belgique": "be",
    "belgium": "be",
    "sverige": "se",
    "sweden": "se",
    "norge": "no",
    "norway": "no",
    "danmark": "dk",
    "denmark": "dk",
    "suomi": "fi",
    "finland": "fi",
    "portugal": "pt",
    "new zealand": "nz",
    "south africa": "za",
    "argentina": "ar",
    "chile": "cl",
    "colombia": "co",
    "singapore": "sg",
    "malaysia": "my",
    "indonesia": "id",
    "thailand": "th",
    "philippines": "ph",
    "vietnam": "vn",
    "egypt": "eg",
    "israel": "il",
    "saudi arabia": "sa",
    "united arab emirates": "ae",
    "uae": "ae",
    "pakistan": "pk",
    "bangladesh": "bd",
    "nigeria": "ng",
    "kenya": "ke",
    "ghana": "gh",
    "taiwan": "tw",
    "hong kong": "hk",
    "czech republic": "cz",
    "czechia": "cz",
    "hungary": "hu",
    "romania": "ro",
    "ukraine": "ua",
    "greece": "gr",
    "croatia": "hr",
    "slovakia": "sk",
    "slovenia": "si",
    "serbia": "rs",
    "iran": "ir",
    "iraq": "iq",
    "jordan": "jo",
    "kuwait": "kw",
    "qatar": "qa",
    "bahrain": "bh",
    "oman": "om",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_raw(value: object) -> str:
    """Convert any input to a stripped string, treating null-likes as ''."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in _NULL_LIKE:
        return ""
    return text


def _basic_normalize(text: str) -> str:
    """
    Apply Unicode NFKC, case-folding, ampersand expansion,
    punctuation → spaces, and whitespace collapse.
    """
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = text.replace("&", " and ")
    # Punctuation and symbols → spaces (preserve letters, digits, spaces).
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    # Underscores are not useful for entity matching.
    text = text.replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def normalize_name(value: object) -> str:
    """
    Normalize a business name to a canonical lowercased string.

    Steps applied (in order):
    1. Null/empty detection → return "".
    2. Unicode NFKC normalization.
    3. Case-folding (Unicode-aware lowercase).
    4. Ampersand expansion (& → and).
    5. Punctuation → spaces.
    6. Legal-form suffix canonicalization
       (e.g. "Pvt Ltd" → "private limited").
    7. Whitespace collapse.

    Returns
    -------
    str
        Normalized business name, or "" for null/empty inputs.
    """
    raw = _clean_raw(value)
    if not raw:
        return ""

    text = _basic_normalize(raw)

    for pattern, replacement in _LEGAL_SUFFIX_PATTERNS:
        text = re.sub(
            pattern,
            replacement,
            text,
            flags=re.IGNORECASE,
        )

    return re.sub(r"\s+", " ", text).strip()


def normalize_address(value: object) -> str:
    """
    Normalize a business address to a canonical lowercased string.

    Steps applied (in order):
    1. Null/empty detection → return "".
    2. Unicode NFKC normalization.
    3. Case-folding.
    4. Punctuation → spaces.
    5. Tokenization (preserving Unicode letters and numbers).
    6. Conservative abbreviation expansion
       (e.g. "rd" → "road", "ave" → "avenue").
    7. Token re-joining and whitespace collapse.

    Note: Numbers are preserved — they are highly discriminative
    in address matching.

    Returns
    -------
    str
        Normalized address string, or "" for null/empty inputs.
    """
    raw = _clean_raw(value)
    if not raw:
        return ""

    text = _basic_normalize(raw)
    tokens = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
    tokens = [_ADDRESS_ABBREVIATIONS.get(tok, tok) for tok in tokens]
    return " ".join(tokens)


def normalize_country(value: object) -> str:
    """
    Normalize a country name or code to an ISO 3166-1 alpha-2 string.

    Matching is case-insensitive and handles common variants
    (e.g. "Deutschland" → "de", "United Kingdom" → "gb").

    If the value is already a 2-letter alphabetic code, it is returned
    lowercased directly.  Unknown values are returned as-is (lowercased)
    so that unknown countries are still grouped consistently during
    candidate generation.

    Returns
    -------
    str
        Two-letter ISO code (lowercase), raw lowercased value for unknowns,
        or "" for null/empty inputs.
    """
    raw = _clean_raw(value)
    if not raw:
        return ""

    lower = raw.strip().casefold()

    # Already a 2-letter code.
    if re.fullmatch(r"[a-z]{2}", lower):
        return lower

    return _COUNTRY_ALIASES.get(lower, lower)
