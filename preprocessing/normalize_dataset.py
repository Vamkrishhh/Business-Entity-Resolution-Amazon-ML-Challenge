from __future__ import annotations

import argparse
import gc
import re
import unicodedata
from pathlib import Path

import pandas as pd


CHUNK_SIZE = 10_000


# ---------------------------------------------------------
# Common
# ---------------------------------------------------------

NULL_VALUES = {"", "null", "none", "nan", "n/a", "na", "nil", "-"}

ADDRESS_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "str": "street",
    "ave": "avenue",
    "av": "avenue",
    "dr": "drive",
    "ln": "lane",
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


def clean_series(s: pd.Series) -> pd.Series:
    s = s.fillna("").astype(str).str.strip()

    mask = s.str.casefold().isin(NULL_VALUES)
    s = s.mask(mask, "")

    return s


def unicode_normalize_series(s: pd.Series) -> pd.Series:
    return s.map(
        lambda x: unicodedata.normalize("NFKC", x)
        if x
        else ""
    )


def basic_normalize(s: pd.Series) -> pd.Series:
    s = unicode_normalize_series(s)
    s = s.str.casefold()
    s = s.str.replace("&", " and ", regex=False)

    # Replace punctuation with spaces.
    s = s.str.replace(
        r"[^\w\s]",
        " ",
        regex=True,
    )

    # Remove underscores.
    s = s.str.replace("_", " ", regex=False)

    # Collapse whitespace.
    s = s.str.replace(
        r"\s+",
        " ",
        regex=True,
    ).str.strip()

    return s


# ---------------------------------------------------------
# Name normalization
# ---------------------------------------------------------

def normalize_names(s: pd.Series) -> pd.DataFrame:

    raw = clean_series(s)

    normalized = basic_normalize(raw)

    # Legal-name variants.
    replacements = [
        (r"\bprivate\s+ltd\b", "private limited"),
        (r"\bpvt\s+ltd\b", "private limited"),
        (r"\bpvt\s+limited\b", "private limited"),
        (r"\bprivate\s+limited\b", "private limited"),
        (r"\bpvt\b", "private"),
        (r"\bltd\b", "limited"),
        (r"\blimited\b", "limited"),
        (r"\bpublic\s+ltd\b", "public limited"),
        (r"\bpublic\s+limited\b", "public limited"),
        (r"\bincorporated\b", "inc"),
        (r"\bcorporation\b", "corp"),
    ]

    for pattern, replacement in replacements:
        normalized = normalized.str.replace(
            pattern,
            replacement,
            regex=True,
        )

    normalized = normalized.str.replace(
        r"\s+",
        " ",
        regex=True,
    ).str.strip()

    tokens = normalized.str.findall(r"[^\W_]+")
    token_text = tokens.str.join(" ")

    compact = token_text.str.replace(
        " ",
        "",
        regex=False,
    )

    return pd.DataFrame(
        {
            "raw_name": raw,
            "normalized_name": normalized,
            "name_tokens": token_text,
            "compact_name": compact,
        },
        index=s.index,
    )


# ---------------------------------------------------------
# Address normalization
# ---------------------------------------------------------

def normalize_addresses(s: pd.Series) -> pd.DataFrame:

    raw = clean_series(s)

    numbers = raw.str.findall(r"\d+").str.join(" ")

    # Candidate only — not guaranteed to be postal codes.
    postal_codes = raw.str.findall(
        r"\b\d{4,6}(?:-\d{4})?\b"
    ).str.join(" ")

    normalized = basic_normalize(raw)

    # Address abbreviations.
    for abbreviation, replacement in ADDRESS_ABBREVIATIONS.items():
        normalized = normalized.str.replace(
            rf"\b{re.escape(abbreviation)}\b",
            replacement,
            regex=True,
        )

    normalized = normalized.str.replace(
        r"\s+",
        " ",
        regex=True,
    ).str.strip()

    tokens = normalized.str.findall(r"[^\W_]+")
    token_text = tokens.str.join(" ")

    return pd.DataFrame(
        {
            "raw_address": raw,
            "normalized_address": normalized,
            "address_tokens": token_text,
            "numbers": numbers,
            "postal_codes": postal_codes,
        },
        index=s.index,
    )


# ---------------------------------------------------------
# Chunk processing
# ---------------------------------------------------------

def normalize_chunk(df: pd.DataFrame) -> pd.DataFrame:

    names = normalize_names(df["business_name"])
    addresses = normalize_addresses(df["business_address"])

    # Add normalized columns.
    df["raw_name"] = names["raw_name"]
    df["normalized_name"] = names["normalized_name"]
    df["name_tokens"] = names["name_tokens"]
    df["compact_name"] = names["compact_name"]

    df["raw_address"] = addresses["raw_address"]
    df["normalized_address"] = addresses["normalized_address"]
    df["address_tokens"] = addresses["address_tokens"]
    df["numbers"] = addresses["numbers"]
    df["postal_codes"] = addresses["postal_codes"]

    return df


# ---------------------------------------------------------
# File processing
# ---------------------------------------------------------

def normalize_file(
    input_path: Path,
    output_path: Path,
    chunk_size: int = CHUNK_SIZE,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    first_chunk = True
    total_rows = 0

    reader = pd.read_csv(
        input_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        chunksize=chunk_size,
    )

    for chunk_number, df in enumerate(reader, start=1):

        normalized_df = normalize_chunk(df)

        normalized_df.to_csv(
            output_path,
            sep="\t",
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )

        total_rows += len(df)

        print(
            f"Chunk {chunk_number}: "
            f"{len(df):,} rows | "
            f"total {total_rows:,}"
        )

        first_chunk = False

        # Explicitly release memory.
        del normalized_df
        del df
        gc.collect()

    print()
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Rows  : {total_rows:,}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input",
        type=Path,
    )

    parser.add_argument(
        "output",
        type=Path,
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}"
        )

    normalize_file(
        args.input,
        args.output,
        args.chunk_size,
    )


if __name__ == "__main__":
    main()