"""
Dataset profiling for Amazon ML Challenge 2026
Business Entity Resolution

Purpose
-------
Profile all source datasets and ground truth without loading the full
2.5 GB dataset into memory.

Outputs
-------
reports/dataset_profile.md
reports/profile_raw_stats.json
reports/profile_examples.json

Usage
-----
From repository root:

    python -m preprocessing.profiling

Or:

    python preprocessing/profiling.py

Expected dataset structure
--------------------------
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
"""

from __future__ import annotations

import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT_DIR / "dataset"
TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

REPORT_DIR = ROOT_DIR / "reports"

CHUNK_SIZE = 50_000

SOURCE_FILES = {
    "train_source1": TRAIN_DIR / "train_source1.tsv",
    "train_source2": TRAIN_DIR / "train_source2.tsv",
    "train_source3": TRAIN_DIR / "train_source3.tsv",
    "test_source1": TEST_DIR / "test_source1.tsv",
    "test_source2": TEST_DIR / "test_source2.tsv",
    "test_source3": TEST_DIR / "test_source3.tsv",
    "train_ground_truth": TRAIN_DIR / "train_ground_truth.tsv",
}


# ============================================================
# COMMON COLUMN DETECTION
# ============================================================

NAME_COLUMN_CANDIDATES = [
    "name",
    "business_name",
    "company_name",
    "entity_name",
    "merchant_name",
    "legal_name",
]

ADDRESS_COLUMN_CANDIDATES = [
    "address",
    "business_address",
    "company_address",
    "entity_address",
    "street",
    "full_address",
]

COUNTRY_COLUMN_CANDIDATES = [
    "country",
    "country_code",
    "country_name",
]

CITY_COLUMN_CANDIDATES = [
    "city",
    "town",
    "locality",
]

STATE_COLUMN_CANDIDATES = [
    "state",
    "state_code",
    "province",
    "region",
]

POSTAL_COLUMN_CANDIDATES = [
    "postal_code",
    "postcode",
    "zip",
    "zip_code",
    "pincode",
    "pin",
]

ID_COLUMN_CANDIDATES = [
    "id",
    "entity_id",
    "business_id",
    "company_id",
    "record_id",
]


# ============================================================
# TEXT HELPERS
# ============================================================

def clean_string(value: Any) -> str:
    """Convert a value to a clean string for analysis."""
    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    return str(value).strip()


def normalize_for_analysis(value: str) -> str:
    """
    Lightweight normalization used ONLY for profiling.

    This is deliberately not the final normalization logic.
    """
    value = clean_string(value).lower()

    value = re.sub(r"\s+", " ", value)

    return value.strip()


def compact_for_analysis(value: str) -> str:
    """
    Remove non-alphanumeric characters for exploratory comparison.

    This is NOT the production normalization.
    """
    value = normalize_for_analysis(value)

    # Exploratory only: preserve Unicode letters/digits instead of
    # deleting non-Latin scripts. This is NOT production normalization.
    return "".join(char for char in value if char.isalnum())


def tokenize(value: str) -> list[str]:
    """Basic exploratory tokenizer."""
    value = normalize_for_analysis(value)

    if not value:
        return []

    return re.findall(r"[a-z0-9]+", value)


def extract_numbers(value: str) -> list[str]:
    """Extract numeric sequences from a string."""
    value = clean_string(value)

    return re.findall(r"\d+", value)


def extract_postal_candidates(value: str) -> list[str]:
    """
    Extract likely postal-code-like numeric/alphanumeric sequences.

    This is intentionally permissive because countries can have different
    postal code formats.
    """
    value = clean_string(value)

    candidates = []

    # Numeric sequences of 4-6 digits
    candidates.extend(re.findall(r"\b\d{4,6}\b", value))

    # UK-like / alphanumeric postal patterns
    candidates.extend(
        re.findall(
            r"\b[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2}\b",
            value.upper(),
        )
    )

    return candidates


# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(
    columns: Iterable[str],
    candidates: list[str],
) -> str | None:
    """
    Find a column using exact candidate matching first,
    then normalized matching.
    """
    columns = list(columns)

    normalized = {
        str(column).strip().lower(): column
        for column in columns
    }

    for candidate in candidates:
        if candidate.lower() in normalized:
            return normalized[candidate.lower()]

    # Secondary fuzzy matching
    for column in columns:
        column_lower = str(column).lower()

        for candidate in candidates:
            if candidate.lower() in column_lower:
                return column

    return None


def detect_columns(columns: Iterable[str]) -> dict[str, str | None]:
    """Detect important semantic columns."""
    columns = list(columns)

    return {
        "id": find_column(columns, ID_COLUMN_CANDIDATES),
        "name": find_column(columns, NAME_COLUMN_CANDIDATES),
        "address": find_column(columns, ADDRESS_COLUMN_CANDIDATES),
        "country": find_column(columns, COUNTRY_COLUMN_CANDIDATES),
        "city": find_column(columns, CITY_COLUMN_CANDIDATES),
        "state": find_column(columns, STATE_COLUMN_CANDIDATES),
        "postal": find_column(columns, POSTAL_COLUMN_CANDIDATES),
    }


# ============================================================
# RESERVOIR SAMPLING
# ============================================================

class ReservoirSampler:
    """
    Fixed-size reservoir sampler.

    Allows us to retain random examples from huge datasets without
    storing every unique value.
    """

    def __init__(self, size: int = 20):
        self.size = size
        self.items: list[str] = []
        self.seen = 0

    def add(self, value: str) -> None:
        value = clean_string(value)

        if not value:
            return

        self.seen += 1

        if len(self.items) < self.size:
            self.items.append(value)
            return

        # Reservoir sampling
        import random

        index = random.randint(1, self.seen)

        if index <= self.size:
            self.items[index - 1] = value

    def get(self) -> list[str]:
        return self.items


# ============================================================
# FIELD PROFILE
# ============================================================

class FieldProfile:
    """Statistics for one semantic field."""

    def __init__(self, sample_size: int = 20):
        self.total = 0
        self.missing = 0
        self.empty = 0

        # value_counts is sufficient to derive exact unique counts and
        # repeated-value statistics; keeping a second set is redundant.
        self.value_counts: Counter[str] = Counter()

        self.lengths: list[int] = []

        self.examples = ReservoirSampler(sample_size)

        self.token_counter: Counter[str] = Counter()
        self.number_counter: Counter[str] = Counter()

        self.punctuation_counter: Counter[str] = Counter()

        self.case_patterns: Counter[str] = Counter()

    def update(self, value: Any) -> None:
        self.total += 1

        value = clean_string(value)

        if not value:
            self.missing += 1
            return

        self.examples.add(value)

        normalized = normalize_for_analysis(value)

        self.value_counts[normalized] += 1

        self.lengths.append(len(value))

        self.token_counter.update(tokenize(value))

        self.number_counter.update(extract_numbers(value))

        for char in value:
            if char in ".,;:-/()'\"&":
                self.punctuation_counter[char] += 1

        if value.isupper():
            self.case_patterns["UPPER"] += 1
        elif value.islower():
            self.case_patterns["LOWER"] += 1
        elif value.istitle():
            self.case_patterns["TITLE"] += 1
        else:
            self.case_patterns["MIXED"] += 1

    def summary(self) -> dict[str, Any]:
        if self.lengths:
            mean_length = statistics.mean(self.lengths)
            median_length = statistics.median(self.lengths)
            min_length = min(self.lengths)
            max_length = max(self.lengths)
        else:
            mean_length = median_length = min_length = max_length = 0

        return {
            "total": self.total,
            "missing": self.missing,
            "missing_pct": (
                self.missing / self.total * 100
                if self.total
                else 0
            ),
            "unique_normalized": len(self.value_counts),
            "repeated_normalized_values": sum(
                1 for count in self.value_counts.values() if count > 1
            ),
            "duplicate_normalized_occurrences": sum(
                max(count - 1, 0) for count in self.value_counts.values()
            ),
            "mean_length": round(mean_length, 2),
            "median_length": median_length,
            "min_length": min_length,
            "max_length": max_length,
            "top_values": self.value_counts.most_common(20),
            "top_tokens": self.token_counter.most_common(30),
            "top_numbers": self.number_counter.most_common(30),
            "punctuation": dict(self.punctuation_counter),
            "case_patterns": dict(self.case_patterns),
            "examples": self.examples.get(),
        }


# ============================================================
# DATASET PROFILE
# ============================================================

class DatasetProfiler:
    """Streaming profiler for one TSV dataset."""

    def __init__(
        self,
        file_path: Path,
        dataset_name: str,
        chunk_size: int = CHUNK_SIZE,
    ):
        self.file_path = file_path
        self.dataset_name = dataset_name
        self.chunk_size = chunk_size

        self.rows = 0
        self.columns: list[str] = []

        self.column_profiles: dict[str, FieldProfile] = {}

        self.detected_columns: dict[str, str | None] = {}

        self.country_counts: Counter[str] = Counter()

        self.ids: set[str] = set()

        self.duplicate_id_count = 0

        self.name_profiles: FieldProfile | None = None
        self.address_profiles: FieldProfile | None = None

        self.name_compact_to_raw: defaultdict[str, set[str]] = defaultdict(set)
        self.address_compact_to_raw: defaultdict[str, set[str]] = defaultdict(set)

        self.normalization_name_collisions: Counter[str] = Counter()
        self.normalization_address_collisions: Counter[str] = Counter()

        self.special_examples = {
            "private_limited_variants": ReservoirSampler(30),
            "road_variants": ReservoirSampler(30),
            "punctuation": ReservoirSampler(30),
            "mixed_case": ReservoirSampler(30),
            "numbers": ReservoirSampler(30),
            "postal_codes": ReservoirSampler(30),
        }

    # --------------------------------------------------------
    # PROCESS FILE
    # --------------------------------------------------------

    def run(self) -> dict[str, Any]:
        print()
        print("=" * 80)
        print(f"Profiling: {self.dataset_name}")
        print(f"File: {self.file_path}")
        print("=" * 80)

        if not self.file_path.exists():
            print(f"WARNING: File not found: {self.file_path}")

            return {
                "dataset": self.dataset_name,
                "file": str(self.file_path),
                "exists": False,
            }

        first_chunk = True

        try:
            reader = pd.read_csv(
                self.file_path,
                sep="\t",
                chunksize=self.chunk_size,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
                quoting=csv.QUOTE_MINIMAL,
                on_bad_lines="warn",
            )

            for chunk_number, chunk in enumerate(reader, start=1):
                if first_chunk:
                    self.columns = list(chunk.columns)

                    self.detected_columns = detect_columns(
                        self.columns
                    )

                    for column in self.columns:
                        self.column_profiles[column] = FieldProfile()

                    first_chunk = False

                    print(
                        f"Columns detected: {self.columns}"
                    )

                    print(
                        f"Semantic columns: "
                        f"{self.detected_columns}"
                    )

                self._process_chunk(chunk, chunk_number)

        except Exception as exc:
            print(
                f"ERROR while profiling {self.file_path}: "
                f"{type(exc).__name__}: {exc}"
            )

            return {
                "dataset": self.dataset_name,
                "file": str(self.file_path),
                "exists": True,
                "error": str(exc),
            }

        result = self._build_result()

        print(
            f"Finished {self.dataset_name}: "
            f"{self.rows:,} rows"
        )

        return result

    # --------------------------------------------------------
    # PROCESS CHUNK
    # --------------------------------------------------------

    def _process_chunk(
        self,
        chunk: pd.DataFrame,
        chunk_number: int,
    ) -> None:

        self.rows += len(chunk)

        print(
            f"  chunk {chunk_number:<5} "
            f"rows processed: {self.rows:,}",
            end="\r",
        )

        # ---------------------------------------------
        # Generic column statistics
        # ---------------------------------------------

        for column in self.columns:
            profile = self.column_profiles[column]

            for value in chunk[column]:
                profile.update(value)

        # ---------------------------------------------
        # Semantic fields
        # ---------------------------------------------

        name_column = self.detected_columns.get("name")
        address_column = self.detected_columns.get("address")
        country_column = self.detected_columns.get("country")
        id_column = self.detected_columns.get("id")

        # Country
        if country_column:
            for value in chunk[country_column]:
                value = clean_string(value)

                if value:
                    self.country_counts[
                        normalize_for_analysis(value)
                    ] += 1

        # IDs
        if id_column:
            for value in chunk[id_column]:
                value = clean_string(value)

                if not value:
                    continue

                if value in self.ids:
                    self.duplicate_id_count += 1
                else:
                    self.ids.add(value)

        # Names
        if name_column:
            if self.name_profiles is None:
                self.name_profiles = FieldProfile()

            for value in chunk[name_column]:
                self._process_name(value)

        # Addresses
        if address_column:
            if self.address_profiles is None:
                self.address_profiles = FieldProfile()

            for value in chunk[address_column]:
                self._process_address(value)

    # --------------------------------------------------------
    # NAME ANALYSIS
    # --------------------------------------------------------

    def _process_name(self, value: Any) -> None:
        value = clean_string(value)

        self.name_profiles.update(value)

        if not value:
            return

        normalized = normalize_for_analysis(value)
        compact = compact_for_analysis(value)

        self.name_compact_to_raw[compact].add(value)

        # Legal suffix / private limited patterns
        upper = value.upper()

        if (
            "PRIVATE LIMITED" in upper
            or "PVT LTD" in upper
            or "PVT. LTD" in upper
            or "PVT. LTD." in upper
            or "PRIVATE LTD" in upper
            or "LIMITED" in upper
        ):
            self.special_examples[
                "private_limited_variants"
            ].add(value)

        # Punctuation
        if re.search(r"[.,;:/()&'\-]", value):
            self.special_examples[
                "punctuation"
            ].add(value)

        # Mixed case
        if not value.isupper() and not value.islower():
            self.special_examples[
                "mixed_case"
            ].add(value)

        # Numbers
        if re.search(r"\d", value):
            self.special_examples[
                "numbers"
            ].add(value)

    # --------------------------------------------------------
    # ADDRESS ANALYSIS
    # --------------------------------------------------------

    def _process_address(self, value: Any) -> None:
        value = clean_string(value)

        self.address_profiles.update(value)

        if not value:
            return

        compact = compact_for_analysis(value)

        self.address_compact_to_raw[compact].add(value)

        # Road/street abbreviations
        upper = value.upper()

        if (
            re.search(r"\bROAD\b", upper)
            or re.search(r"\bRD\b", upper)
            or re.search(r"\bR\.D\.\b", upper)
            or re.search(r"\bSTREET\b", upper)
            or re.search(r"\bST\b", upper)
            or re.search(r"\bLANE\b", upper)
            or re.search(r"\bLN\b", upper)
        ):
            self.special_examples[
                "road_variants"
            ].add(value)

        if re.search(r"[.,;:/()&'\-]", value):
            self.special_examples[
                "punctuation"
            ].add(value)

        if re.search(r"\d", value):
            self.special_examples[
                "numbers"
            ].add(value)

        if extract_postal_candidates(value):
            self.special_examples[
                "postal_codes"
            ].add(value)

    # --------------------------------------------------------
    # BUILD RESULT
    # --------------------------------------------------------

    def _build_result(self) -> dict[str, Any]:

        name_collisions = {
            key: sorted(values)[:20]
            for key, values in self.name_compact_to_raw.items()
            if len(values) > 1
        }

        address_collisions = {
            key: sorted(values)[:20]
            for key, values in self.address_compact_to_raw.items()
            if len(values) > 1
        }

        result = {
            "dataset": self.dataset_name,
            "file": str(self.file_path),
            "exists": True,
            "rows": self.rows,
            "columns": self.columns,
            "detected_columns": self.detected_columns,

            "unique_ids": len(self.ids),
            "duplicate_id_count": self.duplicate_id_count,

            "countries": self.country_counts.most_common(100),

            "columns_profile": {
                column: profile.summary()
                for column, profile in self.column_profiles.items()
            },

            "name_profile": (
                self.name_profiles.summary()
                if self.name_profiles
                else None
            ),

            "address_profile": (
                self.address_profiles.summary()
                if self.address_profiles
                else None
            ),

            "name_compact_collisions": {
                "number_of_collision_groups": len(name_collisions),
                "examples": dict(
                    list(name_collisions.items())[:100]
                ),
            },

            "address_compact_collisions": {
                "number_of_collision_groups": len(address_collisions),
                "examples": dict(
                    list(address_collisions.items())[:100]
                ),
            },

            "special_examples": {
                key: sampler.get()
                for key, sampler in self.special_examples.items()
            },
        }

        return result


# ============================================================
# GROUND TRUTH ANALYSIS
# ============================================================

class GroundTruthProfiler:
    """
    Profile train_ground_truth.tsv using its actual challenge schema:

        source1_entity_id
        matched_entity_ids

    matched_entity_ids is a comma-separated list containing S2-/S3- IDs.
    A Source-1 entity can therefore have zero, one, or many matches.

    The profiler computes the distribution directly from each row instead
    of incorrectly assuming separate Source-2 / Source-3 columns.
    """

    def __init__(
        self,
        file_path: Path,
        source1_files: list[Path],
        chunk_size: int = CHUNK_SIZE,
    ):
        self.file_path = file_path
        self.source1_files = source1_files
        self.chunk_size = chunk_size

        self.rows = 0
        self.columns: list[str] = []

        self.source1_column = "source1_entity_id"
        self.matches_column = "matched_entity_ids"

        self.unique_source1_ids: set[str] = set()
        self.referenced_s2_ids: set[str] = set()
        self.referenced_s3_ids: set[str] = set()

        self.match_distribution: Counter[int] = Counter()
        self.s2_distribution: Counter[int] = Counter()
        self.s3_distribution: Counter[int] = Counter()

        self.one_to_many_examples: dict[str, list[str]] = {}

        self.empty_match_rows = 0
        self.duplicate_source1_rows = 0
        self.duplicate_match_ids = 0
        self.empty_match_tokens = 0
        self.invalid_source1_ids = 0
        self.unknown_match_ids = 0
        self.total_matched_ids = 0
        self.source1_with_both = 0

    def run(self) -> dict[str, Any]:
        print()
        print("=" * 80)
        print("Profiling: train_ground_truth")
        print("=" * 80)

        if not self.file_path.exists():
            print(f"WARNING: Ground truth not found: {self.file_path}")
            return {"exists": False, "file": str(self.file_path)}

        try:
            reader = pd.read_csv(
                self.file_path,
                sep="\t",
                chunksize=self.chunk_size,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
                on_bad_lines="warn",
            )

            first_chunk = True
            for chunk_number, chunk in enumerate(reader, start=1):
                if first_chunk:
                    self.columns = list(chunk.columns)
                    if self.source1_column not in self.columns or self.matches_column not in self.columns:
                        raise ValueError(
                            "Expected ground-truth columns "
                            "source1_entity_id and matched_entity_ids; "
                            f"found {self.columns}"
                        )
                    print(f"Ground-truth columns: {self.columns}")
                    first_chunk = False

                self._process_chunk(chunk, chunk_number)

        except Exception as exc:
            return {
                "exists": True,
                "file": str(self.file_path),
                "error": f"{type(exc).__name__}: {exc}",
            }

        validation = self._validate_references()

        matched_rows = self.rows - self.empty_match_rows
        one_to_many_rows = sum(
            count for matches, count in self.match_distribution.items() if matches > 1
        )
        exactly_one_rows = self.match_distribution.get(1, 0)

        s1_with_s2 = sum(
            count for s2_count, count in self.s2_distribution.items() if s2_count > 0
        )
        s1_with_s3 = sum(
            count for s3_count, count in self.s3_distribution.items() if s3_count > 0
        )
        s1_with_both = self.source1_with_both

        result = {
            "exists": True,
            "file": str(self.file_path),
            "rows": self.rows,
            "columns": self.columns,
            "source1_column": self.source1_column,
            "matched_ids_column": self.matches_column,
            "unique_source1_ids": len(self.unique_source1_ids),
            "duplicate_source1_rows": self.duplicate_source1_rows,
            "invalid_source1_ids": self.invalid_source1_ids,
            "unmatched_source1_rows": self.empty_match_rows,
            "matched_source1_rows": matched_rows,
            "exactly_one_match": exactly_one_rows,
            "one_to_many_rows": one_to_many_rows,
            "total_matched_ids": self.total_matched_ids,
            "s2_matched_ids": len(self.referenced_s2_ids),
            "s3_matched_ids": len(self.referenced_s3_ids),
            "unknown_match_ids": self.unknown_match_ids,
            "duplicate_match_ids": self.duplicate_match_ids,
            "empty_match_tokens": self.empty_match_tokens,
            "source1_with_s2": s1_with_s2,
            "source1_with_s3": s1_with_s3,
            "source1_with_both": s1_with_both,
            "source1_match_distribution": dict(sorted(self.match_distribution.items())),
            "source1_s2_match_distribution": dict(sorted(self.s2_distribution.items())),
            "source1_s3_match_distribution": dict(sorted(self.s3_distribution.items())),
            "one_to_many_examples": self.one_to_many_examples,
            "reference_validation": validation,
        }

        print(f"Total GT rows              : {self.rows:,}")
        print(f"Unique Source-1 IDs        : {len(self.unique_source1_ids):,}")
        print(f"Duplicate Source-1 IDs     : {self.duplicate_source1_rows:,}")
        print(f"Unmatched Source-1 rows    : {self.empty_match_rows:,}")
        print(f"Exactly one match          : {exactly_one_rows:,}")
        print(f"One-to-many rows           : {one_to_many_rows:,}")
        print(f"Total matched IDs          : {result['total_matched_ids']:,}")
        print(f"S2 matched IDs             : {result['s2_matched_ids']:,}")
        print(f"S3 matched IDs             : {result['s3_matched_ids']:,}")
        print(f"Unknown match IDs          : {self.unknown_match_ids:,}")
        print(f"S1 with S2                 : {s1_with_s2:,}")
        print(f"S1 with S3                 : {s1_with_s3:,}")
        print(f"S1 with both               : {s1_with_both:,}")

        return result

    def _process_chunk(self, chunk: pd.DataFrame, chunk_number: int) -> None:
        self.rows += len(chunk)

        for source1, matched in zip(
            chunk[self.source1_column],
            chunk[self.matches_column],
        ):
            source1 = clean_string(source1)
            matched = clean_string(matched)

            if source1 in self.unique_source1_ids:
                self.duplicate_source1_rows += 1
            else:
                self.unique_source1_ids.add(source1)

            if not source1 or not source1.startswith("S1-"):
                self.invalid_source1_ids += 1

            if not matched:
                self.empty_match_rows += 1
                self.match_distribution[0] += 1
                self.s2_distribution[0] += 1
                self.s3_distribution[0] += 1
                continue

            raw_tokens = matched.split(",")
            tokens = [token.strip() for token in raw_tokens]

            if any(not token for token in tokens):
                self.empty_match_tokens += sum(not token for token in tokens)

            valid_tokens = [token for token in tokens if token]
            unique_tokens = list(dict.fromkeys(valid_tokens))
            self.duplicate_match_ids += len(valid_tokens) - len(unique_tokens)

            total_count = len(unique_tokens)
            self.total_matched_ids += total_count
            s2_count = 0
            s3_count = 0

            for token in unique_tokens:
                if token.startswith("S2-"):
                    s2_count += 1
                    self.referenced_s2_ids.add(token)
                elif token.startswith("S3-"):
                    s3_count += 1
                    self.referenced_s3_ids.add(token)
                else:
                    self.unknown_match_ids += 1

            self.match_distribution[total_count] += 1
            self.s2_distribution[s2_count] += 1
            self.s3_distribution[s3_count] += 1
            if s2_count > 0 and s3_count > 0:
                self.source1_with_both += 1

            if total_count > 1 and len(self.one_to_many_examples) < 30:
                self.one_to_many_examples[source1] = unique_tokens

        print(
            f"  ground truth rows processed: {self.rows:,}",
            end="\r",
        )

    def _validate_references(self) -> dict[str, Any]:
        # Validate referenced IDs against the actual training source files.
        expected = {
            "S1": self.unique_source1_ids,
            "S2": self.referenced_s2_ids,
            "S3": self.referenced_s3_ids,
        }
        found = {"S1": set(), "S2": set(), "S3": set()}

        source_files = {
            "S1": self.source1_files[0],
            "S2": self.source1_files[1],
            "S3": self.source1_files[2],
        }

        for prefix, path in source_files.items():
            if not path.exists():
                return {"status": "ERROR", "error": f"Missing source file: {path}"}

            reader = pd.read_csv(
                path,
                sep="\t",
                chunksize=self.chunk_size,
                dtype=str,
                keep_default_na=False,
                na_filter=False,
                usecols=["entity_id"],
            )

            wanted = expected[prefix]
            for chunk in reader:
                for value in chunk["entity_id"]:
                    value = clean_string(value)
                    if value in wanted:
                        found[prefix].add(value)

        missing = {
            prefix: len(expected[prefix] - found[prefix])
            for prefix in expected
        }
        found_counts = {prefix: len(found[prefix]) for prefix in found}

        status = "PASS" if all(value == 0 for value in missing.values()) else "FAIL"
        return {
            "status": status,
            "missing_referenced_ids": missing,
            "found_referenced_ids": found_counts,
        }

# ============================================================
# JSON SERIALIZATION
# ============================================================

def make_json_serializable(value: Any) -> Any:
    """Convert Counters, sets and Path objects into JSON-safe types."""

    if isinstance(value, Counter):
        return dict(value)

    if isinstance(value, set):
        return sorted(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): make_json_serializable(val)
            for key, val in value.items()
        }

    if isinstance(value, list):
        return [
            make_json_serializable(item)
            for item in value
        ]

    return value


# ============================================================
# MARKDOWN REPORT
# ============================================================

def format_number(value: Any) -> str:
    if isinstance(value, (int, float)):
        return f"{value:,}"

    return str(value)


def markdown_table(
    rows: list[list[Any]],
    headers: list[str],
) -> str:

    output = []

    output.append(
        "| "
        + " | ".join(headers)
        + " |"
    )

    output.append(
        "| "
        + " | ".join(["---"] * len(headers))
        + " |"
    )

    for row in rows:
        output.append(
            "| "
            + " | ".join(
                str(value).replace("|", "\\|")
                for value in row
            )
            + " |"
        )

    return "\n".join(output)


def build_markdown_report(
    profiles: dict[str, Any],
    ground_truth: dict[str, Any],
) -> str:

    lines: list[str] = []

    lines.append("# Amazon ML Challenge 2026 — Dataset Profile")
    lines.append("")
    lines.append(
        "Generated by `preprocessing/profiling.py`."
    )
    lines.append("")
    lines.append(
        "This report describes the raw challenge data before "
        "production normalization. Statistics are computed "
        "using streaming/chunk-based processing."
    )
    lines.append("")

    # ========================================================
    # DATASET OVERVIEW
    # ========================================================

    lines.append("## 1. Dataset Overview")
    lines.append("")

    overview_rows = []

    for dataset_name, profile in profiles.items():

        if not profile.get("exists"):
            overview_rows.append(
                [
                    dataset_name,
                    "MISSING",
                    "-",
                    "-",
                    "-",
                ]
            )
            continue

        overview_rows.append(
            [
                dataset_name,
                format_number(profile.get("rows", 0)),
                len(profile.get("columns", [])),
                profile.get(
                    "detected_columns", {}
                ).get("name", "-"),
                profile.get(
                    "detected_columns", {}
                ).get("address", "-"),
            ]
        )

    lines.append(
        markdown_table(
            overview_rows,
            [
                "Dataset",
                "Rows",
                "Columns",
                "Name column",
                "Address column",
            ],
        )
    )

    lines.append("")

    # ========================================================
    # PER DATASET
    # ========================================================

    for dataset_name, profile in profiles.items():

        if not profile.get("exists"):
            continue

        lines.append(
            f"## 2. `{dataset_name}`"
        )
        lines.append("")

        lines.append(
            f"**Rows:** "
            f"{format_number(profile.get('rows', 0))}"
        )

        lines.append("")

        lines.append(
            f"**Columns:** "
            f"{', '.join(profile.get('columns', []))}"
        )

        lines.append("")

        lines.append("### Detected semantic columns")
        lines.append("")

        detected = profile.get(
            "detected_columns",
            {},
        )

        detected_rows = [
            [key, value or "Not detected"]
            for key, value in detected.items()
        ]

        lines.append(
            markdown_table(
                detected_rows,
                ["Field", "Column"],
            )
        )

        lines.append("")

        # ---------------------------------------------
        # Column profile
        # ---------------------------------------------

        lines.append("### Column statistics")
        lines.append("")

        rows = []

        for column, stats in profile.get(
            "columns_profile",
            {},
        ).items():

            rows.append(
                [
                    column,
                    format_number(
                        stats.get("total", 0)
                    ),
                    format_number(
                        stats.get("missing", 0)
                    ),
                    f"{stats.get('missing_pct', 0):.2f}%",
                    format_number(
                        stats.get(
                            "unique_normalized",
                            0,
                        )
                    ),
                    f"{stats.get('mean_length', 0):.2f}",
                    format_number(
                        stats.get(
                            "median_length",
                            0,
                        )
                    ),
                    format_number(
                        stats.get(
                            "max_length",
                            0,
                        )
                    ),
                ]
            )

        lines.append(
            markdown_table(
                rows,
                [
                    "Column",
                    "Rows",
                    "Missing",
                    "Missing %",
                    "Unique",
                    "Mean len",
                    "Median len",
                    "Max len",
                ],
            )
        )

        lines.append("")

        # ---------------------------------------------
        # Country
        # ---------------------------------------------

        countries = profile.get("countries", [])

        if countries:

            lines.append("### Country distribution")
            lines.append("")

            country_rows = [
                [
                    country,
                    format_number(count),
                ]
                for country, count
                in countries[:30]
            ]

            lines.append(
                markdown_table(
                    country_rows,
                    ["Country", "Count"],
                )
            )

            lines.append("")

        # ---------------------------------------------
        # Name
        # ---------------------------------------------

        name_profile = profile.get(
            "name_profile"
        )

        if name_profile:

            lines.append("### Name analysis")
            lines.append("")

            lines.append(
                f"- Unique normalized names: "
                f"{format_number(name_profile.get('unique_normalized', 0))}"
            )

            lines.append(
                f"- Normalized values occurring more than once: "
                f"{format_number(name_profile.get('repeated_normalized_values', 0))}"
            )

            lines.append(
                f"- Mean name length: "
                f"{name_profile.get('mean_length', 0)}"
            )

            lines.append(
                f"- Median name length: "
                f"{name_profile.get('median_length', 0)}"
            )

            lines.append(
                f"- Maximum name length: "
                f"{name_profile.get('max_length', 0)}"
            )

            lines.append("")

            lines.append("#### Common name tokens")
            lines.append("")

            token_rows = [
                [
                    token,
                    format_number(count),
                ]
                for token, count
                in name_profile.get(
                    "top_tokens",
                    [],
                )[:30]
            ]

            lines.append(
                markdown_table(
                    token_rows,
                    ["Token", "Count"],
                )
            )

            lines.append("")

            lines.append(
                "#### Case distribution"
            )
            lines.append("")

            case_rows = [
                [
                    pattern,
                    format_number(count),
                ]
                for pattern, count
                in name_profile.get(
                    "case_patterns",
                    {},
                ).items()
            ]

            lines.append(
                markdown_table(
                    case_rows,
                    ["Case pattern", "Count"],
                )
            )

            lines.append("")

        # ---------------------------------------------
        # Address
        # ---------------------------------------------

        address_profile = profile.get(
            "address_profile"
        )

        if address_profile:

            lines.append("### Address analysis")
            lines.append("")

            lines.append(
                f"- Unique normalized addresses: "
                f"{format_number(address_profile.get('unique_normalized', 0))}"
            )

            lines.append(
                f"- Duplicate normalized addresses: "
                f"{format_number(address_profile.get('repeated_normalized_values', 0))}"
            )

            lines.append(
                f"- Mean address length: "
                f"{address_profile.get('mean_length', 0)}"
            )

            lines.append(
                f"- Median address length: "
                f"{address_profile.get('median_length', 0)}"
            )

            lines.append(
                f"- Maximum address length: "
                f"{address_profile.get('max_length', 0)}"
            )

            lines.append("")

            lines.append(
                "#### Common address tokens"
            )
            lines.append("")

            token_rows = [
                [
                    token,
                    format_number(count),
                ]
                for token, count
                in address_profile.get(
                    "top_tokens",
                    [],
                )[:40]
            ]

            lines.append(
                markdown_table(
                    token_rows,
                    ["Token", "Count"],
                )
            )

            lines.append("")

        # ---------------------------------------------
        # Noise examples
        # ---------------------------------------------

        lines.append("### Noise examples")
        lines.append("")

        special_examples = profile.get(
            "special_examples",
            {},
        )

        for category, examples in special_examples.items():

            title = category.replace(
                "_",
                " ",
            ).title()

            lines.append(
                f"#### {title}"
            )
            lines.append("")

            if examples:
                for example in examples[:30]:
                    lines.append(
                        f"- `{example}`"
                    )
            else:
                lines.append(
                    "- No examples observed."
                )

            lines.append("")

        # ---------------------------------------------
        # Collision analysis
        # ---------------------------------------------

        lines.append(
            "### Exploratory normalization collisions"
        )
        lines.append("")

        name_collisions = profile.get(
            "name_compact_collisions",
            {},
        )

        lines.append(
            f"Compact-name collision groups: "
            f"{format_number(name_collisions.get('number_of_collision_groups', 0))}"
        )

        lines.append("")

        for compact, values in list(
            name_collisions.get(
                "examples",
                {},
            ).items()
        )[:20]:

            lines.append(
                f"- `{compact}` → "
                + ", ".join(
                    f"`{value}`"
                    for value in values
                )
            )

        lines.append("")

        address_collisions = profile.get(
            "address_compact_collisions",
            {},
        )

        lines.append(
            f"Compact-address collision groups: "
            f"{format_number(address_collisions.get('number_of_collision_groups', 0))}"
        )

        lines.append("")

        for compact, values in list(
            address_collisions.get(
                "examples",
                {},
            ).items()
        )[:20]:

            lines.append(
                f"- `{compact}` → "
                + ", ".join(
                    f"`{value}`"
                    for value in values
                )
            )

        lines.append("")

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    lines.append("## 3. Ground Truth Analysis")
    lines.append("")

    if not ground_truth.get("exists"):
        lines.append("Ground-truth file was not available.")
        lines.append("")
    elif ground_truth.get("error"):
        lines.append(f"Error: `{ground_truth['error']}`")
        lines.append("")
    else:
        lines.append(f"- Total ground-truth rows: {format_number(ground_truth.get('rows', 0))}")
        lines.append(f"- Unique Source-1 IDs: {format_number(ground_truth.get('unique_source1_ids', 0))}")
        lines.append(f"- Duplicate Source-1 rows: {format_number(ground_truth.get('duplicate_source1_rows', 0))}")
        lines.append(f"- Unmatched Source-1 rows: {format_number(ground_truth.get('unmatched_source1_rows', 0))}")
        lines.append(f"- Exactly one match: {format_number(ground_truth.get('exactly_one_match', 0))}")
        lines.append(f"- One-to-many rows: {format_number(ground_truth.get('one_to_many_rows', 0))}")
        lines.append(f"- Total matched IDs: {format_number(ground_truth.get('total_matched_ids', 0))}")
        lines.append(f"- Source-2 matched IDs: {format_number(ground_truth.get('s2_matched_ids', 0))}")
        lines.append(f"- Source-3 matched IDs: {format_number(ground_truth.get('s3_matched_ids', 0))}")
        lines.append(f"- Source-1 with Source-2: {format_number(ground_truth.get('source1_with_s2', 0))}")
        lines.append(f"- Source-1 with Source-3: {format_number(ground_truth.get('source1_with_s3', 0))}")
        lines.append(f"- Source-1 with both: {format_number(ground_truth.get('source1_with_both', 0))}")
        lines.append("")

        lines.append("### Source-1 match distribution")
        lines.append("")
        distribution = ground_truth.get("source1_match_distribution", {})
        lines.append(markdown_table(
            [[matches, format_number(count)] for matches, count in sorted(distribution.items())],
            ["Total matches for Source-1", "Source-1 rows"],
        ))
        lines.append("")

        lines.append("### Source-2 matches per Source-1")
        lines.append("")
        s2_distribution = ground_truth.get("source1_s2_match_distribution", {})
        lines.append(markdown_table(
            [[matches, format_number(count)] for matches, count in sorted(s2_distribution.items())],
            ["Source-2 matches", "Source-1 rows"],
        ))
        lines.append("")

        lines.append("### Source-3 matches per Source-1")
        lines.append("")
        s3_distribution = ground_truth.get("source1_s3_match_distribution", {})
        lines.append(markdown_table(
            [[matches, format_number(count)] for matches, count in sorted(s3_distribution.items())],
            ["Source-3 matches", "Source-1 rows"],
        ))
        lines.append("")

        lines.append("### One-to-many examples")
        lines.append("")
        examples = ground_truth.get("one_to_many_examples", {})
        if examples:
            for source1, matches in examples.items():
                lines.append(f"- Source-1 `{source1}` → " + ", ".join(f"`{match}`" for match in matches))
        else:
            lines.append("- No one-to-many relationships observed.")
        lines.append("")

        validation = ground_truth.get("reference_validation", {})
        lines.append("### Ground-truth reference validation")
        lines.append("")
        lines.append(f"- Status: **{validation.get('status', 'UNKNOWN')}**")
        missing = validation.get("missing_referenced_ids", {})
        found = validation.get("found_referenced_ids", {})
        for prefix in ["S1", "S2", "S3"]:
            lines.append(
                f"- {prefix}: found {format_number(found.get(prefix, 0))}, "
                f"missing {format_number(missing.get(prefix, 0))}"
            )
        lines.append("")

        integrity = [
            ("Invalid Source-1 IDs", ground_truth.get("invalid_source1_ids", 0)),
            ("Unknown match-ID prefixes", ground_truth.get("unknown_match_ids", 0)),
            ("Duplicate match IDs within rows", ground_truth.get("duplicate_match_ids", 0)),
            ("Empty match tokens", ground_truth.get("empty_match_tokens", 0)),
        ]
        lines.append("### Ground-truth integrity checks")
        lines.append("")
        lines.append(markdown_table(
            [[name, format_number(value), "PASS" if value == 0 else "CHECK"] for name, value in integrity],
            ["Check", "Count", "Status"],
        ))
        lines.append("")

    # ========================================================
    # NORMALIZATION GUIDANCE
    # ========================================================

    lines.append(
        "## 4. Normalization Design Notes"
    )
    lines.append("")

    lines.append(
        "The profiler intentionally performs only lightweight "
        "exploratory normalization. These transformations should "
        "not automatically become production normalization rules."
    )

    lines.append("")

    lines.append(
        "Production preprocessing should:"
    )

    lines.append(
        "- preserve every original raw field;"
    )
    lines.append(
        "- generate multiple representations instead of replacing "
        "the original value;"
    )
    lines.append(
        "- normalize case and whitespace consistently;"
    )
    lines.append(
        "- standardize only abbreviations supported by observed "
        "dataset noise;"
    )
    lines.append(
        "- preserve meaningful geographic information;"
    )
    lines.append(
        "- preserve numeric/address components;"
    )
    lines.append(
        "- avoid collapsing distinct businesses merely because "
        "their normalized names are identical;"
    )
    lines.append(
        "- validate normalization changes through collision analysis."
    )

    lines.append("")

    lines.append(
        "### Important precision constraint"
    )

    lines.append("")

    lines.append(
        "The challenge uses macro F0.5, so false-positive entity "
        "merges are particularly costly. Aggressive normalization "
        "should therefore be validated against collision patterns "
        "before being used for matching."
    )

    lines.append("")

    lines.append(
        "Example: `ABC Hospital Mumbai` and "
        "`ABC Hospital Pune` should not become the same entity "
        "simply because `Mumbai` and `Pune` were removed."
    )

    lines.append("")

    return "\n".join(lines)


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    data: Any,
    output_path: Path,
) -> None:

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            make_json_serializable(data),
            file,
            indent=2,
            ensure_ascii=False,
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print()
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026")
    print("BUSINESS ENTITY RESOLUTION — DATASET PROFILER")
    print("=" * 80)

    print()
    print(f"Repository root : {ROOT_DIR}")
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Reports directory: {REPORT_DIR}")
    print(f"Chunk size       : {CHUNK_SIZE:,}")

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_profiles: dict[str, Any] = {}

    # ========================================================
    # PROFILE ALL DATASETS
    # ========================================================

    for dataset_name, file_path in SOURCE_FILES.items():

        if dataset_name == "train_ground_truth":
            continue

        profiler = DatasetProfiler(
            file_path=file_path,
            dataset_name=dataset_name,
            chunk_size=CHUNK_SIZE,
        )

        all_profiles[
            dataset_name
        ] = profiler.run()

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    ground_truth_profiler = GroundTruthProfiler(
        file_path=SOURCE_FILES["train_ground_truth"],
        source1_files=[
            SOURCE_FILES["train_source1"],
            SOURCE_FILES["train_source2"],
            SOURCE_FILES["train_source3"],
        ],
        chunk_size=CHUNK_SIZE,
    )

    ground_truth = ground_truth_profiler.run()

    # ========================================================
    # SAVE RAW JSON
    # ========================================================

    json_path = (
        REPORT_DIR
        / "profile_raw_stats.json"
    )

    save_json(
        {
            "datasets": all_profiles,
            "ground_truth": ground_truth,
        },
        json_path,
    )

    # ========================================================
    # SAVE EXAMPLES SEPARATELY
    # ========================================================

    examples = {}

    for dataset_name, profile in all_profiles.items():

        if not profile.get("exists"):
            continue

        examples[dataset_name] = {
            "special_examples": profile.get(
                "special_examples",
                {},
            ),
            "name_collision_examples": profile.get(
                "name_compact_collisions",
                {},
            ).get(
                "examples",
                {},
            ),
            "address_collision_examples": profile.get(
                "address_compact_collisions",
                {},
            ).get(
                "examples",
                {},
            ),
        }

    examples_path = (
        REPORT_DIR
        / "profile_examples.json"
    )

    save_json(
        examples,
        examples_path,
    )

    # ========================================================
    # BUILD MARKDOWN
    # ========================================================

    markdown = build_markdown_report(
        profiles=all_profiles,
        ground_truth=ground_truth,
    )

    report_path = (
        REPORT_DIR
        / "dataset_profile.md"
    )

    report_path.write_text(
        markdown,
        encoding="utf-8",
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print()
    print()
    print("=" * 80)
    print("PROFILING COMPLETE")
    print("=" * 80)

    print()
    print("Generated:")
    print(f"  {report_path}")
    print(f"  {json_path}")
    print(f"  {examples_path}")

    print()
    print(
        "Next step: inspect dataset_profile.md before "
        "deciding production normalization rules."
    )


if __name__ == "__main__":
    main()