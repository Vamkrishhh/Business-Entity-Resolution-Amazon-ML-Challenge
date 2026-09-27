from pathlib import Path
from collections import Counter
import json

import pandas as pd


# ============================================================
# Paths
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT_DIR / "dataset" / "train"
REPORT_DIR = ROOT_DIR / "reports"

GROUND_TRUTH_FILE = TRAIN_DIR / "train_ground_truth.tsv"

SOURCE_FILES = {
    "S1": TRAIN_DIR / "train_source1.tsv",
    "S2": TRAIN_DIR / "train_source2.tsv",
    "S3": TRAIN_DIR / "train_source3.tsv",
}

CHUNK_SIZE = 100_000
MAX_EXAMPLES = 25


# ============================================================
# Helpers
# ============================================================

def add_example(container, value):
    """Keep only the first few deterministic examples."""
    if len(container) < MAX_EXAMPLES:
        container.append(value)


def parse_match_ids(value):
    """
    Parse matched_entity_ids.

    Expected:
        S2-123,S3-456

    Empty values mean no matches.
    """
    if value is None:
        return []

    value = str(value).strip()

    if not value:
        return []

    parts = value.split(",")

    result = []

    for part in parts:
        part = part.strip()

        if part:
            result.append(part)

    return result


def classify_match_id(match_id):
    """Return S2, S3, or UNKNOWN."""
    if match_id.startswith("S2-"):
        return "S2"

    if match_id.startswith("S3-"):
        return "S3"

    return "UNKNOWN"


# ============================================================
# Main ground-truth profiling
# ============================================================

def profile_ground_truth():

    print("=" * 70)
    print("GROUND TRUTH PROFILER")
    print("=" * 70)

    print(f"Repository root : {ROOT_DIR}")
    print(f"Ground truth    : {GROUND_TRUTH_FILE}")
    print()

    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(
            f"Ground truth file not found:\n{GROUND_TRUTH_FILE}"
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total_rows = 0

    seen_s1_ids = set()
    duplicate_s1_ids = set()

    invalid_s1_ids = 0

    empty_match_rows = 0
    exactly_one_match_rows = 0
    multiple_match_rows = 0

    total_match_ids = 0

    s2_match_ids = 0
    s3_match_ids = 0
    unknown_match_ids = 0

    s1_with_s2 = 0
    s1_with_s3 = 0
    s1_with_both = 0

    duplicate_match_id_rows = 0
    empty_tokens = 0

    total_match_distribution = Counter()
    s2_match_distribution = Counter()
    s3_match_distribution = Counter()

    # IDs referenced by the ground truth.
    #
    # IMPORTANT:
    # We store only IDs actually referenced by GT,
    # NOT every ID from the 3 huge source files.
    referenced_s1_ids = set()
    referenced_s2_ids = set()
    referenced_s3_ids = set()

    # Examples
    examples = {
        "empty_matches": [],
        "exactly_one_match": [],
        "multiple_matches": [],
        "s1_with_both_sources": [],
        "duplicate_match_ids": [],
        "unknown_match_ids": [],
        "invalid_s1_ids": [],
        "empty_tokens": [],
    }

    # --------------------------------------------------------
    # Read ground truth
    # --------------------------------------------------------

    usecols = [
        "source1_entity_id",
        "matched_entity_ids",
    ]

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            GROUND_TRUTH_FILE,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            usecols=usecols,
            chunksize=CHUNK_SIZE,
        ),
        start=1,
    ):

        for s1_id, matched_value in zip(
            chunk["source1_entity_id"],
            chunk["matched_entity_ids"],
        ):

            total_rows += 1

            s1_id = str(s1_id).strip()

            # ------------------------------------------------
            # Source 1 ID validation
            # ------------------------------------------------

            if not s1_id.startswith("S1-"):
                invalid_s1_ids += 1

                add_example(
                    examples["invalid_s1_ids"],
                    s1_id,
                )

            if s1_id in seen_s1_ids:
                duplicate_s1_ids.add(s1_id)

            seen_s1_ids.add(s1_id)
            referenced_s1_ids.add(s1_id)

            # ------------------------------------------------
            # Parse matches
            # ------------------------------------------------

            raw_value = str(matched_value).strip()

            if not raw_value:
                match_ids = []

            else:
                raw_parts = raw_value.split(",")

                match_ids = []

                for raw_part in raw_parts:

                    cleaned = raw_part.strip()

                    if not cleaned:
                        empty_tokens += 1

                        add_example(
                            examples["empty_tokens"],
                            {
                                "source1_entity_id": s1_id,
                                "raw_value": raw_value,
                            },
                        )

                    else:
                        match_ids.append(cleaned)

            # ------------------------------------------------
            # Match count
            # ------------------------------------------------

            match_count = len(match_ids)

            total_match_distribution[match_count] += 1
            total_match_ids += match_count

            if match_count == 0:

                empty_match_rows += 1

                add_example(
                    examples["empty_matches"],
                    s1_id,
                )

            elif match_count == 1:

                exactly_one_match_rows += 1

                add_example(
                    examples["exactly_one_match"],
                    {
                        "source1_entity_id": s1_id,
                        "matched_entity_ids": match_ids,
                    },
                )

            else:

                multiple_match_rows += 1

                add_example(
                    examples["multiple_matches"],
                    {
                        "source1_entity_id": s1_id,
                        "matched_entity_ids": match_ids,
                    },
                )

            # ------------------------------------------------
            # Duplicate match IDs inside a single row
            # ------------------------------------------------

            if len(match_ids) != len(set(match_ids)):

                duplicate_match_id_rows += 1

                add_example(
                    examples["duplicate_match_ids"],
                    {
                        "source1_entity_id": s1_id,
                        "matched_entity_ids": match_ids,
                    },
                )

            # ------------------------------------------------
            # Classify S2 / S3
            # ------------------------------------------------

            row_s2 = []
            row_s3 = []
            row_unknown = []

            for match_id in match_ids:

                source_type = classify_match_id(match_id)

                if source_type == "S2":

                    s2_match_ids += 1
                    referenced_s2_ids.add(match_id)
                    row_s2.append(match_id)

                elif source_type == "S3":

                    s3_match_ids += 1
                    referenced_s3_ids.add(match_id)
                    row_s3.append(match_id)

                else:

                    unknown_match_ids += 1
                    row_unknown.append(match_id)

                    add_example(
                        examples["unknown_match_ids"],
                        {
                            "source1_entity_id": s1_id,
                            "match_id": match_id,
                        },
                    )

            # ------------------------------------------------
            # Per-source distributions
            # ------------------------------------------------

            s2_match_distribution[len(row_s2)] += 1
            s3_match_distribution[len(row_s3)] += 1

            if row_s2:
                s1_with_s2 += 1

            if row_s3:
                s1_with_s3 += 1

            if row_s2 and row_s3:

                s1_with_both += 1

                add_example(
                    examples["s1_with_both_sources"],
                    {
                        "source1_entity_id": s1_id,
                        "S2_matches": row_s2,
                        "S3_matches": row_s3,
                    },
                )

        print(
            f"Processed GT chunk {chunk_number} "
            f"| rows: {total_rows:,}"
        )

    # ========================================================
    # Source membership validation
    # ========================================================

    print()
    print("=" * 70)
    print("VALIDATING REFERENCED ENTITY IDs")
    print("=" * 70)

    referenced_sets = {
        "S1": referenced_s1_ids,
        "S2": referenced_s2_ids,
        "S3": referenced_s3_ids,
    }

    missing_ids = {}

    for source_name, source_file in SOURCE_FILES.items():

        print()
        print(f"Scanning {source_name}: {source_file.name}")

        if not source_file.exists():
            raise FileNotFoundError(
                f"Source file not found:\n{source_file}"
            )

        remaining = set(referenced_sets[source_name])

        print(
            f"GT-referenced {source_name} IDs: "
            f"{len(remaining):,}"
        )

        for chunk_number, chunk in enumerate(
            pd.read_csv(
                source_file,
                sep="\t",
                dtype={"entity_id": str},
                usecols=["entity_id"],
                keep_default_na=False,
                na_filter=False,
                chunksize=CHUNK_SIZE,
            ),
            start=1,
        ):

            source_ids = set(
                value.strip()
                for value in chunk["entity_id"]
                if value
            )

            remaining.difference_update(source_ids)

            if not remaining:
                print(
                    f"All referenced {source_name} IDs found."
                )
                break

            if chunk_number % 10 == 0:
                print(
                    f"  scanned chunks: {chunk_number} "
                    f"| remaining: {len(remaining):,}"
                )

        missing_ids[source_name] = remaining

        print(
            f"Missing {source_name} references: "
            f"{len(remaining):,}"
        )

    # ========================================================
    # Final statistics
    # ========================================================

    unique_s1_count = len(seen_s1_ids)

    duplicate_s1_row_count = (
        total_rows - unique_s1_count
    )

    results = {
        "ground_truth": {
            "total_rows": total_rows,
            "unique_source1_ids": unique_s1_count,
            "duplicate_source1_ids": len(duplicate_s1_ids),
            "duplicate_source1_rows": duplicate_s1_row_count,
            "invalid_source1_ids": invalid_s1_ids,

            "empty_match_rows": empty_match_rows,
            "exactly_one_match_rows": exactly_one_match_rows,
            "multiple_match_rows": multiple_match_rows,

            "total_match_ids": total_match_ids,

            "s2_match_ids": s2_match_ids,
            "s3_match_ids": s3_match_ids,
            "unknown_match_ids": unknown_match_ids,

            "source1_with_at_least_one_s2": s1_with_s2,
            "source1_with_at_least_one_s3": s1_with_s3,
            "source1_with_both_s2_and_s3": s1_with_both,

            "duplicate_match_id_rows": duplicate_match_id_rows,
            "empty_tokens": empty_tokens,

            "total_match_distribution": {
                str(k): v
                for k, v in sorted(
                    total_match_distribution.items()
                )
            },

            "s2_match_distribution": {
                str(k): v
                for k, v in sorted(
                    s2_match_distribution.items()
                )
            },

            "s3_match_distribution": {
                str(k): v
                for k, v in sorted(
                    s3_match_distribution.items()
                )
            },
        },

        "validation": {
            source_name: {
                "referenced_ids": len(
                    referenced_sets[source_name]
                ),
                "missing_ids": len(
                    missing_ids[source_name]
                ),
                "valid": len(
                    missing_ids[source_name]
                ) == 0,
            }
            for source_name in ["S1", "S2", "S3"]
        },

        "examples": examples,

        "missing_id_examples": {
            source_name: sorted(
                list(ids)
            )[:MAX_EXAMPLES]
            for source_name, ids in missing_ids.items()
        },
    }

    # ========================================================
    # JSON report
    # ========================================================

    json_path = REPORT_DIR / "ground_truth_profile.json"

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # ========================================================
    # Markdown report
    # ========================================================

    md_path = REPORT_DIR / "ground_truth_profile.md"

    gt = results["ground_truth"]
    validation = results["validation"]

    with open(
        md_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write("# Ground Truth Profile\n\n")

        f.write(
            "> Generated by "
            "`preprocessing/profile_ground_truth.py`.\n"
        )

        f.write(
            "> This report parses `matched_entity_ids` as a "
            "comma-separated list of Source-2 / Source-3 IDs.\n\n"
        )

        # ----------------------------------------------------
        # Overview
        # ----------------------------------------------------

        f.write("## 1. Overview\n\n")

        f.write("| Metric | Value |\n")
        f.write("|---|---:|\n")
        f.write(
            f"| Total ground-truth rows | "
            f"{gt['total_rows']:,} |\n"
        )
        f.write(
            f"| Unique Source-1 IDs | "
            f"{gt['unique_source1_ids']:,} |\n"
        )
        f.write(
            f"| Duplicate Source-1 IDs | "
            f"{gt['duplicate_source1_ids']:,} |\n"
        )
        f.write(
            f"| Invalid Source-1 IDs | "
            f"{gt['invalid_source1_ids']:,} |\n"
        )
        f.write(
            f"| Total matched IDs | "
            f"{gt['total_match_ids']:,} |\n"
        )
        f.write(
            f"| S2 match IDs | "
            f"{gt['s2_match_ids']:,} |\n"
        )
        f.write(
            f"| S3 match IDs | "
            f"{gt['s3_match_ids']:,} |\n"
        )
        f.write(
            f"| Unknown-prefix match IDs | "
            f"{gt['unknown_match_ids']:,} |\n"
        )
        f.write("\n")

        # ----------------------------------------------------
        # Match distribution
        # ----------------------------------------------------

        f.write("## 2. Source-1 Match Distribution\n\n")

        f.write("| Matches for one Source-1 entity | Source-1 rows |\n")
        f.write("|---:|---:|\n")

        for count, rows in sorted(
            gt["total_match_distribution"].items(),
            key=lambda x: int(x[0]),
        ):

            f.write(
                f"| {count} | {rows:,} |\n"
            )

        f.write("\n")

        f.write(
            f"- Unmatched Source-1 rows: "
            f"**{gt['empty_match_rows']:,}**\n"
        )

        f.write(
            f"- Exactly one match: "
            f"**{gt['exactly_one_match_rows']:,}**\n"
        )

        f.write(
            f"- One-to-many (2+ matches): "
            f"**{gt['multiple_match_rows']:,}**\n"
        )

        f.write("\n")

        # ----------------------------------------------------
        # S2 / S3 relationship
        # ----------------------------------------------------

        f.write("## 3. S2 / S3 Relationship\n\n")

        f.write("| Metric | Count |\n")
        f.write("|---|---:|\n")
        f.write(
            f"| Source-1 entities with >=1 S2 match | "
            f"{gt['source1_with_at_least_one_s2']:,} |\n"
        )
        f.write(
            f"| Source-1 entities with >=1 S3 match | "
            f"{gt['source1_with_at_least_one_s3']:,} |\n"
        )
        f.write(
            f"| Source-1 entities with both S2 and S3 | "
            f"{gt['source1_with_both_s2_and_s3']:,} |\n"
        )

        f.write("\n")

        f.write("### S2 matches per Source-1 row\n\n")

        f.write("| S2 matches | Source-1 rows |\n")
        f.write("|---:|---:|\n")

        for count, rows in sorted(
            gt["s2_match_distribution"].items(),
            key=lambda x: int(x[0]),
        ):

            f.write(
                f"| {count} | {rows:,} |\n"
            )

        f.write("\n")

        f.write("### S3 matches per Source-1 row\n\n")

        f.write("| S3 matches | Source-1 rows |\n")
        f.write("|---:|---:|\n")

        for count, rows in sorted(
            gt["s3_match_distribution"].items(),
            key=lambda x: int(x[0]),
        ):

            f.write(
                f"| {count} | {rows:,} |\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        f.write("## 4. Entity-ID Validation\n\n")

        f.write(
            "| Source | GT-referenced IDs | Missing IDs | Valid |\n"
        )
        f.write("|---|---:|---:|---|\n")

        for source_name in ["S1", "S2", "S3"]:

            item = validation[source_name]

            f.write(
                f"| {source_name} | "
                f"{item['referenced_ids']:,} | "
                f"{item['missing_ids']:,} | "
                f"{'YES' if item['valid'] else 'NO'} |\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Integrity checks
        # ----------------------------------------------------

        f.write("## 5. Integrity Checks\n\n")

        checks = [
            (
                "Duplicate Source-1 IDs",
                gt["duplicate_source1_ids"] == 0,
            ),
            (
                "Invalid Source-1 IDs",
                gt["invalid_source1_ids"] == 0,
            ),
            (
                "Unknown match-ID prefixes",
                gt["unknown_match_ids"] == 0,
            ),
            (
                "Duplicate match IDs within rows",
                gt["duplicate_match_id_rows"] == 0,
            ),
            (
                "Missing S1 references",
                validation["S1"]["missing_ids"] == 0,
            ),
            (
                "Missing S2 references",
                validation["S2"]["missing_ids"] == 0,
            ),
            (
                "Missing S3 references",
                validation["S3"]["missing_ids"] == 0,
            ),
        ]

        f.write("| Check | Status |\n")
        f.write("|---|---|\n")

        for name, passed in checks:

            f.write(
                f"| {name} | "
                f"{'PASS' if passed else 'REVIEW'} |\n"
            )

        f.write("\n")

        # ----------------------------------------------------
        # Examples
        # ----------------------------------------------------

        f.write("## 6. Examples\n\n")

        f.write("### Unmatched Source-1 entities\n\n")

        for item in examples["empty_matches"]:
            f.write(f"- `{item}`\n")

        f.write("\n### One-to-many examples\n\n")

        for item in examples["multiple_matches"]:

            f.write(
                f"- `{item['source1_entity_id']}` → "
                f"`{', '.join(item['matched_entity_ids'])}`\n"
            )

        f.write("\n### Entities matching both S2 and S3\n\n")

        for item in examples["s1_with_both_sources"]:

            f.write(
                f"- `{item['source1_entity_id']}` → "
                f"S2: `{', '.join(item['S2_matches'])}`; "
                f"S3: `{', '.join(item['S3_matches'])}`\n"
            )

        f.write("\n### Exactly-one-match examples\n\n")

        for item in examples["exactly_one_match"]:

            f.write(
                f"- `{item['source1_entity_id']}` → "
                f"`{', '.join(item['matched_entity_ids'])}`\n"
            )

        # ----------------------------------------------------
        # Missing references
        # ----------------------------------------------------

        f.write("\n## 7. Missing Referenced IDs\n\n")

        for source_name in ["S1", "S2", "S3"]:

            missing = results["missing_id_examples"][source_name]

            f.write(
                f"### {source_name}\n\n"
            )

            if missing:

                for entity_id in missing:
                    f.write(f"- `{entity_id}`\n")

            else:

                f.write("None.\n")

            f.write("\n")

        # ----------------------------------------------------
        # Notes
        # ----------------------------------------------------

        f.write("## 8. Interpretation Notes\n\n")

        f.write(
            "- **Unmatched** means the Source-1 row has zero "
            "matched IDs.\n"
        )

        f.write(
            "- **Exactly one match** means exactly one S2/S3 "
            "entity ID is listed.\n"
        )

        f.write(
            "- **One-to-many** means two or more matched IDs "
            "are listed for the same Source-1 row.\n"
        )

        f.write(
            "- S2 and S3 are classified from their entity-ID "
            "prefixes (`S2-` and `S3-`). Unknown prefixes are "
            "reported rather than silently assigned.\n"
        )

        f.write(
            "- Original ground-truth values are not modified; "
            "whitespace is stripped only for validation and "
            "counting.\n"
        )

    # ========================================================
    # Console summary
    # ========================================================

    print()
    print("=" * 70)
    print("GROUND TRUTH PROFILE COMPLETE")
    print("=" * 70)

    print(f"Total GT rows              : {total_rows:,}")
    print(f"Unique Source-1 IDs        : {unique_s1_count:,}")
    print(f"Duplicate Source-1 IDs     : {len(duplicate_s1_ids):,}")
    print(f"Unmatched Source-1 rows    : {empty_match_rows:,}")
    print(f"Exactly one match          : {exactly_one_match_rows:,}")
    print(f"One-to-many rows           : {multiple_match_rows:,}")
    print(f"Total matched IDs          : {total_match_ids:,}")
    print(f"S2 matched IDs             : {s2_match_ids:,}")
    print(f"S3 matched IDs             : {s3_match_ids:,}")
    print(f"Unknown match IDs          : {unknown_match_ids:,}")
    print(f"S1 with S2                 : {s1_with_s2:,}")
    print(f"S1 with S3                 : {s1_with_s3:,}")
    print(f"S1 with both               : {s1_with_both:,}")
    print()

    print(f"Markdown report: {md_path}")
    print(f"JSON report    : {json_path}")


if __name__ == "__main__":
    profile_ground_truth()