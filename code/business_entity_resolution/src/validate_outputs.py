"""
validate_outputs.py — Local submission sanity checker.

Verifies both output files against all 15 challenge submission rules
*before* uploading to the leaderboard, so format errors are caught
locally instead of wasting a submission slot.

Checks performed
----------------
 1. matching_results.tsv exists.
 2. candidate_pairs.tsv exists.
 3. matching_results.tsv has exact headers:
      source1_entity_id, matched_entity_ids
 4. candidate_pairs.tsv has exact headers:
      source1_entity_id, candidate_entity_ids
 5. Every test Source 1 entity appears in matching_results.tsv.
 6. Every test Source 1 entity appears in candidate_pairs.tsv.
 7. No duplicate source1_entity_id rows in matching_results.tsv.
 8. No duplicate source1_entity_id rows in candidate_pairs.tsv.
 9. All matched IDs are S2-* or S3-* prefixed.
10. All candidate IDs are S2-* or S3-* prefixed.
11. No duplicate IDs within any comma-separated list.
12. No S1-* IDs appear in matched_entity_ids or candidate_entity_ids.
13. Every predicted match ID is contained in the corresponding
    candidate list for that Source 1 entity.
14. All candidate IDs actually exist in test_source2.tsv or
    test_source3.tsv.
15. All matched IDs actually exist in test_source2.tsv or
    test_source3.tsv.
16. Empty matched_entity_ids cell is accepted for singletons
    (this is the correct format; it is enforced by checks 9–15
    trivially passing on empty lists).

CLI usage
---------
    python src/validate_outputs.py \\
        --matching  output/matching_results.tsv \\
        --candidate output/candidate_pairs.tsv \\
        --test-dir  dataset/test

Exit codes
----------
    0  All checks passed — safe to submit.
    1  One or more checks failed — review printed issues before submitting.

No score is computed and no "validation passed" message is printed unless
all checks actually execute successfully against real files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MATCHING_FILE = "matching_results.tsv"
CANDIDATE_FILE = "candidate_pairs.tsv"

MATCHING_COLS = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_COLS = ("source1_entity_id", "candidate_entity_ids")

TEST_SOURCE1_FILE = "test_source1.tsv"
TEST_SOURCE2_FILE = "test_source2.tsv"
TEST_SOURCE3_FILE = "test_source3.tsv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_ids(raw: object) -> list[str]:
    """
    Parse a comma-separated ID cell into a list (preserving duplicates).

    Returns an empty list for NaN / empty cells.
    """
    s = str(raw).strip() if not pd.isna(raw) else ""
    if not s:
        return []
    return [tok.strip() for tok in s.split(",") if tok.strip()]


def _load_source_ids(path: Path, label: str) -> set[str]:
    """
    Load entity_id values from a source TSV.

    Parameters
    ----------
    path  : Path  Path to the TSV file.
    label : str   Human-readable label for error messages.

    Returns
    -------
    set[str]
        All entity_id values in the file.
    """
    if not path.exists():
        # Caller handles missing test files.
        return set()
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if "entity_id" not in df.columns:
        print(
            f"  ERROR: {label} has no 'entity_id' column.",
            file=sys.stderr,
        )
        return set()
    return set(df["entity_id"].str.strip())


# ---------------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------------


class Validator:
    """
    Runs all 15 validation checks and collects issues.

    After calling ``run()``, inspect ``self.issues`` (list of str) and
    ``self.passed`` (bool).
    """

    def __init__(
        self,
        matching_path: Path,
        candidate_path: Path,
        test_dir: Path,
    ) -> None:
        self.matching_path = matching_path
        self.candidate_path = candidate_path
        self.test_dir = test_dir

        self.issues: list[str] = []
        self.passed: bool = False

        # Populated during run().
        self._matching_df: pd.DataFrame | None = None
        self._candidate_df: pd.DataFrame | None = None
        self._s1_test_ids: set[str] = set()
        self._s2_ids: set[str] = set()
        self._s3_ids: set[str] = set()
        self._target_ids: set[str] = set()

    def _issue(self, msg: str) -> None:
        self.issues.append(msg)

    # ------------------------------------------------------------------
    # File existence and header checks (1–4)
    # ------------------------------------------------------------------

    def _check_file_exists(self, path: Path, label: str) -> bool:
        if not path.exists():
            self._issue(f"[1/2] {label} not found: {path}")
            return False
        return True

    def _check_headers(
        self,
        df: pd.DataFrame,
        expected: tuple[str, str],
        path: Path,
        check_num: str,
    ) -> bool:
        cols = list(df.columns)
        if cols[:2] != list(expected):
            self._issue(
                f"[{check_num}] Wrong headers in {path.name}.\n"
                f"  Expected : {list(expected)}\n"
                f"  Found    : {cols}"
            )
            return False
        return True

    # ------------------------------------------------------------------
    # Entity coverage and duplicate checks (5–8)
    # ------------------------------------------------------------------

    def _check_coverage(
        self,
        df: pd.DataFrame,
        s1_ids: set[str],
        id_col: str,
        file_label: str,
        check_num_coverage: str,
        check_num_dup: str,
    ) -> None:
        found_ids = df[id_col].str.strip().tolist()
        found_set = set(found_ids)

        # Coverage.
        missing = s1_ids - found_set
        if missing:
            sample = sorted(missing)[:5]
            self._issue(
                f"[{check_num_coverage}] {len(missing)} Source 1 entities missing "
                f"from {file_label}: {sample}{'...' if len(missing) > 5 else ''}"
            )

        # Duplicates.
        seen: set[str] = set()
        dups: list[str] = []
        for eid in found_ids:
            if eid in seen and eid not in dups:
                dups.append(eid)
            seen.add(eid)
        if dups:
            self._issue(
                f"[{check_num_dup}] Duplicate source1_entity_id rows in "
                f"{file_label}: {dups[:5]}"
            )

    # ------------------------------------------------------------------
    # ID format and content checks (9–15)
    # ------------------------------------------------------------------

    def _check_id_lists(
        self,
        df: pd.DataFrame,
        id_col: str,
        list_col: str,
        file_label: str,
        target_ids: set[str],
        check_prefix: str,
        check_no_s1: str,
        check_no_dup: str,
        check_exists: str,
        candidate_map: dict[str, set[str]] | None = None,
    ) -> None:
        """
        Validate the comma-separated ID lists in ``list_col``.

        Parameters
        ----------
        candidate_map : dict | None
            If provided (for matching results), checks that every
            predicted ID is in the corresponding candidate list.
        """
        for _, row in df.iterrows():
            eid = str(row[id_col]).strip()
            raw_ids = _parse_ids(row[list_col])

            # Empty list is always valid.
            if not raw_ids:
                continue

            # Check for S1 IDs in prediction lists (check 12).
            s1_in_list = [i for i in raw_ids if i.startswith("S1-")]
            if s1_in_list:
                self._issue(
                    f"[{check_no_s1}] S1-* IDs found in {file_label} for "
                    f"{eid}: {s1_in_list[:3]}"
                )

            # Check prefix format (S2-* or S3-* only) (check 9/10).
            bad_prefix = [
                i for i in raw_ids
                if not i.startswith("S2-") and not i.startswith("S3-")
            ]
            if bad_prefix:
                self._issue(
                    f"[{check_prefix}] Non-S2/S3 IDs in {file_label} for "
                    f"{eid}: {bad_prefix[:3]}"
                )

            # Check for duplicate IDs within the list (check 11).
            if len(raw_ids) != len(set(raw_ids)):
                dups = [i for i in set(raw_ids) if raw_ids.count(i) > 1]
                self._issue(
                    f"[{check_no_dup}] Duplicate IDs in {file_label} for "
                    f"{eid}: {dups[:3]}"
                )

            # Check IDs exist in test source 2/3 (check 14/15).
            nonexistent = [i for i in raw_ids if i not in target_ids]
            if nonexistent:
                self._issue(
                    f"[{check_exists}] IDs in {file_label} for {eid} not "
                    f"found in test_source2/3: {nonexistent[:3]}"
                )

            # Check every predicted match is in candidate list (check 13).
            if candidate_map is not None:
                cands = candidate_map.get(eid, set())
                not_in_cands = [i for i in raw_ids if i not in cands]
                if not_in_cands:
                    self._issue(
                        f"[13] Predicted IDs for {eid} not in candidate list: "
                        f"{not_in_cands[:3]}"
                    )

    # ------------------------------------------------------------------
    # Malformed row check (14)
    # ------------------------------------------------------------------

    def _check_malformed_rows(
        self,
        df: pd.DataFrame,
        expected_cols: tuple[str, str],
        file_label: str,
        check_num: str,
    ) -> None:
        """Detect rows with unexpected extra or missing columns."""
        if len(df.columns) != 2:
            self._issue(
                f"[{check_num}] {file_label} has {len(df.columns)} columns "
                f"(expected 2): {list(df.columns)}"
            )

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run(self) -> bool:
        """
        Execute all validation checks.

        Returns
        -------
        bool
            True if all checks pass (``self.issues`` is empty).
        """
        # --- Checks 1–2: file existence ---
        m_ok = self._check_file_exists(self.matching_path, MATCHING_FILE)
        c_ok = self._check_file_exists(self.candidate_path, CANDIDATE_FILE)
        if not m_ok or not c_ok:
            self.passed = False
            return False

        # Load files.
        try:
            self._matching_df = pd.read_csv(
                self.matching_path,
                sep="\t",
                dtype=str,
                keep_default_na=False,
            )
        except Exception as exc:
            self._issue(f"[1] Could not read {MATCHING_FILE}: {exc}")
            self.passed = False
            return False

        try:
            self._candidate_df = pd.read_csv(
                self.candidate_path,
                sep="\t",
                dtype=str,
                keep_default_na=False,
            )
        except Exception as exc:
            self._issue(f"[2] Could not read {CANDIDATE_FILE}: {exc}")
            self.passed = False
            return False

        # --- Checks 3–4: headers ---
        m_headers_ok = self._check_headers(
            self._matching_df, MATCHING_COLS, self.matching_path, "3"
        )
        c_headers_ok = self._check_headers(
            self._candidate_df, CANDIDATE_COLS, self.candidate_path, "4"
        )

        # Stop early if headers are wrong — subsequent checks would fail.
        if not m_headers_ok or not c_headers_ok:
            self.passed = False
            return False

        # --- Load test source IDs ---
        s1_path = self.test_dir / TEST_SOURCE1_FILE
        s2_path = self.test_dir / TEST_SOURCE2_FILE
        s3_path = self.test_dir / TEST_SOURCE3_FILE

        self._s1_test_ids = _load_source_ids(s1_path, TEST_SOURCE1_FILE)
        self._s2_ids = _load_source_ids(s2_path, TEST_SOURCE2_FILE)
        self._s3_ids = _load_source_ids(s3_path, TEST_SOURCE3_FILE)
        self._target_ids = self._s2_ids | self._s3_ids

        if not self._s1_test_ids:
            print(
                f"WARNING: Could not load Source 1 test IDs from {s1_path}. "
                "Coverage checks (5–6) will be skipped.",
                file=sys.stderr,
            )
        if not self._target_ids:
            print(
                f"WARNING: Could not load Source 2/3 test IDs from {s2_path} / "
                f"{s3_path}. Existence checks (14–15) will be skipped.",
                file=sys.stderr,
            )

        # --- Checks 5–8: coverage and duplicates ---
        if self._s1_test_ids:
            self._check_coverage(
                self._matching_df,
                self._s1_test_ids,
                "source1_entity_id",
                MATCHING_FILE,
                "5",
                "7",
            )
            self._check_coverage(
                self._candidate_df,
                self._s1_test_ids,
                "source1_entity_id",
                CANDIDATE_FILE,
                "6",
                "8",
            )
        else:
            # Still check for duplicate rows even without test source.
            self._check_coverage(
                self._matching_df,
                set(),
                "source1_entity_id",
                MATCHING_FILE,
                "5",
                "7",
            )
            self._check_coverage(
                self._candidate_df,
                set(),
                "source1_entity_id",
                CANDIDATE_FILE,
                "6",
                "8",
            )

        # Build candidate map for check 13.
        candidate_map: dict[str, set[str]] = {}
        for _, row in self._candidate_df.iterrows():
            eid = str(row["source1_entity_id"]).strip()
            candidate_map[eid] = set(_parse_ids(row["candidate_entity_ids"]))

        # --- Checks 9–15: ID list content ---
        # Malformed rows (extra columns) — check 14 (format).
        self._check_malformed_rows(
            self._matching_df, MATCHING_COLS, MATCHING_FILE, "14a"
        )
        self._check_malformed_rows(
            self._candidate_df, CANDIDATE_COLS, CANDIDATE_FILE, "14b"
        )

        # Candidate ID list validation (checks 10, 11, 12, 14).
        self._check_id_lists(
            self._candidate_df,
            id_col="source1_entity_id",
            list_col="candidate_entity_ids",
            file_label=CANDIDATE_FILE,
            target_ids=self._target_ids,
            check_prefix="10",
            check_no_s1="12",
            check_no_dup="11",
            check_exists="14",
        )

        # Matching ID list validation (checks 9, 11, 12, 13, 15).
        self._check_id_lists(
            self._matching_df,
            id_col="source1_entity_id",
            list_col="matched_entity_ids",
            file_label=MATCHING_FILE,
            target_ids=self._target_ids,
            check_prefix="9",
            check_no_s1="12",
            check_no_dup="11",
            check_exists="15",
            candidate_map=candidate_map,
        )

        self.passed = len(self.issues) == 0
        return self.passed


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for validate_outputs.py."""
    parser = argparse.ArgumentParser(
        description=(
            "Validate submission output files against all challenge rules.\n"
            "Exit code 0 = PASS. Exit code 1 = FAIL."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--matching",
        required=True,
        type=Path,
        help="Path to matching_results.tsv.",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        type=Path,
        help="Path to candidate_pairs.tsv.",
    )
    parser.add_argument(
        "--test-dir",
        required=True,
        type=Path,
        help="Directory containing test_source1/2/3.tsv.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Submission Output Validator")
    print("=" * 60)
    print(f"  matching  : {args.matching}")
    print(f"  candidate : {args.candidate}")
    print(f"  test-dir  : {args.test_dir}")
    print()

    validator = Validator(
        matching_path=args.matching,
        candidate_path=args.candidate,
        test_dir=args.test_dir,
    )

    passed = validator.run()

    if validator.issues:
        print(f"Found {len(validator.issues)} issue(s):")
        for i, issue in enumerate(validator.issues, 1):
            print(f"  {i}. {issue}")
        print()

    if passed:
        print("PASS — all checks succeeded. Safe to submit.")
        sys.exit(0)
    else:
        print("FAIL — fix the issues listed above before submitting.")
        sys.exit(1)


if __name__ == "__main__":
    main()
