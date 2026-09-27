"""
evaluate.py — Official macro-averaged F₀.₅ metric for entity resolution.

Implements the exact evaluation formula used by the Amazon ML Challenge:

    F₀.₅ = (1.25 × precision × recall) / (0.25 × precision + recall)

Computed as a *macro-average*: F₀.₅ is calculated per Source 1 entity,
then averaged across all Source 1 entities in the evaluation set.

Singleton handling
------------------
A Source 1 entity with no true matches (singleton) scores:
  - 1.0  when the prediction is also empty (correct empty prediction)
  - 0.0  when any match is predicted (false positive on a singleton)

This is consistent with the challenge rules: correctly identifying
singletons earns full credit; false merges on them are penalized.

Duplicate ID handling
---------------------
Duplicate IDs within a comma-separated list are deduplicated before
scoring. A warning is printed if duplicates are found.

CLI usage
---------
    python src/evaluate.py \\
        --ground-truth dataset/train/train_ground_truth.tsv \\
        --predictions  output/matching_results.tsv \\
        [--examples N]

``--examples N`` prints up to N example rows showing per-entity detail.

All scores are computed from the actual file contents only.
No score is ever hardcoded.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------


def f_beta(
    precision: float,
    recall: float,
    beta: float = 0.5,
) -> float:
    """
    Compute F_β score.

    F_β = (1 + β²) × precision × recall
          ─────────────────────────────────
          (β² × precision) + recall

    When both precision and recall are 0 (empty true and empty pred with
    no true matches) the score is handled upstream; this function is only
    called when at least one of them is non-zero or when a deliberate
    1.0 / 0.0 override applies.

    Parameters
    ----------
    precision : float  Value in [0, 1].
    recall    : float  Value in [0, 1].
    beta      : float  Default 0.5 (precision-heavy).

    Returns
    -------
    float
        F_β in [0, 1], or 0.0 when the denominator is zero.
    """
    beta_sq = beta ** 2
    denom = (beta_sq * precision) + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + beta_sq) * precision * recall / denom


class EntityScore(NamedTuple):
    """Per-entity scoring result."""

    source1_entity_id: str
    true_ids: frozenset[str]
    pred_ids: frozenset[str]
    precision: float
    recall: float
    f05: float


def score_entity(
    entity_id: str,
    true_ids: set[str],
    pred_ids: set[str],
) -> EntityScore:
    """
    Compute precision, recall, and F₀.₅ for a single Source 1 entity.

    Singleton rule
    ~~~~~~~~~~~~~~
    If ``true_ids`` is empty:
      - Empty prediction  → precision=1, recall=1, F₀.₅=1
      - Any prediction    → precision=0, recall=1, F₀.₅=0
      (recall for a true singleton is defined as 1 when pred is also
      empty, matching the challenge's "correctly identifying singletons
      earns 1.0" rule)

    Parameters
    ----------
    entity_id : str
        Source 1 entity ID (for reporting only).
    true_ids : set[str]
        Ground-truth matched entity IDs (may be empty for singletons).
    pred_ids : set[str]
        Predicted matched entity IDs (may be empty).

    Returns
    -------
    EntityScore
        Named tuple with per-entity scores.
    """
    true_fs = frozenset(true_ids)
    pred_fs = frozenset(pred_ids)

    # Singleton: no true matches.
    if not true_fs:
        if not pred_fs:
            # Correct singleton prediction.
            return EntityScore(
                source1_entity_id=entity_id,
                true_ids=true_fs,
                pred_ids=pred_fs,
                precision=1.0,
                recall=1.0,
                f05=1.0,
            )
        else:
            # False positive on a singleton.
            return EntityScore(
                source1_entity_id=entity_id,
                true_ids=true_fs,
                pred_ids=pred_fs,
                precision=0.0,
                recall=1.0,
                f05=0.0,
            )

    # Standard case: true_ids is non-empty.
    tp = len(true_fs & pred_fs)

    precision = tp / len(pred_fs) if pred_fs else 0.0
    recall = tp / len(true_fs)
    score = f_beta(precision, recall, beta=0.5)

    return EntityScore(
        source1_entity_id=entity_id,
        true_ids=true_fs,
        pred_ids=pred_fs,
        precision=precision,
        recall=recall,
        f05=score,
    )


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def _parse_ids(raw: object) -> set[str]:
    """
    Parse a comma-separated ID string into a set.

    Empty strings and NaN values produce an empty set.
    Duplicate IDs within the list are deduplicated (with a warning
    emitted by the caller if needed).
    """
    if pd.isna(raw) or str(raw).strip() == "":
        return set()
    return {tok.strip() for tok in str(raw).split(",") if tok.strip()}


def _load_tsv(path: Path, id_col: str, match_col: str) -> dict[str, set[str]]:
    """
    Load a two-column TSV (entity_id → set of matched IDs).

    Parameters
    ----------
    path      : Path   Path to the TSV file.
    id_col    : str    Column name for the Source 1 entity ID.
    match_col : str    Column name for the comma-separated match list.

    Returns
    -------
    dict[str, set[str]]
        Mapping from Source 1 entity ID to set of matched entity IDs.

    Raises
    ------
    SystemExit
        If the file does not exist or required columns are missing.
    """
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    for col in (id_col, match_col):
        if col not in df.columns:
            print(
                f"ERROR: Column '{col}' not found in {path}.\n"
                f"  Found columns: {list(df.columns)}",
                file=sys.stderr,
            )
            sys.exit(1)

    result: dict[str, set[str]] = {}
    dup_warned: set[str] = set()

    for _, row in df.iterrows():
        eid = str(row[id_col]).strip()
        raw = row[match_col]
        ids = _parse_ids(raw)

        # Warn on duplicates within the original comma-separated list.
        if not pd.isna(raw) and str(raw).strip():
            raw_list = [t.strip() for t in str(raw).split(",") if t.strip()]
            if len(raw_list) != len(set(raw_list)) and eid not in dup_warned:
                print(
                    f"WARNING: Duplicate IDs found for {eid} in {path.name}. "
                    "Deduplicating.",
                    file=sys.stderr,
                )
                dup_warned.add(eid)

        result[eid] = ids

    return result


def load_ground_truth(path: Path) -> dict[str, set[str]]:
    """Load train_ground_truth.tsv → {source1_entity_id: set of matched IDs}."""
    return _load_tsv(
        path,
        id_col="source1_entity_id",
        match_col="matched_entity_ids",
    )


def load_predictions(path: Path) -> dict[str, set[str]]:
    """Load matching_results.tsv → {source1_entity_id: set of predicted IDs}."""
    return _load_tsv(
        path,
        id_col="source1_entity_id",
        match_col="matched_entity_ids",
    )


# ---------------------------------------------------------------------------
# Macro-average evaluation
# ---------------------------------------------------------------------------


def evaluate(
    ground_truth: dict[str, set[str]],
    predictions: dict[str, set[str]],
) -> tuple[float, list[EntityScore]]:
    """
    Compute macro-averaged F₀.₅ over all Source 1 entities in ground truth.

    Entities present in ground truth but absent from predictions are
    treated as empty predictions (no matches predicted).

    Parameters
    ----------
    ground_truth : dict[str, set[str]]
        {source1_entity_id → set of true matched entity IDs}.
    predictions  : dict[str, set[str]]
        {source1_entity_id → set of predicted entity IDs}.

    Returns
    -------
    macro_f05 : float
        Macro-averaged F₀.₅ over all entities.
    scores : list[EntityScore]
        Per-entity scoring details, in input order.
    """
    entity_scores: list[EntityScore] = []

    for eid, true_ids in ground_truth.items():
        pred_ids = predictions.get(eid, set())
        es = score_entity(eid, true_ids, pred_ids)
        entity_scores.append(es)

    if not entity_scores:
        return 0.0, []

    macro_f05 = sum(es.f05 for es in entity_scores) / len(entity_scores)
    return macro_f05, entity_scores


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_examples(
    scores: list[EntityScore],
    n: int,
) -> None:
    """Print up to ``n`` per-entity example rows to stdout."""
    header = (
        f"{'source1_entity_id':<20} "
        f"{'true_ids':<30} "
        f"{'pred_ids':<30} "
        f"{'precision':>9} "
        f"{'recall':>7} "
        f"{'F0.5':>7}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    for es in scores[:n]:
        true_str = ",".join(sorted(es.true_ids)) if es.true_ids else "<none>"
        pred_str = ",".join(sorted(es.pred_ids)) if es.pred_ids else "<none>"
        # Truncate long ID lists for display.
        if len(true_str) > 28:
            true_str = true_str[:25] + "..."
        if len(pred_str) > 28:
            pred_str = pred_str[:25] + "..."
        print(
            f"{es.source1_entity_id:<20} "
            f"{true_str:<30} "
            f"{pred_str:<30} "
            f"{es.precision:>9.4f} "
            f"{es.recall:>7.4f} "
            f"{es.f05:>7.4f}"
        )
    print(sep)


def main() -> None:
    """CLI entry point for evaluate.py."""
    parser = argparse.ArgumentParser(
        description=(
            "Compute macro-averaged F₀.₅ for entity resolution predictions.\n"
            "Scores are derived entirely from the supplied files."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ground-truth",
        required=True,
        type=Path,
        help="Path to train_ground_truth.tsv (tab-separated).",
    )
    parser.add_argument(
        "--predictions",
        required=True,
        type=Path,
        help="Path to matching_results.tsv (tab-separated).",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=0,
        metavar="N",
        help="Print up to N per-entity example rows.",
    )
    args = parser.parse_args()

    print(f"Ground truth : {args.ground_truth}")
    print(f"Predictions  : {args.predictions}")
    print()

    ground_truth = load_ground_truth(args.ground_truth)
    predictions = load_predictions(args.predictions)

    print(f"Ground-truth entities : {len(ground_truth):,}")
    print(f"Prediction entities   : {len(predictions):,}")

    extra = set(predictions) - set(ground_truth)
    if extra:
        print(
            f"WARNING: {len(extra)} prediction entity IDs not in ground truth "
            "(they will be ignored in macro-average).",
            file=sys.stderr,
        )

    missing = set(ground_truth) - set(predictions)
    if missing:
        print(
            f"NOTE: {len(missing)} ground-truth entities have no prediction row "
            "(treated as empty prediction).",
        )

    print()

    macro_f05, scores = evaluate(ground_truth, predictions)

    if args.examples > 0:
        _print_examples(scores, args.examples)
        print()

    print(f"Macro-averaged F₀.₅  : {macro_f05:.6f}")


# ---------------------------------------------------------------------------
# Inline unit tests
# ---------------------------------------------------------------------------


def _run_unit_tests() -> None:
    """
    Lightweight in-memory unit tests for the core metric logic.

    These do not require any files on disk.  Run with:
        python src/evaluate.py --test
    """
    import traceback

    passed = 0
    failed = 0

    def check(name: str, got: object, expected: object) -> None:
        nonlocal passed, failed
        if isinstance(expected, float):
            ok = abs(float(got) - expected) < 1e-9  # type: ignore[arg-type]
        else:
            ok = got == expected
        if ok:
            print(f"  PASS  {name}")
            passed += 1
        else:
            print(f"  FAIL  {name}: got {got!r}, expected {expected!r}")
            failed += 1

    print("Running unit tests...")

    # 1. Exact match
    es = score_entity("S1-001", {"S2-001", "S2-002"}, {"S2-001", "S2-002"})
    check("exact_match precision", es.precision, 1.0)
    check("exact_match recall", es.recall, 1.0)
    check("exact_match f05", es.f05, 1.0)

    # 2. Empty prediction (false negative)
    es = score_entity("S1-002", {"S2-001"}, set())
    check("empty_pred precision", es.precision, 0.0)
    check("empty_pred recall", es.recall, 0.0)
    check("empty_pred f05", es.f05, 0.0)

    # 3. False positive (predict extra)
    es = score_entity("S1-003", {"S2-001"}, {"S2-001", "S2-999"})
    check("false_positive precision", es.precision, 0.5)
    check("false_positive recall", es.recall, 1.0)
    expected_fp = f_beta(0.5, 1.0, 0.5)
    check("false_positive f05", es.f05, expected_fp)

    # 4. False negative (missed match)
    es = score_entity("S1-004", {"S2-001", "S2-002"}, {"S2-001"})
    check("false_negative precision", es.precision, 1.0)
    check("false_negative recall", es.recall, 0.5)
    expected_fn = f_beta(1.0, 0.5, 0.5)
    check("false_negative f05", es.f05, expected_fn)

    # 5. Singleton — correct empty prediction
    es = score_entity("S1-005", set(), set())
    check("singleton_correct f05", es.f05, 1.0)

    # 6. Singleton — wrong prediction
    es = score_entity("S1-006", set(), {"S2-001"})
    check("singleton_wrong f05", es.f05, 0.0)

    # 7. Macro-average over two entities
    gt = {"A": {"x"}, "B": set()}
    pred = {"A": {"x"}, "B": set()}
    macro, _ = evaluate(gt, pred)
    check("macro_avg perfect", macro, 1.0)

    # 8. Macro-average: one correct, one singleton fp
    gt = {"A": {"x"}, "B": set()}
    pred = {"A": {"x"}, "B": {"x"}}
    macro, _ = evaluate(gt, pred)
    check("macro_avg mixed", macro, 0.5)

    # 9. Duplicate IDs in prediction: treated as same entity after dedup
    es = score_entity("S1-007", {"S2-001"}, {"S2-001", "S2-001"})
    # After set dedup the pred_ids = {"S2-001"}, so perfect match
    check("dedup_in_pred recall", es.recall, 1.0)

    print()
    print(f"Results: {passed} passed, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    # Support --test flag for running unit tests.
    if len(sys.argv) == 2 and sys.argv[1] == "--test":
        _run_unit_tests()
    else:
        main()
