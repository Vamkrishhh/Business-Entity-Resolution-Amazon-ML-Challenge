# Documentation_template.md
# Business Entity Resolution — Member 3 Methodology

---

## Executive Summary

This document describes the **Member 3** contribution to the Amazon ML
Challenge Business Entity Resolution pipeline.  The contribution covers
four Python modules:

| Module | Role |
|---|---|
| `normalize.py` | Canonical string normalization of name, address, and country fields |
| `features.py` | Pairwise similarity feature engineering (13 features) |
| `matcher.py` | Supervised binary classifier (`EntityMatcher`) |
| `pipeline.py` | End-to-end inference orchestration (CLI) |

The pipeline follows a standard entity resolution architecture:
**normalize → block → featurize → classify**.  All processing is
deterministic and offline; no external APIs or internet services are used.

> **Results disclaimer:** No numerical performance metrics (F₀.₅,
> precision, recall, leaderboard scores) are reported in this document
> because no evaluation experiment has been executed against the held-out
> test set at the time of writing.  All result claims must come from
> actually running the pipeline and comparing against ground truth.

---

## Methodology

### Pipeline Architecture

```
Source 1 TSV ──┐
Source 2 TSV ──┤──► Normalize ──► Block ──► Featurize ──► Classify ──► Output TSVs
Source 3 TSV ──┘
```

**Step 1 — Load & Normalize**
Each of the three source files is read as a tab-separated DataFrame.
Three fields are normalized per record:

- `business_name` → `name_normalized` (via `normalize_name`)
- `business_address` → `address_normalized` (via `normalize_address`)
- `country` → `country_normalized` (via `normalize_country`)

**Step 2 — Record conversion**
Each DataFrame row is converted to a plain dict (`make_record`) retaining
`entity_id` and the three normalized fields.  Source 2 and Source 3
records are pooled into a single target list.

**Step 3 — Candidate generation (blocking)**
Source 1 and target records are grouped by `country_normalized`.
Only same-country pairs are considered candidates (see §Candidate
Generation below).

**Step 4 — Feature engineering**
For each (Source 1, candidate) pair, `build_features` computes a 13-
element float feature dict.

**Step 5 — Classification**
`EntityMatcher.predict_probability` scores all candidate pairs for a
given Source 1 entity.  Pairs with probability ≥ threshold are included
in `matching_results.tsv`; all pairs regardless of score appear in
`candidate_pairs.tsv`.

---

## Candidate Generation

**Strategy: exact country-code blocking.**

The blocking key is `country_normalized`, a two-letter ISO 3166-1
alpha-2 code produced by `normalize_country`.  All Source 2 and Source 3
records are indexed by this key.  For each Source 1 record, only records
sharing the same country code are retrieved as candidates.

**Rationale:**
- Country is a highly reliable coarse identifier.  Cross-country entity
  merges are extremely rare in commercial entity resolution.
- The blocking step is cheap (dict lookup, O(1) per record) and does not
  require approximate-search data structures.
- Recall upper bound: any true match pair from different countries is
  missed by this blocker.  This is an accepted trade-off given the
  precision-heavy F₀.₅ evaluation metric.

**Country normalization:**
The alias table covers common spelling and language variants
(e.g. "Deutschland" → "de", "United Kingdom" → "gb").  Unknown country
strings are kept as-is (lowercased) so they still group consistently
rather than falling into a single "unknown" bucket.

---

## Matching Model

### Architecture

```
StandardScaler
    │
    └─► GradientBoostingClassifier
            n_estimators  = 300
            max_depth     = 4
            learning_rate = 0.05
            subsample     = 0.8
            random_state  = 42
```

The scaler is included to make the pipeline self-contained and to allow
straightforward replacement with a linear or logistic model.

### Feature Engineering

`build_features(left, right)` produces 13 float-valued features:

#### Name similarity (6 features)

| Feature | Description |
|---|---|
| `name_token_jaccard` | Jaccard similarity of word-token sets after normalization |
| `name_char3gram_jaccard` | Jaccard of 3-character n-gram sets (boundary-padded) |
| `name_char4gram_jaccard` | Jaccard of 4-character n-gram sets |
| `name_length_ratio` | min(len_a, len_b) / max(len_a, len_b) |
| `name_prefix4_match` | Binary: first 4 normalized characters match |
| `name_exact_match` | Binary: full normalized name strings are identical |

Character n-gram features capture partial string overlap and are robust
to insertions, deletions, and transpositions that token-level Jaccard
misses.

#### Address similarity (5 features)

| Feature | Description |
|---|---|
| `addr_token_jaccard` | Jaccard of address word-token sets |
| `addr_char3gram_jaccard` | Jaccard of 3-char n-grams |
| `addr_number_overlap` | Fraction of numeric tokens in Source 1 address found in candidate |
| `addr_length_ratio` | Length ratio |
| `addr_exact_match` | Binary: full normalized addresses are identical |

`addr_number_overlap` is specifically designed to penalize address
mismatches caused by different street numbers, postal codes, or building
numbers — highly discriminative signals.

#### Country & cross features (2 features)

| Feature | Description |
|---|---|
| `country_exact_match` | Binary: ISO country codes agree |
| `name_addr_combined` | 0.6 × name_token_jaccard + 0.4 × addr_token_jaccard |

`name_addr_combined` provides a single composite signal that the
classifier can use as a summary feature, complementing the individual
components.

### Training Protocol

> Training is performed in a separate step not included in this
> inference pipeline.  The procedure below describes the intended
> protocol.

1. Load `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`
   and `train_ground_truth.tsv`.
2. Normalize all records using `normalize_name`, `normalize_address`,
   `normalize_country`.
3. Generate candidate pairs using the same country-blocking strategy
   as inference.
4. For each candidate pair, build a feature dict using `build_features`.
5. Label each pair: 1 if it appears in `train_ground_truth.tsv`, 0
   otherwise.
6. Call `EntityMatcher.train(feature_rows, labels)`.
7. Save with `EntityMatcher.save("output/matcher.joblib")`.

### Threshold

The default classification threshold is **0.85**.  Pairs with
`predict_probability ≥ threshold` are reported as matches.  The
threshold can be tuned on a held-out validation split of the training
data to optimize F₀.₅.

---

## Results & Error Analysis

> **No numerical results are claimed here.**
>
> At the time of writing, the pipeline has not been executed against the
> held-out test set, and no experiment has been run to produce F₀.₅,
> precision, or recall numbers.  Any result figures would need to be
> obtained by:
>
> 1. Training `EntityMatcher` on the training data.
> 2. Running `pipeline.py` against the test sources.
> 3. Submitting `matching_results.tsv` to the leaderboard.
>
> Expected error modes (based on pipeline design, not measured data):
>
> - **Cross-country misses:** True matches where one record has a
>   missing or differently spelled country will be excluded by the
>   blocker.
> - **Name alias misses:** Highly abbreviated names (e.g. "IBM" vs
>   "International Business Machines") will have low token Jaccard and
>   may fall below threshold.
> - **Address format divergence:** Addresses in different scripts or
>   with structural differences (e.g. Japanese kanji vs romanized form)
>   may score poorly on string similarity despite referring to the same
>   location.
> - **False positives:** Common business name fragments (e.g. "trading
>   company", "enterprises") shared by many entities may cause high
>   Jaccard even for unrelated businesses; the address and country
>   features are expected to mitigate this.

---

## Conclusion

The Member 3 pipeline implements a complete, runnable entity resolution
system with:

- **Conservative, lossless normalization** of name, address, and country
  fields, consistent with the team preprocessing module's design
  principles.
- **Efficient country-code blocking** that eliminates the need for an
  all-pairs comparison.
- **13 complementary similarity features** spanning token overlap,
  character n-gram overlap, numeric token agreement, and exact-match
  signals.
- **A trained GradientBoostingClassifier** that combines features into a
  calibrated match probability, applied at a configurable threshold.
- **No external data, APIs, or geocoding services** — the pipeline is
  fully self-contained and compliant with the challenge's fair-play
  constraints.

---

## Appendix

### A. Normalization Design Decisions

**Why canonical strings rather than structured dicts?**
`pipeline.py` stores normalized values as DataFrame columns and passes
them to `build_features` as dict values.  Returning plain strings makes
blocking (country grouping) and feature computation straightforward
without additional field extraction.

**Why ISO 3166-1 alpha-2 as the blocking key?**
Two-letter codes are compact, well-known, and stable.  An alias table
handles common variants without requiring fuzzy matching.

**Why canonicalize legal suffixes (e.g. "Ltd" → "limited") rather than
strip them?**
Stripping legal suffixes removes potentially meaningful distinguishing
information (e.g. "Acme Ltd" and "Acme Corp" may be different entities).
Canonicalization standardizes spelling while preserving the term.

### B. Feature Engineering Design Decisions

**Why both token Jaccard and character n-gram Jaccard for names?**
Token Jaccard is high only when both names share most of the same words.
Character n-grams capture partial overlaps and are more robust to minor
spelling differences, abbreviations within tokens, and OCR errors.

**Why `addr_number_overlap` instead of full address Jaccard only?**
Street numbers and postal codes are the most discriminative address
components.  A pair with identical text but different house numbers
(e.g. "123 Main St" vs "456 Main St") should score very low.
`addr_number_overlap` specifically penalizes numeric discrepancies that
token-level Jaccard might dilute.

**Why a composite `name_addr_combined` feature?**
It gives the classifier a single summary signal that correlates both
name and address agreement, which can be useful as an interaction term
in the decision boundary.

### C. Model Architecture Rationale

**GradientBoostingClassifier:**
Gradient boosting is robust to correlated and redundant features (which
are expected here given multiple name and address similarity measures)
and does not require feature independence assumptions.  It produces
well-calibrated probabilities suitable for threshold-based decisions.

**StandardScaler:**
Included for completeness and to facilitate potential future replacement
with a logistic regression or SVM baseline without changing the pipeline
interface.

### D. File and Column Schemas

**Input (all three source TSVs):**

| Column | Type | Description |
|---|---|---|
| `entity_id` | str | Unique ID (e.g. `S1-00001`, `S2-00002`) |
| `business_name` | str | Raw business name |
| `business_address` | str | Raw business address |
| `country` | str | Country name or ISO code |

**Output — `matching_results.tsv`:**

| Column | Description |
|---|---|
| `source1_entity_id` | Source 1 entity ID |
| `matched_entity_ids` | Comma-separated matched IDs (S2/S3 only), empty if none |

**Output — `candidate_pairs.tsv`:**

| Column | Description |
|---|---|
| `source1_entity_id` | Source 1 entity ID |
| `candidate_entity_ids` | Comma-separated candidate IDs from blocking |
