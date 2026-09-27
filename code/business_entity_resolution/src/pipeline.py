import argparse
from pathlib import Path

import pandas as pd

from normalize import (
    normalize_name,
    normalize_address,
    normalize_country,
)

from features import build_features

from matcher import EntityMatcher


def prepare(df):

    df = df.copy()

    df["name_normalized"] = (
        df["business_name"]
        .map(normalize_name)
    )

    df["address_normalized"] = (
        df["business_address"]
        .map(normalize_address)
    )

    df["country_normalized"] = (
        df["country"]
        .map(normalize_country)
    )

    return df


def make_record(row):

    return {
        "entity_id": row["entity_id"],
        "name_normalized": row["name_normalized"],
        "address_normalized": row["address_normalized"],
        "country_normalized": row["country_normalized"],
    }


def candidate_generation(source1, targets):

    country_index = {}

    for record in targets:

        country = record["country_normalized"]

        country_index.setdefault(
            country,
            []
        ).append(record)

    candidates = {}

    for record in source1:

        country = record["country_normalized"]

        candidates[
            record["entity_id"]
        ] = country_index.get(
            country,
            []
        )

    return candidates


def score_candidates(
    source1,
    candidates,
    matcher,
    threshold,
):

    results = []
    candidate_output = []

    for left in source1:

        sid = left["entity_id"]

        possible = candidates.get(
            sid,
            []
        )

        feature_rows = []

        target_ids = []

        for right in possible:

            feature_rows.append(
                build_features(left, right)
            )

            target_ids.append(
                right["entity_id"]
            )

        if feature_rows:

            probabilities = (
                matcher.predict_probability(
                    feature_rows
                )
            )

        else:
            probabilities = []

        selected = [
            tid
            for tid, probability in zip(
                target_ids,
                probabilities,
            )
            if probability >= threshold
        ]

        candidate_output.append({
            "source1_entity_id": sid,
            "candidate_entity_ids": ",".join(
                dict.fromkeys(target_ids)
            ),
        })

        results.append({
            "source1_entity_id": sid,
            "matched_entity_ids": ",".join(
                dict.fromkeys(selected)
            ),
        })

    return (
        pd.DataFrame(results),
        pd.DataFrame(candidate_output),
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source1",
        required=True,
    )

    parser.add_argument(
        "--source2",
        required=True,
    )

    parser.add_argument(
        "--source3",
        required=True,
    )

    parser.add_argument(
        "--output",
        default="output",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
    )

    args = parser.parse_args()

    source1 = prepare(
        pd.read_csv(
            args.source1,
            sep="\t",
        )
    )

    source2 = prepare(
        pd.read_csv(
            args.source2,
            sep="\t",
        )
    )

    source3 = prepare(
        pd.read_csv(
            args.source3,
            sep="\t",
        )
    )

    s1_records = [
        make_record(row)
        for _, row in source1.iterrows()
    ]

    target_records = [
        make_record(row)
        for _, row in source2.iterrows()
    ]

    target_records.extend(
        make_record(row)
        for _, row in source3.iterrows()
    )

    candidates = candidate_generation(
        s1_records,
        target_records,
    )

    matcher = EntityMatcher()

    # Model training should be performed using
    # labeled candidate pairs from the training set.
    #
    # This pipeline intentionally keeps training
    # and inference separate.

    model_path = Path(
        args.output
    ) / "matcher.joblib"

    matcher.load(model_path)

    matching, candidate_pairs = (
        score_candidates(
            s1_records,
            candidates,
            matcher,
            args.threshold,
        )
    )

    output_dir = Path(args.output)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    matching.to_csv(
        output_dir / "matching_results.tsv",
        sep="\t",
        index=False,
    )

    candidate_pairs.to_csv(
        output_dir / "candidate_pairs.tsv",
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    main()