import argparse
import json
import math
import os
import time

from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import t as student_t

from representation_diagnostic_experiment import (
    all_client_ids,
    collect_client_client_pairs,
    deterministic_seed,
    sample_negative_pairs,
)

from link_prediction_experiment import (
    DESCRIPTOR_SPECS,
    BASELINE_DESCRIPTOR,
    build_random_spanning_forest_split,
    sample_pairs,
    construct_training_graph,
    generate_selected_descriptors,
    filter_pairs,
    rebalance_pair_sets,
    score_pairs,
    binary_metrics_from_scores,
    bootstrap_metric_intervals,
    paired_bootstrap_contrast,
    retrieval_metrics,
    pair_manifest_dataframe,
    resolve_holdout_policy,
)


MULTISEED_SCHEMA_VERSION = "1.0"

DEFAULT_OUTPUT_FOLDER = "output_local_test"

DEFAULT_SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

DEFAULT_TEST_POSITIVES = 1000

DEFAULT_NEGATIVE_RATIO = 1.0

DEFAULT_BOOTSTRAP_REPETITIONS = 500

DEFAULT_D_MAX = 50

DEFAULT_DEGREE_WEIGHT = 1.0

DEFAULT_K_VALUES = (
    10,
    20,
    50,
)

DEFAULT_HOLDOUT_POLICY = "auto"

NUMERICAL_EPS = 1e-12


DESCRIPTOR_COMPARISONS = (
    (
        "eigen_d8_block",
        "hks_d12_global",
    ),
    (
        "eigen_d12_global",
        "hks_d12_global",
    ),
    (
        "eigen_d8_block",
        "eigen_d12_global",
    ),
)


def mean_std_ci95(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    number = len(values)

    if number == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "n": 0,
        }

    mean = float(
        np.mean(values)
    )

    if number == 1:
        return {
            "mean": mean,
            "std": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
            "n": 1,
        }

    std = float(
        np.std(
            values,
            ddof=1,
        )
    )

    standard_error = (
        std
        /
        math.sqrt(number)
    )

    critical_value = float(
        student_t.ppf(
            0.975,
            df=number - 1,
        )
    )

    margin = (
        critical_value
        *
        standard_error
    )

    return {
        "mean": mean,
        "std": std,
        "ci95_low": float(
            mean - margin
        ),
        "ci95_high": float(
            mean + margin
        ),
        "n": int(number),
    }


def make_neighbour_sets(pairs):
    neighbours = defaultdict(
        set
    )

    for first, second in pairs:
        neighbours[
            first
        ].add(
            second
        )

        neighbours[
            second
        ].add(
            first
        )

    return neighbours


def probability_no_relevant_in_top_k(
    candidate_count,
    relevant_count,
    k,
):


    N = int(candidate_count)
    m = int(relevant_count)
    k = min(
        int(k),
        N,
    )

    if m <= 0:
        return 1.0

    if k <= 0:
        return 1.0

    if k > N - m:
        return 0.0

    probability = 1.0

    for offset in range(k):
        numerator = (
            N
            -
            m
            -
            offset
        )

        denominator = (
            N
            -
            offset
        )

        if numerator <= 0:
            return 0.0

        probability *= (
            numerator
            /
            denominator
        )

    return float(
        probability
    )


def expected_random_reciprocal_rank(
    candidate_count,
    relevant_count,
):


    N = int(candidate_count)
    m = int(relevant_count)

    if (
        N <= 0
        or m <= 0
    ):
        return 0.0

    if m > N:
        raise ValueError(
            "relevant_count cannot exceed candidate_count."
        )

    survival = 1.0

    expected_rr = 0.0

    maximum_rank = (
        N
        -
        m
        +
        1
    )

    for rank in range(
        1,
        maximum_rank + 1,
    ):
        remaining_positions = (
            N
            -
            rank
            +
            1
        )

        probability_relevant_now = (
            m
            /
            remaining_positions
        )

        probability_first_relevant = (
            survival
            *
            probability_relevant_now
        )

        expected_rr += (
            probability_first_relevant
            /
            rank
        )

        probability_irrelevant_now = (
            (
                remaining_positions
                -
                m
            )
            /
            remaining_positions
        )

        survival *= (
            probability_irrelevant_now
        )

        if survival <= 1e-15:
            break

    return float(
        expected_rr
    )


def random_retrieval_baseline(
    client_ids,
    training_positive_pairs,
    test_positive_pairs,
    k_values,
):


    client_ids = np.asarray(
        client_ids,
        dtype=str,
    )

    valid_client_set = set(
        client_ids.tolist()
    )

    training_neighbours = (
        make_neighbour_sets(
            training_positive_pairs
        )
    )

    test_neighbours = (
        make_neighbour_sets(
            test_positive_pairs
        )
    )

    recall_values = {
        int(k): []
        for k in k_values
    }

    precision_values = {
        int(k): []
        for k in k_values
    }

    hit_values = {
        int(k): []
        for k in k_values
    }

    reciprocal_rank_values = []

    cache_mrr = {}

    roots = 0

    total_clients = len(
        client_ids
    )

    for root, truth in (
        test_neighbours.items()
    ):
        if root not in valid_client_set:
            continue

        valid_truth = {
            neighbour
            for neighbour
            in truth
            if neighbour in valid_client_set
        }

        if not valid_truth:
            continue

        excluded_training = {
            neighbour
            for neighbour
            in training_neighbours.get(
                root,
                set(),
            )
            if neighbour in valid_client_set
        }

        candidate_count = (
            total_clients
            -
            1
            -
            len(
                excluded_training
            )
        )

        relevant_count = len(
            valid_truth
        )

        if candidate_count <= 0:
            continue

        if relevant_count > candidate_count:
            raise RuntimeError(
                "Random baseline candidate accounting failed."
            )

        roots += 1


        cache_key = (
            candidate_count,
            relevant_count,
        )

        if cache_key not in cache_mrr:
            cache_mrr[
                cache_key
            ] = (
                expected_random_reciprocal_rank(
                    candidate_count,
                    relevant_count,
                )
            )

        reciprocal_rank_values.append(
            cache_mrr[
                cache_key
            ]
        )


        for requested_k in k_values:
            requested_k = int(
                requested_k
            )

            effective_k = min(
                requested_k,
                candidate_count,
            )


            expected_recall = (
                effective_k
                /
                candidate_count
            )


            expected_precision = (
                relevant_count
                /
                candidate_count
            )

            probability_none = (
                probability_no_relevant_in_top_k(
                    candidate_count,
                    relevant_count,
                    effective_k,
                )
            )

            expected_hit_rate = (
                1.0
                -
                probability_none
            )

            recall_values[
                requested_k
            ].append(
                expected_recall
            )

            precision_values[
                requested_k
            ].append(
                expected_precision
            )

            hit_values[
                requested_k
            ].append(
                expected_hit_rate
            )

    result = {
        "random_retrieval_roots": int(
            roots
        ),

        "random_mrr": (
            float(
                np.mean(
                    reciprocal_rank_values
                )
            )
            if reciprocal_rank_values
            else np.nan
        ),
    }

    for k in k_values:
        k = int(k)

        result[
            f"random_recall_at_{k}"
        ] = (
            float(
                np.mean(
                    recall_values[k]
                )
            )
            if recall_values[k]
            else np.nan
        )

        result[
            f"random_precision_at_{k}"
        ] = (
            float(
                np.mean(
                    precision_values[k]
                )
            )
            if precision_values[k]
            else np.nan
        )

        result[
            f"random_hit_rate_at_{k}"
        ] = (
            float(
                np.mean(
                    hit_values[k]
                )
            )
            if hit_values[k]
            else np.nan
        )

    return result


def build_split_dataframe(
    full_positive_pairs,
    forest_pairs,
    test_positive_pairs,
    pair_relations,
):
    forest_set = set(
        forest_pairs
    )

    test_set = set(
        test_positive_pairs
    )

    rows = []

    for pair in full_positive_pairs:
        if pair in test_set:
            split_name = "test"

        elif pair in forest_set:
            split_name = (
                "train_forced_forest"
            )

        else:
            split_name = (
                "train_cycle"
            )

        rows.append({
            "source": pair[0],
            "target": pair[1],
            "split": split_name,
            "relations": "|".join(
                sorted(
                    pair_relations.get(
                        pair,
                        set(),
                    )
                )
            ),
        })

    return pd.DataFrame(
        rows
    )


def run_one_seed(
    original_graph,
    dataset_name,
    clients,
    full_positive_pairs,
    pair_relations,
    seed,
    test_positives,
    negative_ratio,
    holdout_policy,
    bootstrap_repetitions,
    d_max,
    degree_weight,
    k_values,
    seed_folder,
):
    seed_start = time.perf_counter()

    print()
    print(
        "-" * 70
    )
    print(
        f"GRAPH-SPLIT SEED {seed}"
    )
    print(
        "-" * 70
    )


    split = (
        build_random_spanning_forest_split(
            client_ids=clients,
            positive_pairs=(
                full_positive_pairs
            ),
            seed=deterministic_seed(
                dataset_name,
                seed,
                "spanning_forest",
            ),
        )
    )

    forest_pairs = split[
        "forest_pairs"
    ]

    cycle_pairs = split[
        "cycle_pairs"
    ]

    if not cycle_pairs:
        raise RuntimeError(
            f"Seed {seed}: no cycle test candidates."
        )


    test_positive_pairs = sample_pairs(
        cycle_pairs,
        test_positives,
        deterministic_seed(
            dataset_name,
            seed,
            "test_positive_pairs",
        ),
    )


    requested_negatives = int(
        round(
            len(
                test_positive_pairs
            )
            *
            float(
                negative_ratio
            )
        )
    )

    negative_pairs = sample_negative_pairs(
        client_ids=clients,
        full_positive_pairs=(
            full_positive_pairs
        ),
        requested=(
            requested_negatives
        ),
        seed=deterministic_seed(
            dataset_name,
            seed,
            "test_negative_pairs",
        ),
    )


    holdout = (
        construct_training_graph(
            original_graph=(
                original_graph
            ),
            test_positive_pairs=(
                test_positive_pairs
            ),
            dataset_name=(
                dataset_name
            ),
            holdout_policy=(
                holdout_policy
            ),
            seed=seed,
        )
    )

    training_graph = holdout[
        "graph"
    ]


    training_pair_information = (
        collect_client_client_pairs(
            training_graph
        )
    )

    training_positive_pairs = set(
        training_pair_information[
            "pair_relations"
        ].keys()
    )

    test_positive_set = set(
        test_positive_pairs
    )

    leakage_intersection = (
        training_positive_pairs
        &
        test_positive_set
    )

    if leakage_intersection:
        raise RuntimeError(
            f"Seed {seed}: "
            f"{len(leakage_intersection)} "
            "test CC pairs remain in training."
        )

    missing_forest = (
        set(
            forest_pairs
        )
        -
        training_positive_pairs
    )

    if missing_forest:
        raise RuntimeError(
            f"Seed {seed}: "
            f"{len(missing_forest)} "
            "forced forest pairs disappeared."
        )


    generated = (
        generate_selected_descriptors(
            training_graph=training_graph,
            client_ids=clients,
            d_max=d_max,
            degree_weight=degree_weight,
        )
    )

    descriptor_client_ids = (
        generated[
            "client_ids"
        ]
    )

    descriptor_matrices = (
        generated[
            "descriptors"
        ]
    )

    valid_client_set = set(
        descriptor_client_ids.tolist()
    )


    test_positive_pairs = filter_pairs(
        test_positive_pairs,
        valid_client_set,
    )

    negative_pairs = filter_pairs(
        negative_pairs,
        valid_client_set,
    )

    (
        test_positive_pairs,
        negative_pairs,
    ) = rebalance_pair_sets(
        test_positive_pairs,
        negative_pairs,
        seed=deterministic_seed(
            dataset_name,
            seed,
            "rebalance",
        ),
    )

    valid_training_pairs = [
        pair
        for pair
        in training_positive_pairs
        if (
            pair[0]
            in valid_client_set
            and
            pair[1]
            in valid_client_set
        )
    ]


    random_baseline = (
        random_retrieval_baseline(
            client_ids=(
                descriptor_client_ids
            ),
            training_positive_pairs=(
                valid_training_pairs
            ),
            test_positive_pairs=(
                test_positive_pairs
            ),
            k_values=k_values,
        )
    )


    random_binary_baseline = {
        "random_auc": 0.5,

        "random_ap": float(
            len(
                test_positive_pairs
            )
            /
            (
                len(
                    test_positive_pairs
                )
                +
                len(
                    negative_pairs
                )
            )
        ),
    }


    metric_rows = []

    score_cache = {}

    pair_score_dataframe = (
        pair_manifest_dataframe(
            test_positive_pairs,
            negative_pairs,
            pair_relations,
        )
    )

    for specification in (
        DESCRIPTOR_SPECS
    ):
        descriptor_name = (
            specification[
                "name"
            ]
        )

        matrix = descriptor_matrices[
            descriptor_name
        ]

        positive_scores = score_pairs(
            matrix,
            descriptor_client_ids,
            test_positive_pairs,
        )

        negative_scores = score_pairs(
            matrix,
            descriptor_client_ids,
            negative_pairs,
        )

        score_cache[
            descriptor_name
        ] = {
            "positive": (
                positive_scores
            ),
            "negative": (
                negative_scores
            ),
        }

        binary = (
            binary_metrics_from_scores(
                positive_scores,
                negative_scores,
            )
        )

        bootstrap = (
            bootstrap_metric_intervals(
                positive_scores,
                negative_scores,
                repetitions=(
                    bootstrap_repetitions
                ),
                seed=deterministic_seed(
                    dataset_name,
                    seed,
                    descriptor_name,
                    "bootstrap",
                ),
            )
        )

        retrieval = retrieval_metrics(
            descriptor_matrix=matrix,
            client_ids=(
                descriptor_client_ids
            ),
            training_positive_pairs=(
                valid_training_pairs
            ),
            test_positive_pairs=(
                test_positive_pairs
            ),
            k_values=k_values,
        )

        row = {
            "seed": int(
                seed
            ),

            "descriptor": (
                descriptor_name
            ),

            "family": (
                specification[
                    "family"
                ]
            ),

            "dimension": int(
                specification[
                    "dimension"
                ]
            ),

            "normalization": (
                specification[
                    "normalization"
                ]
            ),

            "positive_pairs": int(
                len(
                    test_positive_pairs
                )
            ),

            "negative_pairs": int(
                len(
                    negative_pairs
                )
            ),

            **binary,
            **bootstrap,
            **retrieval,
            **random_binary_baseline,
            **random_baseline,
        }


        for k in k_values:
            k = int(k)

            row[
                f"recall_at_{k}_minus_random"
            ] = (
                row[
                    f"recall_at_{k}"
                ]
                -
                row[
                    f"random_recall_at_{k}"
                ]
            )

            row[
                f"hit_rate_at_{k}_minus_random"
            ] = (
                row[
                    f"hit_rate_at_{k}"
                ]
                -
                row[
                    f"random_hit_rate_at_{k}"
                ]
            )

        row[
            "mrr_minus_random"
        ] = (
            row[
                "mrr"
            ]
            -
            row[
                "random_mrr"
            ]
        )

        metric_rows.append(
            row
        )

        pair_score_dataframe[
            descriptor_name
        ] = np.concatenate(
            [
                positive_scores,
                negative_scores,
            ]
        )

    metrics_dataframe = pd.DataFrame(
        metric_rows
    )


    contrast_rows = []

    for candidate, reference in (
        DESCRIPTOR_COMPARISONS
    ):
        if (
            candidate not in score_cache
            or reference not in score_cache
        ):
            continue

        candidate_scores = (
            score_cache[
                candidate
            ]
        )

        reference_scores = (
            score_cache[
                reference
            ]
        )

        contrast = (
            paired_bootstrap_contrast(
                baseline_positive=(
                    reference_scores[
                        "positive"
                    ]
                ),
                baseline_negative=(
                    reference_scores[
                        "negative"
                    ]
                ),
                candidate_positive=(
                    candidate_scores[
                        "positive"
                    ]
                ),
                candidate_negative=(
                    candidate_scores[
                        "negative"
                    ]
                ),
                repetitions=(
                    bootstrap_repetitions
                ),
                seed=deterministic_seed(
                    dataset_name,
                    seed,
                    candidate,
                    reference,
                    "paired_bootstrap",
                ),
            )
        )

        contrast_rows.append({
            "seed": int(
                seed
            ),
            "candidate": (
                candidate
            ),
            "reference": (
                reference
            ),
            **contrast,
        })

    contrast_dataframe = pd.DataFrame(
        contrast_rows
    )


    relation_counts = defaultdict(
        int
    )

    for pair in test_positive_pairs:
        for relation in (
            pair_relations.get(
                pair,
                set(),
            )
        ):
            relation_counts[
                relation
            ] += 1


    os.makedirs(
        seed_folder,
        exist_ok=True,
    )

    metrics_path = os.path.join(
        seed_folder,
        "metrics.csv",
    )

    contrasts_path = os.path.join(
        seed_folder,
        "paired_contrasts.csv",
    )

    scores_path = os.path.join(
        seed_folder,
        "pairs_and_scores.csv.gz",
    )

    split_path = os.path.join(
        seed_folder,
        "split.csv",
    )

    leakage_path = os.path.join(
        seed_folder,
        "leakage_diagnostics.csv",
    )

    metadata_path = os.path.join(
        seed_folder,
        "metadata.json",
    )

    metrics_dataframe.to_csv(
        metrics_path,
        index=False,
    )

    contrast_dataframe.to_csv(
        contrasts_path,
        index=False,
    )

    pair_score_dataframe.to_csv(
        scores_path,
        index=False,
        compression="gzip",
    )

    build_split_dataframe(
        full_positive_pairs,
        forest_pairs,
        test_positive_pairs,
        pair_relations,
    ).to_csv(
        split_path,
        index=False,
    )

    if holdout[
        "pair_diagnostics"
    ]:
        pd.DataFrame(
            holdout[
                "pair_diagnostics"
            ]
        ).to_csv(
            leakage_path,
            index=False,
        )
    else:
        pd.DataFrame(
            columns=[
                "source",
                "target",
                "original_common_content",
                "content_incidence_edges_removed",
            ]
        ).to_csv(
            leakage_path,
            index=False,
        )

    seed_runtime = float(
        time.perf_counter()
        -
        seed_start
    )

    seed_metadata = {
        "schema_version": (
            MULTISEED_SCHEMA_VERSION
        ),

        "seed": int(seed),

        "holdout_policy": (
            holdout_policy
        ),

        "client_nodes": int(
            len(clients)
        ),

        "original_positive_pairs": int(
            len(
                full_positive_pairs
            )
        ),

        "forest_pairs": int(
            len(
                forest_pairs
            )
        ),

        "cycle_candidates": int(
            len(
                cycle_pairs
            )
        ),

        "final_positive_pairs": int(
            len(
                test_positive_pairs
            )
        ),

        "final_negative_pairs": int(
            len(
                negative_pairs
            )
        ),

        "descriptor_clients": int(
            len(
                descriptor_client_ids
            )
        ),

        "descriptor_failures": (
            generated[
                "failures"
            ]
        ),

        "direct_edges_removed": int(
            holdout[
                "direct_edges_removed"
            ]
        ),

        "content_edges_removed": int(
            holdout[
                "content_edges_removed"
            ]
        ),

        "test_pairs_with_original_common_content": int(
            holdout[
                "test_pairs_with_original_common_content"
            ]
        ),

        "original_common_content_instances": int(
            holdout[
                "original_common_content_instances"
            ]
        ),

        "direct_cc_leakage_remaining": 0,

        "missing_forced_forest_pairs": 0,

        "test_relation_counts": dict(
            relation_counts
        ),

        "random_baseline": {
            **random_binary_baseline,
            **random_baseline,
        },

        "runtime_seconds": (
            seed_runtime
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            seed_metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )


    print(
        f"Seed {seed} complete."
    )

    print(
        f"  positives = "
        f"{len(test_positive_pairs):,}"
    )

    print(
        f"  negatives = "
        f"{len(negative_pairs):,}"
    )

    print(
        f"  CC leakage = 0"
    )

    if holdout_policy == (
        "strict_common_content"
    ):
        print(
            "  common-content leakage = 0"
        )

    print()
    print(
        metrics_dataframe[
            [
                "descriptor",
                "auc",
                "ap",
                "recall_at_20",
                "random_recall_at_20",
                "hit_rate_at_20",
                "random_hit_rate_at_20",
                "mrr",
                "random_mrr",
            ]
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Paired contrasts:"
    )

    print(
        contrast_dataframe[
            [
                "candidate",
                "reference",
                "auc_difference_mean",
                "auc_difference_ci95_low",
                "auc_difference_ci95_high",
                "ap_difference_mean",
                "ap_difference_ci95_low",
                "ap_difference_ci95_high",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        f"\nSeed runtime: "
        f"{seed_runtime:.3f} s"
    )

    return {
        "metrics": (
            metrics_dataframe
        ),
        "contrasts": (
            contrast_dataframe
        ),
        "metadata": (
            seed_metadata
        ),
    }


def aggregate_descriptor_metrics(
    seed_results,
):
    metrics = [
        "auc",
        "ap",
        "positive_similarity_mean",
        "negative_similarity_mean",
        "similarity_mean_gap",
        "recall_at_10",
        "recall_at_20",
        "recall_at_50",
        "hit_rate_at_10",
        "hit_rate_at_20",
        "hit_rate_at_50",
        "mrr",
        "recall_at_10_minus_random",
        "recall_at_20_minus_random",
        "recall_at_50_minus_random",
        "hit_rate_at_10_minus_random",
        "hit_rate_at_20_minus_random",
        "hit_rate_at_50_minus_random",
        "mrr_minus_random",
    ]

    rows = []

    for descriptor, group in (
        seed_results.groupby(
            "descriptor"
        )
    ):
        first = group.iloc[
            0
        ]

        row = {
            "descriptor": descriptor,

            "family": first[
                "family"
            ],

            "dimension": int(
                first[
                    "dimension"
                ]
            ),

            "normalization": first[
                "normalization"
            ],

            "num_split_seeds": int(
                group[
                    "seed"
                ].nunique()
            ),
        }

        for metric in metrics:
            if metric not in (
                group.columns
            ):
                continue

            statistics = mean_std_ci95(
                group[
                    metric
                ].to_numpy(
                    dtype=float
                )
            )

            row[
                f"{metric}__mean"
            ] = statistics[
                "mean"
            ]

            row[
                f"{metric}__std"
            ] = statistics[
                "std"
            ]

            row[
                f"{metric}__ci95_low"
            ] = statistics[
                "ci95_low"
            ]

            row[
                f"{metric}__ci95_high"
            ] = statistics[
                "ci95_high"
            ]

        rows.append(
            row
        )

    dataframe = pd.DataFrame(
        rows
    )

    if len(dataframe):
        dataframe = (
            dataframe
            .sort_values(
                [
                    "auc__mean",
                    "ap__mean",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

        dataframe.insert(
            0,
            "rank",
            np.arange(
                1,
                len(dataframe) + 1,
            ),
        )

    return dataframe


def aggregate_cross_seed_contrasts(
    seed_results,
):


    metrics = [
        "auc",
        "ap",
        "recall_at_20",
        "hit_rate_at_20",
        "mrr",
    ]

    rows = []

    for candidate, reference in (
        DESCRIPTOR_COMPARISONS
    ):

        candidate_rows = seed_results[
            seed_results[
                "descriptor"
            ]
            ==
            candidate
        ]

        reference_rows = seed_results[
            seed_results[
                "descriptor"
            ]
            ==
            reference
        ]

        merged = candidate_rows.merge(
            reference_rows,
            on="seed",
            suffixes=(
                "_candidate",
                "_reference",
            ),
        )

        if not len(merged):
            continue

        for metric in metrics:
            candidate_values = merged[
                f"{metric}_candidate"
            ].to_numpy(
                dtype=float
            )

            reference_values = merged[
                f"{metric}_reference"
            ].to_numpy(
                dtype=float
            )

            differences = (
                candidate_values
                -
                reference_values
            )

            statistics = mean_std_ci95(
                differences
            )

            rows.append({
                "candidate": candidate,
                "reference": reference,
                "metric": metric,

                "paired_split_seeds": int(
                    len(
                        differences
                    )
                ),

                "candidate_mean": float(
                    np.mean(
                        candidate_values
                    )
                ),

                "reference_mean": float(
                    np.mean(
                        reference_values
                    )
                ),

                "difference_mean": (
                    statistics[
                        "mean"
                    ]
                ),

                "difference_std": (
                    statistics[
                        "std"
                    ]
                ),

                "difference_ci95_low": (
                    statistics[
                        "ci95_low"
                    ]
                ),

                "difference_ci95_high": (
                    statistics[
                        "ci95_high"
                    ]
                ),

                "seeds_candidate_better": int(
                    np.sum(
                        differences > 0
                    )
                ),

                "seeds_reference_better": int(
                    np.sum(
                        differences < 0
                    )
                ),

                "seed_ties": int(
                    np.sum(
                        np.abs(
                            differences
                        )
                        <=
                        1e-12
                    )
                ),
            })

    return pd.DataFrame(
        rows
    )


def run_multiseed_validation(
    dataset_name="enron",
    output_folder=DEFAULT_OUTPUT_FOLDER,
    seeds=DEFAULT_SEEDS,
    test_positives=DEFAULT_TEST_POSITIVES,
    negative_ratio=DEFAULT_NEGATIVE_RATIO,
    holdout_policy=DEFAULT_HOLDOUT_POLICY,
    bootstrap_repetitions=(
        DEFAULT_BOOTSTRAP_REPETITIONS
    ),
    d_max=DEFAULT_D_MAX,
    degree_weight=DEFAULT_DEGREE_WEIGHT,
    k_values=DEFAULT_K_VALUES,
):
    total_start = time.perf_counter()

    seeds = tuple(
        int(seed)
        for seed
        in seeds
    )

    if len(seeds) < 2:
        raise ValueError(
            "Use at least two graph-split seeds."
        )

    holdout_policy = (
        resolve_holdout_policy(
            dataset_name,
            holdout_policy,
        )
    )

    if holdout_policy not in {
        "direct_only",
        "strict_common_content",
    }:
        raise ValueError(
            "Invalid holdout policy."
        )


    graph_path = os.path.join(
        output_folder,
        dataset_name,
        f"{dataset_name}_global_graph.gml.gz",
    )

    if not os.path.exists(
        graph_path
    ):
        raise FileNotFoundError(
            f"Missing graph:\n"
            f"  {graph_path}"
        )

    print()
    print(
        "=" * 72
    )
    print(
        "MULTI-SEED LEAKAGE-SAFE LINK-PREDICTION VALIDATION"
    )
    print(
        "=" * 72
    )

    print(
        f"Dataset: {dataset_name}"
    )

    print(
        f"Seeds: {list(seeds)}"
    )

    print(
        f"Holdout policy: "
        f"{holdout_policy}"
    )

    original_graph = nx.read_gml(
        graph_path
    )

    clients = all_client_ids(
        original_graph
    )

    original_pair_information = (
        collect_client_client_pairs(
            original_graph
        )
    )

    pair_relations = (
        original_pair_information[
            "pair_relations"
        ]
    )

    full_positive_pairs = sorted(
        pair_relations.keys()
    )

    print(
        f"Client nodes: "
        f"{len(clients):,}"
    )

    print(
        "Original unique client-client pairs: "
        f"{len(full_positive_pairs):,}"
    )


    experiment_folder = os.path.join(
        output_folder,
        dataset_name,
        "experiments",
        "link_prediction_multiseed",
    )

    os.makedirs(
        experiment_folder,
        exist_ok=True,
    )

    all_metric_frames = []

    all_contrast_frames = []

    seed_metadata = []


    for seed_index, seed in enumerate(
        seeds,
        start=1,
    ):
        print()
        print(
            "=" * 72
        )

        print(
            f"SEED {seed_index}/"
            f"{len(seeds)}: {seed}"
        )

        print(
            "=" * 72
        )

        seed_folder = os.path.join(
            experiment_folder,
            f"seed_{seed}",
        )

        result = run_one_seed(
            original_graph=(
                original_graph
            ),

            dataset_name=(
                dataset_name
            ),

            clients=clients,

            full_positive_pairs=(
                full_positive_pairs
            ),

            pair_relations=(
                pair_relations
            ),

            seed=seed,

            test_positives=(
                test_positives
            ),

            negative_ratio=(
                negative_ratio
            ),

            holdout_policy=(
                holdout_policy
            ),

            bootstrap_repetitions=(
                bootstrap_repetitions
            ),

            d_max=d_max,

            degree_weight=(
                degree_weight
            ),

            k_values=k_values,

            seed_folder=(
                seed_folder
            ),
        )

        all_metric_frames.append(
            result[
                "metrics"
            ]
        )

        all_contrast_frames.append(
            result[
                "contrasts"
            ]
        )

        seed_metadata.append(
            result[
                "metadata"
            ]
        )


    seed_results = pd.concat(
        all_metric_frames,
        ignore_index=True,
    )

    seed_bootstrap_contrasts = (
        pd.concat(
            all_contrast_frames,
            ignore_index=True,
        )
    )

    aggregate_results = (
        aggregate_descriptor_metrics(
            seed_results
        )
    )

    cross_seed_contrasts = (
        aggregate_cross_seed_contrasts(
            seed_results
        )
    )


    seed_results_path = os.path.join(
        experiment_folder,
        "multiseed_seed_results.csv",
    )

    aggregate_path = os.path.join(
        experiment_folder,
        "multiseed_aggregate.csv",
    )

    seed_contrasts_path = os.path.join(
        experiment_folder,
        "multiseed_within_seed_bootstrap_contrasts.csv",
    )

    cross_seed_contrasts_path = os.path.join(
        experiment_folder,
        "multiseed_cross_seed_contrasts.csv",
    )

    metadata_path = os.path.join(
        experiment_folder,
        "multiseed_metadata.json",
    )

    seed_results.to_csv(
        seed_results_path,
        index=False,
    )

    aggregate_results.to_csv(
        aggregate_path,
        index=False,
    )

    seed_bootstrap_contrasts.to_csv(
        seed_contrasts_path,
        index=False,
    )

    cross_seed_contrasts.to_csv(
        cross_seed_contrasts_path,
        index=False,
    )

    total_runtime = float(
        time.perf_counter()
        -
        total_start
    )

    metadata = {
        "schema_version": (
            MULTISEED_SCHEMA_VERSION
        ),

        "dataset": dataset_name,

        "global_graph": graph_path,

        "seeds": list(
            seeds
        ),

        "num_split_seeds": int(
            len(seeds)
        ),

        "holdout_policy": (
            holdout_policy
        ),

        "test_positives_requested_per_seed": int(
            test_positives
        ),

        "negative_ratio": float(
            negative_ratio
        ),

        "bootstrap_repetitions_per_seed": int(
            bootstrap_repetitions
        ),

        "descriptors": list(
            DESCRIPTOR_SPECS
        ),

        "comparisons": [
            {
                "candidate": candidate,
                "reference": reference,
            }
            for candidate, reference
            in DESCRIPTOR_COMPARISONS
        ],

        "important_statistical_note": (
            "Within-seed bootstrap intervals quantify test-pair "
            "sampling uncertainty conditional on a graph split. "
            "Cross-seed mean/std/t intervals quantify variability "
            "across independently generated spanning-forest and "
            "held-out-edge splits. These are distinct sources of "
            "variation and are reported separately."
        ),

        "random_baseline_note": (
            "Random Recall@K, Precision@K and HitRate@K are computed "
            "analytically for each root using its actual candidate "
            "count after excluding training neighbours. Random MRR "
            "is the exact expected reciprocal rank of the first "
            "relevant item under a uniformly random ranking."
        ),

        "selection_rule": (
            "Do not choose a descriptor from one metric alone. "
            "Eigen d8 is preferred over Eigen d12 only if its "
            "predictive performance is comparable/stable across "
            "splits while retaining its lower dimensionality and "
            "therefore lower K-norm noise cost."
        ),

        "privacy": (
            "NONE. This remains a clean descriptor-selection "
            "experiment."
        ),

        "per_seed_metadata": (
            seed_metadata
        ),

        "outputs": {
            "seed_results": os.path.basename(
                seed_results_path
            ),

            "aggregate": os.path.basename(
                aggregate_path
            ),

            "within_seed_bootstrap_contrasts": (
                os.path.basename(
                    seed_contrasts_path
                )
            ),

            "cross_seed_contrasts": (
                os.path.basename(
                    cross_seed_contrasts_path
                )
            ),
        },

        "runtime_seconds": (
            total_runtime
        ),
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )


    print()
    print(
        "=" * 72
    )

    print(
        "MULTI-SEED VALIDATION COMPLETE"
    )

    print(
        "=" * 72
    )

    print(
        f"Graph-split seeds: "
        f"{len(seeds)}"
    )

    print()
    print(
        "Aggregate descriptor results:"
    )

    aggregate_columns = [
        "rank",
        "descriptor",
        "auc__mean",
        "auc__std",
        "auc__ci95_low",
        "auc__ci95_high",
        "ap__mean",
        "ap__std",
        "recall_at_20__mean",
        "recall_at_20_minus_random__mean",
        "hit_rate_at_20__mean",
        "hit_rate_at_20_minus_random__mean",
        "mrr__mean",
        "mrr_minus_random__mean",
    ]

    available_aggregate_columns = [
        column
        for column
        in aggregate_columns
        if column
        in aggregate_results.columns
    ]

    print(
        aggregate_results[
            available_aggregate_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Cross-seed paired comparisons:"
    )

    if len(
        cross_seed_contrasts
    ):
        print(
            cross_seed_contrasts[
                [
                    "candidate",
                    "reference",
                    "metric",
                    "difference_mean",
                    "difference_std",
                    "difference_ci95_low",
                    "difference_ci95_high",
                    "seeds_candidate_better",
                    "seeds_reference_better",
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "Interpretation rule:"
    )

    print(
        "  1. Eigen descriptors should remain above "
        "HKS across independent graph splits."
    )

    print(
        "  2. Check whether Eigen8 vs Eigen12 AUC/AP "
        "differences consistently favor either model."
    )

    print(
        "  3. If Eigen8 and Eigen12 are statistically "
        "comparable, prefer Eigen8 because it has "
        "33.3% fewer dimensions."
    )

    print(
        "  4. Retrieval metrics should also be compared "
        "against their analytical random baselines."
    )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  {seed_results_path}"
    )

    print(
        f"  {aggregate_path}"
    )

    print(
        f"  {seed_contrasts_path}"
    )

    print(
        f"  {cross_seed_contrasts_path}"
    )

    print(
        f"  {metadata_path}"
    )

    print()
    print(
        f"Total runtime: "
        f"{total_runtime:.3f} s"
    )

    print(
        "=" * 72
    )

    print()

    return {
        "seed_results": (
            seed_results
        ),

        "aggregate": (
            aggregate_results
        ),

        "within_seed_contrasts": (
            seed_bootstrap_contrasts
        ),

        "cross_seed_contrasts": (
            cross_seed_contrasts
        ),

        "metadata": (
            metadata
        ),
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Validate leakage-safe local graph descriptors "
            "over multiple independent graph splits."
        )
    )

    parser.add_argument(
        "dataset",
        type=str,
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default=(
            DEFAULT_OUTPUT_FOLDER
        ),
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(
            DEFAULT_SEEDS
        ),
    )

    parser.add_argument(
        "--test-positives",
        type=int,
        default=(
            DEFAULT_TEST_POSITIVES
        ),
    )

    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=(
            DEFAULT_NEGATIVE_RATIO
        ),
    )

    parser.add_argument(
        "--holdout-policy",
        choices=[
            "auto",
            "direct_only",
            "strict_common_content",
        ],
        default=(
            DEFAULT_HOLDOUT_POLICY
        ),
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=(
            DEFAULT_BOOTSTRAP_REPETITIONS
        ),
    )

    parser.add_argument(
        "--Dmax",
        type=int,
        default=(
            DEFAULT_D_MAX
        ),
    )

    parser.add_argument(
        "--degree-weight",
        type=float,
        default=(
            DEFAULT_DEGREE_WEIGHT
        ),
    )

    parser.add_argument(
        "--k",
        nargs="+",
        type=int,
        default=list(
            DEFAULT_K_VALUES
        ),
    )

    args = parser.parse_args()

    run_multiseed_validation(
        dataset_name=(
            args.dataset
        ),

        output_folder=(
            args.output_folder
        ),

        seeds=(
            args.seeds
        ),

        test_positives=(
            args.test_positives
        ),

        negative_ratio=(
            args.negative_ratio
        ),

        holdout_policy=(
            args.holdout_policy
        ),

        bootstrap_repetitions=(
            args.bootstrap
        ),

        d_max=(
            args.Dmax
        ),

        degree_weight=(
            args.degree_weight
        ),

        k_values=(
            args.k
        ),
    )
