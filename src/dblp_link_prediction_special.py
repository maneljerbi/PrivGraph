import argparse
import hashlib
import json
import math
import os
import time

from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import t as student_t

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from link_prediction_experiment import (
    generate_selected_descriptors,
    score_pairs,
)


SCHEMA_VERSION = "dblp-link-special-1.1"

DEFAULT_OUTPUT_FOLDER = "output_local_test"

DEFAULT_SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

DEFAULT_COHORTS = (
    "all_authors",
    "provided_id_only",
)

DEFAULT_TEST_POSITIVES = 1000

DEFAULT_NEGATIVE_RATIO = 1.0

DEFAULT_BOOTSTRAP_REPETITIONS = 500

DEFAULT_D_MAX = 50

DEFAULT_DEGREE_WEIGHT = 1.0


SCORE_ROUND_DECIMALS = 12


DESCRIPTORS = (
    "hks_d12_global",
    "eigen_d8_block",
    "eigen_d12_global",
)

ORIENTATIONS = (
    "similarity",
    "distance",
)


def deterministic_seed(*parts):
    text = "|".join(
        str(part)
        for part in parts
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="little",
        signed=False,
    )


def canonical_pair(
    first,
    second,
):
    first = str(first)
    second = str(second)

    if first <= second:
        return (
            first,
            second,
        )

    return (
        second,
        first,
    )


def pair_endpoints(
    positive_pairs,
    negative_pairs,
):
    endpoints = set()

    for first, second in (
        list(positive_pairs)
        +
        list(negative_pairs)
    ):
        endpoints.add(
            first
        )

        endpoints.add(
            second
        )

    return sorted(
        endpoints
    )


def stabilize_scores(
    scores,
):


    scores = np.asarray(
        scores,
        dtype=float,
    )

    return np.round(
        scores,
        decimals=SCORE_ROUND_DECIMALS,
    )


def mean_std_ci95(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    n = len(
        values
    )

    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }

    mean = float(
        np.mean(
            values
        )
    )

    if n == 1:
        return {
            "n": 1,
            "mean": mean,
            "std": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
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
        math.sqrt(
            n
        )
    )

    critical = float(
        student_t.ppf(
            0.975,
            df=n - 1,
        )
    )

    margin = (
        critical
        *
        standard_error
    )

    return {
        "n": int(
            n
        ),
        "mean": mean,
        "std": std,
        "ci95_low": float(
            mean - margin
        ),
        "ci95_high": float(
            mean + margin
        ),
    }


class UnionFind:

    def __init__(
        self,
        elements,
    ):
        self.parent = {
            element: element
            for element in elements
        }

        self.rank = {
            element: 0
            for element in elements
        }

    def find(
        self,
        element,
    ):
        parent = self.parent[
            element
        ]

        if parent != element:
            self.parent[
                element
            ] = self.find(
                parent
            )

        return self.parent[
            element
        ]

    def union(
        self,
        first,
        second,
    ):
        root_first = self.find(
            first
        )

        root_second = self.find(
            second
        )

        if root_first == root_second:
            return False

        rank_first = self.rank[
            root_first
        ]

        rank_second = self.rank[
            root_second
        ]

        if rank_first < rank_second:
            self.parent[
                root_first
            ] = root_second

        elif rank_first > rank_second:
            self.parent[
                root_second
            ] = root_first

        else:
            self.parent[
                root_second
            ] = root_first

            self.rank[
                root_first
            ] += 1

        return True


def build_dblp_coauthor_index(
    graph,
):
    print()
    print(
        "Building DBLP coauthor index..."
    )

    authors_by_paper = defaultdict(
        set
    )

    papers_by_author = defaultdict(
        set
    )

    authored_edge_count = 0


    for source, target, attributes in graph.edges(
        data=True
    ):
        if str(
            attributes.get(
                "relation",
                "",
            )
        ) != "authored":
            continue

        source_type = str(
            graph.nodes[
                source
            ].get(
                "type",
                "",
            )
        )

        target_type = str(
            graph.nodes[
                target
            ].get(
                "type",
                "",
            )
        )

        if (
            source_type == "author"
            and target_type == "paper"
        ):
            author_id = str(
                source
            )

            paper_id = str(
                target
            )

        elif (
            source_type == "paper"
            and target_type == "author"
        ):
            author_id = str(
                target
            )

            paper_id = str(
                source
            )

        else:
            continue

        authors_by_paper[
            paper_id
        ].add(
            author_id
        )

        papers_by_author[
            author_id
        ].add(
            paper_id
        )

        authored_edge_count += 1


    shared_papers = defaultdict(
        list
    )

    coauthor_occurrences = 0

    multi_author_papers = 0

    exactly_two_author_papers = 0

    for paper_id, authors in authors_by_paper.items():
        authors = sorted(
            authors
        )

        if len(
            authors
        ) < 2:
            continue

        multi_author_papers += 1

        if len(
            authors
        ) == 2:
            exactly_two_author_papers += 1

        for first, second in combinations(
            authors,
            2,
        ):
            pair = canonical_pair(
                first,
                second,
            )

            shared_papers[
                pair
            ].append(
                paper_id
            )

            coauthor_occurrences += 1

    all_pairs = set(
        shared_papers.keys()
    )


    strict_pairs = set()

    for pair, papers in shared_papers.items():
        if len(
            papers
        ) != 1:
            continue

        paper_id = papers[
            0
        ]

        paper_authors = authors_by_paper[
            paper_id
        ]

        if len(
            paper_authors
        ) != 2:
            continue

        strict_pairs.add(
            pair
        )

    print(
        f"  authored physical edges: "
        f"{authored_edge_count:,}"
    )

    print(
        f"  authors with papers: "
        f"{len(papers_by_author):,}"
    )

    print(
        f"  papers with authored edges: "
        f"{len(authors_by_paper):,}"
    )

    print(
        f"  multi-author papers: "
        f"{multi_author_papers:,}"
    )

    print(
        f"  exactly-two-author papers: "
        f"{exactly_two_author_papers:,}"
    )

    print(
        f"  unique coauthor pairs: "
        f"{len(all_pairs):,}"
    )

    print(
        f"  coauthor pair occurrences: "
        f"{coauthor_occurrences:,}"
    )

    print(
        "  strict unique-paper dyadic pairs: "
        f"{len(strict_pairs):,}"
    )

    return {
        "authors_by_paper": authors_by_paper,
        "papers_by_author": papers_by_author,
        "shared_papers": shared_papers,
        "all_pairs": all_pairs,
        "strict_pairs": strict_pairs,
        "authored_edge_count": int(
            authored_edge_count
        ),
        "multi_author_papers": int(
            multi_author_papers
        ),
        "exactly_two_author_papers": int(
            exactly_two_author_papers
        ),
        "coauthor_occurrences": int(
            coauthor_occurrences
        ),
    }


def cohort_author_ids(
    graph,
    cohort,
):
    authors = []

    for node, attributes in graph.nodes(
        data=True
    ):
        if str(
            attributes.get(
                "type",
                "",
            )
        ) != "author":
            continue

        if str(
            attributes.get(
                "role",
                "",
            )
        ) != "client":
            continue

        if cohort == "all_authors":
            authors.append(
                str(
                    node
                )
            )

        elif cohort == "provided_id_only":
            if str(
                attributes.get(
                    "identity_source",
                    "",
                )
            ) == "provided_id":
                authors.append(
                    str(
                        node
                    )
                )

        else:
            raise ValueError(
                f"Unknown cohort: {cohort}"
            )

    return sorted(
        authors
    )


def build_random_spanning_forest(
    authors,
    positive_pairs,
    seed,
):
    authors = list(
        authors
    )

    positive_pairs = sorted(
        positive_pairs
    )

    union_find = UnionFind(
        authors
    )

    rng = np.random.default_rng(
        seed
    )

    permutation = rng.permutation(
        len(
            positive_pairs
        )
    )

    forest_pairs = set()

    for index in permutation:
        first, second = positive_pairs[
            int(
                index
            )
        ]

        if union_find.union(
            first,
            second,
        ):
            forest_pairs.add(
                canonical_pair(
                    first,
                    second,
                )
            )

    return forest_pairs


def sample_positive_pairs(
    candidates,
    requested,
    seed,
):
    candidates = sorted(
        candidates
    )

    if not candidates:
        raise RuntimeError(
            "No eligible DBLP test-positive pairs."
        )

    requested = int(
        requested
    )

    available = len(
        candidates
    )

    if available < requested:
        raise RuntimeError(
            "Not enough eligible DBLP test-positive pairs "
            "for the requested common test size.\n"
            f"  requested={requested:,}\n"
            f"  available={available:,}\n"
            "Choose a smaller common test size for all "
            "seeds and cohorts."
        )

    sample_size = requested

    rng = np.random.default_rng(
        seed
    )

    selected_indices = rng.choice(
        len(
            candidates
        ),
        size=sample_size,
        replace=False,
    )

    return [
        candidates[
            int(
                index
            )
        ]
        for index in selected_indices
    ]


def sample_negative_pairs(
    authors,
    full_positive_pairs,
    requested,
    seed,
):
    authors = np.asarray(
        authors,
        dtype=str,
    )

    positive_set = set(
        full_positive_pairs
    )

    requested = int(
        requested
    )

    if len(
        authors
    ) < 2:
        raise RuntimeError(
            "Too few authors for negative sampling."
        )

    rng = np.random.default_rng(
        seed
    )

    negatives = set()

    maximum_attempts = max(
        requested * 200,
        100000,
    )

    attempts = 0

    while (
        len(
            negatives
        ) < requested
        and attempts < maximum_attempts
    ):
        attempts += 1

        indices = rng.choice(
            len(
                authors
            ),
            size=2,
            replace=False,
        )

        first = str(
            authors[
                indices[
                    0
                ]
            ]
        )

        second = str(
            authors[
                indices[
                    1
                ]
            ]
        )

        pair = canonical_pair(
            first,
            second,
        )

        if pair in positive_set:
            continue

        negatives.add(
            pair
        )

    if len(
        negatives
    ) < requested:
        raise RuntimeError(
            "Could not sample enough true DBLP "
            "negative author pairs."
        )

    return sorted(
        negatives
    )


def authored_edge_records_between(
    graph,
    author_id,
    paper_id,
):
    records = []


    if graph.has_edge(
        author_id,
        paper_id,
    ):
        edge_data = graph.get_edge_data(
            author_id,
            paper_id,
        )

        for key, attributes in edge_data.items():
            if str(
                attributes.get(
                    "relation",
                    "",
                )
            ) == "authored":
                records.append(
                    (
                        author_id,
                        paper_id,
                        key,
                        dict(
                            attributes
                        ),
                    )
                )


    if graph.has_edge(
        paper_id,
        author_id,
    ):
        edge_data = graph.get_edge_data(
            paper_id,
            author_id,
        )

        for key, attributes in edge_data.items():
            if str(
                attributes.get(
                    "relation",
                    "",
                )
            ) == "authored":
                records.append(
                    (
                        paper_id,
                        author_id,
                        key,
                        dict(
                            attributes
                        ),
                    )
                )

    return records


def remove_authored_records(
    graph,
    records,
):
    for (
        source,
        target,
        key,
        _,
    ) in records:
        graph.remove_edge(
            source,
            target,
            key=key,
        )


def restore_authored_records(
    graph,
    records,
):
    for (
        source,
        target,
        key,
        attributes,
    ) in records:
        graph.add_edge(
            source,
            target,
            key=key,
            **attributes,
        )


def select_cut_endpoint(
    pair,
    seed,
):
    first, second = pair

    value = deterministic_seed(
        first,
        second,
        seed,
        "dblp_cut_endpoint",
    )

    if value % 2 == 0:
        return first

    return second


def current_authored_papers(
    graph,
    author_id,
):
    papers = set()

    if author_id not in graph:
        return papers


    for _, target, attributes in graph.out_edges(
        author_id,
        data=True,
    ):
        if str(
            attributes.get(
                "relation",
                "",
            )
        ) != "authored":
            continue

        if str(
            graph.nodes[
                target
            ].get(
                "type",
                "",
            )
        ) == "paper":
            papers.add(
                str(
                    target
                )
            )


    for source, _, attributes in graph.in_edges(
        author_id,
        data=True,
    ):
        if str(
            attributes.get(
                "relation",
                "",
            )
        ) != "authored":
            continue

        if str(
            graph.nodes[
                source
            ].get(
                "type",
                "",
            )
        ) == "paper":
            papers.add(
                str(
                    source
                )
            )

    return papers


def apply_strict_holdout(
    graph,
    test_positive_pairs,
    index,
    seed,
):
    removed_records = []

    diagnostics = []

    shared_papers = index[
        "shared_papers"
    ]

    authors_by_paper = index[
        "authors_by_paper"
    ]

    for pair in test_positive_pairs:
        papers = shared_papers[
            pair
        ]

        if len(
            papers
        ) != 1:
            raise RuntimeError(
                "Strict DBLP test pair unexpectedly "
                "has multiple shared papers."
            )

        paper_id = papers[
            0
        ]

        if len(
            authors_by_paper[
                paper_id
            ]
        ) != 2:
            raise RuntimeError(
                "Strict DBLP test paper unexpectedly "
                "has more than two authors."
            )

        cut_author = select_cut_endpoint(
            pair,
            seed,
        )

        records = authored_edge_records_between(
            graph,
            cut_author,
            paper_id,
        )

        if not records:
            raise RuntimeError(
                "Could not find authored edge for "
                f"held-out DBLP pair {pair}."
            )

        remove_authored_records(
            graph,
            records,
        )

        removed_records.extend(
            records
        )

        diagnostics.append({
            "source": pair[
                0
            ],
            "target": pair[
                1
            ],
            "shared_paper": paper_id,
            "cut_author": cut_author,
            "physical_authored_edges_removed": int(
                len(
                    records
                )
            ),
        })

    return {
        "removed_records": removed_records,
        "diagnostics": diagnostics,
    }


def validate_strict_holdout(
    graph,
    test_positive_pairs,
):
    leakage_pairs = []

    for pair in test_positive_pairs:
        first_papers = current_authored_papers(
            graph,
            pair[
                0
            ],
        )

        second_papers = current_authored_papers(
            graph,
            pair[
                1
            ],
        )

        common = (
            first_papers
            &
            second_papers
        )

        if common:
            leakage_pairs.append({
                "source": pair[
                    0
                ],
                "target": pair[
                    1
                ],
                "remaining_common_papers": int(
                    len(
                        common
                    )
                ),
            })

    if leakage_pairs:
        raise RuntimeError(
            "DBLP leakage validation failed: "
            f"{len(leakage_pairs)} held-out coauthor "
            "pairs still share at least one paper."
        )

    return {
        "remaining_test_pairs_with_shared_paper": 0,
    }


def orient_scores(
    cosine_scores,
    orientation,
):
    cosine_scores = np.asarray(
        cosine_scores,
        dtype=float,
    )

    if orientation == "similarity":
        return cosine_scores

    if orientation == "distance":
        return (
            1.0
            -
            cosine_scores
        )

    raise ValueError(
        f"Unknown orientation: {orientation}"
    )


def binary_metrics(
    positive_scores,
    negative_scores,
):


    positive_scores = stabilize_scores(
        positive_scores
    )

    negative_scores = stabilize_scores(
        negative_scores
    )

    labels = np.concatenate(
        [
            np.ones(
                len(
                    positive_scores
                ),
                dtype=int,
            ),
            np.zeros(
                len(
                    negative_scores
                ),
                dtype=int,
            ),
        ]
    )

    scores = np.concatenate(
        [
            positive_scores,
            negative_scores,
        ]
    )

    return {
        "auc": float(
            roc_auc_score(
                labels,
                scores,
            )
        ),

        "ap": float(
            average_precision_score(
                labels,
                scores,
            )
        ),

        "positive_score_mean": float(
            np.mean(
                positive_scores
            )
        ),

        "positive_score_median": float(
            np.median(
                positive_scores
            )
        ),

        "negative_score_mean": float(
            np.mean(
                negative_scores
            )
        ),

        "negative_score_median": float(
            np.median(
                negative_scores
            )
        ),

        "score_mean_gap": float(
            np.mean(
                positive_scores
            )
            -
            np.mean(
                negative_scores
            )
        ),

        "positive_unique_stabilized": int(
            len(
                np.unique(
                    positive_scores
                )
            )
        ),

        "negative_unique_stabilized": int(
            len(
                np.unique(
                    negative_scores
                )
            )
        ),

        "all_unique_stabilized": int(
            len(
                np.unique(
                    scores
                )
            )
        ),
    }


def bootstrap_metrics(
    positive_scores,
    negative_scores,
    repetitions,
    seed,
):
    positive_scores = np.asarray(
        positive_scores,
        dtype=float,
    )

    negative_scores = np.asarray(
        negative_scores,
        dtype=float,
    )

    rng = np.random.default_rng(
        seed
    )

    auc_values = []

    ap_values = []

    for _ in range(
        int(
            repetitions
        )
    ):
        positive_sample = positive_scores[
            rng.integers(
                0,
                len(
                    positive_scores
                ),
                size=len(
                    positive_scores
                ),
            )
        ]

        negative_sample = negative_scores[
            rng.integers(
                0,
                len(
                    negative_scores
                ),
                size=len(
                    negative_scores
                ),
            )
        ]

        result = binary_metrics(
            positive_sample,
            negative_sample,
        )

        auc_values.append(
            result[
                "auc"
            ]
        )

        ap_values.append(
            result[
                "ap"
            ]
        )

    return {
        "auc_ci95_low": float(
            np.percentile(
                auc_values,
                2.5,
            )
        ),

        "auc_ci95_high": float(
            np.percentile(
                auc_values,
                97.5,
            )
        ),

        "ap_ci95_low": float(
            np.percentile(
                ap_values,
                2.5,
            )
        ),

        "ap_ci95_high": float(
            np.percentile(
                ap_values,
                97.5,
            )
        ),
    }


def run_one_seed_cohort(
    graph,
    dataset_name,
    cohort,
    seed,
    author_ids,
    cohort_positive_pairs,
    cohort_strict_pairs,
    index,
    test_positives,
    negative_ratio,
    bootstrap_repetitions,
    d_max,
    degree_weight,
    seed_folder,
):
    start = time.perf_counter()

    print()
    print(
        "-" * 76
    )

    print(
        f"Cohort: {cohort}"
    )

    print(
        f"Seed: {seed}"
    )

    print(
        "-" * 76
    )


    forest_pairs = build_random_spanning_forest(
        authors=author_ids,
        positive_pairs=cohort_positive_pairs,
        seed=deterministic_seed(
            dataset_name,
            cohort,
            seed,
            "spanning_forest",
        ),
    )

    cycle_pairs = (
        cohort_positive_pairs
        -
        forest_pairs
    )

    eligible_test_pairs = (
        cohort_strict_pairs
        &
        cycle_pairs
    )

    print(
        f"Authors in cohort: "
        f"{len(author_ids):,}"
    )

    print(
        f"Coauthor pairs in cohort: "
        f"{len(cohort_positive_pairs):,}"
    )

    print(
        f"Forced forest pairs: "
        f"{len(forest_pairs):,}"
    )

    print(
        f"Cycle pairs: "
        f"{len(cycle_pairs):,}"
    )

    print(
        "Strict unique-paper dyadic candidates "
        "after forest exclusion: "
        f"{len(eligible_test_pairs):,}"
    )


    positive_pairs = sample_positive_pairs(
        eligible_test_pairs,
        requested=test_positives,
        seed=deterministic_seed(
            dataset_name,
            cohort,
            seed,
            "positive_sample",
        ),
    )


    requested_negatives = int(
        round(
            len(
                positive_pairs
            )
            *
            float(
                negative_ratio
            )
        )
    )

    negative_pairs = sample_negative_pairs(
        authors=author_ids,
        full_positive_pairs=cohort_positive_pairs,
        requested=requested_negatives,
        seed=deterministic_seed(
            dataset_name,
            cohort,
            seed,
            "negative_sample",
        ),
    )

    print(
        f"Test positives: "
        f"{len(positive_pairs):,}"
    )

    print(
        f"Test negatives: "
        f"{len(negative_pairs):,}"
    )


    intervention = None

    try:
        intervention = apply_strict_holdout(
            graph=graph,
            test_positive_pairs=positive_pairs,
            index=index,
            seed=seed,
        )

        validation = validate_strict_holdout(
            graph,
            positive_pairs,
        )

        print(
            "Held-out pairs still sharing a paper: "
            f"{validation['remaining_test_pairs_with_shared_paper']}"
        )

        print(
            "Physical authored edges removed: "
            f"{len(intervention['removed_records']):,}"
        )

        if set(
            positive_pairs
        ) & forest_pairs:
            raise RuntimeError(
                "A held-out pair unexpectedly belongs "
                "to the forced spanning forest."
            )

        print(
            "Forest/test overlap: 0"
        )

        print(
            "Collateral coauthor removal by construction: 0"
        )


        required_clients = pair_endpoints(
            positive_pairs,
            negative_pairs,
        )

        print(
            f"Descriptor clients required: "
            f"{len(required_clients):,}"
        )

        generated = generate_selected_descriptors(
            training_graph=graph,
            client_ids=required_clients,
            d_max=d_max,
            degree_weight=degree_weight,
        )

        descriptor_client_ids = generated[
            "client_ids"
        ]

        descriptor_matrices = generated[
            "descriptors"
        ]

        failures = generated[
            "failures"
        ]

        if failures:
            raise RuntimeError(
                f"Descriptor generation failed for "
                f"{len(failures)} DBLP authors."
            )

        if (
            len(
                descriptor_client_ids
            )
            !=
            len(
                required_clients
            )
        ):
            raise RuntimeError(
                "Not all required DBLP authors received "
                "a descriptor."
            )


        pair_rows = []

        for pair in positive_pairs:
            pair_rows.append({
                "pair_type": "positive",
                "source": pair[
                    0
                ],
                "target": pair[
                    1
                ],
            })

        for pair in negative_pairs:
            pair_rows.append({
                "pair_type": "negative",
                "source": pair[
                    0
                ],
                "target": pair[
                    1
                ],
            })

        pair_dataframe = pd.DataFrame(
            pair_rows
        )


        metric_rows = []

        for descriptor in DESCRIPTORS:
            matrix = descriptor_matrices[
                descriptor
            ]

            positive_cosine = score_pairs(
                matrix,
                descriptor_client_ids,
                positive_pairs,
            )

            negative_cosine = score_pairs(
                matrix,
                descriptor_client_ids,
                negative_pairs,
            )


            pair_dataframe[
                descriptor
            ] = np.concatenate(
                [
                    positive_cosine,
                    negative_cosine,
                ]
            )

            for orientation in ORIENTATIONS:
                positive_scores = orient_scores(
                    positive_cosine,
                    orientation,
                )

                negative_scores = orient_scores(
                    negative_cosine,
                    orientation,
                )

                metrics = binary_metrics(
                    positive_scores,
                    negative_scores,
                )

                intervals = bootstrap_metrics(
                    positive_scores,
                    negative_scores,
                    repetitions=bootstrap_repetitions,
                    seed=deterministic_seed(
                        dataset_name,
                        cohort,
                        seed,
                        descriptor,
                        orientation,
                        "bootstrap",
                    ),
                )

                metric_rows.append({
                    "cohort": cohort,
                    "seed": int(
                        seed
                    ),
                    "descriptor": descriptor,
                    "orientation": orientation,
                    "score_round_decimals": int(
                        SCORE_ROUND_DECIMALS
                    ),
                    "positive_pairs": int(
                        len(
                            positive_pairs
                        )
                    ),
                    "negative_pairs": int(
                        len(
                            negative_pairs
                        )
                    ),
                    **metrics,
                    **intervals,
                })

        metrics_dataframe = pd.DataFrame(
            metric_rows
        )


        orientation_rows = []

        for descriptor in DESCRIPTORS:
            descriptor_results = metrics_dataframe[
                metrics_dataframe[
                    "descriptor"
                ]
                ==
                descriptor
            ]

            similarity_row = descriptor_results[
                descriptor_results[
                    "orientation"
                ]
                ==
                "similarity"
            ].iloc[
                0
            ]

            distance_row = descriptor_results[
                descriptor_results[
                    "orientation"
                ]
                ==
                "distance"
            ].iloc[
                0
            ]

            similarity_auc = float(
                similarity_row[
                    "auc"
                ]
            )

            distance_auc = float(
                distance_row[
                    "auc"
                ]
            )

            similarity_ap = float(
                similarity_row[
                    "ap"
                ]
            )

            distance_ap = float(
                distance_row[
                    "ap"
                ]
            )

            if similarity_auc > distance_auc:
                auc_orientation = "similarity"

            elif distance_auc > similarity_auc:
                auc_orientation = "distance"

            else:
                auc_orientation = "tie"

            if similarity_ap > distance_ap:
                ap_orientation = "similarity"

            elif distance_ap > similarity_ap:
                ap_orientation = "distance"

            else:
                ap_orientation = "tie"

            orientation_rows.append({
                "cohort": cohort,
                "seed": int(
                    seed
                ),
                "descriptor": descriptor,
                "score_round_decimals": int(
                    SCORE_ROUND_DECIMALS
                ),
                "similarity_auc": similarity_auc,
                "distance_auc": distance_auc,
                "auc_preferred_orientation_diagnostic": (
                    auc_orientation
                ),
                "similarity_ap": similarity_ap,
                "distance_ap": distance_ap,
                "ap_preferred_orientation_diagnostic": (
                    ap_orientation
                ),
            })

        orientation_dataframe = pd.DataFrame(
            orientation_rows
        )


        os.makedirs(
            seed_folder,
            exist_ok=True,
        )

        metrics_path = os.path.join(
            seed_folder,
            "metrics.csv",
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

        orientation_path = os.path.join(
            seed_folder,
            "orientation_diagnostic.csv",
        )

        metadata_path = os.path.join(
            seed_folder,
            "metadata.json",
        )

        metrics_dataframe.to_csv(
            metrics_path,
            index=False,
        )

        pair_dataframe.to_csv(
            scores_path,
            index=False,
            compression="gzip",
        )


        split_rows = []

        for pair in positive_pairs:
            shared_paper = index[
                "shared_papers"
            ][
                pair
            ][
                0
            ]

            split_rows.append({
                "source": pair[
                    0
                ],
                "target": pair[
                    1
                ],
                "split": "test_positive",
                "shared_paper": shared_paper,
            })

        for pair in negative_pairs:
            split_rows.append({
                "source": pair[
                    0
                ],
                "target": pair[
                    1
                ],
                "split": "test_negative",
                "shared_paper": "",
            })

        pd.DataFrame(
            split_rows
        ).to_csv(
            split_path,
            index=False,
        )

        pd.DataFrame(
            intervention[
                "diagnostics"
            ]
        ).to_csv(
            leakage_path,
            index=False,
        )

        orientation_dataframe.to_csv(
            orientation_path,
            index=False,
        )


        runtime = float(
            time.perf_counter()
            -
            start
        )

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "dataset": dataset_name,
            "cohort": cohort,
            "seed": int(
                seed
            ),

            "authors_in_cohort": int(
                len(
                    author_ids
                )
            ),

            "coauthor_pairs_in_cohort": int(
                len(
                    cohort_positive_pairs
                )
            ),

            "strict_pairs_in_cohort": int(
                len(
                    cohort_strict_pairs
                )
            ),

            "forest_pairs": int(
                len(
                    forest_pairs
                )
            ),

            "cycle_pairs": int(
                len(
                    cycle_pairs
                )
            ),

            "eligible_test_pairs": int(
                len(
                    eligible_test_pairs
                )
            ),

            "test_positive_pairs": int(
                len(
                    positive_pairs
                )
            ),

            "test_negative_pairs": int(
                len(
                    negative_pairs
                )
            ),

            "descriptor_clients": int(
                len(
                    descriptor_client_ids
                )
            ),

            "descriptor_failures": failures,

            "physical_authored_edges_removed": int(
                len(
                    intervention[
                        "removed_records"
                    ]
                )
            ),

            "remaining_test_pairs_with_shared_paper": 0,

            "forest_test_overlap": 0,

            "score_round_decimals": int(
                SCORE_ROUND_DECIMALS
            ),

            "numerical_stabilization": (
                "All scores used for AUC/AP are rounded to "
                f"{SCORE_ROUND_DECIMALS} decimal places before "
                "evaluation. Raw cosine values remain saved in "
                "pairs_and_scores.csv.gz."
            ),

            "numerical_stabilization_reason": (
                "DBLP HKS produced mathematically/effectively tied "
                "cosine values near 1.0 whose differences were only "
                "at machine precision. Such differences must not "
                "generate artificial ranking performance."
            ),

            "strict_test_definition": (
                "Each held-out positive pair shares exactly one "
                "paper, that paper has exactly two authors, and the "
                "pair is excluded from the randomized spanning "
                "forest."
            ),

            "collateral_relation_statement": (
                "Because every held-out pair is represented by one "
                "exactly-two-author paper, deleting one authorship "
                "incidence removes only that held-out coauthor pair."
            ),

            "privacy": (
                "NONE. This is a clean representation evaluation."
            ),

            "runtime_seconds": runtime,
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
            f"Numerical stabilization: "
            f"{SCORE_ROUND_DECIMALS} decimals"
        )

        print()
        print(
            "Descriptor/orientation results:"
        )

        print(
            metrics_dataframe[
                [
                    "descriptor",
                    "orientation",
                    "auc",
                    "auc_ci95_low",
                    "auc_ci95_high",
                    "ap",
                    "ap_ci95_low",
                    "ap_ci95_high",
                    "positive_score_mean",
                    "negative_score_mean",
                    "all_unique_stabilized",
                ]
            ].to_string(
                index=False
            )
        )

        print()
        print(
            "Orientation diagnostic:"
        )

        print(
            orientation_dataframe.to_string(
                index=False
            )
        )

        print()
        print(
            f"Seed/cohort runtime: "
            f"{runtime:.3f} s"
        )

        return {
            "metrics": metrics_dataframe,
            "orientation": orientation_dataframe,
            "metadata": metadata,
        }

    finally:


        if (
            intervention is not None
            and intervention[
                "removed_records"
            ]
        ):
            restore_authored_records(
                graph,
                intervention[
                    "removed_records"
                ],
            )


def aggregate_metrics(
    all_metrics,
):
    rows = []

    grouped = all_metrics.groupby(
        [
            "cohort",
            "descriptor",
            "orientation",
        ],
        sort=True,
    )

    metric_names = (
        "auc",
        "ap",
        "positive_score_mean",
        "negative_score_mean",
        "score_mean_gap",
        "all_unique_stabilized",
    )

    for (
        cohort,
        descriptor,
        orientation,
    ), group in grouped:

        row = {
            "cohort": cohort,
            "descriptor": descriptor,
            "orientation": orientation,
            "score_round_decimals": int(
                SCORE_ROUND_DECIMALS
            ),
            "split_seeds": int(
                group[
                    "seed"
                ].nunique()
            ),
        }

        for metric in metric_names:
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

    if len(
        dataframe
    ):
        dataframe = (
            dataframe.sort_values(
                [
                    "cohort",
                    "auc__mean",
                    "ap__mean",
                ],
                ascending=[
                    True,
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

    return dataframe


def aggregate_orientation(
    orientation_dataframe,
):
    rows = []

    grouped = orientation_dataframe.groupby(
        [
            "cohort",
            "descriptor",
        ],
        sort=True,
    )

    for (
        cohort,
        descriptor,
    ), group in grouped:

        similarity_auc = group[
            "similarity_auc"
        ].to_numpy(
            dtype=float
        )

        distance_auc = group[
            "distance_auc"
        ].to_numpy(
            dtype=float
        )

        similarity_ap = group[
            "similarity_ap"
        ].to_numpy(
            dtype=float
        )

        distance_ap = group[
            "distance_ap"
        ].to_numpy(
            dtype=float
        )

        rows.append({
            "cohort": cohort,
            "descriptor": descriptor,
            "score_round_decimals": int(
                SCORE_ROUND_DECIMALS
            ),
            "split_seeds": int(
                len(
                    group
                )
            ),

            "mean_similarity_auc": float(
                np.mean(
                    similarity_auc
                )
            ),

            "mean_distance_auc": float(
                np.mean(
                    distance_auc
                )
            ),

            "auc_similarity_wins": int(
                np.sum(
                    similarity_auc
                    >
                    distance_auc
                )
            ),

            "auc_distance_wins": int(
                np.sum(
                    distance_auc
                    >
                    similarity_auc
                )
            ),

            "auc_ties": int(
                np.sum(
                    similarity_auc
                    ==
                    distance_auc
                )
            ),

            "mean_similarity_ap": float(
                np.mean(
                    similarity_ap
                )
            ),

            "mean_distance_ap": float(
                np.mean(
                    distance_ap
                )
            ),

            "ap_similarity_wins": int(
                np.sum(
                    similarity_ap
                    >
                    distance_ap
                )
            ),

            "ap_distance_wins": int(
                np.sum(
                    distance_ap
                    >
                    similarity_ap
                )
            ),

            "ap_ties": int(
                np.sum(
                    similarity_ap
                    ==
                    distance_ap
                )
            ),
        })

    return pd.DataFrame(
        rows
    )


def compare_cohorts(
    aggregate_dataframe,
):
    available_cohorts = set(
        aggregate_dataframe[
            "cohort"
        ]
    )

    if not {
        "all_authors",
        "provided_id_only",
    }.issubset(
        available_cohorts
    ):
        return pd.DataFrame()

    rows = []

    for descriptor in DESCRIPTORS:
        for orientation in ORIENTATIONS:

            all_rows = aggregate_dataframe[
                (
                    aggregate_dataframe[
                        "cohort"
                    ]
                    ==
                    "all_authors"
                )
                &
                (
                    aggregate_dataframe[
                        "descriptor"
                    ]
                    ==
                    descriptor
                )
                &
                (
                    aggregate_dataframe[
                        "orientation"
                    ]
                    ==
                    orientation
                )
            ]

            provided_rows = aggregate_dataframe[
                (
                    aggregate_dataframe[
                        "cohort"
                    ]
                    ==
                    "provided_id_only"
                )
                &
                (
                    aggregate_dataframe[
                        "descriptor"
                    ]
                    ==
                    descriptor
                )
                &
                (
                    aggregate_dataframe[
                        "orientation"
                    ]
                    ==
                    orientation
                )
            ]

            if (
                len(
                    all_rows
                ) != 1
                or len(
                    provided_rows
                ) != 1
            ):
                continue

            all_row = all_rows.iloc[
                0
            ]

            provided_row = provided_rows.iloc[
                0
            ]

            rows.append({
                "descriptor": descriptor,
                "orientation": orientation,

                "all_authors_auc": float(
                    all_row[
                        "auc__mean"
                    ]
                ),

                "provided_id_only_auc": float(
                    provided_row[
                        "auc__mean"
                    ]
                ),

                "provided_minus_all_auc": float(
                    provided_row[
                        "auc__mean"
                    ]
                    -
                    all_row[
                        "auc__mean"
                    ]
                ),

                "all_authors_ap": float(
                    all_row[
                        "ap__mean"
                    ]
                ),

                "provided_id_only_ap": float(
                    provided_row[
                        "ap__mean"
                    ]
                ),

                "provided_minus_all_ap": float(
                    provided_row[
                        "ap__mean"
                    ]
                    -
                    all_row[
                        "ap__mean"
                    ]
                ),
            })

    return pd.DataFrame(
        rows
    )


def run_dblp_special_link_prediction(
    output_folder=DEFAULT_OUTPUT_FOLDER,
    seeds=DEFAULT_SEEDS,
    cohorts=DEFAULT_COHORTS,
    test_positives=DEFAULT_TEST_POSITIVES,
    negative_ratio=DEFAULT_NEGATIVE_RATIO,
    bootstrap_repetitions=DEFAULT_BOOTSTRAP_REPETITIONS,
    d_max=DEFAULT_D_MAX,
    degree_weight=DEFAULT_DEGREE_WEIGHT,
):
    total_start = time.perf_counter()

    dataset_name = "dblp"

    seeds = tuple(
        int(
            seed
        )
        for seed in seeds
    )

    cohorts = tuple(
        str(
            cohort
        )
        for cohort in cohorts
    )

    allowed_cohorts = {
        "all_authors",
        "provided_id_only",
    }

    invalid_cohorts = (
        set(
            cohorts
        )
        -
        allowed_cohorts
    )

    if invalid_cohorts:
        raise ValueError(
            f"Unknown DBLP cohorts: "
            f"{sorted(invalid_cohorts)}"
        )

    graph_path = os.path.join(
        output_folder,
        "dblp",
        "dblp_global_graph.gml.gz",
    )

    if not os.path.exists(
        graph_path
    ):
        raise FileNotFoundError(
            "Missing DBLP prepared graph:\n"
            f"  {graph_path}\n"
            "Run prepare_dblp_special.py first."
        )

    experiment_folder = os.path.join(
        output_folder,
        "dblp",
        "experiments",
        "dblp_link_prediction_special",
    )

    os.makedirs(
        experiment_folder,
        exist_ok=True,
    )

    print()
    print(
        "=" * 80
    )

    print(
        "DBLP SPECIAL LEAKAGE-SAFE COAUTHOR PREDICTION"
    )

    print(
        "=" * 80
    )

    print(
        f"Seeds: "
        f"{list(seeds)}"
    )

    print(
        f"Cohorts: "
        f"{list(cohorts)}"
    )

    print(
        f"Requested test positives per run: "
        f"{test_positives:,}"
    )

    print(
        "Metric score stabilization: "
        f"round to {SCORE_ROUND_DECIMALS} decimals"
    )

    print()
    print(
        "Loading prepared DBLP graph..."
    )

    graph = nx.read_gml(
        graph_path
    )

    print(
        f"Graph nodes: "
        f"{graph.number_of_nodes():,}"
    )

    print(
        f"Graph edges: "
        f"{graph.number_of_edges():,}"
    )


    index = build_dblp_coauthor_index(
        graph
    )

    all_positive_pairs = index[
        "all_pairs"
    ]

    all_strict_pairs = index[
        "strict_pairs"
    ]

    all_metric_frames = []

    all_orientation_frames = []

    all_run_metadata = []


    for cohort in cohorts:
        print()
        print(
            "=" * 80
        )

        print(
            f"COHORT: {cohort}"
        )

        print(
            "=" * 80
        )

        author_ids = cohort_author_ids(
            graph,
            cohort,
        )

        author_set = set(
            author_ids
        )

        cohort_positive_pairs = {
            pair
            for pair in all_positive_pairs
            if (
                pair[
                    0
                ]
                in author_set
                and
                pair[
                    1
                ]
                in author_set
            )
        }

        cohort_strict_pairs = {
            pair
            for pair in all_strict_pairs
            if (
                pair[
                    0
                ]
                in author_set
                and
                pair[
                    1
                ]
                in author_set
            )
        }

        print(
            f"Authors: "
            f"{len(author_ids):,}"
        )

        print(
            f"Coauthor pairs: "
            f"{len(cohort_positive_pairs):,}"
        )

        print(
            "Strict unique-paper dyadic pairs: "
            f"{len(cohort_strict_pairs):,}"
        )

        if len(
            cohort_strict_pairs
        ) < 100:
            raise RuntimeError(
                f"Cohort {cohort} has too few strict "
                "dyadic positive pairs for meaningful "
                "evaluation."
            )


        for seed_index, seed in enumerate(
            seeds,
            start=1,
        ):
            print()
            print(
                "#" * 80
            )

            print(
                f"{cohort} | "
                f"seed {seed_index}/"
                f"{len(seeds)}: "
                f"{seed}"
            )

            print(
                "#" * 80
            )

            seed_folder = os.path.join(
                experiment_folder,
                cohort,
                f"seed_{seed}",
            )

            result = run_one_seed_cohort(
                graph=graph,
                dataset_name=dataset_name,
                cohort=cohort,
                seed=seed,
                author_ids=author_ids,
                cohort_positive_pairs=cohort_positive_pairs,
                cohort_strict_pairs=cohort_strict_pairs,
                index=index,
                test_positives=test_positives,
                negative_ratio=negative_ratio,
                bootstrap_repetitions=bootstrap_repetitions,
                d_max=d_max,
                degree_weight=degree_weight,
                seed_folder=seed_folder,
            )

            all_metric_frames.append(
                result[
                    "metrics"
                ]
            )

            all_orientation_frames.append(
                result[
                    "orientation"
                ]
            )

            all_run_metadata.append(
                result[
                    "metadata"
                ]
            )


    all_metrics = pd.concat(
        all_metric_frames,
        ignore_index=True,
    )

    all_orientation = pd.concat(
        all_orientation_frames,
        ignore_index=True,
    )

    aggregate = aggregate_metrics(
        all_metrics
    )

    orientation_summary = aggregate_orientation(
        all_orientation
    )

    cohort_comparison = compare_cohorts(
        aggregate
    )


    seed_metrics_path = os.path.join(
        experiment_folder,
        "dblp_seed_metrics.csv",
    )

    aggregate_path = os.path.join(
        experiment_folder,
        "dblp_aggregate_metrics.csv",
    )

    orientation_path = os.path.join(
        experiment_folder,
        "dblp_orientation_summary.csv",
    )

    cohort_path = os.path.join(
        experiment_folder,
        "dblp_cohort_comparison.csv",
    )

    metadata_path = os.path.join(
        experiment_folder,
        "dblp_link_prediction_metadata.json",
    )

    all_metrics.to_csv(
        seed_metrics_path,
        index=False,
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    orientation_summary.to_csv(
        orientation_path,
        index=False,
    )

    cohort_comparison.to_csv(
        cohort_path,
        index=False,
    )

    total_runtime = float(
        time.perf_counter()
        -
        total_start
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "dataset": dataset_name,
        "graph": graph_path,

        "seeds": list(
            seeds
        ),

        "cohorts": list(
            cohorts
        ),

        "test_positives_requested": int(
            test_positives
        ),

        "negative_ratio": float(
            negative_ratio
        ),

        "bootstrap_repetitions": int(
            bootstrap_repetitions
        ),

        "descriptors": list(
            DESCRIPTORS
        ),

        "orientations": list(
            ORIENTATIONS
        ),

        "score_round_decimals": int(
            SCORE_ROUND_DECIMALS
        ),

        "metric_score_stabilization": (
            f"Scores are rounded to {SCORE_ROUND_DECIMALS} decimal "
            "places before AUC/AP computation. Raw pairwise cosine "
            "scores are retained in per-seed pair files."
        ),

        "global_coauthor_pairs": int(
            len(
                all_positive_pairs
            )
        ),

        "global_strict_unique_dyadic_pairs": int(
            len(
                all_strict_pairs
            )
        ),

        "strict_holdout_definition": (
            "A test positive pair shares exactly one paper, "
            "that paper has exactly two authors, and the pair "
            "is excluded from a randomized spanning forest. "
            "One authorship incidence is removed before "
            "descriptor computation."
        ),

        "why_dyadic_only": (
            "Removing one authorship incidence from a multi-author "
            "paper can destroy several coauthor relations. Restricting "
            "the primary evaluation to exactly-two-author papers "
            "prevents that collateral modification."
        ),

        "provided_id_only_purpose": (
            "Quantifies sensitivity of results to the DBLP fallback "
            "author identity heuristic."
        ),

        "orientation_warning": (
            "Similarity versus distance is evaluated independently "
            "on DBLP. No orientation learned from Enron is assumed."
        ),

        "privacy": (
            "NONE. This is a clean representation evaluation."
        ),

        "runs": all_run_metadata,

        "runtime_seconds": total_runtime,
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
        "=" * 80
    )

    print(
        "DBLP SPECIAL LINK PREDICTION COMPLETE"
    )

    print(
        "=" * 80
    )

    print()
    print(
        "Aggregate descriptor/orientation results:"
    )

    display_columns = [
        "cohort",
        "descriptor",
        "orientation",
        "auc__mean",
        "auc__std",
        "auc__ci95_low",
        "auc__ci95_high",
        "ap__mean",
        "ap__std",
        "positive_score_mean__mean",
        "negative_score_mean__mean",
        "all_unique_stabilized__mean",
    ]

    print(
        aggregate[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Orientation consistency:"
    )

    print(
        orientation_summary.to_string(
            index=False
        )
    )

    if len(
        cohort_comparison
    ):
        print()
        print(
            "Provided-ID-only minus all-authors:"
        )

        print(
            cohort_comparison.to_string(
                index=False
            )
        )

    print()
    print(
        "INTERPRETATION:"
    )

    print(
        "  1. Scores used for metrics are numerically "
        f"stabilized at {SCORE_ROUND_DECIMALS} decimals."
    )

    print(
        "  2. Raw cosine scores remain saved for "
        "precision diagnostics."
    )

    print(
        "  3. HKS should be treated as collapsed if "
        "its stabilized score count is one and AUC/AP "
        "are approximately 0.5."
    )

    print(
        "  4. Do NOT assume DBLP should use the same "
        "spectral orientation as Enron."
    )

    print(
        "  5. Compare all_authors against "
        "provided_id_only to quantify fallback-identity effects."
    )

    print(
        "  6. The primary test uses only unique-paper "
        "exactly-two-author relations to avoid collateral "
        "holdout modifications."
    )

    print(
        "  7. No privacy noise has been applied."
    )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  {seed_metrics_path}"
    )

    print(
        f"  {aggregate_path}"
    )

    print(
        f"  {orientation_path}"
    )

    print(
        f"  {cohort_path}"
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
        "=" * 80
    )

    return {
        "seed_metrics": all_metrics,
        "aggregate": aggregate,
        "orientation_summary": orientation_summary,
        "cohort_comparison": cohort_comparison,
        "metadata": metadata,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe DBLP coauthor prediction using "
            "strict unique-paper dyadic test relations and "
            "numerically stabilized ranking scores."
        )
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
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
        "--cohorts",
        nargs="+",
        choices=[
            "all_authors",
            "provided_id_only",
        ],
        default=list(
            DEFAULT_COHORTS
        ),
    )

    parser.add_argument(
        "--test-positives",
        type=int,
        default=DEFAULT_TEST_POSITIVES,
    )

    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=DEFAULT_NEGATIVE_RATIO,
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPETITIONS,
    )

    parser.add_argument(
        "--Dmax",
        type=int,
        default=DEFAULT_D_MAX,
    )

    parser.add_argument(
        "--degree-weight",
        type=float,
        default=DEFAULT_DEGREE_WEIGHT,
    )

    args = parser.parse_args()

    run_dblp_special_link_prediction(
        output_folder=args.output_folder,
        seeds=args.seeds,
        cohorts=args.cohorts,
        test_positives=args.test_positives,
        negative_ratio=args.negative_ratio,
        bootstrap_repetitions=args.bootstrap,
        d_max=args.Dmax,
        degree_weight=args.degree_weight,
    )
