import argparse
import hashlib
import json
import math
import os
import time

from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from edge_sensitivity_experiment import (
    build_weighted_projection,
    create_true_ego_graph,
    typed_root_degree,
)

from representation_diagnostic_experiment import (
    all_client_ids,
    build_descriptors_from_profiles,
    collect_client_client_pairs,
    compute_spectral_profile,
    deterministic_seed,
    normalize_rows,
    sample_negative_pairs,
)


LINK_PREDICTION_SCHEMA_VERSION = "2.0"

DEFAULT_OUTPUT_FOLDER = "output_local_test"

DEFAULT_TEST_POSITIVES = 1000

DEFAULT_NEGATIVE_RATIO = 1.0

DEFAULT_BOOTSTRAP_REPETITIONS = 500

DEFAULT_SEED = 0

DEFAULT_D_MAX = 50

DEFAULT_DEGREE_WEIGHT = 1.0

DEFAULT_K_VALUES = (
    10,
    20,
    50,
)

DEFAULT_HOLDOUT_POLICY = "auto"

NUMERICAL_EPS = 1e-12


DESCRIPTOR_SPECS = (
    {
        "name": "hks_d12_global",
        "family": "hks",
        "dimension": 12,
        "normalization": "global",
    },
    {
        "name": "eigen_d8_block",
        "family": "eigen",
        "dimension": 8,
        "normalization": "block",
    },
    {
        "name": "eigen_d12_global",
        "family": "eigen",
        "dimension": 12,
        "normalization": "global",
    },
)

BASELINE_DESCRIPTOR = "hks_d12_global"


class UnionFind:

    def __init__(self, nodes):
        self.parent = {
            node: node
            for node
            in nodes
        }

        self.rank = {
            node: 0
            for node
            in nodes
        }

    def find(self, node):
        parent = self.parent[
            node
        ]

        if parent != node:
            self.parent[
                node
            ] = self.find(
                parent
            )

        return self.parent[
            node
        ]

    def union(self, first, second):
        first_root = self.find(
            first
        )

        second_root = self.find(
            second
        )

        if first_root == second_root:
            return False

        first_rank = self.rank[
            first_root
        ]

        second_rank = self.rank[
            second_root
        ]

        if first_rank < second_rank:
            first_root, second_root = (
                second_root,
                first_root,
            )

        self.parent[
            second_root
        ] = first_root

        if first_rank == second_rank:
            self.rank[
                first_root
            ] += 1

        return True


def undirected_neighbours(
    graph,
    node,
):
    if graph.is_directed():
        return (
            set(
                graph.predecessors(
                    node
                )
            )
            |
            set(
                graph.successors(
                    node
                )
            )
        )

    return set(
        graph.neighbors(
            node
        )
    )


def has_any_edge_between(
    graph,
    first,
    second,
):
    if graph.is_directed():
        return (
            graph.has_edge(
                first,
                second,
            )
            or
            graph.has_edge(
                second,
                first,
            )
        )

    return graph.has_edge(
        first,
        second,
    )


def remove_all_edges_between(
    graph,
    first,
    second,
):


    removed = 0

    if graph.is_directed():

        directions = [
            (
                first,
                second,
            ),
            (
                second,
                first,
            ),
        ]

    else:

        directions = [
            (
                first,
                second,
            )
        ]

    for source, target in directions:

        if graph.is_multigraph():

            if not graph.has_edge(
                source,
                target,
            ):
                continue

            keys = list(
                graph[
                    source
                ][
                    target
                ].keys()
            )

            for key in keys:
                graph.remove_edge(
                    source,
                    target,
                    key,
                )

                removed += 1

        else:

            if graph.has_edge(
                source,
                target,
            ):
                graph.remove_edge(
                    source,
                    target,
                )

                removed += 1

    return removed


def common_content_nodes(
    graph,
    first,
    second,
):


    first_neighbours = (
        undirected_neighbours(
            graph,
            first,
        )
    )

    second_neighbours = (
        undirected_neighbours(
            graph,
            second,
        )
    )

    common = (
        first_neighbours
        &
        second_neighbours
    )

    return {
        node
        for node
        in common
        if str(
            graph.nodes[
                node
            ].get(
                "role",
                "",
            )
        ).lower()
        ==
        "content"
    }


def resolve_holdout_policy(
    dataset_name,
    requested_policy,
):
    requested_policy = str(
        requested_policy
    ).lower()

    if requested_policy != "auto":
        return requested_policy

    dataset = str(
        dataset_name
    ).lower()

    if dataset in {
        "enron",
        "dblp",
    }:
        return (
            "strict_common_content"
        )

    return "direct_only"


def build_random_spanning_forest_split(
    client_ids,
    positive_pairs,
    seed,
):


    positive_pairs = list(
        positive_pairs
    )

    rng = np.random.default_rng(
        seed
    )

    permutation = rng.permutation(
        len(
            positive_pairs
        )
    )

    union_find = UnionFind(
        client_ids
    )

    forest_pairs = []

    cycle_pairs = []

    for index in permutation:

        pair = positive_pairs[
            int(
                index
            )
        ]

        first, second = pair

        if union_find.union(
            first,
            second,
        ):
            forest_pairs.append(
                pair
            )

        else:
            cycle_pairs.append(
                pair
            )

    return {
        "forest_pairs": sorted(
            forest_pairs
        ),
        "cycle_pairs": sorted(
            cycle_pairs
        ),
    }


def sample_pairs(
    pairs,
    requested,
    seed,
):
    pairs = list(
        pairs
    )

    if requested is None:
        return sorted(
            pairs
        )

    requested = min(
        int(
            requested
        ),
        len(
            pairs
        ),
    )

    if requested == len(
        pairs
    ):
        return sorted(
            pairs
        )

    rng = np.random.default_rng(
        seed
    )

    selected = np.sort(
        rng.choice(
            len(
                pairs
            ),
            size=requested,
            replace=False,
        )
    )

    return sorted(
        [
            pairs[
                int(
                    index
                )
            ]
            for index
            in selected
        ]
    )


def construct_training_graph(
    original_graph,
    test_positive_pairs,
    dataset_name,
    holdout_policy,
    seed,
):


    training_graph = (
        original_graph.copy()
    )

    direct_edges_removed = 0

    content_edges_removed = 0

    common_content_pairs = 0

    common_content_nodes_seen = 0

    pair_diagnostics = []


    for first, second in test_positive_pairs:

        removed = remove_all_edges_between(
            training_graph,
            first,
            second,
        )

        direct_edges_removed += (
            removed
        )


    if holdout_policy == (
        "strict_common_content"
    ):

        for first, second in test_positive_pairs:


            original_common = (
                common_content_nodes(
                    original_graph,
                    first,
                    second,
                )
            )

            if original_common:
                common_content_pairs += 1

            common_content_nodes_seen += (
                len(
                    original_common
                )
            )

            pair_removed = 0

            for content_node in sorted(
                original_common,
                key=str,
            ):


                current_common = (
                    common_content_nodes(
                        training_graph,
                        first,
                        second,
                    )
                )

                if content_node not in (
                    current_common
                ):
                    continue

                cut_seed = deterministic_seed(
                    dataset_name,
                    seed,
                    first,
                    second,
                    content_node,
                    "strict_common_content",
                )

                if cut_seed % 2 == 0:
                    endpoint = first
                else:
                    endpoint = second

                removed = (
                    remove_all_edges_between(
                        training_graph,
                        endpoint,
                        content_node,
                    )
                )


                if (
                    removed == 0
                    and content_node
                    in common_content_nodes(
                        training_graph,
                        first,
                        second,
                    )
                ):
                    other_endpoint = (
                        second
                        if endpoint == first
                        else first
                    )

                    removed = (
                        remove_all_edges_between(
                            training_graph,
                            other_endpoint,
                            content_node,
                        )
                    )

                pair_removed += (
                    removed
                )

                content_edges_removed += (
                    removed
                )

            pair_diagnostics.append({
                "source": first,
                "target": second,
                "original_common_content": int(
                    len(
                        original_common
                    )
                ),
                "content_incidence_edges_removed": int(
                    pair_removed
                ),
            })


    remaining_direct = []

    remaining_common_content = []

    for first, second in test_positive_pairs:

        if has_any_edge_between(
            training_graph,
            first,
            second,
        ):
            remaining_direct.append(
                (
                    first,
                    second,
                )
            )

        if holdout_policy == (
            "strict_common_content"
        ):
            common = common_content_nodes(
                training_graph,
                first,
                second,
            )

            if common:
                remaining_common_content.append({
                    "pair": (
                        first,
                        second,
                    ),
                    "count": len(
                        common
                    ),
                })

    if remaining_direct:
        raise RuntimeError(
            "Leakage validation failed: "
            f"{len(remaining_direct)} test pairs "
            "still have a direct client-client edge."
        )

    if remaining_common_content:
        raise RuntimeError(
            "Strict leakage validation failed: "
            f"{len(remaining_common_content)} test pairs "
            "still share at least one direct content node."
        )

    return {
        "graph": training_graph,

        "direct_edges_removed": int(
            direct_edges_removed
        ),

        "content_edges_removed": int(
            content_edges_removed
        ),

        "test_pairs_with_original_common_content": int(
            common_content_pairs
        ),

        "original_common_content_instances": int(
            common_content_nodes_seen
        ),

        "pair_diagnostics": (
            pair_diagnostics
        ),
    }


def generate_selected_descriptors(
    training_graph,
    client_ids,
    d_max,
    degree_weight,
):


    client_ids = [
        str(
            client
        )
        for client
        in client_ids
    ]

    descriptor_lists = {
        specification[
            "name"
        ]: []
        for specification
        in DESCRIPTOR_SPECS
    }

    valid_clients = []

    failures = []

    for client_number, client in enumerate(
        client_ids,
        start=1,
    ):

        if (
            client_number == 1
            or client_number % 100 == 0
            or client_number == len(
                client_ids
            )
        ):
            print(
                "[training descriptors] "
                f"{client_number}/"
                f"{len(client_ids)}"
            )

        try:


            ego = create_true_ego_graph(
                training_graph,
                client,
            )

            degree = typed_root_degree(
                ego,
                client,
            ).astype(
                float
            )

            cc_projection = (
                build_weighted_projection(
                    ego,
                    client,
                    "CC",
                )
            )

            cx_projection = (
                build_weighted_projection(
                    ego,
                    client,
                    "CX",
                )
            )

            cc_profile = (
                compute_spectral_profile(
                    cc_projection,
                    client,
                )
            )

            cx_profile = (
                compute_spectral_profile(
                    cx_projection,
                    client,
                )
            )


            bundle_8_block = (
                build_descriptors_from_profiles(
                    cc_profile=cc_profile,
                    cx_profile=cx_profile,
                    degree=degree,
                    dimension=8,
                    normalization="block",
                    d_max=d_max,
                    degree_weight=degree_weight,
                )
            )


            bundle_12_global = (
                build_descriptors_from_profiles(
                    cc_profile=cc_profile,
                    cx_profile=cx_profile,
                    degree=degree,
                    dimension=12,
                    normalization="global",
                    d_max=d_max,
                    degree_weight=degree_weight,
                )
            )

            descriptors = {
                "hks_d12_global": (
                    bundle_12_global[
                        "hks"
                    ]
                ),

                "eigen_d8_block": (
                    bundle_8_block[
                        "eigen"
                    ]
                ),

                "eigen_d12_global": (
                    bundle_12_global[
                        "eigen"
                    ]
                ),
            }

            for name, descriptor in (
                descriptors.items()
            ):

                descriptor = np.asarray(
                    descriptor,
                    dtype=float,
                )

                if not np.all(
                    np.isfinite(
                        descriptor
                    )
                ):
                    raise RuntimeError(
                        f"{name} contains "
                        "non-finite values."
                    )

                descriptor_lists[
                    name
                ].append(
                    descriptor
                )

            valid_clients.append(
                client
            )

        except Exception as exc:

            failures.append({
                "client_id": (
                    client
                ),
                "error": repr(
                    exc
                ),
            })

    if not valid_clients:
        raise RuntimeError(
            "No valid descriptors generated."
        )

    matrices = {
        name: np.vstack(
            values
        ).astype(
            float
        )
        for name, values
        in descriptor_lists.items()
    }

    return {
        "client_ids": np.asarray(
            valid_clients,
            dtype=str,
        ),

        "descriptors": matrices,

        "failures": failures,
    }


def filter_pairs(
    pairs,
    valid_client_set,
):
    return [
        pair
        for pair
        in pairs
        if (
            pair[
                0
            ]
            in valid_client_set
            and pair[
                1
            ]
            in valid_client_set
        )
    ]


def rebalance_pair_sets(
    positive_pairs,
    negative_pairs,
    seed,
):


    number = min(
        len(
            positive_pairs
        ),
        len(
            negative_pairs
        ),
    )

    if number <= 0:
        raise RuntimeError(
            "No balanced evaluation pairs remain."
        )

    positives = sample_pairs(
        positive_pairs,
        number,
        deterministic_seed(
            seed,
            "rebalance_positive",
        ),
    )

    negatives = sample_pairs(
        negative_pairs,
        number,
        deterministic_seed(
            seed,
            "rebalance_negative",
        ),
    )

    return (
        positives,
        negatives,
    )


def score_pairs(
    descriptor_matrix,
    client_ids,
    pairs,
):
    normalized = normalize_rows(
        descriptor_matrix
    )

    index = {
        str(
            client
        ): position
        for position, client
        in enumerate(
            client_ids
        )
    }

    scores = []

    for first, second in pairs:

        first_index = index[
            first
        ]

        second_index = index[
            second
        ]

        score = float(
            np.dot(
                normalized[
                    first_index
                ],
                normalized[
                    second_index
                ],
            )
        )

        scores.append(
            score
        )

    return np.asarray(
        scores,
        dtype=float,
    )


def binary_metrics_from_scores(
    positive_scores,
    negative_scores,
):
    y_true = np.concatenate(
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

    y_score = np.concatenate(
        [
            positive_scores,
            negative_scores,
        ]
    )

    auc = float(
        roc_auc_score(
            y_true,
            y_score,
        )
    )

    ap = float(
        average_precision_score(
            y_true,
            y_score,
        )
    )

    positive_mean = float(
        np.mean(
            positive_scores
        )
    )

    negative_mean = float(
        np.mean(
            negative_scores
        )
    )

    return {
        "auc": auc,
        "ap": ap,

        "positive_similarity_mean": (
            positive_mean
        ),

        "positive_similarity_median": float(
            np.median(
                positive_scores
            )
        ),

        "negative_similarity_mean": (
            negative_mean
        ),

        "negative_similarity_median": float(
            np.median(
                negative_scores
            )
        ),

        "similarity_mean_gap": float(
            positive_mean
            -
            negative_mean
        ),
    }


def bootstrap_metric_intervals(
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

        positive_indices = (
            rng.integers(
                0,
                len(
                    positive_scores
                ),
                size=len(
                    positive_scores
                ),
            )
        )

        negative_indices = (
            rng.integers(
                0,
                len(
                    negative_scores
                ),
                size=len(
                    negative_scores
                ),
            )
        )

        result = binary_metrics_from_scores(
            positive_scores[
                positive_indices
            ],
            negative_scores[
                negative_indices
            ],
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

    auc_values = np.asarray(
        auc_values,
        dtype=float,
    )

    ap_values = np.asarray(
        ap_values,
        dtype=float,
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


def paired_bootstrap_contrast(
    baseline_positive,
    baseline_negative,
    candidate_positive,
    candidate_negative,
    repetitions,
    seed,
):


    baseline_positive = np.asarray(
        baseline_positive,
        dtype=float,
    )

    baseline_negative = np.asarray(
        baseline_negative,
        dtype=float,
    )

    candidate_positive = np.asarray(
        candidate_positive,
        dtype=float,
    )

    candidate_negative = np.asarray(
        candidate_negative,
        dtype=float,
    )

    if (
        len(
            baseline_positive
        )
        !=
        len(
            candidate_positive
        )
    ):
        raise ValueError(
            "Positive score lengths differ."
        )

    if (
        len(
            baseline_negative
        )
        !=
        len(
            candidate_negative
        )
    ):
        raise ValueError(
            "Negative score lengths differ."
        )

    rng = np.random.default_rng(
        seed
    )

    auc_differences = []

    ap_differences = []

    for _ in range(
        int(
            repetitions
        )
    ):

        positive_indices = rng.integers(
            0,
            len(
                baseline_positive
            ),
            size=len(
                baseline_positive
            ),
        )

        negative_indices = rng.integers(
            0,
            len(
                baseline_negative
            ),
            size=len(
                baseline_negative
            ),
        )

        baseline_result = (
            binary_metrics_from_scores(
                baseline_positive[
                    positive_indices
                ],
                baseline_negative[
                    negative_indices
                ],
            )
        )

        candidate_result = (
            binary_metrics_from_scores(
                candidate_positive[
                    positive_indices
                ],
                candidate_negative[
                    negative_indices
                ],
            )
        )

        auc_differences.append(
            candidate_result[
                "auc"
            ]
            -
            baseline_result[
                "auc"
            ]
        )

        ap_differences.append(
            candidate_result[
                "ap"
            ]
            -
            baseline_result[
                "ap"
            ]
        )

    auc_differences = np.asarray(
        auc_differences,
        dtype=float,
    )

    ap_differences = np.asarray(
        ap_differences,
        dtype=float,
    )

    return {
        "auc_difference_mean": float(
            np.mean(
                auc_differences
            )
        ),

        "auc_difference_ci95_low": float(
            np.percentile(
                auc_differences,
                2.5,
            )
        ),

        "auc_difference_ci95_high": float(
            np.percentile(
                auc_differences,
                97.5,
            )
        ),

        "auc_probability_candidate_better": float(
            np.mean(
                auc_differences
                >
                0
            )
        ),

        "ap_difference_mean": float(
            np.mean(
                ap_differences
            )
        ),

        "ap_difference_ci95_low": float(
            np.percentile(
                ap_differences,
                2.5,
            )
        ),

        "ap_difference_ci95_high": float(
            np.percentile(
                ap_differences,
                97.5,
            )
        ),

        "ap_probability_candidate_better": float(
            np.mean(
                ap_differences
                >
                0
            )
        ),
    }


def make_neighbour_sets(
    pairs,
):
    result = defaultdict(
        set
    )

    for first, second in pairs:

        result[
            first
        ].add(
            second
        )

        result[
            second
        ].add(
            first
        )

    return result


def retrieval_metrics(
    descriptor_matrix,
    client_ids,
    training_positive_pairs,
    test_positive_pairs,
    k_values,
):


    normalized = normalize_rows(
        descriptor_matrix
    )

    client_ids = np.asarray(
        client_ids,
        dtype=str,
    )

    index = {
        client: position
        for position, client
        in enumerate(
            client_ids
        )
    }

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

    maximum_k = max(
        int(
            value
        )
        for value
        in k_values
    )

    recall_values = {
        int(
            k
        ): []
        for k
        in k_values
    }

    hit_values = {
        int(
            k
        ): []
        for k
        in k_values
    }

    precision_values = {
        int(
            k
        ): []
        for k
        in k_values
    }

    reciprocal_ranks = []

    roots_evaluated = 0

    for root, truth in (
        test_neighbours.items()
    ):

        if root not in index:
            continue

        valid_truth = {
            neighbour
            for neighbour
            in truth
            if neighbour in index
        }

        if not valid_truth:
            continue

        root_index = index[
            root
        ]

        scores = (
            normalized
            @
            normalized[
                root_index
            ]
        )

        scores = scores.copy()


        scores[
            root_index
        ] = -np.inf


        for neighbour in (
            training_neighbours.get(
                root,
                set(),
            )
        ):

            if neighbour in index:
                scores[
                    index[
                        neighbour
                    ]
                ] = -np.inf

        finite_count = int(
            np.sum(
                np.isfinite(
                    scores
                )
            )
        )

        if finite_count <= 0:
            continue

        roots_evaluated += 1

        ordering = np.argsort(
            -scores,
            kind="stable",
        )

        ordering = [
            int(
                candidate_index
            )
            for candidate_index
            in ordering
            if np.isfinite(
                scores[
                    candidate_index
                ]
            )
        ]

        ranked_clients = [
            str(
                client_ids[
                    candidate_index
                ]
            )
            for candidate_index
            in ordering
        ]


        first_relevant_rank = None

        for rank, candidate in enumerate(
            ranked_clients,
            start=1,
        ):

            if candidate in valid_truth:
                first_relevant_rank = rank
                break

        if first_relevant_rank is None:
            reciprocal_ranks.append(
                0.0
            )
        else:
            reciprocal_ranks.append(
                1.0
                /
                float(
                    first_relevant_rank
                )
            )


        for k in k_values:

            requested_k = int(
                k
            )

            effective_k = min(
                requested_k,
                len(
                    ranked_clients
                ),
            )

            if effective_k <= 0:
                continue

            retrieved = set(
                ranked_clients[
                    :effective_k
                ]
            )

            hits = len(
                retrieved
                &
                valid_truth
            )

            recall_values[
                requested_k
            ].append(
                hits
                /
                len(
                    valid_truth
                )
            )

            precision_values[
                requested_k
            ].append(
                hits
                /
                effective_k
            )

            hit_values[
                requested_k
            ].append(
                1.0
                if hits > 0
                else 0.0
            )

    result = {
        "retrieval_roots": int(
            roots_evaluated
        ),

        "mrr": (
            float(
                np.mean(
                    reciprocal_ranks
                )
            )
            if reciprocal_ranks
            else np.nan
        ),
    }

    for k in k_values:

        k = int(
            k
        )

        result[
            f"recall_at_{k}"
        ] = (
            float(
                np.mean(
                    recall_values[
                        k
                    ]
                )
            )
            if recall_values[
                k
            ]
            else np.nan
        )

        result[
            f"precision_at_{k}"
        ] = (
            float(
                np.mean(
                    precision_values[
                        k
                    ]
                )
            )
            if precision_values[
                k
            ]
            else np.nan
        )

        result[
            f"hit_rate_at_{k}"
        ] = (
            float(
                np.mean(
                    hit_values[
                        k
                    ]
                )
            )
            if hit_values[
                k
            ]
            else np.nan
        )

    return result


def relation_specific_metrics(
    descriptor_name,
    positive_scores,
    negative_scores,
    positive_pairs,
    pair_relations,
    seed,
):
    pair_to_score = {
        pair: float(
            score
        )
        for pair, score
        in zip(
            positive_pairs,
            positive_scores,
        )
    }

    relation_pairs = defaultdict(
        list
    )

    for pair in positive_pairs:

        for relation in (
            pair_relations.get(
                pair,
                set(),
            )
        ):
            relation_pairs[
                relation
            ].append(
                pair
            )

    rows = []

    rng = np.random.default_rng(
        deterministic_seed(
            descriptor_name,
            seed,
            "relation_negative_scores",
        )
    )

    for relation, pairs in sorted(
        relation_pairs.items()
    ):

        relation_positive_scores = (
            np.asarray(
                [
                    pair_to_score[
                        pair
                    ]
                    for pair
                    in pairs
                ],
                dtype=float,
            )
        )

        number = min(
            len(
                relation_positive_scores
            ),
            len(
                negative_scores
            ),
        )

        if number == 0:
            continue

        if number < len(
            relation_positive_scores
        ):

            positive_indices = (
                rng.choice(
                    len(
                        relation_positive_scores
                    ),
                    size=number,
                    replace=False,
                )
            )

            relation_positive_scores = (
                relation_positive_scores[
                    positive_indices
                ]
            )

        if number == len(
            negative_scores
        ):

            relation_negative_scores = (
                negative_scores
            )

        else:

            negative_indices = (
                rng.choice(
                    len(
                        negative_scores
                    ),
                    size=number,
                    replace=False,
                )
            )

            relation_negative_scores = (
                negative_scores[
                    negative_indices
                ]
            )

        metrics = binary_metrics_from_scores(
            relation_positive_scores,
            relation_negative_scores,
        )

        rows.append({
            "descriptor": (
                descriptor_name
            ),

            "relation": (
                relation
            ),

            "positive_pairs": int(
                number
            ),

            "negative_pairs": int(
                number
            ),

            **metrics,
        })

    return rows


def pair_manifest_dataframe(
    positive_pairs,
    negative_pairs,
    pair_relations,
):
    rows = []

    for first, second in positive_pairs:

        rows.append({
            "pair_type": "positive",
            "source": first,
            "target": second,
            "relations": "|".join(
                sorted(
                    pair_relations.get(
                        (
                            first,
                            second,
                        ),
                        set(),
                    )
                )
            ),
        })

    for first, second in negative_pairs:

        rows.append({
            "pair_type": "negative",
            "source": first,
            "target": second,
            "relations": "",
        })

    return pd.DataFrame(
        rows
    )


def run_link_prediction_experiment(
    dataset_name="enron",
    output_folder=DEFAULT_OUTPUT_FOLDER,
    test_positives=DEFAULT_TEST_POSITIVES,
    negative_ratio=DEFAULT_NEGATIVE_RATIO,
    holdout_policy=DEFAULT_HOLDOUT_POLICY,
    bootstrap_repetitions=(
        DEFAULT_BOOTSTRAP_REPETITIONS
    ),
    d_max=DEFAULT_D_MAX,
    degree_weight=DEFAULT_DEGREE_WEIGHT,
    k_values=DEFAULT_K_VALUES,
    seed=DEFAULT_SEED,
):
    start_time = time.perf_counter()

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
            "holdout_policy must be one of: "
            "auto, direct_only, "
            "strict_common_content"
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
        "=" * 68
    )
    print(
        "LEAKAGE-SAFE DESCRIPTOR LINK PREDICTION"
    )
    print(
        "=" * 68
    )

    print(
        f"Dataset: {dataset_name}"
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

    print(
        f"Client nodes: "
        f"{len(clients):,}"
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

    if not full_positive_pairs:
        raise RuntimeError(
            "No client-client pairs found."
        )

    print(
        "Original unique client-client pairs: "
        f"{len(full_positive_pairs):,}"
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

    test_candidates = split[
        "cycle_pairs"
    ]

    print(
        f"Forced training forest pairs: "
        f"{len(forest_pairs):,}"
    )

    print(
        f"Eligible cycle test pairs: "
        f"{len(test_candidates):,}"
    )

    if not test_candidates:
        raise RuntimeError(
            "No cycle edges available for testing."
        )


    test_positive_pairs = sample_pairs(
        test_candidates,
        test_positives,
        deterministic_seed(
            dataset_name,
            seed,
            "test_positive_pairs",
        ),
    )

    print(
        f"Test positive pairs: "
        f"{len(test_positive_pairs):,}"
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

    print(
        f"Test negative pairs: "
        f"{len(negative_pairs):,}"
    )


    holdout = construct_training_graph(
        original_graph=original_graph,

        test_positive_pairs=(
            test_positive_pairs
        ),

        dataset_name=dataset_name,

        holdout_policy=(
            holdout_policy
        ),

        seed=seed,
    )

    training_graph = holdout[
        "graph"
    ]

    print(
        "Direct physical edges removed: "
        f"{holdout['direct_edges_removed']:,}"
    )

    if holdout_policy == (
        "strict_common_content"
    ):

        print(
            "Test pairs originally sharing content: "
            f"{holdout['test_pairs_with_original_common_content']:,}"
        )

        print(
            "Original common-content instances: "
            f"{holdout['original_common_content_instances']:,}"
        )

        print(
            "Content-incidence physical edges removed: "
            f"{holdout['content_edges_removed']:,}"
        )


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
            "Train/test CC leakage: "
            f"{len(leakage_intersection)} "
            "test pairs remain in training."
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
            "Connectivity validation failed: "
            f"{len(missing_forest)} forced "
            "forest pairs disappeared."
        )

    print(
        "Direct CC train/test leakage: 0"
    )

    print(
        "Forced forest preservation: PASS"
    )

    if holdout_policy == (
        "strict_common_content"
    ):
        print(
            "Direct common-content leakage: 0"
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

    print(
        f"Valid descriptor clients: "
        f"{len(descriptor_client_ids):,}"
    )

    print(
        f"Descriptor failures: "
        f"{len(generated['failures']):,}"
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

    print(
        f"Final positive pairs: "
        f"{len(test_positive_pairs):,}"
    )

    print(
        f"Final negative pairs: "
        f"{len(negative_pairs):,}"
    )


    valid_training_pairs = [
        pair
        for pair
        in training_positive_pairs
        if (
            pair[
                0
            ]
            in valid_client_set
            and pair[
                1
            ]
            in valid_client_set
        )
    ]


    metric_rows = []

    relation_rows = []

    score_cache = {}

    pair_score_table = (
        pair_manifest_dataframe(
            test_positive_pairs,
            negative_pairs,
            pair_relations,
        )
    )

    for descriptor_name, matrix in (
        descriptor_matrices.items()
    ):

        print(
            "[evaluation] "
            f"{descriptor_name}"
        )

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

        metrics = (
            binary_metrics_from_scores(
                positive_scores,
                negative_scores,
            )
        )

        intervals = (
            bootstrap_metric_intervals(
                positive_scores,
                negative_scores,

                repetitions=(
                    bootstrap_repetitions
                ),

                seed=deterministic_seed(
                    dataset_name,
                    descriptor_name,
                    seed,
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

        specification = next(
            specification
            for specification
            in DESCRIPTOR_SPECS
            if specification[
                "name"
            ]
            ==
            descriptor_name
        )

        metric_rows.append({
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

            **metrics,
            **intervals,
            **retrieval,
        })


        pair_score_table[
            descriptor_name
        ] = np.concatenate(
            [
                positive_scores,
                negative_scores,
            ]
        )


        relation_rows.extend(
            relation_specific_metrics(
                descriptor_name=(
                    descriptor_name
                ),

                positive_scores=(
                    positive_scores
                ),

                negative_scores=(
                    negative_scores
                ),

                positive_pairs=(
                    test_positive_pairs
                ),

                pair_relations=(
                    pair_relations
                ),

                seed=seed,
            )
        )

    metrics_dataframe = pd.DataFrame(
        metric_rows
    )

    metrics_dataframe = (
        metrics_dataframe
        .sort_values(
            by=[
                "auc",
                "ap",
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

    metrics_dataframe.insert(
        0,
        "rank",
        np.arange(
            1,
            len(
                metrics_dataframe
            )
            + 1
        ),
    )

    relation_dataframe = pd.DataFrame(
        relation_rows
    )


    if BASELINE_DESCRIPTOR not in (
        score_cache
    ):
        raise RuntimeError(
            "HKS baseline missing."
        )

    baseline_scores = (
        score_cache[
            BASELINE_DESCRIPTOR
        ]
    )

    contrast_rows = []

    for descriptor_name in (
        descriptor_matrices.keys()
    ):

        if descriptor_name == (
            BASELINE_DESCRIPTOR
        ):
            continue

        candidate_scores = (
            score_cache[
                descriptor_name
            ]
        )

        contrast = (
            paired_bootstrap_contrast(
                baseline_positive=(
                    baseline_scores[
                        "positive"
                    ]
                ),

                baseline_negative=(
                    baseline_scores[
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
                    descriptor_name,
                    seed,
                    "paired_contrast",
                ),
            )
        )

        contrast_rows.append({
            "baseline": (
                BASELINE_DESCRIPTOR
            ),

            "candidate": (
                descriptor_name
            ),

            **contrast,
        })

    contrast_dataframe = pd.DataFrame(
        contrast_rows
    )


    test_relation_counts = (
        defaultdict(
            int
        )
    )

    for pair in test_positive_pairs:

        for relation in (
            pair_relations.get(
                pair,
                set(),
            )
        ):

            test_relation_counts[
                relation
            ] += 1


    experiment_folder = os.path.join(
        output_folder,
        dataset_name,
        "experiments",
        "link_prediction_leakage_safe",
    )

    os.makedirs(
        experiment_folder,
        exist_ok=True,
    )

    metrics_path = os.path.join(
        experiment_folder,
        "link_prediction_descriptor_metrics.csv",
    )

    contrast_path = os.path.join(
        experiment_folder,
        "link_prediction_descriptor_contrasts.csv",
    )

    relation_path = os.path.join(
        experiment_folder,
        "link_prediction_by_relation.csv",
    )

    pairs_path = os.path.join(
        experiment_folder,
        "link_prediction_pairs_and_scores.csv.gz",
    )

    split_path = os.path.join(
        experiment_folder,
        "link_prediction_split.csv",
    )

    leakage_path = os.path.join(
        experiment_folder,
        "link_prediction_leakage_diagnostics.csv",
    )

    metadata_path = os.path.join(
        experiment_folder,
        "link_prediction_metadata.json",
    )

    metrics_dataframe.to_csv(
        metrics_path,
        index=False,
    )

    contrast_dataframe.to_csv(
        contrast_path,
        index=False,
    )

    relation_dataframe.to_csv(
        relation_path,
        index=False,
    )

    pair_score_table.to_csv(
        pairs_path,
        index=False,
        compression="gzip",
    )


    split_rows = []

    forest_set = set(
        forest_pairs
    )

    final_test_set = set(
        test_positive_pairs
    )

    for pair in full_positive_pairs:

        if pair in final_test_set:
            split_label = "test"
        elif pair in forest_set:
            split_label = (
                "train_forced_forest"
            )
        else:
            split_label = (
                "train_cycle"
            )

        split_rows.append({
            "source": pair[
                0
            ],
            "target": pair[
                1
            ],
            "split": (
                split_label
            ),
            "relations": "|".join(
                sorted(
                    pair_relations.get(
                        pair,
                        set(),
                    )
                )
            ),
        })

    pd.DataFrame(
        split_rows
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


    runtime = float(
        time.perf_counter()
        -
        start_time
    )

    metadata = {
        "schema_version": (
            LINK_PREDICTION_SCHEMA_VERSION
        ),

        "dataset": (
            dataset_name
        ),

        "global_graph": (
            graph_path
        ),

        "seed": int(
            seed
        ),

        "client_nodes": int(
            len(
                clients
            )
        ),

        "original_client_client_pairs": int(
            len(
                full_positive_pairs
            )
        ),

        "forced_spanning_forest_pairs": int(
            len(
                forest_pairs
            )
        ),

        "eligible_cycle_pairs": int(
            len(
                test_candidates
            )
        ),

        "test_positive_pairs": int(
            len(
                test_positive_pairs
            )
        ),

        "test_negative_pairs": int(
            len(
                negative_pairs
            )
        ),

        "negative_ratio": float(
            negative_ratio
        ),

        "holdout_policy": (
            holdout_policy
        ),

        "direct_physical_edges_removed": int(
            holdout[
                "direct_edges_removed"
            ]
        ),

        "content_incidence_edges_removed": int(
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

        "descriptors": list(
            DESCRIPTOR_SPECS
        ),

        "baseline_descriptor": (
            BASELINE_DESCRIPTOR
        ),

        "bootstrap_repetitions": int(
            bootstrap_repetitions
        ),

        "k_values": [
            int(
                value
            )
            for value
            in k_values
        ],

        "test_relation_counts": dict(
            test_relation_counts
        ),

        "privacy": (
            "NONE. This experiment selects the "
            "clean representation before DP."
        ),

        "split_statement": (
            "The client-client split is created before ego "
            "construction. A randomized spanning forest is forced "
            "into training and test positives are sampled only from "
            "remaining cycle edges."
        ),

        "negative_statement": (
            "Negatives are sampled from pairs absent from the FULL "
            "original client-client positive universe, preventing "
            "sampled false negatives."
        ),

        "leakage_statement": (
            "All direct physical edges between every held-out client "
            "pair are removed before local ego construction. Under "
            "strict_common_content, immediate common-content evidence "
            "between a test pair is also broken by removing one "
            "endpoint-content incidence for each shared content node."
        ),

        "strict_policy_limitation": (
            "strict_common_content is a conservative experimental "
            "control against immediate heterogeneous leakage. It does "
            "not claim removal of every possible longer indirect path."
        ),

        "spanning_forest_limitation": (
            "For connectivity preservation, held-out positives are "
            "sampled from non-forest cycle edges. Therefore the test "
            "set emphasizes redundant/cyclic client-client structure "
            "rather than uniformly sampling every original edge."
        ),

        "outputs": {
            "metrics": os.path.basename(
                metrics_path
            ),

            "contrasts": os.path.basename(
                contrast_path
            ),

            "by_relation": os.path.basename(
                relation_path
            ),

            "pairs_and_scores": os.path.basename(
                pairs_path
            ),

            "split": os.path.basename(
                split_path
            ),

            "leakage_diagnostics": os.path.basename(
                leakage_path
            ),
        },

        "runtime_seconds": (
            runtime
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
        "=" * 68
    )

    print(
        "LEAKAGE-SAFE LINK PREDICTION COMPLETE"
    )

    print(
        "=" * 68
    )

    print(
        f"Holdout policy: "
        f"{holdout_policy}"
    )

    print(
        f"Final positives: "
        f"{len(test_positive_pairs):,}"
    )

    print(
        f"Final negatives: "
        f"{len(negative_pairs):,}"
    )

    print(
        f"Descriptor clients: "
        f"{len(descriptor_client_ids):,}"
    )

    print()
    print(
        "Test relations:"
    )

    for relation, count in sorted(
        test_relation_counts.items(),
        key=lambda item: (
            -item[
                1
            ],
            item[
                0
            ],
        ),
    ):

        print(
            f"  {relation:25s} "
            f"{count:,}"
        )

    print()
    print(
        "Descriptor results:"
    )

    console_columns = [
        "rank",
        "descriptor",
        "auc",
        "auc_ci95_low",
        "auc_ci95_high",
        "ap",
        "ap_ci95_low",
        "ap_ci95_high",
        "recall_at_20",
        "hit_rate_at_20",
        "mrr",
    ]

    print(
        metrics_dataframe[
            console_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "Paired comparison against HKS:"
    )

    if len(
        contrast_dataframe
    ):

        contrast_columns = [
            "candidate",
            "auc_difference_mean",
            "auc_difference_ci95_low",
            "auc_difference_ci95_high",
            "auc_probability_candidate_better",
            "ap_difference_mean",
            "ap_difference_ci95_low",
            "ap_difference_ci95_high",
            "ap_probability_candidate_better",
        ]

        print(
            contrast_dataframe[
                contrast_columns
            ].to_string(
                index=False
            )
        )

    print()
    print(
        "Leakage validation:"
    )

    print(
        "  Direct held-out CC edges remaining: 0"
    )

    if holdout_policy == (
        "strict_common_content"
    ):

        print(
            "  Direct common-content evidence remaining: 0"
        )

        print(
            "  Content-incidence edges removed: "
            f"{holdout['content_edges_removed']:,}"
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  No privacy noise was applied."
    )

    print(
        "  Descriptor egos were built only after the "
        "training graph was constructed."
    )

    print(
        "  If an eigen descriptor does not beat HKS here, "
        "we should not promote it to the privacy pipeline."
    )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  {metrics_path}"
    )

    print(
        f"  {contrast_path}"
    )

    print(
        f"  {relation_path}"
    )

    print(
        f"  {pairs_path}"
    )

    print(
        f"  {split_path}"
    )

    print(
        f"  {leakage_path}"
    )

    print(
        f"  {metadata_path}"
    )

    print()
    print(
        f"Runtime: "
        f"{runtime:.3f} s"
    )

    print(
        "=" * 68
    )

    print()

    return {
        "metrics": (
            metrics_dataframe
        ),

        "contrasts": (
            contrast_dataframe
        ),

        "by_relation": (
            relation_dataframe
        ),

        "metadata": (
            metadata
        ),
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe clean descriptor "
            "link-prediction experiment."
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

    parser.add_argument(
        "--seed",
        type=int,
        default=(
            DEFAULT_SEED
        ),
    )

    args = parser.parse_args()

    run_link_prediction_experiment(
        dataset_name=(
            args.dataset
        ),

        output_folder=(
            args.output_folder
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

        seed=(
            args.seed
        ),
    )
