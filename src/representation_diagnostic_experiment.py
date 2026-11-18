import argparse
import hashlib
import json
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
    hks_times_for_dimension,
    normalized_laplacian_dense,
    typed_root_degree,
)


REPRESENTATION_DIAGNOSTIC_SCHEMA_VERSION = "2.0"

DEFAULT_OUTPUT_FOLDER = "output_local_test"

DEFAULT_DIMENSIONS = (
    4,
    6,
    8,
    12,
)

DEFAULT_NORMALIZATIONS = (
    "block",
    "global",
)

DEFAULT_DESCRIPTOR_FAMILIES = (
    "hks",
    "eigen",
    "moments",
    "hks_eigen",
    "hks_degree",
    "hks_eigen_degree",
)


DEFAULT_DIVERSITY_CLIENTS = 1500

DEFAULT_POSITIVE_PAIRS = 1000

DEFAULT_NEGATIVE_RATIO = 1.0

DEFAULT_D_MAX = 50

DEFAULT_DEGREE_WEIGHT = 1.0

DEFAULT_SEED = 0


DEFAULT_MIN_POSITIVES_FOR_RANKING = 100


DEFAULT_MIN_RELATION_POSITIVES = 20

NUMERICAL_EPS = 1e-12


def deterministic_seed(*parts):
    text = "|".join(str(part) for part in parts)

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="little",
        signed=False,
    )


def normalize_vector(vector):
    vector = np.asarray(
        vector,
        dtype=float,
    )

    norm = float(
        np.linalg.norm(vector)
    )

    if norm <= NUMERICAL_EPS:
        return np.zeros_like(vector)

    return vector / norm


def normalize_rows(matrix):
    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    return np.divide(
        matrix,
        norms,
        out=np.zeros_like(matrix),
        where=norms > NUMERICAL_EPS,
    )


def combine_blocks(
    first,
    second,
    normalization,
):


    first = np.asarray(
        first,
        dtype=float,
    )

    second = np.asarray(
        second,
        dtype=float,
    )

    if normalization == "block":
        combined = np.concatenate(
            [
                normalize_vector(first),
                normalize_vector(second),
            ]
        )

        return normalize_vector(
            combined
        )

    if normalization == "global":
        return normalize_vector(
            np.concatenate(
                [
                    first,
                    second,
                ]
            )
        )

    raise ValueError(
        f"Unknown normalization: {normalization}"
    )


def all_client_ids(graph):


    clients = [
        str(node)
        for node, data
        in graph.nodes(data=True)
        if str(
            data.get(
                "role",
                "",
            )
        ).lower() == "client"
    ]

    return sorted(clients)


def sample_items(
    items,
    requested,
    seed,
):
    items = list(items)

    if requested is None:
        return list(items)

    requested = min(
        int(requested),
        len(items),
    )

    if requested >= len(items):
        return list(items)

    rng = np.random.default_rng(
        seed
    )

    indices = np.sort(
        rng.choice(
            len(items),
            size=requested,
            replace=False,
        )
    )

    return [
        items[int(index)]
        for index in indices
    ]


def compute_spectral_profile(
    projected_graph,
    root,
):


    if root not in projected_graph:
        return {
            "active": False,
            "eigenvalues": np.asarray(
                [],
                dtype=float,
            ),
            "root_weights": np.asarray(
                [],
                dtype=float,
            ),
        }

    root_degree = float(
        projected_graph.degree(
            root,
            weight="weight",
        )
    )

    if root_degree <= 0:
        return {
            "active": False,
            "eigenvalues": np.asarray(
                [],
                dtype=float,
            ),
            "root_weights": np.asarray(
                [],
                dtype=float,
            ),
        }

    (
        laplacian,
        nodes,
        node_index,
    ) = normalized_laplacian_dense(
        projected_graph
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        laplacian
    )


    eigenvalues = np.clip(
        eigenvalues,
        0.0,
        2.0,
    )

    root_position = node_index[
        root
    ]

    root_weights = (
        eigenvectors[
            root_position,
            :
        ]
        ** 2
    )

    return {
        "active": True,
        "eigenvalues": eigenvalues,
        "root_weights": root_weights,
    }


def hks_from_profile(
    profile,
    times,
):
    if not profile["active"]:
        return np.zeros(
            len(times),
            dtype=float,
        )

    eigenvalues = profile[
        "eigenvalues"
    ]

    root_weights = profile[
        "root_weights"
    ]

    values = []

    for time_value in times:
        value = np.sum(
            np.exp(
                -float(time_value)
                * eigenvalues
            )
            * root_weights
        )

        values.append(
            float(value)
        )

    return np.asarray(
        values,
        dtype=float,
    )


def eigenvalue_descriptor(
    profile,
    count,
):


    count = int(count)

    result = np.zeros(
        count,
        dtype=float,
    )

    if not profile["active"]:
        return result

    eigenvalues = np.asarray(
        profile["eigenvalues"],
        dtype=float,
    )

    nontrivial = eigenvalues[
        eigenvalues > 1e-10
    ]

    number = min(
        count,
        len(nontrivial),
    )

    if number > 0:
        result[:number] = (
            nontrivial[:number]
            / 2.0
        )

    return result


def spectral_moment_descriptor(
    profile,
    count,
):


    count = int(count)

    if not profile["active"]:
        return np.zeros(
            count,
            dtype=float,
        )

    eigenvalues = profile[
        "eigenvalues"
    ]

    root_weights = profile[
        "root_weights"
    ]

    values = []

    for power in range(
        2,
        count + 2,
    ):
        raw = float(
            np.sum(
                (
                    eigenvalues
                    ** power
                )
                * root_weights
            )
        )

        bounded = (
            raw
            /
            (
                2.0
                ** power
            )
        )

        values.append(
            bounded
        )

    return np.asarray(
        values,
        dtype=float,
    )


def build_descriptors_from_profiles(
    cc_profile,
    cx_profile,
    degree,
    dimension,
    normalization,
    d_max,
    degree_weight,
):


    block_dimension = (
        int(dimension)
        // 2
    )

    times = hks_times_for_dimension(
        dimension
    )


    cc_hks = hks_from_profile(
        cc_profile,
        times,
    )

    cx_hks = hks_from_profile(
        cx_profile,
        times,
    )

    hks = combine_blocks(
        cc_hks,
        cx_hks,
        normalization,
    )


    cc_eigen = eigenvalue_descriptor(
        cc_profile,
        block_dimension,
    )

    cx_eigen = eigenvalue_descriptor(
        cx_profile,
        block_dimension,
    )

    eigen = combine_blocks(
        cc_eigen,
        cx_eigen,
        normalization,
    )


    cc_moments = (
        spectral_moment_descriptor(
            cc_profile,
            block_dimension,
        )
    )

    cx_moments = (
        spectral_moment_descriptor(
            cx_profile,
            block_dimension,
        )
    )

    moments = combine_blocks(
        cc_moments,
        cx_moments,
        normalization,
    )


    bounded_degree = np.clip(
        np.asarray(
            degree,
            dtype=float,
        ),
        0.0,
        float(d_max),
    )

    degree_features = (
        bounded_degree
        /
        float(d_max)
    )

    degree_features = (
        float(degree_weight)
        * degree_features
    )


    hks_eigen = normalize_vector(
        np.concatenate(
            [
                hks,
                eigen,
            ]
        )
    )

    hks_degree = normalize_vector(
        np.concatenate(
            [
                hks,
                degree_features,
            ]
        )
    )

    hks_eigen_degree = normalize_vector(
        np.concatenate(
            [
                hks,
                eigen,
                degree_features,
            ]
        )
    )

    return {
        "hks": hks,
        "eigen": eigen,
        "moments": moments,
        "hks_eigen": hks_eigen,
        "hks_degree": hks_degree,
        "hks_eigen_degree": (
            hks_eigen_degree
        ),
    }


def generate_descriptors(
    graph,
    required_clients,
    dimensions,
    normalizations,
    families,
    d_max,
    degree_weight,
):


    required_clients = sorted(
        set(
            str(client)
            for client
            in required_clients
        )
    )

    descriptor_lists = {
        (
            dimension,
            normalization,
            family,
        ): []
        for dimension
        in dimensions
        for normalization
        in normalizations
        for family
        in families
    }

    valid_clients = []

    failures = []

    for client_index, client in enumerate(
        required_clients,
        start=1,
    ):
        if (
            client_index == 1
            or client_index % 100 == 0
            or client_index
            == len(required_clients)
        ):
            print(
                "[representation] "
                f"{client_index}/"
                f"{len(required_clients)}"
            )

        try:
            ego = create_true_ego_graph(
                graph,
                client,
            )

            degree = typed_root_degree(
                ego,
                client,
            ).astype(float)

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

            temporary = {}

            for dimension in dimensions:
                for normalization in normalizations:
                    descriptor_bundle = (
                        build_descriptors_from_profiles(
                            cc_profile=cc_profile,
                            cx_profile=cx_profile,
                            degree=degree,
                            dimension=dimension,
                            normalization=(
                                normalization
                            ),
                            d_max=d_max,
                            degree_weight=(
                                degree_weight
                            ),
                        )
                    )

                    for family in families:
                        temporary[
                            (
                                dimension,
                                normalization,
                                family,
                            )
                        ] = np.asarray(
                            descriptor_bundle[
                                family
                            ],
                            dtype=float,
                        )

            valid_clients.append(
                str(client)
            )

            for key, value in temporary.items():
                descriptor_lists[
                    key
                ].append(
                    value
                )

        except Exception as exc:
            failures.append({
                "client_id": str(client),
                "error": repr(exc),
            })

    if not valid_clients:
        raise RuntimeError(
            "No valid descriptors were generated."
        )

    matrices = {}

    for key, values in descriptor_lists.items():
        matrices[key] = np.vstack(
            values
        ).astype(float)

    return {
        "client_ids": np.asarray(
            valid_clients,
            dtype=str,
        ),
        "descriptors": matrices,
        "failures": failures,
    }


def collect_client_client_pairs(
    graph,
):


    pair_relations = defaultdict(
        set
    )

    relation_edge_counts = defaultdict(
        int
    )

    if graph.is_multigraph():
        iterator = graph.edges(
            keys=True,
            data=True,
        )

        for (
            source,
            target,
            key,
            data,
        ) in iterator:
            if source == target:
                continue

            source_role = str(
                graph.nodes[
                    source
                ].get(
                    "role",
                    "",
                )
            ).lower()

            target_role = str(
                graph.nodes[
                    target
                ].get(
                    "role",
                    "",
                )
            ).lower()

            if (
                source_role != "client"
                or target_role != "client"
            ):
                continue

            source = str(source)
            target = str(target)

            pair = tuple(
                sorted(
                    (
                        source,
                        target,
                    )
                )
            )

            relation = str(
                data.get(
                    "relation",
                    "unknown",
                )
            )

            pair_relations[
                pair
            ].add(
                relation
            )

            relation_edge_counts[
                relation
            ] += 1

    else:
        iterator = graph.edges(
            data=True
        )

        for (
            source,
            target,
            data,
        ) in iterator:
            if source == target:
                continue

            source_role = str(
                graph.nodes[
                    source
                ].get(
                    "role",
                    "",
                )
            ).lower()

            target_role = str(
                graph.nodes[
                    target
                ].get(
                    "role",
                    "",
                )
            ).lower()

            if (
                source_role != "client"
                or target_role != "client"
            ):
                continue

            source = str(source)
            target = str(target)

            pair = tuple(
                sorted(
                    (
                        source,
                        target,
                    )
                )
            )

            relation = str(
                data.get(
                    "relation",
                    "unknown",
                )
            )

            pair_relations[
                pair
            ].add(
                relation
            )

            relation_edge_counts[
                relation
            ] += 1

    relation_pair_counts = defaultdict(
        int
    )

    for pair, relations in pair_relations.items():
        for relation in relations:
            relation_pair_counts[
                relation
            ] += 1

    return {
        "pair_relations": dict(
            pair_relations
        ),
        "relation_edge_counts": dict(
            relation_edge_counts
        ),
        "relation_pair_counts": dict(
            relation_pair_counts
        ),
    }


def sample_positive_pairs(
    pair_relations,
    requested,
    seed,
):
    all_pairs = sorted(
        pair_relations.keys()
    )

    if not all_pairs:
        return []

    return sample_items(
        all_pairs,
        requested,
        seed,
    )


def sample_negative_pairs(
    client_ids,
    full_positive_pairs,
    requested,
    seed,
):


    clients = list(
        str(client)
        for client
        in client_ids
    )

    full_positive_set = set(
        full_positive_pairs
    )

    n = len(clients)

    maximum_possible = (
        n
        * (n - 1)
        // 2
        -
        len(full_positive_set)
    )

    requested = min(
        int(requested),
        int(maximum_possible),
    )

    rng = np.random.default_rng(
        seed
    )

    negatives = set()

    attempts = 0

    maximum_attempts = max(
        10000,
        requested * 50,
    )

    while (
        len(negatives) < requested
        and attempts < maximum_attempts
    ):
        attempts += 1

        first_index = int(
            rng.integers(
                0,
                n,
            )
        )

        second_index = int(
            rng.integers(
                0,
                n,
            )
        )

        if first_index == second_index:
            continue

        pair = tuple(
            sorted(
                (
                    clients[
                        first_index
                    ],
                    clients[
                        second_index
                    ],
                )
            )
        )

        if pair in full_positive_set:
            continue

        negatives.add(
            pair
        )


    if len(negatives) < requested:
        for first_index in range(n):
            for second_index in range(
                first_index + 1,
                n,
            ):
                pair = (
                    clients[first_index],
                    clients[second_index],
                )

                if (
                    pair in full_positive_set
                    or pair in negatives
                ):
                    continue

                negatives.add(
                    pair
                )

                if len(negatives) >= requested:
                    break

            if len(negatives) >= requested:
                break

    return sorted(
        negatives
    )


def pair_endpoints(pairs):
    result = set()

    for first, second in pairs:
        result.add(
            str(first)
        )
        result.add(
            str(second)
        )

    return result


def filter_pairs_to_valid_clients(
    pairs,
    valid_client_set,
):
    return [
        pair
        for pair
        in pairs
        if (
            pair[0] in valid_client_set
            and pair[1] in valid_client_set
        )
    ]


def score_pairs(
    signatures,
    client_ids,
    pairs,
):
    normalized = normalize_rows(
        signatures
    )

    index = {
        str(client): position
        for position, client
        in enumerate(client_ids)
    }

    scores = []

    for first, second in pairs:
        first_index = index[
            str(first)
        ]

        second_index = index[
            str(second)
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


def classification_metrics(
    signatures,
    client_ids,
    positive_pairs,
    negative_pairs,
):
    if (
        not positive_pairs
        or not negative_pairs
    ):
        return {
            "clean_auc": np.nan,
            "clean_ap": np.nan,
            "positive_similarity_mean": np.nan,
            "positive_similarity_median": np.nan,
            "negative_similarity_mean": np.nan,
            "negative_similarity_median": np.nan,
            "similarity_mean_gap": np.nan,
        }

    positive_scores = score_pairs(
        signatures,
        client_ids,
        positive_pairs,
    )

    negative_scores = score_pairs(
        signatures,
        client_ids,
        negative_pairs,
    )

    y_true = np.concatenate(
        [
            np.ones(
                len(positive_scores),
                dtype=int,
            ),
            np.zeros(
                len(negative_scores),
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
        "clean_auc": float(
            roc_auc_score(
                y_true,
                y_score,
            )
        ),
        "clean_ap": float(
            average_precision_score(
                y_true,
                y_score,
            )
        ),
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


def upper_triangle_values(
    matrix,
):
    if len(matrix) < 2:
        return np.asarray(
            [],
            dtype=float,
        )

    return matrix[
        np.triu_indices(
            len(matrix),
            k=1,
        )
    ]


def descriptor_diversity_statistics(
    signatures,
):
    signatures = np.asarray(
        signatures,
        dtype=float,
    )

    normalized = normalize_rows(
        signatures
    )

    cosine_matrix = (
        normalized
        @ normalized.T
    )

    cosine_values = (
        upper_triangle_values(
            cosine_matrix
        )
    )

    rounded = np.round(
        signatures,
        decimals=8,
    )

    unique_count = int(
        len(
            np.unique(
                rounded,
                axis=0,
            )
        )
    )

    norms = np.linalg.norm(
        signatures,
        axis=1,
    )

    result = {
        "unique_signature_count": (
            unique_count
        ),
        "unique_signature_fraction": float(
            unique_count
            /
            len(signatures)
        ),
        "zero_signature_fraction": float(
            np.mean(
                norms
                <= NUMERICAL_EPS
            )
        ),
        "mean_signature_norm": float(
            np.mean(norms)
        ),
    }

    if len(cosine_values):
        result.update({
            "mean_pairwise_cosine": float(
                np.mean(
                    cosine_values
                )
            ),
            "median_pairwise_cosine": float(
                np.median(
                    cosine_values
                )
            ),
            "pairwise_cosine_std": float(
                np.std(
                    cosine_values
                )
            ),
            "pairwise_cosine_p05": float(
                np.percentile(
                    cosine_values,
                    5,
                )
            ),
            "pairwise_cosine_p95": float(
                np.percentile(
                    cosine_values,
                    95,
                )
            ),
        })
    else:
        result.update({
            "mean_pairwise_cosine": np.nan,
            "median_pairwise_cosine": np.nan,
            "pairwise_cosine_std": np.nan,
            "pairwise_cosine_p05": np.nan,
            "pairwise_cosine_p95": np.nan,
        })

    return result


def effective_rank_statistics(
    signatures,
):


    matrix = np.asarray(
        signatures,
        dtype=float,
    )

    if len(matrix) < 2:
        return {
            "effective_rank": np.nan,
            "first_pc_variance_fraction": np.nan,
            "first_2_pc_variance_fraction": np.nan,
        }

    centered = (
        matrix
        -
        np.mean(
            matrix,
            axis=0,
            keepdims=True,
        )
    )

    singular_values = np.linalg.svd(
        centered,
        compute_uv=False,
        full_matrices=False,
    )

    variance_values = (
        singular_values
        ** 2
    )

    total = float(
        np.sum(
            variance_values
        )
    )

    if total <= NUMERICAL_EPS:
        return {
            "effective_rank": 0.0,
            "first_pc_variance_fraction": 1.0,
            "first_2_pc_variance_fraction": 1.0,
        }

    squared_total = float(
        np.sum(
            variance_values
            ** 2
        )
    )

    if squared_total <= NUMERICAL_EPS:
        effective_rank = 0.0
    else:
        effective_rank = float(
            (
                total ** 2
            )
            /
            squared_total
        )

    return {
        "effective_rank": effective_rank,
        "first_pc_variance_fraction": float(
            variance_values[0]
            /
            total
        ),
        "first_2_pc_variance_fraction": float(
            np.sum(
                variance_values[:2]
            )
            /
            total
        ),
    }


def top_similarity_tie_statistics(
    signatures,
    tolerance=1e-10,
):
    normalized = normalize_rows(
        signatures
    )

    cosine = (
        normalized
        @ normalized.T
    )

    np.fill_diagonal(
        cosine,
        -np.inf,
    )

    tie_counts = []

    for row_index in range(
        len(cosine)
    ):
        row = cosine[
            row_index
        ]

        finite = row[
            np.isfinite(row)
        ]

        if not len(finite):
            continue

        maximum = float(
            np.max(finite)
        )

        count = int(
            np.sum(
                np.abs(
                    finite
                    -
                    maximum
                )
                <= tolerance
            )
        )

        tie_counts.append(
            count
        )

    if not tie_counts:
        return {
            "mean_top1_tie_count": np.nan,
            "median_top1_tie_count": np.nan,
            "fraction_clients_with_top1_tie": np.nan,
        }

    tie_counts = np.asarray(
        tie_counts,
        dtype=float,
    )

    return {
        "mean_top1_tie_count": float(
            np.mean(
                tie_counts
            )
        ),
        "median_top1_tie_count": float(
            np.median(
                tie_counts
            )
        ),
        "fraction_clients_with_top1_tie": float(
            np.mean(
                tie_counts > 1
            )
        ),
    }


def prepare_relation_evaluations(
    positive_pairs,
    pair_relations,
    common_negative_pairs,
    min_relation_positives,
    dataset_name,
    seed,
):


    relation_positive_pairs = defaultdict(
        list
    )

    for pair in positive_pairs:
        relations = pair_relations.get(
            pair,
            set(),
        )

        for relation in relations:
            relation_positive_pairs[
                relation
            ].append(
                pair
            )

    evaluations = {}

    for relation, relation_positives in sorted(
        relation_positive_pairs.items()
    ):
        desired_negatives = min(
            len(relation_positives),
            len(common_negative_pairs),
        )

        relation_negatives = sample_items(
            common_negative_pairs,
            desired_negatives,
            deterministic_seed(
                dataset_name,
                seed,
                relation,
                "relation_negatives",
            ),
        )

        evaluations[
            relation
        ] = {
            "positive_pairs": (
                relation_positives
            ),
            "negative_pairs": (
                relation_negatives
            ),
            "positive_count": int(
                len(
                    relation_positives
                )
            ),
            "negative_count": int(
                len(
                    relation_negatives
                )
            ),
            "reliable": bool(
                len(relation_positives)
                >=
                int(
                    min_relation_positives
                )
            ),
        }

    return evaluations


def positive_pair_dataframe(
    positive_pairs,
    pair_relations,
):
    rows = []

    for first, second in positive_pairs:
        relations = sorted(
            pair_relations.get(
                (
                    first,
                    second,
                ),
                set(),
            )
        )

        rows.append({
            "source": first,
            "target": second,
            "relations": "|".join(
                relations
            ),
            "relation_count": int(
                len(relations)
            ),
        })

    return pd.DataFrame(
        rows
    )


def negative_pair_dataframe(
    negative_pairs,
):
    return pd.DataFrame(
        [
            {
                "source": first,
                "target": second,
            }
            for first, second
            in negative_pairs
        ]
    )


def run_representation_diagnostic_experiment(
    dataset_name="enron",
    output_folder=DEFAULT_OUTPUT_FOLDER,
    dimensions=DEFAULT_DIMENSIONS,
    normalizations=DEFAULT_NORMALIZATIONS,
    descriptor_families=DEFAULT_DESCRIPTOR_FAMILIES,
    diversity_clients=DEFAULT_DIVERSITY_CLIENTS,
    positive_pairs_requested=DEFAULT_POSITIVE_PAIRS,
    negative_ratio=DEFAULT_NEGATIVE_RATIO,
    d_max=DEFAULT_D_MAX,
    degree_weight=DEFAULT_DEGREE_WEIGHT,
    seed=DEFAULT_SEED,
    min_positives_for_ranking=(
        DEFAULT_MIN_POSITIVES_FOR_RANKING
    ),
    min_relation_positives=(
        DEFAULT_MIN_RELATION_POSITIVES
    ),
):


    start = time.perf_counter()

    dimensions = tuple(
        int(value)
        for value
        in dimensions
    )

    normalizations = tuple(
        str(value)
        for value
        in normalizations
    )

    descriptor_families = tuple(
        str(value)
        for value
        in descriptor_families
    )

    allowed_families = {
        "hks",
        "eigen",
        "moments",
        "hks_eigen",
        "hks_degree",
        "hks_eigen_degree",
    }

    for dimension in dimensions:
        if (
            dimension < 4
            or dimension > 12
            or dimension % 2 != 0
        ):
            raise ValueError(
                "Dimensions must be even "
                "and between 4 and 12."
            )

    for normalization in normalizations:
        if normalization not in {
            "block",
            "global",
        }:
            raise ValueError(
                f"Unknown normalization: "
                f"{normalization}"
            )

    for family in descriptor_families:
        if family not in allowed_families:
            raise ValueError(
                f"Unknown descriptor family: "
                f"{family}"
            )

    if negative_ratio <= 0:
        raise ValueError(
            "negative_ratio must be positive."
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
            f"Missing global graph:\n"
            f"  {graph_path}"
        )

    print()
    print(
        "=" * 60
    )
    print(
        "REPRESENTATION DIAGNOSTIC EXPERIMENT v2"
    )
    print(
        "=" * 60
    )
    print(
        f"Dataset: {dataset_name}"
    )

    graph = nx.read_gml(
        graph_path
    )


    clients = all_client_ids(
        graph
    )

    if len(clients) < 2:
        raise RuntimeError(
            "Fewer than two client nodes found."
        )

    print(
        f"All graph clients: "
        f"{len(clients):,}"
    )


    diversity_population = sample_items(
        clients,
        diversity_clients,
        deterministic_seed(
            dataset_name,
            seed,
            "diversity_clients",
        ),
    )

    print(
        f"Diversity sample: "
        f"{len(diversity_population):,}"
    )


    positive_information = (
        collect_client_client_pairs(
            graph
        )
    )

    pair_relations = (
        positive_information[
            "pair_relations"
        ]
    )

    full_positive_pairs = sorted(
        pair_relations.keys()
    )

    if not full_positive_pairs:
        raise RuntimeError(
            "No direct client-client positive pairs "
            "were found in the graph."
        )

    print(
        "All unique direct client-client pairs: "
        f"{len(full_positive_pairs):,}"
    )


    positive_pairs = sample_positive_pairs(
        pair_relations=pair_relations,
        requested=(
            positive_pairs_requested
        ),
        seed=deterministic_seed(
            dataset_name,
            seed,
            "positive_pairs",
        ),
    )

    print(
        "Sampled positive pairs: "
        f"{len(positive_pairs):,}"
    )


    number_negatives = int(
        round(
            len(positive_pairs)
            *
            float(negative_ratio)
        )
    )

    negative_pairs = sample_negative_pairs(
        client_ids=clients,
        full_positive_pairs=(
            full_positive_pairs
        ),
        requested=number_negatives,
        seed=deterministic_seed(
            dataset_name,
            seed,
            "negative_pairs",
        ),
    )

    print(
        "Common negative pairs: "
        f"{len(negative_pairs):,}"
    )


    positive_endpoints = pair_endpoints(
        positive_pairs
    )

    negative_endpoints = pair_endpoints(
        negative_pairs
    )

    print(
        "Distinct positive endpoints: "
        f"{len(positive_endpoints):,}"
    )

    print(
        "Distinct negative endpoints: "
        f"{len(negative_endpoints):,}"
    )


    required_clients = (
        set(
            diversity_population
        )
        |
        positive_endpoints
        |
        negative_endpoints
    )

    print(
        "Total clients requiring descriptors: "
        f"{len(required_clients):,}"
    )


    generated = generate_descriptors(
        graph=graph,
        required_clients=(
            required_clients
        ),
        dimensions=dimensions,
        normalizations=normalizations,
        families=descriptor_families,
        d_max=d_max,
        degree_weight=degree_weight,
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
        "Valid descriptor clients: "
        f"{len(descriptor_client_ids):,}"
    )

    print(
        "Descriptor failures: "
        f"{len(generated['failures']):,}"
    )


    diversity_population = [
        client
        for client
        in diversity_population
        if client in valid_client_set
    ]

    positive_pairs = (
        filter_pairs_to_valid_clients(
            positive_pairs,
            valid_client_set,
        )
    )

    negative_pairs = (
        filter_pairs_to_valid_clients(
            negative_pairs,
            valid_client_set,
        )
    )

    if not positive_pairs:
        raise RuntimeError(
            "No valid positive pairs remain."
        )

    if not negative_pairs:
        raise RuntimeError(
            "No valid negative pairs remain."
        )


    descriptor_index = {
        str(client): position
        for position, client
        in enumerate(
            descriptor_client_ids
        )
    }

    diversity_indices = np.asarray(
        [
            descriptor_index[
                client
            ]
            for client
            in diversity_population
        ],
        dtype=int,
    )


    relation_evaluations = (
        prepare_relation_evaluations(
            positive_pairs=positive_pairs,
            pair_relations=pair_relations,
            common_negative_pairs=(
                negative_pairs
            ),
            min_relation_positives=(
                min_relation_positives
            ),
            dataset_name=dataset_name,
            seed=seed,
        )
    )


    summary_rows = []

    relation_rows = []

    signatures_output = {
        "client_ids": (
            descriptor_client_ids
        ),
        "diversity_client_ids": (
            np.asarray(
                diversity_population,
                dtype=str,
            )
        ),
    }

    total_descriptors = len(
        descriptor_matrices
    )

    for descriptor_number, (
        key,
        full_matrix,
    ) in enumerate(
        descriptor_matrices.items(),
        start=1,
    ):
        (
            dimension,
            normalization,
            family,
        ) = key

        descriptor_name = (
            f"{family}"
            f"_d{dimension}"
            f"_{normalization}"
        )

        print(
            "[descriptor diagnostic] "
            f"{descriptor_number}/"
            f"{total_descriptors} "
            f"{descriptor_name}"
        )

        diversity_matrix = full_matrix[
            diversity_indices
        ]

        diversity_stats = (
            descriptor_diversity_statistics(
                diversity_matrix
            )
        )

        rank_stats = (
            effective_rank_statistics(
                diversity_matrix
            )
        )

        tie_stats = (
            top_similarity_tie_statistics(
                diversity_matrix
            )
        )

        overall_classification = (
            classification_metrics(
                signatures=full_matrix,
                client_ids=(
                    descriptor_client_ids
                ),
                positive_pairs=(
                    positive_pairs
                ),
                negative_pairs=(
                    negative_pairs
                ),
            )
        )

        classification_reliable = bool(
            len(positive_pairs)
            >=
            int(
                min_positives_for_ranking
            )
        )

        summary_rows.append({
            "descriptor": (
                descriptor_name
            ),
            "family": family,
            "base_dimension": int(
                dimension
            ),
            "actual_dimension": int(
                full_matrix.shape[1]
            ),
            "normalization": (
                normalization
            ),
            "diversity_clients": int(
                len(
                    diversity_population
                )
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
            "positive_endpoints": int(
                len(
                    pair_endpoints(
                        positive_pairs
                    )
                )
            ),
            "classification_reliable": (
                classification_reliable
            ),
            "D_max": int(
                d_max
            ),
            "degree_weight": float(
                degree_weight
            ),
            **diversity_stats,
            **rank_stats,
            **tie_stats,
            **overall_classification,
        })


        for (
            relation,
            relation_evaluation,
        ) in relation_evaluations.items():
            relation_metrics = (
                classification_metrics(
                    signatures=full_matrix,
                    client_ids=(
                        descriptor_client_ids
                    ),
                    positive_pairs=(
                        relation_evaluation[
                            "positive_pairs"
                        ]
                    ),
                    negative_pairs=(
                        relation_evaluation[
                            "negative_pairs"
                        ]
                    ),
                )
            )

            relation_rows.append({
                "descriptor": (
                    descriptor_name
                ),
                "family": family,
                "base_dimension": int(
                    dimension
                ),
                "actual_dimension": int(
                    full_matrix.shape[1]
                ),
                "normalization": (
                    normalization
                ),
                "relation": relation,
                "positive_pairs": int(
                    relation_evaluation[
                        "positive_count"
                    ]
                ),
                "negative_pairs": int(
                    relation_evaluation[
                        "negative_count"
                    ]
                ),
                "relation_result_reliable": bool(
                    relation_evaluation[
                        "reliable"
                    ]
                ),
                **relation_metrics,
            })

        signatures_output[
            descriptor_name
        ] = full_matrix

    summary_dataframe = pd.DataFrame(
        summary_rows
    )

    relation_dataframe = pd.DataFrame(
        relation_rows
    )


    classification_reliable = bool(
        len(positive_pairs)
        >=
        int(
            min_positives_for_ranking
        )
    )

    if classification_reliable:
        ranking_dataframe = (
            summary_dataframe
            .sort_values(
                by=[
                    "clean_auc",
                    "clean_ap",
                    "similarity_mean_gap",
                    "unique_signature_fraction",
                    "mean_pairwise_cosine",
                ],
                ascending=[
                    False,
                    False,
                    False,
                    False,
                    True,
                ],
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

        ranking_basis = (
            "classification-first: "
            "AUC, AP, similarity separation, "
            "then diversity"
        )

    else:
        ranking_dataframe = (
            summary_dataframe
            .sort_values(
                by=[
                    "unique_signature_fraction",
                    "mean_pairwise_cosine",
                    "effective_rank",
                ],
                ascending=[
                    False,
                    True,
                    False,
                ],
                na_position="last",
            )
            .reset_index(
                drop=True
            )
        )

        ranking_basis = (
            "diversity-first because too few "
            "positive pairs were available"
        )

    ranking_dataframe.insert(
        0,
        "screening_rank",
        np.arange(
            1,
            len(
                ranking_dataframe
            )
            + 1,
        ),
    )


    experiment_folder = os.path.join(
        output_folder,
        dataset_name,
        "experiments",
        "representation_diagnostic",
    )

    os.makedirs(
        experiment_folder,
        exist_ok=True,
    )

    summary_path = os.path.join(
        experiment_folder,
        "representation_diagnostic_summary.csv",
    )

    ranking_path = os.path.join(
        experiment_folder,
        "representation_diagnostic_ranking.csv",
    )

    relation_path = os.path.join(
        experiment_folder,
        "representation_diagnostic_by_relation.csv",
    )

    diversity_path = os.path.join(
        experiment_folder,
        "representation_diagnostic_diversity_clients.csv",
    )

    positive_path = os.path.join(
        experiment_folder,
        "representation_diagnostic_positive_pairs.csv",
    )

    negative_path = os.path.join(
        experiment_folder,
        "representation_diagnostic_negative_pairs.csv",
    )

    signatures_path = os.path.join(
        experiment_folder,
        "representation_diagnostic_signatures.npz",
    )

    metadata_path = os.path.join(
        experiment_folder,
        "representation_diagnostic_metadata.json",
    )


    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    ranking_dataframe.to_csv(
        ranking_path,
        index=False,
    )

    relation_dataframe.to_csv(
        relation_path,
        index=False,
    )

    pd.DataFrame({
        "client_id": (
            diversity_population
        )
    }).to_csv(
        diversity_path,
        index=False,
    )

    positive_pair_dataframe(
        positive_pairs,
        pair_relations,
    ).to_csv(
        positive_path,
        index=False,
    )

    negative_pair_dataframe(
        negative_pairs
    ).to_csv(
        negative_path,
        index=False,
    )

    np.savez_compressed(
        signatures_path,
        **signatures_output,
    )


    sampled_relation_counts = {}

    for pair in positive_pairs:
        for relation in pair_relations.get(
            pair,
            set(),
        ):
            sampled_relation_counts[
                relation
            ] = (
                sampled_relation_counts.get(
                    relation,
                    0,
                )
                + 1
            )

    runtime = float(
        time.perf_counter()
        -
        start
    )

    metadata = {
        "schema_version": (
            REPRESENTATION_DIAGNOSTIC_SCHEMA_VERSION
        ),
        "dataset": dataset_name,
        "global_graph": graph_path,

        "all_graph_clients": int(
            len(clients)
        ),

        "diversity_clients_requested": int(
            diversity_clients
        ),

        "diversity_clients_used": int(
            len(
                diversity_population
            )
        ),

        "all_unique_client_client_pairs": int(
            len(
                full_positive_pairs
            )
        ),

        "positive_pairs_requested": int(
            positive_pairs_requested
        ),

        "positive_pairs_used": int(
            len(
                positive_pairs
            )
        ),

        "negative_pairs_used": int(
            len(
                negative_pairs
            )
        ),

        "negative_ratio": float(
            negative_ratio
        ),

        "distinct_positive_endpoints": int(
            len(
                pair_endpoints(
                    positive_pairs
                )
            )
        ),

        "required_descriptor_clients": int(
            len(
                required_clients
            )
        ),

        "valid_descriptor_clients": int(
            len(
                descriptor_client_ids
            )
        ),

        "descriptor_failures": (
            generated[
                "failures"
            ]
        ),

        "dimensions": [
            int(value)
            for value
            in dimensions
        ],

        "normalizations": list(
            normalizations
        ),

        "descriptor_families": list(
            descriptor_families
        ),

        "D_max": int(
            d_max
        ),

        "degree_weight": float(
            degree_weight
        ),

        "seed": int(
            seed
        ),

        "min_positives_for_ranking": int(
            min_positives_for_ranking
        ),

        "classification_reliable": bool(
            classification_reliable
        ),

        "ranking_basis": (
            ranking_basis
        ),

        "full_relation_edge_counts": (
            positive_information[
                "relation_edge_counts"
            ]
        ),

        "full_relation_pair_counts": (
            positive_information[
                "relation_pair_counts"
            ]
        ),

        "sampled_relation_positive_counts": (
            sampled_relation_counts
        ),

        "min_relation_positives": int(
            min_relation_positives
        ),

        "sampling_design": {
            "diversity": (
                "Uniform deterministic sample from all "
                "role=client nodes in the global graph."
            ),
            "positive_pairs": (
                "Uniform deterministic sample from all unique "
                "direct client-client pairs in the global graph."
            ),
            "negative_pairs": (
                "Common deterministic client-client non-edge "
                "sample. Pairs present anywhere in the full "
                "positive universe are excluded."
            ),
        },

        "scope_warning": (
            "AUC and AP are CLEAN representation-screening "
            "statistics. Positive edges are visible to the local "
            "descriptor construction, so these values MUST NOT be "
            "reported as final held-out link-prediction results. "
            "Final downstream evaluation must use the dedicated "
            "train-before-ego-generation link prediction experiment."
        ),

        "descriptor_definitions": {
            "hks": (
                "Root HKS on client-client and "
                "client-content projections."
            ),
            "eigen": (
                "Smallest non-trivial normalized-Laplacian "
                "eigenvalues divided by 2."
            ),
            "moments": (
                "Root spectral moments "
                "sum(lambda^r v(root)^2)/2^r, r>=2."
            ),
            "hks_eigen": (
                "Normalized concatenation of HKS and "
                "eigenvalue descriptors."
            ),
            "hks_degree": (
                "Normalized concatenation of HKS and "
                "bounded typed degrees."
            ),
            "hks_eigen_degree": (
                "Normalized concatenation of HKS, "
                "eigenvalues and bounded typed degrees."
            ),
        },

        "outputs": {
            "summary": os.path.basename(
                summary_path
            ),
            "ranking": os.path.basename(
                ranking_path
            ),
            "by_relation": os.path.basename(
                relation_path
            ),
            "diversity_clients": os.path.basename(
                diversity_path
            ),
            "positive_pairs": os.path.basename(
                positive_path
            ),
            "negative_pairs": os.path.basename(
                negative_path
            ),
            "signatures": os.path.basename(
                signatures_path
            ),
        },

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
        "=" * 60
    )
    print(
        "REPRESENTATION DIAGNOSTIC COMPLETE"
    )
    print(
        "=" * 60
    )

    print(
        f"All graph clients: "
        f"{len(clients):,}"
    )

    print(
        f"Diversity clients: "
        f"{len(diversity_population):,}"
    )

    print(
        f"All CC positive pairs: "
        f"{len(full_positive_pairs):,}"
    )

    print(
        f"Sampled positive pairs: "
        f"{len(positive_pairs):,}"
    )

    print(
        f"Common negative pairs: "
        f"{len(negative_pairs):,}"
    )

    print(
        "Distinct positive endpoints: "
        f"{len(pair_endpoints(positive_pairs)):,}"
    )

    print(
        f"Classification reliable for screening: "
        f"{classification_reliable}"
    )

    print(
        f"Ranking basis: "
        f"{ranking_basis}"
    )

    print()
    print(
        "Sampled client-client relations:"
    )

    if sampled_relation_counts:
        for (
            relation,
            count,
        ) in sorted(
            sampled_relation_counts.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        ):
            print(
                f"  {relation:25s} "
                f"{count:,}"
            )
    else:
        print(
            "  none"
        )

    print()
    print(
        "Top descriptor candidates:"
    )

    console_columns = [
        "screening_rank",
        "descriptor",
        "actual_dimension",
        "unique_signature_fraction",
        "mean_pairwise_cosine",
        "effective_rank",
        "clean_auc",
        "clean_ap",
        "similarity_mean_gap",
    ]

    print(
        ranking_dataframe[
            console_columns
        ]
        .head(20)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Current HKS baselines:"
    )

    hks_rows = (
        ranking_dataframe[
            ranking_dataframe[
                "family"
            ]
            ==
            "hks"
        ]
        .copy()
    )

    hks_columns = [
        "descriptor",
        "unique_signature_fraction",
        "mean_pairwise_cosine",
        "effective_rank",
        "clean_auc",
        "clean_ap",
        "positive_similarity_mean",
        "negative_similarity_mean",
        "similarity_mean_gap",
    ]

    print(
        hks_rows[
            hks_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "  Diversity and structural screening now use "
        "separate populations."
    )

    print(
        "  Positive edges are sampled BEFORE choosing "
        "their endpoints."
    )

    print(
        "  Every descriptor uses the SAME positive and "
        "negative pairs."
    )

    print(
        "  These AUC/AP values are still CLEAN screening "
        "statistics, not final held-out results."
    )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  {summary_path}"
    )
    print(
        f"  {ranking_path}"
    )
    print(
        f"  {relation_path}"
    )
    print(
        f"  {diversity_path}"
    )
    print(
        f"  {positive_path}"
    )
    print(
        f"  {negative_path}"
    )
    print(
        f"  {signatures_path}"
    )
    print(
        f"  {metadata_path}"
    )

    print()
    print(
        f"Runtime: {runtime:.3f} s"
    )

    print(
        "=" * 60
    )
    print()

    return {
        "summary": summary_dataframe,
        "ranking": ranking_dataframe,
        "by_relation": relation_dataframe,
        "metadata": metadata,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Screen invariant local graph descriptors "
            "using separate diversity and structural "
            "evaluation populations."
        )
    )

    parser.add_argument(
        "dataset",
        type=str,
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
    )

    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=list(
            DEFAULT_DIMENSIONS
        ),
    )

    parser.add_argument(
        "--normalizations",
        nargs="+",
        choices=[
            "block",
            "global",
        ],
        default=list(
            DEFAULT_NORMALIZATIONS
        ),
    )

    parser.add_argument(
        "--families",
        nargs="+",
        choices=[
            "hks",
            "eigen",
            "moments",
            "hks_eigen",
            "hks_degree",
            "hks_eigen_degree",
        ],
        default=list(
            DEFAULT_DESCRIPTOR_FAMILIES
        ),
    )

    parser.add_argument(
        "--diversity-clients",
        type=int,
        default=(
            DEFAULT_DIVERSITY_CLIENTS
        ),
    )

    parser.add_argument(
        "--positive-pairs",
        type=int,
        default=(
            DEFAULT_POSITIVE_PAIRS
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
        "--Dmax",
        type=int,
        default=DEFAULT_D_MAX,
    )

    parser.add_argument(
        "--degree-weight",
        type=float,
        default=(
            DEFAULT_DEGREE_WEIGHT
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--min-positives-for-ranking",
        type=int,
        default=(
            DEFAULT_MIN_POSITIVES_FOR_RANKING
        ),
    )

    parser.add_argument(
        "--min-relation-positives",
        type=int,
        default=(
            DEFAULT_MIN_RELATION_POSITIVES
        ),
    )

    args = parser.parse_args()

    run_representation_diagnostic_experiment(
        dataset_name=args.dataset,
        output_folder=args.output_folder,
        dimensions=args.dimensions,
        normalizations=args.normalizations,
        descriptor_families=args.families,
        diversity_clients=(
            args.diversity_clients
        ),
        positive_pairs_requested=(
            args.positive_pairs
        ),
        negative_ratio=(
            args.negative_ratio
        ),
        d_max=args.Dmax,
        degree_weight=(
            args.degree_weight
        ),
        seed=args.seed,
        min_positives_for_ranking=(
            args.min_positives_for_ranking
        ),
        min_relation_positives=(
            args.min_relation_positives
        ),
    )
