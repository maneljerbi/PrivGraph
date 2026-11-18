import argparse
import hashlib
import json
import math
import os
import time

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
    t as student_t,
)

from link_prediction_experiment import (
    generate_selected_descriptors,
)


SCHEMA_VERSION = "eigen-privacy-utility-1.0"

DEFAULT_OUTPUT_FOLDER = "output_local_test"

DEFAULT_DATASETS = (
    "enron",
    "dblp",
    "yelp",
)

DEFAULT_REPRESENTATIONS = (
    "hks_d12_global",
    "eigen_d8_block",
    "eigen_d12_global",
)

DEFAULT_EPSILONS = (
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
    20.0,
)

DEFAULT_SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

DEFAULT_MAX_CLIENTS = 1000

DEFAULT_PAIR_SAMPLES = 50000

DEFAULT_ANCHORS = 200

DEFAULT_C = 1.0

DEFAULT_RHO_SIGNATURE = 0.5

DEFAULT_SENSITIVITY_MODE = "nonnegative_report"

DEFAULT_D_MAX = 50

DEFAULT_DEGREE_WEIGHT = 1.0

NEGATIVE_TOLERANCE = 1e-10

ROUND_DECIMALS = 12

OVERLAP_K = (
    10,
    20,
    50,
)


GRAPH_FILENAMES = {
    "enron": "enron_global_graph.gml.gz",
    "dblp": "dblp_global_graph.gml.gz",
    "yelp": "yelp_global_graph.gml.gz",
}


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


def finite_mean(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(
        values
    ) == 0:

        return np.nan

    return float(
        np.mean(
            values
        )
    )


def finite_median(values):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(
        values
    ) == 0:

        return np.nan

    return float(
        np.median(
            values
        )
    )


def mean_std_ci95(
    values,
):

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
            mean
            -
            margin
        ),

        "ci95_high": float(
            mean
            +
            margin
        ),
    }


def get_client_ids(
    graph,
):

    client_ids = []

    for node_id, attributes in graph.nodes(
        data=True
    ):

        if str(
            attributes.get(
                "role",
                "",
            )
        ) == "client":

            client_ids.append(
                str(
                    node_id
                )
            )

    if not client_ids:

        raise RuntimeError(
            "No nodes with role='client' were found."
        )

    return sorted(
        client_ids
    )


def sample_clients(
    client_ids,
    maximum,
    dataset,
):

    client_ids = np.asarray(
        sorted(
            client_ids
        ),
        dtype=str,
    )

    maximum = int(
        maximum
    )

    if len(
        client_ids
    ) <= maximum:

        return list(
            client_ids
        )

    rng = np.random.default_rng(
        deterministic_seed(
            dataset,
            "privacy_utility_client_sample",
        )
    )

    indices = rng.choice(
        len(
            client_ids
        ),
        size=maximum,
        replace=False,
    )

    selected = sorted(
        str(
            client_ids[
                int(
                    index
                )
            ]
        )
        for index
        in indices
    )

    return selected


def prepare_private_query(
    raw_matrix,
    C,
    representation,
):

    raw_matrix = np.asarray(
        raw_matrix,
        dtype=float,
    )

    if raw_matrix.ndim != 2:

        raise ValueError(
            f"{representation}: descriptor matrix "
            "must be two-dimensional."
        )

    if not np.all(
        np.isfinite(
            raw_matrix
        )
    ):

        raise RuntimeError(
            f"{representation}: descriptor matrix "
            "contains NaN or infinite values."
        )

    total_entries = int(
        raw_matrix.size
    )

    negative_mask = (
        raw_matrix
        <
        0.0
    )

    substantive_negative_mask = (
        raw_matrix
        <
        -NEGATIVE_TOLERANCE
    )

    negative_entries = int(
        np.sum(
            negative_mask
        )
    )

    substantive_negative_entries = int(
        np.sum(
            substantive_negative_mask
        )
    )

    negative_fraction = float(
        negative_entries
        /
        max(
            total_entries,
            1,
        )
    )

    if substantive_negative_entries > 0:

        minimum = float(
            np.min(
                raw_matrix
            )
        )

        raise RuntimeError(
            "\n"
            f"{representation} violates the nonnegative-query "
            "assumption required for the sqrt(2)C sensitivity "
            "argument.\n"
            f"  substantive negative entries="
            f"{substantive_negative_entries:,}\n"
            f"  minimum value={minimum:.12e}\n"
            "\n"
            "Do NOT take absolute values automatically.\n"
            "Either inspect the descriptor definition or run with\n"
            "  --sensitivity-mode generic_report\n"
            "which uses the universally valid 2C clipped-vector "
            "diameter."
        )


    nonnegative = np.maximum(
        raw_matrix,
        0.0,
    )

    raw_norms = np.linalg.norm(
        nonnegative,
        axis=1,
    )

    clipped = np.array(
        nonnegative,
        dtype=float,
        copy=True,
    )

    clip_mask = (
        raw_norms
        >
        float(
            C
        )
    )

    if np.any(
        clip_mask
    ):

        scale = (
            float(
                C
            )
            /
            raw_norms[
                clip_mask
            ]
        )

        clipped[
            clip_mask
        ] *= scale[
            :,
            None,
        ]

    clipped_norms = np.linalg.norm(
        clipped,
        axis=1,
    )

    zero_rows = (
        clipped_norms
        <=
        1e-15
    )

    rounded = np.round(
        clipped,
        decimals=ROUND_DECIMALS,
    )

    if len(
        rounded
    ):

        unique_rows = int(
            np.unique(
                rounded,
                axis=0,
            ).shape[
                0
            ]
        )

    else:

        unique_rows = 0

    component_variance = np.var(
        clipped,
        axis=0,
    )

    return {
        "matrix": clipped,

        "dimension": int(
            clipped.shape[
                1
            ]
        ),

        "raw_norm_mean": float(
            np.mean(
                raw_norms
            )
        ),

        "raw_norm_max": float(
            np.max(
                raw_norms
            )
        ),

        "clipped_norm_mean": float(
            np.mean(
                clipped_norms
            )
        ),

        "clipped_norm_max": float(
            np.max(
                clipped_norms
            )
        ),

        "clip_fraction": float(
            np.mean(
                clip_mask
            )
        ),

        "zero_row_fraction": float(
            np.mean(
                zero_rows
            )
        ),

        "negative_entry_fraction": (
            negative_fraction
        ),

        "substantive_negative_entries": (
            substantive_negative_entries
        ),

        "unique_rows_rounded": (
            unique_rows
        ),

        "mean_component_variance": float(
            np.mean(
                component_variance
            )
        ),

        "max_component_variance": float(
            np.max(
                component_variance
            )
        ),
    }


def analytical_sensitivity(
    C,
    sensitivity_mode,
):

    C = float(
        C
    )

    if sensitivity_mode == "nonnegative_report":

        return float(
            math.sqrt(
                2.0
            )
            *
            C
        )

    if sensitivity_mode == "generic_report":

        return float(
            2.0
            *
            C
        )

    raise ValueError(
        f"Unknown sensitivity mode: "
        f"{sensitivity_mode}"
    )


def sample_unit_direction(
    dimension,
    rng,
):

    while True:

        vector = rng.normal(
            loc=0.0,
            scale=1.0,
            size=int(
                dimension
            ),
        )

        norm = float(
            np.linalg.norm(
                vector
            )
        )

        if norm > 0.0:

            return (
                vector
                /
                norm
            )


def sample_l2_knorm_noise(
    dimension,
    epsilon,
    sensitivity,
    rng,
):

    dimension = int(
        dimension
    )

    epsilon = float(
        epsilon
    )

    sensitivity = float(
        sensitivity
    )

    if dimension <= 0:

        raise ValueError(
            "K-norm dimension must be positive."
        )

    if epsilon <= 0.0:

        raise ValueError(
            "epsilon must be strictly positive."
        )

    if sensitivity <= 0.0:

        raise ValueError(
            "sensitivity must be strictly positive."
        )

    direction = sample_unit_direction(
        dimension,
        rng,
    )

    radius = float(
        rng.gamma(
            shape=dimension,
            scale=(
                sensitivity
                /
                epsilon
            ),
        )
    )

    return (
        radius
        *
        direction
    )


def privatize_matrix(
    clean_matrix,
    epsilon_signature,
    sensitivity,
    dataset,
    representation,
    mechanism_seed,
):

    clean_matrix = np.asarray(
        clean_matrix,
        dtype=float,
    )

    n_clients, dimension = (
        clean_matrix.shape
    )

    private_matrix = np.empty_like(
        clean_matrix,
        dtype=float,
    )

    noise_norms = np.zeros(
        n_clients,
        dtype=float,
    )

    for row_index in range(
        n_clients
    ):

        rng = np.random.default_rng(
            deterministic_seed(
                dataset,
                representation,
                mechanism_seed,
                row_index,
                "knorm_report",
            )
        )

        noise = sample_l2_knorm_noise(
            dimension=dimension,

            epsilon=epsilon_signature,

            sensitivity=sensitivity,

            rng=rng,
        )

        private_matrix[
            row_index
        ] = (
            clean_matrix[
                row_index
            ]
            +
            noise
        )

        noise_norms[
            row_index
        ] = float(
            np.linalg.norm(
                noise
            )
        )

    return {
        "private_matrix": (
            private_matrix
        ),

        "noise_norms": (
            noise_norms
        ),
    }


def rowwise_cosine(
    first_matrix,
    second_matrix,
):

    first_matrix = np.asarray(
        first_matrix,
        dtype=float,
    )

    second_matrix = np.asarray(
        second_matrix,
        dtype=float,
    )

    numerator = np.sum(
        first_matrix
        *
        second_matrix,
        axis=1,
    )

    first_norm = np.linalg.norm(
        first_matrix,
        axis=1,
    )

    second_norm = np.linalg.norm(
        second_matrix,
        axis=1,
    )

    denominator = (
        first_norm
        *
        second_norm
    )

    result = np.zeros(
        len(
            first_matrix
        ),
        dtype=float,
    )

    valid = (
        denominator
        >
        0.0
    )

    result[
        valid
    ] = (
        numerator[
            valid
        ]
        /
        denominator[
            valid
        ]
    )

    result = np.clip(
        result,
        -1.0,
        1.0,
    )

    return result


def vector_preservation_metrics(
    clean_matrix,
    private_matrix,
):

    clean_matrix = np.asarray(
        clean_matrix,
        dtype=float,
    )

    private_matrix = np.asarray(
        private_matrix,
        dtype=float,
    )

    cosine = rowwise_cosine(
        clean_matrix,
        private_matrix,
    )

    l2_error = np.linalg.norm(
        private_matrix
        -
        clean_matrix,
        axis=1,
    )

    clean_norm = np.linalg.norm(
        clean_matrix,
        axis=1,
    )

    relative_error = np.zeros_like(
        l2_error
    )

    nonzero = (
        clean_norm
        >
        1e-15
    )

    relative_error[
        nonzero
    ] = (
        l2_error[
            nonzero
        ]
        /
        clean_norm[
            nonzero
        ]
    )

    relative_error[
        ~nonzero
    ] = np.nan

    return {
        "clean_private_cosine_mean": (
            finite_mean(
                cosine
            )
        ),

        "clean_private_cosine_median": (
            finite_median(
                cosine
            )
        ),

        "clean_private_l2_mean": (
            finite_mean(
                l2_error
            )
        ),

        "clean_private_l2_median": (
            finite_median(
                l2_error
            )
        ),

        "clean_private_relative_l2_mean": (
            finite_mean(
                relative_error
            )
        ),

        "clean_private_relative_l2_median": (
            finite_median(
                relative_error
            )
        ),
    }


def normalize_rows(
    matrix,
):

    matrix = np.asarray(
        matrix,
        dtype=float,
    )

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    output = np.zeros_like(
        matrix,
        dtype=float,
    )

    valid = (
        norms[
            :,
            0
        ]
        >
        1e-15
    )

    output[
        valid
    ] = (
        matrix[
            valid
        ]
        /
        norms[
            valid
        ]
    )

    return output


def sample_pair_indices(
    n_clients,
    requested,
    dataset,
    representation,
):

    n_clients = int(
        n_clients
    )

    requested = int(
        requested
    )

    possible = (
        n_clients
        *
        (
            n_clients
            -
            1
        )
        //
        2
    )

    if possible <= 0:

        return np.empty(
            (
                0,
                2,
            ),
            dtype=int,
        )

    requested = min(
        requested,
        possible,
    )


    if possible <= 200000:

        pairs = np.asarray(
            list(
                combinations(
                    range(
                        n_clients
                    ),
                    2,
                )
            ),
            dtype=int,
        )

        if len(
            pairs
        ) <= requested:

            return pairs

        rng = np.random.default_rng(
            deterministic_seed(
                dataset,
                representation,
                "pair_sample",
            )
        )

        indices = rng.choice(
            len(
                pairs
            ),
            size=requested,
            replace=False,
        )

        return pairs[
            indices
        ]


    rng = np.random.default_rng(
        deterministic_seed(
            dataset,
            representation,
            "pair_sample",
        )
    )

    selected = set()

    while len(
        selected
    ) < requested:

        first = int(
            rng.integers(
                0,
                n_clients,
            )
        )

        second = int(
            rng.integers(
                0,
                n_clients,
            )
        )

        if first == second:

            continue

        if first > second:

            first, second = (
                second,
                first,
            )

        selected.add(
            (
                first,
                second,
            )
        )

    return np.asarray(
        sorted(
            selected
        ),
        dtype=int,
    )


def safe_correlation(
    first,
    second,
    method,
):

    first = np.asarray(
        first,
        dtype=float,
    )

    second = np.asarray(
        second,
        dtype=float,
    )

    valid = (
        np.isfinite(
            first
        )
        &
        np.isfinite(
            second
        )
    )

    first = first[
        valid
    ]

    second = second[
        valid
    ]

    if len(
        first
    ) < 3:

        return np.nan

    if (
        np.std(
            first
        )
        <=
        1e-15
        or
        np.std(
            second
        )
        <=
        1e-15
    ):

        return np.nan

    if method == "spearman":

        value = spearmanr(
            first,
            second,
        ).statistic

    elif method == "pearson":

        value = pearsonr(
            first,
            second,
        ).statistic

    else:

        raise ValueError(
            method
        )

    if not np.isfinite(
        value
    ):

        return np.nan

    return float(
        value
    )


def pair_geometry_metrics(
    clean_matrix,
    private_matrix,
    pair_indices,
):

    clean_normalized = normalize_rows(
        clean_matrix
    )

    private_normalized = normalize_rows(
        private_matrix
    )

    if len(
        pair_indices
    ) == 0:

        return {
            "pair_samples": 0,

            "pair_cosine_spearman": np.nan,

            "pair_cosine_pearson": np.nan,

            "pair_cosine_mae": np.nan,
        }

    first = pair_indices[
        :,
        0
    ]

    second = pair_indices[
        :,
        1
    ]

    clean_scores = np.sum(
        clean_normalized[
            first
        ]
        *
        clean_normalized[
            second
        ],
        axis=1,
    )

    private_scores = np.sum(
        private_normalized[
            first
        ]
        *
        private_normalized[
            second
        ],
        axis=1,
    )

    return {
        "pair_samples": int(
            len(
                pair_indices
            )
        ),

        "pair_cosine_spearman": safe_correlation(
            clean_scores,
            private_scores,
            "spearman",
        ),

        "pair_cosine_pearson": safe_correlation(
            clean_scores,
            private_scores,
            "pearson",
        ),

        "pair_cosine_mae": float(
            np.mean(
                np.abs(
                    clean_scores
                    -
                    private_scores
                )
            )
        ),
    }


def select_anchor_indices(
    n_clients,
    requested,
    dataset,
    representation,
):

    requested = min(
        int(
            requested
        ),
        int(
            n_clients
        ),
    )

    if requested <= 0:

        return np.asarray(
            [],
            dtype=int,
        )

    rng = np.random.default_rng(
        deterministic_seed(
            dataset,
            representation,
            "retrieval_anchors",
        )
    )

    return np.sort(
        rng.choice(
            int(
                n_clients
            ),
            size=requested,
            replace=False,
        )
    )


def overlap_fraction(
    first_indices,
    second_indices,
):

    first_set = set(
        int(
            value
        )
        for value
        in first_indices
    )

    second_set = set(
        int(
            value
        )
        for value
        in second_indices
    )

    if not first_set:

        return np.nan

    return float(
        len(
            first_set
            &
            second_set
        )
        /
        len(
            first_set
        )
    )


def retrieval_preservation_metrics(
    clean_matrix,
    private_matrix,
    anchor_indices,
):

    clean_normalized = normalize_rows(
        clean_matrix
    )

    private_normalized = normalize_rows(
        private_matrix
    )

    n_clients = clean_matrix.shape[
        0
    ]

    metrics = {}

    similarity_values = {
        k: []
        for k
        in OVERLAP_K
    }

    farthest_values = {
        k: []
        for k
        in OVERLAP_K
    }

    for anchor in anchor_indices:

        clean_scores = (
            clean_normalized
            @
            clean_normalized[
                anchor
            ]
        )

        private_scores = (
            private_normalized
            @
            private_normalized[
                anchor
            ]
        )

        clean_scores[
            anchor
        ] = -np.inf

        private_scores[
            anchor
        ] = -np.inf

        for k in OVERLAP_K:

            effective_k = min(
                int(
                    k
                ),
                n_clients - 1,
            )

            if effective_k <= 0:

                continue

            clean_top = np.argpartition(
                clean_scores,
                -effective_k,
            )[
                -effective_k:
            ]

            private_top = np.argpartition(
                private_scores,
                -effective_k,
            )[
                -effective_k:
            ]

            similarity_values[
                k
            ].append(
                overlap_fraction(
                    clean_top,
                    private_top,
                )
            )


        clean_scores[
            anchor
        ] = np.inf

        private_scores[
            anchor
        ] = np.inf

        for k in OVERLAP_K:

            effective_k = min(
                int(
                    k
                ),
                n_clients - 1,
            )

            if effective_k <= 0:

                continue

            clean_bottom = np.argpartition(
                clean_scores,
                effective_k - 1,
            )[
                :effective_k
            ]

            private_bottom = np.argpartition(
                private_scores,
                effective_k - 1,
            )[
                :effective_k
            ]

            farthest_values[
                k
            ].append(
                overlap_fraction(
                    clean_bottom,
                    private_bottom,
                )
            )

    for k in OVERLAP_K:

        metrics[
            f"similarity_overlap_at_{k}"
        ] = finite_mean(
            similarity_values[
                k
            ]
        )

        metrics[
            f"farthest_overlap_at_{k}"
        ] = finite_mean(
            farthest_values[
                k
            ]
        )

        metrics[
            f"random_overlap_at_{k}"
        ] = float(
            min(
                k,
                max(
                    n_clients - 1,
                    1,
                ),
            )
            /
            max(
                n_clients - 1,
                1,
            )
        )

    metrics[
        "retrieval_anchors"
    ] = int(
        len(
            anchor_indices
        )
    )

    return metrics


def evaluate_one_private_run(
    dataset,
    representation,
    clean_matrix,
    epsilon_total,
    mechanism_seed,
    C,
    rho_signature,
    sensitivity_mode,
    pair_indices,
    anchor_indices,
):

    epsilon_total = float(
        epsilon_total
    )

    rho_signature = float(
        rho_signature
    )

    if not (
        0.0
        <
        rho_signature
        <=
        1.0
    ):

        raise ValueError(
            "rho_signature must lie in (0,1]."
        )

    epsilon_signature = (
        rho_signature
        *
        epsilon_total
    )

    sensitivity = analytical_sensitivity(
        C=C,
        sensitivity_mode=sensitivity_mode,
    )

    dimension = int(
        clean_matrix.shape[
            1
        ]
    )

    theoretical_mean_radius = float(
        dimension
        *
        sensitivity
        /
        epsilon_signature
    )

    private_result = privatize_matrix(
        clean_matrix=clean_matrix,

        epsilon_signature=(
            epsilon_signature
        ),

        sensitivity=(
            sensitivity
        ),

        dataset=dataset,

        representation=representation,

        mechanism_seed=(
            mechanism_seed
        ),
    )

    private_matrix = private_result[
        "private_matrix"
    ]

    noise_norms = private_result[
        "noise_norms"
    ]

    vector_metrics = vector_preservation_metrics(
        clean_matrix,
        private_matrix,
    )

    pair_metrics = pair_geometry_metrics(
        clean_matrix,
        private_matrix,
        pair_indices,
    )

    retrieval_metrics = retrieval_preservation_metrics(
        clean_matrix,
        private_matrix,
        anchor_indices,
    )

    return {
        "dataset": (
            dataset
        ),

        "representation": (
            representation
        ),

        "epsilon_total": float(
            epsilon_total
        ),

        "rho_signature": float(
            rho_signature
        ),

        "epsilon_signature": float(
            epsilon_signature
        ),

        "mechanism_seed": int(
            mechanism_seed
        ),

        "sensitivity_mode": (
            sensitivity_mode
        ),

        "sensitivity_l2": float(
            sensitivity
        ),

        "clip_C": float(
            C
        ),

        "dimension": int(
            dimension
        ),

        "theoretical_mean_noise_norm": (
            theoretical_mean_radius
        ),

        "observed_mean_noise_norm": float(
            np.mean(
                noise_norms
            )
        ),

        "observed_median_noise_norm": float(
            np.median(
                noise_norms
            )
        ),

        **vector_metrics,

        **pair_metrics,

        **retrieval_metrics,
    }


def aggregate_results(
    dataframe,
):

    rows = []

    group_columns = [
        "dataset",
        "representation",
        "epsilon_total",
        "epsilon_signature",
        "sensitivity_mode",
    ]

    metric_columns = [
        "observed_mean_noise_norm",
        "clean_private_cosine_mean",
        "clean_private_cosine_median",
        "clean_private_l2_mean",
        "clean_private_relative_l2_mean",
        "pair_cosine_spearman",
        "pair_cosine_pearson",
        "pair_cosine_mae",
        "similarity_overlap_at_10",
        "similarity_overlap_at_20",
        "similarity_overlap_at_50",
        "farthest_overlap_at_10",
        "farthest_overlap_at_20",
        "farthest_overlap_at_50",
    ]

    grouped = dataframe.groupby(
        group_columns,
        sort=True,
    )

    for keys, group in grouped:

        (
            dataset,
            representation,
            epsilon_total,
            epsilon_signature,
            sensitivity_mode,
        ) = keys

        row = {
            "dataset": (
                dataset
            ),

            "representation": (
                representation
            ),

            "epsilon_total": float(
                epsilon_total
            ),

            "epsilon_signature": float(
                epsilon_signature
            ),

            "sensitivity_mode": (
                sensitivity_mode
            ),

            "seeds": int(
                group[
                    "mechanism_seed"
                ].nunique()
            ),

            "dimension": int(
                group[
                    "dimension"
                ].iloc[
                    0
                ]
            ),

            "sensitivity_l2": float(
                group[
                    "sensitivity_l2"
                ].iloc[
                    0
                ]
            ),

            "theoretical_mean_noise_norm": float(
                group[
                    "theoretical_mean_noise_norm"
                ].iloc[
                    0
                ]
            ),

            "random_overlap_at_10": float(
                group[
                    "random_overlap_at_10"
                ].iloc[
                    0
                ]
            ),

            "random_overlap_at_20": float(
                group[
                    "random_overlap_at_20"
                ].iloc[
                    0
                ]
            ),

            "random_overlap_at_50": float(
                group[
                    "random_overlap_at_50"
                ].iloc[
                    0
                ]
            ),
        }

        for metric in metric_columns:

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

    return pd.DataFrame(
        rows
    )


def run_dataset(
    dataset,
    output_folder,
    representations,
    epsilons,
    seeds,
    max_clients,
    pair_samples,
    anchors,
    C,
    rho_signature,
    sensitivity_mode,
    d_max,
    degree_weight,
):

    print()

    print(
        "=" * 80
    )

    print(
        f"DATASET: {dataset.upper()}"
    )

    print(
        "=" * 80
    )

    graph_filename = GRAPH_FILENAMES[
        dataset
    ]

    graph_path = os.path.join(
        output_folder,
        dataset,
        graph_filename,
    )

    if not os.path.exists(
        graph_path
    ):

        raise FileNotFoundError(
            f"Missing prepared graph for {dataset}:\n"
            f"  {graph_path}"
        )

    print(
        f"Loading: {graph_path}"
    )

    graph = nx.read_gml(
        graph_path
    )

    print(
        f"Nodes: "
        f"{graph.number_of_nodes():,}"
    )

    print(
        f"Edges: "
        f"{graph.number_of_edges():,}"
    )

    all_clients = get_client_ids(
        graph
    )

    selected_clients = sample_clients(
        all_clients,
        maximum=max_clients,
        dataset=dataset,
    )

    print(
        f"All clients: "
        f"{len(all_clients):,}"
    )

    print(
        f"Selected clients: "
        f"{len(selected_clients):,}"
    )


    print()

    print(
        "Generating clean descriptors..."
    )

    generated = generate_selected_descriptors(
        training_graph=graph,

        client_ids=selected_clients,

        d_max=d_max,

        degree_weight=degree_weight,
    )

    failures = generated[
        "failures"
    ]

    if failures:

        raise RuntimeError(
            f"Descriptor generation failed for "
            f"{len(failures)} clients."
        )

    returned_clients = [
        str(
            value
        )
        for value
        in generated[
            "client_ids"
        ]
    ]

    if (
        returned_clients
        !=
        selected_clients
    ):

        raise RuntimeError(
            "Descriptor client ordering differs from "
            "the requested deterministic client ordering."
        )

    descriptor_matrices = generated[
        "descriptors"
    ]


    experiment_folder = os.path.join(
        output_folder,
        dataset,
        "experiments",
        "eigen_privacy_utility",
    )

    os.makedirs(
        experiment_folder,
        exist_ok=True,
    )

    run_rows = []

    representation_rows = []


    for representation in representations:

        print()

        print(
            "-" * 80
        )

        print(
            f"Representation: {representation}"
        )

        print(
            "-" * 80
        )

        if representation not in descriptor_matrices:

            raise KeyError(
                f"generate_selected_descriptors() did not "
                f"return '{representation}'."
            )

        raw_matrix = np.asarray(
            descriptor_matrices[
                representation
            ],
            dtype=float,
        )

        query = prepare_private_query(
            raw_matrix=raw_matrix,
            C=C,
            representation=representation,
        )

        clean_matrix = query[
            "matrix"
        ]


        if (
            sensitivity_mode
            ==
            "nonnegative_report"
            and
            query[
                "substantive_negative_entries"
            ]
            >
            0
        ):

            raise RuntimeError(
                "Nonnegative client-report sensitivity requested "
                "for a representation that is not nonnegative."
            )

        print(
            f"Dimension: "
            f"{query['dimension']}"
        )

        print(
            "Unique clipped rows at 12 decimals: "
            f"{query['unique_rows_rounded']:,}/"
            f"{len(clean_matrix):,}"
        )

        print(
            f"Zero-row fraction: "
            f"{query['zero_row_fraction']:.6f}"
        )

        print(
            f"Clip fraction: "
            f"{query['clip_fraction']:.6f}"
        )

        print(
            f"Negative-entry fraction: "
            f"{query['negative_entry_fraction']:.9f}"
        )

        representation_rows.append({
            "dataset": (
                dataset
            ),

            "representation": (
                representation
            ),

            "clients": int(
                len(
                    clean_matrix
                )
            ),

            **{
                key: value
                for key, value
                in query.items()
                if key
                !=
                "matrix"
            },
        })


        pair_indices = sample_pair_indices(
            n_clients=len(
                clean_matrix
            ),

            requested=pair_samples,

            dataset=dataset,

            representation=representation,
        )

        anchor_indices = select_anchor_indices(
            n_clients=len(
                clean_matrix
            ),

            requested=anchors,

            dataset=dataset,

            representation=representation,
        )

        sensitivity = analytical_sensitivity(
            C=C,
            sensitivity_mode=sensitivity_mode,
        )

        print(
            f"Analytical Delta2: "
            f"{sensitivity:.9f}"
        )

        print(
            f"Pair samples: "
            f"{len(pair_indices):,}"
        )

        print(
            f"Retrieval anchors: "
            f"{len(anchor_indices):,}"
        )


        for epsilon_total in epsilons:

            print()

            print(
                f"  epsilon_total="
                f"{epsilon_total:g}"
            )

            epsilon_signature = (
                float(
                    rho_signature
                )
                *
                float(
                    epsilon_total
                )
            )

            print(
                f"  epsilon_signature="
                f"{epsilon_signature:g}"
            )

            epsilon_rows = []

            for mechanism_seed in seeds:

                result = evaluate_one_private_run(
                    dataset=dataset,

                    representation=representation,

                    clean_matrix=clean_matrix,

                    epsilon_total=epsilon_total,

                    mechanism_seed=mechanism_seed,

                    C=C,

                    rho_signature=rho_signature,

                    sensitivity_mode=(
                        sensitivity_mode
                    ),

                    pair_indices=(
                        pair_indices
                    ),

                    anchor_indices=(
                        anchor_indices
                    ),
                )

                run_rows.append(
                    result
                )

                epsilon_rows.append(
                    result
                )

            epsilon_dataframe = pd.DataFrame(
                epsilon_rows
            )

            print(
                "    cosine="
                f"{epsilon_dataframe['clean_private_cosine_mean'].mean():.6f}"
                " | pair Spearman="
                f"{epsilon_dataframe['pair_cosine_spearman'].mean():.6f}"
                " | overlap@20="
                f"{epsilon_dataframe['similarity_overlap_at_20'].mean():.6f}"
                " | farthest@20="
                f"{epsilon_dataframe['farthest_overlap_at_20'].mean():.6f}"
                " | noise="
                f"{epsilon_dataframe['observed_mean_noise_norm'].mean():.6f}"
            )


    run_dataframe = pd.DataFrame(
        run_rows
    )

    representation_dataframe = pd.DataFrame(
        representation_rows
    )

    aggregate_dataframe = aggregate_results(
        run_dataframe
    )

    runs_path = os.path.join(
        experiment_folder,
        "privacy_utility_runs.csv",
    )

    aggregate_path = os.path.join(
        experiment_folder,
        "privacy_utility_aggregate.csv",
    )

    representation_path = os.path.join(
        experiment_folder,
        "representation_diagnostics.csv",
    )

    metadata_path = os.path.join(
        experiment_folder,
        "metadata.json",
    )

    run_dataframe.to_csv(
        runs_path,
        index=False,
    )

    aggregate_dataframe.to_csv(
        aggregate_path,
        index=False,
    )

    representation_dataframe.to_csv(
        representation_path,
        index=False,
    )

    metadata = {
        "schema_version": (
            SCHEMA_VERSION
        ),

        "dataset": (
            dataset
        ),

        "graph": (
            graph_path
        ),

        "clients_total": int(
            len(
                all_clients
            )
        ),

        "clients_selected": int(
            len(
                selected_clients
            )
        ),

        "representations": list(
            representations
        ),

        "epsilons_total": [
            float(
                value
            )
            for value
            in epsilons
        ],

        "mechanism_seeds": [
            int(
                value
            )
            for value
            in seeds
        ],

        "clip_C": float(
            C
        ),

        "rho_signature": float(
            rho_signature
        ),

        "sensitivity_mode": (
            sensitivity_mode
        ),

        "analytical_sensitivity": (
            "sqrt(2) * C for nonnegative complete-report "
            "replacement adjacency; otherwise 2 * C."
        ),

        "privacy_mechanism": (
            "Pure L2 K-norm mechanism. Noise direction is "
            "uniform on the unit sphere and radius follows "
            "Gamma(shape=d, scale=Delta2/epsilon_signature)."
        ),

        "budget_statement": (
            "Only the spectral report is simulated here. "
            "epsilon_signature = rho_signature * epsilon_total. "
            "The remaining budget is reserved for the typed-degree "
            "report used by the complete method."
        ),

        "privacy_warning": (
            "No empirical edge sensitivity is used for calibration."
        ),

        "downstream_warning": (
            "No downstream AUC/AP is computed in this file because "
            "that requires rebuilding the exact leakage-safe training "
            "split before descriptor generation."
        ),

        "pair_samples_requested": int(
            pair_samples
        ),

        "retrieval_anchors_requested": int(
            anchors
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
        f"Saved dataset results to:"
    )

    print(
        f"  {experiment_folder}"
    )

    return {
        "runs": (
            run_dataframe
        ),

        "aggregate": (
            aggregate_dataframe
        ),

        "representations": (
            representation_dataframe
        ),

        "metadata": (
            metadata
        ),
    }


def run_experiment(
    output_folder=DEFAULT_OUTPUT_FOLDER,
    datasets=DEFAULT_DATASETS,
    representations=DEFAULT_REPRESENTATIONS,
    epsilons=DEFAULT_EPSILONS,
    seeds=DEFAULT_SEEDS,
    max_clients=DEFAULT_MAX_CLIENTS,
    pair_samples=DEFAULT_PAIR_SAMPLES,
    anchors=DEFAULT_ANCHORS,
    C=DEFAULT_C,
    rho_signature=DEFAULT_RHO_SIGNATURE,
    sensitivity_mode=DEFAULT_SENSITIVITY_MODE,
    d_max=DEFAULT_D_MAX,
    degree_weight=DEFAULT_DEGREE_WEIGHT,
):

    start = time.perf_counter()

    datasets = tuple(
        str(
            value
        ).lower()
        for value
        in datasets
    )

    representations = tuple(
        str(
            value
        )
        for value
        in representations
    )

    epsilons = tuple(
        float(
            value
        )
        for value
        in epsilons
    )

    seeds = tuple(
        int(
            value
        )
        for value
        in seeds
    )

    unsupported = (
        set(
            datasets
        )
        -
        set(
            GRAPH_FILENAMES
        )
    )

    if unsupported:

        raise ValueError(
            f"Unsupported datasets: "
            f"{sorted(unsupported)}"
        )

    if C <= 0.0:

        raise ValueError(
            "C must be strictly positive."
        )

    if not (
        0.0
        <
        rho_signature
        <=
        1.0
    ):

        raise ValueError(
            "rho_signature must lie in (0,1]."
        )

    print()

    print(
        "=" * 80
    )

    print(
        "LOCAL SPECTRAL REPRESENTATION PRIVACY-UTILITY EXPERIMENT"
    )

    print(
        "=" * 80
    )

    print(
        f"Datasets: "
        f"{list(datasets)}"
    )

    print(
        f"Representations: "
        f"{list(representations)}"
    )

    print(
        f"Total epsilons: "
        f"{list(epsilons)}"
    )

    print(
        f"Mechanism seeds: "
        f"{list(seeds)}"
    )

    print(
        f"Clients per dataset: "
        f"up to {max_clients:,}"
    )

    print(
        f"C: "
        f"{C}"
    )

    print(
        f"Signature budget fraction: "
        f"{rho_signature}"
    )

    print(
        f"Sensitivity mode: "
        f"{sensitivity_mode}"
    )

    if sensitivity_mode == "nonnegative_report":

        print(
            "Analytical Delta2 = sqrt(2) C"
        )

    else:

        print(
            "Analytical Delta2 = 2 C"
        )

    print(
        "Adjacency: complete client-report replacement"
    )

    print(
        "Mechanism: pure L2 K-norm"
    )

    print(
        "=" * 80
    )

    dataset_results = {}

    all_runs = []

    all_aggregates = []

    all_representation_diagnostics = []

    for dataset in datasets:

        result = run_dataset(
            dataset=dataset,

            output_folder=output_folder,

            representations=representations,

            epsilons=epsilons,

            seeds=seeds,

            max_clients=max_clients,

            pair_samples=pair_samples,

            anchors=anchors,

            C=C,

            rho_signature=rho_signature,

            sensitivity_mode=sensitivity_mode,

            d_max=d_max,

            degree_weight=degree_weight,
        )

        dataset_results[
            dataset
        ] = result

        all_runs.append(
            result[
                "runs"
            ]
        )

        all_aggregates.append(
            result[
                "aggregate"
            ]
        )

        all_representation_diagnostics.append(
            result[
                "representations"
            ]
        )


    combined_folder = os.path.join(
        output_folder,
        "experiments",
        "eigen_privacy_utility",
    )

    os.makedirs(
        combined_folder,
        exist_ok=True,
    )

    combined_runs = pd.concat(
        all_runs,
        ignore_index=True,
    )

    combined_aggregate = pd.concat(
        all_aggregates,
        ignore_index=True,
    )

    combined_representations = pd.concat(
        all_representation_diagnostics,
        ignore_index=True,
    )

    combined_runs_path = os.path.join(
        combined_folder,
        "all_privacy_utility_runs.csv",
    )

    combined_aggregate_path = os.path.join(
        combined_folder,
        "all_privacy_utility_aggregate.csv",
    )

    combined_representation_path = os.path.join(
        combined_folder,
        "all_representation_diagnostics.csv",
    )

    combined_runs.to_csv(
        combined_runs_path,
        index=False,
    )

    combined_aggregate.to_csv(
        combined_aggregate_path,
        index=False,
    )

    combined_representations.to_csv(
        combined_representation_path,
        index=False,
    )

    runtime = float(
        time.perf_counter()
        -
        start
    )

    print()

    print(
        "=" * 80
    )

    print(
        "PRIVACY-UTILITY EXPERIMENT COMPLETE"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Representation diagnostics:"
    )

    diagnostic_columns = [
        "dataset",
        "representation",
        "clients",
        "dimension",
        "unique_rows_rounded",
        "zero_row_fraction",
        "clip_fraction",
        "negative_entry_fraction",
        "mean_component_variance",
    ]

    print(
        combined_representations[
            diagnostic_columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Selected privacy-utility summary:"
    )

    summary_columns = [
        "dataset",
        "representation",
        "epsilon_total",
        "epsilon_signature",
        "theoretical_mean_noise_norm",
        "observed_mean_noise_norm__mean",
        "clean_private_cosine_mean__mean",
        "pair_cosine_spearman__mean",
        "similarity_overlap_at_20__mean",
        "farthest_overlap_at_20__mean",
        "random_overlap_at_20",
    ]

    print(
        combined_aggregate[
            summary_columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "PRIVACY INTERPRETATION:"
    )

    print(
        "  1. Calibration uses an analytical complete-report "
        "replacement bound."
    )

    print(
        "  2. No empirical edge-sensitivity estimate is used "
        "for the mechanism."
    )

    print(
        "  3. The sqrt(2)C bound is used only for verified "
        "nonnegative clipped descriptors."
    )

    print(
        "  4. epsilon_signature is only the spectral share "
        "of epsilon_total."
    )

    print(
        "  5. If the typed-degree release is added with the "
        "remaining budget, sequential composition applies."
    )

    print(
        "  6. Retrieval overlap should be compared with the "
        "reported random-overlap baseline."
    )

    print(
        "  7. Good clean performance does not imply useful "
        "private performance."
    )

    print(
        "  8. Downstream private AUC/AP must use separately "
        "constructed leakage-safe training graphs."
    )

    print()

    print(
        "Combined outputs:"
    )

    print(
        f"  {combined_runs_path}"
    )

    print(
        f"  {combined_aggregate_path}"
    )

    print(
        f"  {combined_representation_path}"
    )

    print()

    print(
        f"Total runtime: "
        f"{runtime:.3f} s"
    )

    print(
        "=" * 80
    )

    return {
        "datasets": (
            dataset_results
        ),

        "runs": (
            combined_runs
        ),

        "aggregate": (
            combined_aggregate
        ),

        "representations": (
            combined_representations
        ),
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate HKS and eigenvalue descriptor utility "
            "under a valid pure-LDP L2 K-norm client-report "
            "mechanism."
        )
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[
            "enron",
            "dblp",
            "yelp",
        ],
        default=list(
            DEFAULT_DATASETS
        ),
    )

    parser.add_argument(
        "--representations",
        nargs="+",
        choices=list(
            DEFAULT_REPRESENTATIONS
        ),
        default=list(
            DEFAULT_REPRESENTATIONS
        ),
    )

    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=list(
            DEFAULT_EPSILONS
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
        "--max-clients",
        type=int,
        default=DEFAULT_MAX_CLIENTS,
    )

    parser.add_argument(
        "--pair-samples",
        type=int,
        default=DEFAULT_PAIR_SAMPLES,
    )

    parser.add_argument(
        "--anchors",
        type=int,
        default=DEFAULT_ANCHORS,
    )

    parser.add_argument(
        "--C",
        type=float,
        default=DEFAULT_C,
    )

    parser.add_argument(
        "--rho-signature",
        type=float,
        default=DEFAULT_RHO_SIGNATURE,
    )

    parser.add_argument(
        "--sensitivity-mode",
        choices=[
            "nonnegative_report",
            "generic_report",
        ],
        default=DEFAULT_SENSITIVITY_MODE,
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

    run_experiment(
        output_folder=args.output_folder,

        datasets=args.datasets,

        representations=args.representations,

        epsilons=args.epsilons,

        seeds=args.seeds,

        max_clients=args.max_clients,

        pair_samples=args.pair_samples,

        anchors=args.anchors,

        C=args.C,

        rho_signature=args.rho_signature,

        sensitivity_mode=args.sensitivity_mode,

        d_max=args.Dmax,

        degree_weight=args.degree_weight,
    )
