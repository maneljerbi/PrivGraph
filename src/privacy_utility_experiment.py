import itertools
import json
import math
import os
import time

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from scipy.stats import wilcoxon

from noisy_ego_graphs import (
    DEFAULT_D_MAX,
    DEFAULT_MAX_NEIGHBOUR_CLIENTS,
    DEFAULT_SIGNATURE_FRACTION,
    process_ego_graphs,
)


PRIVACY_UTILITY_SCHEMA_VERSION = "1.0"


DEFAULT_EPSILONS = (
    0.1,
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
)


DEFAULT_SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

DEFAULT_SIGNATURE_METHODS = (
    "raw",
    "abs",
    "hks",
)

DEFAULT_SIGNATURE_MECHANISMS = (
    "l2_knorm",
    "coordinate_laplace",
)

DEFAULT_SIGNATURE_CLIP_NORM = 1.0

DEFAULT_DEGREE_MODE = "role"

DEFAULT_DEGREE_ADJACENCY = "client_report"

DEFAULT_K_VALUES = (
    10,
    20,
    50,
)

CONFIDENCE_LEVEL = 0.95


CORE_METRICS = (

    "signature_cosine_clean_private",
    "signature_relative_l2_error_clean",
    "signature_relative_l2_error_clipped",
    "signature_l2_error_clean",
    "signature_noise_l2_norm",


    "degree_mae_raw",
    "degree_rmse_raw",
    "degree_relative_l1_error_raw",
    "degree_l1_error_raw",
    "degree_l2_error_raw",


    "signature_clipping_fraction",
    "degree_clipping_fraction",


    "clean_private_overlap_at_10",
    "clean_private_overlap_at_20",
    "clean_private_overlap_at_50",


    "runtime_seconds",
)


def safe_float_token(value):

    text = format(
        float(value),
        ".12g",
    )

    return (
        text
        .replace("-", "m")
        .replace(".", "p")
        .replace("+", "")
    )


def validate_sequence(values, name):

    values = list(values)

    if not values:
        raise ValueError(
            f"{name} cannot be empty."
        )

    return values


def mean_or_nan(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        values.mean()
    )


def median_or_nan(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        values.median()
    )


def std_or_nan(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan

    return float(
        values.std(
            ddof=0
        )
    )


def confidence_interval(
    values,
    confidence=CONFIDENCE_LEVEL,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    n = len(values)

    if n == 0:

        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
        }

    mean = float(
        np.mean(values)
    )

    if n == 1:

        return {
            "n": 1,
            "mean": mean,
            "std": 0.0,
            "sem": 0.0,
            "ci_low": mean,
            "ci_high": mean,
        }

    std = float(
        np.std(
            values,
            ddof=1,
        )
    )

    sem = float(
        std
        /
        math.sqrt(n)
    )

    alpha = 1.0 - float(
        confidence
    )

    critical = float(
        student_t.ppf(
            1.0 - alpha / 2.0,
            df=n - 1,
        )
    )

    margin = (
        critical
        * sem
    )

    return {
        "n": int(n),
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci_low": float(
            mean - margin
        ),
        "ci_high": float(
            mean + margin
        ),
    }


def summarize_single_run(
    dataset_name,
    epsilon,
    seed,
    signature_method,
    signature_mechanism,
    signature_fraction,
    diagnostics_df,
    neighbour_df,
    runtime_seconds,
    run_folder,
):


    row = {
        "dataset": str(
            dataset_name
        ),

        "epsilon": float(
            epsilon
        ),

        "seed": int(
            seed
        ),

        "signature_method": str(
            signature_method
        ),

        "signature_mechanism": str(
            signature_mechanism
        ),

        "variant": (
            f"{signature_method}"
            f"__{signature_mechanism}"
        ),

        "signature_fraction": float(
            signature_fraction
        ),

        "degree_fraction": float(
            1.0
            - signature_fraction
        ),

        "num_clients": int(
            len(
                diagnostics_df
            )
        ),

        "runtime_seconds": float(
            runtime_seconds
        ),

        "run_folder": str(
            run_folder
        ),
    }


    signature_metrics = (
        "signature_cosine_clean_private",
        "signature_cosine_clipped_private",
        "signature_l2_error_clean",
        "signature_relative_l2_error_clean",
        "signature_l2_error_clipped",
        "signature_relative_l2_error_clipped",
        "signature_noise_l1_norm",
        "signature_noise_l2_norm",
        "signature_noise_linf_norm",
        "clean_signature_l2_norm",
        "private_signature_l2_norm",
    )

    for metric in signature_metrics:

        if metric not in diagnostics_df.columns:
            continue

        row[
            metric
        ] = mean_or_nan(
            diagnostics_df[
                metric
            ]
        )

        row[
            f"{metric}__median_clients"
        ] = median_or_nan(
            diagnostics_df[
                metric
            ]
        )

        row[
            f"{metric}__std_clients"
        ] = std_or_nan(
            diagnostics_df[
                metric
            ]
        )


    if (
        "signature_was_clipped"
        in diagnostics_df.columns
    ):

        clipped = diagnostics_df[
            "signature_was_clipped"
        ].astype(bool)

        row[
            "signature_clipping_fraction"
        ] = float(
            clipped.mean()
        )

        row[
            "signature_clipping_count"
        ] = int(
            clipped.sum()
        )


    degree_metrics = (
        "degree_mae_raw",
        "degree_rmse_raw",
        "degree_relative_l1_error_raw",
        "degree_l1_error_raw",
        "degree_l2_error_raw",
        "degree_mae_clipped",
        "degree_rmse_clipped",
        "degree_l1_error_clipped",
        "degree_l2_error_clipped",
        "degree_noise_l1_norm",
        "degree_noise_l2_norm",
        "degree_exact_component_fraction_raw",
        "degree_exact_component_fraction_clipped",
    )

    for metric in degree_metrics:

        if metric not in diagnostics_df.columns:
            continue

        row[
            metric
        ] = mean_or_nan(
            diagnostics_df[
                metric
            ]
        )

        row[
            f"{metric}__median_clients"
        ] = median_or_nan(
            diagnostics_df[
                metric
            ]
        )

        row[
            f"{metric}__std_clients"
        ] = std_or_nan(
            diagnostics_df[
                metric
            ]
        )


    if (
        "degree_any_component_clipped"
        in diagnostics_df.columns
    ):

        clipped = diagnostics_df[
            "degree_any_component_clipped"
        ].astype(bool)

        row[
            "degree_clipping_fraction"
        ] = float(
            clipped.mean()
        )

        row[
            "degree_clipping_count"
        ] = int(
            clipped.sum()
        )


    for metric in (
        "degree_negative_before_postprocessing",
        "degree_above_D_max_before_postprocessing",
    ):

        if metric in diagnostics_df.columns:

            row[
                metric
            ] = mean_or_nan(
                diagnostics_df[
                    metric
                ]
            )


    private_component_columns = [
        column
        for column
        in diagnostics_df.columns
        if column.startswith(
            "degree_abs_error_raw__"
        )
    ]

    for column in private_component_columns:

        component = column.split(
            "__",
            1,
        )[1]

        row[
            f"degree_component_mae__{component}"
        ] = mean_or_nan(
            diagnostics_df[
                column
            ]
        )

    raw_component_columns = [
        column
        for column
        in diagnostics_df.columns
        if column.startswith(
            "degree_raw__"
        )
    ]

    for column in raw_component_columns:

        component = column.split(
            "__",
            1,
        )[1]

        row[
            f"degree_component_mean_original__{component}"
        ] = mean_or_nan(
            diagnostics_df[
                column
            ]
        )


    if not neighbour_df.empty:

        overlap_columns = [
            column
            for column
            in neighbour_df.columns
            if column.startswith(
                "clean_private_overlap_at_"
            )
        ]

        row[
            "num_neighbour_clients"
        ] = int(
            len(
                neighbour_df
            )
        )

        for column in overlap_columns:

            row[
                column
            ] = mean_or_nan(
                neighbour_df[
                    column
                ]
            )

            row[
                f"{column}__median_clients"
            ] = median_or_nan(
                neighbour_df[
                    column
                ]
            )

            row[
                f"{column}__std_clients"
            ] = std_or_nan(
                neighbour_df[
                    column
                ]
            )

    else:

        row[
            "num_neighbour_clients"
        ] = 0

    return row


def aggregate_across_seeds(
    seed_results_df,
):


    if seed_results_df.empty:
        return pd.DataFrame()

    group_columns = [
        "dataset",
        "epsilon",
        "signature_method",
        "signature_mechanism",
        "variant",
        "signature_fraction",
        "degree_fraction",
    ]

    excluded_numeric_columns = {
        "epsilon",
        "seed",
        "signature_fraction",
        "degree_fraction",
    }

    numeric_columns = [
        column
        for column
        in seed_results_df.columns
        if (
            pd.api.types.is_numeric_dtype(
                seed_results_df[
                    column
                ]
            )
            and column
            not in excluded_numeric_columns
        )
    ]

    rows = []

    grouped = seed_results_df.groupby(
        group_columns,
        dropna=False,
    )

    for group_values, group in grouped:

        row = {
            column: value
            for column, value
            in zip(
                group_columns,
                group_values,
            )
        }

        row[
            "num_seeds"
        ] = int(
            group[
                "seed"
            ].nunique()
        )

        row[
            "seeds_json"
        ] = json.dumps(
            sorted(
                int(value)
                for value
                in group[
                    "seed"
                ].unique()
            )
        )

        for metric in numeric_columns:

            ci = confidence_interval(
                group[
                    metric
                ].to_numpy(
                    dtype=float
                ),
                confidence=CONFIDENCE_LEVEL,
            )

            row[
                f"{metric}__n"
            ] = ci[
                "n"
            ]

            row[
                f"{metric}__mean"
            ] = ci[
                "mean"
            ]

            row[
                f"{metric}__std"
            ] = ci[
                "std"
            ]

            row[
                f"{metric}__sem"
            ] = ci[
                "sem"
            ]

            row[
                f"{metric}__ci95_low"
            ] = ci[
                "ci_low"
            ]

            row[
                f"{metric}__ci95_high"
            ] = ci[
                "ci_high"
            ]

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def cohen_dz(
    differences,
):

    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(
            differences
        )
    ]

    if len(
        differences
    ) < 2:
        return np.nan

    std = np.std(
        differences,
        ddof=1,
    )

    if std <= 0:
        return (
            0.0
            if np.allclose(
                differences,
                0.0,
            )
            else np.inf
        )

    return float(
        np.mean(
            differences
        )
        /
        std
    )


def paired_wilcoxon(
    first,
    second,
):

    first = np.asarray(
        first,
        dtype=float,
    )

    second = np.asarray(
        second,
        dtype=float,
    )

    mask = (
        np.isfinite(
            first
        )
        &
        np.isfinite(
            second
        )
    )

    first = first[
        mask
    ]

    second = second[
        mask
    ]

    if len(
        first
    ) < 2:
        return np.nan

    differences = (
        first
        - second
    )

    if np.allclose(
        differences,
        0.0,
    ):
        return 1.0

    try:

        result = wilcoxon(
            first,
            second,
            alternative="two-sided",
            zero_method="wilcox",
        )

        return float(
            result.pvalue
        )

    except ValueError:

        return np.nan


def build_pairwise_comparisons(
    seed_results_df,
    metrics=None,
):


    if metrics is None:

        metrics = [
            "signature_cosine_clean_private",
            "signature_relative_l2_error_clean",
            "degree_mae_raw",
            "degree_relative_l1_error_raw",
            "clean_private_overlap_at_10",
            "clean_private_overlap_at_20",
            "clean_private_overlap_at_50",
        ]

    if seed_results_df.empty:
        return pd.DataFrame()

    rows = []

    grouping_columns = [
        "dataset",
        "epsilon",
        "signature_fraction",
    ]

    for group_values, group in (
        seed_results_df.groupby(
            grouping_columns,
            dropna=False,
        )
    ):

        (
            dataset,
            epsilon,
            signature_fraction,
        ) = group_values

        variants = sorted(
            group[
                "variant"
            ].unique()
        )

        for variant_a, variant_b in itertools.combinations(
            variants,
            2,
        ):

            data_a = group[
                group[
                    "variant"
                ]
                == variant_a
            ]

            data_b = group[
                group[
                    "variant"
                ]
                == variant_b
            ]


            shared_seeds = sorted(
                set(
                    data_a[
                        "seed"
                    ]
                )
                &
                set(
                    data_b[
                        "seed"
                    ]
                )
            )

            if not shared_seeds:
                continue

            data_a = (
                data_a
                .set_index(
                    "seed"
                )
                .loc[
                    shared_seeds
                ]
            )

            data_b = (
                data_b
                .set_index(
                    "seed"
                )
                .loc[
                    shared_seeds
                ]
            )

            for metric in metrics:

                if (
                    metric
                    not in data_a.columns
                    or metric
                    not in data_b.columns
                ):
                    continue

                first = pd.to_numeric(
                    data_a[
                        metric
                    ],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )

                second = pd.to_numeric(
                    data_b[
                        metric
                    ],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )

                mask = (
                    np.isfinite(
                        first
                    )
                    &
                    np.isfinite(
                        second
                    )
                )

                first = first[
                    mask
                ]

                second = second[
                    mask
                ]

                if len(
                    first
                ) == 0:
                    continue

                differences = (
                    first
                    - second
                )

                ci = confidence_interval(
                    differences,
                    confidence=CONFIDENCE_LEVEL,
                )

                rows.append({
                    "dataset": dataset,

                    "epsilon": float(
                        epsilon
                    ),

                    "signature_fraction": float(
                        signature_fraction
                    ),

                    "variant_a": (
                        variant_a
                    ),

                    "variant_b": (
                        variant_b
                    ),

                    "metric": metric,

                    "num_paired_seeds": int(
                        len(
                            differences
                        )
                    ),

                    "mean_variant_a": float(
                        np.mean(
                            first
                        )
                    ),

                    "mean_variant_b": float(
                        np.mean(
                            second
                        )
                    ),

                    "mean_difference_a_minus_b": float(
                        np.mean(
                            differences
                        )
                    ),

                    "difference_std": (
                        float(
                            np.std(
                                differences,
                                ddof=1,
                            )
                        )
                        if len(
                            differences
                        )
                        > 1
                        else 0.0
                    ),

                    "difference_ci95_low": (
                        ci[
                            "ci_low"
                        ]
                    ),

                    "difference_ci95_high": (
                        ci[
                            "ci_high"
                        ]
                    ),

                    "cohen_dz": (
                        cohen_dz(
                            differences
                        )
                    ),

                    "wilcoxon_p_value": (
                        paired_wilcoxon(
                            first,
                            second,
                        )
                    ),

                    "seeds_json": json.dumps(
                        [
                            int(
                                seed
                            )
                            for seed
                            in np.asarray(
                                shared_seeds
                            )[
                                mask
                            ]
                        ]
                    ),
                })

    return pd.DataFrame(
        rows
    )


def build_mechanism_comparison_table(
    seed_results_df,
):


    metrics = [
        "signature_cosine_clean_private",
        "signature_relative_l2_error_clean",
        "signature_l2_error_clean",
        "signature_noise_l2_norm",
        "clean_private_overlap_at_10",
        "clean_private_overlap_at_20",
        "clean_private_overlap_at_50",
    ]

    rows = []

    grouping_columns = [
        "dataset",
        "epsilon",
        "signature_method",
        "signature_fraction",
    ]

    for group_values, group in (
        seed_results_df.groupby(
            grouping_columns,
            dropna=False,
        )
    ):

        (
            dataset,
            epsilon,
            signature_method,
            signature_fraction,
        ) = group_values

        knorm = group[
            group[
                "signature_mechanism"
            ]
            == "l2_knorm"
        ]

        laplace = group[
            group[
                "signature_mechanism"
            ]
            == "coordinate_laplace"
        ]

        if (
            knorm.empty
            or laplace.empty
        ):
            continue

        shared_seeds = sorted(
            set(
                knorm[
                    "seed"
                ]
            )
            &
            set(
                laplace[
                    "seed"
                ]
            )
        )

        if not shared_seeds:
            continue

        knorm = (
            knorm
            .set_index(
                "seed"
            )
            .loc[
                shared_seeds
            ]
        )

        laplace = (
            laplace
            .set_index(
                "seed"
            )
            .loc[
                shared_seeds
            ]
        )

        for metric in metrics:

            if (
                metric
                not in knorm.columns
                or metric
                not in laplace.columns
            ):
                continue

            first = pd.to_numeric(
                knorm[
                    metric
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            second = pd.to_numeric(
                laplace[
                    metric
                ],
                errors="coerce",
            ).to_numpy(
                dtype=float
            )

            mask = (
                np.isfinite(
                    first
                )
                &
                np.isfinite(
                    second
                )
            )

            first = first[
                mask
            ]

            second = second[
                mask
            ]

            if len(
                first
            ) == 0:
                continue

            differences = (
                first
                - second
            )

            ci = confidence_interval(
                differences
            )

            rows.append({
                "dataset": dataset,

                "epsilon": float(
                    epsilon
                ),

                "signature_method": (
                    signature_method
                ),

                "signature_fraction": float(
                    signature_fraction
                ),

                "metric": metric,

                "num_paired_seeds": int(
                    len(
                        differences
                    )
                ),

                "mean_l2_knorm": float(
                    np.mean(
                        first
                    )
                ),

                "mean_coordinate_laplace": float(
                    np.mean(
                        second
                    )
                ),

                "mean_difference_knorm_minus_laplace": float(
                    np.mean(
                        differences
                    )
                ),

                "difference_ci95_low": (
                    ci[
                        "ci_low"
                    ]
                ),

                "difference_ci95_high": (
                    ci[
                        "ci_high"
                    ]
                ),

                "cohen_dz": (
                    cohen_dz(
                        differences
                    )
                ),

                "wilcoxon_p_value": (
                    paired_wilcoxon(
                        first,
                        second,
                    )
                ),
            })

    return pd.DataFrame(
        rows
    )


def build_best_variant_table(
    aggregate_df,
):


    if aggregate_df.empty:
        return pd.DataFrame()

    objectives = {

        "signature_cosine_clean_private": "max",
        "clean_private_overlap_at_20": "max",


        "signature_relative_l2_error_clean": "min",
        "degree_mae_raw": "min",
    }

    rows = []

    group_columns = [
        "dataset",
        "epsilon",
        "signature_fraction",
    ]

    for group_values, group in (
        aggregate_df.groupby(
            group_columns,
            dropna=False,
        )
    ):

        (
            dataset,
            epsilon,
            signature_fraction,
        ) = group_values

        for metric, direction in objectives.items():

            column = (
                f"{metric}__mean"
            )

            if column not in group.columns:
                continue

            valid = group[
                np.isfinite(
                    pd.to_numeric(
                        group[
                            column
                        ],
                        errors="coerce",
                    )
                )
            ]

            if valid.empty:
                continue

            if direction == "max":

                best_index = valid[
                    column
                ].idxmax()

            else:

                best_index = valid[
                    column
                ].idxmin()

            best = valid.loc[
                best_index
            ]

            rows.append({
                "dataset": dataset,

                "epsilon": float(
                    epsilon
                ),

                "signature_fraction": float(
                    signature_fraction
                ),

                "metric": metric,

                "direction": direction,

                "best_variant": best[
                    "variant"
                ],

                "best_signature_method": best[
                    "signature_method"
                ],

                "best_signature_mechanism": best[
                    "signature_mechanism"
                ],

                "best_mean": float(
                    best[
                        column
                    ]
                ),

                "best_ci95_low": float(
                    best[
                        f"{metric}__ci95_low"
                    ]
                ),

                "best_ci95_high": float(
                    best[
                        f"{metric}__ci95_high"
                    ]
                ),
            })

    return pd.DataFrame(
        rows
    )


def run_privacy_utility_experiment(
    dataset_name="enron",
    output_folder="output/",
    epsilons=DEFAULT_EPSILONS,
    seeds=DEFAULT_SEEDS,
    signature_methods=DEFAULT_SIGNATURE_METHODS,
    signature_mechanisms=DEFAULT_SIGNATURE_MECHANISMS,
    signature_fraction=DEFAULT_SIGNATURE_FRACTION,
    signature_clip_norm=DEFAULT_SIGNATURE_CLIP_NORM,
    d_max=DEFAULT_D_MAX,
    degree_mode=DEFAULT_DEGREE_MODE,
    degree_components=None,
    degree_adjacency=DEFAULT_DEGREE_ADJACENCY,
    max_clients=None,
    max_neighbour_clients=DEFAULT_MAX_NEIGHBOUR_CLIENTS,
    k_values=DEFAULT_K_VALUES,
    force_rebuild=False,
):


    epsilons = validate_sequence(
        epsilons,
        "epsilons",
    )

    seeds = validate_sequence(
        seeds,
        "seeds",
    )

    signature_methods = validate_sequence(
        signature_methods,
        "signature_methods",
    )

    signature_mechanisms = validate_sequence(
        signature_mechanisms,
        "signature_mechanisms",
    )

    dataset_folder = os.path.join(
        output_folder,
        dataset_name,
    )

    experiment_folder = os.path.join(
        dataset_folder,
        "experiments",
        "privacy_utility",
    )

    os.makedirs(
        experiment_folder,
        exist_ok=True,
    )

    seed_results_path = os.path.join(
        experiment_folder,
        "privacy_utility_seed_results.csv",
    )

    aggregate_path = os.path.join(
        experiment_folder,
        "privacy_utility_aggregate.csv",
    )

    pairwise_path = os.path.join(
        experiment_folder,
        "privacy_utility_pairwise_statistics.csv",
    )

    mechanism_path = os.path.join(
        experiment_folder,
        "privacy_mechanism_comparison.csv",
    )

    best_variant_path = os.path.join(
        experiment_folder,
        "privacy_utility_best_variants.csv",
    )

    metadata_path = os.path.join(
        experiment_folder,
        "privacy_utility_metadata.json",
    )


    seed_rows = []

    total_runs = (
        len(
            epsilons
        )
        *
        len(
            seeds
        )
        *
        len(
            signature_methods
        )
        *
        len(
            signature_mechanisms
        )
    )

    run_number = 0

    experiment_start = time.perf_counter()

    failures = []

    for epsilon in epsilons:

        epsilon = float(
            epsilon
        )

        if epsilon <= 0:

            raise ValueError(
                "All epsilon values must be positive."
            )

        for signature_method in signature_methods:

            for signature_mechanism in signature_mechanisms:

                for seed in seeds:

                    run_number += 1

                    print(
                        "\n"
                        "========================================"
                    )

                    print(
                        "PRIVACY-UTILITY EXPERIMENT"
                    )

                    print(
                        f"Run {run_number}/{total_runs}"
                    )

                    print(
                        f"Dataset: {dataset_name}"
                    )

                    print(
                        f"epsilon: {epsilon:g}"
                    )

                    print(
                        f"signature: {signature_method}"
                    )

                    print(
                        f"mechanism: {signature_mechanism}"
                    )

                    print(
                        f"seed: {seed}"
                    )

                    print(
                        "========================================"
                    )

                    run_start = time.perf_counter()

                    try:

                        result = process_ego_graphs(
                            G=None,

                            epsilon=epsilon,

                            dataset_name=dataset_name,

                            output_folder=output_folder,

                            signature_method=(
                                signature_method
                            ),

                            signature_mechanism=(
                                signature_mechanism
                            ),

                            signature_fraction=(
                                signature_fraction
                            ),

                            signature_clip_norm=(
                                signature_clip_norm
                            ),

                            d_max=d_max,

                            degree_mode=(
                                degree_mode
                            ),

                            degree_components=(
                                degree_components
                            ),

                            degree_adjacency=(
                                degree_adjacency
                            ),

                            seed=int(
                                seed
                            ),

                            max_clients=(
                                max_clients
                            ),

                            max_neighbour_clients=(
                                max_neighbour_clients
                            ),

                            k_values=(
                                k_values
                            ),

                            force_rebuild=(
                                force_rebuild
                            ),
                        )

                        runtime_seconds = float(
                            time.perf_counter()
                            - run_start
                        )

                        row = summarize_single_run(
                            dataset_name=(
                                dataset_name
                            ),

                            epsilon=(
                                epsilon
                            ),

                            seed=(
                                seed
                            ),

                            signature_method=(
                                signature_method
                            ),

                            signature_mechanism=(
                                signature_mechanism
                            ),

                            signature_fraction=(
                                signature_fraction
                            ),

                            diagnostics_df=(
                                result[
                                    "diagnostics"
                                ]
                            ),

                            neighbour_df=(
                                result[
                                    "neighbour_stability"
                                ]
                            ),

                            runtime_seconds=(
                                runtime_seconds
                            ),

                            run_folder=(
                                result[
                                    "run_folder"
                                ]
                            ),
                        )

                        seed_rows.append(
                            row
                        )


                        pd.DataFrame(
                            seed_rows
                        ).to_csv(
                            seed_results_path,
                            index=False,
                        )

                    except Exception as exc:

                        failures.append({
                            "epsilon": float(
                                epsilon
                            ),

                            "signature_method": (
                                signature_method
                            ),

                            "signature_mechanism": (
                                signature_mechanism
                            ),

                            "seed": int(
                                seed
                            ),

                            "error": repr(
                                exc
                            ),
                        })

                        print(
                            "[privacy utility] "
                            "RUN FAILED:"
                        )

                        print(
                            repr(
                                exc
                            )
                        )


    seed_results_df = pd.DataFrame(
        seed_rows
    )

    if seed_results_df.empty:

        raise RuntimeError(
            "Every privacy-utility run failed."
        )

    seed_results_df = (
        seed_results_df
        .sort_values(
            [
                "epsilon",
                "signature_method",
                "signature_mechanism",
                "seed",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    seed_results_df.to_csv(
        seed_results_path,
        index=False,
    )


    aggregate_df = (
        aggregate_across_seeds(
            seed_results_df
        )
    )

    aggregate_df = (
        aggregate_df
        .sort_values(
            [
                "epsilon",
                "signature_method",
                "signature_mechanism",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    aggregate_df.to_csv(
        aggregate_path,
        index=False,
    )


    pairwise_df = (
        build_pairwise_comparisons(
            seed_results_df
        )
    )

    pairwise_df.to_csv(
        pairwise_path,
        index=False,
    )


    mechanism_df = (
        build_mechanism_comparison_table(
            seed_results_df
        )
    )

    mechanism_df.to_csv(
        mechanism_path,
        index=False,
    )


    best_variant_df = (
        build_best_variant_table(
            aggregate_df
        )
    )

    best_variant_df.to_csv(
        best_variant_path,
        index=False,
    )


    total_runtime_seconds = float(
        time.perf_counter()
        - experiment_start
    )

    metadata = {
        "schema_version": (
            PRIVACY_UTILITY_SCHEMA_VERSION
        ),

        "dataset": dataset_name,

        "epsilons": [
            float(
                value
            )
            for value
            in epsilons
        ],

        "seeds": [
            int(
                value
            )
            for value
            in seeds
        ],

        "number_of_seeds": int(
            len(
                seeds
            )
        ),

        "signature_methods": list(
            signature_methods
        ),

        "signature_mechanisms": list(
            signature_mechanisms
        ),

        "signature_fraction": float(
            signature_fraction
        ),

        "degree_fraction": float(
            1.0
            - signature_fraction
        ),

        "signature_clip_norm_C": float(
            signature_clip_norm
        ),

        "D_max": int(
            d_max
        ),

        "degree_mode": (
            degree_mode
        ),

        "degree_components": (
            None
            if degree_components
            is None
            else [
                str(
                    value
                )
                for value
                in degree_components
            ]
        ),

        "degree_adjacency": (
            degree_adjacency
        ),

        "max_clients": (
            None
            if max_clients
            is None
            else int(
                max_clients
            )
        ),

        "max_neighbour_clients": (
            None
            if max_neighbour_clients
            is None
            else int(
                max_neighbour_clients
            )
        ),

        "k_values": [
            int(
                value
            )
            for value
            in k_values
        ],

        "confidence_level": float(
            CONFIDENCE_LEVEL
        ),

        "planned_runs": int(
            total_runs
        ),

        "successful_runs": int(
            len(
                seed_results_df
            )
        ),

        "failed_runs": int(
            len(
                failures
            )
        ),

        "failures": failures,

        "total_runtime_seconds": float(
            total_runtime_seconds
        ),

        "outputs": {
            "seed_results": os.path.basename(
                seed_results_path
            ),

            "aggregate": os.path.basename(
                aggregate_path
            ),

            "pairwise_statistics": os.path.basename(
                pairwise_path
            ),

            "mechanism_comparison": os.path.basename(
                mechanism_path
            ),

            "best_variants": os.path.basename(
                best_variant_path
            ),
        },

        "statistical_design": {
            "independent_replication_unit": (
                "random seed"
            ),

            "cross_seed_summary": (
                "mean, sample standard deviation, "
                "standard error, Student-t 95% CI"
            ),

            "paired_test": (
                "two-sided Wilcoxon signed-rank test "
                "using identical seed indices"
            ),

            "paired_effect_size": (
                "Cohen's dz"
            ),
        },

        "experiment_questions": [
            (
                "How does privacy budget epsilon affect "
                "signature and degree utility?"
            ),

            (
                "Does HKS retain more useful structural "
                "information under LDP than RAW or ABS?"
            ),

            (
                "Does the geometry-aware L2 K-norm mechanism "
                "outperform coordinate-wise Laplace under the "
                "same pure-LDP privacy budget?"
            ),

            (
                "How stable are candidate-neighbour rankings "
                "after local perturbation?"
            ),

            (
                "How much typed-degree information survives "
                "client-report LDP?"
            ),
        ],
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4,
            ensure_ascii=False,
        )


    print(
        "\n"
        "========================================"
    )

    print(
        "PRIVACY-UTILITY EXPERIMENT COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Dataset: {dataset_name}"
    )

    print(
        f"Successful runs: "
        f"{len(seed_results_df)}/{total_runs}"
    )

    print(
        f"Failed runs: "
        f"{len(failures)}"
    )

    print(
        "\nResults:"
    )

    print(
        f"  Seed level:"
        f"\n    {seed_results_path}"
    )

    print(
        f"  Mean/std/95% CI:"
        f"\n    {aggregate_path}"
    )

    print(
        f"  Pairwise statistics:"
        f"\n    {pairwise_path}"
    )

    print(
        f"  K-norm vs Laplace:"
        f"\n    {mechanism_path}"
    )

    print(
        f"  Best variants:"
        f"\n    {best_variant_path}"
    )

    print(
        f"  Metadata:"
        f"\n    {metadata_path}"
    )

    print(
        "========================================\n"
    )

    return {
        "seed_results": (
            seed_results_df
        ),

        "aggregate": (
            aggregate_df
        ),

        "pairwise": (
            pairwise_df
        ),

        "mechanism_comparison": (
            mechanism_df
        ),

        "best_variants": (
            best_variant_df
        ),

        "metadata": metadata,
    }


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Run the journal privacy-utility "
            "experiment across epsilon values, "
            "signature types, mechanisms and seeds."
        )
    )

    parser.add_argument(
        "dataset",
        type=str,
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default="output/",
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
        "--signatures",
        nargs="+",
        choices=[
            "raw",
            "abs",
            "hks",
        ],
        default=list(
            DEFAULT_SIGNATURE_METHODS
        ),
    )

    parser.add_argument(
        "--mechanisms",
        nargs="+",
        choices=[
            "l2_knorm",
            "coordinate_laplace",
        ],
        default=list(
            DEFAULT_SIGNATURE_MECHANISMS
        ),
    )

    parser.add_argument(
        "--signature-fraction",
        type=float,
        default=(
            DEFAULT_SIGNATURE_FRACTION
        ),
    )

    parser.add_argument(
        "--C",
        type=float,
        default=(
            DEFAULT_SIGNATURE_CLIP_NORM
        ),
    )

    parser.add_argument(
        "--D-max",
        type=int,
        default=DEFAULT_D_MAX,
    )

    parser.add_argument(
        "--degree-mode",
        choices=[
            "role",
            "type",
            "relation",
        ],
        default=DEFAULT_DEGREE_MODE,
    )

    parser.add_argument(
        "--degree-adjacency",
        choices=[
            "client_report",
            "incident_edge",
        ],
        default=(
            DEFAULT_DEGREE_ADJACENCY
        ),
    )

    parser.add_argument(
        "--max-clients",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-neighbour-clients",
        type=int,
        default=(
            DEFAULT_MAX_NEIGHBOUR_CLIENTS
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    run_privacy_utility_experiment(
        dataset_name=args.dataset,

        output_folder=args.output_folder,

        epsilons=args.epsilons,

        seeds=args.seeds,

        signature_methods=(
            args.signatures
        ),

        signature_mechanisms=(
            args.mechanisms
        ),

        signature_fraction=(
            args.signature_fraction
        ),

        signature_clip_norm=(
            args.C
        ),

        d_max=args.D_max,

        degree_mode=(
            args.degree_mode
        ),

        degree_adjacency=(
            args.degree_adjacency
        ),

        max_clients=(
            args.max_clients
        ),

        max_neighbour_clients=(
            args.max_neighbour_clients
        ),

        force_rebuild=(
            args.force
        ),
    )
