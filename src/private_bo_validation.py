import hashlib
import json
import os
import time

import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import (
    kendalltau,
    pearsonr,
    spearmanr,
)

from noisy_ego_graphs import (
    load_private_reports,
)

from synthesized_graph import (
    compute_exact_top_candidates,
    compute_private_objective,
    extract_private_targets,
    reconstruct_graph_from_private_reports,
)

from stats import (
    DEFAULT_MAX_MMD_NODES,
    build_category_comparison_table,
    build_degree_distribution_comparison,
    build_metapath_comparison_table,
    calculate_graph_statistics,
    calculate_structural_mmd,
    client_edge_recovery_metrics,
    derive_oracle_objective,
)


BO_VALIDATION_SCHEMA_VERSION = "1.1"

DEFAULT_SEED = 0

DEFAULT_TOP_K_VALUES = (
    3,
    5,
    10,
)

NUMERICAL_EPS = 1e-12
DEFAULT_REPLAY_TOLERANCE = 1e-9


class PrivateReplayMismatchError(RuntimeError):
    pass


def stable_configuration_hash(value):

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def file_sha256(path):

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_correlation(
    x,
    y,
    method="spearman",
):


    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    mask = (
        np.isfinite(
            x
        )
        &
        np.isfinite(
            y
        )
    )

    x = x[
        mask
    ]

    y = y[
        mask
    ]

    if len(
        x
    ) < 2:

        return {
            "n": int(
                len(
                    x
                )
            ),

            "correlation": np.nan,

            "p_value": np.nan,
        }

    if (
        np.allclose(
            x,
            x[
                0
            ],
        )
        or np.allclose(
            y,
            y[
                0
            ],
        )
    ):

        return {
            "n": int(
                len(
                    x
                )
            ),

            "correlation": np.nan,

            "p_value": np.nan,
        }

    try:

        if method == "spearman":

            result = spearmanr(
                x,
                y,
            )

            correlation = float(
                result.statistic
            )

            p_value = float(
                result.pvalue
            )

        elif method == "pearson":

            result = pearsonr(
                x,
                y,
            )

            correlation = float(
                result.statistic
            )

            p_value = float(
                result.pvalue
            )

        elif method == "kendall":

            result = kendalltau(
                x,
                y,
            )

            correlation = float(
                result.statistic
            )

            p_value = float(
                result.pvalue
            )

        else:

            raise ValueError(
                "method must be "
                "'spearman', 'pearson', or 'kendall'."
            )

        return {
            "n": int(
                len(
                    x
                )
            ),

            "correlation": (
                correlation
            ),

            "p_value": (
                p_value
            ),
        }

    except Exception:

        return {
            "n": int(
                len(
                    x
                )
            ),

            "correlation": np.nan,

            "p_value": np.nan,
        }


def safe_relative_regret(
    best_value,
    selected_value,
):


    absolute_regret = float(
        best_value
        -
        selected_value
    )

    relative_regret = float(
        absolute_regret
        /
        max(
            abs(
                float(
                    best_value
                )
            ),
            NUMERICAL_EPS,
        )
    )

    return (
        absolute_regret,
        relative_regret,
    )


def load_original_graph(
    dataset_name,
    output_folder,
):


    path = os.path.join(
        output_folder,
        dataset_name,
        (
            f"{dataset_name}"
            "_global_graph.gml.gz"
        ),
    )

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            "Missing original graph: "
            f"{path}"
        )

    return nx.read_gml(
        path
    )


def load_bo_trace(
    private_run_folder,
):


    path = os.path.join(
        private_run_folder,
        "synthesis",
        "private_bo_trace.csv",
    )

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            "Missing private BO trace: "
            f"{path}"
        )

    trace = pd.read_csv(
        path
    )

    required_columns = {
        "iteration",
        "private_objective",
        "similarity_threshold",
        "client_degree_scale",
        "content_degree_scale",
        "candidate_k",
    }

    missing = (
        required_columns
        -
        set(
            trace.columns
        )
    )

    if missing:

        raise ValueError(
            "BO trace is missing columns: "
            f"{sorted(missing)}"
        )

    return trace


def load_bo_summary(
    private_run_folder,
):


    path = os.path.join(
        private_run_folder,
        "synthesis",
        "private_bo_summary.json",
    )

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            "Missing private BO summary: "
            f"{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def parameters_from_trace_row(
    row,
):


    return {
        "similarity_threshold": float(
            row[
                "similarity_threshold"
            ]
        ),

        "client_degree_scale": float(
            row[
                "client_degree_scale"
            ]
        ),

        "content_degree_scale": float(
            row[
                "content_degree_scale"
            ]
        ),

        "candidate_k": int(
            row[
                "candidate_k"
            ]
        ),
    }


def evaluate_one_candidate_oracle(
    original_graph,
    original_stats,
    synthetic_graph,
    seed,
):


    synthetic_stats = (
        calculate_graph_statistics(
            synthetic_graph,
            seed=seed,
        )
    )


    degree_df = (
        build_degree_distribution_comparison(
            original_graph,
            synthetic_graph,
        )
    )


    (
        category_df,
        category_summary,
    ) = build_category_comparison_table(
        original_graph,
        synthetic_graph,
    )


    (
        metapath_df,
        metapath_summary,
    ) = build_metapath_comparison_table(
        original_graph,
        synthetic_graph,
    )


    structural_mmd = (
        calculate_structural_mmd(
            original_graph,
            synthetic_graph,
            seed=seed,
        )
    )


    edge_recovery = (
        client_edge_recovery_metrics(
            original_graph,
            synthetic_graph,
        )
    )


    oracle = derive_oracle_objective(
        original_stats=(
            original_stats
        ),

        synthetic_stats=(
            synthetic_stats
        ),

        degree_df=(
            degree_df
        ),

        category_summary=(
            category_summary
        ),

        metapath_summary=(
            metapath_summary
        ),

        structural_mmd=(
            structural_mmd
        ),
    )


    result = {
        **oracle,

        "oracle_client_edge_precision": float(
            edge_recovery[
                "precision"
            ]
        ),

        "oracle_client_edge_recall": float(
            edge_recovery[
                "recall"
            ]
        ),

        "oracle_client_edge_f1": float(
            edge_recovery[
                "f1"
            ]
        ),

        "oracle_client_edge_jaccard": float(
            edge_recovery[
                "jaccard"
            ]
        ),

        "oracle_node_count_relative_error": float(
            abs(
                synthetic_graph.number_of_nodes()
                -
                original_graph.number_of_nodes()
            )
            /
            max(
                original_graph.number_of_nodes(),
                1,
            )
        ),

        "oracle_edge_count_relative_error": float(
            abs(
                synthetic_graph.number_of_edges()
                -
                original_graph.number_of_edges()
            )
            /
            max(
                original_graph.number_of_edges(),
                1,
            )
        ),

        "oracle_average_clustering": float(
            synthetic_stats[
                "average_clustering"
            ]
        ),

        "oracle_original_average_clustering": float(
            original_stats[
                "average_clustering"
            ]
        ),

        "oracle_modularity": float(
            synthetic_stats[
                "modularity"
            ]
        ),

        "oracle_original_modularity": float(
            original_stats[
                "modularity"
            ]
        ),
    }

    if structural_mmd is not None:

        result[
            "oracle_mmd_squared"
        ] = float(
            structural_mmd[
                "mmd_squared"
            ]
        )

    else:

        result[
            "oracle_mmd_squared"
        ] = np.nan

    return result


def add_ranking_columns(
    validation_df,
):


    validation_df = (
        validation_df.copy()
    )

    validation_df[
        "private_rank"
    ] = validation_df[
        "private_objective"
    ].rank(
        method="min",
        ascending=False,
    )

    validation_df[
        "oracle_rank"
    ] = validation_df[
        "oracle_objective"
    ].rank(
        method="min",
        ascending=False,
    )

    validation_df[
        "rank_displacement"
    ] = np.abs(
        validation_df[
            "private_rank"
        ]
        -
        validation_df[
            "oracle_rank"
        ]
    )

    return validation_df


def compute_top_k_overlap(
    validation_df,
    k_values=DEFAULT_TOP_K_VALUES,
):


    rows = []

    n = len(
        validation_df
    )

    if n == 0:
        return pd.DataFrame()

    private_order = (
        validation_df
        .sort_values(
            "private_objective",
            ascending=False,
        )
    )

    oracle_order = (
        validation_df
        .sort_values(
            "oracle_objective",
            ascending=False,
        )
    )

    for requested_k in k_values:

        k = min(
            int(
                requested_k
            ),
            n,
        )

        if k <= 0:
            continue

        private_set = set(
            private_order[
                "iteration"
            ]
            .head(
                k
            )
            .astype(
                int
            )
        )

        oracle_set = set(
            oracle_order[
                "iteration"
            ]
            .head(
                k
            )
            .astype(
                int
            )
        )

        intersection = (
            private_set
            &
            oracle_set
        )

        union = (
            private_set
            |
            oracle_set
        )

        rows.append({
            "requested_k": int(
                requested_k
            ),

            "effective_k": int(
                k
            ),

            "intersection_size": int(
                len(
                    intersection
                )
            ),

            "overlap_fraction": float(
                len(
                    intersection
                )
                /
                k
            ),

            "jaccard": float(
                len(
                    intersection
                )
                /
                len(
                    union
                )
                if union
                else 1.0
            ),

            "private_iterations_json": (
                json.dumps(
                    sorted(
                        int(
                            value
                        )
                        for value
                        in private_set
                    )
                )
            ),

            "oracle_iterations_json": (
                json.dumps(
                    sorted(
                        int(
                            value
                        )
                        for value
                        in oracle_set
                    )
                )
            ),
        })

    return pd.DataFrame(
        rows
    )


def compute_convergence_table(
    validation_df,
):


    ordered = (
        validation_df
        .sort_values(
            "iteration"
        )
        .reset_index(
            drop=True
        )
    )

    rows = []

    best_private_value = -np.inf

    best_private_index = None

    best_oracle_value = -np.inf

    best_oracle_index = None

    for position in range(
        len(
            ordered
        )
    ):

        current = ordered.iloc[
            position
        ]

        private_value = float(
            current[
                "private_objective"
            ]
        )

        oracle_value = float(
            current[
                "oracle_objective"
            ]
        )

        if private_value > best_private_value:

            best_private_value = (
                private_value
            )

            best_private_index = (
                position
            )

        if oracle_value > best_oracle_value:

            best_oracle_value = (
                oracle_value
            )

            best_oracle_index = (
                position
            )

        private_selected_row = (
            ordered.iloc[
                best_private_index
            ]
        )

        oracle_selected_row = (
            ordered.iloc[
                best_oracle_index
            ]
        )

        oracle_of_private_best = float(
            private_selected_row[
                "oracle_objective"
            ]
        )

        absolute_regret = float(
            best_oracle_value
            -
            oracle_of_private_best
        )

        relative_regret = float(
            absolute_regret
            /
            max(
                abs(
                    best_oracle_value
                ),
                NUMERICAL_EPS,
            )
        )

        rows.append({
            "iteration": int(
                current[
                    "iteration"
                ]
            ),

            "current_private_objective": (
                private_value
            ),

            "current_oracle_objective": (
                oracle_value
            ),

            "best_private_objective_so_far": float(
                best_private_value
            ),

            "private_selected_iteration_so_far": int(
                private_selected_row[
                    "iteration"
                ]
            ),

            "oracle_of_private_best_so_far": float(
                oracle_of_private_best
            ),

            "best_oracle_objective_so_far": float(
                best_oracle_value
            ),

            "oracle_best_iteration_so_far": int(
                oracle_selected_row[
                    "iteration"
                ]
            ),

            "oracle_regret_of_private_selection": float(
                absolute_regret
            ),

            "relative_oracle_regret_of_private_selection": float(
                relative_regret
            ),
        })

    return pd.DataFrame(
        rows
    )


def build_correlation_table(
    validation_df,
):


    oracle_metrics = [
        "oracle_objective",
        "oracle_total_loss",
        "oracle_degree_loss",
        "oracle_relation_loss",
        "oracle_clustering_loss",
        "oracle_modularity_loss",
        "oracle_metapath_loss",
        "oracle_structural_mmd_loss",
        "oracle_node_count_relative_error",
        "oracle_edge_count_relative_error",
        "oracle_client_edge_precision",
        "oracle_client_edge_recall",
        "oracle_client_edge_f1",
        "oracle_client_edge_jaccard",
    ]

    rows = []

    for oracle_metric in oracle_metrics:
        if oracle_metric not in validation_df.columns:
            continue

        for method in ("spearman", "kendall", "pearson"):
            result = safe_correlation(
                validation_df["private_objective"],
                validation_df[oracle_metric],
                method=method,
            )

            rows.append({
                "private_metric": "private_objective",
                "oracle_metric": oracle_metric,
                "method": method,
                "n": result["n"],
                "correlation": result["correlation"],
                "p_value": result["p_value"],
                "inferential_status": "descriptive_only_adaptive_trace",
            })


    component_pairs = [
        ("degree_loss", "oracle_degree_loss"),
        ("volume_loss", "oracle_edge_count_relative_error"),
        ("client_similarity_loss", "oracle_client_edge_f1"),
        ("content_cohesion_loss", "oracle_metapath_loss"),
    ]

    for private_metric, oracle_metric in component_pairs:
        if (
            private_metric not in validation_df.columns
            or oracle_metric not in validation_df.columns
        ):
            continue

        for method in ("spearman", "kendall", "pearson"):
            result = safe_correlation(
                validation_df[private_metric],
                validation_df[oracle_metric],
                method=method,
            )

            rows.append({
                "private_metric": private_metric,
                "oracle_metric": oracle_metric,
                "method": method,
                "n": result["n"],
                "correlation": result["correlation"],
                "p_value": result["p_value"],
                "inferential_status": "descriptive_only_adaptive_trace",
            })

    return pd.DataFrame(rows)


def compute_selection_statistics(
    validation_df,
):


    if validation_df.empty:
        raise ValueError("validation_df is empty.")

    private_best = validation_df.loc[
        validation_df["private_objective"].idxmax()
    ]
    oracle_best = validation_df.loc[
        validation_df["oracle_objective"].idxmax()
    ]

    private_selected_oracle = float(private_best["oracle_objective"])
    best_oracle_within_trace = float(oracle_best["oracle_objective"])

    absolute_regret, relative_regret = safe_relative_regret(
        best_value=best_oracle_within_trace,
        selected_value=private_selected_oracle,
    )

    same_configuration = bool(
        int(private_best["iteration"]) == int(oracle_best["iteration"])
    )

    private_parameters = {
        "similarity_threshold": float(private_best["similarity_threshold"]),
        "client_degree_scale": float(private_best["client_degree_scale"]),
        "content_degree_scale": float(private_best["content_degree_scale"]),
        "candidate_k": int(private_best["candidate_k"]),
    }

    oracle_within_trace_parameters = {
        "similarity_threshold": float(oracle_best["similarity_threshold"]),
        "client_degree_scale": float(oracle_best["client_degree_scale"]),
        "content_degree_scale": float(oracle_best["content_degree_scale"]),
        "candidate_k": int(oracle_best["candidate_k"]),
    }

    return {
        "selection_scope": (
            "best oracle score among configurations already proposed by "
            "the frozen private BO trace; not an independent oracle BO"
        ),
        "private_selected_iteration": int(private_best["iteration"]),
        "oracle_best_within_private_trace_iteration": int(
            oracle_best["iteration"]
        ),

        "oracle_selected_iteration": int(oracle_best["iteration"]),
        "same_selected_configuration": same_configuration,
        "best_private_objective": float(private_best["private_objective"]),
        "oracle_objective_at_private_selection": private_selected_oracle,
        "best_oracle_objective_within_private_trace": (
            best_oracle_within_trace
        ),

        "best_oracle_objective": best_oracle_within_trace,
        "private_objective_at_oracle_best_within_private_trace": float(
            oracle_best["private_objective"]
        ),

        "private_objective_at_oracle_selection": float(
            oracle_best["private_objective"]
        ),
        "within_trace_oracle_regret_of_private_selection": float(
            absolute_regret
        ),
        "relative_within_trace_oracle_regret": float(relative_regret),

        "oracle_selection_regret": float(absolute_regret),
        "relative_oracle_selection_regret": float(relative_regret),
        "private_selected_parameters": private_parameters,
        "oracle_best_within_private_trace_parameters": (
            oracle_within_trace_parameters
        ),

        "oracle_selected_parameters": oracle_within_trace_parameters,
    }


def save_selected_graphs(
    validation_df,
    client_ids,
    private_signatures,
    private_targets,
    candidate_indices,
    candidate_scores,
    content_capacity,
    validation_folder,
):


    private_best = validation_df.loc[
        validation_df["private_objective"].idxmax()
    ]
    oracle_best = validation_df.loc[
        validation_df["oracle_objective"].idxmax()
    ]

    selections = {
        "private_selected": private_best,
        "oracle_best_within_private_trace": oracle_best,
    }

    paths = {}

    for name, row in selections.items():
        parameters = parameters_from_trace_row(row)

        reconstruction = reconstruct_graph_from_private_reports(
            client_ids=client_ids,
            private_signatures=private_signatures,
            private_targets=private_targets,
            candidate_indices=candidate_indices,
            candidate_scores=candidate_scores,
            parameters=parameters,
            content_capacity=content_capacity,
        )

        graph = reconstruction["graph"]
        graph.graph["validation_selection"] = name
        graph.graph["bo_iteration"] = int(row["iteration"])
        graph.graph["oracle_selection_scope"] = (
            "within_frozen_private_trace_only"
        )

        out_path = os.path.join(
            validation_folder,
            f"{name}_graph.gml.gz",
        )
        nx.write_gml(graph, out_path)
        paths[name] = out_path

    return paths


def validate_private_bo(
    dataset_name,
    private_run_folder,
    output_folder="output/",
    seed=DEFAULT_SEED,
    top_k_values=DEFAULT_TOP_K_VALUES,
    max_evaluations=None,
    save_graphs=True,
    replay_tolerance=DEFAULT_REPLAY_TOLERANCE,
    allow_candidate_failures=False,
    force_rebuild=False,
):


    dataset_name = str(dataset_name)
    private_run_folder = os.path.normpath(str(private_run_folder))
    output_folder = os.path.normpath(str(output_folder))
    seed = int(seed)
    replay_tolerance = float(replay_tolerance)

    if replay_tolerance < 0:
        raise ValueError("replay_tolerance must be non-negative.")

    top_k_values = tuple(int(value) for value in top_k_values)
    if any(value <= 0 for value in top_k_values):
        raise ValueError("All top-k values must be positive.")

    validation_folder = os.path.join(
        private_run_folder,
        "synthesis",
        "bo_validation",
    )
    os.makedirs(validation_folder, exist_ok=True)

    raw_path = os.path.join(
        validation_folder,
        "private_oracle_validation.csv",
    )
    convergence_path = os.path.join(
        validation_folder,
        "private_oracle_convergence.csv",
    )
    correlation_path = os.path.join(
        validation_folder,
        "private_oracle_metric_correlations.csv",
    )
    topk_path = os.path.join(
        validation_folder,
        "private_oracle_topk_overlap.csv",
    )
    ranking_path = os.path.join(
        validation_folder,
        "private_oracle_ranking.csv",
    )
    summary_path = os.path.join(
        validation_folder,
        "private_oracle_validation_summary.json",
    )

    trace_path = os.path.join(
        private_run_folder,
        "synthesis",
        "private_bo_trace.csv",
    )
    bo_summary_path = os.path.join(
        private_run_folder,
        "synthesis",
        "private_bo_summary.json",
    )

    if not os.path.exists(trace_path):
        raise FileNotFoundError(f"Missing private BO trace: {trace_path}")
    if not os.path.exists(bo_summary_path):
        raise FileNotFoundError(f"Missing private BO summary: {bo_summary_path}")

    experiment_start = time.perf_counter()


    reports = load_private_reports(private_run_folder)

    client_ids = np.asarray(reports["client_ids"], dtype=str)
    private_signatures = np.asarray(
        reports["private_signatures"],
        dtype=float,
    )
    private_degrees = np.asarray(
        reports["private_degrees"],
        dtype=int,
    )
    component_names = [
        str(value)
        for value in reports["degree_component_names"]
    ]
    configuration = reports["metadata"]["configuration"]
    degree_mode = str(configuration["degree_mode"])

    if (
        len(client_ids) != private_signatures.shape[0]
        or len(client_ids) != private_degrees.shape[0]
    ):
        raise ValueError("Private report arrays have inconsistent lengths.")

    bo_summary = load_bo_summary(private_run_folder)

    if bo_summary.get("source") != "private_reports_only":
        raise RuntimeError(
            "BO summary does not declare source='private_reports_only'. "
            "Refusing oracle validation because the synthesis provenance "
            "cannot be verified."
        )

    summary_privacy_configuration = bo_summary.get(
        "privacy_run_configuration"
    )
    if summary_privacy_configuration != configuration:
        raise RuntimeError(
            "Private report configuration and BO-summary privacy "
            "configuration do not match. Refusing mixed-run validation."
        )

    synthesis_configuration = bo_summary.get(
        "synthesis_configuration",
        {},
    )
    synthesis_configuration_hash = bo_summary.get(
        "synthesis_configuration_hash"
    )

    if not synthesis_configuration_hash:
        raise RuntimeError(
            "Missing synthesis_configuration_hash. The final RQ5 validator "
            "requires a configuration-aware synthesis artifact."
        )

    degree_cap = synthesis_configuration.get("degree_cap")
    if degree_cap is None:
        raise RuntimeError(
            "Missing degree_cap in the synthesis configuration. RQ5 must "
            "replay the same D_max projection used during private BO."
        )
    degree_cap = int(degree_cap)
    if degree_cap <= 0:
        raise ValueError("Synthesis degree_cap must be positive.")

    recorded_postprocessing = bo_summary.get(
        "private_degree_postprocessing",
        {},
    )
    recorded_degree_cap = recorded_postprocessing.get("degree_cap")
    if (
        recorded_degree_cap is not None
        and int(recorded_degree_cap) != degree_cap
    ):
        raise RuntimeError(
            "BO summary contains inconsistent degree-cap metadata."
        )

    content_capacity = int(bo_summary["content_capacity"])
    max_candidates = int(bo_summary["max_candidates_per_client"])
    candidate_search = str(bo_summary.get("candidate_search", ""))

    if candidate_search != "exact_blockwise_cosine":
        raise RuntimeError(
            "This validator rebuilds the exact blockwise candidate pool, "
            f"but the frozen synthesis used candidate_search={candidate_search!r}."
        )


    private_targets = extract_private_targets(
        private_degrees=private_degrees,
        component_names=component_names,
        degree_mode=degree_mode,
        degree_cap=degree_cap,
    )

    print(
        "[BO validation] Rebuilding frozen PRIVATE candidate pool "
        f"(Dmax={degree_cap})..."
    )
    candidate_indices, candidate_scores = compute_exact_top_candidates(
        private_signatures=private_signatures,
        max_candidates=max_candidates,
    )

    full_trace_df = load_bo_trace(private_run_folder)
    full_trace_df = (
        full_trace_df
        .sort_values("iteration")
        .reset_index(drop=True)
    )

    expected_evaluations = int(bo_summary["bo_evaluations"])
    if len(full_trace_df) != expected_evaluations:
        raise RuntimeError(
            "Frozen BO trace length does not match bo_summary: "
            f"trace={len(full_trace_df)}, summary={expected_evaluations}."
        )

    summary_best_private = float(bo_summary["best_private_objective"])
    trace_best_private = float(full_trace_df["private_objective"].max())
    if abs(summary_best_private - trace_best_private) > replay_tolerance:
        raise RuntimeError(
            "BO summary best_private_objective does not match the frozen "
            "trace maximum."
        )

    if max_evaluations is not None:
        max_evaluations = int(max_evaluations)
        if max_evaluations <= 0:
            raise ValueError("max_evaluations must be positive.")
        trace_df = full_trace_df.iloc[:max_evaluations].copy()
    else:
        trace_df = full_trace_df.copy()


    validation_configuration = {
        "schema_version": BO_VALIDATION_SCHEMA_VERSION,
        "dataset": dataset_name,
        "oracle_evaluation_seed": seed,
        "oracle_seed_policy": "same_fixed_seed_for_every_trace_candidate",
        "top_k_values": list(top_k_values),
        "max_evaluations": max_evaluations,
        "save_graphs": bool(save_graphs),
        "allow_candidate_failures": bool(allow_candidate_failures),
        "replay_tolerance": replay_tolerance,
        "private_bo_trace_sha256": file_sha256(trace_path),
        "synthesis_configuration_hash": synthesis_configuration_hash,
        "privacy_configuration_hash": stable_configuration_hash(
            configuration
        ),
        "degree_cap": degree_cap,
        "candidate_search": candidate_search,
        "max_candidates_per_client": max_candidates,
        "content_capacity": content_capacity,
    }
    validation_configuration_hash = stable_configuration_hash(
        validation_configuration
    )

    expected_selected_graphs = [
        os.path.join(validation_folder, "private_selected_graph.gml.gz"),
        os.path.join(
            validation_folder,
            "oracle_best_within_private_trace_graph.gml.gz",
        ),
    ]

    required_files = [
        raw_path,
        convergence_path,
        correlation_path,
        topk_path,
        ranking_path,
        summary_path,
    ]
    if save_graphs:
        required_files.extend(expected_selected_graphs)

    if (
        not force_rebuild
        and all(os.path.exists(path) for path in required_files)
    ):
        try:
            with open(summary_path, "r", encoding="utf-8") as handle:
                cached_summary = json.load(handle)
        except (OSError, json.JSONDecodeError):
            cached_summary = None

        if (
            isinstance(cached_summary, dict)
            and cached_summary.get("validation_configuration_hash")
            == validation_configuration_hash
        ):
            print(
                "[BO validation] Using configuration-matched cached "
                "validation."
            )
            return {
                "validation": pd.read_csv(raw_path),
                "convergence": pd.read_csv(convergence_path),
                "correlations": pd.read_csv(correlation_path),
                "top_k": pd.read_csv(topk_path),
                "ranking": pd.read_csv(ranking_path),
                "summary": cached_summary,
            }

        print(
            "[BO validation] Existing validation cache does not match "
            "the frozen trace/configuration; rebuilding."
        )

    print(
        "[BO validation] Private trace is frozen. Oracle graph will now "
        "be loaded for post-hoc evaluation only."
    )


    original_graph = load_original_graph(
        dataset_name=dataset_name,
        output_folder=output_folder,
    )

    print(
        "[BO validation] Computing original graph oracle statistics "
        f"with fixed seed={seed}..."
    )
    original_stats = calculate_graph_statistics(
        original_graph,
        seed=seed,
    )

    validation_rows = []
    failures = []
    total = len(trace_df)

    for position, (_, trace_row) in enumerate(trace_df.iterrows()):
        iteration = int(trace_row["iteration"])
        print(
            "[BO validation] "
            f"{position + 1}/{total} iteration={iteration}"
        )
        evaluation_start = time.perf_counter()

        try:
            parameters = parameters_from_trace_row(trace_row)


            reconstruction = reconstruct_graph_from_private_reports(
                client_ids=client_ids,
                private_signatures=private_signatures,
                private_targets=private_targets,
                candidate_indices=candidate_indices,
                candidate_scores=candidate_scores,
                parameters=parameters,
                content_capacity=content_capacity,
            )
            synthetic_graph = reconstruction["graph"]

            recomputed_private = compute_private_objective(
                reconstruction=reconstruction,
                private_targets=private_targets,
                private_signatures=private_signatures,
            )

            trace_private_objective = float(
                trace_row["private_objective"]
            )
            recomputed_private_objective = float(
                recomputed_private["private_objective"]
            )
            private_replay_difference = float(
                abs(
                    trace_private_objective
                    - recomputed_private_objective
                )
            )


            if private_replay_difference > replay_tolerance:
                raise PrivateReplayMismatchError(
                    "Private BO replay mismatch at iteration "
                    f"{iteration}: trace={trace_private_objective:.15g}, "
                    f"replayed={recomputed_private_objective:.15g}, "
                    f"abs_diff={private_replay_difference:.15g}, "
                    f"tolerance={replay_tolerance:.15g}."
                )


            oracle = evaluate_one_candidate_oracle(
                original_graph=original_graph,
                original_stats=original_stats,
                synthetic_graph=synthetic_graph,
                seed=seed,
            )

            runtime = float(
                time.perf_counter() - evaluation_start
            )

            row = {
                "dataset": dataset_name,
                "iteration": iteration,
                "phase": str(trace_row.get("phase", "unknown")),
                **parameters,
                "private_objective": trace_private_objective,
                "recomputed_private_objective": (
                    recomputed_private_objective
                ),
                "private_objective_replay_abs_difference": (
                    private_replay_difference
                ),
                "degree_loss": float(
                    recomputed_private["degree_loss"]
                ),
                "client_degree_loss": float(
                    recomputed_private["client_degree_loss"]
                ),
                "content_degree_loss": float(
                    recomputed_private["content_degree_loss"]
                ),
                "volume_loss": float(
                    recomputed_private["volume_loss"]
                ),
                "client_similarity_loss": float(
                    recomputed_private["client_similarity_loss"]
                ),
                "content_cohesion_loss": float(
                    recomputed_private["content_cohesion_loss"]
                ),
                "synthetic_nodes": int(
                    synthetic_graph.number_of_nodes()
                ),
                "synthetic_edges": int(
                    synthetic_graph.number_of_edges()
                ),
                "oracle_evaluation_seed": seed,
                "oracle_evaluation_runtime_seconds": runtime,
                **oracle,
            }
            validation_rows.append(row)

        except PrivateReplayMismatchError:
            raise

        except Exception as exc:
            failure = {
                "iteration": iteration,
                "error": repr(exc),
            }
            failures.append(failure)
            print("[BO validation] FAILED:")
            print(repr(exc))

            if not allow_candidate_failures:
                raise RuntimeError(
                    "Oracle validation failed for a frozen BO candidate. "
                    "Final RQ5 validation fails closed rather than ranking "
                    "a partial trace. Use --allow-candidate-failures only "
                    "for debugging."
                ) from exc

    validation_df = pd.DataFrame(validation_rows)
    if validation_df.empty:
        raise RuntimeError(
            "No BO configurations were successfully oracle-evaluated."
        )

    if (
        not allow_candidate_failures
        and len(validation_df) != len(trace_df)
    ):
        raise RuntimeError(
            "Not every frozen BO configuration was oracle-evaluated."
        )

    validation_df = add_ranking_columns(validation_df)
    validation_df.to_csv(raw_path, index=False)

    ranking_columns = [
        "iteration",
        "private_objective",
        "oracle_objective",
        "private_rank",
        "oracle_rank",
        "rank_displacement",
        "similarity_threshold",
        "client_degree_scale",
        "content_degree_scale",
        "candidate_k",
    ]
    ranking_df = (
        validation_df[ranking_columns]
        .sort_values("private_rank")
        .reset_index(drop=True)
    )
    ranking_df.to_csv(ranking_path, index=False)

    correlation_df = build_correlation_table(validation_df)
    correlation_df.to_csv(correlation_path, index=False)

    topk_df = compute_top_k_overlap(
        validation_df,
        k_values=top_k_values,
    )
    topk_df.to_csv(topk_path, index=False)

    convergence_df = compute_convergence_table(validation_df)
    convergence_df.to_csv(convergence_path, index=False)

    selection = compute_selection_statistics(validation_df)

    spearman = safe_correlation(
        validation_df["private_objective"],
        validation_df["oracle_objective"],
        method="spearman",
    )
    kendall = safe_correlation(
        validation_df["private_objective"],
        validation_df["oracle_objective"],
        method="kendall",
    )
    pearson = safe_correlation(
        validation_df["private_objective"],
        validation_df["oracle_objective"],
        method="pearson",
    )

    maximum_replay_difference = float(
        validation_df[
            "private_objective_replay_abs_difference"
        ].max()
    )
    mean_replay_difference = float(
        validation_df[
            "private_objective_replay_abs_difference"
        ].mean()
    )

    if maximum_replay_difference > replay_tolerance:
        raise PrivateReplayMismatchError(
            "Maximum private-objective replay difference exceeded the "
            "configured tolerance."
        )

    selected_graph_paths = {}
    if save_graphs:
        selected_graph_paths = save_selected_graphs(
            validation_df=validation_df,
            client_ids=client_ids,
            private_signatures=private_signatures,
            private_targets=private_targets,
            candidate_indices=candidate_indices,
            candidate_scores=candidate_scores,
            content_capacity=content_capacity,
            validation_folder=validation_folder,
        )

    total_runtime = float(time.perf_counter() - experiment_start)

    top_k_summary = {}
    for _, row in topk_df.iterrows():
        key = f"top_{int(row['effective_k'])}"
        top_k_summary[key] = {
            "overlap_fraction": float(row["overlap_fraction"]),
            "jaccard": float(row["jaccard"]),
            "intersection_size": int(row["intersection_size"]),
        }

    summary = {
        "schema_version": BO_VALIDATION_SCHEMA_VERSION,
        "dataset": dataset_name,
        "private_run_folder": private_run_folder,
        "seed": seed,
        "oracle_evaluation_seed": seed,
        "oracle_seed_policy": (
            "same fixed seed for every candidate in the frozen trace"
        ),
        "privacy_configuration": configuration,
        "synthesis_configuration_hash": synthesis_configuration_hash,
        "validation_configuration": validation_configuration,
        "validation_configuration_hash": validation_configuration_hash,
        "private_bo_trace_sha256": validation_configuration[
            "private_bo_trace_sha256"
        ],
        "degree_cap_replayed": degree_cap,
        "private_degree_postprocessing_replayed": private_targets.get(
            "degree_postprocessing"
        ),
        "evaluated_configurations": int(len(validation_df)),
        "failed_configurations": int(len(failures)),
        "failures": failures,
        "private_vs_oracle_objective": {
            "spearman": {
                "correlation": spearman["correlation"],
                "p_value": spearman["p_value"],
                "n": spearman["n"],
            },
            "kendall_tau": {
                "correlation": kendall["correlation"],
                "p_value": kendall["p_value"],
                "n": kendall["n"],
            },
            "pearson": {
                "correlation": pearson["correlation"],
                "p_value": pearson["p_value"],
                "n": pearson["n"],
            },
            "inferential_note": (
                "BO configurations are adaptively proposed; correlation "
                "p-values are descriptive diagnostics, not IID inferential "
                "tests."
            ),
        },
        "selection": selection,
        "oracle_best_scope": (
            "best oracle-scoring configuration within the frozen private "
            "BO trace; not an independently optimized oracle BO"
        ),
        "top_k_overlap": top_k_summary,
        "rank_displacement": {
            "mean": float(validation_df["rank_displacement"].mean()),
            "median": float(validation_df["rank_displacement"].median()),
            "max": float(validation_df["rank_displacement"].max()),
        },
        "private_objective_replay_check": {
            "mean_absolute_difference": mean_replay_difference,
            "maximum_absolute_difference": maximum_replay_difference,
            "tolerance": replay_tolerance,
            "passed": bool(
                maximum_replay_difference <= replay_tolerance
            ),
        },
        "oracle_is_used_by_bo": False,
        "oracle_is_used_by_reconstruction": False,
        "privacy_boundary": {
            "private_trace_frozen_before_original_graph_load": True,
            "candidate_pool_from_private_signatures_only": True,
            "private_targets_from_private_degrees_only": True,
            "degree_projection_is_postprocessing": True,
            "oracle_feedback_to_acquisition": False,
            "oracle_feedback_to_parameter_selection": False,
        },
        "validation_protocol": [
            (
                "The private BO trace is completed and frozen before the "
                "original graph is loaded by this validator."
            ),
            (
                "Every BO configuration is replayed from the same private "
                "reports and the exact same public D_max projection."
            ),
            (
                "The private objective must reproduce the frozen trace "
                "within the configured numerical tolerance before oracle "
                "evaluation is accepted."
            ),
            (
                "Every candidate uses the same fixed oracle evaluation seed "
                "to avoid iteration-specific MMD/statistics noise."
            ),
            (
                "Oracle objective values never influence BO acquisition, "
                "reconstruction, or the released synthetic graph."
            ),
            (
                "Oracle regret is within-trace regret, not regret against an "
                "independently oracle-optimized search."
            ),
        ],
        "selected_graphs": selected_graph_paths,
        "runtime_seconds": total_runtime,
        "outputs": {
            "raw_validation": os.path.basename(raw_path),
            "ranking": os.path.basename(ranking_path),
            "correlations": os.path.basename(correlation_path),
            "top_k_overlap": os.path.basename(topk_path),
            "convergence": os.path.basename(convergence_path),
        },
    }

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(
            summary,
            handle,
            indent=4,
            ensure_ascii=False,
        )

    print("\n========================================")
    print("PRIVATE-vs-ORACLE BO VALIDATION")
    print("========================================")
    print(f"Dataset: {dataset_name}")
    print(f"Configurations evaluated: {len(validation_df)}")
    print(f"Failures: {len(failures)}")
    print(f"Replayed Dmax: {degree_cap}")
    print(f"Fixed oracle evaluation seed: {seed}")

    print("\nPrivate objective vs oracle objective:")
    print(
        f"  Spearman rho: {spearman['correlation']:.4f}"
        if np.isfinite(spearman["correlation"])
        else "  Spearman rho: NaN"
    )
    print(
        f"  Kendall tau: {kendall['correlation']:.4f}"
        if np.isfinite(kendall["correlation"])
        else "  Kendall tau: NaN"
    )
    print(
        f"  Pearson r: {pearson['correlation']:.4f}"
        if np.isfinite(pearson["correlation"])
        else "  Pearson r: NaN"
    )

    print("\nSelection within the frozen private BO trace:")
    print(
        "  Private-selected iteration: "
        f"{selection['private_selected_iteration']}"
    )
    print(
        "  Best-oracle-within-trace iteration: "
        f"{selection['oracle_best_within_private_trace_iteration']}"
    )
    print(
        "  Same configuration: "
        f"{selection['same_selected_configuration']}"
    )
    print(
        "  Oracle score at private selection: "
        f"{selection['oracle_objective_at_private_selection']:.6f}"
    )
    print(
        "  Best oracle score within trace: "
        f"{selection['best_oracle_objective_within_private_trace']:.6f}"
    )
    print(
        "  Within-trace oracle regret: "
        f"{selection['within_trace_oracle_regret_of_private_selection']:.6f}"
    )
    print(
        "  Relative within-trace regret: "
        f"{selection['relative_within_trace_oracle_regret']:.4f}"
    )

    print("\nTop-K overlap:")
    for _, row in topk_df.iterrows():
        print(
            f"  Top-{int(row['effective_k'])}: "
            f"{row['overlap_fraction']:.4f}"
        )

    print("\nPrivate objective replay:")
    print(
        "  Maximum absolute difference: "
        f"{maximum_replay_difference:.12g}"
    )
    print(f"  Tolerance: {replay_tolerance:.12g}")

    print("\nOutputs:")
    print(f"  Raw validation:\n    {raw_path}")
    print(f"  Ranking:\n    {ranking_path}")
    print(f"  Correlations:\n    {correlation_path}")
    print(f"  Top-K overlap:\n    {topk_path}")
    print(f"  Convergence:\n    {convergence_path}")
    print(f"  Summary:\n    {summary_path}")
    print("========================================\n")

    return {
        "validation": validation_df,
        "ranking": ranking_df,
        "correlations": correlation_df,
        "top_k": topk_df,
        "convergence": convergence_df,
        "summary": summary,
    }


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Validate the private Bayesian-optimization "
            "objective against post-hoc oracle graph fidelity."
        )
    )

    parser.add_argument(
        "dataset",
        type=str,
    )

    parser.add_argument(
        "private_run_folder",
        type=str,
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default="output/",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--top-k",
        nargs="+",
        type=int,
        default=list(
            DEFAULT_TOP_K_VALUES
        ),
    )

    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=None,
        help=(
            "Debugging only. "
            "For the final paper evaluate the complete BO trace."
        ),
    )

    parser.add_argument(
        "--no-save-graphs",
        action="store_true",
    )

    parser.add_argument(
        "--replay-tolerance",
        type=float,
        default=DEFAULT_REPLAY_TOLERANCE,
        help=(
            "Maximum allowed absolute difference between the frozen "
            "private BO objective and its deterministic replay."
        ),
    )

    parser.add_argument(
        "--allow-candidate-failures",
        action="store_true",
        help=(
            "Debugging only. Final RQ5 validation should fail closed if "
            "any frozen BO candidate cannot be replayed/evaluated."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    validate_private_bo(
        dataset_name=(
            args.dataset
        ),

        private_run_folder=(
            args.private_run_folder
        ),

        output_folder=(
            args.output_folder
        ),

        seed=args.seed,

        top_k_values=(
            args.top_k
        ),

        max_evaluations=(
            args.max_evaluations
        ),

        save_graphs=(
            not args.no_save_graphs
        ),

        replay_tolerance=(
            args.replay_tolerance
        ),

        allow_candidate_failures=(
            args.allow_candidate_failures
        ),

        force_rebuild=(
            args.force
        ),
    )
