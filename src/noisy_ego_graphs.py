import hashlib
import json
import os
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm

import local_signatures as _local_signatures
import privacy_mechanisms as _privacy_mechanisms

from local_signatures import (
    compute_local_signatures,
    load_local_signatures,
)
from privacy_mechanisms import (
    DEFAULT_D_MAX,
    DEFAULT_SIGNATURE_CLIP_NORM,
    PRIVACY_SCHEMA_VERSION,
    clip_l2,
    extract_typed_degree_vector,
    privatize_client_report,
    split_privacy_budget,
)


PRIVATE_REPORT_SCHEMA_VERSION = "2.1-simulation-only"

DEFAULT_SIGNATURE_METHOD = "hks"
DEFAULT_SIGNATURE_MECHANISM = "l2_knorm"

DEFAULT_SIGNATURE_FRACTION = 0.5

DEFAULT_DEGREE_MODE = "role"
DEFAULT_DEGREE_ADJACENCY = "client_report"

DEFAULT_SEED = 0

DEFAULT_K_VALUES = (
    10,
    20,
    50,
)

DEFAULT_MAX_NEIGHBOUR_CLIENTS = 2000

NUMERICAL_EPS = 1e-12


SIMULATION_ONLY = True


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def implementation_fingerprints():

    return {
        "noisy_ego_graphs.py": _sha256_file(__file__),
        "local_signatures.py": _sha256_file(_local_signatures.__file__),
        "privacy_mechanisms.py": _sha256_file(_privacy_mechanisms.__file__),
    }


def input_archive_fingerprints(dataset_name, input_folder):

    dataset_dir = Path(input_folder) / dataset_name
    files = {
        "local_signatures_npz": dataset_dir / f"{dataset_name}_local_signatures.npz",
        "ego_graph_index_csv_gz": dataset_dir / f"{dataset_name}_ego_graph_index.csv.gz",
    }
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "V2 refuses to create or alter original input files. Missing: "
            + ", ".join(missing)
        )
    return {name: _sha256_file(path) for name, path in files.items()}


def validate_separate_roots(input_folder, reports_output_folder):

    if reports_output_folder is None:
        raise ValueError(
            "V2 requires reports_output_folder (CLI: --reports-output-folder). "
            "Never generate V2 reports in the historical input folder."
        )
    original = Path(input_folder).resolve()
    reports = Path(reports_output_folder).resolve()
    if original == reports or original in reports.parents or reports in original.parents:
        raise ValueError(
            "Input and V2 report output folders must be separate sibling trees: "
            f"input={original}; reports={reports}"
        )
    return reports


def numeric_summary(values):

    values = np.asarray(
        list(values),
        dtype=float,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(values) == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
        }

    return {
        "count": int(
            len(
                values
            )
        ),

        "mean": float(
            np.mean(
                values
            )
        ),

        "std": float(
            np.std(
                values
            )
        ),

        "median": float(
            np.median(
                values
            )
        ),

        "min": float(
            np.min(
                values
            )
        ),

        "max": float(
            np.max(
                values
            )
        ),

        "p25": float(
            np.percentile(
                values,
                25,
            )
        ),

        "p75": float(
            np.percentile(
                values,
                75,
            )
        ),

        "p90": float(
            np.percentile(
                values,
                90,
            )
        ),

        "p95": float(
            np.percentile(
                values,
                95,
            )
        ),

        "p99": float(
            np.percentile(
                values,
                99,
            )
        ),
    }


def safe_float_token(value):

    text = format(
        float(
            value
        ),
        ".12g",
    )

    return (
        text
        .replace(
            "-",
            "m",
        )
        .replace(
            ".",
            "p",
        )
        .replace(
            "+",
            "",
        )
    )


def canonical_json(data):

    return json.dumps(
        data,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=True,
    )


def configuration_hash(config):

    return hashlib.sha256(
        canonical_json(
            config
        ).encode(
            "utf-8"
        )
    ).hexdigest()[:12]


def deterministic_client_seed(
    base_seed,
    dataset_name,
    client_id,
    epsilon,
    signature_method,
    signature_mechanism,
    signature_fraction,
    degree_mode,
):

    text = "|".join(
        [
            str(
                base_seed
            ),
            str(
                dataset_name
            ),
            str(
                client_id
            ),
            format(
                float(
                    epsilon
                ),
                ".17g",
            ),
            str(
                signature_method
            ),
            str(
                signature_mechanism
            ),
            format(
                float(
                    signature_fraction
                ),
                ".17g",
            ),
            str(
                degree_mode
            ),
        ]
    )

    digest = hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).digest()


    return int.from_bytes(
        digest[:4],
        byteorder="little",
        signed=False,
    )


def safe_cosine(
    a,
    b,
):

    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    norm_a = np.linalg.norm(
        a
    )

    norm_b = np.linalg.norm(
        b
    )

    if (
        norm_a <= NUMERICAL_EPS
        or norm_b <= NUMERICAL_EPS
    ):
        return 0.0

    return float(
        np.dot(
            a,
            b,
        )
        /
        (
            norm_a
            * norm_b
        )
    )


def load_or_compute_signatures(
    dataset_name,
    output_folder,
):

    try:

        return load_local_signatures(
            dataset_name=dataset_name,
            output_folder=output_folder,
        )

    except FileNotFoundError:

        compute_local_signatures(
            dataset_name=dataset_name,
            output_folder=output_folder,
        )

        return load_local_signatures(
            dataset_name=dataset_name,
            output_folder=output_folder,
        )


def load_ego_index(
    dataset_name,
    output_folder,
):

    path = os.path.join(
        output_folder,
        dataset_name,
        f"{dataset_name}_ego_graph_index.csv.gz",
    )

    if not os.path.exists(
        path
    ):
        raise FileNotFoundError(
            "Missing revised ego index: "
            f"{path}. "
            "Run create_ego_graphs() first."
        )

    return pd.read_csv(
        path,
        compression="gzip",
        dtype={
            "client_id": str,
            "ego_filename": str,
        },
    )


def load_global_statistics(
    dataset_name,
    output_folder,
):

    path = os.path.join(
        output_folder,
        dataset_name,
        f"{dataset_name}_global_graph_stats.json",
    )

    if not os.path.exists(
        path
    ):
        return {}

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(
            f
        )


def resolve_degree_components(
    dataset_name,
    output_folder,
    degree_mode,
    degree_components=None,
):

    degree_mode = str(
        degree_mode
    ).lower()

    if degree_components is not None:

        components = [
            str(
                component
            )
            for component
            in degree_components
        ]

        if not components:
            raise ValueError(
                "degree_components cannot be empty."
            )

        return components

    if degree_mode == "role":

        return [
            "client",
            "content",
        ]

    global_stats = load_global_statistics(
        dataset_name,
        output_folder,
    )

    if degree_mode == "type":

        counts = global_stats.get(
            "node_type_counts",
            {},
        )

        components = sorted(
            str(
                key
            )
            for key
            in counts.keys()
        )

        if not components:
            raise ValueError(
                "Could not determine global node types. "
                "Rebuild the graph with create_global.py "
                "or provide degree_components explicitly."
            )

        return components

    if degree_mode == "relation":

        counts = global_stats.get(
            "relation_counts",
            {},
        )

        components = sorted(
            str(
                key
            )
            for key
            in counts.keys()
        )

        if not components:
            raise ValueError(
                "Could not determine global relation types. "
                "Rebuild the graph with create_global.py "
                "or provide degree_components explicitly."
            )

        return components

    raise ValueError(
        "degree_mode must be one of "
        "'role', 'type', or 'relation'."
    )


def select_signature_array(
    signatures,
    signature_method,
):

    signature_method = str(
        signature_method
    ).lower()

    if signature_method not in {
        "raw",
        "abs",
        "hks",
    }:
        raise ValueError(
            "signature_method must be one of "
            "'raw', 'abs', or 'hks'."
        )

    return np.asarray(
        signatures[
            signature_method
        ],
        dtype=float,
    )


def compute_signature_error_statistics(
    clean_signature,
    private_signature,
    clip_norm,
):

    clean_signature = np.asarray(
        clean_signature,
        dtype=float,
    )

    private_signature = np.asarray(
        private_signature,
        dtype=float,
    )

    clipped_signature, clip_info = clip_l2(
        clean_signature,
        C=clip_norm,
    )

    clean_difference = (
        private_signature
        - clean_signature
    )

    clipped_difference = (
        private_signature
        - clipped_signature
    )

    clean_l2 = float(
        np.linalg.norm(
            clean_difference
        )
    )

    clipped_l2 = float(
        np.linalg.norm(
            clipped_difference
        )
    )

    clean_norm = float(
        np.linalg.norm(
            clean_signature
        )
    )

    clipped_norm = float(
        np.linalg.norm(
            clipped_signature
        )
    )

    return {
        "clean_signature_l2_norm": float(
            clean_norm
        ),

        "clipped_signature_l2_norm": float(
            clipped_norm
        ),

        "private_signature_l2_norm": float(
            np.linalg.norm(
                private_signature
            )
        ),

        "signature_l2_error_clean": float(
            clean_l2
        ),

        "signature_relative_l2_error_clean": float(
            clean_l2
            /
            max(
                clean_norm,
                NUMERICAL_EPS,
            )
        ),

        "signature_l2_error_clipped": float(
            clipped_l2
        ),

        "signature_relative_l2_error_clipped": float(
            clipped_l2
            /
            max(
                clipped_norm,
                NUMERICAL_EPS,
            )
        ),

        "signature_cosine_clean_private": float(
            safe_cosine(
                clean_signature,
                private_signature,
            )
        ),

        "signature_cosine_clipped_private": float(
            safe_cosine(
                clipped_signature,
                private_signature,
            )
        ),

        "signature_was_clipped": bool(
            clip_info[
                "was_clipped"
            ]
        ),

        "signature_clipping_factor": float(
            clip_info[
                "clipping_factor"
            ]
        ),
    }


def compute_degree_error_statistics(
    raw_degree,
    clipped_degree,
    private_degree,
):

    raw_degree = np.asarray(
        raw_degree,
        dtype=float,
    )

    clipped_degree = np.asarray(
        clipped_degree,
        dtype=float,
    )

    private_degree = np.asarray(
        private_degree,
        dtype=float,
    )

    raw_error = (
        private_degree
        - raw_degree
    )

    clipped_error = (
        private_degree
        - clipped_degree
    )

    q = len(
        raw_degree
    )

    return {
        "degree_components": int(
            q
        ),

        "degree_raw_sum": float(
            np.sum(
                raw_degree
            )
        ),

        "degree_clipped_sum": float(
            np.sum(
                clipped_degree
            )
        ),

        "degree_private_sum": float(
            np.sum(
                private_degree
            )
        ),

        "degree_l1_error_raw": float(
            np.linalg.norm(
                raw_error,
                ord=1,
            )
        ),

        "degree_l2_error_raw": float(
            np.linalg.norm(
                raw_error,
                ord=2,
            )
        ),

        "degree_mae_raw": float(
            np.mean(
                np.abs(
                    raw_error
                )
            )
            if q
            else 0.0
        ),

        "degree_rmse_raw": float(
            np.sqrt(
                np.mean(
                    raw_error ** 2
                )
            )
            if q
            else 0.0
        ),

        "degree_relative_l1_error_raw": float(
            np.linalg.norm(
                raw_error,
                ord=1,
            )
            /
            max(
                np.linalg.norm(
                    raw_degree,
                    ord=1,
                ),
                1.0,
            )
        ),

        "degree_l1_error_clipped": float(
            np.linalg.norm(
                clipped_error,
                ord=1,
            )
        ),

        "degree_l2_error_clipped": float(
            np.linalg.norm(
                clipped_error,
                ord=2,
            )
        ),

        "degree_mae_clipped": float(
            np.mean(
                np.abs(
                    clipped_error
                )
            )
            if q
            else 0.0
        ),

        "degree_rmse_clipped": float(
            np.sqrt(
                np.mean(
                    clipped_error ** 2
                )
            )
            if q
            else 0.0
        ),

        "degree_exact_components_raw": int(
            np.sum(
                private_degree
                == raw_degree
            )
        ),

        "degree_exact_component_fraction_raw": float(
            np.mean(
                private_degree
                == raw_degree
            )
            if q
            else 0.0
        ),

        "degree_exact_components_clipped": int(
            np.sum(
                private_degree
                == clipped_degree
            )
        ),

        "degree_exact_component_fraction_clipped": float(
            np.mean(
                private_degree
                == clipped_degree
            )
            if q
            else 0.0
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

    return np.divide(
        matrix,
        norms,
        out=np.zeros_like(
            matrix
        ),
        where=(
            norms
            > NUMERICAL_EPS
        ),
    )


def cosine_similarity_matrix(
    signatures,
):

    normalized = normalize_rows(
        signatures
    )

    similarity = (
        normalized
        @ normalized.T
    )

    np.fill_diagonal(
        similarity,
        -np.inf,
    )

    return similarity


def top_k_indices(
    similarity,
    k,
):

    n = similarity.shape[
        0
    ]

    if n <= 1:

        return np.empty(
            (
                n,
                0,
            ),
            dtype=int,
        )

    effective_k = min(
        int(
            k
        ),
        n - 1,
    )

    if effective_k <= 0:

        return np.empty(
            (
                n,
                0,
            ),
            dtype=int,
        )

    candidates = np.argpartition(
        -similarity,
        kth=effective_k - 1,
        axis=1,
    )[
        :,
        :effective_k
    ]

    rows = np.arange(
        n
    )[
        :,
        None
    ]

    candidate_scores = similarity[
        rows,
        candidates,
    ]

    order = np.argsort(
        -candidate_scores,
        axis=1,
    )

    return np.take_along_axis(
        candidates,
        order,
        axis=1,
    )


def deterministic_sample_indices(
    client_ids,
    max_clients,
    seed,
):

    n = len(
        client_ids
    )

    if (
        max_clients is None
        or n <= max_clients
    ):
        return np.arange(
            n,
            dtype=int,
        )

    rng = np.random.default_rng(
        seed
    )

    indices = rng.choice(
        n,
        size=int(
            max_clients
        ),
        replace=False,
    )

    return np.sort(
        indices
    )


def compute_private_neighbour_stability(
    client_ids,
    clean_signatures,
    private_signatures,
    k_values=DEFAULT_K_VALUES,
    max_clients=DEFAULT_MAX_NEIGHBOUR_CLIENTS,
    seed=0,
):

    client_ids = np.asarray(
        client_ids,
        dtype=str,
    )

    clean_signatures = np.asarray(
        clean_signatures,
        dtype=float,
    )

    private_signatures = np.asarray(
        private_signatures,
        dtype=float,
    )

    indices = deterministic_sample_indices(
        client_ids,
        max_clients=max_clients,
        seed=seed,
    )

    sampled_ids = client_ids[
        indices
    ]

    clean_sample = clean_signatures[
        indices
    ]

    private_sample = private_signatures[
        indices
    ]

    if len(
        sampled_ids
    ) < 2:
        return pd.DataFrame()

    clean_similarity = cosine_similarity_matrix(
        clean_sample
    )

    private_similarity = cosine_similarity_matrix(
        private_sample
    )

    rows = [
        {
            "client_id": str(
                client_id
            )
        }
        for client_id
        in sampled_ids
    ]

    for k in k_values:

        clean_top = top_k_indices(
            clean_similarity,
            k,
        )

        private_top = top_k_indices(
            private_similarity,
            k,
        )

        effective_k = clean_top.shape[
            1
        ]

        for i in range(
            len(
                sampled_ids
            )
        ):

            if effective_k == 0:

                overlap = 1.0

            else:

                clean_set = set(
                    clean_top[
                        i
                    ].tolist()
                )

                private_set = set(
                    private_top[
                        i
                    ].tolist()
                )

                overlap = (
                    len(
                        clean_set
                        &
                        private_set
                    )
                    /
                    effective_k
                )

            rows[
                i
            ][
                f"clean_private_overlap_at_{int(k)}"
            ] = float(
                overlap
            )

    return pd.DataFrame(
        rows
    )


def build_run_configuration(
    dataset_name,
    epsilon,
    seed,
    signature_method,
    signature_mechanism,
    signature_fraction,
    signature_clip_norm,
    degree_mode,
    degree_components,
    degree_adjacency,
    d_max,
):

    epsilon_signature, epsilon_degree = (
        split_privacy_budget(
            epsilon_total=epsilon,
            signature_fraction=signature_fraction,
        )
    )

    return {
        "private_report_schema_version": (
            PRIVATE_REPORT_SCHEMA_VERSION
        ),

        "simulation_only": True,
        "randomness_mode": "deterministic_replay_not_safe_for_deployment",
        "implementation_sha256": implementation_fingerprints(),

        "privacy_mechanism_schema_version": (
            PRIVACY_SCHEMA_VERSION
        ),

        "dataset": str(
            dataset_name
        ),

        "seed": int(
            seed
        ),

        "epsilon_total": float(
            epsilon
        ),

        "signature_method": str(
            signature_method
        ),

        "signature_mechanism": str(
            signature_mechanism
        ),

        "signature_fraction": float(
            signature_fraction
        ),

        "degree_fraction": float(
            1.0
            - signature_fraction
        ),

        "epsilon_signature": float(
            epsilon_signature
        ),

        "epsilon_degree": float(
            epsilon_degree
        ),

        "signature_clip_norm_C": float(
            signature_clip_norm
        ),

        "degree_mode": str(
            degree_mode
        ),

        "degree_components": [
            str(
                value
            )
            for value
            in degree_components
        ],

        "degree_adjacency": str(
            degree_adjacency
        ),

        "D_max": int(
            d_max
        ),

        "one_private_report_per_client": True,

        "uses_personalised_epsilon": False,

        "uses_original_graph_during_server_reconstruction": False,
    }


def build_run_folder(
    dataset_name,
    output_folder,
    config,
):

    method_folder = (
        f"{config['signature_method']}"
        f"__{config['signature_mechanism']}"
    )

    run_name = (
        f"eps_{safe_float_token(config['epsilon_total'])}"
        f"__rhoS_{safe_float_token(config['signature_fraction'])}"
        f"__C_{safe_float_token(config['signature_clip_norm_C'])}"
        f"__Dmax_{config['D_max']}"
        f"__degree_{config['degree_mode']}"
        f"__adj_{config['degree_adjacency']}"
        f"__seed_{config['seed']}"
        f"__cfg_{configuration_hash(config)}"
    )

    return os.path.join(
        output_folder,
        dataset_name,
        "runs",
        "private_reports",
        method_folder,
        run_name,
    )


def cache_matches_configuration(
    metadata_path,
    config,
):

    if not os.path.exists(
        metadata_path
    ):
        return False

    try:

        with open(
            metadata_path,
            "r",
            encoding="utf-8",
        ) as f:

            metadata = json.load(
                f
            )

        stored_config = metadata.get(
            "configuration",
            {},
        )

        return (
            canonical_json(
                stored_config
            )
            ==
            canonical_json(
                config
            )
        )

    except Exception:
        return False


def build_run_summary(
    diagnostics_df,
    neighbour_df,
    config,
):

    summary = {
        "configuration": config,

        "number_of_clients": int(
            len(
                diagnostics_df
            )
        ),

        "signature": {},

        "degree": {},

        "neighbour_stability": {},
    }

    signature_columns = [
        "clean_signature_l2_norm",
        "clipped_signature_l2_norm",
        "private_signature_l2_norm",
        "signature_l2_error_clean",
        "signature_relative_l2_error_clean",
        "signature_l2_error_clipped",
        "signature_relative_l2_error_clipped",
        "signature_cosine_clean_private",
        "signature_cosine_clipped_private",
        "signature_noise_l1_norm",
        "signature_noise_l2_norm",
        "signature_noise_linf_norm",
    ]

    for column in signature_columns:

        if column in diagnostics_df.columns:

            summary[
                "signature"
            ][
                column
            ] = numeric_summary(
                diagnostics_df[
                    column
                ]
            )

    if (
        "signature_was_clipped"
        in diagnostics_df.columns
    ):

        clipped = diagnostics_df[
            "signature_was_clipped"
        ].astype(
            bool
        )

        summary[
            "signature"
        ][
            "number_signatures_clipped"
        ] = int(
            clipped.sum()
        )

        summary[
            "signature"
        ][
            "fraction_signatures_clipped"
        ] = float(
            clipped.mean()
        )

    degree_columns = [
        "degree_raw_sum",
        "degree_clipped_sum",
        "degree_private_sum",
        "degree_l1_error_raw",
        "degree_l2_error_raw",
        "degree_mae_raw",
        "degree_rmse_raw",
        "degree_relative_l1_error_raw",
        "degree_l1_error_clipped",
        "degree_l2_error_clipped",
        "degree_mae_clipped",
        "degree_rmse_clipped",
        "degree_exact_component_fraction_raw",
        "degree_exact_component_fraction_clipped",
        "degree_noise_l1_norm",
        "degree_noise_l2_norm",
        "degree_components_clipped",
        "degree_negative_before_postprocessing",
        "degree_above_D_max_before_postprocessing",
    ]

    for column in degree_columns:

        if column in diagnostics_df.columns:

            summary[
                "degree"
            ][
                column
            ] = numeric_summary(
                diagnostics_df[
                    column
                ]
            )

    if (
        "degree_any_component_clipped"
        in diagnostics_df.columns
    ):

        clipped = diagnostics_df[
            "degree_any_component_clipped"
        ].astype(
            bool
        )

        summary[
            "degree"
        ][
            "number_clients_with_degree_clipping"
        ] = int(
            clipped.sum()
        )

        summary[
            "degree"
        ][
            "fraction_clients_with_degree_clipping"
        ] = float(
            clipped.mean()
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

        for column in overlap_columns:

            summary[
                "neighbour_stability"
            ][
                column
            ] = numeric_summary(
                neighbour_df[
                    column
                ]
            )

        summary[
            "neighbour_stability"
        ][
            "number_clients_sampled"
        ] = int(
            len(
                neighbour_df
            )
        )

    return summary


def process_ego_graphs(
    G=None,
    epsilon=2.0,
    dataset_name="enron",
    output_folder="output/",
    signature_method=DEFAULT_SIGNATURE_METHOD,
    signature_mechanism=DEFAULT_SIGNATURE_MECHANISM,
    signature_fraction=DEFAULT_SIGNATURE_FRACTION,
    signature_clip_norm=DEFAULT_SIGNATURE_CLIP_NORM,
    d_max=DEFAULT_D_MAX,
    degree_mode=DEFAULT_DEGREE_MODE,
    degree_components=None,
    degree_adjacency=DEFAULT_DEGREE_ADJACENCY,
    seed=DEFAULT_SEED,
    max_clients=None,
    max_neighbour_clients=DEFAULT_MAX_NEIGHBOUR_CLIENTS,
    k_values=DEFAULT_K_VALUES,
    force_rebuild=False,
    reports_output_folder=None,
):


    del G

    epsilon = float(
        epsilon
    )

    signature_method = str(
        signature_method
    ).lower()

    signature_mechanism = str(
        signature_mechanism
    ).lower()

    degree_mode = str(
        degree_mode
    ).lower()

    degree_adjacency = str(
        degree_adjacency
    ).lower()

    seed = int(
        seed
    )

    if force_rebuild:
        raise ValueError(
            "V2 disables force_rebuild to prevent silent replacement of results. "
            "Choose a fresh reports_output_folder instead."
        )
    validate_separate_roots(output_folder, reports_output_folder)
    print(
        "[SIMULATION ONLY] Deterministic noise is reproducible; "
        "never release these reports as genuine private client data."
    )


    degree_components = resolve_degree_components(
        dataset_name=dataset_name,
        output_folder=output_folder,
        degree_mode=degree_mode,
        degree_components=degree_components,
    )


    config = build_run_configuration(
        dataset_name=dataset_name,
        epsilon=epsilon,
        seed=seed,
        signature_method=signature_method,
        signature_mechanism=signature_mechanism,
        signature_fraction=signature_fraction,
        signature_clip_norm=signature_clip_norm,
        degree_mode=degree_mode,
        degree_components=degree_components,
        degree_adjacency=degree_adjacency,
        d_max=d_max,
    )


    config["prepared_input_sha256"] = input_archive_fingerprints(
        dataset_name, output_folder
    )

    run_folder = build_run_folder(
        dataset_name=dataset_name,
        output_folder=reports_output_folder,
        config=config,
    )


    if os.path.exists(run_folder):
        raise FileExistsError(
            f"V2 run folder already exists: {run_folder}. "
            "V2 does not reuse caches or overwrite past results. "
            "Choose a fresh reports_output_folder."
        )

    os.makedirs(run_folder, exist_ok=False)


    restricted_root = str(Path(reports_output_folder).resolve()) + "_SENSITIVE_DO_NOT_SHARE"
    restricted_run_folder = os.path.join(
        restricted_root,
        os.path.relpath(run_folder, reports_output_folder),
    )
    os.makedirs(restricted_run_folder, exist_ok=True)

    reports_path = os.path.join(
        run_folder,
        "private_reports.npz",
    )

    diagnostics_path = os.path.join(
        restricted_run_folder,
        "privacy_diagnostics.csv.gz",
    )

    neighbour_path = os.path.join(
        restricted_run_folder,
        "private_neighbour_stability.csv.gz",
    )

    summary_path = os.path.join(
        restricted_run_folder,
        "privacy_run_summary.json",
    )

    metadata_path = os.path.join(
        run_folder,
        "private_report_metadata.json",
    )


    required_cache_files = [
        reports_path,
        diagnostics_path,
        neighbour_path,
        summary_path,
        metadata_path,
    ]

    if (
        not force_rebuild
        and all(
            os.path.exists(
                path
            )
            for path
            in required_cache_files
        )
        and cache_matches_configuration(
            metadata_path,
            config,
        )
    ):

        print(
            "[privacy] Using cached run:"
        )

        print(
            f"  {run_folder}"
        )

        stored = np.load(
            reports_path,
            allow_pickle=False,
        )

        return {
            "run_folder": run_folder,

            "client_ids": stored[
                "client_ids"
            ],

            "private_signatures": stored[
                "private_signatures"
            ],

            "private_degrees": stored[
                "private_degrees"
            ],

            "degree_component_names": stored[
                "degree_component_names"
            ],

            "diagnostics": pd.read_csv(
                diagnostics_path,
                compression="gzip",
            ),

            "neighbour_stability": pd.read_csv(
                neighbour_path,
                compression="gzip",
            ),

            "configuration": config,
        }


    signatures = load_or_compute_signatures(
        dataset_name=dataset_name,
        output_folder=output_folder,
    )

    all_signature_client_ids = np.asarray(
        signatures[
            "client_ids"
        ],
        dtype=str,
    )

    clean_signature_array = select_signature_array(
        signatures,
        signature_method,
    )

    if (
        len(
            all_signature_client_ids
        )
        != clean_signature_array.shape[
            0
        ]
    ):
        raise ValueError(
            "Signature IDs and signature matrix have "
            "inconsistent lengths."
        )

    signature_by_client = {
        str(
            client_id
        ): clean_signature_array[
            index
        ]
        for index, client_id
        in enumerate(
            all_signature_client_ids
        )
    }


    ego_index = load_ego_index(
        dataset_name,
        output_folder,
    )

    ego_filename_by_client = {
        str(
            row[
                "client_id"
            ]
        ): str(
            row[
                "ego_filename"
            ]
        )
        for _, row
        in ego_index.iterrows()
    }


    client_ids = [
        str(
            client_id
        )
        for client_id
        in all_signature_client_ids
        if str(
            client_id
        )
        in ego_filename_by_client
    ]

    if max_clients is not None:

        max_clients = int(
            max_clients
        )

        if max_clients <= 0:
            raise ValueError(
                "max_clients must be positive."
            )

        client_ids = client_ids[
            :max_clients
        ]

    if not client_ids:
        raise RuntimeError(
            "No clients are available for privatization."
        )

    ego_folder = os.path.join(
        output_folder,
        dataset_name,
        "ego_graphs_local",
    )


    epsilon_signature, epsilon_degree = (
        split_privacy_budget(
            epsilon_total=epsilon,
            signature_fraction=signature_fraction,
        )
    )


    successful_client_ids = []

    successful_clean_signatures = []

    private_signatures = []

    private_degrees = []

    diagnostics_rows = []

    failures = []


    for client_id in tqdm(
        client_ids,
        desc=(
            f"Privatizing client reports "
            f"({dataset_name}, eps={epsilon:g})"
        ),
    ):

        try:

            ego_filename = (
                ego_filename_by_client[
                    client_id
                ]
            )

            ego_path = os.path.join(
                ego_folder,
                ego_filename,
            )

            if not os.path.exists(
                ego_path
            ):
                raise FileNotFoundError(
                    ego_path
                )

            ego_graph = nx.read_gml(
                ego_path
            )

            main_node = str(
                ego_graph.graph.get(
                    "main_node",
                    client_id,
                )
            )

            if main_node not in ego_graph:

                matching_nodes = [
                    node
                    for node
                    in ego_graph.nodes()
                    if str(
                        node
                    ) == main_node
                ]

                if not matching_nodes:
                    raise KeyError(
                        "Main client is missing "
                        "from its ego graph."
                    )

                main_node = matching_nodes[
                    0
                ]


            clean_signature = np.asarray(
                signature_by_client[
                    client_id
                ],
                dtype=float,
            )


            (
                raw_degree_vector,
                returned_component_names,
                degree_extraction_info,
            ) = extract_typed_degree_vector(
                ego_graph=ego_graph,
                main_node=main_node,
                mode=degree_mode,
                component_names=degree_components,
            )

            if (
                list(
                    returned_component_names
                )
                != list(
                    degree_components
                )
            ):
                raise ValueError(
                    "Typed degree coordinates changed "
                    "between clients."
                )


            client_seed = deterministic_client_seed(
                base_seed=seed,
                dataset_name=dataset_name,
                client_id=client_id,
                epsilon=epsilon,
                signature_method=signature_method,
                signature_mechanism=signature_mechanism,
                signature_fraction=signature_fraction,
                degree_mode=degree_mode,
            )

            client_rng = np.random.default_rng(
                client_seed
            )


            (
                private_signature,
                private_degree,
                privacy_info,
            ) = privatize_client_report(
                signature=clean_signature,
                degree_vector=raw_degree_vector,
                epsilon_total=epsilon,
                signature_fraction=signature_fraction,
                signature_mechanism=signature_mechanism,
                signature_clip_norm=signature_clip_norm,
                d_max=d_max,
                degree_adjacency=degree_adjacency,
                rng=client_rng,
            )

            signature_info = privacy_info[
                "signature"
            ]

            degree_info = privacy_info[
                "degree"
            ]

            clipped_degree = np.asarray(
                degree_info[
                    "clipped_degree_vector"
                ],
                dtype=int,
            )


            signature_error_stats = (
                compute_signature_error_statistics(
                    clean_signature=clean_signature,
                    private_signature=private_signature,
                    clip_norm=signature_clip_norm,
                )
            )

            degree_error_stats = (
                compute_degree_error_statistics(
                    raw_degree=raw_degree_vector,
                    clipped_degree=clipped_degree,
                    private_degree=private_degree,
                )
            )

            row = {
                "dataset": dataset_name,

                "client_id": str(
                    client_id
                ),

                "client_seed": int(
                    client_seed
                ),

                "epsilon_total": float(
                    epsilon
                ),

                "epsilon_signature": float(
                    epsilon_signature
                ),

                "epsilon_degree": float(
                    epsilon_degree
                ),

                "signature_fraction": float(
                    signature_fraction
                ),

                "degree_fraction": float(
                    1.0
                    - signature_fraction
                ),

                "signature_method": (
                    signature_method
                ),

                "signature_mechanism": (
                    signature_mechanism
                ),

                "signature_dimension": int(
                    len(
                        clean_signature
                    )
                ),

                "signature_clip_norm_C": float(
                    signature_clip_norm
                ),

                "signature_sensitivity": float(
                    signature_info.get(
                        "sensitivity_l2",
                        signature_info.get(
                            "sensitivity_l1",
                            np.nan,
                        ),
                    )
                ),

                "signature_noise_l1_norm": float(
                    signature_info[
                        "noise_l1_norm"
                    ]
                ),

                "signature_noise_l2_norm": float(
                    signature_info[
                        "noise_l2_norm"
                    ]
                ),

                "signature_noise_linf_norm": float(
                    signature_info[
                        "noise_linf_norm"
                    ]
                ),

                "degree_mode": degree_mode,

                "degree_component_names_json": (
                    json.dumps(
                        degree_components
                    )
                ),

                "degree_adjacency": (
                    degree_adjacency
                ),

                "degree_sensitivity_l1": float(
                    degree_info[
                        "sensitivity_l1"
                    ]
                ),

                "D_max": int(
                    d_max
                ),

                "degree_raw_vector_json": (
                    json.dumps(
                        [
                            int(
                                value
                            )
                            for value
                            in raw_degree_vector
                        ]
                    )
                ),

                "degree_clipped_vector_json": (
                    json.dumps(
                        [
                            int(
                                value
                            )
                            for value
                            in clipped_degree
                        ]
                    )
                ),

                "degree_private_vector_json": (
                    json.dumps(
                        [
                            int(
                                value
                            )
                            for value
                            in private_degree
                        ]
                    )
                ),

                "degree_noise_vector_json": (
                    json.dumps(
                        degree_info[
                            "noise_vector"
                        ]
                    )
                ),

                "degree_components_clipped": int(
                    degree_info[
                        "number_components_clipped"
                    ]
                ),

                "degree_any_component_clipped": bool(
                    degree_info[
                        "was_any_component_clipped"
                    ]
                ),

                "degree_noise_l1_norm": int(
                    degree_info[
                        "noise_l1_norm"
                    ]
                ),

                "degree_noise_l2_norm": float(
                    degree_info[
                        "noise_l2_norm"
                    ]
                ),

                "degree_negative_before_postprocessing": int(
                    degree_info[
                        "number_negative_before_postprocessing"
                    ]
                ),

                "degree_above_D_max_before_postprocessing": int(
                    degree_info[
                        "number_above_D_max_before_postprocessing"
                    ]
                ),

                "degree_discrete_laplace_lambda": float(
                    degree_info[
                        "lambda"
                    ]
                ),

                "degree_discrete_laplace_alpha": float(
                    degree_info[
                        "alpha"
                    ]
                ),

                "degree_geometric_success_probability": float(
                    degree_info[
                        "geometric_success_probability"
                    ]
                ),

                "ego_nodes": int(
                    ego_graph.number_of_nodes()
                ),

                "ego_edges": int(
                    ego_graph.number_of_edges()
                ),

                "raw_degree_sum_from_extraction": int(
                    degree_extraction_info[
                        "raw_degree_sum"
                    ]
                ),

                **signature_error_stats,

                **degree_error_stats,
            }


            for component_index, component_name in enumerate(
                degree_components
            ):

                raw_value = int(
                    raw_degree_vector[
                        component_index
                    ]
                )

                clipped_value = int(
                    clipped_degree[
                        component_index
                    ]
                )

                private_value = int(
                    private_degree[
                        component_index
                    ]
                )

                row[
                    f"degree_raw__{component_name}"
                ] = raw_value

                row[
                    f"degree_clipped__{component_name}"
                ] = clipped_value

                row[
                    f"degree_private__{component_name}"
                ] = private_value

                row[
                    f"degree_signed_error_raw__{component_name}"
                ] = (
                    private_value
                    - raw_value
                )

                row[
                    f"degree_abs_error_raw__{component_name}"
                ] = abs(
                    private_value
                    - raw_value
                )

                row[
                    f"degree_abs_error_clipped__{component_name}"
                ] = abs(
                    private_value
                    - clipped_value
                )


            successful_client_ids.append(
                str(
                    client_id
                )
            )

            successful_clean_signatures.append(
                clean_signature
            )

            private_signatures.append(
                np.asarray(
                    private_signature,
                    dtype=float,
                )
            )

            private_degrees.append(
                np.asarray(
                    private_degree,
                    dtype=int,
                )
            )

            diagnostics_rows.append(
                row
            )

        except Exception as exc:

            failures.append({
                "client_id": str(
                    client_id
                ),

                "error": repr(
                    exc
                ),
            })


    if not successful_client_ids:

        raise RuntimeError(
            "No private client reports were generated."
        )

    private_signature_array = np.vstack(
        private_signatures
    ).astype(
        float
    )

    private_degree_array = np.vstack(
        private_degrees
    ).astype(
        int
    )

    clean_signature_success_array = np.vstack(
        successful_clean_signatures
    ).astype(
        float
    )

    client_ids_array = np.asarray(
        successful_client_ids,
        dtype=str,
    )

    degree_component_array = np.asarray(
        degree_components,
        dtype=str,
    )

    if (
        private_signature_array.shape[
            0
        ]
        != private_degree_array.shape[
            0
        ]
        or private_signature_array.shape[
            0
        ]
        != len(
            client_ids_array
        )
    ):
        raise RuntimeError(
            "Private report arrays have inconsistent lengths."
        )

    if (
        private_degree_array.shape[
            1
        ]
        != len(
            degree_components
        )
    ):
        raise RuntimeError(
            "Private degree dimension does not match "
            "the fixed typed-degree schema."
        )


    np.savez_compressed(
        reports_path,

        client_ids=client_ids_array,

        private_signatures=(
            private_signature_array
        ),

        private_degrees=(
            private_degree_array
        ),

        degree_component_names=(
            degree_component_array
        ),
    )


    diagnostics_df = pd.DataFrame(
        diagnostics_rows
    )

    diagnostics_df.to_csv(
        diagnostics_path,
        index=False,
        compression="gzip",
    )


    neighbour_seed = int(
        seed
        + 7919
    )

    neighbour_df = (
        compute_private_neighbour_stability(
            client_ids=client_ids_array,
            clean_signatures=(
                clean_signature_success_array
            ),
            private_signatures=(
                private_signature_array
            ),
            k_values=k_values,
            max_clients=max_neighbour_clients,
            seed=neighbour_seed,
        )
    )

    if not neighbour_df.empty:

        neighbour_df.insert(
            0,
            "dataset",
            dataset_name,
        )

        neighbour_df.insert(
            1,
            "epsilon_total",
            float(
                epsilon
            ),
        )

        neighbour_df.insert(
            2,
            "seed",
            int(
                seed
            ),
        )

        neighbour_df.insert(
            3,
            "signature_method",
            signature_method,
        )

        neighbour_df.insert(
            4,
            "signature_mechanism",
            signature_mechanism,
        )


    neighbour_df.to_csv(
        neighbour_path,
        index=False,
        compression="gzip",
    )


    summary = build_run_summary(
        diagnostics_df=diagnostics_df,
        neighbour_df=neighbour_df,
        config=config,
    )

    summary[
        "number_failed_clients"
    ] = int(
        len(
            failures
        )
    )

    summary[
        "fraction_successful_clients"
    ] = float(
        len(
            successful_client_ids
        )
        /
        max(
            len(
                client_ids
            ),
            1,
        )
    )


    summary[
        "degree"
    ][
        "components"
    ] = {}

    for component_name in degree_components:

        component_summary = {}

        for prefix in [
            "degree_raw",
            "degree_clipped",
            "degree_private",
            "degree_signed_error_raw",
            "degree_abs_error_raw",
            "degree_abs_error_clipped",
        ]:

            column = (
                f"{prefix}__{component_name}"
            )

            if column in diagnostics_df.columns:

                component_summary[
                    prefix
                ] = numeric_summary(
                    diagnostics_df[
                        column
                    ]
                )

        summary[
            "degree"
        ][
            "components"
        ][
            component_name
        ] = component_summary

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=4,
            ensure_ascii=False,
        )


    metadata = {
        "schema_version": (
            PRIVATE_REPORT_SCHEMA_VERSION
        ),
        "simulation_only": True,
        "warning": (
            "This report uses reproducible deterministic noise; NOT safe "
            "for release to an untrusted curator or for deployment."
        ),
        "diagnostics_separate_from_report_tree": True,

        "configuration": config,

        "configuration_hash": (
            configuration_hash(
                config
            )
        ),

        "requested_clients": int(
            len(
                client_ids
            )
        ),

        "successful_clients": int(
            len(
                successful_client_ids
            )
        ),

        "failed_clients": failures,

        "number_failed_clients": int(
            len(
                failures
            )
        ),

        "private_report_contents": [
            "client_ids",
            "private_signatures",
            "private_degrees",
            "degree_component_names",
        ],

        "private_report_does_not_contain": [
            "clean_signature",
            "clipped_signature",
            "raw_degree_vector",
            "clean_ego_graph",
            "original_edges",
            "oracle_fidelity",
        ],

        "experimental_diagnostics_file": (
            os.path.basename(
                diagnostics_path
            )
        ),

        "private_report_file": (
            os.path.basename(
                reports_path
            )
        ),

        "neighbour_stability_file": (
            os.path.basename(
                neighbour_path
            )
        ),

        "summary_file": (
            os.path.basename(
                summary_path
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

        "neighbour_k_values": [
            int(
                value
            )
            for value
            in k_values
        ],

        "important_privacy_notes": [
            (
                "Exactly one privatized signature and one privatized "
                "typed-degree vector are released per client."
            ),

            (
                "The privacy budget is fixed from public experiment "
                "parameters and is not a function of the client's "
                "private degree."
            ),

            (
                "The main journal configuration uses client_report "
                "adjacency for the typed-degree vector."
            ),

            (
                "Clean quantities are stored only in the separate "
                "experimental diagnostics file and are not available "
                "to the reconstruction algorithm."
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
        f"PRIVATE CLIENT REPORTS: "
        f"{dataset_name}"
    )

    print(
        "========================================"
    )

    print(
        f"epsilon total: "
        f"{epsilon:g}"
    )

    print(
        f"epsilon signature: "
        f"{epsilon_signature:g}"
    )

    print(
        f"epsilon degree: "
        f"{epsilon_degree:g}"
    )

    print(
        f"Signature: "
        f"{signature_method.upper()}"
    )

    print(
        f"Mechanism: "
        f"{signature_mechanism}"
    )

    print(
        f"Degree mode: "
        f"{degree_mode}"
    )

    print(
        f"Degree components: "
        f"{degree_components}"
    )

    print(
        f"Clients requested: "
        f"{len(client_ids):,}"
    )

    print(
        f"Private reports created: "
        f"{len(successful_client_ids):,}"
    )

    print(
        f"Failures: "
        f"{len(failures):,}"
    )

    if not diagnostics_df.empty:

        print(
            "\nSignature utility:"
        )

        print(
            "  Mean clean/private cosine: "
            f"{diagnostics_df['signature_cosine_clean_private'].mean():.4f}"
        )

        print(
            "  Mean relative L2 error: "
            f"{diagnostics_df['signature_relative_l2_error_clean'].mean():.4f}"
        )

        print(
            "\nDegree utility:"
        )

        print(
            "  Mean MAE: "
            f"{diagnostics_df['degree_mae_raw'].mean():.4f}"
        )

        print(
            "  Mean relative L1 error: "
            f"{diagnostics_df['degree_relative_l1_error_raw'].mean():.4f}"
        )

        clipping_fraction = (
            diagnostics_df[
                "degree_any_component_clipped"
            ]
            .astype(
                bool
            )
            .mean()
        )

        print(
            "  Clients requiring degree clipping: "
            f"{100.0 * clipping_fraction:.2f}%"
        )

    if not neighbour_df.empty:

        print(
            "\nClean/private candidate stability:"
        )

        for k in k_values:

            column = (
                f"clean_private_overlap_at_{int(k)}"
            )

            if column in neighbour_df.columns:

                print(
                    f"  Overlap@{k}: "
                    f"{neighbour_df[column].mean():.4f}"
                )

    print(
        "\nSERVER-VISIBLE private reports:"
    )

    print(
        f"  {reports_path}"
    )

    print(
        "\nEXPERIMENT-ONLY diagnostics:"
    )

    print(
        f"  {diagnostics_path}"
    )

    print(
        f"  {neighbour_path}"
    )

    print(
        f"\nSummary:"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"\nRun folder:"
    )

    print(
        f"  {run_folder}"
    )

    print(
        "========================================\n"
    )

    return {
        "run_folder": run_folder,

        "client_ids": client_ids_array,

        "private_signatures": (
            private_signature_array
        ),

        "private_degrees": (
            private_degree_array
        ),

        "degree_component_names": (
            degree_component_array
        ),

        "diagnostics": diagnostics_df,

        "neighbour_stability": (
            neighbour_df
        ),

        "configuration": config,
    }


def load_private_reports(
    run_folder,
):

    reports_path = os.path.join(
        run_folder,
        "private_reports.npz",
    )

    metadata_path = os.path.join(
        run_folder,
        "private_report_metadata.json",
    )

    if not os.path.exists(
        reports_path
    ):
        raise FileNotFoundError(
            f"Missing private reports: "
            f"{reports_path}"
        )

    if not os.path.exists(
        metadata_path
    ):
        raise FileNotFoundError(
            f"Missing private report metadata: "
            f"{metadata_path}"
        )

    stored = np.load(
        reports_path,
        allow_pickle=False,
    )

    with open(
        metadata_path,
        "r",
        encoding="utf-8",
    ) as f:

        metadata = json.load(
            f
        )

    return {
        "client_ids": stored[
            "client_ids"
        ],

        "private_signatures": stored[
            "private_signatures"
        ],

        "private_degrees": stored[
            "private_degrees"
        ],

        "degree_component_names": stored[
            "degree_component_names"
        ],

        "metadata": metadata,
    }


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Generate one locally differentially "
            "private report per client."
        )
    )

    parser.add_argument(
        "dataset",
        type=str,
    )

    parser.add_argument(
        "--epsilon",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--signature",
        choices=[
            "raw",
            "abs",
            "hks",
        ],
        default=DEFAULT_SIGNATURE_METHOD,
    )

    parser.add_argument(
        "--mechanism",
        choices=[
            "l2_knorm",
            "coordinate_laplace",
        ],
        default=DEFAULT_SIGNATURE_MECHANISM,
    )

    parser.add_argument(
        "--signature-fraction",
        type=float,
        default=DEFAULT_SIGNATURE_FRACTION,
    )

    parser.add_argument(
        "--C",
        type=float,
        default=DEFAULT_SIGNATURE_CLIP_NORM,
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
        default=DEFAULT_DEGREE_ADJACENCY,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--max-clients",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--max-neighbour-clients",
        type=int,
        default=DEFAULT_MAX_NEIGHBOUR_CLIENTS,
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default="output/",
        help="Existing READ-ONLY prepared data root (do not overwrite).",
    )

    parser.add_argument(
        "--reports-output-folder",
        type=str,
        required=True,
        help="A separate, NEW output root for V2 simulation reports.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    process_ego_graphs(
        G=None,
        epsilon=args.epsilon,
        dataset_name=args.dataset,
        output_folder=args.output_folder,
        reports_output_folder=args.reports_output_folder,
        signature_method=args.signature,
        signature_mechanism=args.mechanism,
        signature_fraction=(
            args.signature_fraction
        ),
        signature_clip_norm=args.C,
        d_max=args.D_max,
        degree_mode=args.degree_mode,
        degree_adjacency=(
            args.degree_adjacency
        ),
        seed=args.seed,
        max_clients=args.max_clients,
        max_neighbour_clients=(
            args.max_neighbour_clients
        ),
        force_rebuild=args.force,
    )
