import argparse
import hashlib
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone

import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import (
    t as student_t,
    wilcoxon,
)

from noisy_ego_graphs import (
    load_private_reports,
)

from synthesized_graph import (
    compute_exact_top_candidates,
    compute_private_objective,
    extract_private_targets,
    reconstruct_graph_from_private_reports,
    resolve_private_degree_cap,
    run_private_bayesian_optimization,
)

from stats import (
    compute_oracle_metrics,
)


SCHEMA_VERSION = "final-ablation-1.2"

DATASET = "enron"

DEFAULT_OUTPUT_FOLDER = "output_journal_final"

FINAL_SYNTHESIS_RECORDS = os.path.join(
    DEFAULT_OUTPUT_FOLDER,
    "journal_final_synthesis",
    "final_synthesis_run_records.csv",
)


DEFAULT_EPSILONS = (
    1.0,
    3.0,
    5.0,
)

DEFAULT_SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

DEFAULT_VARIANTS = (
    "full_hks",
    "no_private_bo",
    "untyped_total_degree",
    "no_shared_content",
)

VARIANT_DESCRIPTIONS = {
    "full_hks": (
        "Frozen complete RQ4 method: heat-spectrum descriptor "
        "(legacy code label hks), private role-degree vector, private "
        "Bayesian optimization, and shared synthetic content."
    ),
    "no_private_bo": (
        "Use the same frozen private reports but replace private BO with "
        "one deterministic private-only parameter configuration."
    ),
    "untyped_total_degree": (
        "Discard the private client/content degree split by collapsing the "
        "already-private role-degree vector to total degree and reallocating "
        "that total with a fixed public 50/50 rule."
    ),
    "no_shared_content": (
        "Use the same frozen private reports and private BO but set "
        "synthetic content capacity to one, preventing shared content."
    ),
}


FROZEN_SIGNATURE_METHOD = "hks"
FROZEN_SIGNATURE_MECHANISM = "l2_knorm"
FROZEN_SIGNATURE_FRACTION = 0.5
FROZEN_SIGNATURE_CLIP_NORM = 1.0
FROZEN_D_MAX = 50
FROZEN_DEGREE_MODE = "role"
FROZEN_DEGREE_ADJACENCY = "client_report"

FROZEN_MAX_CANDIDATES = 200
FROZEN_BO_EVALUATIONS = 40
FROZEN_BO_INITIAL_POINTS = 8
FROZEN_BO_ACQUISITION_CANDIDATES = 2000
FROZEN_CONTENT_CAPACITY = 4


FROZEN_FIXED_CANDIDATE_K = 50


UNTYPED_PUBLIC_CLIENT_FRACTION = 0.5


FROZEN_ORACLE_SEED = 0


EXPECTED_ENRON_NODES = 14_352
EXPECTED_ENRON_EDGES = 79_750
EXPECTED_ENRON_CLIENTS = 4_352
EXPECTED_ENRON_CONTENT = 10_000

CONFIDENCE_LEVEL = 0.95
NUMERICAL_EPS = 1e-12
REPLAY_TOLERANCE = 1e-9


def utc_now_iso():
    return datetime.now(
        timezone.utc
    ).isoformat()


def normalize_path(path):
    return os.path.normcase(
        os.path.abspath(
            os.path.expanduser(
                str(path)
            )
        )
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


def stable_json_hash(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as handle:

        while True:

            chunk = handle.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def safe_json_dump(value, path):
    os.makedirs(
        os.path.dirname(path),
        exist_ok=True,
    )

    temporary = (
        path
        +
        ".tmp"
    )

    with open(
        temporary,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            value,
            handle,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    os.replace(
        temporary,
        path,
    )


def load_json(path):
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as handle:

        return json.load(
            handle
        )


def ensure_final_output_scope(output_folder, allow_test_output=False):
    normalized = normalize_path(
        output_folder
    )

    suspicious = (
        "output_local_test",
        "local_test",
        "smoke_test",
        "debug_output",
    )

    if (
        any(
            token in normalized
            for token in suspicious
        )
        and not allow_test_output
    ):
        raise RuntimeError(
            "Refusing to run final RQ6 ablations inside a debug/test "
            f"folder: {output_folder}\n"
            "Use output_journal_final, or explicitly pass "
            "--allow-test-output for a smoke test."
        )


def epsilon_mask(series, epsilon):
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=float,
    )

    return np.isclose(
        values,
        float(epsilon),
        rtol=0.0,
        atol=1e-12,
    )


def confidence_interval(values, confidence=CONFIDENCE_LEVEL):
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    n = int(
        len(values)
    )

    if n == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "sem": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
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
            "ci95_low": mean,
            "ci95_high": mean,
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

    alpha = (
        1.0
        -
        float(confidence)
    )

    critical = float(
        student_t.ppf(
            1.0
            -
            alpha / 2.0,
            df=n - 1,
        )
    )

    margin = (
        critical
        *
        sem
    )

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "sem": sem,
        "ci95_low": float(
            mean - margin
        ),
        "ci95_high": float(
            mean + margin
        ),
    }


def paired_wilcoxon(first, second):
    first = np.asarray(
        first,
        dtype=float,
    )

    second = np.asarray(
        second,
        dtype=float,
    )

    mask = (
        np.isfinite(first)
        &
        np.isfinite(second)
    )

    first = first[
        mask
    ]

    second = second[
        mask
    ]

    if len(first) < 2:
        return np.nan

    difference = (
        first
        -
        second
    )

    if np.allclose(
        difference,
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

    except Exception:
        return np.nan


def cohen_dz(differences):
    differences = np.asarray(
        differences,
        dtype=float,
    )

    differences = differences[
        np.isfinite(differences)
    ]

    if len(differences) < 2:
        return np.nan

    std = float(
        np.std(
            differences,
            ddof=1,
        )
    )

    if std <= NUMERICAL_EPS:

        if np.allclose(
            differences,
            0.0,
        ):
            return 0.0

        return np.inf

    return float(
        np.mean(differences)
        /
        std
    )


def synthesis_records_path(output_folder):
    return os.path.join(
        output_folder,
        "journal_final_synthesis",
        "final_synthesis_run_records.csv",
    )


def load_frozen_synthesis_records(
    output_folder,
):
    path = synthesis_records_path(
        output_folder
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Missing frozen RQ4 synthesis records:\n"
            f"  {path}"
        )

    frame = pd.read_csv(
        path
    )

    required = {
        "dataset",
        "epsilon_total",
        "epsilon_signature",
        "epsilon_degree",
        "seed",
        "status",
        "evaluation_scope",
        "private_reports",
        "private_report_failures",
        "best_private_objective",
        "private_run_folder",
    }

    missing = (
        required
        -
        set(
            frame.columns
        )
    )

    if missing:
        raise ValueError(
            "Frozen synthesis records are missing columns: "
            f"{sorted(missing)}"
        )

    frame = frame[
        (
            frame[
                "dataset"
            ].astype(str)
            ==
            DATASET
        )
        &
        (
            frame[
                "status"
            ].astype(str)
            ==
            "complete"
        )
    ].copy()

    if frame.empty:
        raise RuntimeError(
            "No completed frozen Enron RQ4 synthesis records found."
        )

    return frame


def select_frozen_record(
    records,
    epsilon,
    seed,
):
    mask = epsilon_mask(
        records[
            "epsilon_total"
        ],
        epsilon,
    )

    mask &= (
        pd.to_numeric(
            records[
                "seed"
            ],
            errors="coerce",
        )
        ==
        int(seed)
    )

    subset = records[
        mask
    ].copy()

    if len(subset) != 1:
        raise RuntimeError(
            "Expected exactly one frozen RQ4 record for "
            f"epsilon={epsilon}, seed={seed}; found {len(subset)}."
        )

    row = subset.iloc[
        0
    ]

    if str(
        row[
            "evaluation_scope"
        ]
    ) != "full_client_population":
        raise RuntimeError(
            "Frozen RQ4 record is not full-client population."
        )

    report_count = int(
        row[
            "private_reports"
        ]
    )

    report_failures = int(
        row[
            "private_report_failures"
        ]
    )

    if report_count != EXPECTED_ENRON_CLIENTS:
        raise RuntimeError(
            "Unexpected private-report count in frozen RQ4 record: "
            f"{report_count}; expected {EXPECTED_ENRON_CLIENTS}."
        )

    if report_failures != 0:
        raise RuntimeError(
            "Frozen RQ4 record reports private-release failures: "
            f"{report_failures}."
        )

    expected_signature_budget = (
        float(epsilon)
        *
        FROZEN_SIGNATURE_FRACTION
    )

    expected_degree_budget = (
        float(epsilon)
        *
        (
            1.0
            -
            FROZEN_SIGNATURE_FRACTION
        )
    )

    if not np.isclose(
        float(
            row[
                "epsilon_signature"
            ]
        ),
        expected_signature_budget,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Frozen RQ4 signature-budget split does not match 50/50."
        )

    if not np.isclose(
        float(
            row[
                "epsilon_degree"
            ]
        ),
        expected_degree_budget,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Frozen RQ4 degree-budget split does not match 50/50."
        )

    return row


def validate_private_configuration(
    configuration,
    epsilon,
    expected_seed,
):


    expected = {
        "signature_method": FROZEN_SIGNATURE_METHOD,
        "signature_mechanism": FROZEN_SIGNATURE_MECHANISM,
        "degree_mode": FROZEN_DEGREE_MODE,
        "degree_adjacency": FROZEN_DEGREE_ADJACENCY,
    }

    for key, value in expected.items():

        if key not in configuration:
            raise RuntimeError(
                f"Private report metadata is missing required key: {key}"
            )

        if str(
            configuration[
                key
            ]
        ) != str(value):
            raise RuntimeError(
                f"Private report configuration mismatch for {key}: "
                f"{configuration[key]!r} != {value!r}"
            )


    numeric_expected = {
        "signature_fraction": FROZEN_SIGNATURE_FRACTION,
        "degree_fraction": (
            1.0
            -
            FROZEN_SIGNATURE_FRACTION
        ),
        "signature_clip_norm_C": FROZEN_SIGNATURE_CLIP_NORM,
        "D_max": FROZEN_D_MAX,
        "epsilon_total": float(
            epsilon
        ),
        "epsilon_signature": (
            float(
                epsilon
            )
            *
            FROZEN_SIGNATURE_FRACTION
        ),
        "epsilon_degree": (
            float(
                epsilon
            )
            *
            (
                1.0
                -
                FROZEN_SIGNATURE_FRACTION
            )
        ),
    }

    for key, value in numeric_expected.items():

        if key not in configuration:
            raise RuntimeError(
                f"Private report metadata is missing required key: {key}"
            )

        if not np.isclose(
            float(
                configuration[
                    key
                ]
            ),
            float(value),
            atol=1e-12,
            rtol=0.0,
        ):
            raise RuntimeError(
                f"Private report configuration mismatch for {key}: "
                f"{configuration[key]!r} != {value!r}"
            )

    exact_expected = {
        "private_report_schema_version": "2.0",
        "privacy_mechanism_schema_version": "1.0",
        "dataset": DATASET,
        "one_private_report_per_client": True,
        "uses_personalised_epsilon": False,
        "uses_original_graph_during_server_reconstruction": False,
    }

    for key, value in exact_expected.items():

        if key not in configuration:
            raise RuntimeError(
                f"Private report metadata is missing required key: {key}"
            )

        if configuration[
            key
        ] != value:
            raise RuntimeError(
                f"Private report configuration mismatch for {key}: "
                f"{configuration[key]!r} != {value!r}"
            )

    if "seed" not in configuration:
        raise RuntimeError(
            "Private report metadata is missing required key: seed"
        )

    if int(
        configuration[
            "seed"
        ]
    ) != int(
        expected_seed
    ):
        raise RuntimeError(
            "Private report seed mismatch: "
            f"{configuration['seed']!r} != {expected_seed!r}"
        )

    if "degree_components" not in configuration:
        raise RuntimeError(
            "Private report metadata is missing required key: "
            "degree_components"
        )

    degree_components = [
        str(
            value
        )
        for value
        in configuration[
            "degree_components"
        ]
    ]

    if degree_components != [
        "client",
        "content",
    ]:
        raise RuntimeError(
            "Frozen role-degree components mismatch: "
            f"{degree_components!r} != ['client', 'content']"
        )


def prepare_frozen_private_inputs(
    frozen_record,
    epsilon,
    seed,
):
    private_run_folder = str(
        frozen_record[
            "private_run_folder"
        ]
    )

    reports_path = os.path.join(
        private_run_folder,
        "private_reports.npz",
    )

    if not os.path.exists(
        reports_path
    ):
        raise FileNotFoundError(
            "Frozen private report file is missing:\n"
            f"  {reports_path}"
        )

    reports = load_private_reports(
        private_run_folder
    )

    client_ids = np.asarray(
        reports[
            "client_ids"
        ],
        dtype=str,
    )

    private_signatures = np.asarray(
        reports[
            "private_signatures"
        ],
        dtype=float,
    )

    private_degrees = np.asarray(
        reports[
            "private_degrees"
        ],
        dtype=int,
    )

    component_names = [
        str(value)
        for value
        in reports[
            "degree_component_names"
        ]
    ]

    configuration = (
        reports[
            "metadata"
        ][
            "configuration"
        ]
    )

    validate_private_configuration(
        configuration=configuration,
        epsilon=epsilon,
        expected_seed=seed,
    )

    if len(
        client_ids
    ) != EXPECTED_ENRON_CLIENTS:
        raise RuntimeError(
            "Frozen Enron private report does not contain the full "
            f"{EXPECTED_ENRON_CLIENTS}-client population."
        )

    if (
        private_signatures.ndim != 2
        or private_signatures.shape[
            0
        ] != len(
            client_ids
        )
    ):
        raise RuntimeError(
            "Private signature matrix shape is inconsistent with clients."
        )

    if (
        private_degrees.ndim != 2
        or private_degrees.shape[
            0
        ] != len(
            client_ids
        )
    ):
        raise RuntimeError(
            "Private degree matrix shape is inconsistent with clients."
        )

    resolved_degree_cap = resolve_private_degree_cap(
        configuration=configuration,
        explicit_degree_cap=None,
    )

    if int(
        resolved_degree_cap
    ) != FROZEN_D_MAX:
        raise RuntimeError(
            "Frozen private report resolved to unexpected D_max: "
            f"{resolved_degree_cap}; expected {FROZEN_D_MAX}."
        )


    private_targets = extract_private_targets(
        private_degrees=private_degrees,
        component_names=component_names,
        degree_mode=FROZEN_DEGREE_MODE,
        degree_cap=FROZEN_D_MAX,
    )

    return {
        "private_run_folder": private_run_folder,
        "reports_path": reports_path,
        "reports_sha256": sha256_file(
            reports_path
        ),
        "client_ids": client_ids,
        "private_signatures": private_signatures,
        "private_degrees_raw": private_degrees,
        "component_names": component_names,
        "configuration": configuration,
        "private_targets": private_targets,
        "degree_cap": int(
            resolved_degree_cap
        ),
    }


def clone_private_targets(
    private_targets,
):
    output = {
        "private_client_degree": np.asarray(
            private_targets[
                "private_client_degree"
            ],
            dtype=int,
        ).copy(),

        "private_content_degrees": np.asarray(
            private_targets[
                "private_content_degrees"
            ],
            dtype=int,
        ).copy(),

        "content_components": list(
            private_targets[
                "content_components"
            ]
        ),

        "client_component_names": list(
            private_targets[
                "client_component_names"
            ]
        ),
    }

    if "degree_postprocessing" in private_targets:
        output[
            "degree_postprocessing"
        ] = dict(
            private_targets[
                "degree_postprocessing"
            ]
        )

    return output


def collapse_and_publicly_resplit_degree(
    private_targets,
    client_fraction=UNTYPED_PUBLIC_CLIENT_FRACTION,
):


    client_fraction = float(
        client_fraction
    )

    if not (
        0.0
        <= client_fraction
        <= 1.0
    ):
        raise ValueError(
            "client_fraction must lie in [0,1]."
        )

    client_degree = np.asarray(
        private_targets[
            "private_client_degree"
        ],
        dtype=int,
    )

    content_degree = np.asarray(
        private_targets[
            "private_content_degrees"
        ],
        dtype=int,
    )

    content_components = list(
        private_targets[
            "content_components"
        ]
    )

    if (
        content_degree.ndim != 2
        or content_degree.shape[
            1
        ] != 1
        or len(
            content_components
        ) != 1
    ):
        raise RuntimeError(
            "The final untyped-degree ablation is defined for the frozen "
            "role-degree report with exactly one client-content coordinate."
        )

    original_content = content_degree[
        :,
        0
    ].astype(
        int
    )

    total = (
        client_degree
        +
        original_content
    ).astype(
        int
    )

    new_client = np.floor(
        client_fraction
        *
        total
    ).astype(
        int
    )

    new_content = (
        total
        -
        new_client
    ).astype(
        int
    )

    if not np.array_equal(
        new_client
        +
        new_content,
        total,
    ):
        raise RuntimeError(
            "Untyped public split failed to preserve total degree."
        )

    return {
        "private_client_degree": new_client,
        "private_content_degrees": new_content[
            :,
            None
        ],
        "content_components": content_components,
        "client_component_names": [
            "untyped_total_public_50_50"
        ],
        "degree_postprocessing": {
            "source": (
                "already-private feasible role-degree targets"
            ),
            "operation": (
                "collapse_to_total_then_public_fixed_resplit"
            ),
            "public_client_fraction": client_fraction,
            "public_content_fraction": (
                1.0
                -
                client_fraction
            ),
            "total_preserved_exactly": True,
            "privacy_cost": 0.0,
        },
    }


def fixed_private_parameters(
    candidate_scores,
):
    scores = np.asarray(
        candidate_scores,
        dtype=float,
    ).ravel()

    scores = scores[
        np.isfinite(scores)
    ]

    if len(
        scores
    ) == 0:
        raise RuntimeError(
            "No finite private candidate scores for no-BO ablation."
        )

    return {
        "similarity_threshold": float(
            np.median(
                scores
            )
        ),
        "client_degree_scale": 1.0,
        "content_degree_scale": 1.0,
        "candidate_k": int(
            min(
                FROZEN_FIXED_CANDIDATE_K,
                candidate_scores.shape[
                    1
                ],
            )
        ),
    }


def frozen_full_artifacts(
    private_run_folder,
):
    synthesis_folder = os.path.join(
        private_run_folder,
        "synthesis",
    )

    graph_path = os.path.join(
        synthesis_folder,
        "synthetic_graph.gml.gz",
    )

    trace_path = os.path.join(
        synthesis_folder,
        "private_bo_trace.csv",
    )

    summary_path = os.path.join(
        synthesis_folder,
        "private_bo_summary.json",
    )

    for path in (
        graph_path,
        trace_path,
        summary_path,
    ):

        if not os.path.exists(
            path
        ):
            raise FileNotFoundError(
                "Frozen full-method synthesis artifact is missing:\n"
                f"  {path}"
            )

    trace = pd.read_csv(
        trace_path
    )

    if len(
        trace
    ) != FROZEN_BO_EVALUATIONS:
        raise RuntimeError(
            "Frozen full-method trace does not contain exactly "
            f"{FROZEN_BO_EVALUATIONS} configurations: {trace_path}"
        )

    summary = load_json(
        summary_path
    )

    return {
        "graph_path": graph_path,
        "trace_path": trace_path,
        "summary_path": summary_path,
        "summary": summary,
    }


def variant_folder(
    experiment_root,
    epsilon,
    seed,
    variant,
):
    return os.path.join(
        experiment_root,
        (
            "eps_"
            +
            safe_float_token(
                epsilon
            )
        ),
        f"seed_{int(seed)}",
        str(
            variant
        ),
    )


def build_variant_configuration(
    epsilon,
    seed,
    variant,
    private_inputs,
):
    content_capacity = (
        1
        if variant == "no_shared_content"
        else FROZEN_CONTENT_CAPACITY
    )

    selection_method = (
        "frozen_rq4"
        if variant == "full_hks"
        else (
            "fixed_private_parameters"
            if variant == "no_private_bo"
            else "private_bayesian_optimization"
        )
    )

    return {
        "schema_version": SCHEMA_VERSION,
        "dataset": DATASET,
        "epsilon_total": float(
            epsilon
        ),
        "privacy_seed": int(
            seed
        ),
        "variant": str(
            variant
        ),
        "private_reports_sha256": private_inputs[
            "reports_sha256"
        ],
        "private_run_folder": normalize_path(
            private_inputs[
                "private_run_folder"
            ]
        ),
        "degree_cap": FROZEN_D_MAX,
        "degree_mode": FROZEN_DEGREE_MODE,
        "degree_adjacency": FROZEN_DEGREE_ADJACENCY,
        "signature_method": FROZEN_SIGNATURE_METHOD,
        "signature_mechanism": FROZEN_SIGNATURE_MECHANISM,
        "signature_fraction": FROZEN_SIGNATURE_FRACTION,
        "signature_clip_norm": FROZEN_SIGNATURE_CLIP_NORM,
        "max_candidates": FROZEN_MAX_CANDIDATES,
        "bo_evaluations": FROZEN_BO_EVALUATIONS,
        "bo_initial_points": FROZEN_BO_INITIAL_POINTS,
        "bo_acquisition_candidates": (
            FROZEN_BO_ACQUISITION_CANDIDATES
        ),
        "content_capacity": int(
            content_capacity
        ),
        "selection_method": selection_method,
        "fixed_candidate_k": (
            FROZEN_FIXED_CANDIDATE_K
            if variant == "no_private_bo"
            else None
        ),
        "untyped_public_client_fraction": (
            UNTYPED_PUBLIC_CLIENT_FRACTION
            if variant == "untyped_total_degree"
            else None
        ),
    }


def cached_private_phase_is_valid(
    private_result_path,
    graph_path,
    trace_path,
    expected_hash,
):
    if not (
        os.path.exists(
            private_result_path
        )
        and os.path.exists(
            graph_path
        )
        and os.path.exists(
            trace_path
        )
    ):
        return False

    try:
        result = load_json(
            private_result_path
        )

        return (
            str(
                result.get(
                    "schema_version",
                    "",
                )
            )
            ==
            SCHEMA_VERSION
            and str(
                result.get(
                    "configuration_hash",
                    "",
                )
            )
            ==
            str(
                expected_hash
            )
            and str(
                result.get(
                    "phase",
                    "",
                )
            )
            ==
            "private_only_complete"
        )

    except Exception:
        return False


def run_private_variant(
    epsilon,
    seed,
    variant,
    frozen_record,
    private_inputs,
    candidate_indices,
    candidate_scores,
    experiment_root,
    force=False,
):
    folder = variant_folder(
        experiment_root=experiment_root,
        epsilon=epsilon,
        seed=seed,
        variant=variant,
    )

    os.makedirs(
        folder,
        exist_ok=True,
    )

    private_result_path = os.path.join(
        folder,
        "private_ablation_result.json",
    )

    graph_path = os.path.join(
        folder,
        "synthetic_graph.gml.gz",
    )

    trace_path = os.path.join(
        folder,
        "private_selection_trace.csv",
    )

    config = build_variant_configuration(
        epsilon=epsilon,
        seed=seed,
        variant=variant,
        private_inputs=private_inputs,
    )

    config_hash = stable_json_hash(
        config
    )

    if (
        not force
        and cached_private_phase_is_valid(
            private_result_path=private_result_path,
            graph_path=graph_path,
            trace_path=trace_path,
            expected_hash=config_hash,
        )
    ):
        result = load_json(
            private_result_path
        )

        result[
            "private_cache_reused"
        ] = True

        return result

    start = time.perf_counter()

    client_ids = private_inputs[
        "client_ids"
    ]

    private_signatures = private_inputs[
        "private_signatures"
    ]

    base_targets = clone_private_targets(
        private_inputs[
            "private_targets"
        ]
    )

    content_capacity = int(
        config[
            "content_capacity"
        ]
    )


    if variant == "full_hks":

        frozen = frozen_full_artifacts(
            private_inputs[
                "private_run_folder"
            ]
        )

        shutil.copy2(
            frozen[
                "graph_path"
            ],
            graph_path,
        )

        shutil.copy2(
            frozen[
                "trace_path"
            ],
            trace_path,
        )

        frozen_private_objective = float(
            frozen_record[
                "best_private_objective"
            ]
        )

        summary = frozen[
            "summary"
        ]

        summary_objective = summary.get(
            "best_private_objective",
            frozen_private_objective,
        )

        if not np.isclose(
            float(
                summary_objective
            ),
            frozen_private_objective,
            atol=REPLAY_TOLERANCE,
            rtol=0.0,
        ):
            raise RuntimeError(
                "Frozen full-method summary and final synthesis record "
                "disagree on the best private objective."
            )

        parameters = summary.get(
            "best_parameters",
            {},
        )

        private_components = summary.get(
            "best_objective_components",
            {},
        )

        selection_method = "frozen_rq4_reuse"

        private_objective = (
            frozen_private_objective
        )

        source_artifacts = {
            "source_graph_path": frozen[
                "graph_path"
            ],
            "source_trace_path": frozen[
                "trace_path"
            ],
            "source_summary_path": frozen[
                "summary_path"
            ],
        }


    elif variant == "no_private_bo":

        targets = base_targets

        parameters = fixed_private_parameters(
            candidate_scores
        )

        reconstruction = (
            reconstruct_graph_from_private_reports(
                client_ids=client_ids,
                private_signatures=private_signatures,
                private_targets=targets,
                candidate_indices=candidate_indices,
                candidate_scores=candidate_scores,
                parameters=parameters,
                content_capacity=content_capacity,
            )
        )

        objective = compute_private_objective(
            reconstruction=reconstruction,
            private_targets=targets,
            private_signatures=private_signatures,
        )

        graph = reconstruction[
            "graph"
        ]

        nx.write_gml(
            graph,
            graph_path,
        )

        trace = pd.DataFrame(
            [
                {
                    "iteration": 0,
                    **parameters,
                    **objective,
                }
            ]
        )

        trace.to_csv(
            trace_path,
            index=False,
        )

        private_objective = float(
            objective[
                "private_objective"
            ]
        )

        private_components = objective
        selection_method = "fixed_private_parameters"
        source_artifacts = {}


    else:

        if variant == "untyped_total_degree":

            targets = (
                collapse_and_publicly_resplit_degree(
                    private_targets=base_targets,
                    client_fraction=(
                        UNTYPED_PUBLIC_CLIENT_FRACTION
                    ),
                )
            )

        elif variant == "no_shared_content":

            targets = base_targets

        else:
            raise ValueError(
                f"Unsupported final ablation variant: {variant}"
            )

        result = run_private_bayesian_optimization(
            client_ids=client_ids,
            private_signatures=private_signatures,
            private_targets=targets,
            candidate_indices=candidate_indices,
            candidate_scores=candidate_scores,
            seed=int(
                seed
            ),
            evaluations=FROZEN_BO_EVALUATIONS,
            initial_points=FROZEN_BO_INITIAL_POINTS,
            acquisition_candidates=(
                FROZEN_BO_ACQUISITION_CANDIDATES
            ),
            content_capacity=content_capacity,
        )

        graph = result[
            "best_reconstruction"
        ][
            "graph"
        ]

        nx.write_gml(
            graph,
            graph_path,
        )

        result[
            "trace"
        ].to_csv(
            trace_path,
            index=False,
        )

        parameters = result[
            "best_parameters"
        ]

        private_objective = float(
            result[
                "best_private_objective"
            ]
        )

        private_components = result[
            "best_objective_components"
        ]

        selection_method = "private_bayesian_optimization"
        source_artifacts = {}

    if not os.path.exists(
        graph_path
    ):
        raise RuntimeError(
            "Private phase failed to finalize the synthetic graph."
        )

    if not os.path.exists(
        trace_path
    ):
        raise RuntimeError(
            "Private phase failed to finalize the selection trace."
        )

    runtime = float(
        time.perf_counter()
        -
        start
    )

    graph_check = nx.read_gml(
        graph_path
    )

    private_result = {
        "schema_version": SCHEMA_VERSION,
        "phase": "private_only_complete",
        "created_at_utc": utc_now_iso(),
        "configuration_hash": config_hash,
        "configuration": config,
        "dataset": DATASET,
        "epsilon_total": float(
            epsilon
        ),
        "privacy_seed": int(
            seed
        ),
        "variant": variant,
        "description": VARIANT_DESCRIPTIONS[
            variant
        ],
        "selection_method": selection_method,
        "private_objective": float(
            private_objective
        ),
        "private_components": private_components,
        "parameters": parameters,
        "content_capacity": content_capacity,
        "synthetic_nodes": int(
            graph_check.number_of_nodes()
        ),
        "synthetic_edges": int(
            graph_check.number_of_edges()
        ),
        "private_runtime_seconds": runtime,
        "private_reports_sha256": private_inputs[
            "reports_sha256"
        ],
        "degree_postprocessing": private_inputs[
            "private_targets"
        ].get(
            "degree_postprocessing",
            {},
        ),
        "privacy_statement": {
            "new_local_release_generated": False,
            "source": (
                "same already-private frozen RQ4 HKS/degree report"
            ),
            "server_operations_are_postprocessing": True,
            "additional_privacy_cost": 0.0,
            "oracle_available_during_private_phase": False,
        },
        "outputs": {
            "synthetic_graph": graph_path,
            "private_trace": trace_path,
            **source_artifacts,
        },
    }

    safe_json_dump(
        private_result,
        private_result_path,
    )

    return private_result


def original_graph_path(
    output_folder,
):
    return os.path.join(
        output_folder,
        DATASET,
        f"{DATASET}_global_graph.gml.gz",
    )


def load_and_validate_original_graph(
    output_folder,
):
    path = original_graph_path(
        output_folder
    )

    if not os.path.exists(
        path
    ):
        raise FileNotFoundError(
            "Missing frozen Enron original graph:\n"
            f"  {path}"
        )

    graph = nx.read_gml(
        path
    )

    if graph.number_of_nodes() != EXPECTED_ENRON_NODES:
        raise RuntimeError(
            "Wrong Enron graph population: "
            f"{graph.number_of_nodes()} nodes; "
            f"expected {EXPECTED_ENRON_NODES}."
        )

    if graph.number_of_edges() != EXPECTED_ENRON_EDGES:
        raise RuntimeError(
            "Wrong Enron graph population: "
            f"{graph.number_of_edges()} edges; "
            f"expected {EXPECTED_ENRON_EDGES}."
        )

    client_count = sum(
        1
        for _, data
        in graph.nodes(
            data=True
        )
        if data.get(
            "role"
        )
        ==
        "client"
    )

    content_count = sum(
        1
        for _, data
        in graph.nodes(
            data=True
        )
        if data.get(
            "role"
        )
        ==
        "content"
    )

    if client_count != EXPECTED_ENRON_CLIENTS:
        raise RuntimeError(
            "Wrong frozen Enron client population: "
            f"{client_count}; expected {EXPECTED_ENRON_CLIENTS}."
        )

    if content_count != EXPECTED_ENRON_CONTENT:
        raise RuntimeError(
            "Wrong frozen Enron content population: "
            f"{content_count}; expected {EXPECTED_ENRON_CONTENT}."
        )

    return graph


def flatten_oracle_metrics(
    oracle,
):
    objective = oracle[
        "oracle_objective"
    ]

    recovery = oracle[
        "client_edge_recovery"
    ]

    scalar = oracle[
        "scalar_comparison"
    ]

    return {
        "oracle_objective": float(
            objective[
                "oracle_objective"
            ]
        ),
        "oracle_total_loss": float(
            objective[
                "oracle_total_loss"
            ]
        ),
        "oracle_degree_loss": float(
            objective[
                "oracle_degree_loss"
            ]
        ),
        "oracle_relation_loss": float(
            objective[
                "oracle_relation_loss"
            ]
        ),
        "oracle_clustering_loss": float(
            objective[
                "oracle_clustering_loss"
            ]
        ),
        "oracle_modularity_loss": float(
            objective[
                "oracle_modularity_loss"
            ]
        ),
        "oracle_metapath_loss": float(
            objective[
                "oracle_metapath_loss"
            ]
        ),
        "oracle_structural_mmd_loss": float(
            objective[
                "oracle_structural_mmd_loss"
            ]
        ),
        "client_edge_precision": float(
            recovery[
                "precision"
            ]
        ),
        "client_edge_recall": float(
            recovery[
                "recall"
            ]
        ),
        "client_edge_f1": float(
            recovery[
                "f1"
            ]
        ),
        "client_edge_jaccard": float(
            recovery[
                "jaccard"
            ]
        ),
        "node_count_relative_error": float(
            scalar[
                "node_count_relative_error"
            ]
        ),
        "edge_count_relative_error": float(
            scalar[
                "edge_count_relative_error"
            ]
        ),
        "clustering_relative_error": float(
            scalar[
                "clustering_relative_error"
            ]
        ),
    }


def run_oracle_phase(
    private_result,
    original_graph,
    force=False,
):
    graph_path = private_result[
        "outputs"
    ][
        "synthetic_graph"
    ]

    folder = os.path.dirname(
        graph_path
    )

    final_result_path = os.path.join(
        folder,
        "ablation_result.json",
    )

    oracle_folder = os.path.join(
        folder,
        "oracle",
    )

    expected_oracle_hash = stable_json_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "private_configuration_hash": private_result[
                "configuration_hash"
            ],
            "synthetic_graph_sha256": sha256_file(
                graph_path
            ),
            "oracle_seed": FROZEN_ORACLE_SEED,
        }
    )

    if (
        not force
        and os.path.exists(
            final_result_path
        )
    ):
        try:
            cached = load_json(
                final_result_path
            )

            if (
                cached.get(
                    "schema_version"
                )
                ==
                SCHEMA_VERSION
                and cached.get(
                    "oracle_configuration_hash"
                )
                ==
                expected_oracle_hash
                and cached.get(
                    "phase"
                )
                ==
                "posthoc_oracle_complete"
            ):
                cached[
                    "oracle_cache_reused"
                ] = True

                return cached

        except Exception:
            pass

    synthetic_graph = nx.read_gml(
        graph_path
    )

    start = time.perf_counter()

    oracle = compute_oracle_metrics(
        original_graph=original_graph,
        synthetic_graph=synthetic_graph,
        dataset_name=DATASET,
        seed=FROZEN_ORACLE_SEED,
        output_folder=oracle_folder,
    )

    oracle_runtime = float(
        time.perf_counter()
        -
        start
    )

    final_result = {
        "schema_version": SCHEMA_VERSION,
        "phase": "posthoc_oracle_complete",
        "created_at_utc": utc_now_iso(),
        "configuration_hash": private_result[
            "configuration_hash"
        ],
        "oracle_configuration_hash": expected_oracle_hash,
        "dataset": private_result[
            "dataset"
        ],
        "epsilon": private_result[
            "epsilon_total"
        ],
        "seed": private_result[
            "privacy_seed"
        ],
        "variant": private_result[
            "variant"
        ],
        "description": private_result[
            "description"
        ],
        "selection_method": private_result[
            "selection_method"
        ],
        "private_objective": private_result[
            "private_objective"
        ],
        "parameters": private_result[
            "parameters"
        ],
        "content_capacity": private_result[
            "content_capacity"
        ],
        "synthetic_nodes": private_result[
            "synthetic_nodes"
        ],
        "synthetic_edges": private_result[
            "synthetic_edges"
        ],
        "private_runtime_seconds": private_result[
            "private_runtime_seconds"
        ],
        "oracle_runtime_seconds": oracle_runtime,
        "oracle_seed": FROZEN_ORACLE_SEED,
        "privacy_statement": private_result[
            "privacy_statement"
        ],
        "oracle_statement": {
            "role": "post-hoc evaluation only",
            "loaded_only_after_all_private_graphs_were_frozen": True,
            "changes_private_trace_or_graph": False,
        },
        **flatten_oracle_metrics(
            oracle
        ),
    }

    safe_json_dump(
        final_result,
        final_result_path,
    )

    return final_result


def aggregate_ablation_results(
    seed_df,
):
    if seed_df.empty:
        return pd.DataFrame()

    group_columns = [
        "dataset",
        "epsilon",
        "variant",
        "description",
    ]

    excluded = {
        "epsilon",
        "seed",
        "oracle_seed",
    }

    numeric_columns = [
        column
        for column
        in seed_df.columns
        if (
            pd.api.types.is_numeric_dtype(
                seed_df[
                    column
                ]
            )
            and column not in excluded
        )
    ]

    rows = []

    for group_values, group in seed_df.groupby(
        group_columns,
        dropna=False,
        sort=True,
    ):
        row = dict(
            zip(
                group_columns,
                group_values,
            )
        )

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
                int(x)
                for x
                in group[
                    "seed"
                ].unique()
            )
        )

        for metric in numeric_columns:

            summary = confidence_interval(
                pd.to_numeric(
                    group[
                        metric
                    ],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )
            )

            for name, value in summary.items():

                row[
                    f"{metric}__{name}"
                ] = value

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def compare_against_full(
    seed_df,
):
    metrics = [
        "oracle_objective",
        "oracle_total_loss",
        "oracle_degree_loss",
        "oracle_relation_loss",
        "oracle_clustering_loss",
        "oracle_modularity_loss",
        "oracle_metapath_loss",
        "oracle_structural_mmd_loss",
        "client_edge_f1",
        "client_edge_jaccard",
        "node_count_relative_error",
        "edge_count_relative_error",
        "clustering_relative_error",
    ]

    rows = []

    for (
        dataset,
        epsilon,
    ), epsilon_group in seed_df.groupby(
        [
            "dataset",
            "epsilon",
        ],
        dropna=False,
        sort=True,
    ):

        full = epsilon_group[
            epsilon_group[
                "variant"
            ]
            ==
            "full_hks"
        ].copy()

        if full.empty:
            continue

        for variant in sorted(
            set(
                epsilon_group[
                    "variant"
                ]
            )
            -
            {
                "full_hks"
            }
        ):

            alternative = epsilon_group[
                epsilon_group[
                    "variant"
                ]
                ==
                variant
            ].copy()

            shared_seeds = sorted(
                set(
                    full[
                        "seed"
                    ]
                )
                &
                set(
                    alternative[
                        "seed"
                    ]
                )
            )

            if not shared_seeds:
                continue

            full_by_seed = (
                full
                .set_index(
                    "seed"
                )
                .loc[
                    shared_seeds
                ]
            )

            alternative_by_seed = (
                alternative
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
                    not in full_by_seed.columns
                    or metric
                    not in alternative_by_seed.columns
                ):
                    continue

                full_values = pd.to_numeric(
                    full_by_seed[
                        metric
                    ],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )

                alternative_values = pd.to_numeric(
                    alternative_by_seed[
                        metric
                    ],
                    errors="coerce",
                ).to_numpy(
                    dtype=float
                )

                mask = (
                    np.isfinite(
                        full_values
                    )
                    &
                    np.isfinite(
                        alternative_values
                    )
                )

                full_values = full_values[
                    mask
                ]

                alternative_values = (
                    alternative_values[
                        mask
                    ]
                )

                if len(
                    full_values
                ) == 0:
                    continue

                difference = (
                    full_values
                    -
                    alternative_values
                )

                ci = confidence_interval(
                    difference
                )

                rows.append(
                    {
                        "dataset": dataset,
                        "epsilon": float(
                            epsilon
                        ),
                        "full_variant": "full_hks",
                        "ablation_variant": variant,
                        "metric": metric,
                        "paired_seeds": int(
                            len(
                                difference
                            )
                        ),
                        "mean_full": float(
                            np.mean(
                                full_values
                            )
                        ),
                        "mean_ablation": float(
                            np.mean(
                                alternative_values
                            )
                        ),
                        "mean_difference_full_minus_ablation": float(
                            np.mean(
                                difference
                            )
                        ),
                        "difference_ci95_low": ci[
                            "ci95_low"
                        ],
                        "difference_ci95_high": ci[
                            "ci95_high"
                        ],
                        "cohen_dz": cohen_dz(
                            difference
                        ),
                        "wilcoxon_p_value_exploratory": (
                            paired_wilcoxon(
                                full_values,
                                alternative_values,
                            )
                        ),
                        "wilcoxon_note": (
                            "With five paired seeds, the two-sided exact "
                            "Wilcoxon test has very coarse resolution; "
                            "effect sizes and paired CIs are primary."
                        ),
                    }
                )

    return pd.DataFrame(
        rows
    )


def build_contribution_table(
    aggregate_df,
):
    if aggregate_df.empty:
        return pd.DataFrame()

    rows = []

    for (
        dataset,
        epsilon,
    ), group in aggregate_df.groupby(
        [
            "dataset",
            "epsilon",
        ],
        dropna=False,
        sort=True,
    ):

        full = group[
            group[
                "variant"
            ]
            ==
            "full_hks"
        ]

        if full.empty:
            continue

        full = full.iloc[
            0
        ]

        for _, alternative in group.iterrows():

            variant = str(
                alternative[
                    "variant"
                ]
            )

            if variant == "full_hks":
                continue

            rows.append(
                {
                    "dataset": dataset,
                    "epsilon": float(
                        epsilon
                    ),
                    "ablation_variant": variant,
                    "description": VARIANT_DESCRIPTIONS[
                        variant
                    ],
                    "oracle_objective_full": float(
                        full[
                            "oracle_objective__mean"
                        ]
                    ),
                    "oracle_objective_ablation": float(
                        alternative[
                            "oracle_objective__mean"
                        ]
                    ),
                    "oracle_objective_gain_full_minus_ablation": float(
                        full[
                            "oracle_objective__mean"
                        ]
                        -
                        alternative[
                            "oracle_objective__mean"
                        ]
                    ),
                    "oracle_loss_reduction_full_vs_ablation": float(
                        alternative[
                            "oracle_total_loss__mean"
                        ]
                        -
                        full[
                            "oracle_total_loss__mean"
                        ]
                    ),
                    "client_edge_f1_gain_full_minus_ablation": float(
                        full[
                            "client_edge_f1__mean"
                        ]
                        -
                        alternative[
                            "client_edge_f1__mean"
                        ]
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def run_final_ablation(
    output_folder=DEFAULT_OUTPUT_FOLDER,
    epsilons=DEFAULT_EPSILONS,
    seeds=DEFAULT_SEEDS,
    variants=DEFAULT_VARIANTS,
    force=False,
    allow_test_output=False,
):
    ensure_final_output_scope(
        output_folder=output_folder,
        allow_test_output=allow_test_output,
    )

    epsilons = tuple(
        float(x)
        for x
        in epsilons
    )

    seeds = tuple(
        int(x)
        for x
        in seeds
    )

    variants = tuple(
        str(x)
        for x
        in variants
    )

    unknown = (
        set(
            variants
        )
        -
        set(
            VARIANT_DESCRIPTIONS
        )
    )

    if unknown:
        raise ValueError(
            f"Unknown ablation variants: {sorted(unknown)}"
        )

    if "full_hks" not in variants:
        raise ValueError(
            "The final ablation must contain full_hks as the "
            "frozen reference."
        )

    for epsilon in epsilons:

        if epsilon <= 0:
            raise ValueError(
                "Every epsilon must be positive."
            )

    records = load_frozen_synthesis_records(
        output_folder
    )

    experiment_root = os.path.join(
        output_folder,
        "journal_final_ablation",
        DATASET,
    )

    os.makedirs(
        experiment_root,
        exist_ok=True,
    )

    private_manifest_path = os.path.join(
        experiment_root,
        "ablation_private_phase_records.csv",
    )

    seed_results_path = os.path.join(
        experiment_root,
        "ablation_seed_results.csv",
    )

    aggregate_path = os.path.join(
        experiment_root,
        "ablation_aggregate.csv",
    )

    significance_path = os.path.join(
        experiment_root,
        "ablation_paired_comparisons.csv",
    )

    contribution_path = os.path.join(
        experiment_root,
        "ablation_component_contributions.csv",
    )

    metadata_path = os.path.join(
        experiment_root,
        "ablation_metadata.json",
    )

    planned = (
        len(
            epsilons
        )
        *
        len(
            seeds
        )
        *
        len(
            variants
        )
    )

    failures = []

    private_results = []

    experiment_start = time.perf_counter()

    print()
    print("=" * 88)
    print("FINAL RQ6 STRUCTURAL ABLATION")
    print("=" * 88)
    print(f"Dataset: {DATASET}")
    print(f"Epsilons: {list(epsilons)}")
    print(f"Privacy seeds: {list(seeds)}")
    print(f"Variants: {list(variants)}")
    print(f"Planned variant evaluations: {planned}")
    print(
        "Privacy release: REUSE frozen RQ4 reports only; "
        "no new local release."
    )
    print(
        "Oracle graph: will NOT be loaded until every private graph "
        "is frozen."
    )
    print("=" * 88)


    print()
    print("PHASE A: PRIVATE-ONLY ABLATIONS")
    print("-" * 88)

    for epsilon in epsilons:

        for seed in seeds:

            try:
                frozen_record = select_frozen_record(
                    records=records,
                    epsilon=epsilon,
                    seed=seed,
                )

                private_inputs = (
                    prepare_frozen_private_inputs(
                        frozen_record=frozen_record,
                        epsilon=epsilon,
                        seed=seed,
                    )
                )


                candidate_indices = None
                candidate_scores = None

                non_full_requested = any(
                    variant != "full_hks"
                    for variant
                    in variants
                )

                if non_full_requested:

                    (
                        candidate_indices,
                        candidate_scores,
                    ) = compute_exact_top_candidates(
                        private_signatures=(
                            private_inputs[
                                "private_signatures"
                            ]
                        ),
                        max_candidates=(
                            FROZEN_MAX_CANDIDATES
                        ),
                    )

                for variant in variants:

                    try:
                        print(
                            f"[private] eps={epsilon:g} "
                            f"seed={seed} "
                            f"variant={variant}"
                        )

                        result = run_private_variant(
                            epsilon=epsilon,
                            seed=seed,
                            variant=variant,
                            frozen_record=frozen_record,
                            private_inputs=private_inputs,
                            candidate_indices=(
                                candidate_indices
                            ),
                            candidate_scores=(
                                candidate_scores
                            ),
                            experiment_root=(
                                experiment_root
                            ),
                            force=force,
                        )

                        private_results.append(
                            result
                        )

                        pd.DataFrame(
                            [
                                {
                                    "dataset": item[
                                        "dataset"
                                    ],
                                    "epsilon": item[
                                        "epsilon_total"
                                    ],
                                    "seed": item[
                                        "privacy_seed"
                                    ],
                                    "variant": item[
                                        "variant"
                                    ],
                                    "phase": item[
                                        "phase"
                                    ],
                                    "configuration_hash": item[
                                        "configuration_hash"
                                    ],
                                    "private_objective": item[
                                        "private_objective"
                                    ],
                                    "synthetic_nodes": item[
                                        "synthetic_nodes"
                                    ],
                                    "synthetic_edges": item[
                                        "synthetic_edges"
                                    ],
                                    "private_runtime_seconds": item[
                                        "private_runtime_seconds"
                                    ],
                                    "synthetic_graph": item[
                                        "outputs"
                                    ][
                                        "synthetic_graph"
                                    ],
                                }
                                for item
                                in private_results
                            ]
                        ).to_csv(
                            private_manifest_path,
                            index=False,
                        )

                    except Exception as exc:

                        failures.append(
                            {
                                "phase": "private",
                                "epsilon": float(
                                    epsilon
                                ),
                                "seed": int(
                                    seed
                                ),
                                "variant": variant,
                                "error": repr(
                                    exc
                                ),
                            }
                        )

                        print(
                            "[ablation] PRIVATE FAILURE: "
                            f"{repr(exc)}"
                        )

            except Exception as exc:

                for variant in variants:

                    failures.append(
                        {
                            "phase": "private_set",
                            "epsilon": float(
                                epsilon
                            ),
                            "seed": int(
                                seed
                            ),
                            "variant": variant,
                            "error": repr(
                                exc
                            ),
                        }
                    )

                print(
                    "[ablation] PRIVATE SET FAILURE "
                    f"eps={epsilon:g} seed={seed}: {repr(exc)}"
                )

    expected_private = planned

    if len(
        private_results
    ) != expected_private:

        safe_json_dump(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "private_phase_incomplete",
                "planned": expected_private,
                "successful": len(
                    private_results
                ),
                "failures": failures,
            },
            metadata_path,
        )

        raise RuntimeError(
            "Private ablation phase incomplete. "
            f"Successful={len(private_results)}/{expected_private}. "
            "The original graph has NOT been loaded."
        )


    for item in private_results:

        graph_path = item[
            "outputs"
        ][
            "synthetic_graph"
        ]

        trace_path = item[
            "outputs"
        ][
            "private_trace"
        ]

        if not (
            os.path.exists(
                graph_path
            )
            and os.path.exists(
                trace_path
            )
        ):
            raise RuntimeError(
                "Private-phase checkpoint failed before oracle loading."
            )

    print()
    print(
        f"Private phase complete: {len(private_results)}/{planned}."
    )
    print(
        "All synthetic graphs and private traces are frozen."
    )


    print()
    print("PHASE B: POST-HOC ORACLE EVALUATION")
    print("-" * 88)


    original_graph = load_and_validate_original_graph(
        output_folder
    )

    final_results = []

    for private_result in private_results:

        epsilon = float(
            private_result[
                "epsilon_total"
            ]
        )

        seed = int(
            private_result[
                "privacy_seed"
            ]
        )

        variant = str(
            private_result[
                "variant"
            ]
        )

        try:
            print(
                f"[oracle] eps={epsilon:g} "
                f"seed={seed} "
                f"variant={variant}"
            )

            final = run_oracle_phase(
                private_result=private_result,
                original_graph=original_graph,
                force=force,
            )

            final_results.append(
                final
            )

            pd.DataFrame(
                final_results
            ).to_csv(
                seed_results_path,
                index=False,
            )

        except Exception as exc:

            failures.append(
                {
                    "phase": "oracle",
                    "epsilon": epsilon,
                    "seed": seed,
                    "variant": variant,
                    "error": repr(
                        exc
                    ),
                }
            )

            print(
                "[ablation] ORACLE FAILURE: "
                f"{repr(exc)}"
            )

    seed_df = pd.DataFrame(
        final_results
    )

    if len(
        seed_df
    ) != planned:

        safe_json_dump(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "oracle_phase_incomplete",
                "planned": planned,
                "private_phase_successful": len(
                    private_results
                ),
                "oracle_successful": len(
                    seed_df
                ),
                "failures": failures,
            },
            metadata_path,
        )

        raise RuntimeError(
            "Oracle phase incomplete: "
            f"{len(seed_df)}/{planned} results."
        )

    seed_df = (
        seed_df
        .sort_values(
            [
                "epsilon",
                "variant",
                "seed",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    seed_df.to_csv(
        seed_results_path,
        index=False,
    )

    aggregate_df = aggregate_ablation_results(
        seed_df
    )

    aggregate_df.to_csv(
        aggregate_path,
        index=False,
    )

    paired_df = compare_against_full(
        seed_df
    )

    paired_df.to_csv(
        significance_path,
        index=False,
    )

    contribution_df = build_contribution_table(
        aggregate_df
    )

    contribution_df.to_csv(
        contribution_path,
        index=False,
    )

    total_runtime = float(
        time.perf_counter()
        -
        experiment_start
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "status": "complete",
        "dataset": DATASET,
        "frozen_population": {
            "nodes": EXPECTED_ENRON_NODES,
            "edges": EXPECTED_ENRON_EDGES,
            "clients": EXPECTED_ENRON_CLIENTS,
            "content": EXPECTED_ENRON_CONTENT,
            "description": (
                "deterministic first-10,000-message Enron population "
                "used by the frozen journal experiments"
            ),
        },
        "epsilons": [
            float(x)
            for x
            in epsilons
        ],
        "privacy_seeds": [
            int(x)
            for x
            in seeds
        ],
        "variants": {
            variant: VARIANT_DESCRIPTIONS[
                variant
            ]
            for variant
            in variants
        },
        "reference_variant": "full_hks",
        "frozen_settings": {
            "signature_method_code_label": (
                FROZEN_SIGNATURE_METHOD
            ),
            "signature_description": (
                "heat-spectrum descriptor; legacy experiment label hks"
            ),
            "signature_mechanism": (
                FROZEN_SIGNATURE_MECHANISM
            ),
            "signature_fraction": (
                FROZEN_SIGNATURE_FRACTION
            ),
            "degree_fraction": (
                1.0
                -
                FROZEN_SIGNATURE_FRACTION
            ),
            "signature_clip_norm_C": (
                FROZEN_SIGNATURE_CLIP_NORM
            ),
            "D_max": FROZEN_D_MAX,
            "degree_mode": FROZEN_DEGREE_MODE,
            "degree_adjacency": (
                FROZEN_DEGREE_ADJACENCY
            ),
            "max_candidates_per_client": (
                FROZEN_MAX_CANDIDATES
            ),
            "bo_evaluations": (
                FROZEN_BO_EVALUATIONS
            ),
            "bo_initial_points": (
                FROZEN_BO_INITIAL_POINTS
            ),
            "bo_acquisition_candidates": (
                FROZEN_BO_ACQUISITION_CANDIDATES
            ),
            "content_capacity": (
                FROZEN_CONTENT_CAPACITY
            ),
            "fixed_no_bo_candidate_k": (
                FROZEN_FIXED_CANDIDATE_K
            ),
            "untyped_public_client_fraction": (
                UNTYPED_PUBLIC_CLIENT_FRACTION
            ),
            "oracle_seed": (
                FROZEN_ORACLE_SEED
            ),
        },
        "privacy_boundary": {
            "new_private_releases": 0,
            "all_variants_use_same_frozen_rq4_private_report": True,
            "degree_feasibility_projection": (
                "coordinatewise clip to [0,D_max] exactly as final "
                "synthesized_graph.py"
            ),
            "server_side_ablation_additional_privacy_cost": 0.0,
            "original_graph_loaded_after_private_phase": True,
            "oracle_influences_private_selection": False,
        },
        "scientific_scope": {
            "representation_ablation_excluded": True,
            "reason": (
                "RAW/ABS/eigen representation comparisons are already "
                "evaluated in RQ1/RQ2. RQ6 isolates reconstruction "
                "components using the same frozen HKS release."
            ),
            "untyped_ablation_rule": (
                "collapse private typed degree to total, then fixed "
                "public 50/50 resplit; preserves total degree exactly"
            ),
            "no_shared_content_rule": (
                "content capacity = 1"
            ),
            "no_bo_rule": (
                "median private candidate similarity, degree scales 1, "
                "candidate_k 50"
            ),
        },
        "statistical_design": {
            "independent_repetition_unit": "privacy seed",
            "num_seeds_per_cell": len(
                seeds
            ),
            "confidence_interval": (
                "Student-t 95% CI across privacy seeds"
            ),
            "paired_comparisons": (
                "paired by identical epsilon and privacy seed"
            ),
            "wilcoxon": (
                "two-sided, exploratory because n=5 gives coarse exact "
                "resolution"
            ),
            "effect_size": "paired Cohen's dz",
            "oracle_seed_fixed_across_every_run": True,
        },
        "planned_runs": int(
            planned
        ),
        "successful_private_runs": int(
            len(
                private_results
            )
        ),
        "successful_oracle_runs": int(
            len(
                seed_df
            )
        ),
        "failed_runs": int(
            len(
                failures
            )
        ),
        "failures": failures,
        "runtime_seconds": total_runtime,
        "outputs": {
            "private_phase_records": (
                private_manifest_path
            ),
            "seed_results": (
                seed_results_path
            ),
            "aggregate": (
                aggregate_path
            ),
            "paired_comparisons": (
                significance_path
            ),
            "component_contributions": (
                contribution_path
            ),
        },
    }

    safe_json_dump(
        metadata,
        metadata_path,
    )

    print()
    print("=" * 88)
    print("FINAL RQ6 ABLATION COMPLETE")
    print("=" * 88)
    print(f"Successful private runs: {len(private_results)}/{planned}")
    print(f"Successful oracle runs:  {len(seed_df)}/{planned}")
    print(f"Failures:                {len(failures)}")
    print(f"Fixed oracle seed:       {FROZEN_ORACLE_SEED}")
    print()
    print("Mean oracle objective:")
    print()

    display_columns = [
        "epsilon",
        "variant",
        "oracle_objective__mean",
        "oracle_objective__ci95_low",
        "oracle_objective__ci95_high",
    ]

    print(
        aggregate_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()
    print("Outputs:")
    print(f"  {seed_results_path}")
    print(f"  {aggregate_path}")
    print(f"  {significance_path}")
    print(f"  {contribution_path}")
    print(f"  {metadata_path}")
    print("=" * 88)

    return {
        "seed_results": seed_df,
        "aggregate": aggregate_df,
        "paired_comparisons": paired_df,
        "contributions": contribution_df,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Enron RQ6 structural ablation using only "
            "the already-generated RQ4 private reports."
        )
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
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
        "--variants",
        nargs="+",
        choices=list(
            VARIANT_DESCRIPTIONS.keys()
        ),
        default=list(
            DEFAULT_VARIANTS
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Recompute final RQ6 ablation artifacts even if a "
            "configuration-matching cache exists."
        ),
    )

    parser.add_argument(
        "--allow-test-output",
        action="store_true",
        help=(
            "Allow an intentional smoke test inside a path containing "
            "output_local_test/local_test/smoke_test/debug_output."
        ),
    )

    args = parser.parse_args()

    run_final_ablation(
        output_folder=args.output_folder,
        epsilons=tuple(
            float(x)
            for x
            in args.epsilons
        ),
        seeds=tuple(
            int(x)
            for x
            in args.seeds
        ),
        variants=tuple(
            str(x)
            for x
            in args.variants
        ),
        force=args.force,
        allow_test_output=(
            args.allow_test_output
        ),
    )


if __name__ == "__main__":
    main()
