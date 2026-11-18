import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import networkx as nx
import numpy as np


from preprocess_data import (
    PREPROCESS_SCHEMA_VERSION,
    preprocess_data,
)

from create_global import (
    DEFAULT_D_MAX,
    GRAPH_SCHEMA_VERSION,
    create_global_graph,
)

from to_ego_graphs import (
    EGO_SCHEMA_VERSION,
    create_ego_graphs,
)

from local_signatures import (
    SIGNATURE_SCHEMA_VERSION,
    compute_local_signatures,
)

from noisy_ego_graphs import (
    DEFAULT_SIGNATURE_CLIP_NORM,
    DEFAULT_SIGNATURE_FRACTION,
    PRIVATE_REPORT_SCHEMA_VERSION,
    process_ego_graphs,
)

from synthesized_graph import (
    DEFAULT_BO_ACQUISITION_CANDIDATES,
    DEFAULT_BO_EVALUATIONS,
    DEFAULT_BO_INITIAL_POINTS,
    DEFAULT_CONTENT_CAPACITY,
    SYNTHESIS_SCHEMA_VERSION,
    compute_synthesized_global_graph,
)

from stats import (
    STATS_SCHEMA_VERSION,
    compute_metrics,
)


PIPELINE_SCHEMA_VERSION = "3.0"

DEFAULT_OUTPUT_FOLDER = "output/"
DEFAULT_DATA_DIR = "data"

DEFAULT_SEED = 0

DEFAULT_SIGNATURE_METHOD = "hks"
DEFAULT_SIGNATURE_MECHANISM = "l2_knorm"

DEFAULT_DEGREE_MODE = "role"
DEFAULT_DEGREE_ADJACENCY = "client_report"


def utc_now_iso():


    return datetime.now(
        timezone.utc
    ).isoformat()


def sha256_file(
    path,
    chunk_size=1024 * 1024,
):


    if not os.path.exists(
        path
    ):
        return None

    digest = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        while True:

            chunk = f.read(
                chunk_size
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def run_stage(
    stage_name,
    function,
    *args,
    **kwargs,
):


    print(
        "\n"
        "========================================"
    )

    print(
        stage_name.upper()
    )

    print(
        "========================================"
    )

    start = time.perf_counter()

    result = function(
        *args,
        **kwargs,
    )

    runtime = float(
        time.perf_counter()
        -
        start
    )

    print(
        f"\n[{stage_name}] "
        f"completed in "
        f"{runtime:.3f} s"
    )

    return (
        result,
        runtime,
    )


def validate_global_graph(
    graph,
):


    if graph is None:

        raise RuntimeError(
            "Global graph construction returned None."
        )

    if not graph.is_directed():

        raise RuntimeError(
            "Revised global graph must be directed."
        )

    if not graph.is_multigraph():

        raise RuntimeError(
            "Revised global graph must preserve "
            "multiple typed edges."
        )

    if graph.number_of_nodes() == 0:

        raise RuntimeError(
            "Global graph contains no nodes."
        )

    client_nodes = [
        node
        for node, data
        in graph.nodes(
            data=True
        )
        if data.get(
            "role"
        )
        == "client"
    ]

    if not client_nodes:

        raise RuntimeError(
            "Global graph contains no client nodes."
        )

    missing_type = sum(
        1
        for _, data
        in graph.nodes(
            data=True
        )
        if not data.get(
            "type"
        )
    )

    missing_role = sum(
        1
        for _, data
        in graph.nodes(
            data=True
        )
        if not data.get(
            "role"
        )
    )

    if missing_type:

        raise RuntimeError(
            f"{missing_type} nodes have no type."
        )

    if missing_role:

        raise RuntimeError(
            f"{missing_role} nodes have no role."
        )

    missing_relation = 0

    for _, _, _, data in graph.edges(
        keys=True,
        data=True,
    ):

        if not data.get(
            "relation"
        ):

            missing_relation += 1

    if missing_relation:

        raise RuntimeError(
            f"{missing_relation} edges have "
            "no relation label."
        )

    return {
        "nodes": int(
            graph.number_of_nodes()
        ),

        "edges": int(
            graph.number_of_edges()
        ),

        "clients": int(
            len(
                client_nodes
            )
        ),

        "content_nodes": int(
            sum(
                1
                for _, data
                in graph.nodes(
                    data=True
                )
                if data.get(
                    "role"
                )
                == "content"
            )
        ),
    }


def validate_private_report_result(
    private_result,
    expected_epsilon,
):


    client_ids = np.asarray(
        private_result[
            "client_ids"
        ],
        dtype=str,
    )

    private_signatures = np.asarray(
        private_result[
            "private_signatures"
        ],
        dtype=float,
    )

    private_degrees = np.asarray(
        private_result[
            "private_degrees"
        ],
        dtype=int,
    )

    configuration = private_result[
        "configuration"
    ]

    if len(
        client_ids
    ) == 0:

        raise RuntimeError(
            "No private client reports were generated."
        )

    if (
        private_signatures.shape[
            0
        ]
        != len(
            client_ids
        )
    ):

        raise RuntimeError(
            "Private signature count does not "
            "match client count."
        )

    if (
        private_degrees.shape[
            0
        ]
        != len(
            client_ids
        )
    ):

        raise RuntimeError(
            "Private degree count does not "
            "match client count."
        )

    if len(
        set(
            client_ids.tolist()
        )
    ) != len(
        client_ids
    ):

        raise RuntimeError(
            "A client appears more than once in "
            "the private report file."
        )

    epsilon_stored = float(
        configuration[
            "epsilon_total"
        ]
    )

    if not np.isclose(
        epsilon_stored,
        float(
            expected_epsilon
        ),
    ):

        raise RuntimeError(
            "Private-report epsilon does not match "
            "the requested pipeline epsilon."
        )

    epsilon_signature = float(
        configuration[
            "epsilon_signature"
        ]
    )

    epsilon_degree = float(
        configuration[
            "epsilon_degree"
        ]
    )

    if not np.isclose(
        epsilon_signature
        +
        epsilon_degree,
        epsilon_stored,
    ):

        raise RuntimeError(
            "Privacy-budget split does not sum "
            "to epsilon_total."
        )

    if configuration.get(
        "uses_personalised_epsilon",
        True,
    ):

        raise RuntimeError(
            "Personalized/data-dependent epsilon "
            "must not be enabled."
        )

    if not configuration.get(
        "one_private_report_per_client",
        False,
    ):

        raise RuntimeError(
            "Private-report metadata does not confirm "
            "one report per client."
        )

    return {
        "reports": int(
            len(
                client_ids
            )
        ),

        "signature_dimension": int(
            private_signatures.shape[
                1
            ]
        ),

        "degree_dimension": int(
            private_degrees.shape[
                1
            ]
        ),

        "epsilon_total": float(
            epsilon_stored
        ),

        "epsilon_signature": float(
            epsilon_signature
        ),

        "epsilon_degree": float(
            epsilon_degree
        ),
    }


def validate_synthetic_graph(
    graph,
):


    if graph is None:

        raise RuntimeError(
            "Synthesis returned None."
        )

    source = str(
        graph.graph.get(
            "source",
            ""
        )
    )

    if source != "private_reports_only":

        raise RuntimeError(
            "Synthetic graph is not marked as "
            "private-report-only reconstruction."
        )

    if graph.graph.get(
        "contains_original_content_edges",
        True,
    ):

        raise RuntimeError(
            "Synthetic graph metadata indicates "
            "original content edges were used."
        )

    if graph.graph.get(
        "contains_original_content_nodes",
        True,
    ):

        raise RuntimeError(
            "Synthetic graph metadata indicates "
            "original content nodes were reused."
        )

    synthetic_content_nodes = []

    invalid_content_nodes = []

    for node, data in graph.nodes(
        data=True
    ):

        if data.get(
            "role"
        ) != "content":

            continue

        synthetic_content_nodes.append(
            str(
                node
            )
        )

        if not data.get(
            "synthetic",
            False,
        ):

            invalid_content_nodes.append(
                str(
                    node
                )
            )

    if invalid_content_nodes:

        raise RuntimeError(
            "Synthetic graph contains content nodes "
            "not marked as synthetic."
        )

    return {
        "nodes": int(
            graph.number_of_nodes()
        ),

        "edges": int(
            graph.number_of_edges()
        ),

        "synthetic_content_nodes": int(
            len(
                synthetic_content_nodes
            )
        ),
    }


def save_pipeline_manifest(
    dataset_name,
    epsilon,
    seed,
    output_folder,
    private_run_folder,
    configuration,
    stage_runtimes,
    validation,
    preprocessing_result,
):


    manifest_path = os.path.join(
        private_run_folder,
        "pipeline_manifest.json",
    )

    normalized_nodes_path = os.path.join(
        output_folder,
        dataset_name,
        f"{dataset_name}_nodes.csv",
    )

    normalized_edges_path = os.path.join(
        output_folder,
        dataset_name,
        f"{dataset_name}_edges.csv",
    )

    global_graph_path = os.path.join(
        output_folder,
        dataset_name,
        f"{dataset_name}_global_graph.gml.gz",
    )

    local_signatures_path = os.path.join(
        output_folder,
        dataset_name,
        f"{dataset_name}_local_signatures.npz",
    )

    private_reports_path = os.path.join(
        private_run_folder,
        "private_reports.npz",
    )

    synthetic_graph_path = os.path.join(
        private_run_folder,
        "synthesis",
        "synthetic_graph.gml.gz",
    )

    manifest = {
        "pipeline_schema_version": (
            PIPELINE_SCHEMA_VERSION
        ),

        "created_utc": utc_now_iso(),

        "dataset": str(
            dataset_name
        ),

        "epsilon_total": float(
            epsilon
        ),

        "seed": int(
            seed
        ),

        "configuration": (
            configuration
        ),

        "component_schema_versions": {
            "preprocess": (
                PREPROCESS_SCHEMA_VERSION
            ),

            "global_graph": (
                GRAPH_SCHEMA_VERSION
            ),

            "ego_graphs": (
                EGO_SCHEMA_VERSION
            ),

            "local_signatures": (
                SIGNATURE_SCHEMA_VERSION
            ),

            "private_reports": (
                PRIVATE_REPORT_SCHEMA_VERSION
            ),

            "synthesis": (
                SYNTHESIS_SCHEMA_VERSION
            ),

            "statistics": (
                STATS_SCHEMA_VERSION
            ),
        },

        "stage_runtimes_seconds": {
            key: float(
                value
            )
            for key, value
            in stage_runtimes.items()
        },

        "validation": (
            validation
        ),

        "preprocessing": (
            preprocessing_result
        ),

        "artifacts": {
            "normalized_nodes": (
                normalized_nodes_path
            ),

            "normalized_edges": (
                normalized_edges_path
            ),

            "global_graph": (
                global_graph_path
            ),

            "local_signatures": (
                local_signatures_path
            ),

            "private_reports": (
                private_reports_path
            ),

            "synthetic_graph": (
                synthetic_graph_path
            ),
        },

        "artifact_sha256": {
            "normalized_nodes": (
                sha256_file(
                    normalized_nodes_path
                )
            ),

            "normalized_edges": (
                sha256_file(
                    normalized_edges_path
                )
            ),

            "global_graph": (
                sha256_file(
                    global_graph_path
                )
            ),

            "local_signatures": (
                sha256_file(
                    local_signatures_path
                )
            ),

            "private_reports": (
                sha256_file(
                    private_reports_path
                )
            ),

            "synthetic_graph": (
                sha256_file(
                    synthetic_graph_path
                )
            ),
        },

        "privacy_boundary": {
            "client_release": (
                "one privatized signature and one "
                "privatized typed-degree vector per client"
            ),

            "server_visible_file": (
                "private_reports.npz"
            ),

            "server_does_not_receive": [
                "clean ego graphs",
                "clean local signatures",
                "raw typed degrees",
                "original content-node identities",
                "original content edges",
                "oracle graph fidelity",
            ],

            "oracle_used_for_reconstruction": False,

            "oracle_used_for_private_bo": False,
        },
    }

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=4,
            ensure_ascii=False,
            default=str,
        )

    return manifest_path


def run_pipeline(
    dataset_name,
    epsilon=3.0,
    seed=DEFAULT_SEED,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    data_dir=DEFAULT_DATA_DIR,
    force_rebuild=False,


    signature_method=DEFAULT_SIGNATURE_METHOD,
    signature_mechanism=DEFAULT_SIGNATURE_MECHANISM,
    signature_fraction=DEFAULT_SIGNATURE_FRACTION,
    signature_clip_norm=DEFAULT_SIGNATURE_CLIP_NORM,


    d_max=DEFAULT_D_MAX,
    degree_mode=DEFAULT_DEGREE_MODE,
    degree_adjacency=DEFAULT_DEGREE_ADJACENCY,


    bo_evaluations=DEFAULT_BO_EVALUATIONS,
    bo_initial_points=DEFAULT_BO_INITIAL_POINTS,
    bo_acquisition_candidates=(
        DEFAULT_BO_ACQUISITION_CANDIDATES
    ),
    content_capacity=DEFAULT_CONTENT_CAPACITY,


    max_clients=None,


    run_oracle=True,


    preprocess_kwargs=None,
):


    dataset_name = str(
        dataset_name
    )

    epsilon = float(
        epsilon
    )

    seed = int(
        seed
    )

    if epsilon <= 0:

        raise ValueError(
            "epsilon must be positive."
        )

    if preprocess_kwargs is None:

        preprocess_kwargs = {}

    print(
        "\n"
        "############################################################"
    )

    print(
        "# JOURNAL LDP HETEROGENEOUS GRAPH PIPELINE"
    )

    print(
        "############################################################"
    )

    print(
        f"Dataset: {dataset_name}"
    )

    print(
        f"epsilon: {epsilon:g}"
    )

    print(
        f"seed: {seed}"
    )

    print(
        f"signature: "
        f"{signature_method}"
    )

    print(
        f"signature mechanism: "
        f"{signature_mechanism}"
    )

    print(
        f"degree mode: "
        f"{degree_mode}"
    )

    print(
        f"degree adjacency: "
        f"{degree_adjacency}"
    )

    print(
        "############################################################\n"
    )

    pipeline_start = time.perf_counter()

    stage_runtimes = {}

    validation = {}


    (
        preprocessing_result,
        runtime,
    ) = run_stage(
        "1. heterogeneous preprocessing",

        preprocess_data,

        dataset_name=dataset_name,

        data_dir=data_dir,

        output_folder=output_folder,

        force_rebuild=force_rebuild,

        **preprocess_kwargs,
    )

    stage_runtimes[
        "preprocessing"
    ] = runtime


    (
        global_graph,
        runtime,
    ) = run_stage(
        "2. global heterogeneous graph",

        create_global_graph,

        dataset_name=dataset_name,

        output_folder=output_folder,

        force_rebuild=force_rebuild,

        d_max=d_max,
    )

    stage_runtimes[
        "global_graph"
    ] = runtime

    validation[
        "global_graph"
    ] = validate_global_graph(
        global_graph
    )


    (
        ego_result,
        runtime,
    ) = run_stage(
        "3. true local ego graphs",

        create_ego_graphs,

        G=global_graph,

        dataset_name=dataset_name,

        output_folder=output_folder,

        force_rebuild=force_rebuild,

        max_clients=max_clients,

        save_graph_files=True,
    )

    stage_runtimes[
        "ego_graphs"
    ] = runtime


    (
        signature_result,
        runtime,
    ) = run_stage(
        "4. local spectral signatures",

        compute_local_signatures,

        dataset_name=dataset_name,

        output_folder=output_folder,

        force_rebuild=force_rebuild,
    )

    stage_runtimes[
        "local_signatures"
    ] = runtime


    (
        private_result,
        runtime,
    ) = run_stage(
        "5. private client reports",

        process_ego_graphs,

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

        degree_mode=degree_mode,

        degree_components=None,

        degree_adjacency=(
            degree_adjacency
        ),

        seed=seed,

        max_clients=max_clients,

        force_rebuild=force_rebuild,
    )

    stage_runtimes[
        "private_reports"
    ] = runtime

    validation[
        "private_reports"
    ] = validate_private_report_result(
        private_result,
        expected_epsilon=epsilon,
    )

    private_run_folder = private_result[
        "run_folder"
    ]


    (
        synthetic_graph,
        runtime,
    ) = run_stage(
        "6. private Bayesian reconstruction",

        compute_synthesized_global_graph,

        private_run_folder=(
            private_run_folder
        ),

        bo_evaluations=(
            bo_evaluations
        ),

        bo_initial_points=(
            bo_initial_points
        ),

        bo_acquisition_candidates=(
            bo_acquisition_candidates
        ),

        content_capacity=(
            content_capacity
        ),

        seed=seed,

        force_rebuild=force_rebuild,
    )

    stage_runtimes[
        "synthesis"
    ] = runtime

    validation[
        "synthetic_graph"
    ] = validate_synthetic_graph(
        synthetic_graph
    )


    oracle_result = None

    if run_oracle:

        (
            oracle_result,
            runtime,
        ) = run_stage(
            "7. post-hoc oracle evaluation",

            compute_metrics,

            G1=global_graph,

            G2=synthetic_graph,

            epsilon=epsilon,

            dataset_name=dataset_name,

            output_folder=output_folder,

            private_run_folder=(
                private_run_folder
            ),

            seed=seed,
        )

        stage_runtimes[
            "oracle_evaluation"
        ] = runtime


    configuration = {
        "dataset": dataset_name,

        "epsilon_total": float(
            epsilon
        ),

        "seed": int(
            seed
        ),

        "signature_method": (
            signature_method
        ),

        "signature_mechanism": (
            signature_mechanism
        ),

        "signature_fraction": float(
            signature_fraction
        ),

        "degree_fraction": float(
            1.0
            -
            signature_fraction
        ),

        "signature_clip_norm_C": float(
            signature_clip_norm
        ),

        "D_max": int(
            d_max
        ),

        "degree_mode": degree_mode,

        "degree_adjacency": (
            degree_adjacency
        ),

        "bo_evaluations": int(
            bo_evaluations
        ),

        "bo_initial_points": int(
            bo_initial_points
        ),

        "bo_acquisition_candidates": int(
            bo_acquisition_candidates
        ),

        "content_capacity": int(
            content_capacity
        ),

        "max_clients": (
            None
            if max_clients is None
            else int(
                max_clients
            )
        ),

        "oracle_evaluation_enabled": bool(
            run_oracle
        ),
    }

    manifest_path = save_pipeline_manifest(
        dataset_name=dataset_name,

        epsilon=epsilon,

        seed=seed,

        output_folder=output_folder,

        private_run_folder=(
            private_run_folder
        ),

        configuration=(
            configuration
        ),

        stage_runtimes=(
            stage_runtimes
        ),

        validation=validation,

        preprocessing_result=(
            preprocessing_result
        ),
    )

    total_runtime = float(
        time.perf_counter()
        -
        pipeline_start
    )


    print(
        "\n"
        "############################################################"
    )

    print(
        "# CORE PIPELINE COMPLETE"
    )

    print(
        "############################################################"
    )

    print(
        f"Dataset: {dataset_name}"
    )

    print(
        f"epsilon: {epsilon:g}"
    )

    print(
        f"seed: {seed}"
    )

    print(
        "\nOriginal graph:"
    )

    print(
        f"  nodes: "
        f"{global_graph.number_of_nodes():,}"
    )

    print(
        f"  edges: "
        f"{global_graph.number_of_edges():,}"
    )

    print(
        "\nPrivate reports:"
    )

    print(
        f"  clients: "
        f"{len(private_result['client_ids']):,}"
    )

    print(
        f"  signature dimension: "
        f"{private_result['private_signatures'].shape[1]}"
    )

    print(
        f"  degree dimension: "
        f"{private_result['private_degrees'].shape[1]}"
    )

    print(
        "\nSynthetic graph:"
    )

    print(
        f"  nodes: "
        f"{synthetic_graph.number_of_nodes():,}"
    )

    print(
        f"  edges: "
        f"{synthetic_graph.number_of_edges():,}"
    )

    if oracle_result is not None:

        oracle_objective = (
            oracle_result[
                "oracle_objective"
            ][
                "oracle_objective"
            ]
        )

        print(
            "\nOracle objective:"
        )

        print(
            f"  {oracle_objective:.6f}"
        )

    print(
        "\nPrivate run folder:"
    )

    print(
        f"  {private_run_folder}"
    )

    print(
        "\nManifest:"
    )

    print(
        f"  {manifest_path}"
    )

    print(
        f"\nTotal runtime: "
        f"{total_runtime:.3f} s"
    )

    print(
        "############################################################\n"
    )

    return {
        "dataset": dataset_name,

        "epsilon": float(
            epsilon
        ),

        "seed": int(
            seed
        ),

        "global_graph": (
            global_graph
        ),

        "ego_result": (
            ego_result
        ),

        "signature_result": (
            signature_result
        ),

        "private_result": (
            private_result
        ),

        "private_run_folder": (
            private_run_folder
        ),

        "synthetic_graph": (
            synthetic_graph
        ),

        "oracle_result": (
            oracle_result
        ),

        "manifest_path": (
            manifest_path
        ),

        "stage_runtimes": (
            stage_runtimes
        ),

        "total_runtime_seconds": (
            total_runtime
        ),
    }


def run_journal_experiments(
    dataset_name,
    private_run_folder,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    seed=DEFAULT_SEED,

    run_signature_invariance=False,
    run_privacy_utility=False,
    run_bo_validation=False,
    run_ablation=False,
    run_link_prediction=False,
    run_scalability=False,

    force_rebuild=False,
):


    results = {}


    if run_signature_invariance:

        from signature_experiment import (
            run_signature_experiment,
        )

        (
            results[
                "signature_invariance"
            ],
            _,
        ) = run_stage(
            "journal experiment: signature invariance",

            run_signature_experiment,

            dataset_name=dataset_name,

            output_folder=output_folder,

            seed=seed,

            force_rebuild=force_rebuild,
        )


    if run_privacy_utility:

        from privacy_utility_experiment import (
            run_privacy_utility_experiment,
        )

        (
            results[
                "privacy_utility"
            ],
            _,
        ) = run_stage(
            "journal experiment: privacy utility",

            run_privacy_utility_experiment,

            dataset_name=dataset_name,

            output_folder=output_folder,

            force_rebuild=force_rebuild,
        )


    if run_bo_validation:

        from private_bo_validation import (
            validate_private_bo,
        )

        (
            results[
                "bo_validation"
            ],
            _,
        ) = run_stage(
            "journal experiment: private BO validation",

            validate_private_bo,

            dataset_name=dataset_name,

            private_run_folder=(
                private_run_folder
            ),

            output_folder=output_folder,

            seed=seed,

            force_rebuild=force_rebuild,
        )


    if run_ablation:

        from ablation_experiment import (
            run_ablation_experiment,
        )

        (
            results[
                "ablation"
            ],
            _,
        ) = run_stage(
            "journal experiment: ablations",

            run_ablation_experiment,

            dataset_name=dataset_name,

            output_folder=output_folder,

            force_rebuild=force_rebuild,
        )


    if run_link_prediction:

        from link_prediction_experiment import (
            run_link_prediction_experiment,
        )

        (
            results[
                "link_prediction"
            ],
            _,
        ) = run_stage(
            "journal experiment: link prediction",

            run_link_prediction_experiment,

            dataset_name=dataset_name,

            output_folder=output_folder,

            force_rebuild=force_rebuild,
        )


    if run_scalability:

        from ann_scalability_experiment import (
            run_ann_scalability_experiment,
        )

        (
            results[
                "ann_scalability"
            ],
            _,
        ) = run_stage(
            "journal experiment: ANN scalability",

            run_ann_scalability_experiment,

            private_run_folder=(
                private_run_folder
            ),

            dataset_name=dataset_name,

            output_folder=output_folder,

            seed=seed,

            force_rebuild=force_rebuild,
        )

    return results


def build_argument_parser():


    parser = argparse.ArgumentParser(
        description=(
            "Run the revised locally private heterogeneous "
            "graph-synthesis journal pipeline."
        )
    )

    parser.add_argument(
        "-d",
        "--dataset",
        required=True,
        type=str,
        help=(
            "Dataset name, e.g. enron, yelp, dblp_filtered."
        ),
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_DATA_DIR,
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
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
        help=(
            "Debugging only. "
            "Do not use for final paper experiments."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rebuild cached stages."
        ),
    )

    parser.add_argument(
        "--no-oracle",
        action="store_true",
    )


    parser.add_argument(
        "--signature-experiment",
        action="store_true",
    )

    parser.add_argument(
        "--privacy-utility",
        action="store_true",
    )

    parser.add_argument(
        "--bo-validation",
        action="store_true",
    )

    parser.add_argument(
        "--ablation",
        action="store_true",
    )

    parser.add_argument(
        "--link-prediction",
        action="store_true",
    )

    parser.add_argument(
        "--scalability",
        action="store_true",
    )

    parser.add_argument(
        "--all-experiments",
        action="store_true",
        help=(
            "Run every heavy journal experiment after "
            "the core pipeline."
        ),
    )

    return parser


if __name__ == "__main__":

    parser = build_argument_parser()

    args = parser.parse_args()


    result = run_pipeline(
        dataset_name=args.dataset,

        epsilon=3.0,

        seed=args.seed,

        data_dir=args.data_dir,

        output_folder=args.output_folder,

        force_rebuild=args.force,


        signature_method="hks",

        signature_mechanism="l2_knorm",

        signature_fraction=0.5,

        signature_clip_norm=1.0,

        d_max=50,

        degree_mode="role",

        degree_adjacency="client_report",

        bo_evaluations=40,

        bo_initial_points=8,

        bo_acquisition_candidates=2000,

        content_capacity=4,

        max_clients=args.max_clients,

        run_oracle=(
            not args.no_oracle
        ),
    )


    run_all = bool(
        args.all_experiments
    )

    any_experiment = any(
        [
            run_all,
            args.signature_experiment,
            args.privacy_utility,
            args.bo_validation,
            args.ablation,
            args.link_prediction,
            args.scalability,
        ]
    )

    if any_experiment:

        run_journal_experiments(
            dataset_name=args.dataset,

            private_run_folder=(
                result[
                    "private_run_folder"
                ]
            ),

            output_folder=(
                args.output_folder
            ),

            seed=args.seed,

            run_signature_invariance=(
                run_all
                or args.signature_experiment
            ),

            run_privacy_utility=(
                run_all
                or args.privacy_utility
            ),

            run_bo_validation=(
                run_all
                or args.bo_validation
            ),

            run_ablation=(
                run_all
                or args.ablation
            ),

            run_link_prediction=(
                run_all
                or args.link_prediction
            ),

            run_scalability=(
                run_all
                or args.scalability
            ),

            force_rebuild=args.force,
        )
