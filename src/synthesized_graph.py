import hashlib
import heapq
import json
import math
import os
import time
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import norm

from sklearn.gaussian_process import (
    GaussianProcessRegressor,
)

from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)

from noisy_ego_graphs import (
    load_private_reports,
)


SYNTHESIS_SCHEMA_VERSION = "2.1"

DEFAULT_BO_EVALUATIONS = 40

DEFAULT_BO_INITIAL_POINTS = 8

DEFAULT_BO_ACQUISITION_CANDIDATES = 2000

DEFAULT_MAX_CANDIDATES_PER_CLIENT = 200

DEFAULT_MIN_CANDIDATES_PER_CLIENT = 10

DEFAULT_CONTENT_CAPACITY = 4

DEFAULT_RANDOM_SEED = 0

DEFAULT_BLOCK_MEMORY_MB = 256

NUMERICAL_EPS = 1e-12


PRIVATE_OBJECTIVE_WEIGHTS = {
    "degree": 0.60,
    "volume": 0.15,
    "client_similarity": 0.15,
    "content_cohesion": 0.10,
}


CLIENT_COMPONENT_HINTS = {
    "client",
    "user",
    "author",
    "person",
}


def numeric_summary(values):


    values = np.asarray(
        list(
            values
        ),
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


def stable_configuration_hash(configuration):


    payload = json.dumps(
        configuration,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]


def resolve_private_degree_cap(
    configuration,
    explicit_degree_cap=None,
):


    if explicit_degree_cap is not None:
        cap = int(explicit_degree_cap)
        if cap <= 0:
            raise ValueError("degree_cap must be positive.")
        return cap

    candidate_keys = {
        "dmax",
        "degreemax",
        "maxdegree",
        "degreecap",
        "maxdegreepercomponent",
    }

    def normalized_key(value):
        return "".join(
            character.lower()
            for character in str(value)
            if character.isalnum()
        )

    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if normalized_key(key) in candidate_keys:
                    try:
                        cap = int(item)
                    except (TypeError, ValueError):
                        cap = None
                    if cap is not None and cap > 0:
                        return cap

            for item in value.values():
                found = visit(item)
                if found is not None:
                    return found

        elif isinstance(value, (list, tuple)):
            for item in value:
                found = visit(item)
                if found is not None:
                    return found

        return None

    cap = visit(configuration)

    if cap is None:
        raise ValueError(
            "Could not resolve the public degree cap D_max from the "
            "private-run metadata. Pass degree_cap explicitly (CLI: "
            "--degree-cap). The synthesis stage fails closed rather "
            "than optimizing against unbounded noisy degree counts."
        )

    return int(cap)


def safe_component_name(
    value,
):


    value = str(
        value
    )

    return "".join(
        character
        if (
            character.isalnum()
            or character
            in {
                "_",
                "-",
            }
        )
        else "_"
        for character
        in value
    )


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

    normalized = np.divide(
        matrix,
        norms,
        out=np.zeros_like(
            matrix,
        ),
        where=(
            norms
            > NUMERICAL_EPS
        ),
    )

    return normalized


def safe_relative_error(
    observed,
    target,
):


    return float(
        abs(
            float(
                observed
            )
            -
            float(
                target
            )
        )
        /
        max(
            abs(
                float(
                    target
                )
            ),
            1.0,
        )
    )


def classify_degree_components(
    degree_mode,
    component_names,
):


    degree_mode = str(
        degree_mode
    ).lower()

    component_names = [
        str(
            value
        )
        for value
        in component_names
    ]

    if degree_mode == "role":

        client_indices = [
            index
            for index, component
            in enumerate(
                component_names
            )
            if component
            == "client"
        ]

        content_indices = [
            index
            for index, component
            in enumerate(
                component_names
            )
            if component
            == "content"
        ]

        if not client_indices:

            raise ValueError(
                "Role degree schema does not contain "
                "a 'client' component."
            )

        if not content_indices:

            raise ValueError(
                "Role degree schema does not contain "
                "a 'content' component."
            )

        return {
            "client_indices": (
                client_indices
            ),

            "content_indices": (
                content_indices
            ),
        }

    if degree_mode == "type":

        client_indices = []

        content_indices = []

        for index, component in enumerate(
            component_names
        ):

            if component.lower() in CLIENT_COMPONENT_HINTS:

                client_indices.append(
                    index
                )

            else:

                content_indices.append(
                    index
                )

        if not client_indices:

            raise ValueError(
                "Type-based reconstruction could not identify "
                "any client-type coordinate."
            )

        return {
            "client_indices": (
                client_indices
            ),

            "content_indices": (
                content_indices
            ),
        }

    if degree_mode == "relation":

        raise ValueError(
            "Relation-mode private reports require an explicit "
            "public relation endpoint schema before reconstruction. "
            "Use degree_mode='role' for the main journal method "
            "or degree_mode='type' for typed-node reconstruction."
        )

    raise ValueError(
        f"Unsupported degree mode: {degree_mode}"
    )


def compute_exact_top_candidates(
    private_signatures,
    max_candidates=DEFAULT_MAX_CANDIDATES_PER_CLIENT,
    memory_mb=DEFAULT_BLOCK_MEMORY_MB,
):


    X = normalize_rows(
        private_signatures
    )

    n = X.shape[
        0
    ]

    if n == 0:

        raise ValueError(
            "No private signatures supplied."
        )

    if n == 1:

        return (
            np.empty(
                (
                    1,
                    0,
                ),
                dtype=int,
            ),
            np.empty(
                (
                    1,
                    0,
                ),
                dtype=float,
            ),
        )

    max_candidates = min(
        int(
            max_candidates
        ),
        n - 1,
    )

    if max_candidates <= 0:

        raise ValueError(
            "max_candidates must be positive."
        )


    maximum_bytes = (
        int(
            memory_mb
        )
        * 1024
        * 1024
    )

    block_size = max(
        1,
        int(
            maximum_bytes
            /
            max(
                8 * n,
                1,
            )
        ),
    )

    block_size = min(
        block_size,
        512,
    )

    candidate_indices = np.empty(
        (
            n,
            max_candidates,
        ),
        dtype=np.int32,
    )

    candidate_scores = np.empty(
        (
            n,
            max_candidates,
        ),
        dtype=np.float32,
    )

    for start in range(
        0,
        n,
        block_size,
    ):

        end = min(
            start
            + block_size,
            n,
        )

        block = X[
            start:end
        ]

        similarities = (
            block
            @ X.T
        )


        local_rows = np.arange(
            end
            - start
        )

        global_rows = np.arange(
            start,
            end,
        )

        similarities[
            local_rows,
            global_rows,
        ] = -np.inf

        candidates = np.argpartition(
            -similarities,
            kth=max_candidates - 1,
            axis=1,
        )[
            :,
            :max_candidates
        ]

        candidate_values = np.take_along_axis(
            similarities,
            candidates,
            axis=1,
        )

        order = np.argsort(
            -candidate_values,
            axis=1,
        )

        candidates = np.take_along_axis(
            candidates,
            order,
            axis=1,
        )

        candidate_values = np.take_along_axis(
            candidate_values,
            order,
            axis=1,
        )

        candidate_indices[
            start:end
        ] = candidates.astype(
            np.int32
        )

        candidate_scores[
            start:end
        ] = candidate_values.astype(
            np.float32
        )

    return (
        candidate_indices,
        candidate_scores,
    )


def derive_threshold_bounds(
    candidate_scores,
):


    values = np.asarray(
        candidate_scores,
        dtype=float,
    ).ravel()

    values = values[
        np.isfinite(
            values
        )
    ]

    if len(
        values
    ) == 0:

        return (
            -0.25,
            0.95,
        )

    lower = float(
        np.percentile(
            values,
            5,
        )
    )

    upper = float(
        np.percentile(
            values,
            95,
        )
    )

    lower = max(
        -1.0,
        lower,
    )

    upper = min(
        1.0,
        upper,
    )

    if (
        upper
        - lower
        < 0.05
    ):

        center = 0.5 * (
            upper
            + lower
        )

        lower = max(
            -1.0,
            center
            - 0.1,
        )

        upper = min(
            1.0,
            center
            + 0.1,
        )

    return (
        lower,
        upper,
    )


def build_search_space(
    n_clients,
    candidate_scores,
):


    threshold_low, threshold_high = (
        derive_threshold_bounds(
            candidate_scores
        )
    )

    maximum_k = min(
        DEFAULT_MAX_CANDIDATES_PER_CLIENT,
        max(
            n_clients - 1,
            1,
        ),
    )

    minimum_k = min(
        DEFAULT_MIN_CANDIDATES_PER_CLIENT,
        maximum_k,
    )

    return {
        "threshold": (
            float(
                threshold_low
            ),
            float(
                threshold_high
            ),
        ),

        "client_degree_scale": (
            0.50,
            1.50,
        ),

        "content_degree_scale": (
            0.50,
            1.50,
        ),

        "candidate_k": (
            int(
                minimum_k
            ),
            int(
                maximum_k
            ),
        ),
    }


def decode_unit_parameters(
    unit_vector,
    search_space,
):


    unit_vector = np.clip(
        np.asarray(
            unit_vector,
            dtype=float,
        ),
        0.0,
        1.0,
    )

    threshold_low, threshold_high = (
        search_space[
            "threshold"
        ]
    )

    cc_low, cc_high = (
        search_space[
            "client_degree_scale"
        ]
    )

    cx_low, cx_high = (
        search_space[
            "content_degree_scale"
        ]
    )

    k_low, k_high = (
        search_space[
            "candidate_k"
        ]
    )

    threshold = (
        threshold_low
        +
        unit_vector[
            0
        ]
        * (
            threshold_high
            -
            threshold_low
        )
    )

    client_degree_scale = (
        cc_low
        +
        unit_vector[
            1
        ]
        * (
            cc_high
            -
            cc_low
        )
    )

    content_degree_scale = (
        cx_low
        +
        unit_vector[
            2
        ]
        * (
            cx_high
            -
            cx_low
        )
    )

    candidate_k = int(
        round(
            k_low
            +
            unit_vector[
                3
            ]
            * (
                k_high
                -
                k_low
            )
        )
    )

    candidate_k = int(
        np.clip(
            candidate_k,
            k_low,
            k_high,
        )
    )

    return {
        "similarity_threshold": float(
            threshold
        ),

        "client_degree_scale": float(
            client_degree_scale
        ),

        "content_degree_scale": float(
            content_degree_scale
        ),

        "candidate_k": int(
            candidate_k
        ),
    }


def parameter_key(
    parameters,
):


    return (
        round(
            float(
                parameters[
                    "similarity_threshold"
                ]
            ),
            8,
        ),

        round(
            float(
                parameters[
                    "client_degree_scale"
                ]
            ),
            8,
        ),

        round(
            float(
                parameters[
                    "content_degree_scale"
                ]
            ),
            8,
        ),

        int(
            parameters[
                "candidate_k"
            ]
        ),
    )


def extract_private_targets(
    private_degrees,
    component_names,
    degree_mode,
    degree_cap=None,
):


    raw_private_degrees = np.asarray(
        private_degrees,
        dtype=int,
    )

    if raw_private_degrees.ndim != 2:
        raise ValueError(
            "private_degrees must be a two-dimensional matrix."
        )

    negative_entries = int(
        np.sum(raw_private_degrees < 0)
    )

    if degree_cap is None:
        feasible_private_degrees = np.maximum(
            raw_private_degrees,
            0,
        )
        above_cap_entries = 0
    else:
        degree_cap = int(degree_cap)
        if degree_cap <= 0:
            raise ValueError("degree_cap must be positive.")

        above_cap_entries = int(
            np.sum(raw_private_degrees > degree_cap)
        )

        feasible_private_degrees = np.clip(
            raw_private_degrees,
            0,
            degree_cap,
        ).astype(int)

    private_degrees = feasible_private_degrees

    component_names = [
        str(
            value
        )
        for value
        in component_names
    ]

    classification = classify_degree_components(
        degree_mode=degree_mode,
        component_names=component_names,
    )

    client_indices = classification[
        "client_indices"
    ]

    content_indices = classification[
        "content_indices"
    ]


    private_client_degree = np.sum(
        private_degrees[
            :,
            client_indices
        ],
        axis=1,
    ).astype(
        int
    )

    if content_indices:

        private_content_degrees = (
            private_degrees[
                :,
                content_indices
            ].astype(
                int
            )
        )

        content_components = [
            component_names[
                index
            ]
            for index
            in content_indices
        ]

    else:

        private_content_degrees = (
            np.empty(
                (
                    len(
                        private_degrees
                    ),
                    0,
                ),
                dtype=int,
            )
        )

        content_components = []

    return {
        "private_client_degree": (
            private_client_degree
        ),

        "private_content_degrees": (
            private_content_degrees
        ),

        "content_components": (
            content_components
        ),

        "client_component_names": [
            component_names[
                index
            ]
            for index
            in client_indices
        ],

        "degree_postprocessing": {
            "projection": (
                "coordinatewise_clip_to_public_clean_degree_domain"
            ),
            "degree_cap": (
                None if degree_cap is None else int(degree_cap)
            ),
            "raw_negative_entries": int(negative_entries),
            "raw_above_cap_entries": int(above_cap_entries),
            "total_entries": int(raw_private_degrees.size),
            "changed_entries": int(
                np.sum(raw_private_degrees != feasible_private_degrees)
            ),
            "privacy_cost": 0.0,
            "reason": (
                "deterministic post-processing of already private "
                "degree reports into the public feasible query range"
            ),
        },
    }


def reconstruct_client_edges(
    n_clients,
    candidate_indices,
    candidate_scores,
    private_client_degree,
    parameters,
):


    threshold = float(
        parameters[
            "similarity_threshold"
        ]
    )

    degree_scale = float(
        parameters[
            "client_degree_scale"
        ]
    )

    candidate_k = int(
        parameters[
            "candidate_k"
        ]
    )

    candidate_k = min(
        candidate_k,
        candidate_indices.shape[
            1
        ],
    )

    targets = np.rint(
        np.asarray(
            private_client_degree,
            dtype=float,
        )
        * degree_scale
    ).astype(
        int
    )

    targets = np.clip(
        targets,
        0,
        max(
            n_clients
            - 1,
            0,
        ),
    )

    degrees = np.zeros(
        n_clients,
        dtype=int,
    )

    edges = []

    edge_scores = []

    existing_pairs = set()


    order = sorted(
        range(
            n_clients
        ),
        key=lambda index: (
            -targets[
                index
            ],
            index,
        ),
    )


    for i in order:

        if degrees[
            i
        ] >= targets[
            i
        ]:
            continue

        for rank in range(
            candidate_k
        ):

            j = int(
                candidate_indices[
                    i,
                    rank,
                ]
            )

            similarity = float(
                candidate_scores[
                    i,
                    rank,
                ]
            )

            if similarity < threshold:
                break

            if i == j:
                continue

            pair = (
                min(
                    i,
                    j,
                ),
                max(
                    i,
                    j,
                ),
            )

            if pair in existing_pairs:
                continue

            if degrees[
                i
            ] >= targets[
                i
            ]:
                break

            if degrees[
                j
            ] >= targets[
                j
            ]:
                continue

            existing_pairs.add(
                pair
            )

            edges.append(
                pair
            )

            edge_scores.append(
                similarity
            )

            degrees[
                i
            ] += 1

            degrees[
                j
            ] += 1


    maximum_overflow = 1

    for i in order:

        while degrees[
            i
        ] < targets[
            i
        ]:

            added = False

            for rank in range(
                candidate_k
            ):

                j = int(
                    candidate_indices[
                        i,
                        rank,
                    ]
                )

                similarity = float(
                    candidate_scores[
                        i,
                        rank,
                    ]
                )

                if similarity < threshold:
                    break

                if i == j:
                    continue

                pair = (
                    min(
                        i,
                        j,
                    ),
                    max(
                        i,
                        j,
                    ),
                )

                if pair in existing_pairs:
                    continue

                if (
                    degrees[
                        j
                    ]
                    >=
                    targets[
                        j
                    ]
                    +
                    maximum_overflow
                ):
                    continue

                existing_pairs.add(
                    pair
                )

                edges.append(
                    pair
                )

                edge_scores.append(
                    similarity
                )

                degrees[
                    i
                ] += 1

                degrees[
                    j
                ] += 1

                added = True

                break

            if not added:
                break

    return {
        "edges": edges,

        "edge_scores": np.asarray(
            edge_scores,
            dtype=float,
        ),

        "degrees": degrees,

        "scaled_targets": targets,
    }


def reconstruct_content_component(
    component_name,
    component_index,
    private_component_degree,
    private_signatures_normalized,
    candidate_indices,
    parameters,
    content_capacity=DEFAULT_CONTENT_CAPACITY,
):


    degree_scale = float(
        parameters[
            "content_degree_scale"
        ]
    )

    candidate_k = int(
        parameters[
            "candidate_k"
        ]
    )

    n_clients = len(
        private_component_degree
    )

    target = np.rint(
        np.asarray(
            private_component_degree,
            dtype=float,
        )
        * degree_scale
    ).astype(
        int
    )

    target = np.maximum(
        target,
        0,
    )

    remaining = target.copy()

    realized = np.zeros(
        n_clients,
        dtype=int,
    )

    content_nodes = []

    assignments = []

    content_counter = 0


    while np.any(
        remaining > 0
    ):

        maximum_remaining = np.max(
            remaining
        )

        anchors = np.where(
            remaining
            == maximum_remaining
        )[0]

        anchor = int(
            anchors[
                0
            ]
        )

        safe_component = safe_component_name(
            component_name
        )

        content_id = (
            f"synthetic_content__"
            f"{safe_component}__"
            f"{component_index}__"
            f"{content_counter}"
        )

        content_counter += 1

        assigned_clients = [
            anchor
        ]

        remaining[
            anchor
        ] -= 1

        realized[
            anchor
        ] += 1


        effective_k = min(
            candidate_k,
            candidate_indices.shape[
                1
            ],
        )

        for rank in range(
            effective_k
        ):

            if len(
                assigned_clients
            ) >= content_capacity:
                break

            candidate = int(
                candidate_indices[
                    anchor,
                    rank,
                ]
            )

            if candidate == anchor:
                continue

            if candidate in assigned_clients:
                continue

            if remaining[
                candidate
            ] <= 0:
                continue

            assigned_clients.append(
                candidate
            )

            remaining[
                candidate
            ] -= 1

            realized[
                candidate
            ] += 1

        content_nodes.append({
            "node_id": content_id,

            "component": str(
                component_name
            ),

            "clients": list(
                assigned_clients
            ),
        })

        for client_index in assigned_clients:

            assignments.append(
                (
                    int(
                        client_index
                    ),
                    content_id,
                )
            )


    cohesion_scores = []

    for content_node in content_nodes:

        clients = content_node[
            "clients"
        ]

        if len(
            clients
        ) < 2:
            continue

        for i in range(
            len(
                clients
            )
        ):

            for j in range(
                i + 1,
                len(
                    clients
                )
            ):

                u = clients[
                    i
                ]

                v = clients[
                    j
                ]

                similarity = float(
                    np.dot(
                        private_signatures_normalized[
                            u
                        ],
                        private_signatures_normalized[
                            v
                        ],
                    )
                )

                cohesion_scores.append(
                    similarity
                )

    return {
        "content_nodes": content_nodes,

        "assignments": assignments,

        "realized_degrees": realized,

        "scaled_targets": target,

        "cohesion_scores": np.asarray(
            cohesion_scores,
            dtype=float,
        ),
    }


def reconstruct_graph_from_private_reports(
    client_ids,
    private_signatures,
    private_targets,
    candidate_indices,
    candidate_scores,
    parameters,
    content_capacity=DEFAULT_CONTENT_CAPACITY,
):


    client_ids = np.asarray(
        client_ids,
        dtype=str,
    )

    private_signatures = np.asarray(
        private_signatures,
        dtype=float,
    )

    n_clients = len(
        client_ids
    )

    normalized_signatures = normalize_rows(
        private_signatures
    )


    client_result = (
        reconstruct_client_edges(
            n_clients=n_clients,

            candidate_indices=(
                candidate_indices
            ),

            candidate_scores=(
                candidate_scores
            ),

            private_client_degree=(
                private_targets[
                    "private_client_degree"
                ]
            ),

            parameters=parameters,
        )
    )


    private_content = private_targets[
        "private_content_degrees"
    ]

    content_components = private_targets[
        "content_components"
    ]

    content_results = []

    for component_index, component_name in enumerate(
        content_components
    ):

        result = (
            reconstruct_content_component(
                component_name=(
                    component_name
                ),

                component_index=(
                    component_index
                ),

                private_component_degree=(
                    private_content[
                        :,
                        component_index
                    ]
                ),

                private_signatures_normalized=(
                    normalized_signatures
                ),

                candidate_indices=(
                    candidate_indices
                ),

                parameters=parameters,

                content_capacity=(
                    content_capacity
                ),
            )
        )

        content_results.append(
            result
        )


    G_syn = nx.MultiGraph()

    G_syn.graph[
        "schema_version"
    ] = SYNTHESIS_SCHEMA_VERSION

    G_syn.graph[
        "source"
    ] = "private_reports_only"

    G_syn.graph[
        "contains_original_content_edges"
    ] = False

    G_syn.graph[
        "contains_original_content_nodes"
    ] = False

    G_syn.graph[
        "reconstruction_scope"
    ] = (
        "role_or_type_level_undirected_heterogeneous_topology"
    )

    G_syn.graph[
        "preserves_original_relation_labels"
    ] = False

    G_syn.graph[
        "preserves_original_edge_direction"
    ] = False


    for client_id in client_ids:

        G_syn.add_node(
            str(
                client_id
            ),

            role="client",

            type="client",

            synthetic=False,
        )


    for (
        edge_index,
        (
            u_index,
            v_index,
        ),
    ) in enumerate(
        client_result[
            "edges"
        ]
    ):

        similarity = float(
            client_result[
                "edge_scores"
            ][
                edge_index
            ]
        )

        u = str(
            client_ids[
                u_index
            ]
        )

        v = str(
            client_ids[
                v_index
            ]
        )

        G_syn.add_edge(
            u,
            v,

            relation="client_client",

            private_similarity=(
                similarity
            ),
        )


    for component_index, (
        component_name,
        component_result,
    ) in enumerate(
        zip(
            content_components,
            content_results,
        )
    ):

        safe_component = safe_component_name(
            component_name
        )

        for content_node in component_result[
            "content_nodes"
        ]:

            node_id = content_node[
                "node_id"
            ]

            if str(
                component_name
            ) == "content":

                node_type = (
                    "synthetic_content"
                )

            else:

                node_type = str(
                    component_name
                )

            G_syn.add_node(
                node_id,

                role="content",

                type=node_type,

                synthetic=True,

                component=str(
                    component_name
                ),
            )

        for (
            client_index,
            content_id,
        ) in component_result[
            "assignments"
        ]:

            client_id = str(
                client_ids[
                    client_index
                ]
            )

            relation_name = (
                "client_content"
                if str(
                    component_name
                )
                == "content"
                else (
                    f"client_{safe_component}"
                )
            )

            G_syn.add_edge(
                client_id,
                content_id,

                relation=(
                    relation_name
                ),
            )

    return {
        "graph": G_syn,

        "client_result": (
            client_result
        ),

        "content_results": (
            content_results
        ),

        "content_components": (
            content_components
        ),
    }


def compute_private_objective(
    reconstruction,
    private_targets,
    private_signatures,
    objective_weights=PRIVATE_OBJECTIVE_WEIGHTS,
):


    private_client_degree = np.asarray(
        private_targets[
            "private_client_degree"
        ],
        dtype=float,
    )

    private_content = np.asarray(
        private_targets[
            "private_content_degrees"
        ],
        dtype=float,
    )

    synthetic_client_degree = np.asarray(
        reconstruction[
            "client_result"
        ][
            "degrees"
        ],
        dtype=float,
    )


    client_degree_mae = float(
        np.mean(
            np.abs(
                synthetic_client_degree
                -
                private_client_degree
            )
        )
    )

    client_degree_denominator = max(
        float(
            np.mean(
                private_client_degree
            )
        ),
        1.0,
    )

    normalized_client_degree_loss = (
        client_degree_mae
        /
        client_degree_denominator
    )


    if private_content.shape[
        1
    ] > 0:

        content_losses = []

        for component_index, component_result in enumerate(
            reconstruction[
                "content_results"
            ]
        ):

            private_component = (
                private_content[
                    :,
                    component_index
                ]
            )

            synthetic_component = (
                component_result[
                    "realized_degrees"
                ].astype(
                    float
                )
            )

            mae = float(
                np.mean(
                    np.abs(
                        synthetic_component
                        -
                        private_component
                    )
                )
            )

            denominator = max(
                float(
                    np.mean(
                        private_component
                    )
                ),
                1.0,
            )

            content_losses.append(
                mae
                /
                denominator
            )

        normalized_content_degree_loss = float(
            np.mean(
                content_losses
            )
        )

    else:

        normalized_content_degree_loss = 0.0

    degree_loss = float(
        0.5
        * normalized_client_degree_loss
        +
        0.5
        * normalized_content_degree_loss
    )


    target_client_edges = float(
        np.sum(
            private_client_degree
        )
        /
        2.0
    )

    realized_client_edges = float(
        len(
            reconstruction[
                "client_result"
            ][
                "edges"
            ]
        )
    )

    client_volume_loss = (
        safe_relative_error(
            realized_client_edges,
            target_client_edges,
        )
    )

    content_volume_losses = []

    for component_index, component_result in enumerate(
        reconstruction[
            "content_results"
        ]
    ):

        target_volume = float(
            np.sum(
                private_content[
                    :,
                    component_index
                ]
            )
        )

        realized_volume = float(
            len(
                component_result[
                    "assignments"
                ]
            )
        )

        content_volume_losses.append(
            safe_relative_error(
                realized_volume,
                target_volume,
            )
        )

    if content_volume_losses:

        content_volume_loss = float(
            np.mean(
                content_volume_losses
            )
        )

    else:

        content_volume_loss = 0.0

    volume_loss = float(
        0.5
        * client_volume_loss
        +
        0.5
        * content_volume_loss
    )


    edge_scores = reconstruction[
        "client_result"
    ][
        "edge_scores"
    ]

    if len(
        edge_scores
    ):

        mean_client_similarity = float(
            np.mean(
                edge_scores
            )
        )


        client_similarity_quality = (
            mean_client_similarity
            + 1.0
        ) / 2.0

        client_similarity_quality = float(
            np.clip(
                client_similarity_quality,
                0.0,
                1.0,
            )
        )

    else:

        mean_client_similarity = 0.0

        client_similarity_quality = 0.0

    client_similarity_loss = float(
        1.0
        -
        client_similarity_quality
    )


    cohesion_values = []

    for component_result in reconstruction[
        "content_results"
    ]:

        values = component_result[
            "cohesion_scores"
        ]

        if len(
            values
        ):

            cohesion_values.extend(
                values.tolist()
            )

    if cohesion_values:

        mean_content_cohesion = float(
            np.mean(
                cohesion_values
            )
        )

        content_cohesion_quality = (
            mean_content_cohesion
            + 1.0
        ) / 2.0

        content_cohesion_quality = float(
            np.clip(
                content_cohesion_quality,
                0.0,
                1.0,
            )
        )

        content_cohesion_loss = float(
            1.0
            -
            content_cohesion_quality
        )

    else:

        mean_content_cohesion = 0.0

        total_content_assignments = int(
            sum(
                len(component_result["assignments"])
                for component_result
                in reconstruction["content_results"]
            )
        )


        content_cohesion_loss = float(
            1.0 if total_content_assignments > 0 else 0.0
        )


    total_loss = (
        objective_weights[
            "degree"
        ]
        * degree_loss

        +
        objective_weights[
            "volume"
        ]
        * volume_loss

        +
        objective_weights[
            "client_similarity"
        ]
        * client_similarity_loss

        +
        objective_weights[
            "content_cohesion"
        ]
        * content_cohesion_loss
    )

    private_objective = float(
        -total_loss
    )

    return {
        "private_objective": (
            private_objective
        ),

        "private_total_loss": float(
            total_loss
        ),

        "degree_loss": float(
            degree_loss
        ),

        "client_degree_loss": float(
            normalized_client_degree_loss
        ),

        "content_degree_loss": float(
            normalized_content_degree_loss
        ),

        "volume_loss": float(
            volume_loss
        ),

        "client_volume_loss": float(
            client_volume_loss
        ),

        "content_volume_loss": float(
            content_volume_loss
        ),

        "client_similarity_loss": float(
            client_similarity_loss
        ),

        "mean_private_client_edge_similarity": float(
            mean_client_similarity
        ),

        "content_cohesion_loss": float(
            content_cohesion_loss
        ),

        "mean_private_content_cohesion": float(
            mean_content_cohesion
        ),

        "realized_client_edges": int(
            realized_client_edges
        ),

        "target_private_client_edge_volume": float(
            target_client_edges
        ),
    }


def evaluate_private_configuration(
    parameters,
    client_ids,
    private_signatures,
    private_targets,
    candidate_indices,
    candidate_scores,
    content_capacity,
):


    start = time.perf_counter()

    reconstruction = (
        reconstruct_graph_from_private_reports(
            client_ids=(
                client_ids
            ),

            private_signatures=(
                private_signatures
            ),

            private_targets=(
                private_targets
            ),

            candidate_indices=(
                candidate_indices
            ),

            candidate_scores=(
                candidate_scores
            ),

            parameters=parameters,

            content_capacity=(
                content_capacity
            ),
        )
    )

    objective = compute_private_objective(
        reconstruction=(
            reconstruction
        ),

        private_targets=(
            private_targets
        ),

        private_signatures=(
            private_signatures
        ),
    )

    runtime = float(
        time.perf_counter()
        -
        start
    )

    objective[
        "evaluation_runtime_seconds"
    ] = runtime

    objective[
        "synthetic_nodes"
    ] = int(
        reconstruction[
            "graph"
        ].number_of_nodes()
    )

    objective[
        "synthetic_edges"
    ] = int(
        reconstruction[
            "graph"
        ].number_of_edges()
    )

    return (
        reconstruction,
        objective,
    )


def expected_improvement(
    X_candidates,
    gaussian_process,
    best_observed,
    xi=0.01,
):


    mean, std = gaussian_process.predict(
        X_candidates,
        return_std=True,
    )

    std = np.maximum(
        std,
        1e-12,
    )

    improvement = (
        mean
        -
        best_observed
        -
        xi
    )

    z = (
        improvement
        /
        std
    )

    ei = (
        improvement
        * norm.cdf(
            z
        )
        +
        std
        * norm.pdf(
            z
        )
    )

    return ei


def run_private_bayesian_optimization(
    client_ids,
    private_signatures,
    private_targets,
    candidate_indices,
    candidate_scores,
    seed=DEFAULT_RANDOM_SEED,
    evaluations=DEFAULT_BO_EVALUATIONS,
    initial_points=DEFAULT_BO_INITIAL_POINTS,
    acquisition_candidates=DEFAULT_BO_ACQUISITION_CANDIDATES,
    content_capacity=DEFAULT_CONTENT_CAPACITY,
):


    evaluations = int(
        evaluations
    )

    initial_points = int(
        initial_points
    )

    if evaluations <= 0:

        raise ValueError(
            "evaluations must be positive."
        )

    initial_points = min(
        initial_points,
        evaluations,
    )

    rng = np.random.default_rng(
        seed
    )

    search_space = build_search_space(
        n_clients=len(
            client_ids
        ),

        candidate_scores=(
            candidate_scores
        ),
    )

    X_observed = []

    y_observed = []

    trace_rows = []

    evaluated_keys = set()

    best_reconstruction = None

    best_parameters = None

    best_objective = -np.inf

    best_objective_components = None


    initial_unit_points = []


    initial_unit_points.append(
        np.array(
            [
                0.5,
                0.5,
                0.5,
                0.5,
            ],
            dtype=float,
        )
    )

    while len(
        initial_unit_points
    ) < initial_points:

        initial_unit_points.append(
            rng.random(
                4
            )
        )


    for iteration in range(
        evaluations
    ):

        if iteration < initial_points:

            unit_point = (
                initial_unit_points[
                    iteration
                ]
            )

            phase = "initial"

        else:

            X_array = np.vstack(
                X_observed
            )

            y_array = np.asarray(
                y_observed,
                dtype=float,
            )

            kernel = (
                ConstantKernel(
                    1.0,
                    (
                        1e-3,
                        1e3,
                    ),
                )
                *
                Matern(
                    length_scale=np.ones(
                        4
                    ),
                    length_scale_bounds=(
                        1e-2,
                        1e2,
                    ),
                    nu=2.5,
                )
                +
                WhiteKernel(
                    noise_level=1e-6,
                    noise_level_bounds=(
                        1e-10,
                        1e-2,
                    ),
                )
            )

            gaussian_process = (
                GaussianProcessRegressor(
                    kernel=kernel,

                    normalize_y=True,

                    random_state=(
                        int(
                            seed
                        )
                    ),

                    n_restarts_optimizer=2,
                )
            )

            gaussian_process.fit(
                X_array,
                y_array,
            )

            acquisition_points = rng.random(
                (
                    int(
                        acquisition_candidates
                    ),
                    4,
                )
            )

            acquisition_values = (
                expected_improvement(
                    X_candidates=(
                        acquisition_points
                    ),

                    gaussian_process=(
                        gaussian_process
                    ),

                    best_observed=float(
                        np.max(
                            y_array
                        )
                    ),
                )
            )

            order = np.argsort(
                -acquisition_values
            )

            unit_point = None

            for candidate_position in order:

                proposed = (
                    acquisition_points[
                        candidate_position
                    ]
                )

                decoded = decode_unit_parameters(
                    proposed,
                    search_space,
                )

                key = parameter_key(
                    decoded
                )

                if key not in evaluated_keys:

                    unit_point = proposed

                    break

            if unit_point is None:


                unit_point = rng.random(
                    4
                )

            phase = "bayesian_optimization"

        parameters = decode_unit_parameters(
            unit_point,
            search_space,
        )

        key = parameter_key(
            parameters
        )


        duplicate_attempts = 0

        while (
            key in evaluated_keys
            and duplicate_attempts < 100
        ):

            unit_point = rng.random(
                4
            )

            parameters = (
                decode_unit_parameters(
                    unit_point,
                    search_space,
                )
            )

            key = parameter_key(
                parameters
            )

            duplicate_attempts += 1

        evaluated_keys.add(
            key
        )

        (
            reconstruction,
            objective_components,
        ) = evaluate_private_configuration(
            parameters=parameters,

            client_ids=client_ids,

            private_signatures=(
                private_signatures
            ),

            private_targets=(
                private_targets
            ),

            candidate_indices=(
                candidate_indices
            ),

            candidate_scores=(
                candidate_scores
            ),

            content_capacity=(
                content_capacity
            ),
        )

        objective_value = float(
            objective_components[
                "private_objective"
            ]
        )

        X_observed.append(
            np.asarray(
                unit_point,
                dtype=float,
            )
        )

        y_observed.append(
            objective_value
        )

        if objective_value > best_objective:

            best_objective = (
                objective_value
            )

            best_parameters = dict(
                parameters
            )

            best_reconstruction = (
                reconstruction
            )

            best_objective_components = dict(
                objective_components
            )

        trace_row = {
            "iteration": int(
                iteration
            ),

            "phase": phase,

            "unit_threshold": float(
                unit_point[
                    0
                ]
            ),

            "unit_client_degree_scale": float(
                unit_point[
                    1
                ]
            ),

            "unit_content_degree_scale": float(
                unit_point[
                    2
                ]
            ),

            "unit_candidate_k": float(
                unit_point[
                    3
                ]
            ),

            **parameters,

            **objective_components,

            "best_private_objective_so_far": float(
                best_objective
            ),
        }

        trace_rows.append(
            trace_row
        )

        print(
            "[private BO] "
            f"{iteration + 1}/{evaluations} "
            f"J={objective_value:.6f} "
            f"best={best_objective:.6f} "
            f"tau={parameters['similarity_threshold']:.4f} "
            f"k={parameters['candidate_k']} "
            f"gamma_cc={parameters['client_degree_scale']:.3f} "
            f"gamma_cx={parameters['content_degree_scale']:.3f}"
        )

    trace_df = pd.DataFrame(
        trace_rows
    )

    return {
        "best_reconstruction": (
            best_reconstruction
        ),

        "best_parameters": (
            best_parameters
        ),

        "best_private_objective": float(
            best_objective
        ),

        "best_objective_components": (
            best_objective_components
        ),

        "trace": trace_df,

        "search_space": (
            search_space
        ),
    }


def build_private_degree_comparison(
    client_ids,
    private_targets,
    reconstruction,
):


    client_ids = np.asarray(
        client_ids,
        dtype=str,
    )

    private_client_degree = np.asarray(
        private_targets[
            "private_client_degree"
        ],
        dtype=int,
    )

    synthetic_client_degree = np.asarray(
        reconstruction[
            "client_result"
        ][
            "degrees"
        ],
        dtype=int,
    )

    private_content = np.asarray(
        private_targets[
            "private_content_degrees"
        ],
        dtype=int,
    )

    content_components = private_targets[
        "content_components"
    ]

    rows = []

    for index, client_id in enumerate(
        client_ids
    ):

        row = {
            "client_id": str(
                client_id
            ),

            "private_client_degree": int(
                private_client_degree[
                    index
                ]
            ),

            "synthetic_client_degree": int(
                synthetic_client_degree[
                    index
                ]
            ),

            "client_degree_signed_error": int(
                synthetic_client_degree[
                    index
                ]
                -
                private_client_degree[
                    index
                ]
            ),

            "client_degree_absolute_error": int(
                abs(
                    synthetic_client_degree[
                        index
                    ]
                    -
                    private_client_degree[
                        index
                    ]
                )
            ),
        }

        for component_index, component_name in enumerate(
            content_components
        ):

            private_value = int(
                private_content[
                    index,
                    component_index,
                ]
            )

            synthetic_value = int(
                reconstruction[
                    "content_results"
                ][
                    component_index
                ][
                    "realized_degrees"
                ][
                    index
                ]
            )

            safe_component = safe_component_name(
                component_name
            )

            row[
                f"private_content_degree__{safe_component}"
            ] = private_value

            row[
                f"synthetic_content_degree__{safe_component}"
            ] = synthetic_value

            row[
                f"content_degree_abs_error__{safe_component}"
            ] = abs(
                synthetic_value
                -
                private_value
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def compute_synthetic_graph_summary(
    graph,
    private_targets,
    reconstruction,
    best_parameters,
    best_objective_components,
):


    relation_counts = Counter()

    node_type_counts = Counter()

    role_counts = Counter()

    for _, data in graph.nodes(
        data=True
    ):

        node_type_counts[
            str(
                data.get(
                    "type",
                    "unknown",
                )
            )
        ] += 1

        role_counts[
            str(
                data.get(
                    "role",
                    "other",
                )
            )
        ] += 1

    for _, _, _, data in graph.edges(
        keys=True,
        data=True,
    ):

        relation_counts[
            str(
                data.get(
                    "relation",
                    "undefined",
                )
            )
        ] += 1

    client_degrees = reconstruction[
        "client_result"
    ][
        "degrees"
    ]

    content_component_statistics = {}

    for component_name, component_result in zip(
        private_targets[
            "content_components"
        ],
        reconstruction[
            "content_results"
        ],
    ):

        content_component_statistics[
            str(
                component_name
            )
        ] = {
            "synthetic_content_nodes": int(
                len(
                    component_result[
                        "content_nodes"
                    ]
                )
            ),

            "synthetic_client_content_edges": int(
                len(
                    component_result[
                        "assignments"
                    ]
                )
            ),

            "realized_client_degree": numeric_summary(
                component_result[
                    "realized_degrees"
                ]
            ),

            "scaled_target_degree": numeric_summary(
                component_result[
                    "scaled_targets"
                ]
            ),

            "private_cohesion": numeric_summary(
                component_result[
                    "cohesion_scores"
                ]
            ),
        }

    return {
        "schema_version": (
            SYNTHESIS_SCHEMA_VERSION
        ),

        "source": (
            "private_reports_only"
        ),

        "total_nodes": int(
            graph.number_of_nodes()
        ),

        "total_edges": int(
            graph.number_of_edges()
        ),

        "node_type_counts": dict(
            node_type_counts
        ),

        "role_counts": dict(
            role_counts
        ),

        "relation_counts": dict(
            relation_counts
        ),

        "synthetic_client_degree": numeric_summary(
            client_degrees
        ),

        "content_components": (
            content_component_statistics
        ),

        "best_parameters": (
            best_parameters
        ),

        "best_private_objective": (
            best_objective_components
        ),

        "contains_original_content_nodes": False,

        "contains_original_content_edges": False,

        "uses_original_graph_for_optimization": False,

        "reconstruction_scope": (
            "role_or_type_level_undirected_heterogeneous_topology"
        ),

        "preserves_original_relation_labels": False,

        "preserves_original_edge_direction": False,
    }


def compute_synthesized_global_graph(
    private_run_folder,
    bo_evaluations=DEFAULT_BO_EVALUATIONS,
    bo_initial_points=DEFAULT_BO_INITIAL_POINTS,
    bo_acquisition_candidates=DEFAULT_BO_ACQUISITION_CANDIDATES,
    max_candidates_per_client=DEFAULT_MAX_CANDIDATES_PER_CLIENT,
    content_capacity=DEFAULT_CONTENT_CAPACITY,
    seed=None,
    degree_cap=None,
    force_rebuild=False,
):


    synthesis_folder = os.path.join(
        private_run_folder,
        "synthesis",
    )

    os.makedirs(
        synthesis_folder,
        exist_ok=True,
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

    graph_stats_path = os.path.join(
        synthesis_folder,
        "synthetic_private_stats.json",
    )

    degree_comparison_path = os.path.join(
        synthesis_folder,
        "synthetic_vs_private_degree.csv.gz",
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
        str(
            value
        )
        for value
        in reports[
            "degree_component_names"
        ]
    ]

    metadata = reports[
        "metadata"
    ]

    configuration = metadata[
        "configuration"
    ]

    degree_mode = str(
        configuration[
            "degree_mode"
        ]
    )

    if seed is None:

        seed = int(
            configuration.get(
                "seed",
                DEFAULT_RANDOM_SEED,
            )
        )

    seed = int(
        seed
    )

    resolved_degree_cap = resolve_private_degree_cap(
        configuration=configuration,
        explicit_degree_cap=degree_cap,
    )

    synthesis_configuration = {
        "schema_version": SYNTHESIS_SCHEMA_VERSION,
        "bo_evaluations": int(bo_evaluations),
        "bo_initial_points": int(bo_initial_points),
        "bo_acquisition_candidates": int(
            bo_acquisition_candidates
        ),
        "max_candidates_per_client": int(
            max_candidates_per_client
        ),
        "content_capacity": int(content_capacity),
        "seed": int(seed),
        "degree_cap": int(resolved_degree_cap),
        "objective_weights": dict(PRIVATE_OBJECTIVE_WEIGHTS),
        "candidate_search": "exact_blockwise_cosine",
    }

    synthesis_configuration_hash = stable_configuration_hash(
        synthesis_configuration
    )


    if (
        not force_rebuild
        and os.path.exists(graph_path)
        and os.path.exists(trace_path)
        and os.path.exists(summary_path)
    ):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                cached_summary = json.load(f)

            cached_hash = cached_summary.get(
                "synthesis_configuration_hash"
            )
        except (OSError, json.JSONDecodeError):
            cached_hash = None

        if cached_hash == synthesis_configuration_hash:
            print(
                "[synthesis] Using configuration-matched cached "
                "synthetic graph:"
            )
            print(f"  {graph_path}")
            return nx.read_gml(graph_path)

        print(
            "[synthesis] Existing cache does not match the requested "
            "configuration; rebuilding."
        )

    if (
        len(
            client_ids
        )
        != private_signatures.shape[
            0
        ]
        or len(
            client_ids
        )
        != private_degrees.shape[
            0
        ]
    ):

        raise ValueError(
            "Private report arrays have inconsistent lengths."
        )

    if len(
        client_ids
    ) < 2:

        raise ValueError(
            "At least two private client reports "
            "are required for reconstruction."
        )


    private_targets = extract_private_targets(
        private_degrees=(
            private_degrees
        ),

        component_names=(
            component_names
        ),

        degree_mode=(
            degree_mode
        ),

        degree_cap=(
            resolved_degree_cap
        ),
    )


    candidate_start = time.perf_counter()

    (
        candidate_indices,
        candidate_scores,
    ) = compute_exact_top_candidates(
        private_signatures=(
            private_signatures
        ),

        max_candidates=(
            max_candidates_per_client
        ),
    )

    candidate_runtime = float(
        time.perf_counter()
        -
        candidate_start
    )


    bo_start = time.perf_counter()

    bo_result = (
        run_private_bayesian_optimization(
            client_ids=(
                client_ids
            ),

            private_signatures=(
                private_signatures
            ),

            private_targets=(
                private_targets
            ),

            candidate_indices=(
                candidate_indices
            ),

            candidate_scores=(
                candidate_scores
            ),

            seed=seed,

            evaluations=(
                bo_evaluations
            ),

            initial_points=(
                bo_initial_points
            ),

            acquisition_candidates=(
                bo_acquisition_candidates
            ),

            content_capacity=(
                content_capacity
            ),
        )
    )

    bo_runtime = float(
        time.perf_counter()
        -
        bo_start
    )

    best_reconstruction = bo_result[
        "best_reconstruction"
    ]

    best_graph = best_reconstruction[
        "graph"
    ]

    best_parameters = bo_result[
        "best_parameters"
    ]

    best_components = bo_result[
        "best_objective_components"
    ]

    trace_df = bo_result[
        "trace"
    ]


    best_graph.graph[
        "private_run_folder"
    ] = str(
        private_run_folder
    )

    best_graph.graph[
        "bo_seed"
    ] = int(
        seed
    )

    best_graph.graph[
        "private_bo_evaluations"
    ] = int(
        bo_evaluations
    )

    best_graph.graph[
        "private_objective"
    ] = float(
        bo_result[
            "best_private_objective"
        ]
    )

    best_graph.graph[
        "similarity_threshold"
    ] = float(
        best_parameters[
            "similarity_threshold"
        ]
    )

    best_graph.graph[
        "client_degree_scale"
    ] = float(
        best_parameters[
            "client_degree_scale"
        ]
    )

    best_graph.graph[
        "content_degree_scale"
    ] = float(
        best_parameters[
            "content_degree_scale"
        ]
    )

    best_graph.graph[
        "candidate_k"
    ] = int(
        best_parameters[
            "candidate_k"
        ]
    )

    best_graph.graph[
        "content_capacity"
    ] = int(
        content_capacity
    )


    nx.write_gml(
        best_graph,
        graph_path,
    )


    trace_df.to_csv(
        trace_path,
        index=False,
    )


    degree_comparison_df = (
        build_private_degree_comparison(
            client_ids=(
                client_ids
            ),

            private_targets=(
                private_targets
            ),

            reconstruction=(
                best_reconstruction
            ),
        )
    )

    degree_comparison_df.to_csv(
        degree_comparison_path,
        index=False,
        compression="gzip",
    )


    graph_stats = (
        compute_synthetic_graph_summary(
            graph=best_graph,

            private_targets=(
                private_targets
            ),

            reconstruction=(
                best_reconstruction
            ),

            best_parameters=(
                best_parameters
            ),

            best_objective_components=(
                best_components
            ),
        )
    )

    with open(
        graph_stats_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            graph_stats,
            f,
            indent=4,
            ensure_ascii=False,
        )


    bo_summary = {
        "schema_version": (
            SYNTHESIS_SCHEMA_VERSION
        ),

        "source": (
            "private_reports_only"
        ),

        "privacy_run_configuration": (
            configuration
        ),

        "synthesis_configuration": (
            synthesis_configuration
        ),

        "synthesis_configuration_hash": (
            synthesis_configuration_hash
        ),

        "private_degree_postprocessing": (
            private_targets["degree_postprocessing"]
        ),

        "bo_seed": int(
            seed
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

        "candidate_search": (
            "exact_blockwise_cosine"
        ),

        "max_candidates_per_client": int(
            candidate_indices.shape[
                1
            ]
        ),

        "candidate_search_runtime_seconds": float(
            candidate_runtime
        ),

        "bo_runtime_seconds": float(
            bo_runtime
        ),

        "content_capacity": int(
            content_capacity
        ),

        "search_space": (
            bo_result[
                "search_space"
            ]
        ),

        "private_objective_weights": (
            PRIVATE_OBJECTIVE_WEIGHTS
        ),

        "best_parameters": (
            best_parameters
        ),

        "best_private_objective": float(
            bo_result[
                "best_private_objective"
            ]
        ),

        "best_objective_components": (
            best_components
        ),

        "client_degree_components": (
            private_targets[
                "client_component_names"
            ]
        ),

        "content_degree_components": (
            private_targets[
                "content_components"
            ]
        ),

        "server_inputs": [
            "private client IDs",
            "private signatures",
            "private typed-degree vectors",
            "public reconstruction hyperparameters",
        ],

        "server_does_not_use": [
            "original graph",
            "original edges",
            "original content-node identities",
            "clean signatures",
            "clean degree vectors",
            "oracle graph metrics",
        ],

        "reconstruction_scope": (
            "role_or_type_level_undirected_heterogeneous_topology"
        ),

        "preserves_original_relation_labels": False,

        "preserves_original_edge_direction": False,

        "outputs": {
            "synthetic_graph": os.path.basename(
                graph_path
            ),

            "private_bo_trace": os.path.basename(
                trace_path
            ),

            "synthetic_private_stats": os.path.basename(
                graph_stats_path
            ),

            "degree_comparison": os.path.basename(
                degree_comparison_path
            ),
        },
    }

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            bo_summary,
            f,
            indent=4,
            ensure_ascii=False,
        )


    print(
        "\n"
        "========================================"
    )

    print(
        "PRIVATE GRAPH SYNTHESIS COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"Clients: "
        f"{len(client_ids):,}"
    )

    print(
        f"Synthetic nodes: "
        f"{best_graph.number_of_nodes():,}"
    )

    print(
        f"Synthetic edges: "
        f"{best_graph.number_of_edges():,}"
    )

    print(
        f"BO evaluations: "
        f"{bo_evaluations}"
    )

    print(
        "\nBest private objective:"
    )

    print(
        f"  "
        f"{bo_result['best_private_objective']:.6f}"
    )

    print(
        "\nBest parameters:"
    )

    print(
        "  similarity threshold: "
        f"{best_parameters['similarity_threshold']:.6f}"
    )

    print(
        "  client degree scale: "
        f"{best_parameters['client_degree_scale']:.6f}"
    )

    print(
        "  content degree scale: "
        f"{best_parameters['content_degree_scale']:.6f}"
    )

    print(
        "  candidate k: "
        f"{best_parameters['candidate_k']}"
    )

    print(
        "\nPrivate loss components:"
    )

    print(
        "  degree loss: "
        f"{best_components['degree_loss']:.6f}"
    )

    print(
        "  volume loss: "
        f"{best_components['volume_loss']:.6f}"
    )

    print(
        "  client similarity loss: "
        f"{best_components['client_similarity_loss']:.6f}"
    )

    print(
        "  content cohesion loss: "
        f"{best_components['content_cohesion_loss']:.6f}"
    )

    print(
        "\nOutputs:"
    )

    print(
        f"  Graph: "
        f"{graph_path}"
    )

    print(
        f"  BO trace: "
        f"{trace_path}"
    )

    print(
        f"  Private statistics: "
        f"{graph_stats_path}"
    )

    print(
        f"  Degree comparison: "
        f"{degree_comparison_path}"
    )

    print(
        f"  BO summary: "
        f"{summary_path}"
    )

    print(
        "========================================\n"
    )

    return best_graph


def load_synthesized_graph(
    private_run_folder,
):


    graph_path = os.path.join(
        private_run_folder,
        "synthesis",
        "synthetic_graph.gml.gz",
    )

    if not os.path.exists(
        graph_path
    ):

        raise FileNotFoundError(
            f"Missing synthetic graph: "
            f"{graph_path}"
        )

    return nx.read_gml(
        graph_path
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct a heterogeneous graph "
            "using only LDP private reports."
        )
    )

    parser.add_argument(
        "private_run_folder",
        type=str,
        help=(
            "Run directory created by "
            "noisy_ego_graphs.py"
        ),
    )

    parser.add_argument(
        "--evaluations",
        type=int,
        default=(
            DEFAULT_BO_EVALUATIONS
        ),
    )

    parser.add_argument(
        "--initial-points",
        type=int,
        default=(
            DEFAULT_BO_INITIAL_POINTS
        ),
    )

    parser.add_argument(
        "--acquisition-candidates",
        type=int,
        default=(
            DEFAULT_BO_ACQUISITION_CANDIDATES
        ),
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=(
            DEFAULT_MAX_CANDIDATES_PER_CLIENT
        ),
    )

    parser.add_argument(
        "--content-capacity",
        type=int,
        default=(
            DEFAULT_CONTENT_CAPACITY
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--degree-cap",
        type=int,
        default=None,
        help=(
            "Public clean per-coordinate degree cap D_max. "
            "If omitted, resolve it from private-run metadata."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
    )

    args = parser.parse_args()

    compute_synthesized_global_graph(
        private_run_folder=(
            args.private_run_folder
        ),

        bo_evaluations=(
            args.evaluations
        ),

        bo_initial_points=(
            args.initial_points
        ),

        bo_acquisition_candidates=(
            args.acquisition_candidates
        ),

        max_candidates_per_client=(
            args.max_candidates
        ),

        content_capacity=(
            args.content_capacity
        ),

        seed=args.seed,

        degree_cap=args.degree_cap,

        force_rebuild=(
            args.force
        ),
    )
