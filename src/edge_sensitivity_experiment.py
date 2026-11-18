import argparse
import hashlib
import json
import math
import os
import time

from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd


EDGE_SENSITIVITY_SCHEMA_VERSION = "1.0"

DEFAULT_OUTPUT_FOLDER = "output_local_test"

DEFAULT_SEED = 0

DEFAULT_MAX_CLIENTS = 200

DEFAULT_PERTURBATIONS_PER_CATEGORY = 5

DEFAULT_SIGNATURE_DIMENSIONS = (
    2,
    4,
    6,
    8,
    12,
)

DEFAULT_CLIP_NORM_C = 1.0


CANONICAL_HKS_TIMES = np.asarray(
    [
        0.1,
        0.2,
        0.5,
        1.0,
        2.0,
        5.0,
    ],
    dtype=float,
)


def deterministic_seed(
    *parts,
):
    text = "|".join(
        str(part)
        for part
        in parts
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


def safe_role(
    graph,
    node,
):
    return str(
        graph.nodes[
            node
        ].get(
            "role",
            "other",
        )
    ).lower()


def safe_type(
    graph,
    node,
):
    return str(
        graph.nodes[
            node
        ].get(
            "type",
            "unknown",
        )
    ).lower()


def numeric_summary(
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

    if len(
        values
    ) == 0:

        return {
            "count": 0,
            "mean": np.nan,
            "std": np.nan,
            "median": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
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

        "max": float(
            np.max(
                values
            )
        ),
    }


def hks_times_for_dimension(
    dimension,
):


    dimension = int(
        dimension
    )

    if (
        dimension <= 0
        or dimension % 2 != 0
        or dimension > 12
    ):

        raise ValueError(
            "HKS dimension must be an even integer "
            "between 2 and 12."
        )

    per_block = (
        dimension
        //
        2
    )

    if per_block == 1:

        return np.asarray(
            [
                1.0,
            ],
            dtype=float,
        )

    if per_block == 2:

        return np.asarray(
            [
                0.2,
                2.0,
            ],
            dtype=float,
        )

    if per_block == 3:

        return np.asarray(
            [
                0.1,
                1.0,
                5.0,
            ],
            dtype=float,
        )

    if per_block == 4:

        return np.asarray(
            [
                0.1,
                0.5,
                2.0,
                5.0,
            ],
            dtype=float,
        )

    if per_block == 5:

        return np.asarray(
            [
                0.1,
                0.2,
                0.5,
                2.0,
                5.0,
            ],
            dtype=float,
        )

    return CANONICAL_HKS_TIMES.copy()


def get_root_neighbours(
    graph,
    root,
):


    if graph.is_directed():

        return (
            set(
                graph.predecessors(
                    root
                )
            )
            |
            set(
                graph.successors(
                    root
                )
            )
        )

    return set(
        graph.neighbors(
            root
        )
    )


def create_true_ego_graph(
    graph,
    root,
):


    neighbours = get_root_neighbours(
        graph,
        root,
    )

    nodes = {
        root,
        *neighbours,
    }

    return graph.subgraph(
        nodes
    ).copy()


def edge_matches_block(
    graph,
    source,
    target,
    block,
):
    source_role = safe_role(
        graph,
        source,
    )

    target_role = safe_role(
        graph,
        target,
    )

    if block == "CC":

        return (
            source_role == "client"
            and target_role == "client"
        )

    if block == "CX":

        return {
            source_role,
            target_role,
        } == {
            "client",
            "content",
        }

    raise ValueError(
        f"Unknown HKS block: {block}"
    )


def build_weighted_projection(
    ego_graph,
    root,
    block,
):


    projected = nx.Graph()

    projected.add_node(
        root
    )

    if ego_graph.is_multigraph():

        edge_iterator = ego_graph.edges(
            keys=True,
            data=True,
        )

        for (
            source,
            target,
            _,
            data,
        ) in edge_iterator:

            if source == target:
                continue

            if not edge_matches_block(
                ego_graph,
                source,
                target,
                block,
            ):

                continue

            projected.add_node(
                source
            )

            projected.add_node(
                target
            )

            if projected.has_edge(
                source,
                target,
            ):

                projected[
                    source
                ][
                    target
                ][
                    "weight"
                ] += 1.0

            else:

                projected.add_edge(
                    source,
                    target,
                    weight=1.0,
                )

    else:

        for (
            source,
            target,
            data,
        ) in ego_graph.edges(
            data=True
        ):

            if source == target:
                continue

            if not edge_matches_block(
                ego_graph,
                source,
                target,
                block,
            ):

                continue

            weight = float(
                data.get(
                    "weight",
                    1.0,
                )
            )

            projected.add_edge(
                source,
                target,
                weight=weight,
            )

    return projected


def normalized_laplacian_dense(
    graph,
):


    nodes = list(
        graph.nodes()
    )

    n = len(
        nodes
    )

    index = {
        node: position
        for position, node
        in enumerate(
            nodes
        )
    }

    adjacency = np.zeros(
        (
            n,
            n,
        ),
        dtype=float,
    )

    for (
        source,
        target,
        data,
    ) in graph.edges(
        data=True
    ):

        if source == target:
            continue

        i = index[
            source
        ]

        j = index[
            target
        ]

        weight = float(
            data.get(
                "weight",
                1.0,
            )
        )

        adjacency[
            i,
            j
        ] += weight

        adjacency[
            j,
            i
        ] += weight

    degree = np.sum(
        adjacency,
        axis=1,
    )

    combinatorial = (
        np.diag(
            degree
        )
        -
        adjacency
    )

    inverse_sqrt_degree = np.zeros(
        n,
        dtype=float,
    )

    positive = (
        degree
        >
        0
    )

    inverse_sqrt_degree[
        positive
    ] = (
        1.0
        /
        np.sqrt(
            degree[
                positive
            ]
        )
    )

    normalized = (
        inverse_sqrt_degree[
            :,
            None
        ]
        *
        combinatorial
        *
        inverse_sqrt_degree[
            None,
            :
        ]
    )

    normalized = (
        normalized
        +
        normalized.T
    ) / 2.0

    return (
        normalized,
        nodes,
        index,
    )


def hks_root_values(
    graph,
    root,
    times,
):


    if root not in graph:

        raise KeyError(
            f"Root {root!r} missing from projection."
        )

    (
        laplacian,
        nodes,
        index,
    ) = normalized_laplacian_dense(
        graph
    )

    eigenvalues, eigenvectors = np.linalg.eigh(
        laplacian
    )

    eigenvalues = np.maximum(
        eigenvalues,
        0.0,
    )

    root_position = index[
        root
    ]

    squared_root_coordinates = (
        eigenvectors[
            root_position,
            :
        ]
        **
        2
    )

    values = []

    for time_value in times:

        heat = np.exp(
            -float(
                time_value
            )
            *
            eigenvalues
        )

        value = float(
            np.sum(
                heat
                *
                squared_root_coordinates
            )
        )

        values.append(
            value
        )

    return np.asarray(
        values,
        dtype=float,
    )


def normalize_vector(
    vector,
):
    vector = np.asarray(
        vector,
        dtype=float,
    )

    norm = np.linalg.norm(
        vector
    )

    if norm <= 1e-12:

        return np.zeros_like(
            vector
        )

    return (
        vector
        /
        norm
    )


def compute_hks_signature(
    ego_graph,
    root,
    dimension,
):


    times = hks_times_for_dimension(
        dimension
    )

    cc_graph = build_weighted_projection(
        ego_graph,
        root,
        "CC",
    )

    cx_graph = build_weighted_projection(
        ego_graph,
        root,
        "CX",
    )


    if (
        root not in cc_graph
        or cc_graph.degree(
            root,
            weight="weight",
        ) <= 0
    ):
        cc = np.zeros(
            len(times),
            dtype=float,
        )
    else:
        cc = hks_root_values(
            cc_graph,
            root,
            times,
        )

    if (
        root not in cx_graph
        or cx_graph.degree(
            root,
            weight="weight",
        ) <= 0
    ):
        cx = np.zeros(
            len(times),
            dtype=float,
        )
    else:
        cx = hks_root_values(
            cx_graph,
            root,
            times,
        )

    cc = normalize_vector(
        cc
    )

    cx = normalize_vector(
        cx
    )

    combined = np.concatenate(
        [
            cc,
            cx,
        ]
    )

    combined = normalize_vector(
        combined
    )

    return combined


def typed_root_degree(
    ego_graph,
    root,
):


    neighbours = get_root_neighbours(
        ego_graph,
        root,
    )

    client_count = 0
    content_count = 0

    for neighbour in neighbours:

        role = safe_role(
            ego_graph,
            neighbour,
        )

        if role == "client":

            client_count += 1

        elif role == "content":

            content_count += 1

    return np.asarray(
        [
            client_count,
            content_count,
        ],
        dtype=int,
    )


def incident_edge_records(
    graph,
    root,
):


    records = []

    if graph.is_directed():

        if graph.is_multigraph():

            for (
                source,
                target,
                key,
                data,
            ) in graph.out_edges(
                root,
                keys=True,
                data=True,
            ):

                if target == root:
                    continue

                records.append({
                    "source": source,
                    "target": target,
                    "key": key,
                    "direction": "out",
                    "endpoint": target,
                    "relation": str(
                        data.get(
                            "relation",
                            "unknown",
                        )
                    ),
                })

            for (
                source,
                target,
                key,
                data,
            ) in graph.in_edges(
                root,
                keys=True,
                data=True,
            ):

                if source == root:
                    continue

                records.append({
                    "source": source,
                    "target": target,
                    "key": key,
                    "direction": "in",
                    "endpoint": source,
                    "relation": str(
                        data.get(
                            "relation",
                            "unknown",
                        )
                    ),
                })

        else:

            for (
                source,
                target,
                data,
            ) in graph.out_edges(
                root,
                data=True,
            ):

                if target == root:
                    continue

                records.append({
                    "source": source,
                    "target": target,
                    "key": None,
                    "direction": "out",
                    "endpoint": target,
                    "relation": str(
                        data.get(
                            "relation",
                            "unknown",
                        )
                    ),
                })

            for (
                source,
                target,
                data,
            ) in graph.in_edges(
                root,
                data=True,
            ):

                if source == root:
                    continue

                records.append({
                    "source": source,
                    "target": target,
                    "key": None,
                    "direction": "in",
                    "endpoint": source,
                    "relation": str(
                        data.get(
                            "relation",
                            "unknown",
                        )
                    ),
                })

    else:

        if graph.is_multigraph():

            for (
                source,
                target,
                key,
                data,
            ) in graph.edges(
                root,
                keys=True,
                data=True,
            ):

                endpoint = (
                    target
                    if source == root
                    else source
                )

                records.append({
                    "source": source,
                    "target": target,
                    "key": key,
                    "direction": "undirected",
                    "endpoint": endpoint,
                    "relation": str(
                        data.get(
                            "relation",
                            "unknown",
                        )
                    ),
                })

        else:

            for (
                source,
                target,
                data,
            ) in graph.edges(
                root,
                data=True,
            ):

                endpoint = (
                    target
                    if source == root
                    else source
                )

                records.append({
                    "source": source,
                    "target": target,
                    "key": None,
                    "direction": "undirected",
                    "endpoint": endpoint,
                    "relation": str(
                        data.get(
                            "relation",
                            "unknown",
                        )
                    ),
                })

    for record in records:

        endpoint = record[
            "endpoint"
        ]

        endpoint_role = safe_role(
            graph,
            endpoint,
        )

        endpoint_type = safe_type(
            graph,
            endpoint,
        )

        record[
            "endpoint_role"
        ] = endpoint_role

        record[
            "endpoint_type"
        ] = endpoint_type

        if endpoint_role == "client":

            record[
                "edge_class"
            ] = "client_client"

        elif endpoint_role == "content":

            record[
                "edge_class"
            ] = "client_content"

        else:

            record[
                "edge_class"
            ] = "other"

    return records


def root_endpoint_still_adjacent(
    graph,
    root,
    endpoint,
):


    if graph.is_directed():

        return (
            graph.has_edge(
                root,
                endpoint,
            )
            or graph.has_edge(
                endpoint,
                root,
            )
        )

    return graph.has_edge(
        root,
        endpoint,
    )


def ego_after_edge_deletion(
    baseline_ego,
    root,
    edge_record,
):


    modified = baseline_ego.copy()

    source = edge_record[
        "source"
    ]

    target = edge_record[
        "target"
    ]

    key = edge_record[
        "key"
    ]

    endpoint = edge_record[
        "endpoint"
    ]

    if modified.is_multigraph():

        if modified.has_edge(
            source,
            target,
            key,
        ):

            modified.remove_edge(
                source,
                target,
                key,
            )

        else:

            raise KeyError(
                "Selected deletion edge is missing "
                "from the baseline ego."
            )

    else:

        if modified.has_edge(
            source,
            target,
        ):

            modified.remove_edge(
                source,
                target,
            )

        else:

            raise KeyError(
                "Selected deletion edge is missing "
                "from the baseline ego."
            )

    if not root_endpoint_still_adjacent(
        modified,
        root,
        endpoint,
    ):

        if endpoint in modified:

            modified.remove_node(
                endpoint
            )

    return modified


def build_type_pools(
    graph,
):
    pools = {}

    for node, data in graph.nodes(
        data=True
    ):

        node_type = str(
            data.get(
                "type",
                "unknown",
            )
        ).lower()

        pools.setdefault(
            node_type,
            []
        ).append(
            node
        )

    return pools


def choose_non_neighbour(
    graph,
    root,
    required_type,
    type_pools,
    rng,
    maximum_random_attempts=200,
):


    pool = type_pools.get(
        required_type,
        [],
    )

    if not pool:

        return None

    neighbours = get_root_neighbours(
        graph,
        root,
    )

    for _ in range(
        maximum_random_attempts
    ):

        candidate = pool[
            int(
                rng.integers(
                    0,
                    len(
                        pool
                    ),
                )
            )
        ]

        if (
            candidate != root
            and candidate not in neighbours
        ):

            return candidate


    for candidate in pool:

        if (
            candidate != root
            and candidate not in neighbours
        ):

            return candidate

    return None


def ego_after_edge_addition(
    global_graph,
    baseline_ego,
    root,
    template_edge,
    candidate,
):


    node_set = set(
        baseline_ego.nodes()
    )

    node_set.add(
        candidate
    )

    modified = global_graph.subgraph(
        node_set
    ).copy()

    relation = template_edge[
        "relation"
    ]

    direction = template_edge[
        "direction"
    ]

    if direction == "out":

        modified.add_edge(
            root,
            candidate,
            relation=relation,
            perturbation=True,
        )

    elif direction == "in":

        modified.add_edge(
            candidate,
            root,
            relation=relation,
            perturbation=True,
        )

    else:

        modified.add_edge(
            root,
            candidate,
            relation=relation,
            perturbation=True,
        )

    return modified


def edge_multiset(
    graph,
):


    counter = Counter()

    if graph.is_multigraph():

        for (
            source,
            target,
            key,
            data,
        ) in graph.edges(
            keys=True,
            data=True,
        ):

            relation = str(
                data.get(
                    "relation",
                    "unknown",
                )
            )

            counter[
                (
                    str(
                        source
                    ),
                    str(
                        target
                    ),
                    relation,
                )
            ] += 1

    else:

        for (
            source,
            target,
            data,
        ) in graph.edges(
            data=True
        ):

            relation = str(
                data.get(
                    "relation",
                    "unknown",
                )
            )

            counter[
                (
                    str(
                        source
                    ),
                    str(
                        target
                    ),
                    relation,
                )
            ] += 1

    return counter


def edge_multiset_distance(
    first,
    second,
):
    first_counter = edge_multiset(
        first
    )

    second_counter = edge_multiset(
        second
    )

    keys = (
        set(
            first_counter
        )
        |
        set(
            second_counter
        )
    )

    return int(
        sum(
            abs(
                first_counter[
                    key
                ]
                -
                second_counter[
                    key
                ]
            )
            for key
            in keys
        )
    )


def signature_distances(
    clean,
    modified,
):
    delta = (
        np.asarray(
            modified,
            dtype=float,
        )
        -
        np.asarray(
            clean,
            dtype=float,
        )
    )

    return {
        "hks_l1": float(
            np.linalg.norm(
                delta,
                ord=1,
            )
        ),

        "hks_l2": float(
            np.linalg.norm(
                delta,
                ord=2,
            )
        ),

        "hks_linf": float(
            np.linalg.norm(
                delta,
                ord=np.inf,
            )
        ),
    }


def evaluate_perturbation(
    root,
    baseline_ego,
    modified_ego,
    operation,
    edge_class,
    relation,
    direction,
    endpoint_type,
    signature_dimensions,
):


    clean_degree = typed_root_degree(
        baseline_ego,
        root,
    )

    modified_degree = typed_root_degree(
        modified_ego,
        root,
    )

    degree_delta = (
        modified_degree
        -
        clean_degree
    )

    degree_l1 = float(
        np.linalg.norm(
            degree_delta,
            ord=1,
        )
    )

    degree_l2 = float(
        np.linalg.norm(
            degree_delta,
            ord=2,
        )
    )

    degree_linf = float(
        np.linalg.norm(
            degree_delta,
            ord=np.inf,
        )
    )


    if degree_l1 > 1.0 + 1e-12:

        raise RuntimeError(
            "Typed degree changed by more than one under a "
            "single incident-edge perturbation. "
            "This indicates an implementation problem."
        )

    edge_change = edge_multiset_distance(
        baseline_ego,
        modified_ego,
    )

    node_change = len(
        set(
            baseline_ego.nodes()
        )
        ^
        set(
            modified_ego.nodes()
        )
    )

    rows = []

    for dimension in signature_dimensions:

        clean_signature = compute_hks_signature(
            baseline_ego,
            root,
            dimension,
        )

        modified_signature = compute_hks_signature(
            modified_ego,
            root,
            dimension,
        )

        distances = signature_distances(
            clean_signature,
            modified_signature,
        )

        clean_norm = float(
            np.linalg.norm(
                clean_signature
            )
        )

        modified_norm = float(
            np.linalg.norm(
                modified_signature
            )
        )

        clean_nonnegative = bool(
            np.all(
                clean_signature
                >=
                -1e-12
            )
        )

        modified_nonnegative = bool(
            np.all(
                modified_signature
                >=
                -1e-12
            )
        )

        rows.append({
            "client_id": str(
                root
            ),

            "operation": (
                operation
            ),

            "edge_class": (
                edge_class
            ),

            "relation": (
                relation
            ),

            "direction": (
                direction
            ),

            "endpoint_type": (
                endpoint_type
            ),

            "signature_dimension": int(
                dimension
            ),

            "hks_times": json.dumps(
                hks_times_for_dimension(
                    dimension
                ).tolist()
            ),

            **distances,

            "clean_signature_l2": (
                clean_norm
            ),

            "modified_signature_l2": (
                modified_norm
            ),

            "clean_nonnegative": (
                clean_nonnegative
            ),

            "modified_nonnegative": (
                modified_nonnegative
            ),

            "clean_client_degree": int(
                clean_degree[
                    0
                ]
            ),

            "clean_content_degree": int(
                clean_degree[
                    1
                ]
            ),

            "modified_client_degree": int(
                modified_degree[
                    0
                ]
            ),

            "modified_content_degree": int(
                modified_degree[
                    1
                ]
            ),

            "degree_delta_client": int(
                degree_delta[
                    0
                ]
            ),

            "degree_delta_content": int(
                degree_delta[
                    1
                ]
            ),

            "degree_l1": (
                degree_l1
            ),

            "degree_l2": (
                degree_l2
            ),

            "degree_linf": (
                degree_linf
            ),


            "underlying_edge_changes": 1,

            "induced_ego_edge_multiset_change": int(
                edge_change
            ),

            "induced_ego_node_change": int(
                node_change
            ),

            "baseline_ego_nodes": int(
                baseline_ego.number_of_nodes()
            ),

            "baseline_ego_edges": int(
                baseline_ego.number_of_edges()
            ),

            "modified_ego_nodes": int(
                modified_ego.number_of_nodes()
            ),

            "modified_ego_edges": int(
                modified_ego.number_of_edges()
            ),
        })

    return rows


def load_experiment_clients(
    dataset_name,
    output_folder,
    global_graph,
    max_clients,
    seed,
):


    index_path = os.path.join(
        output_folder,
        dataset_name,
        f"{dataset_name}_ego_graph_index.csv.gz",
    )

    clients = []

    if os.path.exists(
        index_path
    ):

        dataframe = pd.read_csv(
            index_path,
            compression="gzip",
            dtype=str,
        )

        possible_columns = [
            "client_id",
            "client",
            "root",
            "main_node",
            "node",
        ]

        client_column = None

        for candidate in possible_columns:

            if candidate in dataframe.columns:

                client_column = candidate
                break

        if client_column is None:

            raise KeyError(
                "Could not locate the client column in "
                f"{index_path}. "
                f"Columns: {list(dataframe.columns)}"
            )

        clients = dataframe[
            client_column
        ].astype(
            str
        ).tolist()

    else:

        clients = [
            str(
                node
            )
            for node, data
            in global_graph.nodes(
                data=True
            )
            if str(
                data.get(
                    "role",
                    ""
                )
            ).lower()
            ==
            "client"
        ]

    clients = [
        client
        for client
        in clients
        if client in global_graph
    ]

    if max_clients is not None:

        maximum = min(
            int(
                max_clients
            ),
            len(
                clients
            ),
        )

        if maximum < len(
            clients
        ):

            rng = np.random.default_rng(
                seed
            )

            selected = np.sort(
                rng.choice(
                    len(
                        clients
                    ),
                    size=maximum,
                    replace=False,
                )
            )

            clients = [
                clients[
                    int(
                        index
                    )
                ]
                for index
                in selected
            ]

    return clients


def sample_deletion_edges(
    records,
    edge_class,
    maximum,
    rng,
):
    eligible = [
        record
        for record
        in records
        if record[
            "edge_class"
        ]
        ==
        edge_class
    ]

    if not eligible:

        return []

    number = min(
        int(
            maximum
        ),
        len(
            eligible
        ),
    )

    if number == len(
        eligible
    ):

        return eligible

    indices = rng.choice(
        len(
            eligible
        ),
        size=number,
        replace=False,
    )

    return [
        eligible[
            int(
                index
            )
        ]
        for index
        in indices
    ]


def generate_addition_cases(
    global_graph,
    root,
    incident_records,
    edge_class,
    type_pools,
    maximum,
    rng,
):


    templates = [
        record
        for record
        in incident_records
        if record[
            "edge_class"
        ]
        ==
        edge_class
    ]

    if not templates:

        return []

    cases = []

    used_candidates = set()

    attempts = 0

    maximum_attempts = max(
        100,
        maximum * 30,
    )

    while (
        len(
            cases
        )
        <
        maximum
        and attempts
        <
        maximum_attempts
    ):

        attempts += 1

        template = templates[
            int(
                rng.integers(
                    0,
                    len(
                        templates
                    ),
                )
            )
        ]

        candidate = choose_non_neighbour(
            graph=global_graph,

            root=root,

            required_type=(
                template[
                    "endpoint_type"
                ]
            ),

            type_pools=(
                type_pools
            ),

            rng=rng,
        )

        if candidate is None:
            continue

        if candidate in used_candidates:
            continue

        used_candidates.add(
            candidate
        )

        cases.append(
            (
                template,
                candidate,
            )
        )

    return cases


def make_summary(
    raw_dataframe,
):
    group_columns = [
        "operation",
        "edge_class",
        "signature_dimension",
    ]

    metrics = [
        "hks_l1",
        "hks_l2",
        "hks_linf",
        "degree_l1",
        "induced_ego_edge_multiset_change",
        "induced_ego_node_change",
    ]

    rows = []

    for (
        group_values,
        group,
    ) in raw_dataframe.groupby(
        group_columns,
        dropna=False,
    ):

        row = {
            key: value
            for key, value
            in zip(
                group_columns,
                group_values,
            )
        }

        row[
            "perturbations"
        ] = int(
            len(
                group
            )
        )

        for metric in metrics:

            summary = numeric_summary(
                group[
                    metric
                ].to_numpy(
                    dtype=float
                )
            )

            for statistic, value in summary.items():

                row[
                    f"{metric}_{statistic}"
                ] = value

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def make_relation_summary(
    raw_dataframe,
):
    group_columns = [
        "operation",
        "edge_class",
        "relation",
        "direction",
        "signature_dimension",
    ]

    rows = []

    for (
        group_values,
        group,
    ) in raw_dataframe.groupby(
        group_columns,
        dropna=False,
    ):

        hks_l2 = numeric_summary(
            group[
                "hks_l2"
            ]
        )

        row = {
            key: value
            for key, value
            in zip(
                group_columns,
                group_values,
            )
        }

        row[
            "perturbations"
        ] = int(
            len(
                group
            )
        )

        row[
            "hks_l2_mean"
        ] = hks_l2[
            "mean"
        ]

        row[
            "hks_l2_p95"
        ] = hks_l2[
            "p95"
        ]

        row[
            "hks_l2_p99"
        ] = hks_l2[
            "p99"
        ]

        row[
            "hks_l2_max"
        ] = hks_l2[
            "max"
        ]

        row[
            "mean_ego_edge_amplification"
        ] = float(
            group[
                "induced_ego_edge_multiset_change"
            ].mean()
        )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def make_maximum_table(
    raw_dataframe,
):
    rows = []

    group_columns = [
        "signature_dimension",
        "operation",
        "edge_class",
    ]

    for (
        group_values,
        group,
    ) in raw_dataframe.groupby(
        group_columns,
        dropna=False,
    ):

        maximum_index = group[
            "hks_l2"
        ].idxmax()

        maximum_row = raw_dataframe.loc[
            maximum_index
        ]

        row = {
            key: value
            for key, value
            in zip(
                group_columns,
                group_values,
            )
        }

        row.update({
            "maximum_observed_hks_l2": float(
                maximum_row[
                    "hks_l2"
                ]
            ),

            "corresponding_hks_l1": float(
                maximum_row[
                    "hks_l1"
                ]
            ),

            "corresponding_hks_linf": float(
                maximum_row[
                    "hks_linf"
                ]
            ),

            "client_id": str(
                maximum_row[
                    "client_id"
                ]
            ),

            "relation": str(
                maximum_row[
                    "relation"
                ]
            ),

            "direction": str(
                maximum_row[
                    "direction"
                ]
            ),

            "endpoint_type": str(
                maximum_row[
                    "endpoint_type"
                ]
            ),

            "ego_edge_amplification": int(
                maximum_row[
                    "induced_ego_edge_multiset_change"
                ]
            ),

            "ego_node_change": int(
                maximum_row[
                    "induced_ego_node_change"
                ]
            ),
        })

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def run_edge_sensitivity_experiment(
    dataset_name="enron",
    output_folder=DEFAULT_OUTPUT_FOLDER,
    max_clients=DEFAULT_MAX_CLIENTS,
    perturbations_per_category=(
        DEFAULT_PERTURBATIONS_PER_CATEGORY
    ),
    signature_dimensions=(
        DEFAULT_SIGNATURE_DIMENSIONS
    ),
    clip_norm_C=DEFAULT_CLIP_NORM_C,
    seed=DEFAULT_SEED,
):


    start = time.perf_counter()

    signature_dimensions = tuple(
        int(
            dimension
        )
        for dimension
        in signature_dimensions
    )

    for dimension in signature_dimensions:

        hks_times_for_dimension(
            dimension
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
            "Missing global graph:\n"
            f"  {graph_path}\n\n"
            "Run the integration/core pipeline first."
        )

    print(
        "\n"
        "============================================================"
    )

    print(
        "EDGE SENSITIVITY EXPERIMENT"
    )

    print(
        "============================================================"
    )

    print(
        f"Dataset: {dataset_name}"
    )

    print(
        f"Loading: {graph_path}"
    )

    global_graph = nx.read_gml(
        graph_path
    )

    if not global_graph.is_directed():

        raise RuntimeError(
            "Expected revised directed heterogeneous global graph."
        )

    if not global_graph.is_multigraph():

        raise RuntimeError(
            "Expected revised MultiDiGraph."
        )


    client_seed = deterministic_seed(
        dataset_name,
        seed,
        "edge_sensitivity_clients",
    )

    clients = load_experiment_clients(
        dataset_name=dataset_name,

        output_folder=output_folder,

        global_graph=global_graph,

        max_clients=max_clients,

        seed=client_seed,
    )

    if not clients:

        raise RuntimeError(
            "No clients available for the experiment."
        )

    print(
        f"Clients: {len(clients):,}"
    )

    print(
        "Signature dimensions: "
        f"{list(signature_dimensions)}"
    )

    print(
        "Perturbations/category/client: "
        f"{perturbations_per_category}"
    )


    type_pools = build_type_pools(
        global_graph
    )

    rows = []

    failures = []

    counters = {
        "deletion_client_client": 0,
        "deletion_client_content": 0,
        "addition_client_client": 0,
        "addition_client_content": 0,
    }


    for client_index, root in enumerate(
        clients,
        start=1,
    ):

        if (
            client_index == 1
            or client_index % 20 == 0
            or client_index == len(
                clients
            )
        ):

            print(
                "[edge sensitivity] "
                f"client {client_index}/{len(clients)}"
            )

        try:

            baseline_ego = create_true_ego_graph(
                global_graph,
                root,
            )

            incident_records = incident_edge_records(
                global_graph,
                root,
            )

            for edge_class in [
                "client_client",
                "client_content",
            ]:


                deletion_seed = deterministic_seed(
                    dataset_name,
                    root,
                    edge_class,
                    seed,
                    "delete",
                )

                deletion_rng = np.random.default_rng(
                    deletion_seed
                )

                deletion_edges = sample_deletion_edges(
                    records=incident_records,

                    edge_class=edge_class,

                    maximum=(
                        perturbations_per_category
                    ),

                    rng=deletion_rng,
                )

                for edge_record in deletion_edges:

                    try:

                        modified_ego = (
                            ego_after_edge_deletion(
                                baseline_ego=(
                                    baseline_ego
                                ),

                                root=root,

                                edge_record=(
                                    edge_record
                                ),
                            )
                        )

                        perturbation_rows = (
                            evaluate_perturbation(
                                root=root,

                                baseline_ego=(
                                    baseline_ego
                                ),

                                modified_ego=(
                                    modified_ego
                                ),

                                operation="delete",

                                edge_class=(
                                    edge_class
                                ),

                                relation=(
                                    edge_record[
                                        "relation"
                                    ]
                                ),

                                direction=(
                                    edge_record[
                                        "direction"
                                    ]
                                ),

                                endpoint_type=(
                                    edge_record[
                                        "endpoint_type"
                                    ]
                                ),

                                signature_dimensions=(
                                    signature_dimensions
                                ),
                            )
                        )

                        rows.extend(
                            perturbation_rows
                        )

                        counters[
                            f"deletion_"
                            f"{edge_class}"
                        ] += 1

                    except Exception as exc:

                        failures.append({
                            "client_id": str(
                                root
                            ),

                            "operation": (
                                "delete"
                            ),

                            "edge_class": (
                                edge_class
                            ),

                            "relation": (
                                edge_record[
                                    "relation"
                                ]
                            ),

                            "error": repr(
                                exc
                            ),
                        })


                addition_seed = deterministic_seed(
                    dataset_name,
                    root,
                    edge_class,
                    seed,
                    "add",
                )

                addition_rng = np.random.default_rng(
                    addition_seed
                )

                addition_cases = generate_addition_cases(
                    global_graph=(
                        global_graph
                    ),

                    root=root,

                    incident_records=(
                        incident_records
                    ),

                    edge_class=edge_class,

                    type_pools=type_pools,

                    maximum=(
                        perturbations_per_category
                    ),

                    rng=addition_rng,
                )

                for (
                    template,
                    candidate,
                ) in addition_cases:

                    try:

                        modified_ego = (
                            ego_after_edge_addition(
                                global_graph=(
                                    global_graph
                                ),

                                baseline_ego=(
                                    baseline_ego
                                ),

                                root=root,

                                template_edge=(
                                    template
                                ),

                                candidate=(
                                    candidate
                                ),
                            )
                        )

                        perturbation_rows = (
                            evaluate_perturbation(
                                root=root,

                                baseline_ego=(
                                    baseline_ego
                                ),

                                modified_ego=(
                                    modified_ego
                                ),

                                operation="add",

                                edge_class=(
                                    edge_class
                                ),

                                relation=(
                                    template[
                                        "relation"
                                    ]
                                ),

                                direction=(
                                    template[
                                        "direction"
                                    ]
                                ),

                                endpoint_type=(
                                    template[
                                        "endpoint_type"
                                    ]
                                ),

                                signature_dimensions=(
                                    signature_dimensions
                                ),
                            )
                        )

                        rows.extend(
                            perturbation_rows
                        )

                        counters[
                            f"addition_"
                            f"{edge_class}"
                        ] += 1

                    except Exception as exc:

                        failures.append({
                            "client_id": str(
                                root
                            ),

                            "operation": (
                                "add"
                            ),

                            "edge_class": (
                                edge_class
                            ),

                            "relation": (
                                template[
                                    "relation"
                                ]
                            ),

                            "error": repr(
                                exc
                            ),
                        })

        except Exception as exc:

            failures.append({
                "client_id": str(
                    root
                ),

                "operation": (
                    "client_setup"
                ),

                "edge_class": (
                    "all"
                ),

                "error": repr(
                    exc
                ),
            })


    if not rows:

        raise RuntimeError(
            "No valid edge perturbations were evaluated."
        )

    raw_dataframe = pd.DataFrame(
        rows
    )


    maximum_degree_l1 = float(
        raw_dataframe[
            "degree_l1"
        ].max()
    )

    if maximum_degree_l1 > 1.0 + 1e-12:

        raise RuntimeError(
            "Observed typed degree L1 change exceeded 1."
        )


    if not raw_dataframe[
        "clean_nonnegative"
    ].all():

        raise RuntimeError(
            "A clean HKS signature contained a negative coordinate."
        )

    if not raw_dataframe[
        "modified_nonnegative"
    ].all():

        raise RuntimeError(
            "A perturbed HKS signature contained a negative coordinate."
        )


    summary_dataframe = make_summary(
        raw_dataframe
    )

    relation_dataframe = make_relation_summary(
        raw_dataframe
    )

    maximum_dataframe = make_maximum_table(
        raw_dataframe
    )


    experiment_folder = os.path.join(
        output_folder,
        dataset_name,
        "experiments",
        "edge_sensitivity",
    )

    os.makedirs(
        experiment_folder,
        exist_ok=True,
    )

    raw_path = os.path.join(
        experiment_folder,
        "edge_sensitivity_raw.csv.gz",
    )

    summary_path = os.path.join(
        experiment_folder,
        "edge_sensitivity_summary.csv",
    )

    relation_path = os.path.join(
        experiment_folder,
        "edge_sensitivity_by_relation.csv",
    )

    maximum_path = os.path.join(
        experiment_folder,
        "edge_sensitivity_maxima.csv",
    )

    metadata_path = os.path.join(
        experiment_folder,
        "edge_sensitivity_metadata.json",
    )

    raw_dataframe.to_csv(
        raw_path,
        index=False,
        compression="gzip",
    )

    summary_dataframe.to_csv(
        summary_path,
        index=False,
    )

    relation_dataframe.to_csv(
        relation_path,
        index=False,
    )

    maximum_dataframe.to_csv(
        maximum_path,
        index=False,
    )


    generic_clipped_l2_bound = (
        2.0
        *
        float(
            clip_norm_C
        )
    )

    nonnegative_hks_l2_bound = (
        math.sqrt(
            2.0
        )
        *
        float(
            clip_norm_C
        )
    )

    runtime = float(
        time.perf_counter()
        -
        start
    )

    metadata = {
        "schema_version": (
            EDGE_SENSITIVITY_SCHEMA_VERSION
        ),

        "dataset": dataset_name,

        "global_graph": (
            graph_path
        ),

        "number_of_clients": int(
            len(
                clients
            )
        ),

        "perturbations_per_category_per_client": int(
            perturbations_per_category
        ),

        "signature_dimensions": [
            int(
                dimension
            )
            for dimension
            in signature_dimensions
        ],

        "dimension_specific_times": {
            str(
                dimension
            ): hks_times_for_dimension(
                dimension
            ).tolist()
            for dimension
            in signature_dimensions
        },

        "seed": int(
            seed
        ),

        "perturbation_counts": (
            counters
        ),

        "failures": (
            failures
        ),

        "number_of_failures": int(
            len(
                failures
            )
        ),

        "degree_result": {
            "degree_vector": (
                "[unique client neighbours, "
                "unique content neighbours]"
            ),

            "maximum_observed_l1_change": float(
                maximum_degree_l1
            ),

            "analytical_incident_edge_l1_sensitivity": (
                1.0
            ),
        },

        "hks_range_bounds": {
            "clip_norm_C": float(
                clip_norm_C
            ),

            "generic_clipped_vector_l2_diameter": float(
                generic_clipped_l2_bound
            ),

            "nonnegative_hks_l2_diameter": float(
                nonnegative_hks_l2_bound
            ),

            "note": (
                "The non-negative HKS range bound follows from "
                "<x,y> >= 0 and ||x||_2, ||y||_2 <= C. "
                "It is a range-based bound and is not a tighter "
                "incident-edge sensitivity proof."
            ),
        },

        "privacy_warning": (
            "The maximum HKS differences in this experiment are "
            "EMPIRICAL OBSERVATIONS. They must not be used directly "
            "as the sensitivity parameter of a pure-DP mechanism. "
            "A formal global analytical bound over every valid "
            "incident-edge neighbouring pair is required first."
        ),

        "ego_amplification_note": (
            "Neighbouring GLOBAL graphs differ in exactly one edge "
            "incident to the root. Because the released local object "
            "is a one-hop induced ego graph, one underlying edge "
            "change can cause several ego nodes/edges to disappear "
            "or appear. The experiment measures this amplification."
        ),

        "outputs": {
            "raw": (
                os.path.basename(
                    raw_path
                )
            ),

            "summary": (
                os.path.basename(
                    summary_path
                )
            ),

            "relation_summary": (
                os.path.basename(
                    relation_path
                )
            ),

            "maxima": (
                os.path.basename(
                    maximum_path
                )
            ),
        },

        "runtime_seconds": float(
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


    print(
        "\n"
        "============================================================"
    )

    print(
        "EDGE SENSITIVITY COMPLETE"
    )

    print(
        "============================================================"
    )

    print(
        f"Clients evaluated: {len(clients):,}"
    )

    print(
        "\nUnderlying perturbations:"
    )

    for key, value in counters.items():

        print(
            f"  {key:30s} {value:,}"
        )

    print(
        "\nTyped degree check:"
    )

    print(
        "  maximum observed L1 change = "
        f"{maximum_degree_l1:.6f}"
    )

    print(
        "  analytical incident-edge bound = 1"
    )

    print(
        "\nObserved HKS L2 differences:"
    )

    overall = (
        raw_dataframe
        .groupby(
            "signature_dimension"
        )[
            "hks_l2"
        ]
        .agg(
            [
                "mean",
                "median",
                "max",
            ]
        )
    )

    for dimension, row in overall.iterrows():

        print(
            f"  d={int(dimension):2d}  "
            f"mean={float(row['mean']):.6f}  "
            f"median={float(row['median']):.6f}  "
            f"max={float(row['max']):.6f}"
        )

    print(
        "\nReference L2 range bounds for C="
        f"{clip_norm_C:g}:"
    )

    print(
        "  generic clipped report: "
        f"{generic_clipped_l2_bound:.6f}"
    )

    print(
        "  non-negative HKS report: "
        f"{nonnegative_hks_l2_bound:.6f}"
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "  Observed maxima are NOT DP sensitivity proofs."
    )

    print(
        "\nOutputs:"
    )

    print(
        f"  {raw_path}"
    )

    print(
        f"  {summary_path}"
    )

    print(
        f"  {relation_path}"
    )

    print(
        f"  {maximum_path}"
    )

    print(
        f"  {metadata_path}"
    )

    print(
        f"\nRuntime: {runtime:.3f} s"
    )

    print(
        "============================================================\n"
    )

    return {
        "raw": (
            raw_dataframe
        ),

        "summary": (
            summary_dataframe
        ),

        "by_relation": (
            relation_dataframe
        ),

        "maxima": (
            maximum_dataframe
        ),

        "metadata": (
            metadata
        ),
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Empirically study local HKS changes under one "
            "client-incident edge modification."
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
        "--max-clients",
        type=int,
        default=(
            DEFAULT_MAX_CLIENTS
        ),
    )

    parser.add_argument(
        "--perturbations",
        type=int,
        default=(
            DEFAULT_PERTURBATIONS_PER_CATEGORY
        ),
    )

    parser.add_argument(
        "--dimensions",
        nargs="+",
        type=int,
        default=list(
            DEFAULT_SIGNATURE_DIMENSIONS
        ),
    )

    parser.add_argument(
        "--C",
        type=float,
        default=(
            DEFAULT_CLIP_NORM_C
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

    run_edge_sensitivity_experiment(
        dataset_name=(
            args.dataset
        ),

        output_folder=(
            args.output_folder
        ),

        max_clients=(
            args.max_clients
        ),

        perturbations_per_category=(
            args.perturbations
        ),

        signature_dimensions=(
            args.dimensions
        ),

        clip_norm_C=(
            args.C
        ),

        seed=(
            args.seed
        ),
    )
