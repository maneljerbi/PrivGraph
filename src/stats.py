import json
import math
import os
import random
from collections import Counter, defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import rbf_kernel


try:
    import community.community_louvain as community_louvain

    HAS_LOUVAIN = True

except ImportError:
    community_louvain = None
    HAS_LOUVAIN = False


STATS_SCHEMA_VERSION = "3.0"

NUMERICAL_EPS = 1e-12

DEFAULT_SEED = 0

DEFAULT_MAX_MMD_NODES = 2000

DEFAULT_MAX_PATH_SOURCES = 256

DEFAULT_EXACT_CLUSTERING_NODES = 20000

DEFAULT_APPROX_CLUSTERING_TRIALS = 2000


ORACLE_OBJECTIVE_WEIGHTS = {
    "degree": 0.30,
    "relation": 0.20,
    "clustering": 0.15,
    "modularity": 0.10,
    "metapath": 0.15,
    "structural_mmd": 0.10,
}


def json_safe_number(value):


    if value is None:
        return None

    try:

        value = float(
            value
        )

    except Exception:
        return None

    if not np.isfinite(
        value
    ):
        return None

    return value


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


def safe_relative_error(
    synthetic,
    original,
):


    return float(
        abs(
            float(
                synthetic
            )
            -
            float(
                original
            )
        )
        /
        max(
            abs(
                float(
                    original
                )
            ),
            1.0,
        )
    )


def safe_absolute_difference(
    value_a,
    value_b,
):


    if (
        value_a is None
        or value_b is None
    ):
        return None

    return float(
        abs(
            float(
                value_a
            )
            -
            float(
                value_b
            )
        )
    )


def to_simple_undirected(
    graph,
):


    H = nx.Graph()

    for node, data in graph.nodes(
        data=True
    ):

        H.add_node(
            str(
                node
            ),
            **dict(
                data
            ),
        )

    if graph.is_multigraph():

        iterator = graph.edges(
            keys=True,
            data=True,
        )

        for u, v, _, _ in iterator:

            u = str(
                u
            )

            v = str(
                v
            )

            if u == v:
                continue

            H.add_edge(
                u,
                v,
            )

    else:

        iterator = graph.edges(
            data=True
        )

        for u, v, _ in iterator:

            u = str(
                u
            )

            v = str(
                v
            )

            if u == v:
                continue

            H.add_edge(
                u,
                v,
            )

    return H


def get_unique_neighbours(
    graph,
    node,
):


    if node not in graph:
        return set()

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


def iterate_typed_edges(
    graph,
):


    if graph.is_multigraph():

        for u, v, _, data in graph.edges(
            keys=True,
            data=True,
        ):

            yield (
                str(
                    u
                ),
                str(
                    v
                ),
                str(
                    data.get(
                        "relation",
                        "undefined",
                    )
                ),
            )

    else:

        for u, v, data in graph.edges(
            data=True,
        ):

            yield (
                str(
                    u
                ),
                str(
                    v
                ),
                str(
                    data.get(
                        "relation",
                        "undefined",
                    )
                ),
            )


def node_role(
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
    )


def node_type(
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
    )


def canonical_node_type(
    graph,
    node,
):


    role = node_role(
        graph,
        node,
    )

    if role == "client":
        return "client"

    if role == "content":

        value = node_type(
            graph,
            node,
        )

        if value == "synthetic_content":
            return "content"

        return value

    return node_type(
        graph,
        node,
    )


def canonical_role_pair(
    graph,
    u,
    v,
):


    roles = sorted(
        [
            node_role(
                graph,
                u,
            ),
            node_role(
                graph,
                v,
            ),
        ]
    )

    return (
        f"{roles[0]}"
        f"--"
        f"{roles[1]}"
    )


def canonical_type_pair(
    graph,
    u,
    v,
):


    types = sorted(
        [
            canonical_node_type(
                graph,
                u,
            ),
            canonical_node_type(
                graph,
                v,
            ),
        ]
    )

    return (
        f"{types[0]}"
        f"--"
        f"{types[1]}"
    )


def canonical_relation_class(
    graph,
    u,
    v,
):


    role_u = node_role(
        graph,
        u,
    )

    role_v = node_role(
        graph,
        v,
    )

    roles = {
        role_u,
        role_v,
    }

    if roles == {
        "client"
    }:

        return "client_client"

    if roles == {
        "client",
        "content",
    }:

        return "client_content"

    if roles == {
        "content"
    }:

        return "content_content"

    return "other"


def count_node_categories(
    graph,
):


    exact_types = Counter()

    canonical_types = Counter()

    roles = Counter()

    for node in graph.nodes():

        exact_types[
            node_type(
                graph,
                node,
            )
        ] += 1

        canonical_types[
            canonical_node_type(
                graph,
                node,
            )
        ] += 1

        roles[
            node_role(
                graph,
                node,
            )
        ] += 1

    return {
        "exact_type_counts": dict(
            exact_types
        ),

        "canonical_type_counts": dict(
            canonical_types
        ),

        "role_counts": dict(
            roles
        ),
    }


def count_edge_categories(
    graph,
):


    exact_relations = Counter()

    role_pairs = Counter()

    type_pairs = Counter()

    canonical_relations = Counter()

    for u, v, relation in iterate_typed_edges(
        graph
    ):

        exact_relations[
            relation
        ] += 1

        role_pairs[
            canonical_role_pair(
                graph,
                u,
                v,
            )
        ] += 1

        type_pairs[
            canonical_type_pair(
                graph,
                u,
                v,
            )
        ] += 1

        canonical_relations[
            canonical_relation_class(
                graph,
                u,
                v,
            )
        ] += 1

    return {
        "exact_relation_counts": dict(
            exact_relations
        ),

        "role_pair_counts": dict(
            role_pairs
        ),

        "canonical_type_pair_counts": dict(
            type_pairs
        ),

        "canonical_relation_counts": dict(
            canonical_relations
        ),
    }


def all_degree_values(
    graph,
):


    return np.asarray(
        [
            graph.degree(
                node
            )
            for node
            in graph.nodes()
        ],
        dtype=float,
    )


def unique_degree_values(
    graph,
):


    return np.asarray(
        [
            len(
                get_unique_neighbours(
                    graph,
                    node,
                )
            )
            for node
            in graph.nodes()
        ],
        dtype=float,
    )


def degree_values_for_nodes(
    graph,
    nodes,
    unique=True,
):


    values = []

    for node in nodes:

        if node not in graph:
            continue

        if unique:

            values.append(
                len(
                    get_unique_neighbours(
                        graph,
                        node,
                    )
                )
            )

        else:

            values.append(
                graph.degree(
                    node
                )
            )

    return np.asarray(
        values,
        dtype=float,
    )


def get_nodes_by_role(
    graph,
    role,
):


    return [
        node
        for node
        in graph.nodes()
        if node_role(
            graph,
            node,
        )
        == role
    ]


def get_nodes_by_canonical_type(
    graph,
    category,
):


    return [
        node
        for node
        in graph.nodes()
        if canonical_node_type(
            graph,
            node,
        )
        == category
    ]


def relation_degree_values(
    graph,
    relation,
    nodes=None,
):


    if nodes is None:

        nodes = list(
            graph.nodes()
        )

    counts = {
        node: 0
        for node
        in nodes
        if node in graph
    }

    for u, v, edge_relation in iterate_typed_edges(
        graph
    ):

        if edge_relation != relation:
            continue

        if u in counts:

            counts[
                u
            ] += 1

        if v in counts:

            counts[
                v
            ] += 1

    return np.asarray(
        [
            counts[
                node
            ]
            for node
            in counts
        ],
        dtype=float,
    )


def canonical_relation_degree_values(
    graph,
    relation_class,
    nodes=None,
):


    if nodes is None:

        nodes = list(
            graph.nodes()
        )

    counts = {
        node: 0
        for node
        in nodes
        if node in graph
    }

    for u, v, _ in iterate_typed_edges(
        graph
    ):

        if (
            canonical_relation_class(
                graph,
                u,
                v,
            )
            != relation_class
        ):
            continue

        if u in counts:

            counts[
                u
            ] += 1

        if v in counts:

            counts[
                v
            ] += 1

    return np.asarray(
        [
            counts[
                node
            ]
            for node
            in counts
        ],
        dtype=float,
    )


def compare_numeric_distributions(
    original_values,
    synthetic_values,
):


    original_values = np.asarray(
        original_values,
        dtype=float,
    )

    synthetic_values = np.asarray(
        synthetic_values,
        dtype=float,
    )

    original_values = original_values[
        np.isfinite(
            original_values
        )
    ]

    synthetic_values = synthetic_values[
        np.isfinite(
            synthetic_values
        )
    ]

    if (
        len(
            original_values
        )
        == 0
        or len(
            synthetic_values
        )
        == 0
    ):

        return {
            "original_count": int(
                len(
                    original_values
                )
            ),

            "synthetic_count": int(
                len(
                    synthetic_values
                )
            ),

            "wasserstein": None,

            "normalized_wasserstein": None,

            "mean_absolute_error": None,

            "median_absolute_error": None,

            "std_absolute_error": None,
        }

    wd = float(
        wasserstein_distance(
            original_values,
            synthetic_values,
        )
    )

    normalization = max(
        float(
            np.mean(
                original_values
            )
        ),
        1.0,
    )

    return {
        "original_count": int(
            len(
                original_values
            )
        ),

        "synthetic_count": int(
            len(
                synthetic_values
            )
        ),

        "wasserstein": float(
            wd
        ),

        "normalized_wasserstein": float(
            wd
            /
            normalization
        ),

        "original_mean": float(
            np.mean(
                original_values
            )
        ),

        "synthetic_mean": float(
            np.mean(
                synthetic_values
            )
        ),

        "mean_absolute_error": float(
            abs(
                np.mean(
                    synthetic_values
                )
                -
                np.mean(
                    original_values
                )
            )
        ),

        "original_median": float(
            np.median(
                original_values
            )
        ),

        "synthetic_median": float(
            np.median(
                synthetic_values
            )
        ),

        "median_absolute_error": float(
            abs(
                np.median(
                    synthetic_values
                )
                -
                np.median(
                    original_values
                )
            )
        ),

        "original_std": float(
            np.std(
                original_values
            )
        ),

        "synthetic_std": float(
            np.std(
                synthetic_values
            )
        ),

        "std_absolute_error": float(
            abs(
                np.std(
                    synthetic_values
                )
                -
                np.std(
                    original_values
                )
            )
        ),
    }


def normalized_proportions(
    counts,
):


    total = float(
        sum(
            counts.values()
        )
    )

    if total <= 0:

        return {
            key: 0.0
            for key
            in counts
        }

    return {
        key: float(
            value
            /
            total
        )
        for key, value
        in counts.items()
    }


def compare_count_dictionaries(
    original_counts,
    synthetic_counts,
):


    keys = sorted(
        set(
            original_counts
        )
        |
        set(
            synthetic_counts
        )
    )

    original_props = normalized_proportions(
        original_counts
    )

    synthetic_props = normalized_proportions(
        synthetic_counts
    )

    rows = []

    l1 = 0.0

    for key in keys:

        original_count = int(
            original_counts.get(
                key,
                0,
            )
        )

        synthetic_count = int(
            synthetic_counts.get(
                key,
                0,
            )
        )

        original_prop = float(
            original_props.get(
                key,
                0.0,
            )
        )

        synthetic_prop = float(
            synthetic_props.get(
                key,
                0.0,
            )
        )

        proportion_error = abs(
            synthetic_prop
            -
            original_prop
        )

        l1 += (
            proportion_error
        )

        rows.append({
            "category": str(
                key
            ),

            "original_count": (
                original_count
            ),

            "synthetic_count": (
                synthetic_count
            ),

            "count_absolute_error": int(
                abs(
                    synthetic_count
                    -
                    original_count
                )
            ),

            "count_relative_error": (
                safe_relative_error(
                    synthetic_count,
                    original_count,
                )
            ),

            "original_proportion": (
                original_prop
            ),

            "synthetic_proportion": (
                synthetic_prop
            ),

            "proportion_absolute_error": float(
                proportion_error
            ),
        })

    return {
        "rows": rows,

        "proportion_l1_error": float(
            l1
        ),

        "total_variation_distance": float(
            0.5
            * l1
        ),
    }


def average_clustering_safe(
    graph,
    seed=DEFAULT_SEED,
):


    H = to_simple_undirected(
        graph
    )

    if H.number_of_nodes() == 0:
        return 0.0

    if H.number_of_edges() == 0:
        return 0.0

    if (
        H.number_of_nodes()
        <= DEFAULT_EXACT_CLUSTERING_NODES
    ):

        return float(
            nx.average_clustering(
                H
            )
        )

    try:

        return float(
            nx.algorithms.approximation.average_clustering(
                H,
                trials=min(
                    DEFAULT_APPROX_CLUSTERING_TRIALS,
                    max(
                        H.number_of_nodes(),
                        100,
                    ),
                ),
                seed=seed,
            )
        )

    except Exception:


        rng = np.random.default_rng(
            seed
        )

        nodes = list(
            H.nodes()
        )

        sample_size = min(
            2000,
            len(
                nodes
            ),
        )

        selected = rng.choice(
            len(
                nodes
            ),
            size=sample_size,
            replace=False,
        )

        selected_nodes = [
            nodes[
                int(
                    index
                )
            ]
            for index in selected
        ]

        values = nx.clustering(
            H,
            nodes=selected_nodes,
        )

        return float(
            np.mean(
                list(
                    values.values()
                )
            )
        )


def client_projection(
    graph,
):


    client_nodes = set(
        get_nodes_by_role(
            graph,
            "client",
        )
    )

    H = nx.Graph()

    H.add_nodes_from(
        client_nodes
    )

    for u, v, _ in iterate_typed_edges(
        graph
    ):

        if (
            u in client_nodes
            and v in client_nodes
            and u != v
        ):

            H.add_edge(
                u,
                v,
            )

    return H


def calculate_modularity_safe(
    graph,
    seed=DEFAULT_SEED,
):


    H = to_simple_undirected(
        graph
    )

    if (
        H.number_of_nodes()
        == 0
        or H.number_of_edges()
        == 0
    ):

        return 0.0

    if HAS_LOUVAIN:

        try:

            partition = (
                community_louvain.best_partition(
                    H,
                    random_state=seed,
                )
            )

            return float(
                community_louvain.modularity(
                    partition,
                    H,
                )
            )

        except Exception:
            pass

    try:

        communities = list(
            nx.algorithms.community.greedy_modularity_communities(
                H
            )
        )

        if not communities:
            return 0.0

        return float(
            nx.algorithms.community.modularity(
                H,
                communities,
            )
        )

    except Exception:

        return 0.0


def degree_assortativity_safe(
    graph,
):


    H = to_simple_undirected(
        graph
    )

    if (
        H.number_of_nodes()
        < 2
        or H.number_of_edges()
        == 0
    ):

        return 0.0

    try:

        value = float(
            nx.degree_assortativity_coefficient(
                H
            )
        )

        if not np.isfinite(
            value
        ):

            return 0.0

        return value

    except Exception:

        return 0.0


def approximate_average_shortest_path_length(
    graph,
    seed=DEFAULT_SEED,
    max_sources=DEFAULT_MAX_PATH_SOURCES,
):


    H = to_simple_undirected(
        graph
    )

    if H.number_of_nodes() <= 1:
        return 0.0

    components = list(
        nx.connected_components(
            H
        )
    )

    if not components:
        return 0.0

    largest = max(
        components,
        key=len,
    )

    H = H.subgraph(
        largest
    )

    n = H.number_of_nodes()

    if n <= 1:
        return 0.0

    nodes = list(
        H.nodes()
    )

    if n <= max_sources:

        selected_nodes = nodes

    else:

        rng = np.random.default_rng(
            seed
        )

        indices = rng.choice(
            n,
            size=int(
                max_sources
            ),
            replace=False,
        )

        selected_nodes = [
            nodes[
                int(
                    index
                )
            ]
            for index
            in indices
        ]

    distances = []

    for source in selected_nodes:

        lengths = nx.single_source_shortest_path_length(
            H,
            source,
        )

        distances.extend(
            distance
            for target, distance
            in lengths.items()
            if target != source
        )

    if not distances:
        return 0.0

    return float(
        np.mean(
            distances
        )
    )


def connectivity_statistics(
    graph,
):


    H = to_simple_undirected(
        graph
    )

    n = H.number_of_nodes()

    if n == 0:

        return {
            "components": 0,
            "largest_component_size": 0,
            "largest_component_fraction": 0.0,
            "isolated_nodes": 0,
            "isolated_fraction": 0.0,
        }

    components = list(
        nx.connected_components(
            H
        )
    )

    component_sizes = [
        len(
            component
        )
        for component
        in components
    ]

    largest = max(
        component_sizes,
        default=0,
    )

    isolated = len(
        list(
            nx.isolates(
                H
            )
        )
    )

    return {
        "components": int(
            len(
                components
            )
        ),

        "largest_component_size": int(
            largest
        ),

        "largest_component_fraction": float(
            largest
            /
            n
        ),

        "isolated_nodes": int(
            isolated
        ),

        "isolated_fraction": float(
            isolated
            /
            n
        ),
    }


def mixing_matrix_counts(
    graph,
    label_mode="role",
):


    counts = Counter()

    for u, v, _ in iterate_typed_edges(
        graph
    ):

        if label_mode == "role":

            first = node_role(
                graph,
                u,
            )

            second = node_role(
                graph,
                v,
            )

        elif label_mode == "canonical_type":

            first = canonical_node_type(
                graph,
                u,
            )

            second = canonical_node_type(
                graph,
                v,
            )

        else:

            raise ValueError(
                "label_mode must be 'role' "
                "or 'canonical_type'."
            )

        first, second = sorted(
            [
                first,
                second,
            ]
        )

        counts[
            f"{first}--{second}"
        ] += 1

    return counts


def count_length_two_metapaths(
    graph,
    label_mode="role",
):


    H = to_simple_undirected(
        graph
    )

    counts = Counter()

    for middle in H.nodes():

        neighbours = list(
            H.neighbors(
                middle
            )
        )

        if len(
            neighbours
        ) < 2:
            continue

        if label_mode == "role":

            middle_label = node_role(
                graph,
                middle,
            )

            neighbour_labels = [
                node_role(
                    graph,
                    neighbour,
                )
                for neighbour
                in neighbours
            ]

        elif label_mode == "canonical_type":

            middle_label = (
                canonical_node_type(
                    graph,
                    middle,
                )
            )

            neighbour_labels = [
                canonical_node_type(
                    graph,
                    neighbour,
                )
                for neighbour
                in neighbours
            ]

        else:

            raise ValueError(
                "label_mode must be 'role' "
                "or 'canonical_type'."
            )

        label_counts = Counter(
            neighbour_labels
        )

        labels = sorted(
            label_counts.keys()
        )


        for label in labels:

            amount = label_counts[
                label
            ]

            if amount >= 2:

                path_name = (
                    f"{label}"
                    f"-{middle_label}-"
                    f"{label}"
                )

                counts[
                    path_name
                ] += (
                    amount
                    * (
                        amount - 1
                    )
                    // 2
                )


        for first_index in range(
            len(
                labels
            )
        ):

            for second_index in range(
                first_index + 1,
                len(
                    labels
                )
            ):

                first = labels[
                    first_index
                ]

                second = labels[
                    second_index
                ]

                path_name = (
                    f"{first}"
                    f"-{middle_label}-"
                    f"{second}"
                )

                counts[
                    path_name
                ] += (
                    label_counts[
                        first
                    ]
                    *
                    label_counts[
                        second
                    ]
                )

    return counts


def node_structural_features(
    graph,
    max_nodes=DEFAULT_MAX_MMD_NODES,
    seed=DEFAULT_SEED,
):


    H = to_simple_undirected(
        graph
    )

    nodes = list(
        H.nodes()
    )

    if not nodes:

        return np.empty(
            (
                0,
                4,
            ),
            dtype=float,
        )

    if (
        max_nodes is not None
        and len(
            nodes
        )
        > max_nodes
    ):

        rng = np.random.default_rng(
            seed
        )

        indices = rng.choice(
            len(
                nodes
            ),
            size=int(
                max_nodes
            ),
            replace=False,
        )

        nodes = [
            nodes[
                int(
                    index
                )
            ]
            for index in indices
        ]

    clustering = nx.clustering(
        H,
        nodes=nodes,
    )

    features = []

    for node in nodes:

        neighbours = list(
            H.neighbors(
                node
            )
        )

        degree = len(
            neighbours
        )

        if degree > 0:

            client_count = sum(
                1
                for neighbour
                in neighbours
                if node_role(
                    graph,
                    neighbour,
                )
                == "client"
            )

            content_count = sum(
                1
                for neighbour
                in neighbours
                if node_role(
                    graph,
                    neighbour,
                )
                == "content"
            )

            client_fraction = (
                client_count
                /
                degree
            )

            content_fraction = (
                content_count
                /
                degree
            )

        else:

            client_fraction = 0.0

            content_fraction = 0.0

        features.append(
            [
                math.log1p(
                    degree
                ),

                float(
                    clustering.get(
                        node,
                        0.0,
                    )
                ),

                float(
                    client_fraction
                ),

                float(
                    content_fraction
                ),
            ]
        )

    return np.asarray(
        features,
        dtype=float,
    )


def median_heuristic_gamma(
    X,
    Y,
    max_points=1000,
    seed=DEFAULT_SEED,
):


    Z = np.vstack(
        [
            X,
            Y,
        ]
    )

    if len(
        Z
    ) < 2:

        return 1.0

    if len(
        Z
    ) > max_points:

        rng = np.random.default_rng(
            seed
        )

        indices = rng.choice(
            len(
                Z
            ),
            size=max_points,
            replace=False,
        )

        Z = Z[
            indices
        ]


    norms = np.sum(
        Z * Z,
        axis=1,
        keepdims=True,
    )

    distances_squared = (
        norms
        +
        norms.T
        -
        2.0
        * (
            Z
            @ Z.T
        )
    )

    distances_squared = np.maximum(
        distances_squared,
        0.0,
    )

    upper = distances_squared[
        np.triu_indices(
            len(
                Z
            ),
            k=1,
        )
    ]

    upper = upper[
        upper
        > NUMERICAL_EPS
    ]

    if len(
        upper
    ) == 0:

        return 1.0

    median = float(
        np.median(
            upper
        )
    )

    if median <= NUMERICAL_EPS:

        return 1.0

    return float(
        1.0
        /
        (
            2.0
            * median
        )
    )


def compute_mmd(
    X,
    Y,
    gamma=None,
    seed=DEFAULT_SEED,
):


    X = np.asarray(
        X,
        dtype=float,
    )

    Y = np.asarray(
        Y,
        dtype=float,
    )

    if (
        len(
            X
        )
        == 0
        or len(
            Y
        )
        == 0
    ):

        return None

    if gamma is None:

        gamma = median_heuristic_gamma(
            X,
            Y,
            seed=seed,
        )

    K_XX = rbf_kernel(
        X,
        X,
        gamma=gamma,
    )

    K_YY = rbf_kernel(
        Y,
        Y,
        gamma=gamma,
    )

    K_XY = rbf_kernel(
        X,
        Y,
        gamma=gamma,
    )

    mmd_squared = (
        np.mean(
            K_XX
        )
        +
        np.mean(
            K_YY
        )
        -
        2.0
        * np.mean(
            K_XY
        )
    )

    return {
        "mmd_squared": float(
            max(
                mmd_squared,
                0.0,
            )
        ),

        "gamma": float(
            gamma
        ),

        "original_samples": int(
            len(
                X
            )
        ),

        "synthetic_samples": int(
            len(
                Y
            )
        ),
    }


def client_edge_set(
    graph,
):


    edges = set()

    for u, v, _ in iterate_typed_edges(
        graph
    ):

        if (
            node_role(
                graph,
                u,
            )
            != "client"
            or node_role(
                graph,
                v,
            )
            != "client"
            or u == v
        ):

            continue

        pair = tuple(
            sorted(
                [
                    str(
                        u
                    ),
                    str(
                        v
                    ),
                ]
            )
        )

        edges.add(
            pair
        )

    return edges


def client_edge_recovery_metrics(
    original_graph,
    synthetic_graph,
):


    original_edges = client_edge_set(
        original_graph
    )

    synthetic_edges = client_edge_set(
        synthetic_graph
    )

    intersection = (
        original_edges
        &
        synthetic_edges
    )

    union = (
        original_edges
        |
        synthetic_edges
    )

    precision = (
        len(
            intersection
        )
        /
        len(
            synthetic_edges
        )
        if synthetic_edges
        else 0.0
    )

    recall = (
        len(
            intersection
        )
        /
        len(
            original_edges
        )
        if original_edges
        else 0.0
    )

    f1 = (
        2.0
        * precision
        * recall
        /
        (
            precision
            +
            recall
        )
        if (
            precision
            +
            recall
        )
        > 0
        else 0.0
    )

    jaccard = (
        len(
            intersection
        )
        /
        len(
            union
        )
        if union
        else 1.0
    )

    return {
        "original_client_edges": int(
            len(
                original_edges
            )
        ),

        "synthetic_client_edges": int(
            len(
                synthetic_edges
            )
        ),

        "correct_client_edges": int(
            len(
                intersection
            )
        ),

        "precision": float(
            precision
        ),

        "recall": float(
            recall
        ),

        "f1": float(
            f1
        ),

        "jaccard": float(
            jaccard
        ),
    }


def calculate_graph_statistics(
    graph,
    seed=DEFAULT_SEED,
):


    H = to_simple_undirected(
        graph
    )

    node_counts = count_node_categories(
        graph
    )

    edge_counts = count_edge_categories(
        graph
    )

    degrees = all_degree_values(
        graph
    )

    unique_degrees = unique_degree_values(
        graph
    )

    client_nodes = get_nodes_by_role(
        graph,
        "client",
    )

    content_nodes = get_nodes_by_role(
        graph,
        "content",
    )

    client_unique_degrees = (
        degree_values_for_nodes(
            graph,
            client_nodes,
            unique=True,
        )
    )

    content_unique_degrees = (
        degree_values_for_nodes(
            graph,
            content_nodes,
            unique=True,
        )
    )

    client_graph = client_projection(
        graph
    )

    connectivity = (
        connectivity_statistics(
            graph
        )
    )

    client_connectivity = (
        connectivity_statistics(
            client_graph
        )
    )

    return {
        "nodes": int(
            graph.number_of_nodes()
        ),

        "edges": int(
            graph.number_of_edges()
        ),

        "simple_edges": int(
            H.number_of_edges()
        ),

        "density": float(
            nx.density(
                H
            )
            if H.number_of_nodes()
            > 1
            else 0.0
        ),

        "node_counts": node_counts,

        "edge_counts": edge_counts,

        "degree": numeric_summary(
            degrees
        ),

        "unique_degree": (
            numeric_summary(
                unique_degrees
            )
        ),

        "client_unique_degree": (
            numeric_summary(
                client_unique_degrees
            )
        ),

        "content_unique_degree": (
            numeric_summary(
                content_unique_degrees
            )
        ),

        "average_clustering": float(
            average_clustering_safe(
                graph,
                seed=seed,
            )
        ),

        "client_average_clustering": float(
            average_clustering_safe(
                client_graph,
                seed=seed,
            )
        ),

        "degree_assortativity": float(
            degree_assortativity_safe(
                graph
            )
        ),

        "client_degree_assortativity": float(
            degree_assortativity_safe(
                client_graph
            )
        ),

        "modularity": float(
            calculate_modularity_safe(
                graph,
                seed=seed,
            )
        ),

        "client_modularity": float(
            calculate_modularity_safe(
                client_graph,
                seed=seed,
            )
        ),

        "average_shortest_path_lcc": float(
            approximate_average_shortest_path_length(
                graph,
                seed=seed,
            )
        ),

        "client_average_shortest_path_lcc": float(
            approximate_average_shortest_path_length(
                client_graph,
                seed=seed,
            )
        ),

        "connectivity": connectivity,

        "client_connectivity": (
            client_connectivity
        ),

        "role_metapaths": dict(
            count_length_two_metapaths(
                graph,
                label_mode="role",
            )
        ),

        "type_metapaths": dict(
            count_length_two_metapaths(
                graph,
                label_mode="canonical_type",
            )
        ),

        "role_mixing": dict(
            mixing_matrix_counts(
                graph,
                label_mode="role",
            )
        ),

        "canonical_type_mixing": dict(
            mixing_matrix_counts(
                graph,
                label_mode="canonical_type",
            )
        ),
    }


def build_degree_distribution_comparison(
    original_graph,
    synthetic_graph,
):


    rows = []


    comparison = compare_numeric_distributions(
        unique_degree_values(
            original_graph
        ),
        unique_degree_values(
            synthetic_graph
        ),
    )

    rows.append({
        "level": "global",

        "category": "all_nodes",

        "degree_definition": (
            "unique_neighbours"
        ),

        **comparison,
    })


    roles = sorted(
        set(
            node_role(
                original_graph,
                node,
            )
            for node
            in original_graph.nodes()
        )
        |
        set(
            node_role(
                synthetic_graph,
                node,
            )
            for node
            in synthetic_graph.nodes()
        )
    )

    for role in roles:

        original_values = (
            degree_values_for_nodes(
                original_graph,

                get_nodes_by_role(
                    original_graph,
                    role,
                ),

                unique=True,
            )
        )

        synthetic_values = (
            degree_values_for_nodes(
                synthetic_graph,

                get_nodes_by_role(
                    synthetic_graph,
                    role,
                ),

                unique=True,
            )
        )

        comparison = (
            compare_numeric_distributions(
                original_values,
                synthetic_values,
            )
        )

        rows.append({
            "level": "role",

            "category": role,

            "degree_definition": (
                "unique_neighbours"
            ),

            **comparison,
        })


    canonical_types = sorted(
        set(
            canonical_node_type(
                original_graph,
                node,
            )
            for node
            in original_graph.nodes()
        )
        |
        set(
            canonical_node_type(
                synthetic_graph,
                node,
            )
            for node
            in synthetic_graph.nodes()
        )
    )

    for category in canonical_types:

        original_values = (
            degree_values_for_nodes(
                original_graph,

                get_nodes_by_canonical_type(
                    original_graph,
                    category,
                ),

                unique=True,
            )
        )

        synthetic_values = (
            degree_values_for_nodes(
                synthetic_graph,

                get_nodes_by_canonical_type(
                    synthetic_graph,
                    category,
                ),

                unique=True,
            )
        )

        comparison = (
            compare_numeric_distributions(
                original_values,
                synthetic_values,
            )
        )

        rows.append({
            "level": "canonical_type",

            "category": category,

            "degree_definition": (
                "unique_neighbours"
            ),

            **comparison,
        })


    relation_classes = [
        "client_client",
        "client_content",
        "content_content",
        "other",
    ]

    for relation_class in relation_classes:

        original_values = (
            canonical_relation_degree_values(
                original_graph,
                relation_class,
            )
        )

        synthetic_values = (
            canonical_relation_degree_values(
                synthetic_graph,
                relation_class,
            )
        )

        comparison = (
            compare_numeric_distributions(
                original_values,
                synthetic_values,
            )
        )

        rows.append({
            "level": "canonical_relation",

            "category": relation_class,

            "degree_definition": (
                "incident_edge_count"
            ),

            **comparison,
        })

    return pd.DataFrame(
        rows
    )


def build_category_comparison_table(
    original_graph,
    synthetic_graph,
):


    original_nodes = count_node_categories(
        original_graph
    )

    synthetic_nodes = count_node_categories(
        synthetic_graph
    )

    original_edges = count_edge_categories(
        original_graph
    )

    synthetic_edges = count_edge_categories(
        synthetic_graph
    )

    groups = {
        "node_role": (
            original_nodes[
                "role_counts"
            ],
            synthetic_nodes[
                "role_counts"
            ],
        ),

        "node_exact_type": (
            original_nodes[
                "exact_type_counts"
            ],
            synthetic_nodes[
                "exact_type_counts"
            ],
        ),

        "node_canonical_type": (
            original_nodes[
                "canonical_type_counts"
            ],
            synthetic_nodes[
                "canonical_type_counts"
            ],
        ),

        "edge_exact_relation": (
            original_edges[
                "exact_relation_counts"
            ],
            synthetic_edges[
                "exact_relation_counts"
            ],
        ),

        "edge_role_pair": (
            original_edges[
                "role_pair_counts"
            ],
            synthetic_edges[
                "role_pair_counts"
            ],
        ),

        "edge_canonical_type_pair": (
            original_edges[
                "canonical_type_pair_counts"
            ],
            synthetic_edges[
                "canonical_type_pair_counts"
            ],
        ),

        "edge_canonical_relation": (
            original_edges[
                "canonical_relation_counts"
            ],
            synthetic_edges[
                "canonical_relation_counts"
            ],
        ),
    }

    rows = []

    summary = {}

    for level, (
        original_counts,
        synthetic_counts,
    ) in groups.items():

        comparison = (
            compare_count_dictionaries(
                original_counts,
                synthetic_counts,
            )
        )

        summary[
            level
        ] = {
            "proportion_l1_error": (
                comparison[
                    "proportion_l1_error"
                ]
            ),

            "total_variation_distance": (
                comparison[
                    "total_variation_distance"
                ]
            ),
        }

        for item in comparison[
            "rows"
        ]:

            rows.append({
                "level": level,

                **item,
            })

    return (
        pd.DataFrame(
            rows
        ),
        summary,
    )


def build_metapath_comparison_table(
    original_graph,
    synthetic_graph,
):


    rows = []

    summaries = {}

    for mode in [
        "role",
        "canonical_type",
    ]:

        original_counts = (
            count_length_two_metapaths(
                original_graph,
                label_mode=mode,
            )
        )

        synthetic_counts = (
            count_length_two_metapaths(
                synthetic_graph,
                label_mode=mode,
            )
        )

        comparison = (
            compare_count_dictionaries(
                original_counts,
                synthetic_counts,
            )
        )

        summaries[
            mode
        ] = {
            "proportion_l1_error": (
                comparison[
                    "proportion_l1_error"
                ]
            ),

            "total_variation_distance": (
                comparison[
                    "total_variation_distance"
                ]
            ),
        }

        for item in comparison[
            "rows"
        ]:

            rows.append({
                "metapath_mode": mode,

                "metapath": item[
                    "category"
                ],

                "original_count": item[
                    "original_count"
                ],

                "synthetic_count": item[
                    "synthetic_count"
                ],

                "count_absolute_error": item[
                    "count_absolute_error"
                ],

                "count_relative_error": item[
                    "count_relative_error"
                ],

                "original_proportion": item[
                    "original_proportion"
                ],

                "synthetic_proportion": item[
                    "synthetic_proportion"
                ],

                "proportion_absolute_error": item[
                    "proportion_absolute_error"
                ],
            })

    return (
        pd.DataFrame(
            rows
        ),
        summaries,
    )


def calculate_structural_mmd(
    original_graph,
    synthetic_graph,
    seed=DEFAULT_SEED,
):


    original_features = (
        node_structural_features(
            original_graph,
            max_nodes=(
                DEFAULT_MAX_MMD_NODES
            ),
            seed=seed,
        )
    )

    synthetic_features = (
        node_structural_features(
            synthetic_graph,
            max_nodes=(
                DEFAULT_MAX_MMD_NODES
            ),
            seed=seed + 1,
        )
    )

    return compute_mmd(
        original_features,
        synthetic_features,
        gamma=None,
        seed=seed,
    )


def derive_oracle_objective(
    original_stats,
    synthetic_stats,
    degree_df,
    category_summary,
    metapath_summary,
    structural_mmd,
):


    degree_rows = degree_df[
        degree_df[
            "level"
        ]
        .isin(
            [
                "role",
            ]
        )
    ]

    degree_values = pd.to_numeric(
        degree_rows[
            "normalized_wasserstein"
        ],
        errors="coerce",
    )

    degree_values = degree_values[
        np.isfinite(
            degree_values
        )
    ]

    if len(
        degree_values
    ):

        degree_loss = float(
            degree_values.mean()
        )

    else:

        degree_loss = 1.0


    relation_loss = float(
        category_summary[
            "edge_canonical_relation"
        ][
            "total_variation_distance"
        ]
    )


    original_clustering = float(
        original_stats[
            "average_clustering"
        ]
    )

    synthetic_clustering = float(
        synthetic_stats[
            "average_clustering"
        ]
    )

    clustering_loss = (
        safe_relative_error(
            synthetic_clustering,
            original_clustering,
        )
    )


    modularity_loss = float(
        abs(
            synthetic_stats[
                "modularity"
            ]
            -
            original_stats[
                "modularity"
            ]
        )
    )


    metapath_loss = float(
        metapath_summary[
            "role"
        ][
            "total_variation_distance"
        ]
    )


    if (
        structural_mmd is None
        or structural_mmd.get(
            "mmd_squared"
        )
        is None
    ):

        mmd_loss = 1.0

    else:

        mmd_loss = float(
            structural_mmd[
                "mmd_squared"
            ]
        )

    losses = {
        "degree": float(
            degree_loss
        ),

        "relation": float(
            relation_loss
        ),

        "clustering": float(
            clustering_loss
        ),

        "modularity": float(
            modularity_loss
        ),

        "metapath": float(
            metapath_loss
        ),

        "structural_mmd": float(
            mmd_loss
        ),
    }

    total_loss = 0.0

    for component, weight in (
        ORACLE_OBJECTIVE_WEIGHTS.items()
    ):

        total_loss += (
            float(
                weight
            )
            *
            losses[
                component
            ]
        )

    return {
        "oracle_objective": float(
            -total_loss
        ),

        "oracle_total_loss": float(
            total_loss
        ),

        "oracle_degree_loss": float(
            degree_loss
        ),

        "oracle_relation_loss": float(
            relation_loss
        ),

        "oracle_clustering_loss": float(
            clustering_loss
        ),

        "oracle_modularity_loss": float(
            modularity_loss
        ),

        "oracle_metapath_loss": float(
            metapath_loss
        ),

        "oracle_structural_mmd_loss": float(
            mmd_loss
        ),

        "weights": dict(
            ORACLE_OBJECTIVE_WEIGHTS
        ),
    }


def compute_oracle_metrics(
    original_graph,
    synthetic_graph,
    dataset_name="unknown",
    seed=DEFAULT_SEED,
    output_folder=None,
):


    original_stats = (
        calculate_graph_statistics(
            original_graph,
            seed=seed,
        )
    )

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


    scalar_comparison = {
        "node_count_relative_error": (
            safe_relative_error(
                synthetic_graph.number_of_nodes(),
                original_graph.number_of_nodes(),
            )
        ),

        "edge_count_relative_error": (
            safe_relative_error(
                synthetic_graph.number_of_edges(),
                original_graph.number_of_edges(),
            )
        ),

        "density_absolute_error": float(
            abs(
                synthetic_stats[
                    "density"
                ]
                -
                original_stats[
                    "density"
                ]
            )
        ),

        "clustering_absolute_error": float(
            abs(
                synthetic_stats[
                    "average_clustering"
                ]
                -
                original_stats[
                    "average_clustering"
                ]
            )
        ),

        "clustering_relative_error": (
            safe_relative_error(
                synthetic_stats[
                    "average_clustering"
                ],
                original_stats[
                    "average_clustering"
                ],
            )
        ),

        "client_clustering_absolute_error": float(
            abs(
                synthetic_stats[
                    "client_average_clustering"
                ]
                -
                original_stats[
                    "client_average_clustering"
                ]
            )
        ),

        "degree_assortativity_absolute_error": float(
            abs(
                synthetic_stats[
                    "degree_assortativity"
                ]
                -
                original_stats[
                    "degree_assortativity"
                ]
            )
        ),

        "client_degree_assortativity_absolute_error": float(
            abs(
                synthetic_stats[
                    "client_degree_assortativity"
                ]
                -
                original_stats[
                    "client_degree_assortativity"
                ]
            )
        ),

        "modularity_absolute_error": float(
            abs(
                synthetic_stats[
                    "modularity"
                ]
                -
                original_stats[
                    "modularity"
                ]
            )
        ),

        "client_modularity_absolute_error": float(
            abs(
                synthetic_stats[
                    "client_modularity"
                ]
                -
                original_stats[
                    "client_modularity"
                ]
            )
        ),

        "average_path_absolute_error": float(
            abs(
                synthetic_stats[
                    "average_shortest_path_lcc"
                ]
                -
                original_stats[
                    "average_shortest_path_lcc"
                ]
            )
        ),
    }


    oracle_objective = derive_oracle_objective(
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
        "schema_version": (
            STATS_SCHEMA_VERSION
        ),

        "dataset": str(
            dataset_name
        ),

        "seed": int(
            seed
        ),

        "evaluation_mode": (
            "oracle_experiment_only"
        ),

        "original": (
            original_stats
        ),

        "synthetic": (
            synthetic_stats
        ),

        "scalar_comparison": (
            scalar_comparison
        ),

        "category_summary": (
            category_summary
        ),

        "metapath_summary": (
            metapath_summary
        ),

        "structural_mmd": (
            structural_mmd
        ),

        "client_edge_recovery": (
            edge_recovery
        ),

        "oracle_objective": (
            oracle_objective
        ),
    }


    if output_folder is not None:

        os.makedirs(
            output_folder,
            exist_ok=True,
        )

        summary_path = os.path.join(
            output_folder,
            "oracle_metrics.json",
        )

        degree_path = os.path.join(
            output_folder,
            "oracle_degree_distributions.csv",
        )

        category_path = os.path.join(
            output_folder,
            "oracle_category_metrics.csv",
        )

        metapath_path = os.path.join(
            output_folder,
            "oracle_metapath_metrics.csv",
        )

        degree_df.to_csv(
            degree_path,
            index=False,
        )

        category_df.to_csv(
            category_path,
            index=False,
        )

        metapath_df.to_csv(
            metapath_path,
            index=False,
        )

        with open(
            summary_path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                result,
                f,
                indent=4,
                ensure_ascii=False,
            )

    return result


def compute_metrics(
    G1,
    G2,
    epsilon=None,
    dataset_name="unknown",
    output_folder="output/",
    private_run_folder=None,
    seed=DEFAULT_SEED,
):


    if G1 is None:

        original_graph_path = os.path.join(
            output_folder,
            dataset_name,
            (
                f"{dataset_name}"
                "_global_graph.gml.gz"
            ),
        )

        if not os.path.exists(
            original_graph_path
        ):

            raise FileNotFoundError(
                f"Missing original graph: "
                f"{original_graph_path}"
            )

        G1 = nx.read_gml(
            original_graph_path
        )

    if G2 is None:

        if private_run_folder is None:

            raise ValueError(
                "G2 is None and no private_run_folder "
                "was provided."
            )

        synthetic_graph_path = os.path.join(
            private_run_folder,
            "synthesis",
            "synthetic_graph.gml.gz",
        )

        if not os.path.exists(
            synthetic_graph_path
        ):

            raise FileNotFoundError(
                f"Missing synthetic graph: "
                f"{synthetic_graph_path}"
            )

        G2 = nx.read_gml(
            synthetic_graph_path
        )

    if private_run_folder is not None:

        oracle_output_folder = os.path.join(
            private_run_folder,
            "synthesis",
            "oracle",
        )

    else:

        epsilon_token = (
            "unknown"
            if epsilon is None
            else str(
                epsilon
            ).replace(
                ".",
                "p",
            )
        )

        oracle_output_folder = os.path.join(
            output_folder,
            dataset_name,
            "oracle_evaluation",
            f"epsilon_{epsilon_token}",
        )

    result = compute_oracle_metrics(
        original_graph=G1,

        synthetic_graph=G2,

        dataset_name=(
            dataset_name
        ),

        seed=seed,

        output_folder=(
            oracle_output_folder
        ),
    )


    objective = result[
        "oracle_objective"
    ]

    edge_recovery = result[
        "client_edge_recovery"
    ]

    print(
        "\n"
        "========================================"
    )

    print(
        "ORACLE GRAPH EVALUATION"
    )

    print(
        "========================================"
    )

    print(
        f"Dataset: "
        f"{dataset_name}"
    )

    if epsilon is not None:

        print(
            f"epsilon: "
            f"{epsilon}"
        )

    print(
        "\nOracle objective:"
    )

    print(
        f"  "
        f"{objective['oracle_objective']:.6f}"
    )

    print(
        "\nOracle loss components:"
    )

    print(
        "  degree: "
        f"{objective['oracle_degree_loss']:.6f}"
    )

    print(
        "  relation: "
        f"{objective['oracle_relation_loss']:.6f}"
    )

    print(
        "  clustering: "
        f"{objective['oracle_clustering_loss']:.6f}"
    )

    print(
        "  modularity: "
        f"{objective['oracle_modularity_loss']:.6f}"
    )

    print(
        "  meta-path: "
        f"{objective['oracle_metapath_loss']:.6f}"
    )

    print(
        "  structural MMD: "
        f"{objective['oracle_structural_mmd_loss']:.6f}"
    )

    print(
        "\nClient-edge recovery:"
    )

    print(
        f"  Precision: "
        f"{edge_recovery['precision']:.4f}"
    )

    print(
        f"  Recall: "
        f"{edge_recovery['recall']:.4f}"
    )

    print(
        f"  F1: "
        f"{edge_recovery['f1']:.4f}"
    )

    print(
        f"  Jaccard: "
        f"{edge_recovery['jaccard']:.4f}"
    )

    print(
        "\nResults:"
    )

    print(
        f"  {oracle_output_folder}"
    )

    print(
        "========================================\n"
    )

    return result


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Oracle evaluation of an LDP "
            "synthetic graph."
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

    args = parser.parse_args()

    compute_metrics(
        G1=None,

        G2=None,

        epsilon=None,

        dataset_name=(
            args.dataset
        ),

        output_folder=(
            args.output_folder
        ),

        private_run_folder=(
            args.private_run_folder
        ),

        seed=args.seed,
    )
