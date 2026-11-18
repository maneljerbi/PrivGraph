import json
import os
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from tqdm import tqdm


SIGNATURE_SCHEMA_VERSION = "1.0"


RAW_SIGNATURE_DIM = 12


HKS_TIMES = np.array(
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


DENSE_EIGEN_THRESHOLD = 500


MAX_HKS_EIGENPAIRS = 64


SPARSE_EIGEN_BUFFER = 8


EIGENVALUE_ATOL = 1e-8
EIGENVALUE_RTOL = 1e-6
NORM_EPS = 1e-12


EGO_FOLDER_NAME = "ego_graphs_local"


def l2_normalize(vector, eps=NORM_EPS):

    vector = np.asarray(
        vector,
        dtype=float,
    )

    norm = np.linalg.norm(
        vector
    )

    if norm <= eps:
        return np.zeros_like(
            vector,
            dtype=float,
        )

    return vector / norm


def pad_or_truncate(vector, dimension):

    vector = np.asarray(
        vector,
        dtype=float,
    )

    result = np.zeros(
        dimension,
        dtype=float,
    )

    usable = min(
        len(vector),
        dimension,
    )

    if usable > 0:
        result[:usable] = vector[:usable]

    return result


def to_simple_undirected_weighted(graph):

    H = nx.Graph()

    for node, data in graph.nodes(
        data=True
    ):
        H.add_node(
            node,
            **dict(data),
        )

    if graph.is_multigraph():

        for u, v, _, data in graph.edges(
            keys=True,
            data=True,
        ):

            if u == v:
                continue

            if H.has_edge(
                u,
                v,
            ):
                H[u][v]["weight"] += 1.0

            else:
                H.add_edge(
                    u,
                    v,
                    weight=1.0,
                )

    else:

        for u, v, data in graph.edges(
            data=True
        ):

            if u == v:
                continue

            if H.has_edge(
                u,
                v,
            ):
                H[u][v]["weight"] += 1.0

            else:
                H.add_edge(
                    u,
                    v,
                    weight=1.0,
                )

    return H


def get_unique_neighbours(graph, node):

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


def build_role_projection(
    ego_graph,
    main_node,
    neighbour_role,
):

    if main_node not in ego_graph:
        raise KeyError(
            f"Main node {main_node!r} is missing "
            "from its ego graph."
        )

    neighbours = get_unique_neighbours(
        ego_graph,
        main_node,
    )

    selected_neighbours = {
        node
        for node in neighbours
        if ego_graph.nodes[
            node
        ].get(
            "role"
        ) == neighbour_role
    }

    selected_nodes = {
        main_node
    } | selected_neighbours

    subgraph = ego_graph.subgraph(
        selected_nodes
    ).copy()

    return to_simple_undirected_weighted(
        subgraph
    )


def ordered_nodelist(graph, main_node):

    if main_node not in graph:
        raise KeyError(
            f"{main_node!r} is not present in graph."
        )

    others = sorted(
        [
            node
            for node in graph.nodes()
            if node != main_node
        ],
        key=str,
    )

    return [
        main_node
    ] + others


def normalized_laplacian_dense(
    graph,
    nodelist,
):

    A = nx.to_numpy_array(
        graph,
        nodelist=nodelist,
        weight="weight",
        dtype=float,
    )

    degrees = np.sum(
        A,
        axis=1,
    )

    inv_sqrt = np.zeros_like(
        degrees,
        dtype=float,
    )

    nonzero = degrees > 0

    inv_sqrt[
        nonzero
    ] = 1.0 / np.sqrt(
        degrees[
            nonzero
        ]
    )

    D_inv_sqrt = np.diag(
        inv_sqrt
    )

    identity = np.eye(
        len(
            nodelist
        ),
        dtype=float,
    )

    L = (
        identity
        -
        D_inv_sqrt
        @ A
        @ D_inv_sqrt
    )


    isolated = np.where(
        ~nonzero
    )[0]

    for idx in isolated:
        L[idx, idx] = 0.0


    L = 0.5 * (
        L + L.T
    )

    return L


def compute_spectral_decomposition(
    graph,
    main_node,
    requested_eigenpairs,
    dense_threshold=DENSE_EIGEN_THRESHOLD,
    sparse_buffer=SPARSE_EIGEN_BUFFER,
):

    graph = to_simple_undirected_weighted(
        graph
    )

    nodelist = ordered_nodelist(
        graph,
        main_node,
    )

    n = len(
        nodelist
    )

    if n == 0:
        raise ValueError(
            "Cannot compute a spectrum for an empty graph."
        )


    if n == 1:

        return {
            "eigenvalues": np.array(
                [0.0],
                dtype=float,
            ),

            "eigenvectors": np.array(
                [[1.0]],
                dtype=float,
            ),

            "nodelist": nodelist,

            "main_index": 0,

            "full_spectrum": True,

            "spectrum_truncated": False,
        }


    if n <= dense_threshold:

        L = normalized_laplacian_dense(
            graph,
            nodelist,
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            L
        )

        order = np.argsort(
            eigenvalues
        )

        eigenvalues = np.asarray(
            eigenvalues[
                order
            ],
            dtype=float,
        )

        eigenvectors = np.asarray(
            eigenvectors[
                :,
                order
            ],
            dtype=float,
        )

        return {
            "eigenvalues": eigenvalues,

            "eigenvectors": eigenvectors,

            "nodelist": nodelist,

            "main_index": 0,

            "full_spectrum": True,

            "spectrum_truncated": False,
        }


    L = nx.normalized_laplacian_matrix(
        graph,
        nodelist=nodelist,
        weight="weight",
    ).astype(
        float
    )

    requested_eigenpairs = max(
        1,
        int(
            requested_eigenpairs
        ),
    )

    k = min(
        n - 1,
        requested_eigenpairs
        + sparse_buffer,
    )

    if k <= 0:

        return {
            "eigenvalues": np.array(
                [0.0],
                dtype=float,
            ),

            "eigenvectors": np.ones(
                (
                    n,
                    1,
                ),
                dtype=float,
            )
            / np.sqrt(
                n
            ),

            "nodelist": nodelist,

            "main_index": 0,

            "full_spectrum": False,

            "spectrum_truncated": True,
        }

    try:

        eigenvalues, eigenvectors = eigsh(
            L,
            k=k,
            which="SM",
            tol=1e-6,
        )

    except Exception:


        L_dense = L.toarray()

        eigenvalues, eigenvectors = np.linalg.eigh(
            L_dense
        )

        order = np.argsort(
            eigenvalues
        )

        eigenvalues = eigenvalues[
            order
        ]

        eigenvectors = eigenvectors[
            :,
            order
        ]

        return {
            "eigenvalues": np.asarray(
                eigenvalues,
                dtype=float,
            ),

            "eigenvectors": np.asarray(
                eigenvectors,
                dtype=float,
            ),

            "nodelist": nodelist,

            "main_index": 0,

            "full_spectrum": True,

            "spectrum_truncated": False,
        }

    order = np.argsort(
        eigenvalues
    )

    eigenvalues = np.asarray(
        eigenvalues[
            order
        ],
        dtype=float,
    )

    eigenvectors = np.asarray(
        eigenvectors[
            :,
            order
        ],
        dtype=float,
    )

    return {
        "eigenvalues": eigenvalues,

        "eigenvectors": eigenvectors,

        "nodelist": nodelist,

        "main_index": 0,

        "full_spectrum": False,

        "spectrum_truncated": True,
    }


def eigenvalues_close(
    a,
    b,
    atol=EIGENVALUE_ATOL,
    rtol=EIGENVALUE_RTOL,
):

    return bool(
        np.isclose(
            a,
            b,
            atol=atol,
            rtol=rtol,
        )
    )


def repeated_eigenspaces(
    eigenvalues,
    atol=EIGENVALUE_ATOL,
    rtol=EIGENVALUE_RTOL,
):

    eigenvalues = np.asarray(
        eigenvalues,
        dtype=float,
    )

    groups = []

    if len(
        eigenvalues
    ) < 2:
        return groups

    start = 0

    for i in range(
        1,
        len(
            eigenvalues
        ),
    ):

        if eigenvalues_close(
            eigenvalues[
                i
            ],
            eigenvalues[
                start
            ],
            atol=atol,
            rtol=rtol,
        ):
            continue

        multiplicity = (
            i - start
        )

        if multiplicity > 1:

            groups.append({
                "start": int(
                    start
                ),

                "end": int(
                    i - 1
                ),

                "multiplicity": int(
                    multiplicity
                ),

                "eigenvalue": float(
                    np.mean(
                        eigenvalues[
                            start:i
                        ]
                    )
                ),
            })

        start = i

    multiplicity = (
        len(
            eigenvalues
        )
        - start
    )

    if multiplicity > 1:

        groups.append({
            "start": int(
                start
            ),

            "end": int(
                len(
                    eigenvalues
                )
                - 1
            ),

            "multiplicity": int(
                multiplicity
            ),

            "eigenvalue": float(
                np.mean(
                    eigenvalues[
                        start:
                    ]
                )
            ),
        })

    return groups


def extend_cutoff_for_repeated_eigenspace(
    eigenvalues,
    cutoff,
):

    eigenvalues = np.asarray(
        eigenvalues,
        dtype=float,
    )

    if len(
        eigenvalues
    ) == 0:
        return 0

    cutoff = min(
        max(
            int(
                cutoff
            ),
            1,
        ),
        len(
            eigenvalues
        ),
    )

    if cutoff >= len(
        eigenvalues
    ):
        return cutoff

    boundary_value = eigenvalues[
        cutoff - 1
    ]

    while (
        cutoff
        < len(
            eigenvalues
        )
        and eigenvalues_close(
            eigenvalues[
                cutoff
            ],
            boundary_value,
        )
    ):
        cutoff += 1

    return cutoff


def select_nontrivial_eigen_indices(
    eigenvalues,
    dimension,
):

    eigenvalues = np.asarray(
        eigenvalues,
        dtype=float,
    )

    indices = [
        idx
        for idx, value
        in enumerate(
            eigenvalues
        )
        if value > EIGENVALUE_ATOL
    ]

    return indices[
        :dimension
    ]


def compute_raw_signature_from_decomposition(
    decomposition,
    dimension=RAW_SIGNATURE_DIM,
):

    eigenvalues = decomposition[
        "eigenvalues"
    ]

    eigenvectors = decomposition[
        "eigenvectors"
    ]

    main_index = decomposition[
        "main_index"
    ]

    indices = select_nontrivial_eigen_indices(
        eigenvalues,
        dimension,
    )

    if not indices:

        return np.zeros(
            dimension,
            dtype=float,
        )

    raw = eigenvectors[
        main_index,
        indices,
    ]

    raw = pad_or_truncate(
        raw,
        dimension,
    )


    return l2_normalize(
        raw
    )


def compute_abs_signature_from_raw(
    raw_signature
):

    absolute = np.abs(
        np.asarray(
            raw_signature,
            dtype=float,
        )
    )

    return l2_normalize(
        absolute
    )


def compute_hks_from_decomposition(
    decomposition,
    times=HKS_TIMES,
    max_eigenpairs=MAX_HKS_EIGENPAIRS,
):

    eigenvalues = np.asarray(
        decomposition[
            "eigenvalues"
        ],
        dtype=float,
    )

    eigenvectors = np.asarray(
        decomposition[
            "eigenvectors"
        ],
        dtype=float,
    )

    main_index = int(
        decomposition[
            "main_index"
        ]
    )

    times = np.asarray(
        times,
        dtype=float,
    )

    if len(
        eigenvalues
    ) == 0:

        return np.zeros(
            len(
                times
            ),
            dtype=float,
        ), 0

    requested = min(
        int(
            max_eigenpairs
        ),
        len(
            eigenvalues
        ),
    )

    selected_count = (
        extend_cutoff_for_repeated_eigenspace(
            eigenvalues,
            requested,
        )
    )

    selected_eigenvalues = eigenvalues[
        :selected_count
    ]

    main_coordinates_squared = (
        eigenvectors[
            main_index,
            :selected_count,
        ]
        ** 2
    )

    values = []

    for t in times:

        heat_weights = np.exp(
            -float(
                t
            )
            * selected_eigenvalues
        )

        value = np.sum(
            heat_weights
            * main_coordinates_squared
        )

        values.append(
            float(
                value
            )
        )

    return np.asarray(
        values,
        dtype=float,
    ), int(
        selected_count
    )


def compute_role_hks_block(
    ego_graph,
    main_node,
    neighbour_role,
    times=HKS_TIMES,
):

    projection = build_role_projection(
        ego_graph,
        main_node,
        neighbour_role,
    )

    neighbours = get_unique_neighbours(
        projection,
        main_node,
    )

    if (
        projection.number_of_nodes()
        <= 1
        or len(
            neighbours
        ) == 0
        or projection.number_of_edges()
        == 0
    ):

        diagnostics = {
            "nodes": int(
                projection.number_of_nodes()
            ),

            "edges": int(
                projection.number_of_edges()
            ),

            "neighbours": int(
                len(
                    neighbours
                )
            ),

            "eigenpairs_used": 0,

            "spectrum_truncated": False,

            "repeated_eigenspaces": 0,

            "largest_multiplicity": 1,

            "smallest_eigenvalues": [],
        }

        return (
            np.zeros(
                len(
                    times
                ),
                dtype=float,
            ),
            diagnostics,
        )

    decomposition = compute_spectral_decomposition(
        projection,
        main_node,
        requested_eigenpairs=MAX_HKS_EIGENPAIRS,
    )

    block, used = compute_hks_from_decomposition(
        decomposition,
        times=times,
        max_eigenpairs=MAX_HKS_EIGENPAIRS,
    )

    groups = repeated_eigenspaces(
        decomposition[
            "eigenvalues"
        ]
    )

    largest_multiplicity = max(
        [
            group[
                "multiplicity"
            ]
            for group in groups
        ],
        default=1,
    )

    diagnostics = {
        "nodes": int(
            projection.number_of_nodes()
        ),

        "edges": int(
            projection.number_of_edges()
        ),

        "neighbours": int(
            len(
                neighbours
            )
        ),

        "eigenpairs_used": int(
            used
        ),

        "spectrum_truncated": bool(
            decomposition[
                "spectrum_truncated"
            ]
        ),

        "repeated_eigenspaces": int(
            len(
                groups
            )
        ),

        "largest_multiplicity": int(
            largest_multiplicity
        ),

        "smallest_eigenvalues": [
            float(
                value
            )
            for value
            in decomposition[
                "eigenvalues"
            ][
                :min(
                    20,
                    len(
                        decomposition[
                            "eigenvalues"
                        ]
                    ),
                )
            ]
        ],
    }


    block = l2_normalize(
        block
    )

    return (
        block,
        diagnostics,
    )


def compute_hks_signature(
    ego_graph,
    main_node,
    times=HKS_TIMES,
):

    client_block, client_diag = (
        compute_role_hks_block(
            ego_graph=ego_graph,
            main_node=main_node,
            neighbour_role="client",
            times=times,
        )
    )

    content_block, content_diag = (
        compute_role_hks_block(
            ego_graph=ego_graph,
            main_node=main_node,
            neighbour_role="content",
            times=times,
        )
    )

    signature = np.concatenate(
        [
            client_block,
            content_block,
        ]
    )

    signature = l2_normalize(
        signature
    )

    diagnostics = {
        "client_block": client_diag,
        "content_block": content_diag,
    }

    return (
        signature,
        diagnostics,
    )


def compute_all_local_signatures(
    ego_graph,
    main_node,
    raw_dimension=RAW_SIGNATURE_DIM,
    hks_times=HKS_TIMES,
):

    full_projection = (
        to_simple_undirected_weighted(
            ego_graph
        )
    )

    if (
        full_projection.number_of_nodes()
        <= 1
        or full_projection.number_of_edges()
        == 0
    ):

        raw = np.zeros(
            raw_dimension,
            dtype=float,
        )

        absolute = np.zeros(
            raw_dimension,
            dtype=float,
        )

        full_groups = []

        full_diagnostics = {
            "full_nodes": int(
                full_projection.number_of_nodes()
            ),

            "full_edges": int(
                full_projection.number_of_edges()
            ),

            "full_spectrum_truncated": False,

            "full_repeated_eigenspaces": 0,

            "full_largest_multiplicity": 1,

            "full_smallest_eigenvalues": [],
        }

    else:

        full_decomposition = (
            compute_spectral_decomposition(
                full_projection,
                main_node,
                requested_eigenpairs=raw_dimension,
            )
        )

        raw = (
            compute_raw_signature_from_decomposition(
                full_decomposition,
                dimension=raw_dimension,
            )
        )

        absolute = (
            compute_abs_signature_from_raw(
                raw
            )
        )

        full_groups = repeated_eigenspaces(
            full_decomposition[
                "eigenvalues"
            ]
        )

        largest_multiplicity = max(
            [
                group[
                    "multiplicity"
                ]
                for group in full_groups
            ],
            default=1,
        )

        full_diagnostics = {
            "full_nodes": int(
                full_projection.number_of_nodes()
            ),

            "full_edges": int(
                full_projection.number_of_edges()
            ),

            "full_spectrum_truncated": bool(
                full_decomposition[
                    "spectrum_truncated"
                ]
            ),

            "full_repeated_eigenspaces": int(
                len(
                    full_groups
                )
            ),

            "full_largest_multiplicity": int(
                largest_multiplicity
            ),

            "full_smallest_eigenvalues": [
                float(
                    value
                )
                for value
                in full_decomposition[
                    "eigenvalues"
                ][
                    :min(
                        20,
                        len(
                            full_decomposition[
                                "eigenvalues"
                            ]
                        ),
                    )
                ]
            ],
        }

    hks, hks_diagnostics = (
        compute_hks_signature(
            ego_graph,
            main_node,
            times=hks_times,
        )
    )

    diagnostics = {
        **full_diagnostics,

        "client_block": (
            hks_diagnostics[
                "client_block"
            ]
        ),

        "content_block": (
            hks_diagnostics[
                "content_block"
            ]
        ),
    }

    return {
        "raw": raw,
        "abs": absolute,
        "hks": hks,
        "diagnostics": diagnostics,
    }


def compute_local_signatures(
    dataset_name="enron",
    output_folder="output/",
    force_rebuild=False,
    max_clients=None,
):

    dataset_folder = os.path.join(
        output_folder,
        dataset_name,
    )

    ego_folder = os.path.join(
        dataset_folder,
        EGO_FOLDER_NAME,
    )

    ego_index_path = os.path.join(
        dataset_folder,
        f"{dataset_name}_ego_graph_index.csv.gz",
    )

    signatures_path = os.path.join(
        dataset_folder,
        f"{dataset_name}_local_signatures.npz",
    )

    stats_path = os.path.join(
        dataset_folder,
        f"{dataset_name}_local_signature_stats.csv.gz",
    )

    metadata_path = os.path.join(
        dataset_folder,
        f"{dataset_name}_local_signature_metadata.json",
    )


    if (
        not force_rebuild
        and os.path.exists(
            signatures_path
        )
        and os.path.exists(
            stats_path
        )
        and os.path.exists(
            metadata_path
        )
    ):

        try:

            with open(
                metadata_path,
                "r",
                encoding="utf-8",
            ) as f:

                metadata = json.load(
                    f
                )

            if (
                metadata.get(
                    "schema_version"
                )
                == SIGNATURE_SCHEMA_VERSION
            ):

                stored = np.load(
                    signatures_path,
                    allow_pickle=False,
                )

                stats_df = pd.read_csv(
                    stats_path,
                    compression="gzip",
                )

                return {
                    "client_ids": stored[
                        "client_ids"
                    ],

                    "raw": stored[
                        "raw"
                    ],

                    "abs": stored[
                        "abs"
                    ],

                    "hks": stored[
                        "hks"
                    ],

                    "stats": stats_df,
                }

        except Exception:
            pass


    if not os.path.isdir(
        ego_folder
    ):
        raise FileNotFoundError(
            "Missing revised ego-graph folder: "
            f"{ego_folder}. "
            "Run create_ego_graphs() first."
        )

    if not os.path.exists(
        ego_index_path
    ):
        raise FileNotFoundError(
            "Missing ego index: "
            f"{ego_index_path}. "
            "Run create_ego_graphs() first."
        )

    ego_index = pd.read_csv(
        ego_index_path,
        compression="gzip",
        dtype={
            "client_id": str,
            "ego_filename": str,
        },
    )

    if max_clients is not None:

        if max_clients <= 0:
            raise ValueError(
                "max_clients must be positive."
            )

        ego_index = ego_index.iloc[
            :max_clients
        ].copy()


    client_ids = []

    raw_signatures = []

    abs_signatures = []

    hks_signatures = []

    stats_rows = []

    failures = []

    for _, index_row in tqdm(
        ego_index.iterrows(),
        total=len(
            ego_index
        ),
        desc=(
            f"Computing local signatures "
            f"({dataset_name})"
        ),
    ):

        client_id = str(
            index_row[
                "client_id"
            ]
        )

        ego_filename = str(
            index_row[
                "ego_filename"
            ]
        )

        ego_path = os.path.join(
            ego_folder,
            ego_filename,
        )

        try:

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


                matching = [
                    node
                    for node
                    in ego_graph.nodes()
                    if str(
                        node
                    ) == main_node
                ]

                if not matching:
                    raise KeyError(
                        "Main node is not present in ego graph."
                    )

                main_node = matching[
                    0
                ]

            result = (
                compute_all_local_signatures(
                    ego_graph=ego_graph,
                    main_node=main_node,
                    raw_dimension=RAW_SIGNATURE_DIM,
                    hks_times=HKS_TIMES,
                )
            )

            raw = np.asarray(
                result[
                    "raw"
                ],
                dtype=float,
            )

            absolute = np.asarray(
                result[
                    "abs"
                ],
                dtype=float,
            )

            hks = np.asarray(
                result[
                    "hks"
                ],
                dtype=float,
            )

            diagnostics = result[
                "diagnostics"
            ]

            client_diag = diagnostics[
                "client_block"
            ]

            content_diag = diagnostics[
                "content_block"
            ]

            client_ids.append(
                client_id
            )

            raw_signatures.append(
                raw
            )

            abs_signatures.append(
                absolute
            )

            hks_signatures.append(
                hks
            )

            stats_rows.append({
                "dataset": dataset_name,

                "client_id": client_id,

                "ego_filename": ego_filename,

                "ego_nodes": int(
                    ego_graph.number_of_nodes()
                ),

                "ego_edges": int(
                    ego_graph.number_of_edges()
                ),

                "raw_dimension": int(
                    len(
                        raw
                    )
                ),

                "abs_dimension": int(
                    len(
                        absolute
                    )
                ),

                "hks_dimension": int(
                    len(
                        hks
                    )
                ),

                "raw_norm": float(
                    np.linalg.norm(
                        raw
                    )
                ),

                "abs_norm": float(
                    np.linalg.norm(
                        absolute
                    )
                ),

                "hks_norm": float(
                    np.linalg.norm(
                        hks
                    )
                ),

                "raw_zero": bool(
                    np.linalg.norm(
                        raw
                    )
                    <= NORM_EPS
                ),

                "hks_zero": bool(
                    np.linalg.norm(
                        hks
                    )
                    <= NORM_EPS
                ),

                "full_spectrum_truncated": bool(
                    diagnostics[
                        "full_spectrum_truncated"
                    ]
                ),

                "full_repeated_eigenspaces": int(
                    diagnostics[
                        "full_repeated_eigenspaces"
                    ]
                ),

                "full_largest_multiplicity": int(
                    diagnostics[
                        "full_largest_multiplicity"
                    ]
                ),

                "full_smallest_eigenvalues_json": (
                    json.dumps(
                        diagnostics[
                            "full_smallest_eigenvalues"
                        ]
                    )
                ),

                "cc_nodes": int(
                    client_diag[
                        "nodes"
                    ]
                ),

                "cc_edges": int(
                    client_diag[
                        "edges"
                    ]
                ),

                "cc_neighbours": int(
                    client_diag[
                        "neighbours"
                    ]
                ),

                "cc_eigenpairs_used": int(
                    client_diag[
                        "eigenpairs_used"
                    ]
                ),

                "cc_spectrum_truncated": bool(
                    client_diag[
                        "spectrum_truncated"
                    ]
                ),

                "cc_repeated_eigenspaces": int(
                    client_diag[
                        "repeated_eigenspaces"
                    ]
                ),

                "cc_largest_multiplicity": int(
                    client_diag[
                        "largest_multiplicity"
                    ]
                ),

                "cc_smallest_eigenvalues_json": (
                    json.dumps(
                        client_diag[
                            "smallest_eigenvalues"
                        ]
                    )
                ),

                "cx_nodes": int(
                    content_diag[
                        "nodes"
                    ]
                ),

                "cx_edges": int(
                    content_diag[
                        "edges"
                    ]
                ),

                "cx_neighbours": int(
                    content_diag[
                        "neighbours"
                    ]
                ),

                "cx_eigenpairs_used": int(
                    content_diag[
                        "eigenpairs_used"
                    ]
                ),

                "cx_spectrum_truncated": bool(
                    content_diag[
                        "spectrum_truncated"
                    ]
                ),

                "cx_repeated_eigenspaces": int(
                    content_diag[
                        "repeated_eigenspaces"
                    ]
                ),

                "cx_largest_multiplicity": int(
                    content_diag[
                        "largest_multiplicity"
                    ]
                ),

                "cx_smallest_eigenvalues_json": (
                    json.dumps(
                        content_diag[
                            "smallest_eigenvalues"
                        ]
                    )
                ),

                "has_repeated_eigenspace": bool(
                    diagnostics[
                        "full_repeated_eigenspaces"
                    ]
                    > 0
                    or client_diag[
                        "repeated_eigenspaces"
                    ]
                    > 0
                    or content_diag[
                        "repeated_eigenspaces"
                    ]
                    > 0
                ),
            })

        except Exception as exc:

            failures.append({
                "client_id": client_id,
                "ego_filename": ego_filename,
                "error": repr(
                    exc
                ),
            })


    if len(
        client_ids
    ) == 0:

        raise RuntimeError(
            f"No local signatures were successfully computed "
            f"for {dataset_name}."
        )

    client_ids_array = np.asarray(
        client_ids,
        dtype=str,
    )

    raw_array = np.vstack(
        raw_signatures
    ).astype(
        float
    )

    abs_array = np.vstack(
        abs_signatures
    ).astype(
        float
    )

    hks_array = np.vstack(
        hks_signatures
    ).astype(
        float
    )

    stats_df = pd.DataFrame(
        stats_rows
    )


    np.savez_compressed(
        signatures_path,

        client_ids=client_ids_array,

        raw=raw_array,

        abs=abs_array,

        hks=hks_array,
    )


    stats_df.to_csv(
        stats_path,
        index=False,
        compression="gzip",
    )


    repeated_mask = (
        stats_df[
            "has_repeated_eigenspace"
        ].astype(
            bool
        )
    )

    raw_zero_mask = (
        stats_df[
            "raw_zero"
        ].astype(
            bool
        )
    )

    hks_zero_mask = (
        stats_df[
            "hks_zero"
        ].astype(
            bool
        )
    )

    metadata = {
        "dataset": dataset_name,

        "schema_version": SIGNATURE_SCHEMA_VERSION,

        "number_of_clients_requested": int(
            len(
                ego_index
            )
        ),

        "number_of_clients_processed": int(
            len(
                client_ids
            )
        ),

        "number_of_failures": int(
            len(
                failures
            )
        ),

        "failures": failures,

        "methods": [
            "raw",
            "abs",
            "hks",
        ],

        "raw_signature_dimension": int(
            RAW_SIGNATURE_DIM
        ),

        "hks_time_scales": [
            float(
                value
            )
            for value
            in HKS_TIMES
        ],

        "hks_blocks": [
            "client-client",
            "client-content",
        ],

        "hks_signature_dimension": int(
            len(
                HKS_TIMES
            )
            * 2
        ),

        "dense_eigen_threshold": int(
            DENSE_EIGEN_THRESHOLD
        ),

        "max_hks_eigenpairs": int(
            MAX_HKS_EIGENPAIRS
        ),

        "sparse_eigen_buffer": int(
            SPARSE_EIGEN_BUFFER
        ),

        "eigenvalue_absolute_tolerance": float(
            EIGENVALUE_ATOL
        ),

        "eigenvalue_relative_tolerance": float(
            EIGENVALUE_RTOL
        ),

        "number_clients_with_repeated_eigenspace": int(
            repeated_mask.sum()
        ),

        "fraction_clients_with_repeated_eigenspace": float(
            repeated_mask.mean()
        ),

        "number_zero_raw_signatures": int(
            raw_zero_mask.sum()
        ),

        "fraction_zero_raw_signatures": float(
            raw_zero_mask.mean()
        ),

        "number_zero_hks_signatures": int(
            hks_zero_mask.sum()
        ),

        "fraction_zero_hks_signatures": float(
            hks_zero_mask.mean()
        ),

        "number_full_spectra_truncated": int(
            stats_df[
                "full_spectrum_truncated"
            ].astype(
                bool
            ).sum()
        ),

        "number_cc_spectra_truncated": int(
            stats_df[
                "cc_spectrum_truncated"
            ].astype(
                bool
            ).sum()
        ),

        "number_cx_spectra_truncated": int(
            stats_df[
                "cx_spectrum_truncated"
            ].astype(
                bool
            ).sum()
        ),

        "mean_ego_nodes": float(
            stats_df[
                "ego_nodes"
            ].mean()
        ),

        "median_ego_nodes": float(
            stats_df[
                "ego_nodes"
            ].median()
        ),

        "mean_cc_neighbours": float(
            stats_df[
                "cc_neighbours"
            ].mean()
        ),

        "mean_cx_neighbours": float(
            stats_df[
                "cx_neighbours"
            ].mean()
        ),

        "signatures_file": os.path.basename(
            signatures_path
        ),

        "statistics_file": os.path.basename(
            stats_path
        ),
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
        f"LOCAL SPECTRAL SIGNATURES: "
        f"{dataset_name}"
    )

    print(
        "========================================"
    )

    print(
        f"Clients processed: "
        f"{len(client_ids):,}"
    )

    print(
        f"Failures: "
        f"{len(failures):,}"
    )

    print(
        f"RAW dimension: "
        f"{raw_array.shape[1]}"
    )

    print(
        f"HKS dimension: "
        f"{hks_array.shape[1]}"
    )

    print(
        "Clients with repeated eigenspaces: "
        f"{int(repeated_mask.sum()):,} "
        f"("
        f"{100.0 * repeated_mask.mean():.2f}%"
        f")"
    )

    print(
        "Zero RAW signatures: "
        f"{int(raw_zero_mask.sum()):,}"
    )

    print(
        "Zero HKS signatures: "
        f"{int(hks_zero_mask.sum()):,}"
    )

    print(
        f"\nSignatures: "
        f"{signatures_path}"
    )

    print(
        f"Statistics: "
        f"{stats_path}"
    )

    print(
        f"Metadata: "
        f"{metadata_path}"
    )

    print(
        "========================================\n"
    )

    return {
        "client_ids": client_ids_array,

        "raw": raw_array,

        "abs": abs_array,

        "hks": hks_array,

        "stats": stats_df,
    }


def load_local_signatures(
    dataset_name="enron",
    output_folder="output/",
):

    dataset_folder = os.path.join(
        output_folder,
        dataset_name,
    )

    signatures_path = os.path.join(
        dataset_folder,
        f"{dataset_name}_local_signatures.npz",
    )

    if not os.path.exists(
        signatures_path
    ):
        raise FileNotFoundError(
            f"Missing local signatures: "
            f"{signatures_path}"
        )

    data = np.load(
        signatures_path,
        allow_pickle=False,
    )

    return {
        "client_ids": data[
            "client_ids"
        ],

        "raw": data[
            "raw"
        ],

        "abs": data[
            "abs"
        ],

        "hks": data[
            "hks"
        ],
    }
