import math
from typing import Optional, Sequence

import networkx as nx
import numpy as np


PRIVACY_SCHEMA_VERSION = "1.0"

DEFAULT_SIGNATURE_CLIP_NORM = 1.0

DEFAULT_D_MAX = 50

DEFAULT_DEGREE_MODE = "role"

NUMERICAL_EPS = 1e-12


def make_rng(seed=None, rng=None):


    if (
        seed is not None
        and rng is not None
    ):
        raise ValueError(
            "Provide either seed or rng, not both."
        )

    if rng is not None:

        if not isinstance(
            rng,
            np.random.Generator,
        ):
            raise TypeError(
                "rng must be an instance of "
                "numpy.random.Generator."
            )

        return rng

    return np.random.default_rng(
        seed
    )


def validate_epsilon(epsilon):


    epsilon = float(
        epsilon
    )

    if not np.isfinite(
        epsilon
    ):
        raise ValueError(
            "epsilon must be finite."
        )

    if epsilon <= 0:
        raise ValueError(
            "epsilon must be strictly positive."
        )

    return epsilon


def validate_clip_norm(C):


    C = float(
        C
    )

    if not np.isfinite(
        C
    ):
        raise ValueError(
            "C must be finite."
        )

    if C <= 0:
        raise ValueError(
            "C must be strictly positive."
        )

    return C


def validate_d_max(d_max):


    d_max = int(
        d_max
    )

    if d_max <= 0:
        raise ValueError(
            "d_max must be strictly positive."
        )

    return d_max


def clip_l2(
    vector,
    C=DEFAULT_SIGNATURE_CLIP_NORM,
):


    C = validate_clip_norm(
        C
    )

    vector = np.asarray(
        vector,
        dtype=float,
    )

    if vector.ndim != 1:
        raise ValueError(
            "vector must be one-dimensional."
        )

    if not np.all(
        np.isfinite(
            vector
        )
    ):
        raise ValueError(
            "vector contains NaN or infinite values."
        )

    original_norm = float(
        np.linalg.norm(
            vector,
            ord=2,
        )
    )

    if original_norm <= C:

        clipped = vector.copy()

        clipping_factor = 1.0

        was_clipped = False

    else:

        clipping_factor = (
            C
            /
            max(
                original_norm,
                NUMERICAL_EPS,
            )
        )

        clipped = (
            vector
            * clipping_factor
        )

        was_clipped = True

    clipped_norm = float(
        np.linalg.norm(
            clipped,
            ord=2,
        )
    )

    diagnostics = {
        "clip_norm_C": float(
            C
        ),

        "original_l2_norm": float(
            original_norm
        ),

        "clipped_l2_norm": float(
            clipped_norm
        ),

        "clipping_factor": float(
            clipping_factor
        ),

        "was_clipped": bool(
            was_clipped
        ),
    }

    return (
        clipped,
        diagnostics,
    )


def sample_uniform_unit_direction(
    dimension,
    rng,
):


    dimension = int(
        dimension
    )

    if dimension <= 0:
        raise ValueError(
            "dimension must be positive."
        )

    if dimension == 1:

        sign = rng.choice(
            np.array(
                [
                    -1.0,
                    1.0,
                ],
                dtype=float,
            )
        )

        return np.array(
            [
                sign
            ],
            dtype=float,
        )

    while True:

        direction = rng.normal(
            loc=0.0,
            scale=1.0,
            size=dimension,
        )

        norm = np.linalg.norm(
            direction,
            ord=2,
        )

        if norm > NUMERICAL_EPS:

            return (
                direction
                /
                norm
            )


def apply_l2_knorm_mechanism(
    vector,
    epsilon,
    C=DEFAULT_SIGNATURE_CLIP_NORM,
    seed=None,
    rng=None,
):


    epsilon = validate_epsilon(
        epsilon
    )

    C = validate_clip_norm(
        C
    )

    rng = make_rng(
        seed=seed,
        rng=rng,
    )

    clipped, clipping_diagnostics = (
        clip_l2(
            vector,
            C=C,
        )
    )

    dimension = int(
        len(
            clipped
        )
    )

    if dimension <= 0:
        raise ValueError(
            "Cannot privatize an empty vector."
        )

    delta_l2 = float(
        2.0
        * C
    )

    radial_scale = float(
        delta_l2
        /
        epsilon
    )

    direction = (
        sample_uniform_unit_direction(
            dimension,
            rng,
        )
    )

    radius = float(
        rng.gamma(
            shape=float(
                dimension
            ),
            scale=radial_scale,
        )
    )

    noise = (
        radius
        * direction
    )

    privatized = (
        clipped
        + noise
    )

    diagnostics = {
        "mechanism": "l2_knorm",

        "privacy_type": "pure_ldp",

        "epsilon": float(
            epsilon
        ),

        "dimension": int(
            dimension
        ),

        "sensitivity_l2": float(
            delta_l2
        ),

        "radial_gamma_shape": float(
            dimension
        ),

        "radial_gamma_scale": float(
            radial_scale
        ),

        "noise_l2_norm": float(
            np.linalg.norm(
                noise,
                ord=2,
            )
        ),

        "noise_l1_norm": float(
            np.linalg.norm(
                noise,
                ord=1,
            )
        ),

        "noise_linf_norm": float(
            np.linalg.norm(
                noise,
                ord=np.inf,
            )
        ),

        "output_l2_norm": float(
            np.linalg.norm(
                privatized,
                ord=2,
            )
        ),

        **clipping_diagnostics,
    }

    return (
        privatized,
        diagnostics,
    )


def apply_coordinate_laplace_mechanism(
    vector,
    epsilon,
    C=DEFAULT_SIGNATURE_CLIP_NORM,
    seed=None,
    rng=None,
):


    epsilon = validate_epsilon(
        epsilon
    )

    C = validate_clip_norm(
        C
    )

    rng = make_rng(
        seed=seed,
        rng=rng,
    )

    clipped, clipping_diagnostics = (
        clip_l2(
            vector,
            C=C,
        )
    )

    dimension = int(
        len(
            clipped
        )
    )

    if dimension <= 0:
        raise ValueError(
            "Cannot privatize an empty vector."
        )

    delta_l1 = float(
        2.0
        * C
        * math.sqrt(
            dimension
        )
    )

    scale = float(
        delta_l1
        /
        epsilon
    )

    noise = rng.laplace(
        loc=0.0,
        scale=scale,
        size=dimension,
    )

    privatized = (
        clipped
        + noise
    )

    diagnostics = {
        "mechanism": "coordinate_laplace",

        "privacy_type": "pure_ldp",

        "epsilon": float(
            epsilon
        ),

        "dimension": int(
            dimension
        ),

        "sensitivity_l1": float(
            delta_l1
        ),

        "laplace_scale": float(
            scale
        ),

        "noise_l2_norm": float(
            np.linalg.norm(
                noise,
                ord=2,
            )
        ),

        "noise_l1_norm": float(
            np.linalg.norm(
                noise,
                ord=1,
            )
        ),

        "noise_linf_norm": float(
            np.linalg.norm(
                noise,
                ord=np.inf,
            )
        ),

        "output_l2_norm": float(
            np.linalg.norm(
                privatized,
                ord=2,
            )
        ),

        **clipping_diagnostics,
    }

    return (
        privatized,
        diagnostics,
    )


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


def extract_role_degree_vector(
    ego_graph,
    main_node,
    roles=(
        "client",
        "content",
    ),
):


    if main_node not in ego_graph:
        raise KeyError(
            f"Main node {main_node!r} "
            "is not present in ego graph."
        )

    neighbours = get_unique_neighbours(
        ego_graph,
        main_node,
    )

    counts = []

    component_names = []

    for role in roles:

        role = str(
            role
        )

        count = sum(
            1
            for neighbour in neighbours
            if ego_graph.nodes[
                neighbour
            ].get(
                "role"
            ) == role
        )

        counts.append(
            int(
                count
            )
        )

        component_names.append(
            role
        )

    vector = np.asarray(
        counts,
        dtype=int,
    )

    diagnostics = {
        "degree_mode": "role",

        "component_names": list(
            component_names
        ),

        "number_of_components": int(
            len(
                component_names
            )
        ),

        "total_unique_neighbours": int(
            len(
                neighbours
            )
        ),

        "raw_degree_sum": int(
            np.sum(
                vector
            )
        ),
    }

    return (
        vector,
        component_names,
        diagnostics,
    )


def extract_type_degree_vector(
    ego_graph,
    main_node,
    node_types=None,
):


    if main_node not in ego_graph:
        raise KeyError(
            f"Main node {main_node!r} "
            "is not present in ego graph."
        )

    neighbours = get_unique_neighbours(
        ego_graph,
        main_node,
    )

    if node_types is None:

        node_types = sorted(
            {
                str(
                    ego_graph.nodes[
                        neighbour
                    ].get(
                        "type",
                        "unknown",
                    )
                )
                for neighbour in neighbours
            }
        )

    else:

        node_types = [
            str(
                value
            )
            for value in node_types
        ]

    counts = []

    for node_type in node_types:

        count = sum(
            1
            for neighbour in neighbours
            if str(
                ego_graph.nodes[
                    neighbour
                ].get(
                    "type",
                    "unknown",
                )
            )
            == node_type
        )

        counts.append(
            int(
                count
            )
        )

    vector = np.asarray(
        counts,
        dtype=int,
    )

    diagnostics = {
        "degree_mode": "type",

        "component_names": list(
            node_types
        ),

        "number_of_components": int(
            len(
                node_types
            )
        ),

        "total_unique_neighbours": int(
            len(
                neighbours
            )
        ),

        "raw_degree_sum": int(
            np.sum(
                vector
            )
        ),
    }

    return (
        vector,
        list(
            node_types
        ),
        diagnostics,
    )


def get_incident_relation_counts(
    ego_graph,
    main_node,
):


    counts = {}

    if ego_graph.is_multigraph():

        if ego_graph.is_directed():

            incoming = ego_graph.in_edges(
                main_node,
                keys=True,
                data=True,
            )

            outgoing = ego_graph.out_edges(
                main_node,
                keys=True,
                data=True,
            )

            for _, _, _, data in incoming:

                relation = str(
                    data.get(
                        "relation",
                        "undefined",
                    )
                )

                counts[
                    relation
                ] = (
                    counts.get(
                        relation,
                        0,
                    )
                    + 1
                )

            for _, _, _, data in outgoing:

                relation = str(
                    data.get(
                        "relation",
                        "undefined",
                    )
                )

                counts[
                    relation
                ] = (
                    counts.get(
                        relation,
                        0,
                    )
                    + 1
                )

        else:

            for _, _, _, data in ego_graph.edges(
                main_node,
                keys=True,
                data=True,
            ):

                relation = str(
                    data.get(
                        "relation",
                        "undefined",
                    )
                )

                counts[
                    relation
                ] = (
                    counts.get(
                        relation,
                        0,
                    )
                    + 1
                )

    else:

        if ego_graph.is_directed():

            for _, _, data in ego_graph.in_edges(
                main_node,
                data=True,
            ):

                relation = str(
                    data.get(
                        "relation",
                        "undefined",
                    )
                )

                counts[
                    relation
                ] = (
                    counts.get(
                        relation,
                        0,
                    )
                    + 1
                )

            for _, _, data in ego_graph.out_edges(
                main_node,
                data=True,
            ):

                relation = str(
                    data.get(
                        "relation",
                        "undefined",
                    )
                )

                counts[
                    relation
                ] = (
                    counts.get(
                        relation,
                        0,
                    )
                    + 1
                )

        else:

            for _, _, data in ego_graph.edges(
                main_node,
                data=True,
            ):

                relation = str(
                    data.get(
                        "relation",
                        "undefined",
                    )
                )

                counts[
                    relation
                ] = (
                    counts.get(
                        relation,
                        0,
                    )
                    + 1
                )

    return counts


def extract_relation_degree_vector(
    ego_graph,
    main_node,
    relations=None,
):


    counts = get_incident_relation_counts(
        ego_graph,
        main_node,
    )

    if relations is None:

        relations = sorted(
            counts.keys()
        )

    else:

        relations = [
            str(
                relation
            )
            for relation in relations
        ]

    vector = np.asarray(
        [
            int(
                counts.get(
                    relation,
                    0,
                )
            )
            for relation in relations
        ],
        dtype=int,
    )

    diagnostics = {
        "degree_mode": "relation",

        "component_names": list(
            relations
        ),

        "number_of_components": int(
            len(
                relations
            )
        ),

        "raw_degree_sum": int(
            np.sum(
                vector
            )
        ),
    }

    return (
        vector,
        list(
            relations
        ),
        diagnostics,
    )


def extract_typed_degree_vector(
    ego_graph,
    main_node,
    mode=DEFAULT_DEGREE_MODE,
    component_names=None,
):


    mode = str(
        mode
    ).lower()

    if mode == "role":

        roles = (
            component_names
            if component_names is not None
            else (
                "client",
                "content",
            )
        )

        return extract_role_degree_vector(
            ego_graph,
            main_node,
            roles=roles,
        )

    if mode == "type":

        return extract_type_degree_vector(
            ego_graph,
            main_node,
            node_types=component_names,
        )

    if mode == "relation":

        return extract_relation_degree_vector(
            ego_graph,
            main_node,
            relations=component_names,
        )

    raise ValueError(
        "mode must be one of: "
        "'role', 'type', 'relation'."
    )


def clip_degree_vector(
    degree_vector,
    d_max=DEFAULT_D_MAX,
):


    d_max = validate_d_max(
        d_max
    )

    degree_vector = np.asarray(
        degree_vector,
        dtype=float,
    )

    if degree_vector.ndim != 1:
        raise ValueError(
            "degree_vector must be one-dimensional."
        )

    if not np.all(
        np.isfinite(
            degree_vector
        )
    ):
        raise ValueError(
            "degree_vector contains NaN or infinite values."
        )

    original = degree_vector.copy()

    clipped = np.clip(
        degree_vector,
        0,
        d_max,
    )


    clipped = np.rint(
        clipped
    ).astype(
        int
    )

    component_was_clipped = (
        np.rint(
            original
        ).astype(
            int
        )
        != clipped
    )

    diagnostics = {
        "D_max": int(
            d_max
        ),

        "number_of_components": int(
            len(
                clipped
            )
        ),

        "number_components_clipped": int(
            np.sum(
                component_was_clipped
            )
        ),

        "fraction_components_clipped": float(
            np.mean(
                component_was_clipped
            )
            if len(
                component_was_clipped
            )
            else 0.0
        ),

        "was_any_component_clipped": bool(
            np.any(
                component_was_clipped
            )
        ),

        "original_degree_sum": float(
            np.sum(
                original
            )
        ),

        "clipped_degree_sum": int(
            np.sum(
                clipped
            )
        ),

        "original_degree_vector": [
            float(
                value
            )
            for value in original
        ],

        "clipped_degree_vector": [
            int(
                value
            )
            for value in clipped
        ],
    }

    return (
        clipped,
        diagnostics,
    )


def degree_vector_l1_sensitivity(
    number_of_components,
    d_max=DEFAULT_D_MAX,
    adjacency="client_report",
):


    q = int(
        number_of_components
    )

    if q <= 0:
        raise ValueError(
            "number_of_components must be positive."
        )

    d_max = validate_d_max(
        d_max
    )

    adjacency = str(
        adjacency
    ).lower()

    if adjacency == "client_report":

        return float(
            q
            * d_max
        )

    if adjacency == "incident_edge":

        return 1.0

    raise ValueError(
        "adjacency must be either "
        "'client_report' or 'incident_edge'."
    )


def sample_discrete_laplace(
    epsilon,
    sensitivity,
    size,
    rng,
):


    epsilon = validate_epsilon(
        epsilon
    )

    sensitivity = float(
        sensitivity
    )

    if (
        not np.isfinite(
            sensitivity
        )
        or sensitivity <= 0
    ):
        raise ValueError(
            "sensitivity must be finite and positive."
        )

    size = int(
        size
    )

    if size <= 0:
        raise ValueError(
            "size must be positive."
        )

    lambda_parameter = float(
        epsilon
        /
        sensitivity
    )


    geometric_probability = float(
        -np.expm1(
            -lambda_parameter
        )
    )


    geometric_probability = max(
        geometric_probability,
        np.finfo(
            float
        ).tiny,
    )

    geometric_probability = min(
        geometric_probability,
        1.0,
    )

    g1 = (
        rng.geometric(
            geometric_probability,
            size=size,
        )
        - 1
    )

    g2 = (
        rng.geometric(
            geometric_probability,
            size=size,
        )
        - 1
    )

    noise = (
        g1
        - g2
    ).astype(
        int
    )

    diagnostics = {
        "lambda": float(
            lambda_parameter
        ),

        "alpha": float(
            math.exp(
                -lambda_parameter
            )
        ),

        "geometric_success_probability": float(
            geometric_probability
        ),
    }

    return (
        noise,
        diagnostics,
    )


def apply_discrete_laplace_degree_mechanism(
    degree_vector,
    epsilon,
    d_max=DEFAULT_D_MAX,
    adjacency="client_report",
    postprocess=True,
    seed=None,
    rng=None,
):


    epsilon = validate_epsilon(
        epsilon
    )

    d_max = validate_d_max(
        d_max
    )

    rng = make_rng(
        seed=seed,
        rng=rng,
    )

    clipped, clipping_diagnostics = (
        clip_degree_vector(
            degree_vector,
            d_max=d_max,
        )
    )

    q = int(
        len(
            clipped
        )
    )

    if q <= 0:
        raise ValueError(
            "Cannot privatize an empty degree vector."
        )

    sensitivity = (
        degree_vector_l1_sensitivity(
            number_of_components=q,
            d_max=d_max,
            adjacency=adjacency,
        )
    )

    noise, noise_parameters = (
        sample_discrete_laplace(
            epsilon=epsilon,
            sensitivity=sensitivity,
            size=q,
            rng=rng,
        )
    )

    raw_noisy = (
        clipped.astype(
            np.int64
        )
        +
        noise.astype(
            np.int64
        )
    )

    if postprocess:

        privatized = np.clip(
            raw_noisy,
            0,
            d_max,
        ).astype(
            int
        )

    else:

        privatized = raw_noisy.astype(
            int
        )

    diagnostics = {
        "mechanism": "discrete_laplace",

        "privacy_type": "pure_ldp",

        "epsilon": float(
            epsilon
        ),

        "adjacency": str(
            adjacency
        ),

        "number_of_components": int(
            q
        ),

        "sensitivity_l1": float(
            sensitivity
        ),

        "noise_vector": [
            int(
                value
            )
            for value in noise
        ],

        "noise_l1_norm": int(
            np.linalg.norm(
                noise,
                ord=1,
            )
        ),

        "noise_l2_norm": float(
            np.linalg.norm(
                noise,
                ord=2,
            )
        ),

        "raw_noisy_degree_vector": [
            int(
                value
            )
            for value in raw_noisy
        ],

        "postprocessed_degree_vector": [
            int(
                value
            )
            for value in privatized
        ],

        "postprocessing_enabled": bool(
            postprocess
        ),

        "number_negative_before_postprocessing": int(
            np.sum(
                raw_noisy < 0
            )
        ),

        "number_above_D_max_before_postprocessing": int(
            np.sum(
                raw_noisy > d_max
            )
        ),

        **noise_parameters,

        **clipping_diagnostics,
    }

    return (
        privatized,
        diagnostics,
    )


def split_privacy_budget(
    epsilon_total,
    signature_fraction=0.5,
):


    epsilon_total = validate_epsilon(
        epsilon_total
    )

    signature_fraction = float(
        signature_fraction
    )

    if not (
        0.0
        < signature_fraction
        < 1.0
    ):
        raise ValueError(
            "signature_fraction must lie strictly between 0 and 1."
        )

    epsilon_signature = float(
        epsilon_total
        * signature_fraction
    )

    epsilon_degree = float(
        epsilon_total
        - epsilon_signature
    )


    if (
        epsilon_signature <= 0
        or epsilon_degree <= 0
    ):
        raise ValueError(
            "Both privacy sub-budgets must be positive."
        )

    return (
        epsilon_signature,
        epsilon_degree,
    )


def privatize_client_report(
    signature,
    degree_vector,
    epsilon_total,
    signature_fraction=0.5,
    signature_mechanism="l2_knorm",
    signature_clip_norm=DEFAULT_SIGNATURE_CLIP_NORM,
    d_max=DEFAULT_D_MAX,
    degree_adjacency="client_report",
    seed=None,
    rng=None,
):


    epsilon_total = validate_epsilon(
        epsilon_total
    )

    rng = make_rng(
        seed=seed,
        rng=rng,
    )

    (
        epsilon_signature,
        epsilon_degree,
    ) = split_privacy_budget(
        epsilon_total=epsilon_total,
        signature_fraction=signature_fraction,
    )

    signature_mechanism = str(
        signature_mechanism
    ).lower()


    if signature_mechanism == "l2_knorm":

        (
            private_signature,
            signature_diagnostics,
        ) = apply_l2_knorm_mechanism(
            vector=signature,
            epsilon=epsilon_signature,
            C=signature_clip_norm,
            rng=rng,
        )

    elif signature_mechanism in {
        "coordinate_laplace",
        "laplace",
    }:

        (
            private_signature,
            signature_diagnostics,
        ) = apply_coordinate_laplace_mechanism(
            vector=signature,
            epsilon=epsilon_signature,
            C=signature_clip_norm,
            rng=rng,
        )

    else:

        raise ValueError(
            "Unknown signature mechanism: "
            f"{signature_mechanism}. "
            "Supported mechanisms are "
            "'l2_knorm' and 'coordinate_laplace'."
        )


    (
        private_degree,
        degree_diagnostics,
    ) = (
        apply_discrete_laplace_degree_mechanism(
            degree_vector=degree_vector,
            epsilon=epsilon_degree,
            d_max=d_max,
            adjacency=degree_adjacency,
            postprocess=True,
            rng=rng,
        )
    )

    diagnostics = {
        "privacy_schema_version": (
            PRIVACY_SCHEMA_VERSION
        ),

        "privacy_type": "pure_ldp",

        "epsilon_total": float(
            epsilon_total
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

        "composition_epsilon": float(
            epsilon_signature
            + epsilon_degree
        ),

        "signature_mechanism": (
            signature_mechanism
        ),

        "degree_mechanism": (
            "discrete_laplace"
        ),

        "degree_adjacency": str(
            degree_adjacency
        ),

        "signature": (
            signature_diagnostics
        ),

        "degree": (
            degree_diagnostics
        ),
    }

    return (
        np.asarray(
            private_signature,
            dtype=float,
        ),
        np.asarray(
            private_degree,
            dtype=int,
        ),
        diagnostics,
    )


def describe_privacy_configuration(
    epsilon_total,
    signature_fraction=0.5,
    signature_dimension=12,
    signature_clip_norm=DEFAULT_SIGNATURE_CLIP_NORM,
    degree_components=2,
    d_max=DEFAULT_D_MAX,
    degree_adjacency="client_report",
):


    (
        epsilon_signature,
        epsilon_degree,
    ) = split_privacy_budget(
        epsilon_total=epsilon_total,
        signature_fraction=signature_fraction,
    )

    signature_clip_norm = (
        validate_clip_norm(
            signature_clip_norm
        )
    )

    d_max = validate_d_max(
        d_max
    )

    signature_dimension = int(
        signature_dimension
    )

    degree_components = int(
        degree_components
    )

    if signature_dimension <= 0:
        raise ValueError(
            "signature_dimension must be positive."
        )

    if degree_components <= 0:
        raise ValueError(
            "degree_components must be positive."
        )

    delta_signature_l2 = float(
        2.0
        * signature_clip_norm
    )

    delta_signature_l1_laplace = float(
        2.0
        * signature_clip_norm
        * math.sqrt(
            signature_dimension
        )
    )

    delta_degree_l1 = (
        degree_vector_l1_sensitivity(
            number_of_components=(
                degree_components
            ),
            d_max=d_max,
            adjacency=degree_adjacency,
        )
    )

    return {
        "privacy_schema_version": (
            PRIVACY_SCHEMA_VERSION
        ),

        "privacy_type": "pure_ldp",

        "epsilon_total": float(
            epsilon_total
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

        "signature_dimension": int(
            signature_dimension
        ),

        "signature_clip_norm_C": float(
            signature_clip_norm
        ),

        "signature_l2_sensitivity": float(
            delta_signature_l2
        ),

        "coordinate_laplace_l1_sensitivity": float(
            delta_signature_l1_laplace
        ),

        "degree_components": int(
            degree_components
        ),

        "D_max": int(
            d_max
        ),

        "degree_adjacency": str(
            degree_adjacency
        ),

        "degree_l1_sensitivity": float(
            delta_degree_l1
        ),

        "sequential_composition": {
            "epsilon_signature_plus_epsilon_degree": float(
                epsilon_signature
                + epsilon_degree
            ),

            "equals_epsilon_total": bool(
                np.isclose(
                    epsilon_signature
                    + epsilon_degree,
                    epsilon_total,
                )
            ),
        },
    }


def run_sanity_checks():


    rng = np.random.default_rng(
        12345
    )


    vector = np.array(
        [
            3.0,
            4.0,
        ]
    )

    clipped, info = clip_l2(
        vector,
        C=1.0,
    )

    assert np.isclose(
        np.linalg.norm(
            clipped
        ),
        1.0,
    )

    assert info[
        "was_clipped"
    ]


    private_vector, info = (
        apply_l2_knorm_mechanism(
            vector=np.array(
                [
                    0.2,
                    0.4,
                ]
            ),
            epsilon=1.0,
            C=1.0,
            rng=rng,
        )
    )

    assert private_vector.shape == (
        2,
    )

    assert (
        info[
            "sensitivity_l2"
        ]
        == 2.0
    )


    degree = np.array(
        [
            10,
            100,
        ]
    )

    clipped_degree, info = (
        clip_degree_vector(
            degree,
            d_max=50,
        )
    )

    assert np.array_equal(
        clipped_degree,
        np.array(
            [
                10,
                50,
            ]
        ),
    )


    assert (
        degree_vector_l1_sensitivity(
            number_of_components=2,
            d_max=50,
            adjacency="client_report",
        )
        == 100.0
    )


    epsilon_s, epsilon_d = (
        split_privacy_budget(
            epsilon_total=3.0,
            signature_fraction=0.5,
        )
    )

    assert np.isclose(
        epsilon_s
        + epsilon_d,
        3.0,
    )

    print(
        "privacy_mechanisms.py sanity checks passed."
    )


if __name__ == "__main__":

    run_sanity_checks()
