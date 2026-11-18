import argparse
import hashlib
import json
import math
import os
import time

from collections import (
    Counter,
    defaultdict,
)

import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import t as student_t

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from link_prediction_experiment import (
    generate_selected_descriptors,
)


SCHEMA_VERSION = "yelp-link-special-1.1"

DEFAULT_OUTPUT_FOLDER = "output_local_test"

DEFAULT_SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

DEFAULT_TEST_POSITIVES = 500

DEFAULT_BOOTSTRAP_REPETITIONS = 500

DEFAULT_D_MAX = 50

DEFAULT_DEGREE_WEIGHT = 1.0

SCORE_ROUND_DECIMALS = 12


DESCRIPTORS = (
    "hks_d12_global",
    "eigen_d8_block",
    "eigen_d12_global",
)

ORIENTATIONS = (
    "similarity",
    "distance",
)


def deterministic_seed(*parts):

    text = "|".join(
        str(part)
        for part in parts
    )

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:4],
        byteorder="little",
        signed=False,
    )


def stabilize_scores(
    scores,
):

    scores = np.asarray(
        scores,
        dtype=float,
    )

    return np.round(
        scores,
        decimals=SCORE_ROUND_DECIMALS,
    )


def cosine_score(
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

    first_norm = float(
        np.linalg.norm(
            first
        )
    )

    second_norm = float(
        np.linalg.norm(
            second
        )
    )

    if (
        first_norm <= 0.0
        or second_norm <= 0.0
    ):
        return 0.0

    value = float(
        np.dot(
            first,
            second,
        )
        /
        (
            first_norm
            *
            second_norm
        )
    )

    return float(
        np.clip(
            value,
            -1.0,
            1.0,
        )
    )


def canonical_interaction(
    user_id,
    business_id,
):

    return (
        str(
            user_id
        ),
        str(
            business_id
        ),
    )


def mean_std_ci95(
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

    n = len(
        values
    )

    if n == 0:

        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
        }

    mean = float(
        np.mean(
            values
        )
    )

    if n == 1:

        return {
            "n": 1,
            "mean": mean,
            "std": 0.0,
            "ci95_low": mean,
            "ci95_high": mean,
        }

    std = float(
        np.std(
            values,
            ddof=1,
        )
    )

    standard_error = (
        std
        /
        math.sqrt(
            n
        )
    )

    critical = float(
        student_t.ppf(
            0.975,
            df=n - 1,
        )
    )

    margin = (
        critical
        *
        standard_error
    )

    return {
        "n": int(
            n
        ),

        "mean": mean,

        "std": std,

        "ci95_low": float(
            mean
            -
            margin
        ),

        "ci95_high": float(
            mean
            +
            margin
        ),
    }


def find_review_or_tip_endpoints(
    graph,
    node_id,
    node_type,
):

    users = set()

    businesses = set()

    if node_type == "review":

        user_relation = "wrote"

        business_relation = "about"

    elif node_type == "tip":

        user_relation = "wrote_tip"

        business_relation = "about_tip"

    else:

        raise ValueError(
            f"Unsupported interaction node type: "
            f"{node_type}"
        )


    for source, _, attributes in graph.in_edges(
        node_id,
        data=True,
    ):

        relation = str(
            attributes.get(
                "relation",
                "",
            )
        )

        source_type = str(
            graph.nodes[
                source
            ].get(
                "type",
                "",
            )
        )

        if (
            relation == user_relation
            and source_type == "user"
        ):

            users.add(
                str(
                    source
                )
            )

    for _, target, attributes in graph.out_edges(
        node_id,
        data=True,
    ):

        relation = str(
            attributes.get(
                "relation",
                "",
            )
        )

        target_type = str(
            graph.nodes[
                target
            ].get(
                "type",
                "",
            )
        )

        if (
            relation == business_relation
            and target_type == "business"
        ):

            businesses.add(
                str(
                    target
                )
            )


    for _, target, attributes in graph.out_edges(
        node_id,
        data=True,
    ):

        relation = str(
            attributes.get(
                "relation",
                "",
            )
        )

        target_type = str(
            graph.nodes[
                target
            ].get(
                "type",
                "",
            )
        )

        if (
            relation == user_relation
            and target_type == "user"
        ):

            users.add(
                str(
                    target
                )
            )

    for source, _, attributes in graph.in_edges(
        node_id,
        data=True,
    ):

        relation = str(
            attributes.get(
                "relation",
                "",
            )
        )

        source_type = str(
            graph.nodes[
                source
            ].get(
                "type",
                "",
            )
        )

        if (
            relation == business_relation
            and source_type == "business"
        ):

            businesses.add(
                str(
                    source
                )
            )

    return (
        users,
        businesses,
    )


def build_yelp_interaction_index(
    graph,
):

    print()

    print(
        "Building Yelp User-Business interaction index..."
    )

    interaction_nodes = defaultdict(
        lambda: {
            "review": [],
            "tip": [],
        }
    )

    invalid_content_nodes = []

    review_nodes = 0

    tip_nodes = 0

    for node_id, attributes in graph.nodes(
        data=True
    ):

        node_type = str(
            attributes.get(
                "type",
                "",
            )
        )

        if node_type not in {
            "review",
            "tip",
        }:

            continue

        if node_type == "review":

            review_nodes += 1

        else:

            tip_nodes += 1

        users, businesses = (
            find_review_or_tip_endpoints(
                graph,
                node_id,
                node_type,
            )
        )

        if (
            len(
                users
            ) != 1
            or
            len(
                businesses
            ) != 1
        ):

            invalid_content_nodes.append({
                "node": str(
                    node_id
                ),

                "type": node_type,

                "users": int(
                    len(
                        users
                    )
                ),

                "businesses": int(
                    len(
                        businesses
                    )
                ),
            })

            continue

        user_id = next(
            iter(
                users
            )
        )

        business_id = next(
            iter(
                businesses
            )
        )

        pair = canonical_interaction(
            user_id,
            business_id,
        )

        interaction_nodes[
            pair
        ][
            node_type
        ].append(
            str(
                node_id
            )
        )

    if invalid_content_nodes:

        example = invalid_content_nodes[
            :5
        ]

        raise RuntimeError(
            "Prepared Yelp graph contains Review/Tip nodes "
            "without exactly one User and one Business endpoint.\n"
            f"Invalid nodes: {len(invalid_content_nodes):,}\n"
            f"Examples: {example}"
        )

    interactions = set(
        interaction_nodes.keys()
    )

    user_businesses = defaultdict(
        set
    )

    business_users = defaultdict(
        set
    )

    evidence_counts = Counter()

    for (
        user_id,
        business_id,
    ), evidence in interaction_nodes.items():

        user_businesses[
            user_id
        ].add(
            business_id
        )

        business_users[
            business_id
        ].add(
            user_id
        )

        has_review = bool(
            evidence[
                "review"
            ]
        )

        has_tip = bool(
            evidence[
                "tip"
            ]
        )

        if (
            has_review
            and has_tip
        ):

            evidence_type = (
                "review_and_tip"
            )

        elif has_review:

            evidence_type = (
                "review_only"
            )

        elif has_tip:

            evidence_type = (
                "tip_only"
            )

        else:

            raise RuntimeError(
                "Interaction has no Review or Tip evidence."
            )

        evidence_counts[
            evidence_type
        ] += 1

    print(
        f"  review nodes: "
        f"{review_nodes:,}"
    )

    print(
        f"  tip nodes: "
        f"{tip_nodes:,}"
    )

    print(
        f"  unique User-Business interactions: "
        f"{len(interactions):,}"
    )

    print(
        f"  users with interactions: "
        f"{len(user_businesses):,}"
    )

    print(
        f"  businesses with interactions: "
        f"{len(business_users):,}"
    )

    print(
        f"  review-only pairs: "
        f"{evidence_counts['review_only']:,}"
    )

    print(
        f"  tip-only pairs: "
        f"{evidence_counts['tip_only']:,}"
    )

    print(
        f"  review+tip pairs: "
        f"{evidence_counts['review_and_tip']:,}"
    )

    return {
        "interaction_nodes": (
            interaction_nodes
        ),

        "interactions": (
            interactions
        ),

        "user_businesses": (
            user_businesses
        ),

        "business_users": (
            business_users
        ),

        "evidence_counts": (
            evidence_counts
        ),
    }


def sample_holdout_positives(
    index,
    requested,
    seed,
):

    requested = int(
        requested
    )

    user_businesses = index[
        "user_businesses"
    ]

    business_users = index[
        "business_users"
    ]

    candidates = []

    for pair in sorted(
        index[
            "interactions"
        ]
    ):

        user_id, business_id = pair

        if len(
            user_businesses[
                user_id
            ]
        ) < 2:

            continue

        if len(
            business_users[
                business_id
            ]
        ) < 2:

            continue

        candidates.append(
            pair
        )

    rng = np.random.default_rng(
        seed
    )

    permutation = rng.permutation(
        len(
            candidates
        )
    )

    selected = []

    removed_user_count = Counter()

    removed_business_count = Counter()

    for candidate_index in permutation:

        user_id, business_id = candidates[
            int(
                candidate_index
            )
        ]

        user_degree = len(
            user_businesses[
                user_id
            ]
        )

        business_degree = len(
            business_users[
                business_id
            ]
        )


        if (
            removed_user_count[
                user_id
            ]
            >=
            user_degree - 1
        ):

            continue

        if (
            removed_business_count[
                business_id
            ]
            >=
            business_degree - 1
        ):

            continue

        selected.append(
            (
                user_id,
                business_id,
            )
        )

        removed_user_count[
            user_id
        ] += 1

        removed_business_count[
            business_id
        ] += 1

        if len(
            selected
        ) >= requested:

            break

    if len(
        selected
    ) < requested:

        raise RuntimeError(
            "Not enough leakage-safe Yelp positives "
            "for the requested common test size.\n"
            f"  requested={requested:,}\n"
            f"  selected={len(selected):,}\n"
            f"  basic candidates={len(candidates):,}"
        )

    return {
        "positive_pairs": (
            selected
        ),

        "basic_candidate_count": int(
            len(
                candidates
            )
        ),

        "removed_user_count": dict(
            removed_user_count
        ),

        "removed_business_count": dict(
            removed_business_count
        ),
    }


def build_training_interactions(
    full_interactions,
    positive_pairs,
):

    positive_set = set(
        positive_pairs
    )

    training_pairs = (
        set(
            full_interactions
        )
        -
        positive_set
    )

    training_user_businesses = defaultdict(
        set
    )

    training_business_users = defaultdict(
        set
    )

    for user_id, business_id in training_pairs:

        training_user_businesses[
            user_id
        ].add(
            business_id
        )

        training_business_users[
            business_id
        ].add(
            user_id
        )

    return {
        "pairs": (
            training_pairs
        ),

        "user_businesses": (
            training_user_businesses
        ),

        "business_users": (
            training_business_users
        ),
    }


def build_business_degree_index(
    training_business_users,
):

    degree_by_business = {}

    businesses_by_degree = defaultdict(
        list
    )

    for business_id, users in (
        training_business_users.items()
    ):

        degree = int(
            len(
                users
            )
        )

        if degree <= 0:

            continue

        business_id = str(
            business_id
        )

        degree_by_business[
            business_id
        ] = degree

        businesses_by_degree[
            degree
        ].append(
            business_id
        )

    for degree in businesses_by_degree:

        businesses_by_degree[
            degree
        ] = sorted(
            businesses_by_degree[
                degree
            ]
        )

    available_degrees = sorted(
        businesses_by_degree.keys()
    )

    return {
        "degree_by_business": (
            degree_by_business
        ),

        "businesses_by_degree": (
            businesses_by_degree
        ),

        "available_degrees": (
            available_degrees
        ),
    }


def ordered_degree_candidates(
    positive_degree,
    available_degrees,
):

    positive_log_degree = float(
        math.log1p(
            positive_degree
        )
    )

    return sorted(
        available_degrees,
        key=lambda candidate_degree: (
            abs(
                math.log1p(
                    candidate_degree
                )
                -
                positive_log_degree
            ),

            abs(
                candidate_degree
                -
                positive_degree
            ),

            candidate_degree,
        ),
    )


def sample_degree_matched_negatives(
    positive_pairs,
    full_interactions,
    training_business_users,
    seed,
):


    full_positive_set = set(
        full_interactions
    )

    degree_index = build_business_degree_index(
        training_business_users
    )

    degree_by_business = degree_index[
        "degree_by_business"
    ]

    businesses_by_degree = degree_index[
        "businesses_by_degree"
    ]

    available_degrees = degree_index[
        "available_degrees"
    ]

    if not available_degrees:

        raise RuntimeError(
            "No Yelp businesses with training support."
        )

    rng = np.random.default_rng(
        seed
    )

    negatives = []

    used_negative_pairs = set()

    diagnostics = []

    for pair_index, (
        user_id,
        positive_business,
    ) in enumerate(
        positive_pairs
    ):

        positive_business = str(
            positive_business
        )

        positive_degree = int(
            degree_by_business.get(
                positive_business,
                0,
            )
        )

        if positive_degree <= 0:

            raise RuntimeError(
                "Held-out positive business lost all "
                "training support despite holdout guard."
            )

        ordered_degrees = (
            ordered_degree_candidates(
                positive_degree,
                available_degrees,
            )
        )

        selected_business = None

        selected_degree = None

        selected_log_distance = None


        for candidate_degree in ordered_degrees:

            candidates = businesses_by_degree[
                candidate_degree
            ]

            valid_candidates = []

            for candidate_business in candidates:

                candidate_pair = canonical_interaction(
                    user_id,
                    candidate_business,
                )


                if candidate_pair in full_positive_set:

                    continue

                if candidate_pair in used_negative_pairs:

                    continue

                valid_candidates.append(
                    candidate_business
                )

            if not valid_candidates:

                continue

            selected_business = str(
                valid_candidates[
                    int(
                        rng.integers(
                            0,
                            len(
                                valid_candidates
                            ),
                        )
                    )
                ]
            )

            selected_degree = int(
                candidate_degree
            )

            selected_log_distance = float(
                abs(
                    math.log1p(
                        positive_degree
                    )
                    -
                    math.log1p(
                        selected_degree
                    )
                )
            )

            break

        if selected_business is None:

            raise RuntimeError(
                "Could not construct a popularity-matched "
                f"negative for positive pair {pair_index}."
            )

        negative_pair = canonical_interaction(
            user_id,
            selected_business,
        )

        used_negative_pairs.add(
            negative_pair
        )

        negatives.append(
            negative_pair
        )

        absolute_degree_difference = int(
            abs(
                positive_degree
                -
                selected_degree
            )
        )

        relative_degree_difference = float(
            absolute_degree_difference
            /
            max(
                positive_degree,
                1,
            )
        )

        diagnostics.append({
            "pair_index": int(
                pair_index
            ),

            "user": str(
                user_id
            ),

            "positive_business": (
                positive_business
            ),

            "negative_business": (
                selected_business
            ),

            "positive_business_train_degree": int(
                positive_degree
            ),

            "negative_business_train_degree": int(
                selected_degree
            ),

            "exact_degree_match": int(
                positive_degree
                ==
                selected_degree
            ),

            "absolute_degree_difference": int(
                absolute_degree_difference
            ),

            "relative_degree_difference": float(
                relative_degree_difference
            ),

            "log_degree_distance": float(
                selected_log_distance
            ),
        })

    return {
        "negative_pairs": (
            negatives
        ),

        "diagnostics": (
            diagnostics
        ),

        "degree_by_business": (
            degree_by_business
        ),
    }


def capture_node_record(
    graph,
    node_id,
):

    node_attributes = dict(
        graph.nodes[
            node_id
        ]
    )

    edge_records = []

    seen_edges = set()

    for (
        source,
        target,
        key,
        attributes,
    ) in graph.in_edges(
        node_id,
        keys=True,
        data=True,
    ):

        edge_key = (
            str(
                source
            ),

            str(
                target
            ),

            key,
        )

        if edge_key in seen_edges:

            continue

        seen_edges.add(
            edge_key
        )

        edge_records.append(
            (
                source,
                target,
                key,
                dict(
                    attributes
                ),
            )
        )

    for (
        source,
        target,
        key,
        attributes,
    ) in graph.out_edges(
        node_id,
        keys=True,
        data=True,
    ):

        edge_key = (
            str(
                source
            ),

            str(
                target
            ),

            key,
        )

        if edge_key in seen_edges:

            continue

        seen_edges.add(
            edge_key
        )

        edge_records.append(
            (
                source,
                target,
                key,
                dict(
                    attributes
                ),
            )
        )

    return {
        "node_id": (
            node_id
        ),

        "node_attributes": (
            node_attributes
        ),

        "edges": (
            edge_records
        ),
    }


def validate_dedicated_interaction_node(
    graph,
    node_id,
    user_id,
    business_id,
):

    node_type = str(
        graph.nodes[
            node_id
        ].get(
            "type",
            "",
        )
    )

    if node_type not in {
        "review",
        "tip",
    }:

        raise RuntimeError(
            "Unexpected held-out Yelp evidence node type: "
            f"{node_type}"
        )

    neighboring_nodes = set()

    for source, _ in graph.in_edges(
        node_id
    ):

        neighboring_nodes.add(
            str(
                source
            )
        )

    for _, target in graph.out_edges(
        node_id
    ):

        neighboring_nodes.add(
            str(
                target
            )
        )

    expected = {
        str(
            user_id
        ),

        str(
            business_id
        ),
    }

    if not neighboring_nodes.issubset(
        expected
    ):

        raise RuntimeError(
            "Held-out Review/Tip node has unexpected "
            "relationships. Removing it could create "
            "collateral graph changes.\n"
            f"  node={node_id}\n"
            f"  neighbors={sorted(neighboring_nodes)}\n"
            f"  expected subset={sorted(expected)}"
        )


def apply_yelp_holdout(
    graph,
    positive_pairs,
    index,
):

    removed_records = []

    diagnostics = []

    already_removed_nodes = set()

    interaction_nodes = index[
        "interaction_nodes"
    ]

    for pair_index, (
        user_id,
        business_id,
    ) in enumerate(
        positive_pairs
    ):

        evidence = interaction_nodes[
            (
                user_id,
                business_id,
            )
        ]

        review_nodes = list(
            evidence[
                "review"
            ]
        )

        tip_nodes = list(
            evidence[
                "tip"
            ]
        )

        nodes_to_remove = (
            review_nodes
            +
            tip_nodes
        )

        if not nodes_to_remove:

            raise RuntimeError(
                "Selected Yelp positive has no evidence nodes."
            )

        for node_id in nodes_to_remove:

            if node_id in already_removed_nodes:

                raise RuntimeError(
                    "A Yelp interaction evidence node belongs "
                    "to more than one selected User-Business pair."
                )

            validate_dedicated_interaction_node(
                graph,
                node_id,
                user_id,
                business_id,
            )

            record = capture_node_record(
                graph,
                node_id,
            )

            removed_records.append(
                record
            )

            graph.remove_node(
                node_id
            )

            already_removed_nodes.add(
                node_id
            )

        if (
            review_nodes
            and tip_nodes
        ):

            evidence_type = (
                "review_and_tip"
            )

        elif review_nodes:

            evidence_type = (
                "review_only"
            )

        else:

            evidence_type = (
                "tip_only"
            )

        diagnostics.append({
            "pair_index": int(
                pair_index
            ),

            "user": (
                user_id
            ),

            "business": (
                business_id
            ),

            "evidence_type": (
                evidence_type
            ),

            "review_nodes_removed": int(
                len(
                    review_nodes
                )
            ),

            "tip_nodes_removed": int(
                len(
                    tip_nodes
                )
            ),

            "total_content_nodes_removed": int(
                len(
                    nodes_to_remove
                )
            ),
        })

    return {
        "removed_records": (
            removed_records
        ),

        "diagnostics": (
            diagnostics
        ),
    }


def restore_yelp_holdout(
    graph,
    intervention,
):


    for record in intervention[
        "removed_records"
    ]:

        graph.add_node(
            record[
                "node_id"
            ],

            **record[
                "node_attributes"
            ],
        )


    for record in intervention[
        "removed_records"
    ]:

        for (
            source,
            target,
            key,
            attributes,
        ) in record[
            "edges"
        ]:

            graph.add_edge(
                source,
                target,
                key=key,
                **attributes,
            )


def current_interaction_exists(
    graph,
    user_id,
    business_id,
):

    candidate_content = set()


    for _, target, attributes in graph.out_edges(
        user_id,
        data=True,
    ):

        relation = str(
            attributes.get(
                "relation",
                "",
            )
        )

        if relation in {
            "wrote",
            "wrote_tip",
        }:

            candidate_content.add(
                str(
                    target
                )
            )


    for source, _, attributes in graph.in_edges(
        user_id,
        data=True,
    ):

        relation = str(
            attributes.get(
                "relation",
                "",
            )
        )

        if relation in {
            "wrote",
            "wrote_tip",
        }:

            candidate_content.add(
                str(
                    source
                )
            )

    for content_id in candidate_content:

        if content_id not in graph:

            continue

        content_type = str(
            graph.nodes[
                content_id
            ].get(
                "type",
                "",
            )
        )

        if content_type == "review":

            expected_relation = (
                "about"
            )

        elif content_type == "tip":

            expected_relation = (
                "about_tip"
            )

        else:

            continue


        for _, target, attributes in graph.out_edges(
            content_id,
            data=True,
        ):

            if (
                str(
                    target
                )
                ==
                str(
                    business_id
                )
                and
                str(
                    attributes.get(
                        "relation",
                        "",
                    )
                )
                ==
                expected_relation
            ):

                return True


        for source, _, attributes in graph.in_edges(
            content_id,
            data=True,
        ):

            if (
                str(
                    source
                )
                ==
                str(
                    business_id
                )
                and
                str(
                    attributes.get(
                        "relation",
                        "",
                    )
                )
                ==
                expected_relation
            ):

                return True

    return False


def validate_holdout(
    graph,
    positive_pairs,
):

    leaking = []

    for (
        user_id,
        business_id,
    ) in positive_pairs:

        if current_interaction_exists(
            graph,
            user_id,
            business_id,
        ):

            leaking.append(
                (
                    user_id,
                    business_id,
                )
            )

    if leaking:

        raise RuntimeError(
            "Yelp leakage validation failed: "
            f"{len(leaking)} held-out User-Business pairs "
            "still have Review/Tip evidence."
        )

    return {
        "remaining_heldout_interactions": 0,
    }


def required_descriptor_users(
    positive_pairs,
    negative_pairs,
    training_business_users,
):

    required = set()

    candidate_businesses = set()

    for user_id, business_id in (
        list(
            positive_pairs
        )
        +
        list(
            negative_pairs
        )
    ):

        required.add(
            user_id
        )

        candidate_businesses.add(
            business_id
        )


    for business_id in candidate_businesses:

        for user_id in training_business_users.get(
            business_id,
            set(),
        ):

            required.add(
                user_id
            )

    return {
        "users": sorted(
            required
        ),

        "businesses": sorted(
            candidate_businesses
        ),
    }


def build_business_profiles(
    descriptor_matrix,
    descriptor_client_ids,
    businesses,
    training_business_users,
):

    client_index = {
        str(
            client_id
        ): index
        for index, client_id
        in enumerate(
            descriptor_client_ids
        )
    }

    profiles = {}

    support = {}

    missing_training_users = 0

    for business_id in businesses:

        vectors = []

        for user_id in sorted(
            training_business_users.get(
                business_id,
                set(),
            )
        ):

            row_index = client_index.get(
                str(
                    user_id
                )
            )

            if row_index is None:

                missing_training_users += 1

                continue

            vectors.append(
                descriptor_matrix[
                    row_index
                ]
            )

        if not vectors:

            raise RuntimeError(
                "Could not construct business prototype for "
                f"{business_id}: no training user descriptor."
            )

        matrix = np.asarray(
            vectors,
            dtype=float,
        )

        centroid = np.mean(
            matrix,
            axis=0,
        )

        profiles[
            business_id
        ] = centroid

        support[
            business_id
        ] = int(
            len(
                vectors
            )
        )

    return {
        "profiles": (
            profiles
        ),

        "support": (
            support
        ),

        "missing_training_users": int(
            missing_training_users
        ),

        "client_index": (
            client_index
        ),
    }


def score_user_business_pairs(
    descriptor_matrix,
    descriptor_client_ids,
    business_profiles,
    pairs,
):

    client_index = {
        str(
            client_id
        ): index
        for index, client_id
        in enumerate(
            descriptor_client_ids
        )
    }

    scores = []

    for user_id, business_id in pairs:

        row_index = client_index.get(
            str(
                user_id
            )
        )

        if row_index is None:

            raise RuntimeError(
                f"Missing descriptor for Yelp user "
                f"{user_id}."
            )

        profile = business_profiles.get(
            business_id
        )

        if profile is None:

            raise RuntimeError(
                f"Missing training profile for Yelp business "
                f"{business_id}."
            )

        score = cosine_score(
            descriptor_matrix[
                row_index
            ],
            profile,
        )

        scores.append(
            score
        )

    return np.asarray(
        scores,
        dtype=float,
    )


def orient_scores(
    cosine_scores,
    orientation,
):

    cosine_scores = np.asarray(
        cosine_scores,
        dtype=float,
    )

    if orientation == "similarity":

        return cosine_scores

    if orientation == "distance":

        return (
            1.0
            -
            cosine_scores
        )

    raise ValueError(
        f"Unknown orientation: "
        f"{orientation}"
    )


def binary_metrics(
    positive_scores,
    negative_scores,
):

    positive_scores = stabilize_scores(
        positive_scores
    )

    negative_scores = stabilize_scores(
        negative_scores
    )

    labels = np.concatenate(
        [
            np.ones(
                len(
                    positive_scores
                ),
                dtype=int,
            ),

            np.zeros(
                len(
                    negative_scores
                ),
                dtype=int,
            ),
        ]
    )

    scores = np.concatenate(
        [
            positive_scores,
            negative_scores,
        ]
    )

    return {
        "auc": float(
            roc_auc_score(
                labels,
                scores,
            )
        ),

        "ap": float(
            average_precision_score(
                labels,
                scores,
            )
        ),

        "positive_score_mean": float(
            np.mean(
                positive_scores
            )
        ),

        "positive_score_median": float(
            np.median(
                positive_scores
            )
        ),

        "negative_score_mean": float(
            np.mean(
                negative_scores
            )
        ),

        "negative_score_median": float(
            np.median(
                negative_scores
            )
        ),

        "score_mean_gap": float(
            np.mean(
                positive_scores
            )
            -
            np.mean(
                negative_scores
            )
        ),

        "positive_unique_stabilized": int(
            len(
                np.unique(
                    positive_scores
                )
            )
        ),

        "negative_unique_stabilized": int(
            len(
                np.unique(
                    negative_scores
                )
            )
        ),

        "all_unique_stabilized": int(
            len(
                np.unique(
                    scores
                )
            )
        ),
    }


def paired_bootstrap_metrics(
    positive_scores,
    negative_scores,
    repetitions,
    seed,
):

    positive_scores = np.asarray(
        positive_scores,
        dtype=float,
    )

    negative_scores = np.asarray(
        negative_scores,
        dtype=float,
    )

    if (
        len(
            positive_scores
        )
        !=
        len(
            negative_scores
        )
    ):

        raise RuntimeError(
            "Paired Yelp bootstrap requires the same number "
            "of positive and negative pairs."
        )

    n = len(
        positive_scores
    )

    rng = np.random.default_rng(
        seed
    )

    auc_values = []

    ap_values = []

    for _ in range(
        int(
            repetitions
        )
    ):

        indices = rng.integers(
            0,
            n,
            size=n,
        )

        result = binary_metrics(
            positive_scores[
                indices
            ],
            negative_scores[
                indices
            ],
        )

        auc_values.append(
            result[
                "auc"
            ]
        )

        ap_values.append(
            result[
                "ap"
            ]
        )

    return {
        "auc_ci95_low": float(
            np.percentile(
                auc_values,
                2.5,
            )
        ),

        "auc_ci95_high": float(
            np.percentile(
                auc_values,
                97.5,
            )
        ),

        "ap_ci95_low": float(
            np.percentile(
                ap_values,
                2.5,
            )
        ),

        "ap_ci95_high": float(
            np.percentile(
                ap_values,
                97.5,
            )
        ),
    }


def evaluate_popularity_baseline(
    positive_pairs,
    negative_pairs,
    degree_by_business,
):

    positive_scores = np.asarray(
        [
            math.log1p(
                degree_by_business[
                    business_id
                ]
            )
            for _, business_id
            in positive_pairs
        ],
        dtype=float,
    )

    negative_scores = np.asarray(
        [
            math.log1p(
                degree_by_business[
                    business_id
                ]
            )
            for _, business_id
            in negative_pairs
        ],
        dtype=float,
    )

    return (
        positive_scores,
        negative_scores,
    )


def run_one_seed(
    graph,
    index,
    seed,
    test_positives,
    bootstrap_repetitions,
    d_max,
    degree_weight,
    seed_folder,
):

    start = time.perf_counter()

    print()

    print(
        "#" * 80
    )

    print(
        f"Yelp seed: {seed}"
    )

    print(
        "#" * 80
    )


    positive_selection = sample_holdout_positives(
        index=index,

        requested=test_positives,

        seed=deterministic_seed(
            "yelp",
            seed,
            "positive_sample",
        ),
    )

    positive_pairs = positive_selection[
        "positive_pairs"
    ]

    print(
        f"Basic eligible positive interactions: "
        f"{positive_selection['basic_candidate_count']:,}"
    )

    print(
        f"Test positives: "
        f"{len(positive_pairs):,}"
    )


    training = build_training_interactions(
        full_interactions=index[
            "interactions"
        ],

        positive_pairs=(
            positive_pairs
        ),
    )

    training_user_businesses = training[
        "user_businesses"
    ]

    training_business_users = training[
        "business_users"
    ]

    unsupported_users = [
        user_id
        for user_id, _
        in positive_pairs
        if len(
            training_user_businesses.get(
                user_id,
                set(),
            )
        ) == 0
    ]

    unsupported_businesses = [
        business_id
        for _, business_id
        in positive_pairs
        if len(
            training_business_users.get(
                business_id,
                set(),
            )
        ) == 0
    ]

    if unsupported_users:

        raise RuntimeError(
            "Positive holdout isolated "
            f"{len(unsupported_users)} Yelp users."
        )

    if unsupported_businesses:

        raise RuntimeError(
            "Positive holdout isolated "
            f"{len(unsupported_businesses)} Yelp businesses."
        )

    print(
        "Held-out users without training support: 0"
    )

    print(
        "Held-out businesses without training support: 0"
    )


    negative_result = sample_degree_matched_negatives(
        positive_pairs=positive_pairs,

        full_interactions=index[
            "interactions"
        ],

        training_business_users=(
            training_business_users
        ),

        seed=deterministic_seed(
            "yelp",
            seed,
            "negative_sample",
        ),
    )

    negative_pairs = negative_result[
        "negative_pairs"
    ]

    negative_diagnostics = negative_result[
        "diagnostics"
    ]

    degree_by_business = negative_result[
        "degree_by_business"
    ]

    print(
        f"Test negatives: "
        f"{len(negative_pairs):,}"
    )

    exact_degree_matches = sum(
        1
        for row
        in negative_diagnostics
        if row[
            "exact_degree_match"
        ] == 1
    )

    mean_degree_difference = float(
        np.mean(
            [
                row[
                    "absolute_degree_difference"
                ]
                for row
                in negative_diagnostics
            ]
        )
    )

    mean_relative_degree_difference = float(
        np.mean(
            [
                row[
                    "relative_degree_difference"
                ]
                for row
                in negative_diagnostics
            ]
        )
    )

    mean_log_degree_distance = float(
        np.mean(
            [
                row[
                    "log_degree_distance"
                ]
                for row
                in negative_diagnostics
            ]
        )
    )

    max_degree_difference = int(
        max(
            row[
                "absolute_degree_difference"
            ]
            for row
            in negative_diagnostics
        )
    )

    print(
        "Negative business exact degree matches: "
        f"{exact_degree_matches:,}/"
        f"{len(negative_diagnostics):,}"
    )

    print(
        "Mean absolute business-degree difference: "
        f"{mean_degree_difference:.6f}"
    )

    print(
        "Mean relative business-degree difference: "
        f"{mean_relative_degree_difference:.6f}"
    )

    print(
        "Mean log1p business-degree distance: "
        f"{mean_log_degree_distance:.9f}"
    )

    print(
        "Maximum absolute business-degree difference: "
        f"{max_degree_difference:,}"
    )


    intervention = None

    try:

        intervention = apply_yelp_holdout(
            graph=graph,

            positive_pairs=(
                positive_pairs
            ),

            index=index,
        )

        leakage = validate_holdout(
            graph,
            positive_pairs,
        )

        print(
            "Held-out interactions still represented "
            "by Review/Tip paths: "
            f"{leakage['remaining_heldout_interactions']}"
        )

        removed_content_nodes = len(
            intervention[
                "removed_records"
            ]
        )

        removed_reviews = sum(
            row[
                "review_nodes_removed"
            ]
            for row
            in intervention[
                "diagnostics"
            ]
        )

        removed_tips = sum(
            row[
                "tip_nodes_removed"
            ]
            for row
            in intervention[
                "diagnostics"
            ]
        )

        print(
            f"Removed Review nodes: "
            f"{removed_reviews:,}"
        )

        print(
            f"Removed Tip nodes: "
            f"{removed_tips:,}"
        )

        print(
            f"Total interaction content nodes removed: "
            f"{removed_content_nodes:,}"
        )


        requirements = required_descriptor_users(
            positive_pairs=positive_pairs,

            negative_pairs=negative_pairs,

            training_business_users=(
                training_business_users
            ),
        )

        required_users = requirements[
            "users"
        ]

        candidate_businesses = requirements[
            "businesses"
        ]

        print(
            f"Candidate businesses requiring profiles: "
            f"{len(candidate_businesses):,}"
        )

        print(
            f"Training/query users requiring descriptors: "
            f"{len(required_users):,}"
        )


        generated = generate_selected_descriptors(
            training_graph=graph,

            client_ids=(
                required_users
            ),

            d_max=d_max,

            degree_weight=(
                degree_weight
            ),
        )

        descriptor_client_ids = generated[
            "client_ids"
        ]

        descriptor_matrices = generated[
            "descriptors"
        ]

        failures = generated[
            "failures"
        ]

        if failures:

            raise RuntimeError(
                f"Descriptor generation failed for "
                f"{len(failures)} Yelp users."
            )

        if (
            len(
                descriptor_client_ids
            )
            !=
            len(
                required_users
            )
        ):

            raise RuntimeError(
                "Not all required Yelp users received "
                "a descriptor."
            )


        pair_rows = []

        intervention_by_pair = {
            (
                row[
                    "user"
                ],
                row[
                    "business"
                ],
            ): row
            for row
            in intervention[
                "diagnostics"
            ]
        }

        negative_by_index = {
            int(
                row[
                    "pair_index"
                ]
            ): row
            for row
            in negative_diagnostics
        }

        for pair_index, (
            user_id,
            business_id,
        ) in enumerate(
            positive_pairs
        ):

            intervention_row = intervention_by_pair[
                (
                    user_id,
                    business_id,
                )
            ]

            pair_rows.append({
                "pair_index": int(
                    pair_index
                ),

                "pair_type": (
                    "positive"
                ),

                "user": (
                    user_id
                ),

                "business": (
                    business_id
                ),

                "evidence_type": intervention_row[
                    "evidence_type"
                ],

                "business_train_degree": int(
                    degree_by_business[
                        business_id
                    ]
                ),

                "matched_positive_business": (
                    business_id
                ),

                "degree_match_exact": 1,

                "absolute_degree_difference": 0,

                "relative_degree_difference": 0.0,

                "log_degree_distance": 0.0,
            })

        for pair_index, (
            user_id,
            business_id,
        ) in enumerate(
            negative_pairs
        ):

            negative_row = negative_by_index[
                pair_index
            ]

            pair_rows.append({
                "pair_index": int(
                    pair_index
                ),

                "pair_type": (
                    "negative"
                ),

                "user": (
                    user_id
                ),

                "business": (
                    business_id
                ),

                "evidence_type": (
                    "none"
                ),

                "business_train_degree": int(
                    degree_by_business[
                        business_id
                    ]
                ),

                "matched_positive_business": (
                    negative_row[
                        "positive_business"
                    ]
                ),

                "degree_match_exact": int(
                    negative_row[
                        "exact_degree_match"
                    ]
                ),

                "absolute_degree_difference": int(
                    negative_row[
                        "absolute_degree_difference"
                    ]
                ),

                "relative_degree_difference": float(
                    negative_row[
                        "relative_degree_difference"
                    ]
                ),

                "log_degree_distance": float(
                    negative_row[
                        "log_degree_distance"
                    ]
                ),
            })

        pair_dataframe = pd.DataFrame(
            pair_rows
        )

        metric_rows = []

        business_profile_rows = []


        for descriptor in DESCRIPTORS:

            descriptor_matrix = descriptor_matrices[
                descriptor
            ]

            profile_result = build_business_profiles(
                descriptor_matrix=(
                    descriptor_matrix
                ),

                descriptor_client_ids=(
                    descriptor_client_ids
                ),

                businesses=(
                    candidate_businesses
                ),

                training_business_users=(
                    training_business_users
                ),
            )

            profiles = profile_result[
                "profiles"
            ]

            support = profile_result[
                "support"
            ]

            if (
                profile_result[
                    "missing_training_users"
                ]
                != 0
            ):

                raise RuntimeError(
                    "Business profile construction encountered "
                    "training users without descriptors."
                )

            positive_cosine = score_user_business_pairs(
                descriptor_matrix=(
                    descriptor_matrix
                ),

                descriptor_client_ids=(
                    descriptor_client_ids
                ),

                business_profiles=(
                    profiles
                ),

                pairs=(
                    positive_pairs
                ),
            )

            negative_cosine = score_user_business_pairs(
                descriptor_matrix=(
                    descriptor_matrix
                ),

                descriptor_client_ids=(
                    descriptor_client_ids
                ),

                business_profiles=(
                    profiles
                ),

                pairs=(
                    negative_pairs
                ),
            )


            pair_dataframe[
                descriptor
            ] = np.concatenate(
                [
                    positive_cosine,
                    negative_cosine,
                ]
            )

            for business_id in candidate_businesses:

                business_profile_rows.append({
                    "descriptor": (
                        descriptor
                    ),

                    "business": (
                        business_id
                    ),

                    "training_user_support": int(
                        support[
                            business_id
                        ]
                    ),
                })

            for orientation in ORIENTATIONS:

                positive_scores = orient_scores(
                    positive_cosine,
                    orientation,
                )

                negative_scores = orient_scores(
                    negative_cosine,
                    orientation,
                )

                metrics = binary_metrics(
                    positive_scores,
                    negative_scores,
                )

                intervals = paired_bootstrap_metrics(
                    positive_scores,
                    negative_scores,

                    repetitions=(
                        bootstrap_repetitions
                    ),

                    seed=deterministic_seed(
                        "yelp",
                        seed,
                        descriptor,
                        orientation,
                        "paired_bootstrap",
                    ),
                )

                metric_rows.append({
                    "seed": int(
                        seed
                    ),

                    "method": (
                        descriptor
                    ),

                    "orientation": (
                        orientation
                    ),

                    "score_round_decimals": int(
                        SCORE_ROUND_DECIMALS
                    ),

                    "positive_pairs": int(
                        len(
                            positive_pairs
                        )
                    ),

                    "negative_pairs": int(
                        len(
                            negative_pairs
                        )
                    ),

                    **metrics,
                    **intervals,
                })


        (
            positive_popularity,
            negative_popularity,
        ) = evaluate_popularity_baseline(
            positive_pairs,
            negative_pairs,
            degree_by_business,
        )

        pair_dataframe[
            "business_popularity_score"
        ] = np.concatenate(
            [
                positive_popularity,
                negative_popularity,
            ]
        )

        popularity_metrics = binary_metrics(
            positive_popularity,
            negative_popularity,
        )

        popularity_intervals = (
            paired_bootstrap_metrics(
                positive_popularity,
                negative_popularity,

                repetitions=(
                    bootstrap_repetitions
                ),

                seed=deterministic_seed(
                    "yelp",
                    seed,
                    "popularity",
                    "paired_bootstrap",
                ),
            )
        )

        metric_rows.append({
            "seed": int(
                seed
            ),

            "method": (
                "business_popularity"
            ),

            "orientation": (
                "higher_degree"
            ),

            "score_round_decimals": int(
                SCORE_ROUND_DECIMALS
            ),

            "positive_pairs": int(
                len(
                    positive_pairs
                )
            ),

            "negative_pairs": int(
                len(
                    negative_pairs
                )
            ),

            **popularity_metrics,
            **popularity_intervals,
        })

        metrics_dataframe = pd.DataFrame(
            metric_rows
        )

        business_profile_dataframe = pd.DataFrame(
            business_profile_rows
        )


        orientation_rows = []

        for descriptor in DESCRIPTORS:

            subset = metrics_dataframe[
                metrics_dataframe[
                    "method"
                ]
                ==
                descriptor
            ]

            similarity_row = subset[
                subset[
                    "orientation"
                ]
                ==
                "similarity"
            ].iloc[
                0
            ]

            distance_row = subset[
                subset[
                    "orientation"
                ]
                ==
                "distance"
            ].iloc[
                0
            ]

            similarity_auc = float(
                similarity_row[
                    "auc"
                ]
            )

            distance_auc = float(
                distance_row[
                    "auc"
                ]
            )

            similarity_ap = float(
                similarity_row[
                    "ap"
                ]
            )

            distance_ap = float(
                distance_row[
                    "ap"
                ]
            )

            if similarity_auc > distance_auc:

                auc_preference = (
                    "similarity"
                )

            elif distance_auc > similarity_auc:

                auc_preference = (
                    "distance"
                )

            else:

                auc_preference = (
                    "tie"
                )

            if similarity_ap > distance_ap:

                ap_preference = (
                    "similarity"
                )

            elif distance_ap > similarity_ap:

                ap_preference = (
                    "distance"
                )

            else:

                ap_preference = (
                    "tie"
                )

            orientation_rows.append({
                "seed": int(
                    seed
                ),

                "descriptor": (
                    descriptor
                ),

                "similarity_auc": (
                    similarity_auc
                ),

                "distance_auc": (
                    distance_auc
                ),

                "auc_preferred_orientation": (
                    auc_preference
                ),

                "similarity_ap": (
                    similarity_ap
                ),

                "distance_ap": (
                    distance_ap
                ),

                "ap_preferred_orientation": (
                    ap_preference
                ),
            })

        orientation_dataframe = pd.DataFrame(
            orientation_rows
        )


        os.makedirs(
            seed_folder,
            exist_ok=True,
        )

        metrics_path = os.path.join(
            seed_folder,
            "metrics.csv",
        )

        scores_path = os.path.join(
            seed_folder,
            "pairs_and_scores.csv.gz",
        )

        leakage_path = os.path.join(
            seed_folder,
            "holdout_diagnostics.csv",
        )

        negative_path = os.path.join(
            seed_folder,
            "negative_matching.csv",
        )

        profile_path = os.path.join(
            seed_folder,
            "business_profile_support.csv",
        )

        orientation_path = os.path.join(
            seed_folder,
            "orientation_diagnostic.csv",
        )

        metadata_path = os.path.join(
            seed_folder,
            "metadata.json",
        )

        metrics_dataframe.to_csv(
            metrics_path,
            index=False,
        )

        pair_dataframe.to_csv(
            scores_path,
            index=False,
            compression="gzip",
        )

        pd.DataFrame(
            intervention[
                "diagnostics"
            ]
        ).to_csv(
            leakage_path,
            index=False,
        )

        pd.DataFrame(
            negative_diagnostics
        ).to_csv(
            negative_path,
            index=False,
        )

        business_profile_dataframe.to_csv(
            profile_path,
            index=False,
        )

        orientation_dataframe.to_csv(
            orientation_path,
            index=False,
        )


        runtime = float(
            time.perf_counter()
            -
            start
        )

        metadata = {
            "schema_version": (
                SCHEMA_VERSION
            ),

            "dataset": (
                "yelp"
            ),

            "seed": int(
                seed
            ),

            "full_interactions": int(
                len(
                    index[
                        "interactions"
                    ]
                )
            ),

            "eligible_positive_interactions": int(
                positive_selection[
                    "basic_candidate_count"
                ]
            ),

            "test_positive_pairs": int(
                len(
                    positive_pairs
                )
            ),

            "test_negative_pairs": int(
                len(
                    negative_pairs
                )
            ),

            "removed_review_nodes": int(
                removed_reviews
            ),

            "removed_tip_nodes": int(
                removed_tips
            ),

            "removed_content_nodes": int(
                removed_content_nodes
            ),

            "remaining_heldout_interactions": 0,

            "candidate_businesses": int(
                len(
                    candidate_businesses
                )
            ),

            "required_user_descriptors": int(
                len(
                    required_users
                )
            ),

            "descriptor_failures": (
                failures
            ),

            "negative_exact_degree_matches": int(
                exact_degree_matches
            ),

            "negative_total": int(
                len(
                    negative_diagnostics
                )
            ),

            "negative_exact_degree_match_fraction": float(
                exact_degree_matches
                /
                max(
                    len(
                        negative_diagnostics
                    ),
                    1,
                )
            ),

            "negative_mean_absolute_degree_difference": float(
                mean_degree_difference
            ),

            "negative_mean_relative_degree_difference": float(
                mean_relative_degree_difference
            ),

            "negative_mean_log_degree_distance": float(
                mean_log_degree_distance
            ),

            "negative_max_absolute_degree_difference": int(
                max_degree_difference
            ),

            "score_round_decimals": int(
                SCORE_ROUND_DECIMALS
            ),

            "cross_role_scoring": (
                "Business profile is the mean descriptor of "
                "training users interacting with that business. "
                "User-Business score is cosine(query user descriptor, "
                "business training-user centroid)."
            ),

            "negative_sampling": (
                "Each positive and negative share the same user. "
                "The negative business is absent from the complete "
                "original interaction relation and is selected to "
                "minimize absolute log1p training-degree difference "
                "from the corresponding positive business. Exact "
                "training-degree matches are preferred."
            ),

            "holdout_policy": (
                "All Review and Tip content nodes connecting each "
                "held-out User-Business pair are removed before "
                "descriptor generation."
            ),

            "numerical_stabilization": (
                f"Scores used for ranking metrics are rounded to "
                f"{SCORE_ROUND_DECIMALS} decimal places. Raw spectral "
                "cosine scores remain saved in pairs_and_scores.csv.gz."
            ),

            "privacy": (
                "NONE. This is a clean representation evaluation."
            ),

            "runtime_seconds": (
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


        print()

        print(
            "Descriptor results:"
        )

        display_columns = [
            "method",
            "orientation",
            "auc",
            "auc_ci95_low",
            "auc_ci95_high",
            "ap",
            "ap_ci95_low",
            "ap_ci95_high",
            "positive_score_mean",
            "negative_score_mean",
            "all_unique_stabilized",
        ]

        print(
            metrics_dataframe[
                display_columns
            ].to_string(
                index=False
            )
        )

        print()

        print(
            "Orientation diagnostic:"
        )

        print(
            orientation_dataframe.to_string(
                index=False
            )
        )

        print()

        print(
            f"Seed runtime: "
            f"{runtime:.3f} s"
        )

        return {
            "metrics": (
                metrics_dataframe
            ),

            "orientation": (
                orientation_dataframe
            ),

            "metadata": (
                metadata
            ),
        }

    finally:

        if intervention is not None:

            restore_yelp_holdout(
                graph,
                intervention,
            )


def aggregate_metrics(
    all_metrics,
):

    rows = []

    grouped = all_metrics.groupby(
        [
            "method",
            "orientation",
        ],
        sort=True,
    )

    metric_names = (
        "auc",
        "ap",
        "positive_score_mean",
        "negative_score_mean",
        "score_mean_gap",
        "all_unique_stabilized",
    )

    for (
        method,
        orientation,
    ), group in grouped:

        row = {
            "method": (
                method
            ),

            "orientation": (
                orientation
            ),

            "split_seeds": int(
                group[
                    "seed"
                ].nunique()
            ),
        }

        for metric in metric_names:

            statistics = mean_std_ci95(
                group[
                    metric
                ].to_numpy(
                    dtype=float
                )
            )

            row[
                f"{metric}__mean"
            ] = statistics[
                "mean"
            ]

            row[
                f"{metric}__std"
            ] = statistics[
                "std"
            ]

            row[
                f"{metric}__ci95_low"
            ] = statistics[
                "ci95_low"
            ]

            row[
                f"{metric}__ci95_high"
            ] = statistics[
                "ci95_high"
            ]

        rows.append(
            row
        )

    dataframe = pd.DataFrame(
        rows
    )

    if len(
        dataframe
    ):

        dataframe = (
            dataframe.sort_values(
                [
                    "auc__mean",
                    "ap__mean",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .reset_index(
                drop=True
            )
        )

    return dataframe


def aggregate_orientation(
    all_orientation,
):

    rows = []

    grouped = all_orientation.groupby(
        "descriptor",
        sort=True,
    )

    for descriptor, group in grouped:

        similarity_auc = group[
            "similarity_auc"
        ].to_numpy(
            dtype=float
        )

        distance_auc = group[
            "distance_auc"
        ].to_numpy(
            dtype=float
        )

        similarity_ap = group[
            "similarity_ap"
        ].to_numpy(
            dtype=float
        )

        distance_ap = group[
            "distance_ap"
        ].to_numpy(
            dtype=float
        )

        rows.append({
            "descriptor": (
                descriptor
            ),

            "split_seeds": int(
                len(
                    group
                )
            ),

            "mean_similarity_auc": float(
                np.mean(
                    similarity_auc
                )
            ),

            "mean_distance_auc": float(
                np.mean(
                    distance_auc
                )
            ),

            "auc_similarity_wins": int(
                np.sum(
                    similarity_auc
                    >
                    distance_auc
                )
            ),

            "auc_distance_wins": int(
                np.sum(
                    distance_auc
                    >
                    similarity_auc
                )
            ),

            "auc_ties": int(
                np.sum(
                    similarity_auc
                    ==
                    distance_auc
                )
            ),

            "mean_similarity_ap": float(
                np.mean(
                    similarity_ap
                )
            ),

            "mean_distance_ap": float(
                np.mean(
                    distance_ap
                )
            ),

            "ap_similarity_wins": int(
                np.sum(
                    similarity_ap
                    >
                    distance_ap
                )
            ),

            "ap_distance_wins": int(
                np.sum(
                    distance_ap
                    >
                    similarity_ap
                )
            ),

            "ap_ties": int(
                np.sum(
                    similarity_ap
                    ==
                    distance_ap
                )
            ),
        })

    return pd.DataFrame(
        rows
    )


def run_yelp_special_link_prediction(
    output_folder=DEFAULT_OUTPUT_FOLDER,
    seeds=DEFAULT_SEEDS,
    test_positives=DEFAULT_TEST_POSITIVES,
    bootstrap_repetitions=DEFAULT_BOOTSTRAP_REPETITIONS,
    d_max=DEFAULT_D_MAX,
    degree_weight=DEFAULT_DEGREE_WEIGHT,
):

    total_start = time.perf_counter()

    seeds = tuple(
        int(
            seed
        )
        for seed in seeds
    )

    graph_path = os.path.join(
        output_folder,
        "yelp",
        "yelp_global_graph.gml.gz",
    )

    if not os.path.exists(
        graph_path
    ):

        raise FileNotFoundError(
            "Missing prepared Yelp graph:\n"
            f"  {graph_path}\n"
            "Run prepare_yelp_special.py first."
        )

    experiment_folder = os.path.join(
        output_folder,
        "yelp",
        "experiments",
        "yelp_link_prediction_special",
    )

    os.makedirs(
        experiment_folder,
        exist_ok=True,
    )

    print()

    print(
        "=" * 80
    )

    print(
        "YELP SPECIAL LEAKAGE-SAFE USER-BUSINESS PREDICTION"
    )

    print(
        "=" * 80
    )

    print(
        f"Seeds: "
        f"{list(seeds)}"
    )

    print(
        f"Requested test positives per run: "
        f"{test_positives:,}"
    )

    print(
        f"Score stabilization: "
        f"{SCORE_ROUND_DECIMALS} decimals"
    )

    print(
        "Negative matching: "
        "nearest actual training business degree"
    )

    print()

    print(
        "Loading prepared Yelp graph..."
    )

    graph = nx.read_gml(
        graph_path
    )

    print(
        f"Graph nodes: "
        f"{graph.number_of_nodes():,}"
    )

    print(
        f"Graph edges: "
        f"{graph.number_of_edges():,}"
    )


    index = build_yelp_interaction_index(
        graph
    )

    all_metric_frames = []

    all_orientation_frames = []

    all_metadata = []


    for seed_index, seed in enumerate(
        seeds,
        start=1,
    ):

        print()

        print(
            "=" * 80
        )

        print(
            f"SEED {seed_index}/"
            f"{len(seeds)}: "
            f"{seed}"
        )

        print(
            "=" * 80
        )

        seed_folder = os.path.join(
            experiment_folder,
            f"seed_{seed}",
        )

        result = run_one_seed(
            graph=graph,

            index=index,

            seed=seed,

            test_positives=(
                test_positives
            ),

            bootstrap_repetitions=(
                bootstrap_repetitions
            ),

            d_max=d_max,

            degree_weight=(
                degree_weight
            ),

            seed_folder=(
                seed_folder
            ),
        )

        all_metric_frames.append(
            result[
                "metrics"
            ]
        )

        all_orientation_frames.append(
            result[
                "orientation"
            ]
        )

        all_metadata.append(
            result[
                "metadata"
            ]
        )


    all_metrics = pd.concat(
        all_metric_frames,
        ignore_index=True,
    )

    all_orientation = pd.concat(
        all_orientation_frames,
        ignore_index=True,
    )

    aggregate = aggregate_metrics(
        all_metrics
    )

    orientation_summary = aggregate_orientation(
        all_orientation
    )


    seed_metrics_path = os.path.join(
        experiment_folder,
        "yelp_seed_metrics.csv",
    )

    aggregate_path = os.path.join(
        experiment_folder,
        "yelp_aggregate_metrics.csv",
    )

    orientation_path = os.path.join(
        experiment_folder,
        "yelp_orientation_summary.csv",
    )

    metadata_path = os.path.join(
        experiment_folder,
        "yelp_link_prediction_metadata.json",
    )

    all_metrics.to_csv(
        seed_metrics_path,
        index=False,
    )

    aggregate.to_csv(
        aggregate_path,
        index=False,
    )

    orientation_summary.to_csv(
        orientation_path,
        index=False,
    )

    total_runtime = float(
        time.perf_counter()
        -
        total_start
    )

    metadata = {
        "schema_version": (
            SCHEMA_VERSION
        ),

        "dataset": (
            "yelp"
        ),

        "graph": (
            graph_path
        ),

        "seeds": list(
            seeds
        ),

        "test_positives": int(
            test_positives
        ),

        "bootstrap_repetitions": int(
            bootstrap_repetitions
        ),

        "descriptors": list(
            DESCRIPTORS
        ),

        "orientations": list(
            ORIENTATIONS
        ),

        "score_round_decimals": int(
            SCORE_ROUND_DECIMALS
        ),

        "target_relation": (
            "Derived User-Business interaction through Review "
            "and/or Tip evidence."
        ),

        "holdout_policy": (
            "All Review and Tip content nodes for held-out "
            "User-Business pairs are removed before descriptors."
        ),

        "business_representation": (
            "Mean descriptor of remaining training users "
            "interacting with each candidate business."
        ),

        "negative_policy": (
            "Same query user as the positive. Negative business "
            "must be absent from the complete original interaction "
            "relation and is selected by minimizing absolute log1p "
            "training-degree difference from the corresponding "
            "positive business. Exact degree matches are preferred."
        ),

        "popularity_baseline": (
            "log1p training business interaction degree."
        ),

        "privacy": (
            "NONE. Clean representation evaluation."
        ),

        "runs": (
            all_metadata
        ),

        "runtime_seconds": (
            total_runtime
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


    print()

    print(
        "=" * 80
    )

    print(
        "YELP SPECIAL LINK PREDICTION COMPLETE"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Aggregate metrics:"
    )

    display_columns = [
        "method",
        "orientation",
        "auc__mean",
        "auc__std",
        "auc__ci95_low",
        "auc__ci95_high",
        "ap__mean",
        "ap__std",
        "positive_score_mean__mean",
        "negative_score_mean__mean",
        "all_unique_stabilized__mean",
    ]

    print(
        aggregate[
            display_columns
        ].to_string(
            index=False
        )
    )

    print()

    print(
        "Orientation consistency:"
    )

    print(
        orientation_summary.to_string(
            index=False
        )
    )

    print()

    print(
        "INTERPRETATION:"
    )

    print(
        "  1. The target is User-Business interaction, "
        "not an invented Yelp friendship relation."
    )

    print(
        "  2. Every positive interaction is removed "
        "before descriptor generation."
    )

    print(
        "  3. All Review/Tip evidence for a held-out "
        "pair is removed."
    )

    print(
        "  4. Business profiles contain only "
        "training-user descriptor information."
    )

    print(
        "  5. Every positive and its corresponding "
        "negative use the same query user."
    )

    print(
        "  6. Negative businesses are matched using "
        "actual training business degree."
    )

    print(
        "  7. Exact training-degree matches are "
        "preferred whenever available."
    )

    print(
        "  8. Business popularity is evaluated as "
        "an explicit control baseline."
    )

    print(
        "  9. No differential privacy noise is used."
    )

    print()

    print(
        "Outputs:"
    )

    print(
        f"  {seed_metrics_path}"
    )

    print(
        f"  {aggregate_path}"
    )

    print(
        f"  {orientation_path}"
    )

    print(
        f"  {metadata_path}"
    )

    print()

    print(
        f"Total runtime: "
        f"{total_runtime:.3f} s"
    )

    print(
        "=" * 80
    )

    return {
        "seed_metrics": (
            all_metrics
        ),

        "aggregate": (
            aggregate
        ),

        "orientation_summary": (
            orientation_summary
        ),

        "metadata": (
            metadata
        ),
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Leakage-safe Yelp User-Business interaction "
            "prediction using training-only business profiles "
            "and tight business-degree-matched negatives."
        )
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default=(
            DEFAULT_OUTPUT_FOLDER
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
        "--test-positives",
        type=int,
        default=(
            DEFAULT_TEST_POSITIVES
        ),
    )

    parser.add_argument(
        "--bootstrap",
        type=int,
        default=(
            DEFAULT_BOOTSTRAP_REPETITIONS
        ),
    )

    parser.add_argument(
        "--Dmax",
        type=int,
        default=(
            DEFAULT_D_MAX
        ),
    )

    parser.add_argument(
        "--degree-weight",
        type=float,
        default=(
            DEFAULT_DEGREE_WEIGHT
        ),
    )

    args = parser.parse_args()

    run_yelp_special_link_prediction(
        output_folder=(
            args.output_folder
        ),

        seeds=(
            args.seeds
        ),

        test_positives=(
            args.test_positives
        ),

        bootstrap_repetitions=(
            args.bootstrap
        ),

        d_max=(
            args.Dmax
        ),

        degree_weight=(
            args.degree_weight
        ),
    )
