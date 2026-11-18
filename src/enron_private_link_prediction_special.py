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

from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from scipy.stats import (
    pearsonr,
    spearmanr,
    t as student_t,
)

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)

from link_prediction_experiment import (
    generate_selected_descriptors,
)


SCHEMA_VERSION = "enron-private-link-special-2.0"

DEFAULT_OUTPUT_FOLDER = "output_local_test"

TARGET_RELATION = "shared_text"

DEFAULT_REPRESENTATIONS = (
    "hks_d12_global",
    "eigen_d8_block",
    "eigen_d12_global",
)

REPRESENTATION_ORIENTATION = {
    "hks_d12_global": "distance",
    "eigen_d8_block": "similarity",
    "eigen_d12_global": "similarity",
}

DEFAULT_EPSILONS = (
    0.5,
    1.0,
    2.0,
    3.0,
    5.0,
    10.0,
    20.0,
)

DEFAULT_SPLIT_SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

DEFAULT_MECHANISM_SEEDS = (
    0,
    1,
    2,
    3,
    4,
)

DEFAULT_TEST_POSITIVES = 80

DEFAULT_C = 1.0

DEFAULT_RHO_SIGNATURE = 0.5

DEFAULT_SENSITIVITY_MODE = "nonnegative_report"

DEFAULT_D_MAX = 50

DEFAULT_DEGREE_WEIGHT = 1.0

SCORE_ROUND_DECIMALS = 12

NEGATIVE_TOLERANCE = 1e-10


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


def canonical_pair(
    first,
    second,
):

    first = str(first)
    second = str(second)

    if first <= second:

        return (
            first,
            second,
        )

    return (
        second,
        first,
    )


def mean_std_ci95(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    n = len(values)

    if n == 0:

        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
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
        math.sqrt(n)
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
        "n": int(n),
        "mean": mean,
        "std": std,
        "ci95_low": float(
            mean - margin
        ),
        "ci95_high": float(
            mean + margin
        ),
    }


def build_enron_index(
    graph,
):

    print()
    print(
        "Building Enron target/content index..."
    )

    clients = set()
    contents = set()

    for node_id, attributes in graph.nodes(
        data=True
    ):

        node_id = str(node_id)

        role = str(
            attributes.get(
                "role",
                "",
            )
        )

        if role == "client":

            clients.add(
                node_id
            )

        elif role == "content":

            contents.add(
                node_id
            )

    if not clients:

        raise RuntimeError(
            "No role='client' nodes found."
        )

    if not contents:

        raise RuntimeError(
            "No role='content' nodes found."
        )

    target_pairs = set()

    target_edge_multiplicity = Counter()

    target_physical_edges = 0

    client_contents = defaultdict(set)

    content_clients = defaultdict(set)

    relation_counts = Counter()

    for (
        source,
        target,
        key,
        attributes,
    ) in graph.edges(
        keys=True,
        data=True,
    ):

        source = str(source)
        target = str(target)

        relation = str(
            attributes.get(
                "relation",
                "",
            )
        )

        relation_counts[
            relation
        ] += 1

        source_role = str(
            graph.nodes[
                source
            ].get(
                "role",
                "",
            )
        )

        target_role = str(
            graph.nodes[
                target
            ].get(
                "role",
                "",
            )
        )


        if (
            relation == TARGET_RELATION
            and
            source_role == "client"
            and
            target_role == "client"
            and
            source != target
        ):

            pair = canonical_pair(
                source,
                target,
            )

            target_pairs.add(
                pair
            )

            target_edge_multiplicity[
                pair
            ] += 1

            target_physical_edges += 1


        if (
            source_role == "client"
            and
            target_role == "content"
        ):

            client_contents[
                source
            ].add(
                target
            )

            content_clients[
                target
            ].add(
                source
            )

        elif (
            source_role == "content"
            and
            target_role == "client"
        ):

            client_contents[
                target
            ].add(
                source
            )

            content_clients[
                source
            ].add(
                target
            )

    strict_pairs = {}

    common_content_counts = []

    for pair in sorted(
        target_pairs
    ):

        first, second = pair

        common_contents = (
            client_contents.get(
                first,
                set(),
            )
            &
            client_contents.get(
                second,
                set(),
            )
        )

        if not common_contents:

            continue

        strict_pairs[
            pair
        ] = tuple(
            sorted(
                common_contents
            )
        )

        common_content_counts.append(
            len(
                common_contents
            )
        )

    clients = sorted(
        clients
    )

    contents = sorted(
        contents
    )

    print(
        f"  clients: "
        f"{len(clients):,}"
    )

    print(
        f"  content nodes: "
        f"{len(contents):,}"
    )

    print(
        f"  physical {TARGET_RELATION} edges: "
        f"{target_physical_edges:,}"
    )

    print(
        f"  unique target pairs: "
        f"{len(target_pairs):,}"
    )

    print(
        "  target pairs with common-content evidence: "
        f"{len(strict_pairs):,}"
    )

    if common_content_counts:

        print(
            "  mean common contents per target pair: "
            f"{np.mean(common_content_counts):.4f}"
        )

        print(
            "  maximum common contents per target pair: "
            f"{max(common_content_counts):,}"
        )

    print(
        "  relation counts:"
    )

    for relation in sorted(
        relation_counts
    ):

        label = (
            relation
            if relation
            else "<empty>"
        )

        print(
            f"    {label}: "
            f"{relation_counts[relation]:,}"
        )

    return {
        "clients": clients,
        "client_set": set(
            clients
        ),
        "contents": contents,
        "content_set": set(
            contents
        ),
        "target_pairs": target_pairs,
        "target_edge_multiplicity": (
            target_edge_multiplicity
        ),
        "target_physical_edges": int(
            target_physical_edges
        ),
        "client_contents": (
            client_contents
        ),
        "content_clients": (
            content_clients
        ),
        "strict_pairs": (
            strict_pairs
        ),
        "relation_counts": dict(
            relation_counts
        ),
    }


class UnionFind:

    def __init__(
        self,
        elements,
    ):

        self.parent = {
            element: element
            for element in elements
        }

        self.rank = {
            element: 0
            for element in elements
        }

    def find(
        self,
        element,
    ):

        parent = self.parent[
            element
        ]

        if parent != element:

            self.parent[
                element
            ] = self.find(
                parent
            )

        return self.parent[
            element
        ]

    def union(
        self,
        first,
        second,
    ):

        first_root = self.find(
            first
        )

        second_root = self.find(
            second
        )

        if first_root == second_root:

            return False

        first_rank = self.rank[
            first_root
        ]

        second_rank = self.rank[
            second_root
        ]

        if first_rank < second_rank:

            first_root, second_root = (
                second_root,
                first_root,
            )

        self.parent[
            second_root
        ] = first_root

        if first_rank == second_rank:

            self.rank[
                first_root
            ] += 1

        return True


def build_random_spanning_forest(
    clients,
    target_pairs,
    split_seed,
):

    edges = sorted(
        target_pairs
    )

    rng = np.random.default_rng(
        deterministic_seed(
            "enron",
            split_seed,
            "spanning_forest",
        )
    )

    if edges:

        permutation = rng.permutation(
            len(edges)
        )

        edges = [
            edges[
                int(index)
            ]
            for index in permutation
        ]

    union_find = UnionFind(
        clients
    )

    forest = set()

    for first, second in edges:

        if union_find.union(
            first,
            second,
        ):

            forest.add(
                canonical_pair(
                    first,
                    second,
                )
            )

    return forest


def is_criterion_c_pair(
    pair,
    strict_pairs,
    content_clients,
):

    pair = canonical_pair(
        pair[
            0
        ],
        pair[
            1
        ],
    )

    common_contents = strict_pairs.get(
        pair,
        (),
    )

    if not common_contents:

        return False

    expected_clients = {
        pair[
            0
        ],
        pair[
            1
        ],
    }

    for content_id in common_contents:

        actual_clients = set(
            content_clients.get(
                content_id,
                set(),
            )
        )

        if actual_clients != expected_clients:

            return False

    return True


def build_criterion_c_candidates(
    strict_pairs,
    forest_pairs,
    content_clients,
):

    candidates = []

    for pair in sorted(
        strict_pairs
    ):

        if pair in forest_pairs:

            continue

        if not is_criterion_c_pair(
            pair=pair,
            strict_pairs=strict_pairs,
            content_clients=content_clients,
        ):

            continue

        candidates.append(
            pair
        )

    return candidates


def sample_criterion_c_positives(
    candidates,
    requested,
    split_seed,
):

    requested = int(
        requested
    )

    candidates = list(
        sorted(
            candidates
        )
    )

    if len(
        candidates
    ) < requested:

        raise RuntimeError(
            "Not enough Criterion-C Enron positives.\n"
            f"  split_seed={split_seed}\n"
            f"  requested={requested:,}\n"
            f"  available={len(candidates):,}"
        )

    rng = np.random.default_rng(
        deterministic_seed(
            "enron",
            split_seed,
            "criterion_c_positive_sample",
        )
    )

    indices = rng.choice(
        len(
            candidates
        ),
        size=requested,
        replace=False,
    )

    return [
        candidates[
            int(index)
        ]
        for index in indices
    ]


def choose_removed_endpoint(
    pair,
    content_id,
    split_seed,
):

    first, second = pair

    seed = deterministic_seed(
        "enron",
        split_seed,
        first,
        second,
        content_id,
        "criterion_c_removed_endpoint",
    )

    if (
        seed
        %
        2
    ) == 0:

        return first

    return second


def build_criterion_c_pair_plans(
    positive_pairs,
    strict_pairs,
    content_clients,
    split_seed,
):

    pair_plans = {}

    reserved_incidence = set()

    for pair in positive_pairs:

        pair = canonical_pair(
            pair[
                0
            ],
            pair[
                1
            ],
        )

        if not is_criterion_c_pair(
            pair=pair,
            strict_pairs=strict_pairs,
            content_clients=content_clients,
        ):

            raise RuntimeError(
                "Selected positive is not Criterion C:\n"
                f"  pair={pair}"
            )

        common_contents = tuple(
            sorted(
                strict_pairs[
                    pair
                ]
            )
        )

        local_plan = []

        for content_id in common_contents:

            expected = {
                pair[
                    0
                ],
                pair[
                    1
                ],
            }

            actual = set(
                content_clients.get(
                    content_id,
                    set(),
                )
            )

            if actual != expected:

                raise RuntimeError(
                    "Criterion-C content validation failed.\n"
                    f"  pair={pair}\n"
                    f"  content={content_id}\n"
                    f"  expected={sorted(expected)}\n"
                    f"  actual={sorted(actual)}"
                )

            removed_client = choose_removed_endpoint(
                pair=pair,
                content_id=content_id,
                split_seed=split_seed,
            )

            other_client = (
                pair[
                    1
                ]
                if removed_client
                ==
                pair[
                    0
                ]
                else pair[
                    0
                ]
            )

            incidence = (
                removed_client,
                content_id,
            )

            if incidence in reserved_incidence:

                raise RuntimeError(
                    "Duplicate planned client-content incidence:\n"
                    f"  {incidence}"
                )

            reserved_incidence.add(
                incidence
            )

            local_plan.append({
                "client": (
                    removed_client
                ),
                "other_client": (
                    other_client
                ),
                "content": (
                    content_id
                ),
            })

        pair_plans[
            pair
        ] = local_plan

    return pair_plans


def sample_test_negatives(
    clients,
    target_pairs,
    positive_pairs,
    requested,
    split_seed,
):

    clients = list(
        sorted(
            clients
        )
    )

    target_pairs = set(
        target_pairs
    )

    positive_pairs = set(
        positive_pairs
    )

    requested = int(
        requested
    )

    rng = np.random.default_rng(
        deterministic_seed(
            "enron",
            split_seed,
            "negative_sample",
        )
    )

    negatives = set()

    attempts = 0

    maximum_attempts = max(
        requested * 2000,
        100000,
    )

    while len(
        negatives
    ) < requested:

        attempts += 1

        if attempts > maximum_attempts:

            raise RuntimeError(
                "Could not sample enough true Enron negatives."
            )

        first_index = int(
            rng.integers(
                0,
                len(
                    clients
                ),
            )
        )

        second_index = int(
            rng.integers(
                0,
                len(
                    clients
                ),
            )
        )

        if first_index == second_index:

            continue

        pair = canonical_pair(
            clients[
                first_index
            ],
            clients[
                second_index
            ],
        )

        if pair in target_pairs:

            continue

        if pair in positive_pairs:

            continue

        negatives.add(
            pair
        )

    return sorted(
        negatives
    )


def matching_edges_between(
    graph,
    first,
    second,
    relation=None,
):

    records = []

    seen = set()

    for (
        source,
        target,
    ) in (
        (
            first,
            second,
        ),
        (
            second,
            first,
        ),
    ):

        if not graph.has_edge(
            source,
            target,
        ):

            continue

        edge_data = graph.get_edge_data(
            source,
            target,
        )

        for key, attributes in edge_data.items():

            edge_relation = str(
                attributes.get(
                    "relation",
                    "",
                )
            )

            if (
                relation is not None
                and
                edge_relation != relation
            ):

                continue

            identity = (
                str(source),
                str(target),
                key,
            )

            if identity in seen:

                continue

            seen.add(
                identity
            )

            records.append({
                "source": source,
                "target": target,
                "key": key,
                "attributes": dict(
                    attributes
                ),
            })

    return records


def physical_target_pairs(
    graph,
):

    pairs = set()

    for (
        source,
        target,
        key,
        attributes,
    ) in graph.edges(
        keys=True,
        data=True,
    ):

        if str(
            attributes.get(
                "relation",
                "",
            )
        ) != TARGET_RELATION:

            continue

        source = str(source)
        target = str(target)

        if source == target:

            continue

        if (
            str(
                graph.nodes[
                    source
                ].get(
                    "role",
                    "",
                )
            )
            !=
            "client"
        ):

            continue

        if (
            str(
                graph.nodes[
                    target
                ].get(
                    "role",
                    "",
                )
            )
            !=
            "client"
        ):

            continue

        pairs.add(
            canonical_pair(
                source,
                target,
            )
        )

    return pairs


def build_common_content_pair_set(
    graph,
    content_ids,
):

    pairs = set()

    for content_id in content_ids:

        if graph.is_directed():

            neighbours = (
                set(
                    graph.predecessors(
                        content_id
                    )
                )
                |
                set(
                    graph.successors(
                        content_id
                    )
                )
            )

        else:

            neighbours = set(
                graph.neighbors(
                    content_id
                )
            )

        clients = sorted(
            str(node_id)
            for node_id in neighbours
            if str(
                graph.nodes[
                    node_id
                ].get(
                    "role",
                    "",
                )
            )
            ==
            "client"
        )

        if len(
            clients
        ) < 2:

            continue

        for first, second in combinations(
            clients,
            2,
        ):

            pairs.add(
                canonical_pair(
                    first,
                    second,
                )
            )

    return pairs


def apply_criterion_c_holdout(
    graph,
    positive_pairs,
    pair_plans,
):

    removed_edges = []

    removed_edge_ids = set()

    pair_diagnostics = []

    total_direct_edges = 0

    total_incidence_edges = 0

    for pair_index, pair in enumerate(
        positive_pairs
    ):

        first, second = pair


        direct_records = matching_edges_between(
            graph=graph,
            first=first,
            second=second,
            relation=TARGET_RELATION,
        )

        if not direct_records:

            raise RuntimeError(
                "Selected positive has no physical shared_text "
                "edge at intervention time.\n"
                f"  pair={pair}"
            )

        pair_direct_edges = 0

        for record in direct_records:

            identity = (
                str(
                    record[
                        "source"
                    ]
                ),
                str(
                    record[
                        "target"
                    ]
                ),
                record[
                    "key"
                ],
            )

            if identity in removed_edge_ids:

                continue

            graph.remove_edge(
                record[
                    "source"
                ],
                record[
                    "target"
                ],
                key=record[
                    "key"
                ],
            )

            removed_edges.append(
                record
            )

            removed_edge_ids.add(
                identity
            )

            pair_direct_edges += 1

            total_direct_edges += 1


        pair_incidence_edges = 0

        for removal in pair_plans[
            pair
        ]:

            client_id = removal[
                "client"
            ]

            content_id = removal[
                "content"
            ]

            records = matching_edges_between(
                graph=graph,
                first=client_id,
                second=content_id,
                relation=None,
            )

            if not records:

                raise RuntimeError(
                    "Planned Criterion-C client-content "
                    "incidence is missing.\n"
                    f"  pair={pair}\n"
                    f"  client={client_id}\n"
                    f"  content={content_id}"
                )

            incidence_removed = 0

            for record in records:

                identity = (
                    str(
                        record[
                            "source"
                        ]
                    ),
                    str(
                        record[
                            "target"
                        ]
                    ),
                    record[
                        "key"
                    ],
                )

                if identity in removed_edge_ids:

                    continue

                graph.remove_edge(
                    record[
                        "source"
                    ],
                    record[
                        "target"
                    ],
                    key=record[
                        "key"
                    ],
                )

                removed_edges.append(
                    record
                )

                removed_edge_ids.add(
                    identity
                )

                incidence_removed += 1

                pair_incidence_edges += 1

                total_incidence_edges += 1

            if incidence_removed == 0:

                raise RuntimeError(
                    "No physical edge removed for planned "
                    "Criterion-C incidence."
                )

        pair_diagnostics.append({
            "pair_index": int(
                pair_index
            ),
            "client_a": first,
            "client_b": second,
            "direct_shared_text_edges_removed": int(
                pair_direct_edges
            ),
            "shared_dyadic_contents_broken": int(
                len(
                    pair_plans[
                        pair
                    ]
                )
            ),
            "client_content_physical_edges_removed": int(
                pair_incidence_edges
            ),
        })

    return {
        "removed_edges": (
            removed_edges
        ),
        "pair_diagnostics": (
            pair_diagnostics
        ),
        "direct_edges_removed": int(
            total_direct_edges
        ),
        "incidence_edges_removed": int(
            total_incidence_edges
        ),
    }


def restore_holdout(
    graph,
    intervention,
):

    for record in intervention[
        "removed_edges"
    ]:

        graph.add_edge(
            record[
                "source"
            ],
            record[
                "target"
            ],
            key=record[
                "key"
            ],
            **record[
                "attributes"
            ],
        )


def validate_exact_holdout(
    graph,
    positive_pairs,
    forest_pairs,
    physical_before,
    common_before,
    content_ids,
):

    test_pairs = set(
        positive_pairs
    )

    forest_overlap = int(
        len(
            test_pairs
            &
            forest_pairs
        )
    )

    physical_after = physical_target_pairs(
        graph
    )

    common_after = build_common_content_pair_set(
        graph=graph,
        content_ids=content_ids,
    )

    physical_lost = (
        physical_before
        -
        physical_after
    )

    physical_gained = (
        physical_after
        -
        physical_before
    )

    common_lost = (
        common_before
        -
        common_after
    )

    common_gained = (
        common_after
        -
        common_before
    )

    heldout_physical_remaining = int(
        len(
            test_pairs
            &
            physical_after
        )
    )

    heldout_common_remaining = int(
        len(
            test_pairs
            &
            common_after
        )
    )

    physical_non_test_lost = (
        physical_lost
        -
        test_pairs
    )

    common_non_test_lost = (
        common_lost
        -
        test_pairs
    )

    missing_expected_physical_losses = (
        test_pairs
        -
        physical_lost
    )

    missing_expected_common_losses = (
        test_pairs
        -
        common_lost
    )

    checks = {
        "forest_test_overlap": int(
            forest_overlap
        ),
        "heldout_physical_target_remaining": int(
            heldout_physical_remaining
        ),
        "heldout_common_content_remaining": int(
            heldout_common_remaining
        ),
        "physical_non_test_pairs_lost": int(
            len(
                physical_non_test_lost
            )
        ),
        "common_content_non_test_pairs_lost": int(
            len(
                common_non_test_lost
            )
        ),
        "physical_pairs_gained": int(
            len(
                physical_gained
            )
        ),
        "common_content_pairs_gained": int(
            len(
                common_gained
            )
        ),
        "test_physical_pairs_not_lost": int(
            len(
                missing_expected_physical_losses
            )
        ),
        "test_common_content_pairs_not_lost": int(
            len(
                missing_expected_common_losses
            )
        ),
    }

    failures = {
        key: value
        for key, value in checks.items()
        if value != 0
    }

    if failures:

        raise RuntimeError(
            "Criterion-C exact holdout validation failed:\n"
            +
            "\n".join(
                f"  {key}={value}"
                for key, value in failures.items()
            )
        )

    return {
        "checks": checks,
        "physical_pairs_before": int(
            len(
                physical_before
            )
        ),
        "physical_pairs_after": int(
            len(
                physical_after
            )
        ),
        "physical_pairs_lost": int(
            len(
                physical_lost
            )
        ),
        "common_pairs_before": int(
            len(
                common_before
            )
        ),
        "common_pairs_after": int(
            len(
                common_after
            )
        ),
        "common_pairs_lost": int(
            len(
                common_lost
            )
        ),
    }


def prepare_query(
    raw_matrix,
    C,
    sensitivity_mode,
    representation,
):

    raw_matrix = np.asarray(
        raw_matrix,
        dtype=float,
    )

    if raw_matrix.ndim != 2:

        raise RuntimeError(
            f"{representation}: descriptor matrix "
            "must be two-dimensional."
        )

    if not np.all(
        np.isfinite(
            raw_matrix
        )
    ):

        raise RuntimeError(
            f"{representation}: descriptor contains "
            "NaN or infinite values."
        )

    substantive_negative_count = int(
        np.sum(
            raw_matrix
            <
            -NEGATIVE_TOLERANCE
        )
    )

    negative_entry_count = int(
        np.sum(
            raw_matrix
            <
            0.0
        )
    )

    if sensitivity_mode == "nonnegative_report":

        if substantive_negative_count > 0:

            raise RuntimeError(
                f"{representation} contains substantive "
                "negative entries and cannot use the "
                "sqrt(2)C nonnegative-report sensitivity bound.\n"
                f"  count={substantive_negative_count:,}\n"
                f"  minimum={np.min(raw_matrix):.12e}"
            )


        query_matrix = np.maximum(
            raw_matrix,
            0.0,
        )

    elif sensitivity_mode == "generic_report":

        query_matrix = np.array(
            raw_matrix,
            dtype=float,
            copy=True,
        )

    else:

        raise ValueError(
            f"Unknown sensitivity mode: "
            f"{sensitivity_mode}"
        )

    norms = np.linalg.norm(
        query_matrix,
        axis=1,
    )

    clip_mask = (
        norms
        >
        float(C)
    )

    clipped_matrix = np.array(
        query_matrix,
        dtype=float,
        copy=True,
    )

    if np.any(
        clip_mask
    ):

        clipped_matrix[
            clip_mask
        ] *= (
            float(C)
            /
            norms[
                clip_mask
            ]
        )[
            :,
            None
        ]

    clipped_norms = np.linalg.norm(
        clipped_matrix,
        axis=1,
    )

    rounded = np.round(
        clipped_matrix,
        decimals=SCORE_ROUND_DECIMALS,
    )

    component_variance = np.var(
        clipped_matrix,
        axis=0,
    )

    return {
        "matrix": clipped_matrix,
        "dimension": int(
            clipped_matrix.shape[
                1
            ]
        ),
        "clip_fraction": float(
            np.mean(
                clip_mask
            )
        ),
        "zero_row_fraction": float(
            np.mean(
                clipped_norms
                <=
                1e-15
            )
        ),
        "negative_entry_fraction": float(
            negative_entry_count
            /
            max(
                raw_matrix.size,
                1,
            )
        ),
        "substantive_negative_entries": int(
            substantive_negative_count
        ),
        "unique_rows_rounded": int(
            np.unique(
                rounded,
                axis=0,
            ).shape[
                0
            ]
        ),
        "mean_component_variance": float(
            np.mean(
                component_variance
            )
        ),
        "max_component_variance": float(
            np.max(
                component_variance
            )
        ),
    }


def analytical_sensitivity(
    C,
    sensitivity_mode,
):

    if sensitivity_mode == "nonnegative_report":

        return float(
            math.sqrt(
                2.0
            )
            *
            float(C)
        )

    if sensitivity_mode == "generic_report":

        return float(
            2.0
            *
            float(C)
        )

    raise ValueError(
        sensitivity_mode
    )


def sample_unit_direction(
    dimension,
    rng,
):

    while True:

        vector = rng.normal(
            size=int(
                dimension
            )
        )

        norm = float(
            np.linalg.norm(
                vector
            )
        )

        if norm > 0.0:

            return (
                vector
                /
                norm
            )


def privatize_matrix(
    clean_matrix,
    client_ids,
    epsilon_signature,
    sensitivity,
    split_seed,
    representation,
    mechanism_seed,
):

    clean_matrix = np.asarray(
        clean_matrix,
        dtype=float,
    )

    if len(
        client_ids
    ) != len(
        clean_matrix
    ):

        raise RuntimeError(
            "client_ids and clean_matrix have different lengths."
        )

    n_clients, dimension = (
        clean_matrix.shape
    )

    private_matrix = np.empty_like(
        clean_matrix,
        dtype=float,
    )

    noise_norms = np.zeros(
        n_clients,
        dtype=float,
    )

    for row_index, client_id in enumerate(
        client_ids
    ):

        rng = np.random.default_rng(
            deterministic_seed(
                "enron",
                split_seed,
                representation,
                mechanism_seed,
                client_id,
                "private_report",
            )
        )

        direction = sample_unit_direction(
            dimension=dimension,
            rng=rng,
        )

        radius = float(
            rng.gamma(
                shape=dimension,
                scale=(
                    float(
                        sensitivity
                    )
                    /
                    float(
                        epsilon_signature
                    )
                ),
            )
        )

        private_matrix[
            row_index
        ] = (
            clean_matrix[
                row_index
            ]
            +
            radius
            *
            direction
        )

        noise_norms[
            row_index
        ] = radius

    return {
        "matrix": private_matrix,
        "noise_norms": (
            noise_norms
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

    result = np.zeros_like(
        matrix
    )

    valid = (
        norms[
            :,
            0
        ]
        >
        1e-15
    )

    result[
        valid
    ] = (
        matrix[
            valid
        ]
        /
        norms[
            valid
        ]
    )

    return result


def cosine_pair_scores(
    matrix,
    client_ids,
    pairs,
):

    normalized = normalize_rows(
        matrix
    )

    client_index = {
        str(client_id): row_index
        for row_index, client_id
        in enumerate(
            client_ids
        )
    }

    scores = []

    for first, second in pairs:

        if (
            first not in client_index
            or
            second not in client_index
        ):

            raise RuntimeError(
                "Evaluation pair missing descriptor:\n"
                f"  ({first}, {second})"
            )

        score = float(
            np.dot(
                normalized[
                    client_index[
                        first
                    ]
                ],
                normalized[
                    client_index[
                        second
                    ]
                ],
            )
        )

        scores.append(
            float(
                np.clip(
                    score,
                    -1.0,
                    1.0,
                )
            )
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
        orientation
    )


def stabilize_scores(
    scores,
):

    return np.round(
        np.asarray(
            scores,
            dtype=float,
        ),
        decimals=SCORE_ROUND_DECIMALS,
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
        "negative_score_mean": float(
            np.mean(
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
        "unique_scores": int(
            len(
                np.unique(
                    scores
                )
            )
        ),
    }


def safe_correlation(
    first,
    second,
    method,
):

    first = np.asarray(
        first,
        dtype=float,
    )

    second = np.asarray(
        second,
        dtype=float,
    )

    valid = (
        np.isfinite(
            first
        )
        &
        np.isfinite(
            second
        )
    )

    first = first[
        valid
    ]

    second = second[
        valid
    ]

    if len(
        first
    ) < 3:

        return np.nan

    if (
        np.std(
            first
        )
        <=
        1e-15
        or
        np.std(
            second
        )
        <=
        1e-15
    ):

        return np.nan

    if method == "spearman":

        value = spearmanr(
            first,
            second,
        ).statistic

    elif method == "pearson":

        value = pearsonr(
            first,
            second,
        ).statistic

    else:

        raise ValueError(
            method
        )

    if not np.isfinite(
        value
    ):

        return np.nan

    return float(
        value
    )


def rowwise_cosine(
    clean_matrix,
    private_matrix,
):

    clean_matrix = np.asarray(
        clean_matrix,
        dtype=float,
    )

    private_matrix = np.asarray(
        private_matrix,
        dtype=float,
    )

    numerator = np.sum(
        clean_matrix
        *
        private_matrix,
        axis=1,
    )

    clean_norm = np.linalg.norm(
        clean_matrix,
        axis=1,
    )

    private_norm = np.linalg.norm(
        private_matrix,
        axis=1,
    )

    denominator = (
        clean_norm
        *
        private_norm
    )

    result = np.zeros(
        len(
            clean_matrix
        ),
        dtype=float,
    )

    valid = (
        denominator
        >
        1e-15
    )

    result[
        valid
    ] = (
        numerator[
            valid
        ]
        /
        denominator[
            valid
        ]
    )

    return np.clip(
        result,
        -1.0,
        1.0,
    )


def run_split(
    graph,
    index,
    split_seed,
    representations,
    epsilons,
    mechanism_seeds,
    test_positives,
    C,
    rho_signature,
    sensitivity_mode,
    d_max,
    degree_weight,
    split_folder,
):

    split_start = time.perf_counter()

    print()
    print(
        "#" * 80
    )

    print(
        f"ENRON CRITERION-C | split seed {split_seed}"
    )

    print(
        "#" * 80
    )


    forest_pairs = build_random_spanning_forest(
        clients=index[
            "clients"
        ],
        target_pairs=index[
            "target_pairs"
        ],
        split_seed=split_seed,
    )

    criterion_c_candidates = build_criterion_c_candidates(
        strict_pairs=index[
            "strict_pairs"
        ],
        forest_pairs=forest_pairs,
        content_clients=index[
            "content_clients"
        ],
    )

    positive_pairs = sample_criterion_c_positives(
        candidates=criterion_c_candidates,
        requested=test_positives,
        split_seed=split_seed,
    )

    negative_pairs = sample_test_negatives(
        clients=index[
            "clients"
        ],
        target_pairs=index[
            "target_pairs"
        ],
        positive_pairs=positive_pairs,
        requested=len(
            positive_pairs
        ),
        split_seed=split_seed,
    )

    pair_plans = build_criterion_c_pair_plans(
        positive_pairs=positive_pairs,
        strict_pairs=index[
            "strict_pairs"
        ],
        content_clients=index[
            "content_clients"
        ],
        split_seed=split_seed,
    )

    shared_content_counts = [
        len(
            pair_plans[
                pair
            ]
        )
        for pair in positive_pairs
    ]

    print(
        f"Target pairs: "
        f"{len(index['target_pairs']):,}"
    )

    print(
        f"Forest pairs: "
        f"{len(forest_pairs):,}"
    )

    print(
        "Criterion-C non-forest candidates: "
        f"{len(criterion_c_candidates):,}"
    )

    print(
        f"Test positives: "
        f"{len(positive_pairs):,}"
    )

    print(
        f"Test negatives: "
        f"{len(negative_pairs):,}"
    )

    print(
        "Shared dyadic contents to break: "
        f"{sum(shared_content_counts):,}"
    )

    print(
        "Mean shared dyadic contents per positive: "
        f"{np.mean(shared_content_counts):.4f}"
    )

    print(
        "Maximum shared dyadic contents per positive: "
        f"{max(shared_content_counts):,}"
    )


    physical_before = physical_target_pairs(
        graph
    )

    common_before = build_common_content_pair_set(
        graph=graph,
        content_ids=index[
            "contents"
        ],
    )

    intervention = None

    try:


        intervention = apply_criterion_c_holdout(
            graph=graph,
            positive_pairs=positive_pairs,
            pair_plans=pair_plans,
        )

        validation = validate_exact_holdout(
            graph=graph,
            positive_pairs=positive_pairs,
            forest_pairs=forest_pairs,
            physical_before=physical_before,
            common_before=common_before,
            content_ids=index[
                "contents"
            ],
        )

        print()
        print(
            "Exact structural validation:"
        )

        for key, value in validation[
            "checks"
        ].items():

            print(
                f"  {key}: "
                f"{value}"
            )

        print(
            "  physical shared_text edges removed: "
            f"{intervention['direct_edges_removed']:,}"
        )

        print(
            "  physical client-content edges removed: "
            f"{intervention['incidence_edges_removed']:,}"
        )


        required_clients = sorted(
            set(
                client_id
                for pair in (
                    list(
                        positive_pairs
                    )
                    +
                    list(
                        negative_pairs
                    )
                )
                for client_id in pair
            )
        )

        print(
            "Clients requiring descriptor reports: "
            f"{len(required_clients):,}"
        )


        generated = generate_selected_descriptors(
            training_graph=graph,
            client_ids=required_clients,
            d_max=d_max,
            degree_weight=degree_weight,
        )

        failures = generated[
            "failures"
        ]

        if failures:

            raise RuntimeError(
                "Descriptor generation failed for "
                f"{len(failures)} Enron clients."
            )

        descriptor_client_ids = [
            str(value)
            for value in generated[
                "client_ids"
            ]
        ]

        if descriptor_client_ids != required_clients:

            raise RuntimeError(
                "Descriptor client ordering differs from "
                "the deterministic requested ordering."
            )

        descriptor_matrices = generated[
            "descriptors"
        ]

        os.makedirs(
            split_folder,
            exist_ok=True,
        )


        pair_rows = []

        for pair_index, pair in enumerate(
            positive_pairs
        ):

            pair_rows.append({
                "pair_index": int(
                    pair_index
                ),
                "label": 1,
                "client_a": pair[
                    0
                ],
                "client_b": pair[
                    1
                ],
                "criterion_c": 1,
                "shared_dyadic_content_count": int(
                    len(
                        pair_plans[
                            pair
                        ]
                    )
                ),
            })

        for pair_index, pair in enumerate(
            negative_pairs
        ):

            pair_rows.append({
                "pair_index": int(
                    pair_index
                ),
                "label": 0,
                "client_a": pair[
                    0
                ],
                "client_b": pair[
                    1
                ],
                "criterion_c": 0,
                "shared_dyadic_content_count": np.nan,
            })

        pd.DataFrame(
            pair_rows
        ).to_csv(
            os.path.join(
                split_folder,
                "evaluation_pairs.csv",
            ),
            index=False,
        )

        pd.DataFrame(
            intervention[
                "pair_diagnostics"
            ]
        ).to_csv(
            os.path.join(
                split_folder,
                "holdout_burden.csv",
            ),
            index=False,
        )


        clean_rows = []

        private_rows = []

        representation_rows = []

        pair_score_rows = []


        for representation in representations:

            print()
            print(
                f"Representation: "
                f"{representation}"
            )

            if representation not in descriptor_matrices:

                raise RuntimeError(
                    "generate_selected_descriptors() did not "
                    f"return {representation}."
                )

            orientation = (
                REPRESENTATION_ORIENTATION[
                    representation
                ]
            )

            raw_matrix = np.asarray(
                descriptor_matrices[
                    representation
                ],
                dtype=float,
            )

            query = prepare_query(
                raw_matrix=raw_matrix,
                C=C,
                sensitivity_mode=sensitivity_mode,
                representation=representation,
            )

            clean_matrix = query[
                "matrix"
            ]

            dimension = int(
                query[
                    "dimension"
                ]
            )

            sensitivity = analytical_sensitivity(
                C=C,
                sensitivity_mode=sensitivity_mode,
            )

            print(
                f"  orientation: "
                f"{orientation}"
            )

            print(
                f"  dimension: "
                f"{dimension}"
            )

            print(
                "  unique clipped rows: "
                f"{query['unique_rows_rounded']:,}/"
                f"{len(clean_matrix):,}"
            )

            print(
                f"  clip fraction: "
                f"{query['clip_fraction']:.6f}"
            )

            print(
                f"  negative-entry fraction: "
                f"{query['negative_entry_fraction']:.9f}"
            )

            print(
                f"  analytical Delta2: "
                f"{sensitivity:.9f}"
            )

            representation_rows.append({
                "split_seed": int(
                    split_seed
                ),
                "representation": (
                    representation
                ),
                "orientation": (
                    orientation
                ),
                "clients": int(
                    len(
                        clean_matrix
                    )
                ),
                "dimension": int(
                    dimension
                ),
                "unique_rows_rounded": int(
                    query[
                        "unique_rows_rounded"
                    ]
                ),
                "clip_fraction": float(
                    query[
                        "clip_fraction"
                    ]
                ),
                "zero_row_fraction": float(
                    query[
                        "zero_row_fraction"
                    ]
                ),
                "negative_entry_fraction": float(
                    query[
                        "negative_entry_fraction"
                    ]
                ),
                "substantive_negative_entries": int(
                    query[
                        "substantive_negative_entries"
                    ]
                ),
                "mean_component_variance": float(
                    query[
                        "mean_component_variance"
                    ]
                ),
                "max_component_variance": float(
                    query[
                        "max_component_variance"
                    ]
                ),
                "sensitivity_l2": float(
                    sensitivity
                ),
            })


            clean_positive_cosine = cosine_pair_scores(
                matrix=clean_matrix,
                client_ids=descriptor_client_ids,
                pairs=positive_pairs,
            )

            clean_negative_cosine = cosine_pair_scores(
                matrix=clean_matrix,
                client_ids=descriptor_client_ids,
                pairs=negative_pairs,
            )

            clean_positive_scores = orient_scores(
                cosine_scores=clean_positive_cosine,
                orientation=orientation,
            )

            clean_negative_scores = orient_scores(
                cosine_scores=clean_negative_cosine,
                orientation=orientation,
            )

            clean_metrics = binary_metrics(
                positive_scores=clean_positive_scores,
                negative_scores=clean_negative_scores,
            )

            print(
                f"  clean AUC="
                f"{clean_metrics['auc']:.6f}"
                " | AP="
                f"{clean_metrics['ap']:.6f}"
            )

            clean_rows.append({
                "split_seed": int(
                    split_seed
                ),
                "representation": (
                    representation
                ),
                "orientation": (
                    orientation
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
                "clean_auc": float(
                    clean_metrics[
                        "auc"
                    ]
                ),
                "clean_ap": float(
                    clean_metrics[
                        "ap"
                    ]
                ),
                "clean_positive_score_mean": float(
                    clean_metrics[
                        "positive_score_mean"
                    ]
                ),
                "clean_negative_score_mean": float(
                    clean_metrics[
                        "negative_score_mean"
                    ]
                ),
                "clean_score_mean_gap": float(
                    clean_metrics[
                        "score_mean_gap"
                    ]
                ),
                "clean_unique_scores": int(
                    clean_metrics[
                        "unique_scores"
                    ]
                ),
            })

            clean_all_scores = np.concatenate(
                [
                    clean_positive_scores,
                    clean_negative_scores,
                ]
            )


            for epsilon_total in epsilons:

                epsilon_total = float(
                    epsilon_total
                )

                epsilon_signature = (
                    float(
                        rho_signature
                    )
                    *
                    epsilon_total
                )

                theoretical_mean_noise = float(
                    dimension
                    *
                    sensitivity
                    /
                    epsilon_signature
                )

                epsilon_auc = []

                epsilon_ap = []

                epsilon_noise = []

                for mechanism_seed in mechanism_seeds:

                    private_result = privatize_matrix(
                        clean_matrix=clean_matrix,
                        client_ids=descriptor_client_ids,
                        epsilon_signature=epsilon_signature,
                        sensitivity=sensitivity,
                        split_seed=split_seed,
                        representation=representation,
                        mechanism_seed=mechanism_seed,
                    )

                    private_matrix = private_result[
                        "matrix"
                    ]

                    noise_norms = private_result[
                        "noise_norms"
                    ]

                    private_positive_cosine = cosine_pair_scores(
                        matrix=private_matrix,
                        client_ids=descriptor_client_ids,
                        pairs=positive_pairs,
                    )

                    private_negative_cosine = cosine_pair_scores(
                        matrix=private_matrix,
                        client_ids=descriptor_client_ids,
                        pairs=negative_pairs,
                    )

                    private_positive_scores = orient_scores(
                        cosine_scores=private_positive_cosine,
                        orientation=orientation,
                    )

                    private_negative_scores = orient_scores(
                        cosine_scores=private_negative_cosine,
                        orientation=orientation,
                    )

                    private_metrics = binary_metrics(
                        positive_scores=private_positive_scores,
                        negative_scores=private_negative_scores,
                    )

                    private_all_scores = np.concatenate(
                        [
                            private_positive_scores,
                            private_negative_scores,
                        ]
                    )

                    stable_clean = stabilize_scores(
                        clean_all_scores
                    )

                    stable_private = stabilize_scores(
                        private_all_scores
                    )

                    pair_spearman = safe_correlation(
                        stable_clean,
                        stable_private,
                        "spearman",
                    )

                    pair_pearson = safe_correlation(
                        stable_clean,
                        stable_private,
                        "pearson",
                    )

                    pair_mae = float(
                        np.mean(
                            np.abs(
                                stable_clean
                                -
                                stable_private
                            )
                        )
                    )

                    report_cosines = rowwise_cosine(
                        clean_matrix,
                        private_matrix,
                    )

                    observed_mean_noise = float(
                        np.mean(
                            noise_norms
                        )
                    )

                    private_rows.append({
                        "split_seed": int(
                            split_seed
                        ),
                        "mechanism_seed": int(
                            mechanism_seed
                        ),
                        "representation": (
                            representation
                        ),
                        "orientation": (
                            orientation
                        ),
                        "epsilon_total": float(
                            epsilon_total
                        ),
                        "epsilon_signature": float(
                            epsilon_signature
                        ),
                        "rho_signature": float(
                            rho_signature
                        ),
                        "actual_release_privacy": (
                            "epsilon_signature-LDP"
                        ),
                        "sensitivity_mode": (
                            sensitivity_mode
                        ),
                        "sensitivity_l2": float(
                            sensitivity
                        ),
                        "clip_C": float(C),
                        "dimension": int(
                            dimension
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
                        "clean_auc": float(
                            clean_metrics[
                                "auc"
                            ]
                        ),
                        "clean_ap": float(
                            clean_metrics[
                                "ap"
                            ]
                        ),
                        "private_auc": float(
                            private_metrics[
                                "auc"
                            ]
                        ),
                        "private_ap": float(
                            private_metrics[
                                "ap"
                            ]
                        ),
                        "auc_change": float(
                            private_metrics[
                                "auc"
                            ]
                            -
                            clean_metrics[
                                "auc"
                            ]
                        ),
                        "ap_change": float(
                            private_metrics[
                                "ap"
                            ]
                            -
                            clean_metrics[
                                "ap"
                            ]
                        ),
                        "clean_private_pair_spearman": (
                            pair_spearman
                        ),
                        "clean_private_pair_pearson": (
                            pair_pearson
                        ),
                        "clean_private_pair_score_mae": float(
                            pair_mae
                        ),
                        "clean_private_report_cosine_mean": float(
                            np.mean(
                                report_cosines
                            )
                        ),
                        "clean_private_report_cosine_median": float(
                            np.median(
                                report_cosines
                            )
                        ),
                        "theoretical_mean_noise_norm": float(
                            theoretical_mean_noise
                        ),
                        "observed_mean_noise_norm": float(
                            observed_mean_noise
                        ),
                        "observed_median_noise_norm": float(
                            np.median(
                                noise_norms
                            )
                        ),
                    })

                    epsilon_auc.append(
                        private_metrics[
                            "auc"
                        ]
                    )

                    epsilon_ap.append(
                        private_metrics[
                            "ap"
                        ]
                    )

                    epsilon_noise.append(
                        observed_mean_noise
                    )


                    for pair_index, score in enumerate(
                        private_positive_scores
                    ):

                        pair_score_rows.append({
                            "split_seed": int(
                                split_seed
                            ),
                            "mechanism_seed": int(
                                mechanism_seed
                            ),
                            "representation": (
                                representation
                            ),
                            "orientation": (
                                orientation
                            ),
                            "epsilon_total": float(
                                epsilon_total
                            ),
                            "epsilon_signature": float(
                                epsilon_signature
                            ),
                            "pair_index": int(
                                pair_index
                            ),
                            "label": 1,
                            "clean_score": float(
                                clean_positive_scores[
                                    pair_index
                                ]
                            ),
                            "private_score": float(
                                score
                            ),
                        })

                    for pair_index, score in enumerate(
                        private_negative_scores
                    ):

                        pair_score_rows.append({
                            "split_seed": int(
                                split_seed
                            ),
                            "mechanism_seed": int(
                                mechanism_seed
                            ),
                            "representation": (
                                representation
                            ),
                            "orientation": (
                                orientation
                            ),
                            "epsilon_total": float(
                                epsilon_total
                            ),
                            "epsilon_signature": float(
                                epsilon_signature
                            ),
                            "pair_index": int(
                                pair_index
                            ),
                            "label": 0,
                            "clean_score": float(
                                clean_negative_scores[
                                    pair_index
                                ]
                            ),
                            "private_score": float(
                                score
                            ),
                        })

                print(
                    f"    eps={epsilon_total:g}"
                    " | eps_sig="
                    f"{epsilon_signature:g}"
                    " | private AUC="
                    f"{np.mean(epsilon_auc):.6f}"
                    " | AP="
                    f"{np.mean(epsilon_ap):.6f}"
                    " | theory noise="
                    f"{theoretical_mean_noise:.6f}"
                    " | observed="
                    f"{np.mean(epsilon_noise):.6f}"
                )


        clean_dataframe = pd.DataFrame(
            clean_rows
        )

        private_dataframe = pd.DataFrame(
            private_rows
        )

        representation_dataframe = pd.DataFrame(
            representation_rows
        )

        pair_score_dataframe = pd.DataFrame(
            pair_score_rows
        )

        clean_dataframe.to_csv(
            os.path.join(
                split_folder,
                "clean_metrics.csv",
            ),
            index=False,
        )

        private_dataframe.to_csv(
            os.path.join(
                split_folder,
                "private_runs.csv",
            ),
            index=False,
        )

        representation_dataframe.to_csv(
            os.path.join(
                split_folder,
                "representation_diagnostics.csv",
            ),
            index=False,
        )

        pair_score_dataframe.to_csv(
            os.path.join(
                split_folder,
                "private_pair_scores.csv.gz",
            ),
            index=False,
            compression="gzip",
        )

        split_runtime = float(
            time.perf_counter()
            -
            split_start
        )

        metadata = {
            "schema_version": (
                SCHEMA_VERSION
            ),
            "dataset": (
                "enron"
            ),
            "split_seed": int(
                split_seed
            ),
            "target_relation": (
                TARGET_RELATION
            ),
            "holdout_criterion": (
                "Criterion C: every common content shared "
                "by a held-out pair has exactly the two "
                "endpoints as its distinct client neighbours."
            ),
            "criterion_c_candidates": int(
                len(
                    criterion_c_candidates
                )
            ),
            "test_positives": int(
                len(
                    positive_pairs
                )
            ),
            "test_negatives": int(
                len(
                    negative_pairs
                )
            ),
            "forest_pairs": int(
                len(
                    forest_pairs
                )
            ),
            "mean_shared_dyadic_contents": float(
                np.mean(
                    shared_content_counts
                )
            ),
            "maximum_shared_dyadic_contents": int(
                max(
                    shared_content_counts
                )
            ),
            "physical_shared_text_edges_removed": int(
                intervention[
                    "direct_edges_removed"
                ]
            ),
            "physical_client_content_edges_removed": int(
                intervention[
                    "incidence_edges_removed"
                ]
            ),
            "exact_structural_validation": (
                validation
            ),
            "required_client_reports": int(
                len(
                    required_clients
                )
            ),
            "representations": list(
                representations
            ),
            "orientations": {
                representation:
                    REPRESENTATION_ORIENTATION[
                        representation
                    ]
                for representation in representations
            },
            "primary_representation": (
                "hks_d12_global"
            ),
            "primary_orientation": (
                "distance"
            ),
            "epsilons_total": [
                float(
                    value
                )
                for value in epsilons
            ],
            "mechanism_seeds": [
                int(
                    value
                )
                for value in mechanism_seeds
            ],
            "clip_C": float(C),
            "rho_signature": float(
                rho_signature
            ),
            "sensitivity_mode": (
                sensitivity_mode
            ),
            "privacy_adjacency": (
                "complete client-report replacement"
            ),
            "privacy_mechanism": (
                "pure L2 K-norm"
            ),
            "analytical_sensitivity": (
                "sqrt(2)C for the mathematically "
                "nonnegative L2-clipped query range; "
                "2C in generic_report mode."
            ),
            "actual_privacy_guarantee_in_this_experiment": (
                "epsilon_signature-LDP because only the "
                "descriptor report is released."
            ),
            "epsilon_total_interpretation": (
                "Reference composed synthesis budget. "
                "epsilon_signature=rho_signature*epsilon_total; "
                "the remaining budget is reserved but unused "
                "in this downstream experiment."
            ),
            "one_report_per_client_per_private_run": (
                True
            ),
            "score_orientation_frozen_before_private_evaluation": (
                True
            ),
            "empirical_sensitivity_used_for_calibration": (
                False
            ),
            "runtime_seconds": float(
                split_runtime
            ),
        }

        with open(
            os.path.join(
                split_folder,
                "metadata.json",
            ),
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
            f"Split runtime: "
            f"{split_runtime:.3f} s"
        )

        return {
            "clean": (
                clean_dataframe
            ),
            "private": (
                private_dataframe
            ),
            "representation_diagnostics": (
                representation_dataframe
            ),
            "metadata": (
                metadata
            ),
        }

    finally:

        if intervention is not None:

            restore_holdout(
                graph,
                intervention,
            )


def average_mechanism_seeds(
    private_runs,
):

    group_columns = [
        "split_seed",
        "representation",
        "orientation",
        "epsilon_total",
        "epsilon_signature",
        "sensitivity_mode",
    ]

    metric_columns = [
        "clean_auc",
        "clean_ap",
        "private_auc",
        "private_ap",
        "auc_change",
        "ap_change",
        "clean_private_pair_spearman",
        "clean_private_pair_pearson",
        "clean_private_pair_score_mae",
        "clean_private_report_cosine_mean",
        "observed_mean_noise_norm",
    ]

    rows = []

    for keys, group in private_runs.groupby(
        group_columns,
        sort=True,
    ):

        (
            split_seed,
            representation,
            orientation,
            epsilon_total,
            epsilon_signature,
            sensitivity_mode,
        ) = keys

        row = {
            "split_seed": int(
                split_seed
            ),
            "representation": (
                representation
            ),
            "orientation": (
                orientation
            ),
            "epsilon_total": float(
                epsilon_total
            ),
            "epsilon_signature": float(
                epsilon_signature
            ),
            "sensitivity_mode": (
                sensitivity_mode
            ),
            "mechanism_seeds": int(
                group[
                    "mechanism_seed"
                ].nunique()
            ),
            "dimension": int(
                group[
                    "dimension"
                ].iloc[
                    0
                ]
            ),
            "sensitivity_l2": float(
                group[
                    "sensitivity_l2"
                ].iloc[
                    0
                ]
            ),
            "theoretical_mean_noise_norm": float(
                group[
                    "theoretical_mean_noise_norm"
                ].iloc[
                    0
                ]
            ),
        }

        for metric in metric_columns:

            values = group[
                metric
            ].to_numpy(
                dtype=float
            )

            finite = values[
                np.isfinite(
                    values
                )
            ]

            if len(
                finite
            ) == 0:

                row[
                    f"{metric}__mechanism_mean"
                ] = np.nan

                row[
                    f"{metric}__mechanism_std"
                ] = np.nan

            else:

                row[
                    f"{metric}__mechanism_mean"
                ] = float(
                    np.mean(
                        finite
                    )
                )

                row[
                    f"{metric}__mechanism_std"
                ] = float(
                    np.std(
                        finite,
                        ddof=1,
                    )
                    if len(
                        finite
                    )
                    >
                    1
                    else 0.0
                )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


def aggregate_across_splits(
    split_means,
):

    group_columns = [
        "representation",
        "orientation",
        "epsilon_total",
        "epsilon_signature",
        "sensitivity_mode",
    ]

    metric_bases = [
        "clean_auc",
        "clean_ap",
        "private_auc",
        "private_ap",
        "auc_change",
        "ap_change",
        "clean_private_pair_spearman",
        "clean_private_pair_pearson",
        "clean_private_pair_score_mae",
        "clean_private_report_cosine_mean",
        "observed_mean_noise_norm",
    ]

    rows = []

    for keys, group in split_means.groupby(
        group_columns,
        sort=True,
    ):

        (
            representation,
            orientation,
            epsilon_total,
            epsilon_signature,
            sensitivity_mode,
        ) = keys

        row = {
            "representation": (
                representation
            ),
            "orientation": (
                orientation
            ),
            "epsilon_total": float(
                epsilon_total
            ),
            "epsilon_signature": float(
                epsilon_signature
            ),
            "sensitivity_mode": (
                sensitivity_mode
            ),
            "split_seeds": int(
                group[
                    "split_seed"
                ].nunique()
            ),
            "dimension": int(
                group[
                    "dimension"
                ].iloc[
                    0
                ]
            ),
            "sensitivity_l2": float(
                group[
                    "sensitivity_l2"
                ].iloc[
                    0
                ]
            ),
            "theoretical_mean_noise_norm": float(
                group[
                    "theoretical_mean_noise_norm"
                ].iloc[
                    0
                ]
            ),
        }

        for metric in metric_bases:

            column = (
                f"{metric}"
                f"__mechanism_mean"
            )

            statistics = mean_std_ci95(
                group[
                    column
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

    return pd.DataFrame(
        rows
    )


def run_experiment(
    output_folder=DEFAULT_OUTPUT_FOLDER,
    representations=DEFAULT_REPRESENTATIONS,
    epsilons=DEFAULT_EPSILONS,
    split_seeds=DEFAULT_SPLIT_SEEDS,
    mechanism_seeds=DEFAULT_MECHANISM_SEEDS,
    test_positives=DEFAULT_TEST_POSITIVES,
    C=DEFAULT_C,
    rho_signature=DEFAULT_RHO_SIGNATURE,
    sensitivity_mode=DEFAULT_SENSITIVITY_MODE,
    d_max=DEFAULT_D_MAX,
    degree_weight=DEFAULT_DEGREE_WEIGHT,
):

    total_start = time.perf_counter()

    representations = tuple(
        str(value)
        for value in representations
    )

    epsilons = tuple(
        float(value)
        for value in epsilons
    )

    split_seeds = tuple(
        int(value)
        for value in split_seeds
    )

    mechanism_seeds = tuple(
        int(value)
        for value in mechanism_seeds
    )

    test_positives = int(
        test_positives
    )

    if test_positives <= 0:

        raise ValueError(
            "test_positives must be positive."
        )

    for representation in representations:

        if representation not in REPRESENTATION_ORIENTATION:

            raise ValueError(
                "No frozen orientation for representation: "
                f"{representation}"
            )

    if float(C) <= 0.0:

        raise ValueError(
            "C must be positive."
        )

    if not (
        0.0
        <
        float(
            rho_signature
        )
        <=
        1.0
    ):

        raise ValueError(
            "rho_signature must be in (0,1]."
        )

    for epsilon in epsilons:

        if epsilon <= 0.0:

            raise ValueError(
                "All epsilon values must be positive."
            )

    graph_path = os.path.join(
        output_folder,
        "enron",
        "enron_global_graph.gml.gz",
    )

    if not os.path.exists(
        graph_path
    ):

        raise FileNotFoundError(
            "Missing prepared Enron graph:\n"
            f"  {graph_path}"
        )

    experiment_folder = os.path.join(
        output_folder,
        "enron",
        "experiments",
        "enron_private_link_prediction_special",
    )

    os.makedirs(
        experiment_folder,
        exist_ok=True,
    )

    frozen_orientations = {
        representation:
            REPRESENTATION_ORIENTATION[
                representation
            ]
        for representation in representations
    }

    print()
    print(
        "=" * 80
    )

    print(
        "ENRON PRIVATE CRITERION-C LINK PREDICTION"
    )

    print(
        "=" * 80
    )

    print(
        f"Representations: "
        f"{list(representations)}"
    )

    print(
        f"Frozen orientations: "
        f"{frozen_orientations}"
    )

    print(
        f"Total epsilons: "
        f"{list(epsilons)}"
    )

    print(
        f"Split seeds: "
        f"{list(split_seeds)}"
    )

    print(
        f"Mechanism seeds: "
        f"{list(mechanism_seeds)}"
    )

    print(
        f"Criterion-C positives per split: "
        f"{test_positives}"
    )

    print(
        f"C: "
        f"{C}"
    )

    print(
        f"rho_signature: "
        f"{rho_signature}"
    )

    print(
        f"Sensitivity mode: "
        f"{sensitivity_mode}"
    )

    print(
        "Actual released privacy channel: "
        "epsilon_signature-LDP descriptor"
    )

    print(
        "Privacy adjacency: "
        "complete client-report replacement"
    )

    print(
        "Mechanism: pure L2 K-norm"
    )

    print(
        "=" * 80
    )

    print()
    print(
        "Loading prepared Enron graph..."
    )

    graph = nx.read_gml(
        graph_path
    )

    if not graph.is_multigraph():

        raise RuntimeError(
            "Expected prepared Enron graph to be "
            "a MultiGraph/MultiDiGraph."
        )

    print(
        f"Graph nodes: "
        f"{graph.number_of_nodes():,}"
    )

    print(
        f"Graph edges: "
        f"{graph.number_of_edges():,}"
    )

    index = build_enron_index(
        graph
    )

    all_clean = []

    all_private = []

    all_representation = []

    all_metadata = []

    for split_seed in split_seeds:

        split_folder = os.path.join(
            experiment_folder,
            f"split_seed_{split_seed}",
        )

        result = run_split(
            graph=graph,
            index=index,
            split_seed=split_seed,
            representations=representations,
            epsilons=epsilons,
            mechanism_seeds=mechanism_seeds,
            test_positives=test_positives,
            C=C,
            rho_signature=rho_signature,
            sensitivity_mode=sensitivity_mode,
            d_max=d_max,
            degree_weight=degree_weight,
            split_folder=split_folder,
        )

        all_clean.append(
            result[
                "clean"
            ]
        )

        all_private.append(
            result[
                "private"
            ]
        )

        all_representation.append(
            result[
                "representation_diagnostics"
            ]
        )

        all_metadata.append(
            result[
                "metadata"
            ]
        )

    clean_dataframe = pd.concat(
        all_clean,
        ignore_index=True,
    )

    private_dataframe = pd.concat(
        all_private,
        ignore_index=True,
    )

    representation_dataframe = pd.concat(
        all_representation,
        ignore_index=True,
    )

    split_mean_dataframe = average_mechanism_seeds(
        private_dataframe
    )

    aggregate_dataframe = aggregate_across_splits(
        split_mean_dataframe
    )

    clean_path = os.path.join(
        experiment_folder,
        "enron_private_clean_metrics.csv",
    )

    private_path = os.path.join(
        experiment_folder,
        "enron_private_all_runs.csv",
    )

    split_mean_path = os.path.join(
        experiment_folder,
        "enron_private_split_means.csv",
    )

    aggregate_path = os.path.join(
        experiment_folder,
        "enron_private_aggregate.csv",
    )

    representation_path = os.path.join(
        experiment_folder,
        "enron_private_representation_diagnostics.csv",
    )

    metadata_path = os.path.join(
        experiment_folder,
        "enron_private_metadata.json",
    )

    clean_dataframe.to_csv(
        clean_path,
        index=False,
    )

    private_dataframe.to_csv(
        private_path,
        index=False,
    )

    split_mean_dataframe.to_csv(
        split_mean_path,
        index=False,
    )

    aggregate_dataframe.to_csv(
        aggregate_path,
        index=False,
    )

    representation_dataframe.to_csv(
        representation_path,
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
            "enron"
        ),
        "graph": (
            graph_path
        ),
        "target_relation": (
            TARGET_RELATION
        ),
        "holdout": (
            "Criterion C"
        ),
        "holdout_definition": (
            "Every content shared by a test-positive pair "
            "has exactly the two test endpoints as its "
            "distinct client neighbours."
        ),
        "fixed_test_positives": int(
            test_positives
        ),
        "representations": list(
            representations
        ),
        "primary_representation": (
            "hks_d12_global"
        ),
        "primary_orientation": (
            "distance"
        ),
        "orientations": (
            frozen_orientations
        ),
        "epsilons_total": [
            float(
                value
            )
            for value in epsilons
        ],
        "split_seeds": [
            int(
                value
            )
            for value in split_seeds
        ],
        "mechanism_seeds": [
            int(
                value
            )
            for value in mechanism_seeds
        ],
        "clip_C": float(C),
        "rho_signature": float(
            rho_signature
        ),
        "sensitivity_mode": (
            sensitivity_mode
        ),
        "privacy_adjacency": (
            "complete client-report replacement"
        ),
        "privacy_mechanism": (
            "pure L2 K-norm"
        ),
        "actual_privacy_guarantee": (
            "epsilon_signature-LDP for this experiment"
        ),
        "epsilon_total_interpretation": (
            "Composed full-pipeline reference budget. "
            "Only rho_signature*epsilon_total is actually "
            "consumed here."
        ),
        "analytical_nonnegative_report_sensitivity": (
            "sqrt(2)C"
        ),
        "generic_report_sensitivity": (
            "2C"
        ),
        "empirical_sensitivity_used_for_calibration": (
            False
        ),
        "one_private_report_per_client_per_run": (
            True
        ),
        "final_inference_unit": (
            "graph split after averaging mechanism seeds"
        ),
        "runs": (
            all_metadata
        ),
        "runtime_seconds": float(
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
        "ENRON PRIVATE CRITERION-C EXPERIMENT COMPLETE"
    )

    print(
        "=" * 80
    )

    print()
    print(
        "Final aggregate across graph splits:"
    )

    summary_columns = [
        "representation",
        "orientation",
        "epsilon_total",
        "epsilon_signature",
        "clean_auc__mean",
        "private_auc__mean",
        "private_auc__ci95_low",
        "private_auc__ci95_high",
        "clean_ap__mean",
        "private_ap__mean",
        "private_ap__ci95_low",
        "private_ap__ci95_high",
        "auc_change__mean",
        "ap_change__mean",
        "clean_private_pair_spearman__mean",
        "clean_private_report_cosine_mean__mean",
        "theoretical_mean_noise_norm",
        "observed_mean_noise_norm__mean",
    ]

    print(
        aggregate_dataframe[
            summary_columns
        ].to_string(
            index=False
        )
    )

    print()
    print(
        "FINAL PROTOCOL CHECKS:"
    )

    print(
        "  1. Positive population is Criterion C only."
    )

    print(
        "  2. Test size is fixed before private evaluation."
    )

    print(
        "  3. Randomized target spanning forest is preserved."
    )

    print(
        "  4. All direct held-out shared_text evidence is removed."
    )

    print(
        "  5. All held-out common-content evidence is removed."
    )

    print(
        "  6. Exact validation requires zero non-test "
        "physical target loss."
    )

    print(
        "  7. Exact validation requires zero non-test "
        "common-content pair loss."
    )

    print(
        "  8. HKS distance and Eigen similarity orientations "
        "are frozen."
    )

    print(
        "  9. Every client releases one report per private run."
    )

    print(
        " 10. Privacy uses analytical complete-report sensitivity."
    )

    print(
        " 11. Empirical sensitivity is never used for calibration."
    )

    print(
        " 12. Mechanism seeds are averaged within graph split "
        "before final inference."
    )

    print()
    print(
        "Outputs:"
    )

    print(
        f"  {clean_path}"
    )

    print(
        f"  {private_path}"
    )

    print(
        f"  {split_mean_path}"
    )

    print(
        f"  {aggregate_path}"
    )

    print(
        f"  {representation_path}"
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
        "clean": (
            clean_dataframe
        ),
        "private_runs": (
            private_dataframe
        ),
        "split_means": (
            split_mean_dataframe
        ),
        "aggregate": (
            aggregate_dataframe
        ),
        "representation_diagnostics": (
            representation_dataframe
        ),
        "metadata": (
            metadata
        ),
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Final Criterion-C leakage-safe private "
            "Enron client-client link prediction."
        )
    )

    parser.add_argument(
        "--output-folder",
        type=str,
        default=DEFAULT_OUTPUT_FOLDER,
    )

    parser.add_argument(
        "--representations",
        nargs="+",
        choices=list(
            DEFAULT_REPRESENTATIONS
        ),
        default=list(
            DEFAULT_REPRESENTATIONS
        ),
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
        "--split-seeds",
        nargs="+",
        type=int,
        default=list(
            DEFAULT_SPLIT_SEEDS
        ),
    )

    parser.add_argument(
        "--mechanism-seeds",
        nargs="+",
        type=int,
        default=list(
            DEFAULT_MECHANISM_SEEDS
        ),
    )

    parser.add_argument(
        "--test-positives",
        type=int,
        default=DEFAULT_TEST_POSITIVES,
    )

    parser.add_argument(
        "--C",
        type=float,
        default=DEFAULT_C,
    )

    parser.add_argument(
        "--rho-signature",
        type=float,
        default=DEFAULT_RHO_SIGNATURE,
    )

    parser.add_argument(
        "--sensitivity-mode",
        choices=[
            "nonnegative_report",
            "generic_report",
        ],
        default=DEFAULT_SENSITIVITY_MODE,
    )

    parser.add_argument(
        "--Dmax",
        type=int,
        default=DEFAULT_D_MAX,
    )

    parser.add_argument(
        "--degree-weight",
        type=float,
        default=DEFAULT_DEGREE_WEIGHT,
    )

    args = parser.parse_args()

    run_experiment(
        output_folder=args.output_folder,
        representations=args.representations,
        epsilons=args.epsilons,
        split_seeds=args.split_seeds,
        mechanism_seeds=args.mechanism_seeds,
        test_positives=args.test_positives,
        C=args.C,
        rho_signature=args.rho_signature,
        sensitivity_mode=args.sensitivity_mode,
        d_max=args.Dmax,
        degree_weight=args.degree_weight,
    )
