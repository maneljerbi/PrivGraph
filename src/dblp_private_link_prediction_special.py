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


SCHEMA_VERSION = "dblp-private-link-special-1.0"

DEFAULT_OUTPUT_FOLDER = "output_local_test"

DEFAULT_COHORTS = (
    "provided_id_only",
    "all_authors",
)

DEFAULT_REPRESENTATIONS = (
    "eigen_d8_block",
    "eigen_d12_global",
    "hks_d12_global",
)

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

DEFAULT_TEST_POSITIVES = 500

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

    first = str(
        first
    )

    second = str(
        second
    )

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


def build_dblp_index(
    graph,
):

    print()

    print(
        "Building DBLP coauthor index..."
    )

    author_papers = defaultdict(
        set
    )

    paper_authors = defaultdict(
        set
    )

    authored_edges = 0

    for (
        source,
        target,
        key,
        attributes,
    ) in graph.edges(
        keys=True,
        data=True,
    ):

        relation = str(
            attributes.get(
                "relation",
                "",
            )
        )

        if relation != "authored":

            continue

        source_type = str(
            graph.nodes[
                source
            ].get(
                "type",
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
            source_type == "author"
            and
            target_type == "paper"
        ):

            author_id = str(
                source
            )

            paper_id = str(
                target
            )

        elif (
            source_type == "paper"
            and
            target_type == "author"
        ):


            author_id = str(
                target
            )

            paper_id = str(
                source
            )

        else:

            continue

        authored_edges += 1

        author_papers[
            author_id
        ].add(
            paper_id
        )

        paper_authors[
            paper_id
        ].add(
            author_id
        )

    if not author_papers:

        raise RuntimeError(
            "No DBLP authored relations were found."
        )


    pair_papers = defaultdict(
        set
    )

    multi_author_papers = 0

    exactly_two_author_papers = 0

    coauthor_occurrences = 0

    for paper_id, authors in paper_authors.items():

        authors = sorted(
            authors
        )

        if len(
            authors
        ) < 2:

            continue

        multi_author_papers += 1

        if len(
            authors
        ) == 2:

            exactly_two_author_papers += 1

        for first_index in range(
            len(
                authors
            )
        ):

            for second_index in range(
                first_index + 1,
                len(
                    authors
                ),
            ):

                pair = canonical_pair(
                    authors[
                        first_index
                    ],
                    authors[
                        second_index
                    ],
                )

                pair_papers[
                    pair
                ].add(
                    paper_id
                )

                coauthor_occurrences += 1

    coauthor_pairs = set(
        pair_papers.keys()
    )

    strict_pairs = {}

    for pair, shared_papers in pair_papers.items():

        if len(
            shared_papers
        ) != 1:

            continue

        paper_id = next(
            iter(
                shared_papers
            )
        )

        if len(
            paper_authors[
                paper_id
            ]
        ) != 2:

            continue

        strict_pairs[
            pair
        ] = paper_id

    print(
        f"  authored physical edges: "
        f"{authored_edges:,}"
    )

    print(
        f"  authors with papers: "
        f"{len(author_papers):,}"
    )

    print(
        f"  papers with authored edges: "
        f"{len(paper_authors):,}"
    )

    print(
        f"  multi-author papers: "
        f"{multi_author_papers:,}"
    )

    print(
        f"  exactly-two-author papers: "
        f"{exactly_two_author_papers:,}"
    )

    print(
        f"  unique coauthor pairs: "
        f"{len(coauthor_pairs):,}"
    )

    print(
        f"  coauthor pair occurrences: "
        f"{coauthor_occurrences:,}"
    )

    print(
        f"  strict unique-paper dyadic pairs: "
        f"{len(strict_pairs):,}"
    )

    return {
        "author_papers": (
            author_papers
        ),

        "paper_authors": (
            paper_authors
        ),

        "pair_papers": (
            pair_papers
        ),

        "coauthor_pairs": (
            coauthor_pairs
        ),

        "strict_pairs": (
            strict_pairs
        ),

        "authored_edges": int(
            authored_edges
        ),

        "multi_author_papers": int(
            multi_author_papers
        ),

        "exactly_two_author_papers": int(
            exactly_two_author_papers
        ),

        "coauthor_occurrences": int(
            coauthor_occurrences
        ),
    }


def get_cohort_authors(
    graph,
    cohort,
):

    authors = []

    for node_id, attributes in graph.nodes(
        data=True
    ):

        if str(
            attributes.get(
                "type",
                "",
            )
        ) != "author":

            continue

        if cohort == "all_authors":

            authors.append(
                str(
                    node_id
                )
            )

            continue

        if cohort == "provided_id_only":

            if str(
                attributes.get(
                    "identity_source",
                    "",
                )
            ) == "provided_id":

                authors.append(
                    str(
                        node_id
                    )
                )

            continue

        raise ValueError(
            f"Unsupported DBLP cohort: "
            f"{cohort}"
        )

    authors = sorted(
        set(
            authors
        )
    )

    if not authors:

        raise RuntimeError(
            f"No authors found for cohort "
            f"{cohort}."
        )

    return authors


def restrict_index_to_cohort(
    full_index,
    cohort_authors,
):

    cohort_set = set(
        cohort_authors
    )

    coauthor_pairs = {
        pair
        for pair
        in full_index[
            "coauthor_pairs"
        ]
        if (
            pair[
                0
            ]
            in cohort_set
            and
            pair[
                1
            ]
            in cohort_set
        )
    }

    strict_pairs = {
        pair: paper_id
        for pair, paper_id
        in full_index[
            "strict_pairs"
        ].items()
        if (
            pair[
                0
            ]
            in cohort_set
            and
            pair[
                1
            ]
            in cohort_set
        )
    }

    return {
        "authors": (
            cohort_authors
        ),

        "author_set": (
            cohort_set
        ),

        "coauthor_pairs": (
            coauthor_pairs
        ),

        "strict_pairs": (
            strict_pairs
        ),
    }


class UnionFind:

    def __init__(
        self,
        elements,
    ):

        self.parent = {
            element: element
            for element
            in elements
        }

        self.rank = {
            element: 0
            for element
            in elements
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

        root_first = self.find(
            first
        )

        root_second = self.find(
            second
        )

        if root_first == root_second:

            return False

        rank_first = self.rank[
            root_first
        ]

        rank_second = self.rank[
            root_second
        ]

        if rank_first < rank_second:

            root_first, root_second = (
                root_second,
                root_first,
            )

        self.parent[
            root_second
        ] = root_first

        if rank_first == rank_second:

            self.rank[
                root_first
            ] += 1

        return True


def build_random_spanning_forest(
    authors,
    coauthor_pairs,
    cohort,
    split_seed,
):


    edges = sorted(
        coauthor_pairs
    )

    rng = np.random.default_rng(
        deterministic_seed(
            "dblp",
            cohort,
            split_seed,
            "spanning_forest",
        )
    )

    if edges:

        permutation = rng.permutation(
            len(
                edges
            )
        )

        edges = [
            edges[
                int(
                    index
                )
            ]
            for index
            in permutation
        ]

    union_find = UnionFind(
        authors
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


def sample_test_positives(
    strict_pairs,
    forest_pairs,
    requested,
    cohort,
    split_seed,
):

    candidates = sorted(
        pair
        for pair
        in strict_pairs
        if pair not in forest_pairs
    )

    requested = int(
        requested
    )

    if len(
        candidates
    ) < requested:

        raise RuntimeError(
            "Not enough strict DBLP test-positive pairs.\n"
            f"  cohort={cohort}\n"
            f"  split_seed={split_seed}\n"
            f"  requested={requested:,}\n"
            f"  available={len(candidates):,}\n"
            "Use one common smaller test size for every "
            "cohort and split."
        )

    rng = np.random.default_rng(
        deterministic_seed(
            "dblp",
            cohort,
            split_seed,
            "test_positive_sample",
        )
    )

    selected_indices = rng.choice(
        len(
            candidates
        ),
        size=requested,
        replace=False,
    )

    selected = [
        candidates[
            int(
                index
            )
        ]
        for index
        in selected_indices
    ]

    return {
        "pairs": (
            selected
        ),

        "candidate_count": int(
            len(
                candidates
            )
        ),
    }


def sample_test_negatives(
    cohort_authors,
    full_coauthor_pairs,
    positive_pairs,
    requested,
    cohort,
    split_seed,
):

    requested = int(
        requested
    )

    full_coauthor_pairs = set(
        full_coauthor_pairs
    )

    positive_pairs = set(
        positive_pairs
    )

    authors = list(
        sorted(
            cohort_authors
        )
    )

    if len(
        authors
    ) < 2:

        raise RuntimeError(
            "Not enough authors for negative sampling."
        )

    rng = np.random.default_rng(
        deterministic_seed(
            "dblp",
            cohort,
            split_seed,
            "negative_sample",
        )
    )

    negatives = set()

    attempts = 0

    maximum_attempts = max(
        requested * 1000,
        100000,
    )

    while len(
        negatives
    ) < requested:

        attempts += 1

        if attempts > maximum_attempts:

            raise RuntimeError(
                "Could not sample enough true DBLP negatives."
            )

        first_index = int(
            rng.integers(
                0,
                len(
                    authors
                ),
            )
        )

        second_index = int(
            rng.integers(
                0,
                len(
                    authors
                ),
            )
        )

        if first_index == second_index:

            continue

        pair = canonical_pair(
            authors[
                first_index
            ],
            authors[
                second_index
            ],
        )

        if pair in full_coauthor_pairs:

            continue

        if pair in positive_pairs:

            continue

        negatives.add(
            pair
        )

    return sorted(
        negatives
    )


def authored_edge_records(
    graph,
    author_id,
    paper_id,
):

    records = []

    author_id = str(
        author_id
    )

    paper_id = str(
        paper_id
    )


    if (
        author_id in graph
        and
        paper_id in graph
    ):

        if graph.has_edge(
            author_id,
            paper_id,
        ):

            edge_data = graph.get_edge_data(
                author_id,
                paper_id,
            )

            for key, attributes in edge_data.items():

                if str(
                    attributes.get(
                        "relation",
                        "",
                    )
                ) == "authored":

                    records.append({
                        "source": (
                            author_id
                        ),

                        "target": (
                            paper_id
                        ),

                        "key": (
                            key
                        ),

                        "attributes": dict(
                            attributes
                        ),
                    })


        if graph.has_edge(
            paper_id,
            author_id,
        ):

            edge_data = graph.get_edge_data(
                paper_id,
                author_id,
            )

            for key, attributes in edge_data.items():

                if str(
                    attributes.get(
                        "relation",
                        "",
                    )
                ) == "authored":

                    records.append({
                        "source": (
                            paper_id
                        ),

                        "target": (
                            author_id
                        ),

                        "key": (
                            key
                        ),

                        "attributes": dict(
                            attributes
                        ),
                    })

    return records


def choose_removed_author(
    pair,
    cohort,
    split_seed,
):

    first, second = pair

    value = deterministic_seed(
        "dblp",
        cohort,
        split_seed,
        first,
        second,
        "removed_endpoint",
    )

    if (
        value
        %
        2
    ) == 0:

        return first

    return second


def apply_holdout(
    graph,
    positive_pairs,
    strict_pairs,
    cohort,
    split_seed,
):

    removed_edges = []

    diagnostics = []

    for pair_index, pair in enumerate(
        positive_pairs
    ):

        paper_id = strict_pairs[
            pair
        ]

        removed_author = choose_removed_author(
            pair,
            cohort,
            split_seed,
        )

        records = authored_edge_records(
            graph,
            removed_author,
            paper_id,
        )

        if len(
            records
        ) != 1:

            raise RuntimeError(
                "Expected exactly one physical authored edge "
                "for the DBLP strict holdout.\n"
                f"  pair={pair}\n"
                f"  paper={paper_id}\n"
                f"  removed_author={removed_author}\n"
                f"  found={len(records)}"
            )

        record = records[
            0
        ]

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

        diagnostics.append({
            "pair_index": int(
                pair_index
            ),

            "author_a": (
                pair[
                    0
                ]
            ),

            "author_b": (
                pair[
                    1
                ]
            ),

            "paper": (
                paper_id
            ),

            "removed_author": (
                removed_author
            ),

            "removed_edge_source": (
                record[
                    "source"
                ]
            ),

            "removed_edge_target": (
                record[
                    "target"
                ]
            ),

            "removed_edge_key": str(
                record[
                    "key"
                ]
            ),
        })

    return {
        "removed_edges": (
            removed_edges
        ),

        "diagnostics": (
            diagnostics
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


def current_author_papers(
    graph,
    author_id,
):

    papers = set()

    author_id = str(
        author_id
    )

    for _, target, attributes in graph.out_edges(
        author_id,
        data=True,
    ):

        if str(
            attributes.get(
                "relation",
                "",
            )
        ) != "authored":

            continue

        if str(
            graph.nodes[
                target
            ].get(
                "type",
                "",
            )
        ) == "paper":

            papers.add(
                str(
                    target
                )
            )

    for source, _, attributes in graph.in_edges(
        author_id,
        data=True,
    ):

        if str(
            attributes.get(
                "relation",
                "",
            )
        ) != "authored":

            continue

        if str(
            graph.nodes[
                source
            ].get(
                "type",
                "",
            )
        ) == "paper":

            papers.add(
                str(
                    source
                )
            )

    return papers


def validate_holdout(
    graph,
    positive_pairs,
    forest_pairs,
):

    remaining_shared_papers = 0

    forest_overlap = 0

    for first, second in positive_pairs:

        first_papers = current_author_papers(
            graph,
            first,
        )

        second_papers = current_author_papers(
            graph,
            second,
        )

        shared = (
            first_papers
            &
            second_papers
        )

        if shared:

            remaining_shared_papers += 1

        if canonical_pair(
            first,
            second,
        ) in forest_pairs:

            forest_overlap += 1

    if remaining_shared_papers != 0:

        raise RuntimeError(
            "DBLP leakage validation failed: "
            f"{remaining_shared_papers} held-out pairs "
            "still share a paper."
        )

    if forest_overlap != 0:

        raise RuntimeError(
            "DBLP split validation failed: "
            f"{forest_overlap} test pairs belong to the "
            "spanning forest."
        )

    return {
        "heldout_pairs_still_sharing_paper": int(
            remaining_shared_papers
        ),

        "forest_test_overlap": int(
            forest_overlap
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
            f"{representation}: expected a matrix."
        )

    if not np.all(
        np.isfinite(
            raw_matrix
        )
    ):

        raise RuntimeError(
            f"{representation}: NaN/Inf descriptor value."
        )

    substantive_negative = (
        raw_matrix
        <
        -NEGATIVE_TOLERANCE
    )

    substantive_negative_count = int(
        np.sum(
            substantive_negative
        )
    )

    all_negative_count = int(
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
                "negative values and cannot use the "
                "sqrt(2)C nonnegative-report bound.\n"
                f"Negative entries below "
                f"-{NEGATIVE_TOLERANCE}: "
                f"{substantive_negative_count:,}"
            )


        query = np.maximum(
            raw_matrix,
            0.0,
        )

    elif sensitivity_mode == "generic_report":


        query = np.array(
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
        query,
        axis=1,
    )

    clip_mask = (
        norms
        >
        float(
            C
        )
    )

    clipped = np.array(
        query,
        dtype=float,
        copy=True,
    )

    if np.any(
        clip_mask
    ):

        clipped[
            clip_mask
        ] *= (
            float(
                C
            )
            /
            norms[
                clip_mask
            ]
        )[
            :,
            None
        ]

    clipped_norms = np.linalg.norm(
        clipped,
        axis=1,
    )

    rounded = np.round(
        clipped,
        decimals=SCORE_ROUND_DECIMALS,
    )

    unique_rows = int(
        np.unique(
            rounded,
            axis=0,
        ).shape[
            0
        ]
    )

    return {
        "matrix": (
            clipped
        ),

        "dimension": int(
            clipped.shape[
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
            all_negative_count
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
            unique_rows
        ),

        "mean_component_variance": float(
            np.mean(
                np.var(
                    clipped,
                    axis=0,
                )
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
            float(
                C
            )
        )

    if sensitivity_mode == "generic_report":

        return float(
            2.0
            *
            float(
                C
            )
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
            0.0,
            1.0,
            size=int(
                dimension
            ),
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
    epsilon_signature,
    sensitivity,
    cohort,
    split_seed,
    representation,
    mechanism_seed,
):

    clean_matrix = np.asarray(
        clean_matrix,
        dtype=float,
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

    for row_index in range(
        n_clients
    ):

        rng = np.random.default_rng(
            deterministic_seed(
                "dblp",
                cohort,
                split_seed,
                representation,
                mechanism_seed,
                row_index,
                "private_report",
            )
        )

        direction = sample_unit_direction(
            dimension,
            rng,
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

        noise = (
            radius
            *
            direction
        )

        private_matrix[
            row_index
        ] = (
            clean_matrix[
                row_index
            ]
            +
            noise
        )

        noise_norms[
            row_index
        ] = radius

    return {
        "matrix": (
            private_matrix
        ),

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
        matrix,
        dtype=float,
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


def score_pairs(
    matrix,
    client_ids,
    pairs,
):

    normalized = normalize_rows(
        matrix
    )

    index = {
        str(
            client_id
        ): row_index
        for row_index, client_id
        in enumerate(
            client_ids
        )
    }

    scores = []

    for first, second in pairs:

        first_index = index.get(
            str(
                first
            )
        )

        second_index = index.get(
            str(
                second
            )
        )

        if (
            first_index is None
            or
            second_index is None
        ):

            raise RuntimeError(
                "Missing descriptor for DBLP evaluation pair."
            )

        value = float(
            np.dot(
                normalized[
                    first_index
                ],
                normalized[
                    second_index
                ],
            )
        )

        scores.append(
            float(
                np.clip(
                    value,
                    -1.0,
                    1.0,
                )
            )
        )

    return np.asarray(
        scores,
        dtype=float,
    )


def stabilize_scores(
    values,
):

    return np.round(
        np.asarray(
            values,
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

    if (
        len(
            first
        ) < 3
        or
        np.std(
            first
        ) <= 1e-15
        or
        np.std(
            second
        ) <= 1e-15
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
    full_index,
    cohort,
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
        f"{cohort} | split seed {split_seed}"
    )

    print(
        "#" * 80
    )

    cohort_authors = get_cohort_authors(
        graph,
        cohort,
    )

    cohort_index = restrict_index_to_cohort(
        full_index,
        cohort_authors,
    )

    print(
        f"Authors in cohort: "
        f"{len(cohort_authors):,}"
    )

    print(
        f"Coauthor pairs in cohort: "
        f"{len(cohort_index['coauthor_pairs']):,}"
    )

    print(
        f"Strict dyadic pairs in cohort: "
        f"{len(cohort_index['strict_pairs']):,}"
    )


    forest_pairs = build_random_spanning_forest(
        authors=cohort_authors,

        coauthor_pairs=(
            cohort_index[
                "coauthor_pairs"
            ]
        ),

        cohort=cohort,

        split_seed=split_seed,
    )

    positive_result = sample_test_positives(
        strict_pairs=(
            cohort_index[
                "strict_pairs"
            ]
        ),

        forest_pairs=(
            forest_pairs
        ),

        requested=test_positives,

        cohort=cohort,

        split_seed=split_seed,
    )

    positive_pairs = positive_result[
        "pairs"
    ]

    negative_pairs = sample_test_negatives(
        cohort_authors=(
            cohort_authors
        ),

        full_coauthor_pairs=(
            cohort_index[
                "coauthor_pairs"
            ]
        ),

        positive_pairs=(
            positive_pairs
        ),

        requested=len(
            positive_pairs
        ),

        cohort=cohort,

        split_seed=split_seed,
    )

    print(
        f"Forced forest pairs: "
        f"{len(forest_pairs):,}"
    )

    print(
        "Strict candidates after forest exclusion: "
        f"{positive_result['candidate_count']:,}"
    )

    print(
        f"Test positives: "
        f"{len(positive_pairs):,}"
    )

    print(
        f"Test negatives: "
        f"{len(negative_pairs):,}"
    )


    intervention = None

    try:

        intervention = apply_holdout(
            graph=graph,

            positive_pairs=(
                positive_pairs
            ),

            strict_pairs=(
                cohort_index[
                    "strict_pairs"
                ]
            ),

            cohort=cohort,

            split_seed=split_seed,
        )

        validation = validate_holdout(
            graph=graph,

            positive_pairs=(
                positive_pairs
            ),

            forest_pairs=(
                forest_pairs
            ),
        )

        print(
            "Held-out pairs still sharing a paper: "
            f"{validation['heldout_pairs_still_sharing_paper']}"
        )

        print(
            "Forest/test overlap: "
            f"{validation['forest_test_overlap']}"
        )

        print(
            "Physical authored edges removed: "
            f"{len(intervention['removed_edges']):,}"
        )

        print(
            "Collateral coauthor removal by construction: 0"
        )


        required_authors = sorted(
            set(
                author
                for pair
                in (
                    list(
                        positive_pairs
                    )
                    +
                    list(
                        negative_pairs
                    )
                )
                for author
                in pair
            )
        )

        print(
            "Authors requiring descriptor reports: "
            f"{len(required_authors):,}"
        )


        generated = generate_selected_descriptors(
            training_graph=graph,

            client_ids=(
                required_authors
            ),

            d_max=d_max,

            degree_weight=(
                degree_weight
            ),
        )

        failures = generated[
            "failures"
        ]

        if failures:

            raise RuntimeError(
                "DBLP descriptor generation failed for "
                f"{len(failures)} authors."
            )

        descriptor_client_ids = [
            str(
                value
            )
            for value
            in generated[
                "client_ids"
            ]
        ]

        if (
            descriptor_client_ids
            !=
            required_authors
        ):

            raise RuntimeError(
                "Descriptor author ordering differs from "
                "the requested deterministic ordering."
            )

        descriptor_matrices = generated[
            "descriptors"
        ]


        os.makedirs(
            split_folder,
            exist_ok=True,
        )

        pair_rows = []

        for pair_index, (
            first,
            second,
        ) in enumerate(
            positive_pairs
        ):

            pair_rows.append({
                "pair_index": int(
                    pair_index
                ),

                "label": 1,

                "author_a": (
                    first
                ),

                "author_b": (
                    second
                ),
            })

        for pair_index, (
            first,
            second,
        ) in enumerate(
            negative_pairs
        ):

            pair_rows.append({
                "pair_index": int(
                    pair_index
                ),

                "label": 0,

                "author_a": (
                    first
                ),

                "author_b": (
                    second
                ),
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
                "diagnostics"
            ]
        ).to_csv(
            os.path.join(
                split_folder,
                "holdout_diagnostics.csv",
            ),
            index=False,
        )

        clean_rows = []

        private_rows = []

        report_diagnostic_rows = []

        pair_score_rows = []


        for representation in representations:

            print()

            print(
                f"Representation: "
                f"{representation}"
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

                sensitivity_mode=(
                    sensitivity_mode
                ),

                representation=(
                    representation
                ),
            )

            clean_matrix = query[
                "matrix"
            ]

            sensitivity = analytical_sensitivity(
                C=C,

                sensitivity_mode=(
                    sensitivity_mode
                ),
            )

            dimension = int(
                query[
                    "dimension"
                ]
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

            report_diagnostic_rows.append({
                "cohort": (
                    cohort
                ),

                "split_seed": int(
                    split_seed
                ),

                "representation": (
                    representation
                ),

                "authors": int(
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

                "sensitivity_l2": float(
                    sensitivity
                ),
            })


            clean_positive_scores = score_pairs(
                matrix=clean_matrix,

                client_ids=(
                    descriptor_client_ids
                ),

                pairs=(
                    positive_pairs
                ),
            )

            clean_negative_scores = score_pairs(
                matrix=clean_matrix,

                client_ids=(
                    descriptor_client_ids
                ),

                pairs=(
                    negative_pairs
                ),
            )

            clean_metrics = binary_metrics(
                clean_positive_scores,
                clean_negative_scores,
            )

            clean_rows.append({
                "cohort": (
                    cohort
                ),

                "split_seed": int(
                    split_seed
                ),

                "representation": (
                    representation
                ),

                "orientation": (
                    "similarity"
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

                "clean_unique_scores": int(
                    clean_metrics[
                        "unique_scores"
                    ]
                ),
            })

            print(
                f"  clean AUC="
                f"{clean_metrics['auc']:.6f}"
                " | AP="
                f"{clean_metrics['ap']:.6f}"
            )


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

                theoretical_mean_noise = (
                    dimension
                    *
                    sensitivity
                    /
                    epsilon_signature
                )

                epsilon_private_auc = []

                epsilon_private_ap = []

                for mechanism_seed in mechanism_seeds:

                    private_result = privatize_matrix(
                        clean_matrix=(
                            clean_matrix
                        ),

                        epsilon_signature=(
                            epsilon_signature
                        ),

                        sensitivity=(
                            sensitivity
                        ),

                        cohort=cohort,

                        split_seed=(
                            split_seed
                        ),

                        representation=(
                            representation
                        ),

                        mechanism_seed=(
                            mechanism_seed
                        ),
                    )

                    private_matrix = private_result[
                        "matrix"
                    ]

                    noise_norms = private_result[
                        "noise_norms"
                    ]

                    private_positive_scores = score_pairs(
                        matrix=private_matrix,

                        client_ids=(
                            descriptor_client_ids
                        ),

                        pairs=(
                            positive_pairs
                        ),
                    )

                    private_negative_scores = score_pairs(
                        matrix=private_matrix,

                        client_ids=(
                            descriptor_client_ids
                        ),

                        pairs=(
                            negative_pairs
                        ),
                    )

                    private_metrics = binary_metrics(
                        private_positive_scores,
                        private_negative_scores,
                    )

                    private_all_scores = np.concatenate(
                        [
                            private_positive_scores,
                            private_negative_scores,
                        ]
                    )

                    report_cosines = rowwise_cosine(
                        clean_matrix,
                        private_matrix,
                    )

                    pair_spearman = safe_correlation(
                        stabilize_scores(
                            clean_all_scores
                        ),

                        stabilize_scores(
                            private_all_scores
                        ),

                        "spearman",
                    )

                    pair_pearson = safe_correlation(
                        stabilize_scores(
                            clean_all_scores
                        ),

                        stabilize_scores(
                            private_all_scores
                        ),

                        "pearson",
                    )

                    pair_score_mae = float(
                        np.mean(
                            np.abs(
                                stabilize_scores(
                                    clean_all_scores
                                )
                                -
                                stabilize_scores(
                                    private_all_scores
                                )
                            )
                        )
                    )

                    row = {
                        "cohort": (
                            cohort
                        ),

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
                            "similarity"
                        ),

                        "epsilon_total": float(
                            epsilon_total
                        ),

                        "rho_signature": float(
                            rho_signature
                        ),

                        "epsilon_signature": float(
                            epsilon_signature
                        ),

                        "sensitivity_mode": (
                            sensitivity_mode
                        ),

                        "sensitivity_l2": float(
                            sensitivity
                        ),

                        "clip_C": float(
                            C
                        ),

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

                        "private_positive_score_mean": float(
                            private_metrics[
                                "positive_score_mean"
                            ]
                        ),

                        "private_negative_score_mean": float(
                            private_metrics[
                                "negative_score_mean"
                            ]
                        ),

                        "private_unique_scores": int(
                            private_metrics[
                                "unique_scores"
                            ]
                        ),

                        "clean_private_pair_spearman": (
                            pair_spearman
                        ),

                        "clean_private_pair_pearson": (
                            pair_pearson
                        ),

                        "clean_private_pair_score_mae": float(
                            pair_score_mae
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
                            np.mean(
                                noise_norms
                            )
                        ),

                        "observed_median_noise_norm": float(
                            np.median(
                                noise_norms
                            )
                        ),
                    }

                    private_rows.append(
                        row
                    )

                    epsilon_private_auc.append(
                        private_metrics[
                            "auc"
                        ]
                    )

                    epsilon_private_ap.append(
                        private_metrics[
                            "ap"
                        ]
                    )


                    for pair_index, score in enumerate(
                        private_positive_scores
                    ):

                        pair_score_rows.append({
                            "cohort": (
                                cohort
                            ),

                            "split_seed": int(
                                split_seed
                            ),

                            "mechanism_seed": int(
                                mechanism_seed
                            ),

                            "representation": (
                                representation
                            ),

                            "epsilon_total": float(
                                epsilon_total
                            ),

                            "pair_index": int(
                                pair_index
                            ),

                            "label": 1,

                            "private_score_raw": float(
                                score
                            ),

                            "private_score_stabilized": float(
                                np.round(
                                    score,
                                    SCORE_ROUND_DECIMALS,
                                )
                            ),
                        })

                    for pair_index, score in enumerate(
                        private_negative_scores
                    ):

                        pair_score_rows.append({
                            "cohort": (
                                cohort
                            ),

                            "split_seed": int(
                                split_seed
                            ),

                            "mechanism_seed": int(
                                mechanism_seed
                            ),

                            "representation": (
                                representation
                            ),

                            "epsilon_total": float(
                                epsilon_total
                            ),

                            "pair_index": int(
                                pair_index
                            ),

                            "label": 0,

                            "private_score_raw": float(
                                score
                            ),

                            "private_score_stabilized": float(
                                np.round(
                                    score,
                                    SCORE_ROUND_DECIMALS,
                                )
                            ),
                        })

                print(
                    f"    eps={epsilon_total:g}"
                    " | eps_sig="
                    f"{epsilon_signature:g}"
                    " | private AUC="
                    f"{np.mean(epsilon_private_auc):.6f}"
                    " | AP="
                    f"{np.mean(epsilon_private_ap):.6f}"
                    " | noise="
                    f"{theoretical_mean_noise:.6f}"
                )


        clean_dataframe = pd.DataFrame(
            clean_rows
        )

        private_dataframe = pd.DataFrame(
            private_rows
        )

        report_dataframe = pd.DataFrame(
            report_diagnostic_rows
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

        report_dataframe.to_csv(
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
                "dblp"
            ),

            "cohort": (
                cohort
            ),

            "split_seed": int(
                split_seed
            ),

            "authors_in_cohort": int(
                len(
                    cohort_authors
                )
            ),

            "coauthor_pairs": int(
                len(
                    cohort_index[
                        "coauthor_pairs"
                    ]
                )
            ),

            "strict_pairs": int(
                len(
                    cohort_index[
                        "strict_pairs"
                    ]
                )
            ),

            "forest_pairs": int(
                len(
                    forest_pairs
                )
            ),

            "strict_candidates_after_forest": int(
                positive_result[
                    "candidate_count"
                ]
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

            "heldout_pairs_still_sharing_paper": int(
                validation[
                    "heldout_pairs_still_sharing_paper"
                ]
            ),

            "forest_test_overlap": int(
                validation[
                    "forest_test_overlap"
                ]
            ),

            "physical_authored_edges_removed": int(
                len(
                    intervention[
                        "removed_edges"
                    ]
                )
            ),

            "collateral_coauthor_removal_by_construction": 0,

            "required_author_reports": int(
                len(
                    required_authors
                )
            ),

            "representations": list(
                representations
            ),

            "epsilons_total": [
                float(
                    value
                )
                for value
                in epsilons
            ],

            "mechanism_seeds": [
                int(
                    value
                )
                for value
                in mechanism_seeds
            ],

            "rho_signature": float(
                rho_signature
            ),

            "sensitivity_mode": (
                sensitivity_mode
            ),

            "clip_C": float(
                C
            ),

            "score_orientation": (
                "cosine similarity; frozen from clean DBLP study"
            ),

            "privacy_adjacency": (
                "complete client-report replacement"
            ),

            "privacy_mechanism": (
                "pure L2 K-norm"
            ),

            "privacy_calibration": (
                "sqrt(2)C for verified nonnegative clipped reports; "
                "2C for generic clipped reports"
            ),

            "privacy_accounting": (
                "epsilon_signature = rho_signature * epsilon_total. "
                "The unused remainder is reserved for the typed-degree "
                "release of the complete synthesis method."
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
                report_dataframe
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
        "cohort",
        "split_seed",
        "representation",
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

    grouped = private_runs.groupby(
        group_columns,
        sort=True,
    )

    for keys, group in grouped:

        (
            cohort,
            split_seed,
            representation,
            epsilon_total,
            epsilon_signature,
            sensitivity_mode,
        ) = keys

        row = {
            "cohort": (
                cohort
            ),

            "split_seed": int(
                split_seed
            ),

            "representation": (
                representation
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
            ):

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
                    ) > 1
                    else 0.0
                )

            else:

                row[
                    f"{metric}__mechanism_mean"
                ] = np.nan

                row[
                    f"{metric}__mechanism_std"
                ] = np.nan

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
        "cohort",
        "representation",
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

    grouped = split_means.groupby(
        group_columns,
        sort=True,
    )

    for keys, group in grouped:

        (
            cohort,
            representation,
            epsilon_total,
            epsilon_signature,
            sensitivity_mode,
        ) = keys

        row = {
            "cohort": (
                cohort
            ),

            "representation": (
                representation
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
    cohorts=DEFAULT_COHORTS,
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

    cohorts = tuple(
        str(
            value
        )
        for value
        in cohorts
    )

    representations = tuple(
        str(
            value
        )
        for value
        in representations
    )

    epsilons = tuple(
        float(
            value
        )
        for value
        in epsilons
    )

    split_seeds = tuple(
        int(
            value
        )
        for value
        in split_seeds
    )

    mechanism_seeds = tuple(
        int(
            value
        )
        for value
        in mechanism_seeds
    )

    if C <= 0.0:

        raise ValueError(
            "C must be strictly positive."
        )

    if not (
        0.0
        <
        rho_signature
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
        "dblp",
        "dblp_global_graph.gml.gz",
    )

    if not os.path.exists(
        graph_path
    ):

        raise FileNotFoundError(
            "Missing prepared DBLP graph:\n"
            f"  {graph_path}\n"
            "Run prepare_dblp_special.py first."
        )

    experiment_folder = os.path.join(
        output_folder,
        "dblp",
        "experiments",
        "dblp_private_link_prediction_special",
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
        "DBLP PRIVATE LEAKAGE-SAFE COAUTHOR PREDICTION"
    )

    print(
        "=" * 80
    )

    print(
        f"Cohorts: "
        f"{list(cohorts)}"
    )

    print(
        f"Representations: "
        f"{list(representations)}"
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
        f"Test positives per split: "
        f"{test_positives:,}"
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
        "Score orientation: cosine similarity "
        "(frozen from clean DBLP evaluation)"
    )

    print(
        "Privacy adjacency: complete client-report replacement"
    )

    print(
        "Mechanism: pure L2 K-norm"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Loading prepared DBLP graph..."
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

    full_index = build_dblp_index(
        graph
    )

    all_clean = []

    all_private = []

    all_representation_diagnostics = []

    all_metadata = []


    for cohort in cohorts:

        for split_seed in split_seeds:

            split_folder = os.path.join(
                experiment_folder,
                cohort,
                f"split_seed_{split_seed}",
            )

            result = run_split(
                graph=graph,

                full_index=(
                    full_index
                ),

                cohort=cohort,

                split_seed=(
                    split_seed
                ),

                representations=(
                    representations
                ),

                epsilons=(
                    epsilons
                ),

                mechanism_seeds=(
                    mechanism_seeds
                ),

                test_positives=(
                    test_positives
                ),

                C=C,

                rho_signature=(
                    rho_signature
                ),

                sensitivity_mode=(
                    sensitivity_mode
                ),

                d_max=d_max,

                degree_weight=(
                    degree_weight
                ),

                split_folder=(
                    split_folder
                ),
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

            all_representation_diagnostics.append(
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
        all_representation_diagnostics,
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
        "dblp_private_clean_metrics.csv",
    )

    private_path = os.path.join(
        experiment_folder,
        "dblp_private_all_runs.csv",
    )

    split_mean_path = os.path.join(
        experiment_folder,
        "dblp_private_split_means.csv",
    )

    aggregate_path = os.path.join(
        experiment_folder,
        "dblp_private_aggregate.csv",
    )

    representation_path = os.path.join(
        experiment_folder,
        "dblp_private_representation_diagnostics.csv",
    )

    metadata_path = os.path.join(
        experiment_folder,
        "dblp_private_metadata.json",
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
            "dblp"
        ),

        "graph": (
            graph_path
        ),

        "cohorts": list(
            cohorts
        ),

        "representations": list(
            representations
        ),

        "primary_representation": (
            "eigen_d8_block"
        ),

        "secondary_representation": (
            "eigen_d12_global"
        ),

        "negative_control": (
            "hks_d12_global"
        ),

        "score_orientation": (
            "cosine similarity"
        ),

        "orientation_selection": (
            "Frozen from clean leakage-safe DBLP experiments "
            "before examining private results."
        ),

        "epsilons_total": [
            float(
                value
            )
            for value
            in epsilons
        ],

        "split_seeds": [
            int(
                value
            )
            for value
            in split_seeds
        ],

        "mechanism_seeds": [
            int(
                value
            )
            for value
            in mechanism_seeds
        ],

        "test_positives_per_split": int(
            test_positives
        ),

        "clip_C": float(
            C
        ),

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

        "privacy_calibration": (
            "Delta2=sqrt(2)C for verified nonnegative "
            "L2-clipped reports; Delta2=2C for generic reports."
        ),

        "privacy_budget": (
            "epsilon_signature=rho_signature*epsilon_total. "
            "The remaining total budget is reserved for the "
            "typed-degree report of the complete synthesis method."
        ),

        "one_report_per_author": True,

        "holdout_policy": (
            "Positive test relations are strict unique-paper "
            "exactly-two-author pairs excluded from a randomized "
            "spanning forest. One authorship incidence is removed "
            "before descriptor computation."
        ),

        "negative_policy": (
            "True non-coauthor pairs from the complete original "
            "coauthor relation."
        ),

        "statistical_unit": (
            "Graph split. Mechanism seeds are averaged within "
            "split before across-split inference."
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
        "DBLP PRIVATE LINK PREDICTION COMPLETE"
    )

    print(
        "=" * 80
    )

    print()

    print(
        "Final aggregate across graph splits:"
    )

    summary_columns = [
        "cohort",
        "representation",
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
        "PRIVACY / EVALUATION CHECKS:"
    )

    print(
        "  1. Test pairs are selected before descriptor "
        "generation."
    )

    print(
        "  2. Held-out authorship evidence is removed before "
        "descriptor generation."
    )

    print(
        "  3. Every held-out pair is verified to share zero "
        "papers after intervention."
    )

    print(
        "  4. The score orientation is frozen to similarity."
    )

    print(
        "  5. Representation identities are frozen before "
        "private evaluation."
    )

    print(
        "  6. Each author releases only one report per "
        "epsilon/mechanism seed."
    )

    print(
        "  7. Repeated pair appearances reuse the same private "
        "author report."
    )

    print(
        "  8. LDP calibration uses an analytical complete-report "
        "sensitivity bound."
    )

    print(
        "  9. Empirical edge sensitivity is never used for "
        "calibration."
    )

    print(
        " 10. Mechanism draws are averaged within graph split "
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
            "Leakage-safe DBLP private coauthor prediction "
            "with pure-LDP spectral client reports."
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
        "--cohorts",
        nargs="+",
        choices=[
            "provided_id_only",
            "all_authors",
        ],
        default=list(
            DEFAULT_COHORTS
        ),
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
        default=(
            DEFAULT_TEST_POSITIVES
        ),
    )

    parser.add_argument(
        "--C",
        type=float,
        default=(
            DEFAULT_C
        ),
    )

    parser.add_argument(
        "--rho-signature",
        type=float,
        default=(
            DEFAULT_RHO_SIGNATURE
        ),
    )

    parser.add_argument(
        "--sensitivity-mode",
        choices=[
            "nonnegative_report",
            "generic_report",
        ],
        default=(
            DEFAULT_SENSITIVITY_MODE
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

    run_experiment(
        output_folder=(
            args.output_folder
        ),

        cohorts=(
            args.cohorts
        ),

        representations=(
            args.representations
        ),

        epsilons=(
            args.epsilons
        ),

        split_seeds=(
            args.split_seeds
        ),

        mechanism_seeds=(
            args.mechanism_seeds
        ),

        test_positives=(
            args.test_positives
        ),

        C=(
            args.C
        ),

        rho_signature=(
            args.rho_signature
        ),

        sensitivity_mode=(
            args.sensitivity_mode
        ),

        d_max=(
            args.Dmax
        ),

        degree_weight=(
            args.degree_weight
        ),
    )
