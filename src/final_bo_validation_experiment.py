import argparse
import json
import math
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from private_bo_validation import validate_private_bo

SCHEMA_VERSION = "final-bo-validation-matrix-1.0"
DEFAULT_RECORDS = os.path.join(
    "output_journal_final",
    "journal_final_synthesis",
    "final_synthesis_run_records.csv",
)
DEFAULT_OUTPUT_FOLDER = "output_journal_final"
DEFAULT_DATASETS = ("enron",)
DEFAULT_EPSILONS = (0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
DEFAULT_PRIVACY_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_ORACLE_SEED = 0
DEFAULT_TOP_K = (3, 5, 10)
REPLAY_TOLERANCE = 1e-9


def utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def safe_json_dump(value, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=4, ensure_ascii=False, default=str)


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan
    return value if np.isfinite(value) else np.nan


def nested_get(mapping, keys, default=np.nan):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def epsilon_match(series, epsilon):
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return np.isclose(values, float(epsilon), rtol=0.0, atol=1e-12)


def ci95(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = int(len(values))
    if n == 0:
        return {"n": 0, "mean": np.nan, "std": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    mean = float(np.mean(values))
    if n == 1:
        return {"n": 1, "mean": mean, "std": 0.0, "ci95_low": mean, "ci95_high": mean}
    std = float(np.std(values, ddof=1))
    critical = float(student_t.ppf(0.975, df=n - 1))
    half_width = float(critical * std / math.sqrt(n))
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "ci95_low": mean - half_width,
        "ci95_high": mean + half_width,
    }


def aggregate_numeric_columns(frame, group_columns, metric_columns):
    rows = []
    for group_key, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(group_key, tuple):
            group_key = (group_key,)
        row = {column: value for column, value in zip(group_columns, group_key)}
        if "privacy_seed" in group.columns:
            row["seed_count"] = int(group["privacy_seed"].nunique())
        for metric in metric_columns:
            if metric not in group.columns:
                continue
            summary = ci95(pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float))
            for key, value in summary.items():
                row[f"{metric}__{key}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def load_final_synthesis_records(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing final synthesis run records:\n  {path}")
    frame = pd.read_csv(path)
    required = {"dataset", "epsilon_total", "seed", "private_run_folder"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Final synthesis records are missing columns: {sorted(missing)}")
    if "status" in frame.columns:
        frame = frame[frame["status"].astype(str).str.lower() == "complete"].copy()
    if frame.empty:
        raise RuntimeError("No completed final synthesis runs were found.")
    return frame


def select_final_run(records, dataset, epsilon, privacy_seed):
    mask = records["dataset"].astype(str) == str(dataset)
    mask &= epsilon_match(records["epsilon_total"], epsilon)
    mask &= pd.to_numeric(records["seed"], errors="coerce") == int(privacy_seed)
    subset = records[mask].copy()
    if len(subset) != 1:
        raise RuntimeError(
            "Expected exactly one final synthesis record for:\n"
            f"  dataset={dataset}\n  epsilon={epsilon}\n  privacy_seed={privacy_seed}\n"
            f"Found: {len(subset)}"
        )
    return subset.iloc[0]


def top_k_overlap_from_summary(summary, requested_k):
    top_k = summary.get("top_k_overlap", {})
    key = f"top_{int(requested_k)}"
    if key in top_k:
        return safe_float(top_k[key].get("overlap_fraction", np.nan))
    return np.nan


def extract_run_record(dataset, epsilon, privacy_seed, private_run_folder, validation_result, oracle_seed, top_k_values):
    summary = validation_result["summary"]
    selection = summary.get("selection", {})
    replay = summary.get("private_objective_replay_check", {})
    rank = summary.get("rank_displacement", {})

    row = {
        "dataset": str(dataset),
        "epsilon_total": float(epsilon),
        "privacy_seed": int(privacy_seed),
        "oracle_seed": int(oracle_seed),
        "private_run_folder": str(private_run_folder),
        "evaluated_configurations": int(summary.get("evaluated_configurations", 0)),
        "failed_configurations": int(summary.get("failed_configurations", 0)),
        "spearman": safe_float(nested_get(summary, ("private_vs_oracle_objective", "spearman", "correlation"))),
        "kendall_tau": safe_float(nested_get(summary, ("private_vs_oracle_objective", "kendall_tau", "correlation"))),
        "pearson": safe_float(nested_get(summary, ("private_vs_oracle_objective", "pearson", "correlation"))),
        "private_selected_iteration": safe_float(selection.get("private_selected_iteration", np.nan)),
        "best_oracle_within_trace_iteration": safe_float(
            selection.get("best_oracle_within_trace_iteration", selection.get("oracle_selected_iteration", np.nan))
        ),
        "same_selected_configuration": float(bool(selection.get("same_selected_configuration", False))),
        "oracle_objective_at_private_selection": safe_float(selection.get("oracle_objective_at_private_selection", np.nan)),
        "best_oracle_objective_within_trace": safe_float(
            selection.get("best_oracle_objective_within_trace", selection.get("best_oracle_objective", np.nan))
        ),
        "within_trace_oracle_regret": safe_float(
            selection.get("within_trace_oracle_regret", selection.get("oracle_selection_regret", np.nan))
        ),
        "relative_within_trace_oracle_regret": safe_float(
            selection.get("relative_within_trace_oracle_regret", selection.get("relative_oracle_selection_regret", np.nan))
        ),
        "rank_displacement_mean": safe_float(rank.get("mean", np.nan)),
        "rank_displacement_median": safe_float(rank.get("median", np.nan)),
        "rank_displacement_max": safe_float(rank.get("max", np.nan)),
        "private_replay_mean_abs_difference": safe_float(replay.get("mean_absolute_difference", np.nan)),
        "private_replay_max_abs_difference": safe_float(replay.get("maximum_absolute_difference", np.nan)),
        "validation_runtime_seconds": safe_float(summary.get("runtime_seconds", np.nan)),
    }
    for k in top_k_values:
        row[f"top_{int(k)}_overlap"] = top_k_overlap_from_summary(summary, k)
    return row


def run_matrix(
    records_path=DEFAULT_RECORDS,
    output_folder=DEFAULT_OUTPUT_FOLDER,
    datasets=DEFAULT_DATASETS,
    epsilons=DEFAULT_EPSILONS,
    privacy_seeds=DEFAULT_PRIVACY_SEEDS,
    oracle_seed=DEFAULT_ORACLE_SEED,
    top_k_values=DEFAULT_TOP_K,
    max_evaluations=None,
    force=False,
):
    start = time.perf_counter()
    records = load_final_synthesis_records(records_path)
    result_folder = os.path.join(output_folder, "journal_final_bo_validation")
    os.makedirs(result_folder, exist_ok=True)

    run_records_path = os.path.join(result_folder, "bo_validation_run_records.csv")
    aggregate_path = os.path.join(result_folder, "bo_validation_aggregate.csv")
    convergence_records_path = os.path.join(result_folder, "bo_validation_convergence_records.csv")
    convergence_aggregate_path = os.path.join(result_folder, "bo_validation_convergence_aggregate.csv")
    metadata_path = os.path.join(result_folder, "bo_validation_metadata.json")

    planned = len(datasets) * len(epsilons) * len(privacy_seeds)

    print("\n" + "=" * 80)
    print("FINAL JOURNAL PRIVATE-vs-ORACLE BO VALIDATION MATRIX")
    print("=" * 80)
    print(f"Datasets: {list(datasets)}")
    print(f"Epsilons: {[float(x) for x in epsilons]}")
    print(f"Privacy seeds: {[int(x) for x in privacy_seeds]}")
    print(f"Fixed oracle seed: {int(oracle_seed)}")
    print(f"Top-K: {[int(x) for x in top_k_values]}")
    print("Evaluations per run: " + ("complete frozen trace" if max_evaluations is None else str(int(max_evaluations))))
    print(f"Planned runs: {planned}")
    print("=" * 80)

    run_rows = []
    convergence_rows = []
    failures = []

    for dataset in datasets:
        for epsilon in epsilons:
            for privacy_seed in privacy_seeds:
                print("\n" + "#" * 80)
                print("BO VALIDATION RUN")
                print("#" * 80)
                print(f"dataset={dataset}")
                print(f"epsilon_total={float(epsilon):g}")
                print(f"privacy_seed={int(privacy_seed)}")
                print(f"oracle_seed={int(oracle_seed)}")

                try:
                    source_row = select_final_run(records, dataset, epsilon, privacy_seed)
                    private_run_folder = str(source_row["private_run_folder"])

                    if not os.path.isdir(private_run_folder):
                        raise FileNotFoundError(f"Private run folder does not exist:\n  {private_run_folder}")

                    trace_path = os.path.join(private_run_folder, "synthesis", "private_bo_trace.csv")
                    if not os.path.exists(trace_path):
                        raise FileNotFoundError(f"Frozen BO trace is missing:\n  {trace_path}")

                    validation = validate_private_bo(
                        dataset_name=dataset,
                        private_run_folder=private_run_folder,
                        output_folder=output_folder,
                        seed=int(oracle_seed),
                        top_k_values=tuple(int(value) for value in top_k_values),
                        max_evaluations=max_evaluations,
                        save_graphs=False,
                        force_rebuild=bool(force),
                    )

                    run_row = extract_run_record(
                        dataset,
                        epsilon,
                        privacy_seed,
                        private_run_folder,
                        validation,
                        oracle_seed,
                        top_k_values,
                    )

                    replay_max = safe_float(run_row["private_replay_max_abs_difference"])
                    if not np.isfinite(replay_max) or replay_max > REPLAY_TOLERANCE:
                        raise RuntimeError(
                            "Private objective replay validation failed.\n"
                            f"  replay_max={replay_max}\n  tolerance={REPLAY_TOLERANCE}"
                        )

                    if int(run_row["failed_configurations"]) != 0:
                        raise RuntimeError("One or more BO configurations failed oracle evaluation.")

                    run_rows.append(run_row)

                    convergence = validation["convergence"].copy()
                    convergence.insert(0, "dataset", str(dataset))
                    convergence.insert(1, "epsilon_total", float(epsilon))
                    convergence.insert(2, "privacy_seed", int(privacy_seed))
                    convergence.insert(3, "oracle_seed", int(oracle_seed))
                    convergence_rows.append(convergence)

                    print("[matrix] completed")
                    print(f"  Spearman: {run_row['spearman']:.6f}")
                    print(f"  within-trace regret: {run_row['within_trace_oracle_regret']:.6f}")
                    print(f"  replay max difference: {replay_max:.3e}")

                except Exception as exc:
                    failures.append({
                        "dataset": str(dataset),
                        "epsilon_total": float(epsilon),
                        "privacy_seed": int(privacy_seed),
                        "error": repr(exc),
                    })
                    print("[matrix] FAILED")
                    print(repr(exc))

    if not run_rows:
        raise RuntimeError("No BO validation runs completed successfully.")

    run_df = pd.DataFrame(run_rows).sort_values(
        ["dataset", "epsilon_total", "privacy_seed"]
    ).reset_index(drop=True)
    run_df.to_csv(run_records_path, index=False)

    metric_columns = [
        "spearman",
        "kendall_tau",
        "pearson",
        "same_selected_configuration",
        "oracle_objective_at_private_selection",
        "best_oracle_objective_within_trace",
        "within_trace_oracle_regret",
        "relative_within_trace_oracle_regret",
        "rank_displacement_mean",
        "rank_displacement_median",
        "rank_displacement_max",
        "private_replay_mean_abs_difference",
        "private_replay_max_abs_difference",
        "validation_runtime_seconds",
    ] + [f"top_{int(k)}_overlap" for k in top_k_values]

    aggregate_df = aggregate_numeric_columns(
        run_df,
        ["dataset", "epsilon_total"],
        metric_columns,
    )
    aggregate_df.to_csv(aggregate_path, index=False)

    if convergence_rows:
        convergence_df = pd.concat(convergence_rows, ignore_index=True).sort_values(
            ["dataset", "epsilon_total", "privacy_seed", "iteration"]
        ).reset_index(drop=True)
        convergence_df.to_csv(convergence_records_path, index=False)

        convergence_metrics = [
            "best_private_objective_so_far",
            "oracle_of_private_best_so_far",
            "best_oracle_objective_so_far",
            "oracle_regret_of_private_selection",
            "relative_oracle_regret_of_private_selection",
        ]
        convergence_aggregate_df = aggregate_numeric_columns(
            convergence_df,
            ["dataset", "epsilon_total", "iteration"],
            convergence_metrics,
        )
        convergence_aggregate_df.to_csv(convergence_aggregate_path, index=False)

    total_runtime = float(time.perf_counter() - start)
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": utc_now_iso(),
        "records_path": records_path,
        "output_folder": output_folder,
        "datasets": [str(value) for value in datasets],
        "epsilons": [float(value) for value in epsilons],
        "privacy_seeds": [int(value) for value in privacy_seeds],
        "fixed_oracle_seed": int(oracle_seed),
        "top_k_values": [int(value) for value in top_k_values],
        "max_evaluations": None if max_evaluations is None else int(max_evaluations),
        "replay_tolerance": float(REPLAY_TOLERANCE),
        "planned_runs": int(planned),
        "successful_runs": int(len(run_df)),
        "failed_runs": int(len(failures)),
        "failures": failures,
        "interpretation": {
            "oracle_role": "post-hoc evaluation only; oracle values never alter the frozen private BO trace",
            "oracle_best_definition": "best oracle fidelity among configurations already proposed in the frozen private BO trace",
            "not_an_oracle_optimizer": True,
            "same_oracle_seed_across_all_runs": True,
        },
        "outputs": {
            "run_records": run_records_path,
            "aggregate": aggregate_path,
            "convergence_records": convergence_records_path if convergence_rows else None,
            "convergence_aggregate": convergence_aggregate_path if convergence_rows else None,
        },
        "total_runtime_seconds": total_runtime,
    }
    safe_json_dump(metadata, metadata_path)

    print("\n" + "=" * 80)
    print("FINAL BO VALIDATION MATRIX COMPLETE")
    print("=" * 80)
    print(f"Successful runs: {len(run_df)}")
    print(f"Failed runs: {len(failures)}")
    print(f"Records: {run_records_path}")
    print(f"Aggregate: {aggregate_path}")
    if convergence_rows:
        print(f"Convergence records: {convergence_records_path}")
        print(f"Convergence aggregate: {convergence_aggregate_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Total runtime: {total_runtime:.3f} s")
    print("=" * 80)

    return {
        "records": run_df,
        "aggregate": aggregate_df,
        "failures": failures,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch post-hoc private-vs-oracle validation over the frozen final synthesis matrix."
    )
    parser.add_argument("--records", type=str, default=DEFAULT_RECORDS)
    parser.add_argument("--output-folder", type=str, default=DEFAULT_OUTPUT_FOLDER)
    parser.add_argument("--datasets", nargs="+", type=str, default=list(DEFAULT_DATASETS))
    parser.add_argument("--epsilons", nargs="+", type=float, default=list(DEFAULT_EPSILONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_PRIVACY_SEEDS))
    parser.add_argument(
        "--oracle-seed",
        type=int,
        default=DEFAULT_ORACLE_SEED,
        help="Fixed post-hoc oracle seed used for every private run.",
    )
    parser.add_argument("--top-k", nargs="+", type=int, default=list(DEFAULT_TOP_K))
    parser.add_argument(
        "--max-evaluations",
        type=int,
        default=None,
        help="Debug only. Omit for the final paper so the complete frozen BO trace is evaluated.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuilding per-run validator outputs instead of reusing configuration-matched caches.",
    )
    args = parser.parse_args()

    run_matrix(
        records_path=args.records,
        output_folder=args.output_folder,
        datasets=tuple(args.datasets),
        epsilons=tuple(float(value) for value in args.epsilons),
        privacy_seeds=tuple(int(value) for value in args.seeds),
        oracle_seed=int(args.oracle_seed),
        top_k_values=tuple(int(value) for value in args.top_k),
        max_evaluations=args.max_evaluations,
        force=bool(args.force),
    )


if __name__ == "__main__":
    main()
