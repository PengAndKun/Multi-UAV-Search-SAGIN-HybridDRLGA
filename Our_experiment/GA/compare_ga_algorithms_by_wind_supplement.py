import argparse
import os
import sys
import time

import numpy as np

# Headless defaults for environments that run without display/audio.
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Our_experiment.GA import ga_deployment_seed_search_2_no_offloading as algo_no_offload
from Our_experiment.GA.ga_vis_common import build_agents_and_env
from Our_experiment.GA.ga_vis_common import resolve_path_template
from Our_experiment.GA.ga_vis_common import save_json
from Our_experiment.GA.ga_vis_common import utc_now_iso
from Our_experiment.GA.ga_vis_common import validate_traj_range


WIND_CLASS_MAP = {
    11: "Low Wind",
    23: "Moderate Wind",
    4800: "Strong Wind",
}

ALGORITHM_ID = "no_ga_drl_only_no_offloading"
ALGORITHM_LABEL = "No-GA + No-Offloading"
ALGORITHM_FILE = "Our_experiment/GA/compare_ga_algorithms_by_wind_supplement.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the No-GA + No-Offloading baseline across multiple wind classes, "
            "save a supplement JSON, and print a terminal table."
        )
    )
    parser.add_argument("--num-uav", type=int, default=4, help="Number of UAVs.")
    parser.add_argument("--infra-seed", type=int, default=999999, help="Infrastructure seed.")
    parser.add_argument("--terrain-seed", type=int, default=10, help="Terrain seed.")
    parser.add_argument(
        "--wind-seeds",
        type=str,
        default="11,23,4800",
        help="Comma-separated wind seeds. Default represents Low/Moderate/Strong wind.",
    )
    parser.add_argument(
        "--eval-traj-seed-start",
        type=int,
        default=0,
        help="Evaluation trajectory seed start (inclusive).",
    )
    parser.add_argument(
        "--eval-traj-seed-end",
        type=int,
        default=200,
        help="Evaluation trajectory seed end (inclusive).",
    )
    parser.add_argument(
        "--eval-repetitions",
        type=int,
        default=1,
        help="Evaluation rollout repetitions per trajectory seed.",
    )
    parser.add_argument(
        "--rollout-seed-base",
        type=int,
        default=2026,
        help="Base seed used to compose rollout randomness.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=(
            "Our_experiment/GA/data/"
            "ga_algorithm_comparison_supplement_i{infra_seed}_g{terrain_seed}_winds{wind_tag}.json"
        ),
        help="Output supplement JSON path.",
    )
    parser.add_argument(
        "--lifetime-unit",
        type=str,
        default="min",
        choices=["min", "s", "steps"],
        help="Lifetime unit shown in the terminal table.",
    )
    parser.add_argument(
        "--table-format",
        type=str,
        default="markdown",
        choices=["markdown", "plain"],
        help="Terminal table format.",
    )
    return parser.parse_args()


def parse_wind_seeds(text):
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) == 0:
        raise ValueError("wind-seeds is empty.")
    return [int(p) for p in parts]


def wind_class_name(wind_seed):
    return WIND_CLASS_MAP.get(int(wind_seed), f"Wind Seed {int(wind_seed)}")


def build_config(args):
    validate_traj_range(args.eval_traj_seed_start, args.eval_traj_seed_end)
    if args.num_uav < 1:
        raise ValueError("num-uav must be >= 1")
    if args.eval_repetitions < 1:
        raise ValueError("eval-repetitions must be >= 1")
    return {
        "num_uav": int(args.num_uav),
        "infra_seed": int(args.infra_seed),
        "terrain_seed": int(args.terrain_seed),
        "eval_traj_seed_start": int(args.eval_traj_seed_start),
        "eval_traj_seed_end": int(args.eval_traj_seed_end),
        "eval_repetitions": int(args.eval_repetitions),
        "rollout_seed_base": int(args.rollout_seed_base),
    }


def make_eval_traj_seeds(cfg):
    return list(range(cfg["eval_traj_seed_start"], cfg["eval_traj_seed_end"] + 1))


def to_coverage_metrics(metrics):
    metrics = dict(metrics)
    mean_unc = float(metrics["mean_average_uncertainty"])
    std_unc = float(metrics["std_average_uncertainty"])
    metrics["mean_coverage"] = float(1.0 - mean_unc)
    metrics["std_coverage"] = float(std_unc)
    metrics["mean_coverage_percent"] = float((1.0 - mean_unc) * 100.0)
    metrics["std_coverage_percent"] = float(std_unc * 100.0)
    return metrics


def format_pm(mean_value, std_value, digits=2):
    return f"{float(mean_value):.{digits}f} ± {float(std_value):.{digits}f}"


def format_lifetime(metrics, unit):
    if unit == "steps":
        return format_pm(metrics["mean_uav_lifetime_steps"], metrics["std_uav_lifetime_steps"], digits=2)
    if unit == "s":
        return format_pm(metrics["mean_uav_lifetime_seconds"], metrics["std_uav_lifetime_seconds"], digits=2)
    return format_pm(
        float(metrics["mean_uav_lifetime_seconds"]) / 60.0,
        float(metrics["std_uav_lifetime_seconds"]) / 60.0,
        digits=2,
    )


def format_coverage(metrics):
    return format_pm(metrics["mean_coverage_percent"], metrics["std_coverage_percent"], digits=2)


def lifetime_label(unit):
    if unit == "steps":
        return "Lifetime (steps)"
    if unit == "s":
        return "Lifetime (s)"
    return "Lifetime (min)"


def markdown_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def plain_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt_row(headers), sep]
    for row in rows:
        lines.append(fmt_row(row))
    return "\n".join(lines)


def build_headers(lifetime_unit, wind_seeds):
    headers = ["Variant", "Offloading", "GA"]
    for wind_seed in wind_seeds:
        headers.append(f"{wind_class_name(wind_seed)} {lifetime_label(lifetime_unit)}")
        headers.append(f"{wind_class_name(wind_seed)} Coverage (%)")
    return headers


def build_rows(metrics_by_wind, lifetime_unit, wind_seeds):
    row = [ALGORITHM_LABEL, "No", "No"]
    for wind_seed in wind_seeds:
        metrics = metrics_by_wind[str(wind_seed)]["metrics"]
        row.append(format_lifetime(metrics, lifetime_unit))
        row.append(format_coverage(metrics))
    return [row]


def print_table(metrics_by_wind, lifetime_unit, table_format, wind_seeds):
    headers = build_headers(lifetime_unit, wind_seeds)
    rows = build_rows(metrics_by_wind, lifetime_unit, wind_seeds)
    if table_format == "plain":
        print(plain_table(headers, rows))
    else:
        print(markdown_table(headers, rows))


def make_default_positions(env):
    positions = [tuple(env.gird_position[i]) for i in range(env.N)]
    return positions, list(positions)


def compose_rollout_seed(base_seed, traj_seed, repetition):
    value = int(base_seed)
    value = value * 1000003 + int(traj_seed) * 7919 + int(repetition) * 97
    return int(value % 2147483647)


def evaluate_no_ga_no_offloading(
    env,
    agent,
    wind_seed,
    terrain_seed,
    infra_seed,
    eval_traj_seeds,
    eval_repetitions,
    rollout_seed_base,
):
    starts, ends = make_default_positions(env)
    uncertainty_values = []
    lifetime_steps_values = []
    lifetime_seconds_values = []

    for traj_seed in eval_traj_seeds:
        for rep in range(eval_repetitions):
            rollout_seed = compose_rollout_seed(rollout_seed_base, traj_seed, rep)
            rollout_result = algo_no_offload.trajectory_execution_no_offloading(
                env=env,
                agent=agent,
                position_starts=starts,
                position_ends=ends,
                wind_seed=int(wind_seed),
                terrain_seed=int(terrain_seed),
                infra_seed=int(infra_seed),
                traj_seed=int(traj_seed),
                rollout_seed=int(rollout_seed),
            )
            uncertainty_values.append(float(rollout_result["avg_uncertainty"]))
            lifetime_steps_values.append(float(rollout_result["avg_uav_lifetime_steps"]))
            lifetime_seconds_values.append(float(rollout_result["avg_uav_lifetime_seconds"]))

    if len(uncertainty_values) == 0:
        raise RuntimeError("No evaluation rollouts were executed for the supplement baseline.")

    metrics = {
        "mean_average_uncertainty": float(np.mean(uncertainty_values)),
        "std_average_uncertainty": float(np.std(uncertainty_values)),
        "mean_uav_lifetime_steps": float(np.mean(lifetime_steps_values)),
        "std_uav_lifetime_steps": float(np.std(lifetime_steps_values)),
        "mean_uav_lifetime_seconds": float(np.mean(lifetime_seconds_values)),
        "std_uav_lifetime_seconds": float(np.std(lifetime_seconds_values)),
        "num_traj_seeds": int(len(eval_traj_seeds)),
        "eval_repetitions": int(eval_repetitions),
        "num_eval_rollouts": int(len(uncertainty_values)),
    }
    return to_coverage_metrics(metrics), starts, ends


def run_algorithm_no_ga_no_offloading(cfg, wind_seed):
    print("=" * 100)
    print(f"[{ALGORITHM_LABEL}] wind_seed={wind_seed} ({wind_class_name(wind_seed)}) evaluation started.")

    env, agent, _ = build_agents_and_env(num_uav=cfg["num_uav"])

    t0 = time.time()
    eval_metrics, starts, ends = evaluate_no_ga_no_offloading(
        env=env,
        agent=agent,
        wind_seed=int(wind_seed),
        terrain_seed=cfg["terrain_seed"],
        infra_seed=cfg["infra_seed"],
        eval_traj_seeds=make_eval_traj_seeds(cfg),
        eval_repetitions=cfg["eval_repetitions"],
        rollout_seed_base=cfg["rollout_seed_base"],
    )
    eval_elapsed = float(time.time() - t0)

    print(
        f"[{ALGORITHM_LABEL}] wind_seed={wind_seed} done. "
        f"Mean uncertainty={eval_metrics['mean_average_uncertainty']:.6f}, "
        f"Coverage={eval_metrics['mean_coverage_percent']:.2f}%, "
        f"Mean lifetime={eval_metrics['mean_uav_lifetime_steps']:.2f} steps."
    )

    return {
        "algorithm_id": ALGORITHM_ID,
        "algorithm_label": ALGORITHM_LABEL,
        "algorithm_file": ALGORITHM_FILE,
        "wind_seed": int(wind_seed),
        "wind_class": wind_class_name(wind_seed),
        "search_elapsed_sec": 0.0,
        "eval_elapsed_sec": eval_elapsed,
        "search": {
            "ga_objective_mean_average_uncertainty": None,
            "best_run_traj_seed": None,
            "best_step_count": None,
            "best_iteration_seed_set": [],
            "iteration_sampled_traj_seeds": [],
        },
        "best": {
            "best_solution": None,
            "start_positions": [[int(p[0]), int(p[1])] for p in starts],
            "end_positions": [[int(p[0]), int(p[1])] for p in ends],
            "trajectory_actions": None,
            "offload_actions": None,
        },
        "metrics": eval_metrics,
    }


def build_grouped_summary(runs):
    by_wind = {}
    by_algorithm = {}

    for run in runs:
        wind_key = str(run["wind_seed"])
        alg_key = run["algorithm_id"]

        if wind_key not in by_wind:
            by_wind[wind_key] = {
                "wind_seed": int(run["wind_seed"]),
                "wind_class": run["wind_class"],
                "algorithms": {},
            }
        by_wind[wind_key]["algorithms"][alg_key] = {
            "algorithm_label": run["algorithm_label"],
            "metrics": run["metrics"],
            "search_elapsed_sec": run["search_elapsed_sec"],
            "ga_objective_mean_average_uncertainty": run["search"]["ga_objective_mean_average_uncertainty"],
        }

        if alg_key not in by_algorithm:
            by_algorithm[alg_key] = {
                "algorithm_label": run["algorithm_label"],
                "algorithm_file": run["algorithm_file"],
                "winds": {},
            }
        by_algorithm[alg_key]["winds"][wind_key] = {
            "wind_seed": int(run["wind_seed"]),
            "wind_class": run["wind_class"],
            "metrics": run["metrics"],
            "search_elapsed_sec": run["search_elapsed_sec"],
            "ga_objective_mean_average_uncertainty": run["search"]["ga_objective_mean_average_uncertainty"],
        }

    return by_wind, by_algorithm


def main():
    args = parse_args()
    cfg = build_config(args)
    wind_seeds = parse_wind_seeds(args.wind_seeds)
    wind_tag = "_".join(str(w) for w in wind_seeds)

    runs = []
    total_start = time.time()
    for wind_seed in wind_seeds:
        runs.append(run_algorithm_no_ga_no_offloading(cfg, wind_seed))

    by_wind, by_algorithm = build_grouped_summary(runs)
    total_elapsed = float(time.time() - total_start)

    output_json = resolve_path_template(
        args.output_json,
        infra_seed=cfg["infra_seed"],
        terrain_seed=cfg["terrain_seed"],
        wind_tag=wind_tag,
    )

    result = {
        "created_at": utc_now_iso(),
        "summary_type": "supplement_no_ga_no_offloading_comparison_by_wind",
        "config": {
            "num_uav": cfg["num_uav"],
            "infra_seed": cfg["infra_seed"],
            "terrain_seed": cfg["terrain_seed"],
            "wind_seeds": [int(w) for w in wind_seeds],
            "wind_classes": {str(w): wind_class_name(w) for w in wind_seeds},
            "eval_traj_seed_start": cfg["eval_traj_seed_start"],
            "eval_traj_seed_end": cfg["eval_traj_seed_end"],
            "eval_repetitions": cfg["eval_repetitions"],
            "rollout_seed_base": cfg["rollout_seed_base"],
            "coverage_definition": "coverage = 1 - uncertainty",
            "num_total_runs": int(len(runs)),
            "total_elapsed_sec": total_elapsed,
        },
        "runs": runs,
        "comparison_by_wind": by_wind,
        "comparison_by_algorithm": by_algorithm,
    }
    save_json(output_json, result)

    print("-" * 100)
    print("GA supplement comparison by wind completed.")
    print(f"Output JSON: {output_json}")
    print("")
    print_table(by_wind, args.lifetime_unit, args.table_format, wind_seeds)


if __name__ == "__main__":
    main()
