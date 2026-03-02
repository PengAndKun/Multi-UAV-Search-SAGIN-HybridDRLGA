import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np

# Headless defaults for environments that run without display/audio
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Our_experiment.GA import ga_deployment_seed_search_2 as algo_ga_offload
from Our_experiment.GA import ga_deployment_seed_search_2_no_offloading as algo_ga_no_offload
from Our_experiment.GA.ga_vis_common import build_agents_and_env
from Our_experiment.GA.ga_vis_common import resolve_path_template
from Our_experiment.GA.ga_vis_common import save_json
from Our_experiment.GA.ga_vis_common import set_seed
from Our_experiment.GA.ga_vis_common import utc_now_iso
from Our_experiment.HCSAC.UAV_VIS_offloading_2 import visualize_trajectory as vis


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare three groups across different UAV counts: "
            "GA+Offloading, GA+No-Offloading, and No-GA+Offloading."
        )
    )
    parser.add_argument(
        "--uav-counts",
        type=str,
        default="1,2,3,4,5,6",
        help="Comma-separated UAV counts (e.g. 1,2,3,4,5,6).",
    )
    parser.add_argument("--infra-seed", type=int, default=999999, help="Infrastructure seed.")
    parser.add_argument("--terrain-seed", type=int, default=10, help="Terrain seed.")
    parser.add_argument("--wind-seed", type=int, default=4800, help="Wind seed.")

    # GA search config (used by two GA groups)
    parser.add_argument("--traj-seed-min", type=int, default=0, help="GA search trajectory seed pool min.")
    parser.add_argument("--traj-seed-max", type=int, default=200, help="GA search trajectory seed pool max.")
    parser.add_argument("--traj-seed-sample-size", type=int, default=10, help="GA search sampled trajectory seed count.")
    parser.add_argument("--sample-with-replacement", action="store_true", help="GA search seed sampling with replacement.")
    parser.add_argument("--iterations", type=int, default=20, help="GA generation count.")
    parser.add_argument("--population-size", type=int, default=12, help="GA population size.")
    parser.add_argument("--repetitions", type=int, default=1, help="GA rollout repetitions per sampled seed.")
    parser.add_argument("--mutation-rate", type=float, default=0.1, help="GA mutation rate.")
    parser.add_argument("--ga-seed", type=int, default=2026, help="GA random seed.")
    parser.add_argument("--verbose-seed-progress", action="store_true", help="Print per-seed progress from GA scripts.")

    # Unified evaluation config (used by all 3 groups)
    parser.add_argument(
        "--eval-traj-seed-pool-min",
        type=int,
        default=0,
        help="Evaluation trajectory seed pool min (inclusive).",
    )
    parser.add_argument(
        "--eval-traj-seed-pool-max",
        type=int,
        default=200,
        help="Evaluation trajectory seed pool max (inclusive).",
    )
    parser.add_argument(
        "--eval-traj-seed-sample-size",
        type=int,
        default=10,
        help="Evaluation sampled trajectory seed count.",
    )
    parser.add_argument(
        "--eval-traj-seed-sampler-seed",
        type=int,
        default=2026,
        help="RNG seed used to sample evaluation trajectory seeds.",
    )
    parser.add_argument(
        "--eval-sample-with-replacement",
        action="store_true",
        help="Evaluation seed sampling with replacement.",
    )
    parser.add_argument("--eval-repetitions", type=int, default=1, help="Evaluation repetitions per trajectory seed.")

    parser.add_argument(
        "--output-json",
        type=str,
        default=(
            "Our_experiment/GA/data/"
            "ga_three_group_uav_count_comparison_w{wind_seed}_g{terrain_seed}_i{infra_seed}.json"
        ),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--coverage-plot-path",
        type=str,
        default=(
            "Our_experiment/GA/data/"
            "ga_three_group_coverage_vs_uav_w{wind_seed}_g{terrain_seed}_i{infra_seed}.png"
        ),
        help="Output coverage plot path.",
    )
    parser.add_argument(
        "--uncertainty-plot-path",
        type=str,
        default=(
            "Our_experiment/GA/data/"
            "ga_three_group_uncertainty_vs_uav_w{wind_seed}_g{terrain_seed}_i{infra_seed}.png"
        ),
        help="Output uncertainty plot path.",
    )
    parser.add_argument(
        "--table-format",
        type=str,
        default="plain",
        choices=["plain", "markdown"],
        help="Terminal table format.",
    )
    return parser.parse_args()


def parse_uav_counts(text):
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) == 0:
        raise ValueError("uav-counts is empty.")
    values = [int(p) for p in parts]
    for v in values:
        if v < 1:
            raise ValueError("All uav-counts must be >= 1.")
    return values


def normalize_output_path(path):
    if os.path.isabs(path):
        return os.path.normpath(path)
    if path.startswith("Our_experiment/"):
        return os.path.normpath(os.path.join(PROJECT_ROOT, path))
    return os.path.normpath(os.path.abspath(path))


def validate_args(args):
    if args.traj_seed_max < args.traj_seed_min:
        raise ValueError("traj-seed-max must be >= traj-seed-min")
    if args.eval_traj_seed_pool_max < args.eval_traj_seed_pool_min:
        raise ValueError("eval-traj-seed-pool-max must be >= eval-traj-seed-pool-min")
    if args.population_size < 2:
        raise ValueError("population-size must be >= 2")
    if args.iterations < 1:
        raise ValueError("iterations must be >= 1")
    if args.repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if args.traj_seed_sample_size < 1:
        raise ValueError("traj-seed-sample-size must be >= 1")
    if args.eval_traj_seed_sample_size < 1:
        raise ValueError("eval-traj-seed-sample-size must be >= 1")
    if args.eval_repetitions < 1:
        raise ValueError("eval-repetitions must be >= 1")
    if not (0.0 <= args.mutation_rate <= 1.0):
        raise ValueError("mutation-rate must be in [0, 1]")


def sample_eval_traj_seeds(args):
    pool = np.arange(int(args.eval_traj_seed_pool_min), int(args.eval_traj_seed_pool_max) + 1, dtype=np.int64)
    if len(pool) == 0:
        raise RuntimeError("No evaluation trajectory seed pool.")
    replace = bool(args.eval_sample_with_replacement) or int(args.eval_traj_seed_sample_size) > len(pool)
    rng = np.random.default_rng(int(args.eval_traj_seed_sampler_seed))
    sampled = rng.choice(pool, size=int(args.eval_traj_seed_sample_size), replace=replace)
    return [int(s) for s in sampled.tolist()], bool(replace)


def rollout_full_rl_with_lifetime(
    env,
    agent,
    offload_agent,
    position_starts,
    position_ends,
    wind_seed,
    terrain_seed,
    infra_seed,
    traj_seed,
    rollout_seed,
):
    set_seed(rollout_seed)
    state = env.reset(
        seed=traj_seed,
        positions=position_starts,
        destinations=position_ends,
        wind_seed=wind_seed,
        terrain_seed=terrain_seed,
        infra_seed=infra_seed,
    )

    done = False
    step_count = 0
    uav_lifetime_steps = np.full(env.N, np.nan, dtype=np.float64)
    avg_uncertainty = float(np.mean(env.uncertainty_matrix))

    while not done:
        step_count += 1
        actions = [agent.take_action(state[n]) for n in range(env.N)]
        next_state, _, done_move = env.step(actions)

        offload_data = env.get_obs_2()
        offload_actions = offload_agent.take_action(offload_data)
        _, _, done_offload = env.step_offload(offload_actions)

        done = bool(done_move or done_offload)
        state = next_state
        avg_uncertainty = float(np.mean(env.uncertainty_matrix))

        for i, uav in enumerate(env.uavs):
            if np.isnan(uav_lifetime_steps[i]) and bool(uav["done"]):
                uav_lifetime_steps[i] = float(step_count)

    if env.N > 0:
        uav_lifetime_steps = np.where(np.isnan(uav_lifetime_steps), float(step_count), uav_lifetime_steps)
        avg_uav_lifetime_steps = float(np.mean(uav_lifetime_steps))
    else:
        uav_lifetime_steps = np.array([], dtype=np.float64)
        avg_uav_lifetime_steps = float(step_count)
    avg_uav_lifetime_seconds = float(avg_uav_lifetime_steps * float(env.T))

    return {
        "avg_uncertainty": float(avg_uncertainty),
        "avg_uav_lifetime_steps": float(avg_uav_lifetime_steps),
        "avg_uav_lifetime_seconds": float(avg_uav_lifetime_seconds),
        "step_count": int(step_count),
    }


def to_coverage_metrics(metrics):
    mean_unc = float(metrics["mean_average_uncertainty"])
    std_unc = float(metrics["std_average_uncertainty"])
    metrics = dict(metrics)
    metrics["mean_coverage_percent"] = float((1.0 - mean_unc) * 100.0)
    metrics["std_coverage_percent"] = float(std_unc * 100.0)
    return metrics


def evaluate_ga_offloading_solution(
    env,
    agent,
    offload_agent,
    num_uav,
    best_solution,
    wind_seed,
    terrain_seed,
    infra_seed,
    eval_traj_seeds,
    eval_repetitions,
    ga_seed,
):
    starts, ends = algo_ga_offload.split_chromosome(best_solution, num_uav)
    uncertainty_values = []
    lifetime_steps_values = []
    lifetime_seconds_values = []

    for seed_idx, traj_seed in enumerate(eval_traj_seeds):
        for rep in range(eval_repetitions):
            rollout_seed = algo_ga_offload.compose_rollout_seed(
                ga_seed=ga_seed + 7919,
                iteration=99991,
                population_index=0,
                seed_index=seed_idx,
                repetition=rep,
            )
            stats = rollout_full_rl_with_lifetime(
                env=env,
                agent=agent,
                offload_agent=offload_agent,
                position_starts=starts,
                position_ends=ends,
                wind_seed=wind_seed,
                terrain_seed=terrain_seed,
                infra_seed=infra_seed,
                traj_seed=int(traj_seed),
                rollout_seed=rollout_seed,
            )
            uncertainty_values.append(float(stats["avg_uncertainty"]))
            lifetime_steps_values.append(float(stats["avg_uav_lifetime_steps"]))
            lifetime_seconds_values.append(float(stats["avg_uav_lifetime_seconds"]))

    if len(uncertainty_values) == 0:
        raise RuntimeError("No evaluation rollouts for GA+Offloading.")

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
    return to_coverage_metrics(metrics)


def evaluate_no_ga_offloading(
    env,
    agent,
    offload_agent,
    wind_seed,
    terrain_seed,
    infra_seed,
    eval_traj_seeds,
    eval_repetitions,
):
    uncertainty_values = []
    lifetime_steps_values = []
    lifetime_seconds_values = []

    for traj_seed in eval_traj_seeds:
        for _ in range(eval_repetitions):
            set_seed(int(traj_seed))
            stats = vis(
                agent,
                offload_agent,
                env,
                seed=int(traj_seed),
                return_stats=True,
                wind_seed=int(wind_seed),
                terrain_seed=int(terrain_seed),
                traj_seed=int(traj_seed),
                infra_seed=int(infra_seed),
            )
            uncertainty_values.append(float(stats["avg_uncertainty"]))
            lifetime_steps_values.append(float(stats.get("avg_uav_lifetime_steps", np.nan)))
            lifetime_seconds_values.append(float(stats.get("avg_uav_lifetime_seconds", np.nan)))

    if len(uncertainty_values) == 0:
        raise RuntimeError("No evaluation rollouts for No-GA+Offloading.")

    metrics = {
        "mean_average_uncertainty": float(np.mean(uncertainty_values)),
        "std_average_uncertainty": float(np.std(uncertainty_values)),
        "mean_uav_lifetime_steps": float(np.nanmean(lifetime_steps_values)),
        "std_uav_lifetime_steps": float(np.nanstd(lifetime_steps_values)),
        "mean_uav_lifetime_seconds": float(np.nanmean(lifetime_seconds_values)),
        "std_uav_lifetime_seconds": float(np.nanstd(lifetime_seconds_values)),
        "num_traj_seeds": int(len(eval_traj_seeds)),
        "eval_repetitions": int(eval_repetitions),
        "num_eval_rollouts": int(len(uncertainty_values)),
    }
    return to_coverage_metrics(metrics)


def run_ga_offloading_group(args, num_uav, eval_traj_seeds):
    set_seed(args.ga_seed)
    seed_sampler_rng = np.random.default_rng(args.ga_seed)
    env, agent, offload_agent = build_agents_and_env(num_uav=num_uav)
    traj_seed_pool = np.arange(args.traj_seed_min, args.traj_seed_max + 1, dtype=np.int64)

    t0 = time.time()
    ga_result = algo_ga_offload.genetic_algorithm(
        iterations=args.iterations,
        population_size=args.population_size,
        repetitions=args.repetitions,
        mutation_rate=args.mutation_rate,
        env=env,
        agent=agent,
        offload_agent=offload_agent,
        num_uav=num_uav,
        env_lx=env.Lx,
        env_ly=env.Ly,
        wind_seed=int(args.wind_seed),
        terrain_seed=int(args.terrain_seed),
        infra_seed=int(args.infra_seed),
        traj_seed_pool=traj_seed_pool,
        traj_seed_sample_size=args.traj_seed_sample_size,
        sample_with_replacement=bool(args.sample_with_replacement),
        seed_sampler_rng=seed_sampler_rng,
        ga_seed=int(args.ga_seed),
        verbose_seed_progress=bool(args.verbose_seed_progress),
    )
    search_elapsed = float(time.time() - t0)

    best_solution = ga_result["best_solution"]
    metrics = evaluate_ga_offloading_solution(
        env=env,
        agent=agent,
        offload_agent=offload_agent,
        num_uav=num_uav,
        best_solution=best_solution,
        wind_seed=int(args.wind_seed),
        terrain_seed=int(args.terrain_seed),
        infra_seed=int(args.infra_seed),
        eval_traj_seeds=eval_traj_seeds,
        eval_repetitions=int(args.eval_repetitions),
        ga_seed=int(args.ga_seed),
    )
    starts, ends = algo_ga_offload.split_chromosome(best_solution, num_uav)

    return {
        "group_id": "ga_offloading",
        "group_label": "GA + Offloading",
        "search_elapsed_sec": search_elapsed,
        "search": {
            "ga_objective_mean_average_uncertainty": float(ga_result["best_fitness"]),
            "best_run_traj_seed": (
                None if ga_result["best_run_traj_seed"] is None else int(ga_result["best_run_traj_seed"])
            ),
            "best_step_count": int(ga_result["best_step_count"]),
            "best_iteration_seed_set": [int(s) for s in ga_result["best_iteration_seed_set"]],
        },
        "best": {
            "best_solution": [[int(p[0]), int(p[1])] for p in best_solution],
            "start_positions": [[int(p[0]), int(p[1])] for p in starts],
            "end_positions": [[int(p[0]), int(p[1])] for p in ends],
        },
        "metrics": metrics,
    }


def run_ga_no_offloading_group(args, num_uav, eval_traj_seeds):
    set_seed(args.ga_seed)
    seed_sampler_rng = np.random.default_rng(args.ga_seed)
    env, agent, _ = build_agents_and_env(num_uav=num_uav)
    traj_seed_pool = np.arange(args.traj_seed_min, args.traj_seed_max + 1, dtype=np.int64)

    t0 = time.time()
    ga_result = algo_ga_no_offload.genetic_algorithm(
        iterations=args.iterations,
        population_size=args.population_size,
        repetitions=args.repetitions,
        mutation_rate=args.mutation_rate,
        env=env,
        agent=agent,
        num_uav=num_uav,
        env_lx=env.Lx,
        env_ly=env.Ly,
        wind_seed=int(args.wind_seed),
        terrain_seed=int(args.terrain_seed),
        infra_seed=int(args.infra_seed),
        traj_seed_pool=traj_seed_pool,
        traj_seed_sample_size=args.traj_seed_sample_size,
        sample_with_replacement=bool(args.sample_with_replacement),
        seed_sampler_rng=seed_sampler_rng,
        ga_seed=int(args.ga_seed),
        verbose_seed_progress=bool(args.verbose_seed_progress),
    )
    search_elapsed = float(time.time() - t0)

    best_solution = ga_result["best_solution"]
    metrics = algo_ga_no_offload.evaluate_solution_metrics(
        env=env,
        agent=agent,
        num_uav=num_uav,
        best_solution=best_solution,
        wind_seed=int(args.wind_seed),
        terrain_seed=int(args.terrain_seed),
        infra_seed=int(args.infra_seed),
        eval_traj_seeds=eval_traj_seeds,
        eval_repetitions=int(args.eval_repetitions),
        ga_seed=int(args.ga_seed),
    )
    metrics = to_coverage_metrics(metrics)
    starts, ends = algo_ga_no_offload.split_chromosome(best_solution, num_uav)

    return {
        "group_id": "ga_no_offloading",
        "group_label": "GA + No Offloading",
        "search_elapsed_sec": search_elapsed,
        "search": {
            "ga_objective_mean_average_uncertainty": float(ga_result["best_fitness"]),
            "best_run_traj_seed": (
                None if ga_result["best_run_traj_seed"] is None else int(ga_result["best_run_traj_seed"])
            ),
            "best_step_count": int(ga_result["best_step_count"]),
            "best_iteration_seed_set": [int(s) for s in ga_result["best_iteration_seed_set"]],
        },
        "best": {
            "best_solution": [[int(p[0]), int(p[1])] for p in best_solution],
            "start_positions": [[int(p[0]), int(p[1])] for p in starts],
            "end_positions": [[int(p[0]), int(p[1])] for p in ends],
        },
        "metrics": metrics,
    }


def run_no_ga_offloading_group(args, num_uav, eval_traj_seeds):
    set_seed(args.ga_seed)
    env, agent, offload_agent = build_agents_and_env(num_uav=num_uav)

    t0 = time.time()
    metrics = evaluate_no_ga_offloading(
        env=env,
        agent=agent,
        offload_agent=offload_agent,
        wind_seed=int(args.wind_seed),
        terrain_seed=int(args.terrain_seed),
        infra_seed=int(args.infra_seed),
        eval_traj_seeds=eval_traj_seeds,
        eval_repetitions=int(args.eval_repetitions),
    )
    eval_elapsed = float(time.time() - t0)

    return {
        "group_id": "no_ga_offloading",
        "group_label": "No-GA + Offloading",
        "search_elapsed_sec": 0.0,
        "eval_elapsed_sec": eval_elapsed,
        "search": {
            "ga_objective_mean_average_uncertainty": None,
            "best_run_traj_seed": None,
            "best_step_count": None,
            "best_iteration_seed_set": [],
        },
        "best": {
            "best_solution": None,
            "start_positions": None,
            "end_positions": None,
        },
        "metrics": metrics,
    }


def format_pm(mean_value, std_value, digits=2):
    return f"{float(mean_value):.{digits}f} ± {float(std_value):.{digits}f}"


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


def markdown_table(headers, rows):
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def print_terminal_table(per_uav_results, table_format):
    headers = [
        "UAVs",
        "GA+Offload Coverage(%)",
        "GA+NoOffload Coverage(%)",
        "NoGA+Offload Coverage(%)",
        "GA+Offload AvgUnc",
        "GA+NoOffload AvgUnc",
        "NoGA+Offload AvgUnc",
    ]
    rows = []
    for item in per_uav_results:
        ga_off = item["groups"]["ga_offloading"]["metrics"]
        ga_no = item["groups"]["ga_no_offloading"]["metrics"]
        no_ga = item["groups"]["no_ga_offloading"]["metrics"]

        rows.append(
            [
                str(item["num_uav"]),
                format_pm(ga_off["mean_coverage_percent"], ga_off["std_coverage_percent"], digits=2),
                format_pm(ga_no["mean_coverage_percent"], ga_no["std_coverage_percent"], digits=2),
                format_pm(no_ga["mean_coverage_percent"], no_ga["std_coverage_percent"], digits=2),
                format_pm(ga_off["mean_average_uncertainty"], ga_off["std_average_uncertainty"], digits=4),
                format_pm(ga_no["mean_average_uncertainty"], ga_no["std_average_uncertainty"], digits=4),
                format_pm(no_ga["mean_average_uncertainty"], no_ga["std_average_uncertainty"], digits=4),
            ]
        )

    print("")
    if table_format == "markdown":
        print(markdown_table(headers, rows))
    else:
        print(plain_table(headers, rows))


def save_plots(per_uav_results, coverage_plot_path, uncertainty_plot_path, args):
    uav_counts = [item["num_uav"] for item in per_uav_results]

    ga_off_cov = [item["groups"]["ga_offloading"]["metrics"]["mean_coverage_percent"] for item in per_uav_results]
    ga_no_cov = [item["groups"]["ga_no_offloading"]["metrics"]["mean_coverage_percent"] for item in per_uav_results]
    no_ga_cov = [item["groups"]["no_ga_offloading"]["metrics"]["mean_coverage_percent"] for item in per_uav_results]

    ga_off_unc = [item["groups"]["ga_offloading"]["metrics"]["mean_average_uncertainty"] for item in per_uav_results]
    ga_no_unc = [item["groups"]["ga_no_offloading"]["metrics"]["mean_average_uncertainty"] for item in per_uav_results]
    no_ga_unc = [item["groups"]["no_ga_offloading"]["metrics"]["mean_average_uncertainty"] for item in per_uav_results]

    os.makedirs(os.path.dirname(coverage_plot_path), exist_ok=True)
    os.makedirs(os.path.dirname(uncertainty_plot_path), exist_ok=True)

    # Coverage plot
    plt.figure(figsize=(10, 6))
    plt.plot(uav_counts, ga_off_cov, "o-", color="blue", label="GA + Offloading")
    plt.plot(uav_counts, ga_no_cov, "s--", color="red", label="GA + No Offloading")
    plt.plot(uav_counts, no_ga_cov, "^-.", color="green", label="No-GA + Offloading")
    plt.title(
        f"Final Coverage Comparison ({args.wind_seed=}, {args.terrain_seed=}, {args.infra_seed=})".replace("args.", "")
    )
    plt.xlabel("Number of UAVs")
    plt.ylabel("Final Coverage (%)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(coverage_plot_path, dpi=220)
    plt.close()

    # Uncertainty plot (style similar to provided figure)
    plt.figure(figsize=(10, 6))
    plt.plot(uav_counts, ga_off_unc, "o-", color="blue", label="GA + Offloading")
    plt.plot(uav_counts, ga_no_unc, "s--", color="red", label="GA + No Offloading")
    plt.plot(uav_counts, no_ga_unc, "^-.", color="green", label="No-GA + Offloading")
    plt.title("Comparison of Uncertainty with/without Offloading")
    plt.xlabel("Number of UAVs")
    plt.ylabel("Average Uncertainty")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(uncertainty_plot_path, dpi=220)
    plt.close()


def main():
    args = parse_args()
    validate_args(args)

    uav_counts = parse_uav_counts(args.uav_counts)
    eval_traj_seeds, eval_replace_used = sample_eval_traj_seeds(args)
    if len(eval_traj_seeds) == 0:
        raise RuntimeError("No evaluation trajectory seeds selected.")
    print(
        f"[Evaluation Seeds] pool={args.eval_traj_seed_pool_min}-{args.eval_traj_seed_pool_max}, "
        f"sample_size={args.eval_traj_seed_sample_size}, replace={eval_replace_used}, "
        f"sampler_seed={args.eval_traj_seed_sampler_seed}, seeds={eval_traj_seeds}"
    )

    per_uav_results = []
    t_all = time.time()

    for num_uav in uav_counts:
        print("=" * 100)
        print(f"[UAV Count {num_uav}] Running 3-group comparison...")

        ga_off = run_ga_offloading_group(args, num_uav, eval_traj_seeds)
        print(
            f"  GA+Offloading: coverage={ga_off['metrics']['mean_coverage_percent']:.2f}% "
            f"unc={ga_off['metrics']['mean_average_uncertainty']:.6f}"
        )

        ga_no = run_ga_no_offloading_group(args, num_uav, eval_traj_seeds)
        print(
            f"  GA+NoOffloading: coverage={ga_no['metrics']['mean_coverage_percent']:.2f}% "
            f"unc={ga_no['metrics']['mean_average_uncertainty']:.6f}"
        )

        no_ga = run_no_ga_offloading_group(args, num_uav, eval_traj_seeds)
        print(
            f"  NoGA+Offloading: coverage={no_ga['metrics']['mean_coverage_percent']:.2f}% "
            f"unc={no_ga['metrics']['mean_average_uncertainty']:.6f}"
        )

        per_uav_results.append(
            {
                "num_uav": int(num_uav),
                "groups": {
                    "ga_offloading": ga_off,
                    "ga_no_offloading": ga_no,
                    "no_ga_offloading": no_ga,
                },
            }
        )

    total_elapsed = float(time.time() - t_all)

    output_json = resolve_path_template(
        args.output_json,
        wind_seed=int(args.wind_seed),
        terrain_seed=int(args.terrain_seed),
        infra_seed=int(args.infra_seed),
    )
    output_json = normalize_output_path(output_json)
    coverage_plot_path = resolve_path_template(
        args.coverage_plot_path,
        wind_seed=int(args.wind_seed),
        terrain_seed=int(args.terrain_seed),
        infra_seed=int(args.infra_seed),
    )
    coverage_plot_path = normalize_output_path(coverage_plot_path)
    uncertainty_plot_path = resolve_path_template(
        args.uncertainty_plot_path,
        wind_seed=int(args.wind_seed),
        terrain_seed=int(args.terrain_seed),
        infra_seed=int(args.infra_seed),
    )
    uncertainty_plot_path = normalize_output_path(uncertainty_plot_path)

    result = {
        "created_at": utc_now_iso(),
        "summary_type": "uav_count_three_group_comparison",
        "groups": [
            {"group_id": "ga_offloading", "group_label": "GA + Offloading"},
            {"group_id": "ga_no_offloading", "group_label": "GA + No Offloading"},
            {"group_id": "no_ga_offloading", "group_label": "No-GA + Offloading"},
        ],
        "config": {
            "uav_counts": [int(x) for x in uav_counts],
            "infra_seed": int(args.infra_seed),
            "terrain_seed": int(args.terrain_seed),
            "wind_seed": int(args.wind_seed),
            "ga_search": {
                "traj_seed_min": int(args.traj_seed_min),
                "traj_seed_max": int(args.traj_seed_max),
                "traj_seed_sample_size": int(args.traj_seed_sample_size),
                "sample_with_replacement": bool(args.sample_with_replacement),
                "iterations": int(args.iterations),
                "population_size": int(args.population_size),
                "repetitions": int(args.repetitions),
                "mutation_rate": float(args.mutation_rate),
                "ga_seed": int(args.ga_seed),
            },
            "evaluation": {
                "eval_traj_seed_pool_min": int(args.eval_traj_seed_pool_min),
                "eval_traj_seed_pool_max": int(args.eval_traj_seed_pool_max),
                "eval_traj_seed_sample_size": int(args.eval_traj_seed_sample_size),
                "eval_traj_seed_sampler_seed": int(args.eval_traj_seed_sampler_seed),
                "eval_sample_with_replacement": bool(args.eval_sample_with_replacement),
                "eval_replace_used": bool(eval_replace_used),
                "eval_traj_seeds": [int(s) for s in eval_traj_seeds],
                "eval_repetitions": int(args.eval_repetitions),
                "num_eval_traj_seeds": int(len(eval_traj_seeds)),
            },
            "total_elapsed_sec": total_elapsed,
        },
        "per_uav_count_results": per_uav_results,
        "artifacts": {
            "coverage_plot_path": coverage_plot_path,
            "uncertainty_plot_path": uncertainty_plot_path,
        },
    }
    save_json(output_json, result)

    save_plots(per_uav_results, coverage_plot_path, uncertainty_plot_path, args)

    print("-" * 100)
    print(f"Output JSON: {output_json}")
    print(f"Coverage plot: {coverage_plot_path}")
    print(f"Uncertainty plot: {uncertainty_plot_path}")
    print(f"Total elapsed: {total_elapsed:.1f}s")
    print_terminal_table(per_uav_results, args.table_format)


if __name__ == "__main__":
    main()
