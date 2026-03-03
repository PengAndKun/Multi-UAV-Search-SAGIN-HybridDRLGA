import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Our_experiment.GA.ga_deployment_seed_search_2 import genetic_algorithm
from Our_experiment.GA.ga_deployment_seed_search_2 import split_chromosome
from Our_experiment.GA.ga_vis_common import build_agents_and_env
from Our_experiment.GA.ga_vis_common import resolve_path_template
from Our_experiment.GA.ga_vis_common import set_seed
from Our_experiment.GA.ga_vis_common import validate_traj_range


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run GA deployment search and save a coverage convergence plot. "
            "Coverage is defined as 1 - uncertainty."
        )
    )
    parser.add_argument("--num-uav", type=int, default=4, help="Number of UAVs.")
    parser.add_argument("--wind-seed", type=int, default=4800, help="Wind seed.")
    parser.add_argument("--terrain-seed", type=int, default=10, help="Terrain seed.")
    parser.add_argument("--infra-seed", type=int, default=999999, help="Infrastructure seed.")
    parser.add_argument("--traj-seed-min", type=int, default=0, help="Trajectory seed pool min (inclusive).")
    parser.add_argument("--traj-seed-max", type=int, default=200, help="Trajectory seed pool max (inclusive).")
    parser.add_argument(
        "--traj-seed-sample-size",
        type=int,
        default=10,
        help="Number of random trajectory seeds sampled per iteration.",
    )
    parser.add_argument(
        "--sample-with-replacement",
        action="store_true",
        help="Sample trajectory seeds with replacement.",
    )
    parser.add_argument("--iterations", type=int, default=30, help="GA generation count.")
    parser.add_argument("--population-size", type=int, default=12, help="GA population size.")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Rollout repetitions for each sampled trajectory seed and chromosome.",
    )
    parser.add_argument("--mutation-rate", type=float, default=0.1, help="Mutation probability per gene.")
    parser.add_argument("--ga-seed", type=int, default=2026, help="GA random seed.")
    parser.add_argument(
        "--verbose-seed-progress",
        action="store_true",
        help="Print per-sampled-seed progress for each chromosome evaluation.",
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=(
            "data/GA/"
            "ga_coverage_convergence_random{sample_size}_w{wind_seed}_g{terrain_seed}_i{infra_seed}"
            "_pool{seed_min}_{seed_max}.png"
        ),
        help="Output coverage convergence plot path.",
    )
    return parser.parse_args()


def validate_args(args):
    validate_traj_range(args.traj_seed_min, args.traj_seed_max)
    if args.population_size < 2:
        raise ValueError("population-size must be >= 2")
    if args.iterations < 1:
        raise ValueError("iterations must be >= 1")
    if args.repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if args.traj_seed_sample_size < 1:
        raise ValueError("traj-seed-sample-size must be >= 1")
    if not (0.0 <= args.mutation_rate <= 1.0):
        raise ValueError("mutation-rate must be in [0, 1]")


def fitness_to_coverage_curves(ga_result):
    fitness_max = np.array(ga_result["fitness_set_max"], dtype=np.float64)
    fitness_min = np.array(ga_result["fitness_set_min"], dtype=np.float64)
    fitness_mean = np.array(ga_result["fitness_set_mean"], dtype=np.float64)
    return {
        "coverage_max": 1.0 - fitness_min,
        "coverage_mean": 1.0 - fitness_mean,
        "coverage_min": 1.0 - fitness_max,
    }


def save_coverage_convergence_plot(coverage_max, coverage_mean, coverage_min, output_path):
    generations = np.arange(1, len(coverage_mean) + 1, dtype=np.int64)
    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    plt.figure(figsize=(11, 7))
    plt.plot(generations, coverage_max, color="#4C72B0", linewidth=2.4, label="Coverage Max")
    plt.plot(generations, coverage_mean, color="#DD8452", linewidth=2.4, label="Coverage Mean")
    plt.plot(generations, coverage_min, color="#55A868", linewidth=2.4, label="Coverage Min")
    plt.title("GA Coverage Plot", fontsize=18)
    plt.xlabel("Generation", fontsize=14)
    plt.ylabel("Coverage", fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.45)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def main():
    args = parse_args()
    validate_args(args)

    set_seed(args.ga_seed)
    seed_sampler_rng = np.random.default_rng(args.ga_seed)

    env, agent, offload_agent = build_agents_and_env(num_uav=args.num_uav)
    traj_seed_pool = np.arange(args.traj_seed_min, args.traj_seed_max + 1, dtype=np.int64)
    if len(traj_seed_pool) == 0:
        raise ValueError("trajectory seed pool is empty.")

    t0 = time.time()
    ga_result = genetic_algorithm(
        iterations=args.iterations,
        population_size=args.population_size,
        repetitions=args.repetitions,
        mutation_rate=args.mutation_rate,
        env=env,
        agent=agent,
        offload_agent=offload_agent,
        num_uav=args.num_uav,
        env_lx=env.Lx,
        env_ly=env.Ly,
        wind_seed=args.wind_seed,
        terrain_seed=args.terrain_seed,
        infra_seed=args.infra_seed,
        traj_seed_pool=traj_seed_pool,
        traj_seed_sample_size=args.traj_seed_sample_size,
        sample_with_replacement=args.sample_with_replacement,
        seed_sampler_rng=seed_sampler_rng,
        ga_seed=args.ga_seed,
        verbose_seed_progress=args.verbose_seed_progress,
    )
    total_elapsed = float(time.time() - t0)

    best_solution = ga_result["best_solution"]
    if best_solution is None:
        raise RuntimeError("GA did not produce a valid best solution.")
    starts, ends = split_chromosome(best_solution, args.num_uav)

    curves = fitness_to_coverage_curves(ga_result)
    plot_path = resolve_path_template(
        args.plot_path,
        wind_seed=args.wind_seed,
        terrain_seed=args.terrain_seed,
        infra_seed=args.infra_seed,
        seed_min=args.traj_seed_min,
        seed_max=args.traj_seed_max,
        sample_size=args.traj_seed_sample_size,
    )
    save_coverage_convergence_plot(
        coverage_max=curves["coverage_max"],
        coverage_mean=curves["coverage_mean"],
        coverage_min=curves["coverage_min"],
        output_path=plot_path,
    )

    best_coverage = float(1.0 - ga_result["best_fitness"])

    print("-" * 90)
    print("GA coverage convergence plotting completed.")
    print(f"Coverage plot: {plot_path}")
    print(f"Best solution: {best_solution}")
    print(f"Best start positions: {starts}")
    print(f"Best end positions: {ends}")
    print(f"Best mean coverage: {best_coverage:.6f}")
    print(f"Best mean average uncertainty: {ga_result['best_fitness']:.6f}")
    print(f"Best run traj_seed: {ga_result['best_run_traj_seed']}")
    print(f"Best iteration sampled seeds: {ga_result['best_iteration_seed_set']}")
    print(
        f"Trajectory seed sampling config: min={args.traj_seed_min}, "
        f"max={args.traj_seed_max}, sample_size={args.traj_seed_sample_size}, "
        f"with_replacement={bool(args.sample_with_replacement)}"
    )
    print(f"Total elapsed: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
