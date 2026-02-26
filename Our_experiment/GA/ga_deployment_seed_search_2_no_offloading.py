import argparse
import os
import pickle
import random
import sys
import time

import numpy as np


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Our_experiment.GA.ga_vis_common import build_agents_and_env
from Our_experiment.GA.ga_vis_common import resolve_path_template
from Our_experiment.GA.ga_vis_common import save_json
from Our_experiment.GA.ga_vis_common import set_seed
from Our_experiment.GA.ga_vis_common import utc_now_iso
from Our_experiment.GA.ga_vis_common import validate_traj_range


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "GA deployment search with random trajectory seeds per iteration (no offloading policy). "
            "Offloading action is fixed to local processing (action=0) for all UAVs."
        )
    )
    parser.add_argument("--num-uav", type=int, default=4, help="Number of UAVs.")
    parser.add_argument("--wind-seed", type=int, default=4800, help="Wind seed.")
    parser.add_argument("--terrain-seed", type=int, default=10, help="Terrain seed.")
    parser.add_argument("--infra-seed", type=int, default=999999, help="Infrastructure seed.")
    parser.add_argument(
        "--traj-seed-min",
        type=int,
        default=0,
        help="Minimum trajectory seed in the sampling pool (inclusive).",
    )
    parser.add_argument(
        "--traj-seed-max",
        type=int,
        default=200,
        help="Maximum trajectory seed in the sampling pool (inclusive).",
    )
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
    parser.add_argument("--iterations", type=int, default=20, help="GA generation count.")
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
        "--eval-traj-seed-start",
        type=int,
        default=None,
        help="Trajectory seed start used for final metric summary. Default: traj-seed-min.",
    )
    parser.add_argument(
        "--eval-traj-seed-end",
        type=int,
        default=None,
        help="Trajectory seed end used for final metric summary. Default: traj-seed-max.",
    )
    parser.add_argument(
        "--eval-repetitions",
        type=int,
        default=1,
        help="Rollout repetitions for each eval trajectory seed when computing final statistics.",
    )
    parser.add_argument(
        "--result-json",
        type=str,
        default=(
            "Our_experiment/GA/data/"
            "ga_best_deployment_nooffload_random{sample_size}_w{wind_seed}_g{terrain_seed}_i{infra_seed}"
            "_pool{seed_min}_{seed_max}.json"
        ),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--result-pkl",
        type=str,
        default=(
            "Our_experiment/GA/data/"
            "ga_best_deployment_nooffload_random{sample_size}_w{wind_seed}_g{terrain_seed}_i{infra_seed}"
            "_pool{seed_min}_{seed_max}.pkl"
        ),
        help="Output PKL path.",
    )
    return parser.parse_args()


def init_population(population_size, num_uav, env_lx, env_ly):
    population = []
    for _ in range(population_size):
        chromosome_start = []
        chromosome_end = []
        for _ in range(num_uav):
            x_start = int(np.random.randint(0, env_lx))
            y_start = int(np.random.randint(0, env_ly))
            x_end = int(np.random.randint(0, env_lx))
            y_end = int(np.random.randint(0, env_ly))
            chromosome_start.append((x_start, y_start))
            chromosome_end.append((x_end, y_end))
        population.append(chromosome_start + chromosome_end)
    return population


def selection(population, fitnesses):
    pairs = list(zip(fitnesses, population))
    pairs_sorted = sorted(pairs, key=lambda x: x[0])
    n = max(2, len(population) // 2)
    selected = [pair[1] for pair in pairs_sorted[:n]]
    return selected


def crossover(parent1, parent2):
    cross_point = int(np.random.randint(0, len(parent1)))
    child1 = parent1[:cross_point] + parent2[cross_point:]
    child2 = parent2[:cross_point] + parent1[cross_point:]
    return child1, child2


def mutate(chromosome, env_lx, env_ly, mutation_rate=0.1):
    for i in range(len(chromosome)):
        if np.random.rand() < mutation_rate:
            p = int(np.random.randint(0, env_lx * env_ly))
            x = int(p // env_ly)
            y = int(p % env_ly)
            chromosome[i] = (x, y)
    return chromosome


def split_chromosome(chromosome, num_uav):
    starts = [tuple(chromosome[i]) for i in range(num_uav)]
    ends = [tuple(chromosome[num_uav + i]) for i in range(num_uav)]
    return starts, ends


def sample_traj_seeds(seed_pool, sample_size, rng, with_replacement=False):
    if len(seed_pool) == 0:
        raise ValueError("trajectory seed pool is empty.")
    replace = bool(with_replacement) or sample_size > len(seed_pool)
    sampled = rng.choice(seed_pool, size=sample_size, replace=replace)
    return [int(s) for s in sampled.tolist()]


def trajectory_execution_no_offloading(
    env,
    agent,
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
    trajectory_actions = []
    uav_lifetime_steps = np.full(env.N, np.nan, dtype=np.float64)
    avg_uncertainty = float(np.mean(env.uncertainty_matrix))
    while not done:
        step_count += 1
        actions = [agent.take_action(state[n]) for n in range(env.N)]
        trajectory_actions.append(actions)
        next_state, _, done_move = env.step(actions)
        local_offload_actions = [0] * env.N
        _, _, done_offload = env.step_offload(local_offload_actions)
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
        "step_count": int(step_count),
        "trajectory_actions": trajectory_actions,
        "uav_lifetime_steps": uav_lifetime_steps.tolist(),
        "avg_uav_lifetime_steps": float(avg_uav_lifetime_steps),
        "avg_uav_lifetime_seconds": float(avg_uav_lifetime_seconds),
    }


def compose_rollout_seed(ga_seed, iteration, population_index, seed_index, repetition):
    value = ga_seed
    value = value * 1000003 + iteration * 7919
    value = value + population_index * 3571 + seed_index * 1013 + repetition * 97
    return int(value % 2147483647)


def evaluate_chromosome(
    chromosome,
    env,
    agent,
    num_uav,
    wind_seed,
    terrain_seed,
    infra_seed,
    traj_seeds,
    repetitions,
    ga_seed,
    iteration,
    pop_index,
    pop_size,
    verbose_seed_progress=False,
):
    starts, ends = split_chromosome(chromosome, num_uav)
    seed_best_fitnesses = []

    single_best_fitness = 1.0
    single_best_step_count = 0
    single_best_trajectory_actions = []
    single_best_traj_seed = None

    t0_candidate = time.time()
    total_seed_count = len(traj_seeds)
    for seed_idx, traj_seed in enumerate(traj_seeds):
        if verbose_seed_progress:
            print(
                f"  [seed-progress] iteration={iteration} population={pop_index}/{pop_size - 1} "
                f"seed={seed_idx + 1}/{total_seed_count} (traj_seed={traj_seed})"
            )

        r_fitness = 1.0
        r_step_count = 0
        r_trajectory_actions = []

        for rep in range(repetitions):
            rollout_seed = compose_rollout_seed(
                ga_seed=ga_seed,
                iteration=iteration,
                population_index=pop_index,
                seed_index=seed_idx,
                repetition=rep,
            )
            rollout_result = trajectory_execution_no_offloading(
                env=env,
                agent=agent,
                position_starts=starts,
                position_ends=ends,
                wind_seed=wind_seed,
                terrain_seed=terrain_seed,
                infra_seed=infra_seed,
                traj_seed=traj_seed,
                rollout_seed=rollout_seed,
            )
            _fitness = float(rollout_result["avg_uncertainty"])
            _step_count = int(rollout_result["step_count"])
            _traj_actions = list(rollout_result["trajectory_actions"])

            if _fitness < r_fitness:
                r_fitness = float(_fitness)
                r_step_count = int(_step_count)
                r_trajectory_actions = list(_traj_actions)

        seed_best_fitnesses.append(float(r_fitness))
        if r_fitness < single_best_fitness:
            single_best_fitness = float(r_fitness)
            single_best_step_count = int(r_step_count)
            single_best_trajectory_actions = list(r_trajectory_actions)
            single_best_traj_seed = int(traj_seed)

    candidate_fitness = float(np.mean(seed_best_fitnesses)) if seed_best_fitnesses else float("inf")
    elapsed = float(time.time() - t0_candidate)
    return {
        "candidate_fitness": candidate_fitness,
        "single_best_fitness": float(single_best_fitness),
        "single_best_step_count": int(single_best_step_count),
        "single_best_trajectory_actions": single_best_trajectory_actions,
        "single_best_traj_seed": single_best_traj_seed,
        "elapsed_sec": elapsed,
    }


def genetic_algorithm(
    iterations,
    population_size,
    repetitions,
    mutation_rate,
    env,
    agent,
    num_uav,
    env_lx,
    env_ly,
    wind_seed,
    terrain_seed,
    infra_seed,
    traj_seed_pool,
    traj_seed_sample_size,
    sample_with_replacement,
    seed_sampler_rng,
    ga_seed,
    verbose_seed_progress=False,
):
    population = init_population(population_size, num_uav, env_lx, env_ly)

    best_fitness = 1.0
    best_solution = None
    best_trajectory_actions = []
    best_step_count = 0
    best_run_traj_seed = None
    best_iteration_seed_set = []

    fitness_set_max = []
    fitness_set_min = []
    fitness_set_mean = []
    iteration_sampled_traj_seeds = []

    for iteration in range(iterations):
        traj_seeds = sample_traj_seeds(
            seed_pool=traj_seed_pool,
            sample_size=traj_seed_sample_size,
            rng=seed_sampler_rng,
            with_replacement=sample_with_replacement,
        )
        iteration_sampled_traj_seeds.append([int(s) for s in traj_seeds])

        print("=" * 90)
        print(f"Iteration {iteration + 1}/{iterations} started.")
        print(f"Iteration {iteration + 1}/{iterations} sampled traj seeds: {traj_seeds}")

        fitnesses = []
        b_fitness = 1.0
        generation_best_trajectory_actions = []
        generation_best_step_count = 0
        generation_best_traj_seed = None

        for i, chromo in enumerate(population):
            result = evaluate_chromosome(
                chromosome=chromo,
                env=env,
                agent=agent,
                num_uav=num_uav,
                wind_seed=wind_seed,
                terrain_seed=terrain_seed,
                infra_seed=infra_seed,
                traj_seeds=traj_seeds,
                repetitions=repetitions,
                ga_seed=ga_seed,
                iteration=iteration,
                pop_index=i,
                pop_size=population_size,
                verbose_seed_progress=verbose_seed_progress,
            )
            candidate_fitness = float(result["candidate_fitness"])
            single_best_fitness = float(result["single_best_fitness"])
            fitnesses.append(candidate_fitness)

            if single_best_fitness < b_fitness:
                b_fitness = single_best_fitness
                generation_best_trajectory_actions = list(result["single_best_trajectory_actions"])
                generation_best_step_count = int(result["single_best_step_count"])
                generation_best_traj_seed = result["single_best_traj_seed"]

            print(
                f"Iteration {iteration}, population {i},Chromosome: {chromo} "
                f"current_best_fitness {candidate_fitness:.6f}  best_fitness: {b_fitness:.6f} "
                f"_step_count: {int(result['single_best_step_count'])} "
                f"len:{len(result['single_best_trajectory_actions'])} "
                f"elapsed:{result['elapsed_sec']:.1f}s"
            )

        fit_min = float(np.min(fitnesses))
        fit_max = float(np.max(fitnesses))
        fit_mean = float(np.mean(fitnesses))
        print(
            f"Iteration {iteration}, Best Average Uncertainty: {fit_min:.6f} "
            f"max {fit_max:.6f} mean {fit_mean:.6f}"
        )
        fitness_set_max.append(fit_max)
        fitness_set_min.append(fit_min)
        fitness_set_mean.append(fit_mean)

        current_best = fit_min
        current_best_idx = int(np.argmin(fitnesses))
        if best_solution is None or current_best < best_fitness:
            best_fitness = float(current_best)
            best_solution = list(population[current_best_idx])
            best_trajectory_actions = list(generation_best_trajectory_actions)
            best_step_count = int(generation_best_step_count)
            best_run_traj_seed = generation_best_traj_seed
            best_iteration_seed_set = list(traj_seeds)
            print(
                f"  -> New global best at iteration {iteration}: "
                f"fitness={best_fitness:.6f}, traj_seed={best_run_traj_seed}, step_count={best_step_count}"
            )

        parents = selection(population, fitnesses)
        next_generation = []
        while len(next_generation) < population_size:
            p1, p2 = random.sample(parents, 2)
            child1, child2 = crossover(p1, p2)
            next_generation.append(list(child1))
            if len(next_generation) < population_size:
                next_generation.append(list(child2))

        new_population = []
        for _chromo in next_generation:
            new_population.append(mutate(_chromo.copy(), env_lx, env_ly, mutation_rate=mutation_rate))
        population = new_population

        print(f"Iteration {iteration}, Final Best Average Uncertainty: {best_fitness:.6f}")

    return {
        "best_solution": best_solution,
        "best_fitness": best_fitness,
        "trajectory_actions": best_trajectory_actions,
        "fitness_set_max": fitness_set_max,
        "fitness_set_min": fitness_set_min,
        "fitness_set_mean": fitness_set_mean,
        "best_step_count": best_step_count,
        "best_run_traj_seed": best_run_traj_seed,
        "best_iteration_seed_set": best_iteration_seed_set,
        "iteration_sampled_traj_seeds": iteration_sampled_traj_seeds,
    }


def evaluate_solution_metrics(
    env,
    agent,
    num_uav,
    best_solution,
    wind_seed,
    terrain_seed,
    infra_seed,
    eval_traj_seeds,
    eval_repetitions,
    ga_seed,
):
    starts, ends = split_chromosome(best_solution, num_uav)
    uncertainty_values = []
    lifetime_steps_values = []
    lifetime_seconds_values = []

    for seed_idx, traj_seed in enumerate(eval_traj_seeds):
        for rep in range(eval_repetitions):
            rollout_seed = compose_rollout_seed(
                ga_seed=ga_seed + 7919,
                iteration=99991,
                population_index=0,
                seed_index=seed_idx,
                repetition=rep,
            )
            rollout_result = trajectory_execution_no_offloading(
                env=env,
                agent=agent,
                position_starts=starts,
                position_ends=ends,
                wind_seed=wind_seed,
                terrain_seed=terrain_seed,
                infra_seed=infra_seed,
                traj_seed=int(traj_seed),
                rollout_seed=rollout_seed,
            )
            uncertainty_values.append(float(rollout_result["avg_uncertainty"]))
            lifetime_steps_values.append(float(rollout_result["avg_uav_lifetime_steps"]))
            lifetime_seconds_values.append(float(rollout_result["avg_uav_lifetime_seconds"]))

    if len(uncertainty_values) == 0:
        raise RuntimeError("No evaluation rollouts were executed for the best solution.")

    return {
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


def main():
    args = parse_args()
    validate_traj_range(args.traj_seed_min, args.traj_seed_max)
    if args.population_size < 2:
        raise ValueError("population-size must be >= 2")
    if args.iterations < 1:
        raise ValueError("iterations must be >= 1")
    if args.repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if args.traj_seed_sample_size < 1:
        raise ValueError("traj-seed-sample-size must be >= 1")
    if args.eval_repetitions < 1:
        raise ValueError("eval-repetitions must be >= 1")
    if not (0.0 <= args.mutation_rate <= 1.0):
        raise ValueError("mutation-rate must be in [0, 1]")

    eval_traj_seed_start = int(args.eval_traj_seed_start) if args.eval_traj_seed_start is not None else int(args.traj_seed_min)
    eval_traj_seed_end = int(args.eval_traj_seed_end) if args.eval_traj_seed_end is not None else int(args.traj_seed_max)
    validate_traj_range(eval_traj_seed_start, eval_traj_seed_end)
    eval_traj_seeds = list(range(eval_traj_seed_start, eval_traj_seed_end + 1))

    set_seed(args.ga_seed)
    seed_sampler_rng = np.random.default_rng(args.ga_seed)

    env, agent, _ = build_agents_and_env(num_uav=args.num_uav)
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

    eval_metrics = evaluate_solution_metrics(
        env=env,
        agent=agent,
        num_uav=args.num_uav,
        best_solution=best_solution,
        wind_seed=args.wind_seed,
        terrain_seed=args.terrain_seed,
        infra_seed=args.infra_seed,
        eval_traj_seeds=eval_traj_seeds,
        eval_repetitions=args.eval_repetitions,
        ga_seed=args.ga_seed,
    )

    result_json = resolve_path_template(
        args.result_json,
        wind_seed=args.wind_seed,
        terrain_seed=args.terrain_seed,
        infra_seed=args.infra_seed,
        seed_min=args.traj_seed_min,
        seed_max=args.traj_seed_max,
        sample_size=args.traj_seed_sample_size,
    )
    result_pkl = resolve_path_template(
        args.result_pkl,
        wind_seed=args.wind_seed,
        terrain_seed=args.terrain_seed,
        infra_seed=args.infra_seed,
        seed_min=args.traj_seed_min,
        seed_max=args.traj_seed_max,
        sample_size=args.traj_seed_sample_size,
    )

    result_data = {
        "created_at": utc_now_iso(),
        "mode": "no_offloading_local_only",
        "seeds": {
            "wind_seed": int(args.wind_seed),
            "terrain_seed": int(args.terrain_seed),
            "infra_seed": int(args.infra_seed),
            "traj_seed_start": int(args.traj_seed_min),
            "traj_seed_end": int(args.traj_seed_max),
            "traj_seed_mode": "random_per_iteration",
            "traj_seed_pool_min": int(args.traj_seed_min),
            "traj_seed_pool_max": int(args.traj_seed_max),
            "traj_seed_sample_size": int(args.traj_seed_sample_size),
            "sample_with_replacement": bool(args.sample_with_replacement),
        },
        "evaluation_seeds": {
            "eval_traj_seed_start": int(eval_traj_seed_start),
            "eval_traj_seed_end": int(eval_traj_seed_end),
            "eval_repetitions": int(args.eval_repetitions),
            "eval_traj_seed_count": int(len(eval_traj_seeds)),
        },
        "ga": {
            "iterations": int(args.iterations),
            "population_size": int(args.population_size),
            "repetitions": int(args.repetitions),
            "mutation_rate": float(args.mutation_rate),
            "ga_seed": int(args.ga_seed),
            "elapsed_sec": total_elapsed,
            "iteration_sampled_traj_seeds": ga_result["iteration_sampled_traj_seeds"],
        },
        "grid": {
            "lx": int(env.Lx),
            "ly": int(env.Ly),
            "grid_cell_size_m": float(getattr(env, "grid_cell_size_m", env.X / env.Lx)),
        },
        "best": {
            "ga_objective_mean_average_uncertainty": float(ga_result["best_fitness"]),
            "mean_average_uncertainty": float(eval_metrics["mean_average_uncertainty"]),
            "std_average_uncertainty": float(eval_metrics["std_average_uncertainty"]),
            "mean_uav_lifetime_steps": float(eval_metrics["mean_uav_lifetime_steps"]),
            "std_uav_lifetime_steps": float(eval_metrics["std_uav_lifetime_steps"]),
            "mean_uav_lifetime_seconds": float(eval_metrics["mean_uav_lifetime_seconds"]),
            "std_uav_lifetime_seconds": float(eval_metrics["std_uav_lifetime_seconds"]),
            "best_solution": [[int(p[0]), int(p[1])] for p in best_solution],
            "start_positions": [[int(p[0]), int(p[1])] for p in starts],
            "end_positions": [[int(p[0]), int(p[1])] for p in ends],
            "deployment_positions": [[int(p[0]), int(p[1])] for p in starts],
            "deployment_destinations": [[int(p[0]), int(p[1])] for p in ends],
            "best_run_traj_seed": (
                None if ga_result["best_run_traj_seed"] is None else int(ga_result["best_run_traj_seed"])
            ),
            "best_step_count": int(ga_result["best_step_count"]),
            "best_iteration_seed_set": [int(s) for s in ga_result["best_iteration_seed_set"]],
            "trajectory_actions": ga_result["trajectory_actions"],
        },
        "convergence": {
            "fitness_max": [float(x) for x in ga_result["fitness_set_max"]],
            "fitness_min": [float(x) for x in ga_result["fitness_set_min"]],
            "fitness_mean": [float(x) for x in ga_result["fitness_set_mean"]],
        },
    }
    save_json(result_json, result_data)

    pkl_parent = os.path.dirname(result_pkl)
    if pkl_parent:
        os.makedirs(pkl_parent, exist_ok=True)
    with open(result_pkl, "wb") as f:
        pickle.dump(
            {
                "best_solution": best_solution,
                "best_fitness": ga_result["best_fitness"],
                "trajectory_actions": ga_result["trajectory_actions"],
                "eval_metrics": eval_metrics,
            },
            f,
        )

    print("-" * 90)
    print("GA deployment search (random traj seeds per iteration, no offloading policy) completed.")
    print(f"Result JSON: {result_json}")
    print(f"Result PKL: {result_pkl}")
    print(f"Best solution: {best_solution}")
    print(f"Best start positions: {starts}")
    print(f"Best end positions: {ends}")
    print(f"Best run traj_seed: {ga_result['best_run_traj_seed']}")
    print(f"Best iteration sampled seeds: {ga_result['best_iteration_seed_set']}")
    print(
        f"Trajectory seed sampling config: min={args.traj_seed_min}, "
        f"max={args.traj_seed_max}, sample_size={args.traj_seed_sample_size}, "
        f"with_replacement={bool(args.sample_with_replacement)}"
    )
    print(
        f"Evaluation traj seeds: start={eval_traj_seed_start}, end={eval_traj_seed_end}, "
        f"count={len(eval_traj_seeds)}, repetitions={args.eval_repetitions}"
    )
    print(f"GA objective mean average uncertainty: {ga_result['best_fitness']:.6f}")
    print(f"Mean Average uncertainty: {eval_metrics['mean_average_uncertainty']:.6f}")
    print(f"Std Average uncertainty: {eval_metrics['std_average_uncertainty']:.6f}")
    print(
        f"Mean UAV lifetime: {eval_metrics['mean_uav_lifetime_steps']:.2f} steps "
        f"({eval_metrics['mean_uav_lifetime_seconds']:.2f} s)"
    )
    print(
        f"Std UAV lifetime: {eval_metrics['std_uav_lifetime_steps']:.2f} steps "
        f"({eval_metrics['std_uav_lifetime_seconds']:.2f} s)"
    )
    print(f"Total elapsed: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
