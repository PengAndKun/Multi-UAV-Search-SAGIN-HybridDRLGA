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
            "GA deployment search aligned with GA_Original.ipynb. "
            "Chromosome = [start_positions + end_positions]. "
            "Fitness = mean best uncertainty over trajectory seed range."
        )
    )
    parser.add_argument("--num-uav", type=int, default=4, help="Number of UAVs.")
    parser.add_argument("--wind-seed", type=int, default=4800, help="Wind seed.")
    parser.add_argument("--terrain-seed", type=int, default=10, help="Terrain seed.")
    parser.add_argument("--infra-seed", type=int, default=999999, help="Infrastructure seed.")
    parser.add_argument("--traj-seed-start", type=int, default=0, help="Trajectory seed start (inclusive).")
    parser.add_argument("--traj-seed-end", type=int, default=200, help="Trajectory seed end (inclusive).")
    parser.add_argument("--iterations", type=int, default=8, help="GA generation count.")
    parser.add_argument("--population-size", type=int, default=12, help="GA population size.")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Rollout repetitions for each trajectory seed and chromosome.",
    )
    parser.add_argument("--mutation-rate", type=float, default=0.1, help="Mutation probability per gene.")
    parser.add_argument("--ga-seed", type=int, default=2026, help="GA random seed.")
    parser.add_argument(
        "--verbose-seed-progress",
        action="store_true",
        help="Print progress for each trajectory seed inside candidate evaluation.",
    )
    parser.add_argument(
        "--result-json",
        type=str,
        default=(
            "Our_experiment/GA/data/"
            "ga_best_deployment_w{wind_seed}_g{terrain_seed}_i{infra_seed}_t{start}_{end}.json"
        ),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--result-pkl",
        type=str,
        default=(
            "Our_experiment/GA/data/"
            "ga_best_deployment_w{wind_seed}_g{terrain_seed}_i{infra_seed}_t{start}_{end}.pkl"
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


def trajectory_execution(
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
    trajectory_actions = []
    offloading_actions = []
    while not done:
        step_count += 1
        actions = [agent.take_action(state[n]) for n in range(env.N)]
        trajectory_actions.append(actions)
        next_state, _, done = env.step(actions)
        state = next_state

        offload_data = env.get_obs_2()
        offload_actions = offload_agent.take_action(offload_data)
        offloading_actions.append(offload_actions)
        _, _, done = env.step_offload(offload_actions)

    avg_uncertainty = float(np.mean(env.uncertainty_matrix))
    return avg_uncertainty, step_count, trajectory_actions, offloading_actions


def compose_rollout_seed(ga_seed, iteration, population_index, seed_index, repetition):
    value = ga_seed
    value = value * 1000003 + iteration * 7919
    value = value + population_index * 3571 + seed_index * 1013 + repetition * 97
    return int(value % 2147483647)


def evaluate_chromosome(
    chromosome,
    env,
    agent,
    offload_agent,
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
    single_best_offload_actions = []
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
        r_offload_actions = []

        for rep in range(repetitions):
            rollout_seed = compose_rollout_seed(
                ga_seed=ga_seed,
                iteration=iteration,
                population_index=pop_index,
                seed_index=seed_idx,
                repetition=rep,
            )
            _fitness, _step_count, _traj_actions, _offload_actions = trajectory_execution(
                env=env,
                agent=agent,
                offload_agent=offload_agent,
                position_starts=starts,
                position_ends=ends,
                wind_seed=wind_seed,
                terrain_seed=terrain_seed,
                infra_seed=infra_seed,
                traj_seed=traj_seed,
                rollout_seed=rollout_seed,
            )

            if _fitness < r_fitness:
                r_fitness = float(_fitness)
                r_step_count = int(_step_count)
                r_trajectory_actions = list(_traj_actions)
                r_offload_actions = list(_offload_actions)

        seed_best_fitnesses.append(float(r_fitness))
        if r_fitness < single_best_fitness:
            single_best_fitness = float(r_fitness)
            single_best_step_count = int(r_step_count)
            single_best_trajectory_actions = list(r_trajectory_actions)
            single_best_offload_actions = list(r_offload_actions)
            single_best_traj_seed = int(traj_seed)

    candidate_fitness = float(np.mean(seed_best_fitnesses)) if seed_best_fitnesses else float("inf")
    elapsed = float(time.time() - t0_candidate)
    return {
        "candidate_fitness": candidate_fitness,
        "single_best_fitness": float(single_best_fitness),
        "single_best_step_count": int(single_best_step_count),
        "single_best_trajectory_actions": single_best_trajectory_actions,
        "single_best_offload_actions": single_best_offload_actions,
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
    offload_agent,
    num_uav,
    env_lx,
    env_ly,
    wind_seed,
    terrain_seed,
    infra_seed,
    traj_seeds,
    ga_seed,
    verbose_seed_progress=False,
):
    population = init_population(population_size, num_uav, env_lx, env_ly)

    best_fitness = 1.0
    best_solution = None
    best_trajectory_actions = []
    best_offload_actions = []
    best_step_count = 0
    best_run_traj_seed = None

    fitness_set_max = []
    fitness_set_min = []
    fitness_set_mean = []

    for iteration in range(iterations):
        print("=" * 90)
        print(f"Iteration {iteration + 1}/{iterations} started.")

        fitnesses = []
        b_fitness = 1.0
        generation_best_trajectory_actions = []
        generation_best_offload_actions = []
        generation_best_step_count = 0
        generation_best_traj_seed = None

        for i, chromo in enumerate(population):
            result = evaluate_chromosome(
                chromosome=chromo,
                env=env,
                agent=agent,
                offload_agent=offload_agent,
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
                generation_best_offload_actions = list(result["single_best_offload_actions"])
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
            f"max {fit_max:.6f} men {fit_mean:.6f}"
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
            best_offload_actions = list(generation_best_offload_actions)
            best_step_count = int(generation_best_step_count)
            best_run_traj_seed = generation_best_traj_seed
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
        "offload_actions": best_offload_actions,
        "fitness_set_max": fitness_set_max,
        "fitness_set_min": fitness_set_min,
        "fitness_set_mean": fitness_set_mean,
        "best_step_count": best_step_count,
        "best_run_traj_seed": best_run_traj_seed,
    }


def main():
    args = parse_args()
    validate_traj_range(args.traj_seed_start, args.traj_seed_end)
    if args.population_size < 2:
        raise ValueError("population-size must be >= 2")
    if args.iterations < 1:
        raise ValueError("iterations must be >= 1")
    if args.repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if not (0.0 <= args.mutation_rate <= 1.0):
        raise ValueError("mutation-rate must be in [0, 1]")

    set_seed(args.ga_seed)
    env, agent, offload_agent = build_agents_and_env(num_uav=args.num_uav)
    traj_seeds = list(range(args.traj_seed_start, args.traj_seed_end + 1))

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
        traj_seeds=traj_seeds,
        ga_seed=args.ga_seed,
        verbose_seed_progress=args.verbose_seed_progress,
    )
    total_elapsed = float(time.time() - t0)

    best_solution = ga_result["best_solution"]
    if best_solution is None:
        raise RuntimeError("GA did not produce a valid best solution.")
    starts, ends = split_chromosome(best_solution, args.num_uav)

    result_json = resolve_path_template(
        args.result_json,
        wind_seed=args.wind_seed,
        terrain_seed=args.terrain_seed,
        infra_seed=args.infra_seed,
        start=args.traj_seed_start,
        end=args.traj_seed_end,
    )
    result_pkl = resolve_path_template(
        args.result_pkl,
        wind_seed=args.wind_seed,
        terrain_seed=args.terrain_seed,
        infra_seed=args.infra_seed,
        start=args.traj_seed_start,
        end=args.traj_seed_end,
    )

    result_data = {
        "created_at": utc_now_iso(),
        "seeds": {
            "wind_seed": int(args.wind_seed),
            "terrain_seed": int(args.terrain_seed),
            "infra_seed": int(args.infra_seed),
            "traj_seed_start": int(args.traj_seed_start),
            "traj_seed_end": int(args.traj_seed_end),
        },
        "ga": {
            "iterations": int(args.iterations),
            "population_size": int(args.population_size),
            "repetitions": int(args.repetitions),
            "mutation_rate": float(args.mutation_rate),
            "ga_seed": int(args.ga_seed),
            "elapsed_sec": total_elapsed,
        },
        "grid": {
            "lx": int(env.Lx),
            "ly": int(env.Ly),
            "grid_cell_size_m": float(getattr(env, "grid_cell_size_m", env.X / env.Lx)),
        },
        "best": {
            "mean_average_uncertainty": float(ga_result["best_fitness"]),
            "best_solution": [[int(p[0]), int(p[1])] for p in best_solution],
            "start_positions": [[int(p[0]), int(p[1])] for p in starts],
            "end_positions": [[int(p[0]), int(p[1])] for p in ends],
            "deployment_positions": [[int(p[0]), int(p[1])] for p in starts],
            "deployment_destinations": [[int(p[0]), int(p[1])] for p in ends],
            "best_run_traj_seed": (
                None if ga_result["best_run_traj_seed"] is None else int(ga_result["best_run_traj_seed"])
            ),
            "best_step_count": int(ga_result["best_step_count"]),
            "trajectory_actions": ga_result["trajectory_actions"],
            "offload_actions": ga_result["offload_actions"],
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
                "offload_actions": ga_result["offload_actions"],
            },
            f,
        )

    print("-" * 90)
    print("GA deployment search completed.")
    print(f"Result JSON: {result_json}")
    print(f"Result PKL: {result_pkl}")
    print(f"Best solution: {best_solution}")
    print(f"Best mean average uncertainty: {ga_result['best_fitness']:.6f}")
    print(f"Best start positions: {starts}")
    print(f"Best end positions: {ends}")
    print(f"Best run traj_seed: {ga_result['best_run_traj_seed']}")
    print(f"Total elapsed: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
