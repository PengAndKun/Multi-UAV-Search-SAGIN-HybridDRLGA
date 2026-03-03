import argparse
import os
import sys
import time

import numpy as np

# Headless defaults for environments that run without display/audio
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Our_experiment.GA import ga_deployment_seed_search_2 as algo_rl_offload
from Our_experiment.GA import ga_deployment_seed_search_2_no_offloading as algo_no_offload
from Our_experiment.GA import ga_deployment_seed_search_2_rule_based_offloading_2 as algo_simple_greedy_offload
from Our_experiment.GA.ga_vis_common import build_agents_and_env
from Our_experiment.GA.ga_vis_common import resolve_path_template
from Our_experiment.GA.ga_vis_common import save_json
from Our_experiment.GA.ga_vis_common import set_seed
from Our_experiment.GA.ga_vis_common import utc_now_iso
from Our_experiment.GA.ga_vis_common import validate_traj_range
from Our_experiment.HCSAC.UAV_VIS_offloading_2 import visualize_trajectory as vis


WIND_CLASS_MAP = {
    11: "Low Wind",
    23: "Moderate Wind",
    4800: "Strong Wind",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run and compare GA-based algorithms plus a No-GA baseline across wind classes. "
            "The script reruns all algorithms, recomputes lifetime/uncertainty/coverage statistics, "
            "and writes a single comparison JSON."
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
    parser.add_argument("--traj-seed-min", type=int, default=0, help="Trajectory seed pool minimum (inclusive).")
    parser.add_argument("--traj-seed-max", type=int, default=200, help="Trajectory seed pool maximum (inclusive).")
    parser.add_argument(
        "--traj-seed-sample-size",
        type=int,
        default=10,
        help="Number of trajectory seeds randomly sampled per GA iteration.",
    )
    parser.add_argument(
        "--sample-with-replacement",
        action="store_true",
        help="Sample trajectory seeds with replacement during GA search.",
    )
    parser.add_argument("--iterations", type=int, default=20, help="GA generation count for all algorithms.")
    parser.add_argument("--population-size", type=int, default=12, help="GA population size for all algorithms.")
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Rollout repetitions per sampled trajectory seed during GA search.",
    )
    parser.add_argument("--mutation-rate", type=float, default=0.1, help="GA mutation rate.")
    parser.add_argument("--ga-seed", type=int, default=2026, help="Shared GA random seed.")
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
        "--output-json",
        type=str,
        default=(
            "Our_experiment/GA/data/"
            "ga_algorithm_comparison_i{infra_seed}_g{terrain_seed}_winds{wind_tag}.json"
        ),
        help="Output comparison JSON path.",
    )
    parser.add_argument(
        "--verbose-seed-progress",
        action="store_true",
        help="Print per-seed progress from underlying GA search loops.",
    )
    return parser.parse_args()


def parse_wind_seeds(text):
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if len(parts) == 0:
        raise ValueError("wind-seeds is empty.")
    return [int(p) for p in parts]


def wind_class_name(wind_seed):
    return WIND_CLASS_MAP.get(int(wind_seed), f"Wind Seed {int(wind_seed)}")


def build_common_config(args):
    validate_traj_range(args.traj_seed_min, args.traj_seed_max)
    validate_traj_range(args.eval_traj_seed_start, args.eval_traj_seed_end)
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

    return {
        "num_uav": int(args.num_uav),
        "infra_seed": int(args.infra_seed),
        "terrain_seed": int(args.terrain_seed),
        "traj_seed_min": int(args.traj_seed_min),
        "traj_seed_max": int(args.traj_seed_max),
        "traj_seed_sample_size": int(args.traj_seed_sample_size),
        "sample_with_replacement": bool(args.sample_with_replacement),
        "iterations": int(args.iterations),
        "population_size": int(args.population_size),
        "repetitions": int(args.repetitions),
        "mutation_rate": float(args.mutation_rate),
        "ga_seed": int(args.ga_seed),
        "eval_traj_seed_start": int(args.eval_traj_seed_start),
        "eval_traj_seed_end": int(args.eval_traj_seed_end),
        "eval_repetitions": int(args.eval_repetitions),
        "verbose_seed_progress": bool(args.verbose_seed_progress),
    }


def make_traj_seed_pool(cfg):
    return np.arange(cfg["traj_seed_min"], cfg["traj_seed_max"] + 1, dtype=np.int64)


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


def evaluate_full_rl_offloading_solution(
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
    starts, ends = algo_rl_offload.split_chromosome(best_solution, num_uav)
    uncertainty_values = []
    lifetime_steps_values = []
    lifetime_seconds_values = []

    for seed_idx, traj_seed in enumerate(eval_traj_seeds):
        for rep in range(eval_repetitions):
            rollout_seed = algo_rl_offload.compose_rollout_seed(
                ga_seed=ga_seed + 7919,
                iteration=99991,
                population_index=0,
                seed_index=seed_idx,
                repetition=rep,
            )
            rollout_result = rollout_full_rl_with_lifetime(
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
            uncertainty_values.append(float(rollout_result["avg_uncertainty"]))
            lifetime_steps_values.append(float(rollout_result["avg_uav_lifetime_steps"]))
            lifetime_seconds_values.append(float(rollout_result["avg_uav_lifetime_seconds"]))

    if len(uncertainty_values) == 0:
        raise RuntimeError("No evaluation rollouts were executed for the RL + offloading solution.")

    return to_coverage_metrics({
        "mean_average_uncertainty": float(np.mean(uncertainty_values)),
        "std_average_uncertainty": float(np.std(uncertainty_values)),
        "mean_uav_lifetime_steps": float(np.mean(lifetime_steps_values)),
        "std_uav_lifetime_steps": float(np.std(lifetime_steps_values)),
        "mean_uav_lifetime_seconds": float(np.mean(lifetime_seconds_values)),
        "std_uav_lifetime_seconds": float(np.std(lifetime_seconds_values)),
        "num_traj_seeds": int(len(eval_traj_seeds)),
        "eval_repetitions": int(eval_repetitions),
        "num_eval_rollouts": int(len(uncertainty_values)),
    })


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
        raise RuntimeError("No evaluation rollouts were executed for the No-GA baseline.")

    return to_coverage_metrics({
        "mean_average_uncertainty": float(np.mean(uncertainty_values)),
        "std_average_uncertainty": float(np.std(uncertainty_values)),
        "mean_uav_lifetime_steps": float(np.nanmean(lifetime_steps_values)),
        "std_uav_lifetime_steps": float(np.nanstd(lifetime_steps_values)),
        "mean_uav_lifetime_seconds": float(np.nanmean(lifetime_seconds_values)),
        "std_uav_lifetime_seconds": float(np.nanstd(lifetime_seconds_values)),
        "num_traj_seeds": int(len(eval_traj_seeds)),
        "eval_repetitions": int(eval_repetitions),
        "num_eval_rollouts": int(len(uncertainty_values)),
    })


def run_algorithm_rl_offloading(cfg, wind_seed):
    algorithm_id = "ga_deployment_seed_search_2"
    algorithm_label = "RL Trajectory + Offloading"
    print("=" * 100)
    print(f"[{algorithm_label}] wind_seed={wind_seed} ({wind_class_name(wind_seed)}) search started.")

    set_seed(cfg["ga_seed"])
    seed_sampler_rng = np.random.default_rng(cfg["ga_seed"])
    env, agent, offload_agent = build_agents_and_env(num_uav=cfg["num_uav"])
    traj_seed_pool = make_traj_seed_pool(cfg)

    t0 = time.time()
    ga_result = algo_rl_offload.genetic_algorithm(
        iterations=cfg["iterations"],
        population_size=cfg["population_size"],
        repetitions=cfg["repetitions"],
        mutation_rate=cfg["mutation_rate"],
        env=env,
        agent=agent,
        offload_agent=offload_agent,
        num_uav=cfg["num_uav"],
        env_lx=env.Lx,
        env_ly=env.Ly,
        wind_seed=int(wind_seed),
        terrain_seed=cfg["terrain_seed"],
        infra_seed=cfg["infra_seed"],
        traj_seed_pool=traj_seed_pool,
        traj_seed_sample_size=cfg["traj_seed_sample_size"],
        sample_with_replacement=cfg["sample_with_replacement"],
        seed_sampler_rng=seed_sampler_rng,
        ga_seed=cfg["ga_seed"],
        verbose_seed_progress=cfg["verbose_seed_progress"],
    )
    search_elapsed = float(time.time() - t0)

    best_solution = ga_result["best_solution"]
    if best_solution is None:
        raise RuntimeError(f"{algorithm_id} did not produce a valid best solution.")

    eval_metrics = evaluate_full_rl_offloading_solution(
        env=env,
        agent=agent,
        offload_agent=offload_agent,
        num_uav=cfg["num_uav"],
        best_solution=best_solution,
        wind_seed=int(wind_seed),
        terrain_seed=cfg["terrain_seed"],
        infra_seed=cfg["infra_seed"],
        eval_traj_seeds=make_eval_traj_seeds(cfg),
        eval_repetitions=cfg["eval_repetitions"],
        ga_seed=cfg["ga_seed"],
    )
    starts, ends = algo_rl_offload.split_chromosome(best_solution, cfg["num_uav"])

    print(
        f"[{algorithm_label}] wind_seed={wind_seed} done. "
        f"Mean uncertainty={eval_metrics['mean_average_uncertainty']:.6f}, "
        f"Coverage={eval_metrics['mean_coverage_percent']:.2f}%, "
        f"Mean lifetime={eval_metrics['mean_uav_lifetime_steps']:.2f} steps."
    )

    return {
        "algorithm_id": algorithm_id,
        "algorithm_label": algorithm_label,
        "algorithm_file": "Our_experiment/GA/ga_deployment_seed_search_2.py",
        "wind_seed": int(wind_seed),
        "wind_class": wind_class_name(wind_seed),
        "search_elapsed_sec": search_elapsed,
        "search": {
            "ga_objective_mean_average_uncertainty": float(ga_result["best_fitness"]),
            "best_run_traj_seed": (
                None if ga_result["best_run_traj_seed"] is None else int(ga_result["best_run_traj_seed"])
            ),
            "best_step_count": int(ga_result["best_step_count"]),
            "best_iteration_seed_set": [int(s) for s in ga_result["best_iteration_seed_set"]],
            "iteration_sampled_traj_seeds": ga_result["iteration_sampled_traj_seeds"],
        },
        "best": {
            "best_solution": [[int(p[0]), int(p[1])] for p in best_solution],
            "start_positions": [[int(p[0]), int(p[1])] for p in starts],
            "end_positions": [[int(p[0]), int(p[1])] for p in ends],
            "trajectory_actions": ga_result["trajectory_actions"],
            "offload_actions": ga_result["offload_actions"],
        },
        "metrics": eval_metrics,
    }


def run_algorithm_no_offloading(cfg, wind_seed):
    algorithm_id = "ga_deployment_seed_search_2_no_offloading"
    algorithm_label = "RL Trajectory + Local Processing Only"
    print("=" * 100)
    print(f"[{algorithm_label}] wind_seed={wind_seed} ({wind_class_name(wind_seed)}) search started.")

    set_seed(cfg["ga_seed"])
    seed_sampler_rng = np.random.default_rng(cfg["ga_seed"])
    env, agent, _ = build_agents_and_env(num_uav=cfg["num_uav"])
    traj_seed_pool = make_traj_seed_pool(cfg)

    t0 = time.time()
    ga_result = algo_no_offload.genetic_algorithm(
        iterations=cfg["iterations"],
        population_size=cfg["population_size"],
        repetitions=cfg["repetitions"],
        mutation_rate=cfg["mutation_rate"],
        env=env,
        agent=agent,
        num_uav=cfg["num_uav"],
        env_lx=env.Lx,
        env_ly=env.Ly,
        wind_seed=int(wind_seed),
        terrain_seed=cfg["terrain_seed"],
        infra_seed=cfg["infra_seed"],
        traj_seed_pool=traj_seed_pool,
        traj_seed_sample_size=cfg["traj_seed_sample_size"],
        sample_with_replacement=cfg["sample_with_replacement"],
        seed_sampler_rng=seed_sampler_rng,
        ga_seed=cfg["ga_seed"],
        verbose_seed_progress=cfg["verbose_seed_progress"],
    )
    search_elapsed = float(time.time() - t0)

    best_solution = ga_result["best_solution"]
    if best_solution is None:
        raise RuntimeError(f"{algorithm_id} did not produce a valid best solution.")

    eval_metrics = to_coverage_metrics(algo_no_offload.evaluate_solution_metrics(
        env=env,
        agent=agent,
        num_uav=cfg["num_uav"],
        best_solution=best_solution,
        wind_seed=int(wind_seed),
        terrain_seed=cfg["terrain_seed"],
        infra_seed=cfg["infra_seed"],
        eval_traj_seeds=make_eval_traj_seeds(cfg),
        eval_repetitions=cfg["eval_repetitions"],
        ga_seed=cfg["ga_seed"],
    ))
    starts, ends = algo_no_offload.split_chromosome(best_solution, cfg["num_uav"])

    print(
        f"[{algorithm_label}] wind_seed={wind_seed} done. "
        f"Mean uncertainty={eval_metrics['mean_average_uncertainty']:.6f}, "
        f"Coverage={eval_metrics['mean_coverage_percent']:.2f}%, "
        f"Mean lifetime={eval_metrics['mean_uav_lifetime_steps']:.2f} steps."
    )

    return {
        "algorithm_id": algorithm_id,
        "algorithm_label": algorithm_label,
        "algorithm_file": "Our_experiment/GA/ga_deployment_seed_search_2_no_offloading.py",
        "wind_seed": int(wind_seed),
        "wind_class": wind_class_name(wind_seed),
        "search_elapsed_sec": search_elapsed,
        "search": {
            "ga_objective_mean_average_uncertainty": float(ga_result["best_fitness"]),
            "best_run_traj_seed": (
                None if ga_result["best_run_traj_seed"] is None else int(ga_result["best_run_traj_seed"])
            ),
            "best_step_count": int(ga_result["best_step_count"]),
            "best_iteration_seed_set": [int(s) for s in ga_result["best_iteration_seed_set"]],
            "iteration_sampled_traj_seeds": ga_result["iteration_sampled_traj_seeds"],
        },
        "best": {
            "best_solution": [[int(p[0]), int(p[1])] for p in best_solution],
            "start_positions": [[int(p[0]), int(p[1])] for p in starts],
            "end_positions": [[int(p[0]), int(p[1])] for p in ends],
            "trajectory_actions": ga_result["trajectory_actions"],
        },
        "metrics": eval_metrics,
    }


def run_algorithm_simple_greedy_offloading(cfg, wind_seed):
    algorithm_id = "ga_deployment_seed_search_2_rule_based_offloading_2"
    algorithm_label = "Simple Greedy Trajectory + Offloading"
    print("=" * 100)
    print(f"[{algorithm_label}] wind_seed={wind_seed} ({wind_class_name(wind_seed)}) search started.")

    set_seed(cfg["ga_seed"])
    seed_sampler_rng = np.random.default_rng(cfg["ga_seed"])
    env, _, offload_agent = build_agents_and_env(num_uav=cfg["num_uav"])
    traj_seed_pool = make_traj_seed_pool(cfg)

    t0 = time.time()
    ga_result = algo_simple_greedy_offload.genetic_algorithm(
        iterations=cfg["iterations"],
        population_size=cfg["population_size"],
        repetitions=cfg["repetitions"],
        mutation_rate=cfg["mutation_rate"],
        env=env,
        offload_agent=offload_agent,
        num_uav=cfg["num_uav"],
        env_lx=env.Lx,
        env_ly=env.Ly,
        wind_seed=int(wind_seed),
        terrain_seed=cfg["terrain_seed"],
        infra_seed=cfg["infra_seed"],
        traj_seed_pool=traj_seed_pool,
        traj_seed_sample_size=cfg["traj_seed_sample_size"],
        sample_with_replacement=cfg["sample_with_replacement"],
        seed_sampler_rng=seed_sampler_rng,
        ga_seed=cfg["ga_seed"],
        verbose_seed_progress=cfg["verbose_seed_progress"],
    )
    search_elapsed = float(time.time() - t0)

    best_solution = ga_result["best_solution"]
    if best_solution is None:
        raise RuntimeError(f"{algorithm_id} did not produce a valid best solution.")

    eval_metrics = to_coverage_metrics(algo_simple_greedy_offload.evaluate_solution_metrics(
        env=env,
        offload_agent=offload_agent,
        num_uav=cfg["num_uav"],
        best_solution=best_solution,
        wind_seed=int(wind_seed),
        terrain_seed=cfg["terrain_seed"],
        infra_seed=cfg["infra_seed"],
        eval_traj_seeds=make_eval_traj_seeds(cfg),
        eval_repetitions=cfg["eval_repetitions"],
        ga_seed=cfg["ga_seed"],
    ))
    starts, ends = algo_simple_greedy_offload.split_chromosome(best_solution, cfg["num_uav"])

    print(
        f"[{algorithm_label}] wind_seed={wind_seed} done. "
        f"Mean uncertainty={eval_metrics['mean_average_uncertainty']:.6f}, "
        f"Coverage={eval_metrics['mean_coverage_percent']:.2f}%, "
        f"Mean lifetime={eval_metrics['mean_uav_lifetime_steps']:.2f} steps."
    )

    return {
        "algorithm_id": algorithm_id,
        "algorithm_label": algorithm_label,
        "algorithm_file": "Our_experiment/GA/ga_deployment_seed_search_2_rule_based_offloading_2.py",
        "wind_seed": int(wind_seed),
        "wind_class": wind_class_name(wind_seed),
        "search_elapsed_sec": search_elapsed,
        "search": {
            "ga_objective_mean_average_uncertainty": float(ga_result["best_fitness"]),
            "best_run_traj_seed": (
                None if ga_result["best_run_traj_seed"] is None else int(ga_result["best_run_traj_seed"])
            ),
            "best_step_count": int(ga_result["best_step_count"]),
            "best_iteration_seed_set": [int(s) for s in ga_result["best_iteration_seed_set"]],
            "iteration_sampled_traj_seeds": ga_result["iteration_sampled_traj_seeds"],
        },
        "best": {
            "best_solution": [[int(p[0]), int(p[1])] for p in best_solution],
            "start_positions": [[int(p[0]), int(p[1])] for p in starts],
            "end_positions": [[int(p[0]), int(p[1])] for p in ends],
            "trajectory_actions": ga_result["trajectory_actions"],
            "offload_actions": ga_result["offload_actions"],
        },
        "metrics": eval_metrics,
    }


def run_algorithm_no_ga_offloading(cfg, wind_seed):
    algorithm_id = "no_ga_drl_only_offloading"
    algorithm_label = "No-GA (DRL-only)"
    print("=" * 100)
    print(f"[{algorithm_label}] wind_seed={wind_seed} ({wind_class_name(wind_seed)}) evaluation started.")

    set_seed(cfg["ga_seed"])
    env, agent, offload_agent = build_agents_and_env(num_uav=cfg["num_uav"])

    t0 = time.time()
    eval_metrics = evaluate_no_ga_offloading(
        env=env,
        agent=agent,
        offload_agent=offload_agent,
        wind_seed=int(wind_seed),
        terrain_seed=cfg["terrain_seed"],
        infra_seed=cfg["infra_seed"],
        eval_traj_seeds=make_eval_traj_seeds(cfg),
        eval_repetitions=cfg["eval_repetitions"],
    )
    eval_elapsed = float(time.time() - t0)

    print(
        f"[{algorithm_label}] wind_seed={wind_seed} done. "
        f"Mean uncertainty={eval_metrics['mean_average_uncertainty']:.6f}, "
        f"Coverage={eval_metrics['mean_coverage_percent']:.2f}%, "
        f"Mean lifetime={eval_metrics['mean_uav_lifetime_steps']:.2f} steps."
    )

    return {
        "algorithm_id": algorithm_id,
        "algorithm_label": algorithm_label,
        "algorithm_file": "Our_experiment/HCSAC/HCSAC_vis_offloading_seed_executor_range.py",
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
            "start_positions": None,
            "end_positions": None,
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
    cfg = build_common_config(args)
    wind_seeds = parse_wind_seeds(args.wind_seeds)
    wind_tag = "_".join(str(w) for w in wind_seeds)

    all_runs = []
    total_start = time.time()

    for wind_seed in wind_seeds:
        all_runs.append(run_algorithm_rl_offloading(cfg, wind_seed))
        all_runs.append(run_algorithm_no_offloading(cfg, wind_seed))
        all_runs.append(run_algorithm_simple_greedy_offloading(cfg, wind_seed))
        all_runs.append(run_algorithm_no_ga_offloading(cfg, wind_seed))

    by_wind, by_algorithm = build_grouped_summary(all_runs)
    total_elapsed = float(time.time() - total_start)

    output_json = resolve_path_template(
        args.output_json,
        infra_seed=cfg["infra_seed"],
        terrain_seed=cfg["terrain_seed"],
        wind_tag=wind_tag,
    )

    result = {
        "created_at": utc_now_iso(),
        "summary_type": "from_scratch_ga_algorithm_comparison_by_wind",
        "config": {
            "num_uav": cfg["num_uav"],
            "infra_seed": cfg["infra_seed"],
            "terrain_seed": cfg["terrain_seed"],
            "wind_seeds": [int(w) for w in wind_seeds],
            "wind_classes": {str(w): wind_class_name(w) for w in wind_seeds},
            "traj_seed_min": cfg["traj_seed_min"],
            "traj_seed_max": cfg["traj_seed_max"],
            "traj_seed_sample_size": cfg["traj_seed_sample_size"],
            "sample_with_replacement": cfg["sample_with_replacement"],
            "iterations": cfg["iterations"],
            "population_size": cfg["population_size"],
            "repetitions": cfg["repetitions"],
            "mutation_rate": cfg["mutation_rate"],
            "ga_seed": cfg["ga_seed"],
            "eval_traj_seed_start": cfg["eval_traj_seed_start"],
            "eval_traj_seed_end": cfg["eval_traj_seed_end"],
            "eval_repetitions": cfg["eval_repetitions"],
            "coverage_definition": "coverage = 1 - uncertainty",
            "num_total_runs": int(len(all_runs)),
            "total_elapsed_sec": total_elapsed,
        },
        "runs": all_runs,
        "comparison_by_wind": by_wind,
        "comparison_by_algorithm": by_algorithm,
    }
    save_json(output_json, result)

    print("-" * 100)
    print("GA algorithm comparison by wind completed.")
    print(f"Output JSON: {output_json}")
    print(f"Wind seeds: {wind_seeds}")
    print(f"Total runs: {len(all_runs)}")
    print(f"Total elapsed: {total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
