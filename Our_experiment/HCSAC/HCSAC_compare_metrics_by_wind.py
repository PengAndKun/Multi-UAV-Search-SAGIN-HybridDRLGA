import argparse
import json
import os
import random
import sys

# Headless default for batch execution
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from Our_experiment.HCSAC.ENV.UAVenv_SAC_Original import UAVEnv as UAVenv
from Our_experiment.HCSAC.ENV.UAVenv_SAC_Original import SAC
from Our_experiment.HCSAC.UAV_VIS_offloading_2 import visualize_trajectory as vis
from Our_experiment.HCSAC import UAV_SAVE

import numpy as np
import pygame
import torch

# Skip waits for faster batch mode
pygame.time.wait = lambda ms: None


WIND_CLASS_MAP = {
    11: "Low Wind",
    23: "Moderate Wind",
    4800: "Strong Wind",
}

WIND_ORDER = [11, 23, 4800]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recompute HCSAC uncertainty/lifetime metrics across multiple wind seeds, "
            "save a single JSON summary, and print a terminal table."
        )
    )
    parser.add_argument("--infra-seed", type=int, default=999999, help="Infrastructure seed.")
    parser.add_argument("--terrain-seed", type=int, default=10, help="Terrain seed.")
    parser.add_argument(
        "--wind-seeds",
        type=str,
        default="11,23,4800",
        help="Comma-separated wind seeds. Default corresponds to Low/Moderate/Strong wind.",
    )
    parser.add_argument(
        "--traj-seed-mode",
        type=str,
        default="random",
        choices=["random", "range"],
        help="Trajectory seed mode: random sample (default) or contiguous range.",
    )
    parser.add_argument(
        "--traj-seed-list",
        type=str,
        default=None,
        help="Manual seed list, comma-separated. If set, overrides mode.",
    )
    parser.add_argument("--traj-seed-start", type=int, default=0, help="Trajectory seed start for range mode.")
    parser.add_argument("--traj-seed-end", type=int, default=200, help="Trajectory seed end for range mode.")
    parser.add_argument("--traj-seed-pool-min", type=int, default=0, help="Trajectory seed pool min for random mode.")
    parser.add_argument("--traj-seed-pool-max", type=int, default=200, help="Trajectory seed pool max for random mode.")
    parser.add_argument("--traj-seed-sample-size", type=int, default=10, help="Random mode: sampled trajectory seed count.")
    parser.add_argument("--traj-seed-sampler-seed", type=int, default=2026, help="Random mode: RNG seed.")
    parser.add_argument(
        "--sample-with-replacement",
        action="store_true",
        help="Random mode: sample trajectory seeds with replacement.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=(
            "Our_experiment/HCSAC/data/"
            "hcsac_metrics_by_wind_g{terrain_seed}_i{infra_seed}_{seed_tag}_winds{wind_tag}.json"
        ),
        help="Output JSON path.",
    )
    parser.add_argument(
        "--table-format",
        type=str,
        default="plain",
        choices=["plain", "markdown"],
        help="Terminal table format.",
    )
    parser.add_argument(
        "--lifetime-unit",
        type=str,
        default="min",
        choices=["min", "s", "steps"],
        help="Lifetime unit shown in terminal table.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent_dir(path):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_json(path, data):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_seed_list(seed_list_text):
    if seed_list_text is None:
        return None
    parts = [p.strip() for p in seed_list_text.split(",") if p.strip()]
    if not parts:
        raise ValueError("traj-seed-list is empty.")
    return [int(p) for p in parts]


def parse_wind_seeds(text):
    parts = [p.strip() for p in str(text).split(",") if p.strip()]
    if not parts:
        raise ValueError("wind-seeds is empty.")
    return [int(p) for p in parts]


def summarize_seed_list(traj_seeds, limit=25):
    if len(traj_seeds) <= limit:
        return str(traj_seeds)
    head = traj_seeds[:limit]
    return f"{head} ... (total {len(traj_seeds)})"


def select_traj_seeds(args):
    manual_list = parse_seed_list(args.traj_seed_list)
    if manual_list is not None:
        return manual_list, "manual-list", f"list{len(manual_list)}"

    if args.traj_seed_mode == "range":
        if args.traj_seed_end < args.traj_seed_start:
            raise ValueError("traj-seed-end must be >= traj-seed-start")
        traj_seeds = list(range(args.traj_seed_start, args.traj_seed_end + 1))
        return traj_seeds, f"range({args.traj_seed_start}-{args.traj_seed_end})", f"range{args.traj_seed_start}_{args.traj_seed_end}"

    if args.traj_seed_pool_max < args.traj_seed_pool_min:
        raise ValueError("traj-seed-pool-max must be >= traj-seed-pool-min")
    if args.traj_seed_sample_size < 1:
        raise ValueError("traj-seed-sample-size must be >= 1")

    pool = np.arange(args.traj_seed_pool_min, args.traj_seed_pool_max + 1, dtype=np.int64)
    if len(pool) == 0:
        raise ValueError("trajectory seed pool is empty.")

    rng = np.random.default_rng(args.traj_seed_sampler_seed)
    replace = bool(args.sample_with_replacement) or args.traj_seed_sample_size > len(pool)
    sampled = rng.choice(pool, size=args.traj_seed_sample_size, replace=replace)
    traj_seeds = [int(s) for s in sampled.tolist()]
    return traj_seeds, (
        f"random(pool={args.traj_seed_pool_min}-{args.traj_seed_pool_max}, "
        f"n={args.traj_seed_sample_size}, replace={replace}, seed={args.traj_seed_sampler_seed})"
    ), f"random{args.traj_seed_sample_size}_s{args.traj_seed_sampler_seed}"


def wind_class_name(wind_seed):
    return WIND_CLASS_MAP.get(int(wind_seed), f"Wind Seed {int(wind_seed)}")


def build_agents_and_env():
    env = UAVenv(4)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    hidden_dim = 128
    gamma = 0.99
    tau = 0.005
    actor_lr = 3e-4
    critic_lr = 3e-4
    alpha_lr = 1e-4

    state_dim = env.state_dim
    action_dim = env.action_dim
    offload_state_dim = env.offload_state_dim
    offload_action_dim = env.offload_action_dim
    target_entropy = -np.log(action_dim)
    target_entropy_offload = -np.log(offload_action_dim)

    agent = SAC(
        state_dim,
        hidden_dim,
        action_dim,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        tau,
        gamma,
        device,
    )
    offload_agent = SAC(
        offload_state_dim,
        hidden_dim,
        offload_action_dim,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy_offload,
        tau,
        gamma,
        device,
        type="GCN",
    )

    fly_model = os.path.join(PROJECT_ROOT, "Our_experiment", "HCSAC", "data", "sac_model_fly")
    offload_model = os.path.join(PROJECT_ROOT, "Our_experiment", "HCSAC", "data", "sac_model_offload")
    agent = UAV_SAVE.load_sac_agent(agent, path=fly_model, device=device)
    offload_agent = UAV_SAVE.load_sac_agent(offload_agent, path=offload_model, device=device)
    return env, agent, offload_agent


def compute_metrics_for_wind(env, agent, offload_agent, wind_seed, terrain_seed, infra_seed, traj_seeds):
    uncertainty_list = []
    lifetime_steps_list = []
    lifetime_seconds_list = []

    for traj_seed in traj_seeds:
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
        uncertainty_list.append(float(stats["avg_uncertainty"]))
        lifetime_steps_list.append(float(stats.get("avg_uav_lifetime_steps", np.nan)))
        lifetime_seconds_list.append(float(stats.get("avg_uav_lifetime_seconds", np.nan)))

    if len(uncertainty_list) == 0:
        raise RuntimeError(f"No rollout statistics produced for wind_seed={wind_seed}")

    return {
        "wind_seed": int(wind_seed),
        "wind_class": wind_class_name(wind_seed),
        "num_traj_seeds": int(len(traj_seeds)),
        "mean_average_uncertainty": float(np.mean(uncertainty_list)),
        "std_average_uncertainty": float(np.std(uncertainty_list)),
        "mean_uav_lifetime_steps": float(np.nanmean(lifetime_steps_list)),
        "std_uav_lifetime_steps": float(np.nanstd(lifetime_steps_list)),
        "mean_uav_lifetime_seconds": float(np.nanmean(lifetime_seconds_list)),
        "std_uav_lifetime_seconds": float(np.nanstd(lifetime_seconds_list)),
    }


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


def format_uncertainty(metrics):
    return format_pm(metrics["mean_average_uncertainty"], metrics["std_average_uncertainty"], digits=4)


def lifetime_label(unit):
    if unit == "steps":
        return "Lifetime (steps)"
    if unit == "s":
        return "Lifetime (s)"
    return "Lifetime (min)"


def build_headers(lifetime_unit, wind_seeds):
    headers = ["Variant"]
    for wind_seed in wind_seeds:
        headers.append(f"{wind_class_name(wind_seed)} {lifetime_label(lifetime_unit)}")
        headers.append(f"{wind_class_name(wind_seed)} Avg Unc")
    return headers


def build_rows(metrics_by_wind, lifetime_unit, wind_seeds):
    row = ["HCSAC"]
    for wind_seed in wind_seeds:
        metrics = metrics_by_wind[str(int(wind_seed))]
        row.append(format_lifetime(metrics, lifetime_unit))
        row.append(format_uncertainty(metrics))
    return [row]


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


def print_table(metrics_by_wind, lifetime_unit, table_format, wind_seeds):
    headers = build_headers(lifetime_unit, wind_seeds)
    rows = build_rows(metrics_by_wind, lifetime_unit, wind_seeds)
    print("")
    if table_format == "markdown":
        print(markdown_table(headers, rows))
    else:
        print(plain_table(headers, rows))


def main():
    args = parse_args()
    wind_seeds = parse_wind_seeds(args.wind_seeds)
    traj_seeds, traj_seed_source_desc, seed_tag = select_traj_seeds(args)
    if len(traj_seeds) == 0:
        raise RuntimeError("No trajectory seeds selected.")

    env, agent, offload_agent = build_agents_and_env()
    metrics_by_wind = {}

    for wind_seed in wind_seeds:
        print(f"[HCSAC] wind_seed={int(wind_seed)} ({wind_class_name(wind_seed)}) started.")
        metrics = compute_metrics_for_wind(
            env=env,
            agent=agent,
            offload_agent=offload_agent,
            wind_seed=int(wind_seed),
            terrain_seed=int(args.terrain_seed),
            infra_seed=int(args.infra_seed),
            traj_seeds=traj_seeds,
        )
        metrics_by_wind[str(int(wind_seed))] = metrics
        print(
            f"[HCSAC] wind_seed={int(wind_seed)} done. "
            f"Mean uncertainty={metrics['mean_average_uncertainty']:.6f}, "
            f"Mean lifetime={metrics['mean_uav_lifetime_steps']:.2f} steps."
        )

    wind_tag = "_".join(str(int(w)) for w in wind_seeds)
    output_json = args.output_json.format(
        terrain_seed=int(args.terrain_seed),
        infra_seed=int(args.infra_seed),
        wind_tag=wind_tag,
        seed_tag=seed_tag,
    )

    result = {
        "created_at": __import__("datetime").datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "algorithm_id": "HCSAC",
        "algorithm_label": "HCSAC",
        "algorithm_file": "Our_experiment/HCSAC/HCSAC_vis_offloading_seed_executor_range.py",
        "config": {
            "infra_seed": int(args.infra_seed),
            "terrain_seed": int(args.terrain_seed),
            "wind_seeds": [int(w) for w in wind_seeds],
            "wind_classes": {str(int(w)): wind_class_name(w) for w in wind_seeds},
            "traj_seed_source": traj_seed_source_desc,
            "traj_seeds": [int(s) for s in traj_seeds],
            "traj_seed_count": int(len(traj_seeds)),
            "seed_tag": seed_tag,
        },
        "comparison_by_wind": metrics_by_wind,
    }
    save_json(output_json, result)

    print("-" * 100)
    print(f"Output JSON: {output_json}")
    print(f"Trajectory seed source: {traj_seed_source_desc}")
    print(f"Trajectory seeds used ({len(traj_seeds)}): {summarize_seed_list(traj_seeds)}")
    print_table(metrics_by_wind, args.lifetime_unit, args.table_format, wind_seeds)


if __name__ == "__main__":
    main()
