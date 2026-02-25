import os
import sys
import json
import argparse
import random

# Headless default for batch execution
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from Our_experiment.HCSAC.ENV.UAVenv_SAC_Original import UAVEnv as UAVenv
from Our_experiment.HCSAC.ENV.UAVenv_SAC_Original import SAC
from Our_experiment.HCSAC.UAV_VIS_offloading_2 import visualize_trajectory as vis
from Our_experiment.HCSAC import UAV_SAVE

import torch
import numpy as np
import matplotlib.pyplot as plt
import pygame

# Skip waits for faster batch mode
pygame.time.wait = lambda ms: None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze UAV visit frequency for traj seeds 0-200 under a single wind field."
    )
    parser.add_argument(
        "--wind-seed",
        type=int,
        default=None,
        help="Directly set one wind seed. If omitted, representative seed will be taken from --wind-class in catalog.",
    )
    parser.add_argument(
        "--terrain-seed",
        type=int,
        default=None,
        help="Seed controlling terrain difficulty map generation. Default follows wind seed.",
    )
    parser.add_argument(
        "--infra-seed",
        type=int,
        default=None,
        help="Seed controlling random GBS/HAPS ground positions. Default follows wind seed.",
    )
    parser.add_argument(
        "--wind-class",
        type=str,
        default="Moderate Wind",
        choices=["Low Wind", "Moderate Wind", "Strong Wind"],
        help="Wind class used when --wind-seed is not provided.",
    )
    parser.add_argument(
        "--wind-catalog-json",
        type=str,
        default="Our_experiment/HCSAC/data/wind_seed_classes_5000.json",
        help="Wind seed catalog JSON path produced by wind_seed_catalog_builder.py",
    )
    parser.add_argument("--traj-seed-start", type=int, default=0, help="Trajectory seed start (inclusive).")
    parser.add_argument("--traj-seed-end", type=int, default=200, help="Trajectory seed end (inclusive).")
    parser.add_argument(
        "--heatmap-path",
        type=str,
        default="Our_experiment/HCSAC/data/visit_frequency_w{wind_seed}_g{terrain_seed}_t{start}_{end}_i{infra_seed}.png",
        help="Output heatmap path. Supports {wind_seed}/{terrain_seed}/{start}/{end}/{infra_seed} placeholders.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="Our_experiment/HCSAC/data/visit_frequency_w{wind_seed}_g{terrain_seed}_t{start}_{end}_i{infra_seed}.md",
        help="Output report path. Supports {wind_seed}/{terrain_seed}/{start}/{end}/{infra_seed} placeholders.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

    agent = UAV_SAVE.load_sac_agent(agent, path="../HCSAC/data/sac_model_fly", device=device)
    offload_agent = UAV_SAVE.load_sac_agent(offload_agent, path="../HCSAC/data/sac_model_offload", device=device)
    return env, agent, offload_agent


def load_wind_representatives(catalog_json_path):
    with open(catalog_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    reps = data.get("representative_seeds", {})
    for key in ["Low Wind", "Moderate Wind", "Strong Wind"]:
        if key not in reps:
            raise KeyError(f"Missing representative seed for class: {key}")
    return reps


def top_k_cells(freq_map, k=5):
    flat_idx = np.argsort(freq_map.ravel())[::-1][:k]
    w = freq_map.shape[1]
    result = []
    for idx in flat_idx:
        x = int(idx // w)
        y = int(idx % w)
        result.append((x, y, float(freq_map[x, y])))
    return result


def generate_commentary(
    freq_maps_by_uav,
    uncertainty_list,
    wind_seed,
    terrain_seed,
    infra_seed,
    gbs_position,
    haps_position,
    grid_cell_size_m,
    wind_class,
):
    lines = []
    lines.append("# Visit Frequency Commentary (Single Wind Field)")
    lines.append("")
    lines.append(f"- Wind seed: `{wind_seed}`")
    lines.append(f"- Terrain seed: `{terrain_seed}`")
    lines.append(f"- Infrastructure seed: `{infra_seed}`")
    lines.append(f"- Wind label: `{wind_class}`")
    lines.append(f"- Grid cell size: `{grid_cell_size_m:.0f} m`")
    lines.append(
        f"- GBS position (grid): `({gbs_position[0]:.2f}, {gbs_position[1]:.2f})`, "
        f"HAPS position (grid): `({haps_position[0]:.2f}, {haps_position[1]:.2f})`"
    )
    lines.append("")

    avg_unc = float(np.mean(uncertainty_list))

    lines.append("## Uncertainty Summary")
    lines.append("")
    lines.append(f"- Mean Average uncertainty = `{avg_unc:.6f}`")
    lines.append("")
    lines.append("## Hotspot Cells by UAV (Top-5 Frequency)")
    lines.append("")
    for uav_idx in range(freq_maps_by_uav.shape[0]):
        lines.append(f"### UAV {uav_idx}")
        for x, y, v in top_k_cells(freq_maps_by_uav[uav_idx], k=5):
            lines.append(f"- cell ({x}, {y}): frequency `{v:.6f}`")
        lines.append("")

    return "\n".join(lines) + "\n"


def save_visit_frequency_heatmap(
    freq_maps_by_uav,
    output_path,
    wind_seed,
    terrain_seed,
    infra_seed,
    gbs_position,
    haps_position,
    grid_cell_size_m,
    wind_class,
):
    n_uav = freq_maps_by_uav.shape[0]
    cols = 2 if n_uav > 1 else 1
    rows = int(np.ceil(n_uav / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 12))
    axes = np.array(axes).reshape(-1)

    # Match typography with vis_offloading_seed_executor.py
    title_fs = 24
    axis_label_fs = 20
    tick_fs = 16
    suptitle_fs = 24
    cbar_label_fs = 20
    cbar_tick_fs = 16

    vmin = float(np.min(freq_maps_by_uav))
    vmax = float(np.max(freq_maps_by_uav))
    mappable = None
    for uav_idx in range(n_uav):
        ax = axes[uav_idx]
        mappable = ax.imshow(
            freq_maps_by_uav[uav_idx].T,
            cmap="viridis",
            origin="lower",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"UAV {uav_idx}", fontsize=title_fs)
        ax.set_xlabel("Grid X", fontsize=axis_label_fs)
        ax.set_ylabel("Grid Y", fontsize=axis_label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.scatter(gbs_position[0], gbs_position[1], marker="X", s=90, c="cyan", edgecolors="black", linewidths=1.0, label="GBS")
        ax.scatter(haps_position[0], haps_position[1], marker="^", s=90, c="lime", edgecolors="black", linewidths=1.0, label="HAPS")
        if uav_idx == 0:
            ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
    for i in range(n_uav, len(axes)):
        axes[i].axis("off")

    # Use a dedicated right-side colorbar axis to avoid overlap with subplot area.
    cbar_ax = fig.add_axes([0.92, 0.13, 0.024, 0.74])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label("Visit Frequency", fontsize=cbar_label_fs, labelpad=14)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

    title_line_1 = "UAV Visit Frequency by UAV"
    wind_suffix = ""
    if wind_class and wind_class != "Custom Wind Seed":
        wind_suffix = f", {wind_class}"
    title_line_2 = (
        f"w={wind_seed}, g={terrain_seed}, i={infra_seed}, "
        f"cell={grid_cell_size_m:.0f}m{wind_suffix}"
    )
    fig.suptitle(
        f"{title_line_1}\n{title_line_2}",
        fontsize=suptitle_fs,
        x=0.52,  # Center over subplot+colorbar content block.
        y=0.985,
    )
    plt.tight_layout(rect=[0.0, 0.0, 0.90, 0.92])
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def main():
    args = parse_args()
    if args.traj_seed_end < args.traj_seed_start:
        raise ValueError("traj-seed-end must be >= traj-seed-start")

    reps = load_wind_representatives(args.wind_catalog_json)
    wind_seed = int(args.wind_seed) if args.wind_seed is not None else int(reps[args.wind_class])
    terrain_seed = int(args.terrain_seed) if args.terrain_seed is not None else wind_seed
    infra_seed = int(args.infra_seed) if args.infra_seed is not None else wind_seed
    wind_class = args.wind_class if args.wind_seed is None else f"Custom Wind Seed"

    env, agent, offload_agent = build_agents_and_env()
    env.reset(seed=0, wind_seed=wind_seed, terrain_seed=terrain_seed, infra_seed=infra_seed)
    gbs_position = np.array(env.gbs_position, dtype=np.float64).copy()
    haps_position = np.array(env.haps_position, dtype=np.float64).copy()
    grid_cell_size_m = float(getattr(env, "grid_cell_size_m", env.X / env.Lx))

    visit_sum = None
    visit_sum_by_uav = None
    uncertainty_list = []

    for traj_seed in range(args.traj_seed_start, args.traj_seed_end + 1):
        set_seed(traj_seed)
        stats = vis(
            agent,
            offload_agent,
            env,
            seed=traj_seed,
            return_stats=True,
            wind_seed=wind_seed,
            terrain_seed=terrain_seed,
            traj_seed=traj_seed,
            infra_seed=infra_seed,
        )
        visit_count = stats["visit_count"].astype(np.float64)
        visit_count_by_uav = stats["visit_count_by_uav"].astype(np.float64)
        if visit_sum is None:
            visit_sum = np.zeros_like(visit_count, dtype=np.float64)
        if visit_sum_by_uav is None:
            visit_sum_by_uav = np.zeros_like(visit_count_by_uav, dtype=np.float64)
        visit_sum += visit_count
        visit_sum_by_uav += visit_count_by_uav
        uncertainty_list.append(float(stats["avg_uncertainty"]))

    print(
        f"wind_seed={wind_seed}, wind_class={wind_class}, "
        f"traj_seed_range={args.traj_seed_start}-{args.traj_seed_end}, "
        f"runs={args.traj_seed_end - args.traj_seed_start + 1}"
    )

    freq_maps_by_uav = np.zeros_like(visit_sum_by_uav, dtype=np.float64)
    for uav_idx in range(visit_sum_by_uav.shape[0]):
        uav_total = float(np.sum(visit_sum_by_uav[uav_idx]))
        if uav_total > 0:
            freq_maps_by_uav[uav_idx] = visit_sum_by_uav[uav_idx] / uav_total
        else:
            freq_maps_by_uav[uav_idx] = visit_sum_by_uav[uav_idx]

    heatmap_path = args.heatmap_path.format(
        wind_seed=wind_seed,
        terrain_seed=terrain_seed,
        start=args.traj_seed_start,
        end=args.traj_seed_end,
        infra_seed=infra_seed,
    )
    report_path = args.report_path.format(
        wind_seed=wind_seed,
        terrain_seed=terrain_seed,
        start=args.traj_seed_start,
        end=args.traj_seed_end,
        infra_seed=infra_seed,
    )
    os.makedirs(os.path.dirname(heatmap_path), exist_ok=True)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)

    save_visit_frequency_heatmap(
        freq_maps_by_uav,
        heatmap_path,
        wind_seed,
        terrain_seed,
        infra_seed,
        gbs_position,
        haps_position,
        grid_cell_size_m,
        wind_class,
    )
    commentary = generate_commentary(
        freq_maps_by_uav,
        uncertainty_list,
        wind_seed,
        terrain_seed,
        infra_seed,
        gbs_position,
        haps_position,
        grid_cell_size_m,
        wind_class,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(commentary)

    print("-" * 60)
    print("Visit frequency analysis done.")
    print(f"Wind used: class={wind_class}, wind_seed={wind_seed}")
    print(f"Terrain seed: {terrain_seed}")
    print(f"Infrastructure seed: {infra_seed}")
    print(
        "GBS/HAPS positions (grid): "
        f"GBS=({gbs_position[0]:.2f}, {gbs_position[1]:.2f}), "
        f"HAPS=({haps_position[0]:.2f}, {haps_position[1]:.2f})"
    )
    print(f"Grid cell size: {grid_cell_size_m:.0f} m")
    print(f"Heatmap saved: {heatmap_path}")
    print(f"Commentary saved: {report_path}")


if __name__ == "__main__":
    main()
