import argparse
import json
import os
import random
import sys

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
        description=(
            "Aggregate offloading heatmaps under one wind field. "
            "Supports random trajectory seed sampling (GA-style)."
        )
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
        help="Manual seed list, comma-separated (e.g. 1,5,9). If set, overrides mode.",
    )
    parser.add_argument("--traj-seed-start", type=int, default=0, help="Trajectory seed start (inclusive) for range mode.")
    parser.add_argument("--traj-seed-end", type=int, default=200, help="Trajectory seed end (inclusive) for range mode.")
    parser.add_argument("--traj-seed-pool-min", type=int, default=0, help="Trajectory seed pool min (inclusive) for random mode.")
    parser.add_argument("--traj-seed-pool-max", type=int, default=200, help="Trajectory seed pool max (inclusive) for random mode.")
    parser.add_argument("--traj-seed-sample-size", type=int, default=10, help="Random mode: number of sampled trajectory seeds.")
    parser.add_argument("--traj-seed-sampler-seed", type=int, default=2026, help="Random mode: RNG seed for trajectory sampling.")
    parser.add_argument(
        "--sample-with-replacement",
        action="store_true",
        help="Random mode: sample trajectory seeds with replacement.",
    )
    parser.add_argument(
        "--heatmap-path",
        type=str,
        default="Our_experiment/HCSAC/data/offloading_frequency_w{wind_seed}_g{terrain_seed}_{seed_tag}_i{infra_seed}.png",
        help="Output heatmap path. Supports {wind_seed}/{terrain_seed}/{start}/{end}/{infra_seed}/{seed_tag} placeholders.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default="Our_experiment/HCSAC/data/offloading_frequency_w{wind_seed}_g{terrain_seed}_{seed_tag}_i{infra_seed}.md",
        help="Output report path. Supports {wind_seed}/{terrain_seed}/{start}/{end}/{infra_seed}/{seed_tag} placeholders.",
    )
    parser.add_argument(
        "--offload-metric",
        type=str,
        default="count",
        choices=["count", "frequency"],
        help="Aggregation metric for offloading heatmaps: raw count or normalized frequency.",
    )
    parser.add_argument(
        "--terrain-map-path",
        type=str,
        default="Our_experiment/HCSAC/data/terrain_difficulty_w{wind_seed}_g{terrain_seed}_i{infra_seed}.png",
        help="Output terrain difficulty map path. Supports {wind_seed}/{terrain_seed}/{infra_seed} placeholders.",
    )
    parser.add_argument(
        "--wind-map-path",
        type=str,
        default="Our_experiment/HCSAC/data/wind_field_w{wind_seed}_g{terrain_seed}_i{infra_seed}.png",
        help="Output wind field map path. Supports {wind_seed}/{terrain_seed}/{infra_seed} placeholders.",
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


def parse_seed_list(seed_list_text):
    if seed_list_text is None:
        return None
    parts = [p.strip() for p in seed_list_text.split(",") if p.strip()]
    if not parts:
        raise ValueError("traj-seed-list is empty.")
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


def generate_commentary(
    freq_maps_by_device,
    wind_seed,
    terrain_seed,
    infra_seed,
    gbs_position,
    haps_position,
    grid_cell_size_m,
    wind_label,
    traj_seeds,
    traj_seed_source_desc,
    metric,
):
    device_labels = ["BS", "HAPS", "LEO", "CE"]
    lines = []
    lines.append("# Offloading Frequency Commentary")
    lines.append("")
    lines.append(f"- Wind seed: `{wind_seed}`")
    lines.append(f"- Terrain seed: `{terrain_seed}`")
    lines.append(f"- Infrastructure seed: `{infra_seed}`")
    lines.append(f"- Wind label: `{wind_label}`")
    lines.append(f"- Trajectory seed source: `{traj_seed_source_desc}`")
    lines.append(f"- Trajectory seeds used ({len(traj_seeds)}): `{summarize_seed_list(traj_seeds)}`")
    lines.append(f"- Grid cell size: `{grid_cell_size_m:.0f} m`")
    lines.append(
        f"- GBS position (grid): `({gbs_position[0]:.2f}, {gbs_position[1]:.2f})`, "
        f"HAPS position (grid): `({haps_position[0]:.2f}, {haps_position[1]:.2f})`"
    )
    lines.append("")

    metric_name = "Count" if metric == "count" else "Frequency"
    lines.append(f"## Top Hotspot Cells by Device (Top-5 {metric_name})")
    lines.append("")
    for i, dev in enumerate(device_labels):
        lines.append(f"### {dev}")
        for x, y, v in top_k_cells(freq_maps_by_device[i], k=5):
            if metric == "count":
                lines.append(f"- cell ({x}, {y}): count `{int(round(v))}`")
            else:
                lines.append(f"- cell ({x}, {y}): frequency `{v:.6f}`")
        lines.append("")

    return "\n".join(lines) + "\n"


def save_offloading_frequency_heatmap(
    freq_maps_by_device,
    output_path,
    wind_seed,
    terrain_seed,
    infra_seed,
    gbs_position,
    haps_position,
    grid_cell_size_m,
    wind_label,
    metric,
    avg_uncertainty,
):
    # Match typography with vis_offloading_visit_frequency_by_wind_class.py
    title_fs = 24
    axis_label_fs = 20
    tick_fs = 16
    suptitle_fs = 24
    cbar_label_fs = 20
    cbar_tick_fs = 16

    device_labels = ["BS", "HAPS", "LEO", "CE"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    vmin = float(np.min(freq_maps_by_device))
    vmax = float(np.max(freq_maps_by_device))

    mappable = None
    for i, dev in enumerate(device_labels):
        ax = axes[i]
        mappable = ax.imshow(
            freq_maps_by_device[i].T,
            cmap="hot",
            origin="lower",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(dev, fontsize=title_fs)
        ax.set_xlabel("Grid X", fontsize=axis_label_fs)
        ax.set_ylabel("Grid Y", fontsize=axis_label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.scatter(gbs_position[0], gbs_position[1], marker="X", s=110, c="cyan", edgecolors="black", linewidths=1.0, label="GBS")
        ax.scatter(haps_position[0], haps_position[1], marker="^", s=110, c="lime", edgecolors="black", linewidths=1.0, label="HAPS")
        if i == 0:
            ax.legend(loc="upper right", fontsize=12, framealpha=0.9)

    # Dedicated right-side colorbar axis to avoid overlap
    cbar_ax = fig.add_axes([0.92, 0.13, 0.024, 0.74])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar_label = "Offloading Count" if metric == "count" else "Offloading Frequency"
    cbar.set_label(cbar_label, fontsize=cbar_label_fs, labelpad=14)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

    metric_text = "Count" if metric == "count" else "Frequency"
    title_line_1 = f"Offloading {metric_text} by Device"
    wind_suffix = ""
    if wind_label and wind_label != "Custom Wind Seed":
        wind_suffix = f", {wind_label}"
    title_line_2 = (
        f"w={wind_seed}, g={terrain_seed}, i={infra_seed}, "
        f"cell={grid_cell_size_m:.0f}m, avg_unc={avg_uncertainty:.6f}{wind_suffix}"
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


def save_terrain_difficulty_map(
    task_matrix,
    output_path,
    wind_seed,
    terrain_seed,
    infra_seed,
    gbs_position,
    haps_position,
    grid_cell_size_m,
):
    title_fs = 24
    axis_label_fs = 20
    tick_fs = 16
    suptitle_fs = 28
    cbar_label_fs = 20
    cbar_tick_fs = 16

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    mappable = ax.imshow(
        task_matrix.T,
        cmap="YlOrRd",
        origin="lower",
        interpolation="nearest",
        vmin=1,
        vmax=4,
    )
    ax.set_title("Terrain Difficulty", fontsize=title_fs)
    ax.set_xlabel("Grid X", fontsize=axis_label_fs)
    ax.set_ylabel("Grid Y", fontsize=axis_label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.scatter(gbs_position[0], gbs_position[1], marker="X", s=140, c="cyan", edgecolors="black", linewidths=1.0, label="GBS")
    ax.scatter(haps_position[0], haps_position[1], marker="^", s=140, c="lime", edgecolors="black", linewidths=1.0, label="HAPS")
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)

    cbar_ax = fig.add_axes([0.92, 0.13, 0.024, 0.74])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label("Difficulty (1-4)", fontsize=cbar_label_fs, labelpad=14)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

    fig.suptitle(
        f"Terrain Difficulty Map (wind_seed={wind_seed}, terrain_seed={terrain_seed}, "
        f"infra_seed={infra_seed}, cell={grid_cell_size_m:.0f}m)",
        fontsize=suptitle_fs,
        y=0.98,
    )
    plt.tight_layout(rect=[0.0, 0.0, 0.90, 0.95])
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def save_wind_field_map(
    wind_u,
    wind_v,
    output_path,
    wind_seed,
    terrain_seed,
    infra_seed,
    gbs_position,
    haps_position,
    grid_cell_size_m,
):
    title_fs = 24
    axis_label_fs = 20
    tick_fs = 16
    suptitle_fs = 28
    cbar_label_fs = 20
    cbar_tick_fs = 16

    speed = np.sqrt(wind_u ** 2 + wind_v ** 2)
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    mappable = ax.imshow(
        speed.T,
        cmap="Blues",
        origin="lower",
        interpolation="nearest",
    )

    # Overlay wind direction vectors.
    grid_x = np.arange(wind_u.shape[0])
    grid_y = np.arange(wind_u.shape[1])
    X, Y = np.meshgrid(grid_x, grid_y, indexing="xy")
    ax.quiver(
        X,
        Y,
        wind_u.T,
        wind_v.T,
        color="black",
        scale=80,
        width=0.0025,
        alpha=0.6,
    )

    ax.set_title("Wind Field", fontsize=title_fs)
    ax.set_xlabel("Grid X", fontsize=axis_label_fs)
    ax.set_ylabel("Grid Y", fontsize=axis_label_fs)
    ax.tick_params(axis="both", labelsize=tick_fs)
    ax.scatter(gbs_position[0], gbs_position[1], marker="X", s=140, c="cyan", edgecolors="black", linewidths=1.0, label="GBS")
    ax.scatter(haps_position[0], haps_position[1], marker="^", s=140, c="lime", edgecolors="black", linewidths=1.0, label="HAPS")
    ax.legend(loc="upper right", fontsize=12, framealpha=0.9)

    cbar_ax = fig.add_axes([0.92, 0.13, 0.024, 0.74])
    cbar = fig.colorbar(mappable, cax=cbar_ax)
    cbar.set_label("Wind Speed", fontsize=cbar_label_fs, labelpad=14)
    cbar.ax.tick_params(labelsize=cbar_tick_fs)

    fig.suptitle(
        f"Wind Field Map (wind_seed={wind_seed}, terrain_seed={terrain_seed}, "
        f"infra_seed={infra_seed}, cell={grid_cell_size_m:.0f}m)",
        fontsize=suptitle_fs,
        y=0.98,
    )
    plt.tight_layout(rect=[0.0, 0.0, 0.90, 0.95])
    plt.savefig(output_path, dpi=220, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def main():
    args = parse_args()

    reps = load_wind_representatives(args.wind_catalog_json)
    wind_seed = int(args.wind_seed) if args.wind_seed is not None else int(reps[args.wind_class])
    terrain_seed = int(args.terrain_seed) if args.terrain_seed is not None else wind_seed
    infra_seed = int(args.infra_seed) if args.infra_seed is not None else wind_seed
    wind_label = args.wind_class if args.wind_seed is None else "Custom Wind Seed"
    traj_seeds, traj_seed_source_desc, seed_tag = select_traj_seeds(args)
    if len(traj_seeds) == 0:
        raise RuntimeError("No trajectory seeds selected.")

    env, agent, offload_agent = build_agents_and_env()

    # Before rollouts: output terrain difficulty map and wind field map.
    env.reset(seed=0, wind_seed=wind_seed, terrain_seed=terrain_seed, infra_seed=infra_seed)
    gbs_position = np.array(env.gbs_position, dtype=np.float64).copy()
    haps_position = np.array(env.haps_position, dtype=np.float64).copy()
    grid_cell_size_m = float(getattr(env, "grid_cell_size_m", env.X / env.Lx))
    terrain_map_path = args.terrain_map_path.format(wind_seed=wind_seed, terrain_seed=terrain_seed, infra_seed=infra_seed)
    wind_map_path = args.wind_map_path.format(wind_seed=wind_seed, terrain_seed=terrain_seed, infra_seed=infra_seed)
    os.makedirs(os.path.dirname(terrain_map_path), exist_ok=True)
    os.makedirs(os.path.dirname(wind_map_path), exist_ok=True)
    save_terrain_difficulty_map(
        env.task_matrix.astype(np.float64),
        terrain_map_path,
        wind_seed,
        terrain_seed,
        infra_seed,
        gbs_position,
        haps_position,
        grid_cell_size_m,
    )
    save_wind_field_map(
        env.wind_u.astype(np.float64),
        env.wind_v.astype(np.float64),
        wind_map_path,
        wind_seed,
        terrain_seed,
        infra_seed,
        gbs_position,
        haps_position,
        grid_cell_size_m,
    )

    # Device order from env: [L, BS, HAPS, LEO, CE], we only keep 4 offloading devices [BS,HAPS,LEO,CE]
    offload_sum_by_device = None  # shape: [4, grid_x, grid_y]
    uncertainty_list = []

    for traj_seed in traj_seeds:
        set_seed(int(traj_seed))
        stats = vis(
            agent,
            offload_agent,
            env,
            seed=int(traj_seed),
            return_stats=True,
            wind_seed=wind_seed,
            terrain_seed=terrain_seed,
            traj_seed=int(traj_seed),
            infra_seed=infra_seed,
        )

        heatmaps_all_targets = stats["offload_heatmaps_by_target"].astype(np.float64)
        heatmaps_devices = heatmaps_all_targets[1:5]  # [BS,HAPS,LEO,CE]

        if offload_sum_by_device is None:
            offload_sum_by_device = np.zeros_like(heatmaps_devices, dtype=np.float64)

        offload_sum_by_device += heatmaps_devices
        uncertainty_list.append(float(stats["avg_uncertainty"]))

    # Use raw offloading counts by default; optional normalized frequency is still available.
    if args.offload_metric == "count":
        freq_maps_by_device = offload_sum_by_device.copy()
    else:
        freq_maps_by_device = np.zeros_like(offload_sum_by_device, dtype=np.float64)
        for i in range(offload_sum_by_device.shape[0]):
            total_i = float(np.sum(offload_sum_by_device[i]))
            if total_i > 0:
                freq_maps_by_device[i] = offload_sum_by_device[i] / total_i
            else:
                freq_maps_by_device[i] = offload_sum_by_device[i]

    seed_start = int(min(traj_seeds))
    seed_end = int(max(traj_seeds))
    heatmap_path = args.heatmap_path.format(
        wind_seed=wind_seed,
        terrain_seed=terrain_seed,
        start=seed_start,
        end=seed_end,
        seed_tag=seed_tag,
        infra_seed=infra_seed,
    )
    report_path = args.report_path.format(
        wind_seed=wind_seed,
        terrain_seed=terrain_seed,
        start=seed_start,
        end=seed_end,
        seed_tag=seed_tag,
        infra_seed=infra_seed,
    )
    heatmap_dir = os.path.dirname(heatmap_path)
    report_dir = os.path.dirname(report_path)
    if heatmap_dir:
        os.makedirs(heatmap_dir, exist_ok=True)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    mean_unc = float(np.mean(uncertainty_list)) if uncertainty_list else float("nan")
    save_offloading_frequency_heatmap(
        freq_maps_by_device,
        heatmap_path,
        wind_seed,
        terrain_seed,
        infra_seed,
        gbs_position,
        haps_position,
        grid_cell_size_m,
        wind_label,
        args.offload_metric,
        mean_unc,
    )
    commentary = generate_commentary(
        freq_maps_by_device,
        wind_seed,
        terrain_seed,
        infra_seed,
        gbs_position,
        haps_position,
        grid_cell_size_m,
        wind_label,
        traj_seeds,
        traj_seed_source_desc,
        args.offload_metric,
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(commentary)

    print("-" * 60)
    print("Offloading frequency analysis done.")
    print(f"Wind used: class={wind_label}, wind_seed={wind_seed}")
    print(f"Terrain seed: {terrain_seed}")
    print(f"Infrastructure seed: {infra_seed}")
    print(
        "GBS/HAPS positions (grid): "
        f"GBS=({gbs_position[0]:.2f}, {gbs_position[1]:.2f}), "
        f"HAPS=({haps_position[0]:.2f}, {haps_position[1]:.2f})"
    )
    print(f"Grid cell size: {grid_cell_size_m:.0f} m")
    print(f"Trajectory seed source: {traj_seed_source_desc}")
    print(f"Trajectory seeds used ({len(traj_seeds)}): {summarize_seed_list(traj_seeds)}")
    print(f"Offloading metric: {args.offload_metric}")
    print(f"Mean Average uncertainty: {mean_unc:.6f}")
    print(f"Terrain map saved: {terrain_map_path}")
    print(f"Wind field map saved: {wind_map_path}")
    print(f"Heatmap saved: {heatmap_path}")
    print(f"Commentary saved: {report_path}")


if __name__ == "__main__":
    main()
