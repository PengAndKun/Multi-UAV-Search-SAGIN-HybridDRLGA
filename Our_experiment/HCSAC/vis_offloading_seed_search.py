import sys
import os
import argparse
import random

# Headless mode: disable visualization window/audio by default
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

# Skip all pygame waits in seed-search mode to speed up runs
pygame.time.wait = lambda ms: None


def parse_args():
    parser = argparse.ArgumentParser(description="Search best seed by minimum average uncertainty.")
    parser.add_argument("--wind-seed", type=int, default=0, help="Seed controlling wind field/environment reset.")
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
    parser.add_argument("--traj-seed-start", type=int, default=0, help="Start trajectory seed (inclusive).")
    parser.add_argument("--traj-seed-end", type=int, default=9, help="End trajectory seed (inclusive).")
    parser.add_argument(
        "--heatmap-path",
        type=str,
        default="Our_experiment/HCSAC/data/best_offloading_heatmap_w{wind_seed}_g{terrain_seed}_t{traj_seed}_i{infra_seed}.png",
        help="Output path for best-seed 4-device heatmap. Supports {wind_seed}/{terrain_seed}/{traj_seed}/{infra_seed} placeholders.",
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


def save_offloading_heatmap_by_device(
    offload_heatmaps_by_target,
    offload_targets,
    output_path,
    wind_seed,
    terrain_seed,
    traj_seed,
    infra_seed,
    gbs_position,
    haps_position,
    grid_cell_size_m,
):
    title_fs = 18
    axis_label_fs = 15
    tick_fs = 13
    suptitle_fs = 20
    cbar_label_fs = 15
    cbar_tick_fs = 13

    device_indices = [1, 2, 3, 4]  # BS/HAPS/LEO/CE
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    mappable = None
    for plot_idx, (ax, idx) in enumerate(zip(axes, device_indices)):
        mappable = ax.imshow(
            offload_heatmaps_by_target[idx].T,
            cmap="hot",
            origin="lower",
            interpolation="nearest",
        )
        ax.set_title(f"{offload_targets[idx]} (w={wind_seed}, g={terrain_seed}, t={traj_seed}, i={infra_seed})", fontsize=title_fs)
        ax.set_xlabel("Grid X", fontsize=axis_label_fs)
        ax.set_ylabel("Grid Y", fontsize=axis_label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)
        ax.scatter(gbs_position[0], gbs_position[1], marker="X", s=90, c="cyan", edgecolors="black", linewidths=1.0, label="GBS")
        ax.scatter(haps_position[0], haps_position[1], marker="^", s=90, c="lime", edgecolors="black", linewidths=1.0, label="HAPS")
        if plot_idx == 0:
            ax.legend(loc="upper right", fontsize=10, framealpha=0.9)

    if mappable is not None:
        cbar_ax = fig.add_axes([0.885, 0.13, 0.024, 0.74])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label("Offloading Count", fontsize=cbar_label_fs)
        cbar.ax.tick_params(labelsize=cbar_tick_fs)

    fig.suptitle(
        f"Offloading Heatmaps by Device (BS/HAPS/LEO/CE) "
        f"[wind={wind_seed}, terrain={terrain_seed}, traj={traj_seed}, infra={infra_seed}, cell={grid_cell_size_m:.0f}m]",
        y=0.98,
        fontsize=suptitle_fs,
    )
    plt.tight_layout(rect=[0.0, 0.0, 0.87, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def main():
    args = parse_args()
    if args.traj_seed_end < args.traj_seed_start:
        raise ValueError("traj-seed-end must be >= traj-seed-start")
    terrain_seed = args.terrain_seed if args.terrain_seed is not None else args.wind_seed
    infra_seed = args.infra_seed if args.infra_seed is not None else args.wind_seed

    env, agent, offload_agent = build_agents_and_env()

    best_traj_seed = None
    best_uncertainty = float("inf")
    best_stats = None

    for traj_seed in range(args.traj_seed_start, args.traj_seed_end + 1):
        set_seed(traj_seed)
        stats = vis(
            agent,
            offload_agent,
            env,
            seed=traj_seed,
            return_stats=True,
            wind_seed=args.wind_seed,
            terrain_seed=terrain_seed,
            traj_seed=traj_seed,
            infra_seed=infra_seed,
        )
        avg_uncertainty = float(stats["avg_uncertainty"])
        print(
            f"wind_seed={args.wind_seed}, terrain_seed={terrain_seed}, "
            f"traj_seed={traj_seed}, average_uncertainty={avg_uncertainty:.6f}"
        )

        if avg_uncertainty < best_uncertainty:
            best_uncertainty = avg_uncertainty
            best_traj_seed = traj_seed
            best_stats = {
                "offload_heatmaps_by_target": stats["offload_heatmaps_by_target"].copy(),
                "offload_targets": list(stats["offload_targets"]),
                "gbs_position": np.array(stats["gbs_position"], dtype=np.float64).copy(),
                "haps_position": np.array(stats["haps_position"], dtype=np.float64).copy(),
                "grid_cell_size_m": float(stats["grid_cell_size_m"]),
            }

    if best_traj_seed is None or best_stats is None:
        raise RuntimeError("No seed result was produced.")

    output_path = args.heatmap_path.format(
        wind_seed=args.wind_seed,
        terrain_seed=terrain_seed,
        traj_seed=best_traj_seed,
        infra_seed=infra_seed,
    )
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    save_offloading_heatmap_by_device(
        best_stats["offload_heatmaps_by_target"],
        best_stats["offload_targets"],
        output_path,
        args.wind_seed,
        terrain_seed,
        best_traj_seed,
        infra_seed,
        best_stats["gbs_position"],
        best_stats["haps_position"],
        best_stats["grid_cell_size_m"],
    )

    print("-" * 60)
    print(f"Best wind_seed: {args.wind_seed}")
    print(f"Terrain seed: {terrain_seed}")
    print(f"Infrastructure seed: {infra_seed}")
    print(
        "GBS/HAPS positions (grid): "
        f"GBS=({best_stats['gbs_position'][0]:.2f}, {best_stats['gbs_position'][1]:.2f}), "
        f"HAPS=({best_stats['haps_position'][0]:.2f}, {best_stats['haps_position'][1]:.2f})"
    )
    print(f"Grid cell size: {best_stats['grid_cell_size_m']:.0f} m")
    print(f"Best traj_seed: {best_traj_seed}")
    print(f"Minimum average uncertainty: {best_uncertainty:.6f}")
    print(f"Best-seed heatmap saved to: {output_path}")


if __name__ == "__main__":
    main()
