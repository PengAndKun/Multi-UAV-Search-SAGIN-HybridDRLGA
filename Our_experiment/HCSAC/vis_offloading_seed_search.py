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
    parser.add_argument("--seed-start", type=int, default=0, help="Start seed (inclusive).")
    parser.add_argument("--seed-end", type=int, default=9, help="End seed (inclusive).")
    parser.add_argument(
        "--heatmap-path",
        type=str,
        default="Our_experiment/HCSAC/data/best_offloading_heatmap_devices_seed_{seed}.png",
        help="Output path for best-seed 4-device heatmap. Supports {seed} placeholder.",
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


def save_offloading_heatmap_by_device(offload_heatmaps_by_target, offload_targets, output_path, seed):
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
    for ax, idx in zip(axes, device_indices):
        mappable = ax.imshow(
            offload_heatmaps_by_target[idx].T,
            cmap="hot",
            origin="lower",
            interpolation="nearest",
        )
        ax.set_title(f"{offload_targets[idx]} (seed={seed})", fontsize=title_fs)
        ax.set_xlabel("Grid X", fontsize=axis_label_fs)
        ax.set_ylabel("Grid Y", fontsize=axis_label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)

    if mappable is not None:
        cbar_ax = fig.add_axes([0.92, 0.13, 0.02, 0.74])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label("Offloading Count", fontsize=cbar_label_fs)
        cbar.ax.tick_params(labelsize=cbar_tick_fs)

    fig.suptitle("Offloading Heatmaps by Device (BS/HAPS/LEO/CE)", y=0.98, fontsize=suptitle_fs)
    plt.tight_layout(rect=[0.0, 0.0, 0.9, 0.96])
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    if args.seed_end < args.seed_start:
        raise ValueError("seed-end must be >= seed-start")

    env, agent, offload_agent = build_agents_and_env()

    best_seed = None
    best_uncertainty = float("inf")
    best_stats = None

    for seed in range(args.seed_start, args.seed_end + 1):
        set_seed(seed)
        stats = vis(agent, offload_agent, env, seed=seed, return_stats=True)
        avg_uncertainty = float(stats["avg_uncertainty"])
        print(f"seed={seed}, average_uncertainty={avg_uncertainty:.6f}")

        if avg_uncertainty < best_uncertainty:
            best_uncertainty = avg_uncertainty
            best_seed = seed
            best_stats = {
                "offload_heatmaps_by_target": stats["offload_heatmaps_by_target"].copy(),
                "offload_targets": list(stats["offload_targets"]),
            }

    if best_seed is None or best_stats is None:
        raise RuntimeError("No seed result was produced.")

    output_path = args.heatmap_path.format(seed=best_seed)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    save_offloading_heatmap_by_device(
        best_stats["offload_heatmaps_by_target"],
        best_stats["offload_targets"],
        output_path,
        best_seed,
    )

    print("-" * 60)
    print(f"Best seed: {best_seed}")
    print(f"Minimum average uncertainty: {best_uncertainty:.6f}")
    print(f"Best-seed heatmap saved to: {output_path}")


if __name__ == "__main__":
    main()
