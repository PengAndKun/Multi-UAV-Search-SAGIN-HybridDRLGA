import sys
import os
import argparse
import random

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


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single seed with seed-search-consistent behavior.")
    parser.add_argument("--seed", type=int, default=None, help="Legacy single seed (maps to both wind/traj if others unset).")
    parser.add_argument("--wind-seed", type=int, default=None, help="Seed controlling wind field/environment reset.")
    parser.add_argument("--traj-seed", type=int, default=None, help="Seed controlling stochastic trajectory sampling.")
    parser.add_argument(
        "--heatmap-path",
        type=str,
        default="Our_experiment/HCSAC/data/replay_offloading_heatmap_w{wind_seed}_t{traj_seed}.png",
        help="Output path for 4-device offloading heatmap. Supports {wind_seed}/{traj_seed} placeholders.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show pygame visualization window. Default is headless (no window).",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_display(show):
    if show:
        os.environ.pop("SDL_VIDEODRIVER", None)
        os.environ.pop("SDL_AUDIODRIVER", None)
    else:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
        # Match seed-search behavior in headless mode: skip pygame waits.
        pygame.time.wait = lambda ms: None


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


def save_offloading_heatmap_by_device(offload_heatmaps_by_target, offload_targets, output_path, wind_seed, traj_seed):
    title_fs = 24
    axis_label_fs = 20
    tick_fs = 16
    suptitle_fs = 28
    cbar_label_fs = 20
    cbar_tick_fs = 16

    device_indices = [1, 2, 3, 4]  # BS/HAPS/LEO/CE
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    mappable = None
    for ax, idx in zip(axes, device_indices):
        mappable = ax.imshow(
            offload_heatmaps_by_target[idx].T,
            cmap="hot",
            origin="lower",
            interpolation="nearest",
        )
        ax.set_title(f"{offload_targets[idx]} (w={wind_seed}, t={traj_seed})", fontsize=title_fs)
        ax.set_xlabel("Grid X", fontsize=axis_label_fs)
        ax.set_ylabel("Grid Y", fontsize=axis_label_fs)
        ax.tick_params(axis="both", labelsize=tick_fs)

    if mappable is not None:
        cbar_ax = fig.add_axes([0.885, 0.13, 0.024, 0.74])
        cbar = fig.colorbar(mappable, cax=cbar_ax)
        cbar.set_label("Offloading Count", fontsize=cbar_label_fs, labelpad=14)
        cbar.ax.tick_params(labelsize=cbar_tick_fs)

    fig.suptitle(
        f"Offloading Heatmaps by Device (BS/HAPS/LEO/CE) [wind={wind_seed}, traj={traj_seed}]",
        y=0.98,
        fontsize=suptitle_fs,
    )
    plt.tight_layout(rect=[0.0, 0.0, 0.87, 0.96])
    plt.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def main():
    args = parse_args()
    configure_display(args.show)

    # Resolve dual seeds with backward compatibility.
    if args.wind_seed is None and args.traj_seed is None:
        if args.seed is None:
            raise ValueError("Please provide --wind-seed and --traj-seed, or legacy --seed.")
        wind_seed = args.seed
        traj_seed = args.seed
    else:
        wind_seed = args.wind_seed if args.wind_seed is not None else args.seed
        traj_seed = args.traj_seed if args.traj_seed is not None else args.seed
        if wind_seed is None or traj_seed is None:
            raise ValueError("Both wind_seed and traj_seed must be resolved.")

    env, agent, offload_agent = build_agents_and_env()
    # Keep timing consistent with seed-search: set trajectory seed right before rollout.
    set_seed(traj_seed)
    stats = vis(
        agent,
        offload_agent,
        env,
        seed=traj_seed,
        return_stats=True,
        wind_seed=wind_seed,
        traj_seed=traj_seed,
    )

    output_path = args.heatmap_path.format(wind_seed=wind_seed, traj_seed=traj_seed)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    save_offloading_heatmap_by_device(
        stats["offload_heatmaps_by_target"],
        stats["offload_targets"],
        output_path,
        wind_seed,
        traj_seed,
    )

    print(f"Replay wind_seed: {wind_seed}")
    print(f"Replay traj_seed: {traj_seed}")
    print(f"Visualization shown: {args.show}")
    print(f"Average uncertainty: {float(stats['avg_uncertainty']):.6f}")
    print(f"Offloading heatmaps (BS/HAPS/LEO/CE) saved to: {output_path}")


if __name__ == "__main__":
    main()
