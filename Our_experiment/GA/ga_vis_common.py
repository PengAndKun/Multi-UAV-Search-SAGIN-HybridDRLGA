import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from Our_experiment.HCSAC.ENV.UAVenv_SAC_Original import SAC
from Our_experiment.HCSAC.ENV.UAVenv_SAC_Original import UAVEnv as UAVenv
from Our_experiment.HCSAC import UAV_SAVE


OFFLOAD_TARGETS = ["L", "BS", "HAPS", "LEO", "CE"]


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


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def utc_now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def resolve_path_template(path_template, **kwargs):
    return path_template.format(**kwargs) if "{" in path_template else path_template


def validate_traj_range(start, end):
    if end < start:
        raise ValueError("traj-seed-end must be >= traj-seed-start")


def normalize_positions(positions, num_uav, lx, ly):
    if positions is None or len(positions) != num_uav:
        raise ValueError(f"deployment positions must contain exactly {num_uav} points.")
    parsed = []
    used = set()
    for p in positions:
        if not isinstance(p, (list, tuple)) or len(p) != 2:
            raise ValueError(f"invalid position format: {p}")
        x = int(p[0])
        y = int(p[1])
        if not (0 <= x < lx and 0 <= y < ly):
            raise ValueError(f"position out of bounds: {(x, y)}")
        if (x, y) in used:
            raise ValueError(f"duplicate deployment position: {(x, y)}")
        used.add((x, y))
        parsed.append((x, y))
    return parsed


def apply_deployment(env, deployment_positions):
    positions = normalize_positions(deployment_positions, env.N, env.Lx, env.Ly)
    base = list(env.gird_position)
    if len(base) < env.N:
        base.extend([(0, 0)] * (env.N - len(base)))
    for i, pos in enumerate(positions):
        base[i] = pos
    env.gird_position = base
    return positions


def build_agents_and_env(num_uav=4):
    env = UAVenv(num_uav)
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

    fly_model = os.path.join(project_root, "Our_experiment", "HCSAC", "data", "sac_model_fly")
    offload_model = os.path.join(project_root, "Our_experiment", "HCSAC", "data", "sac_model_offload")
    agent = UAV_SAVE.load_sac_agent(agent, path=fly_model, device=device)
    offload_agent = UAV_SAVE.load_sac_agent(offload_agent, path=offload_model, device=device)
    return env, agent, offload_agent


def rollout_once(
    env,
    agent,
    offload_agent,
    deployment_positions,
    wind_seed,
    terrain_seed,
    infra_seed,
    traj_seed,
    deployment_destinations=None,
    return_actions=False,
):
    positions = normalize_positions(deployment_positions, env.N, env.Lx, env.Ly)
    if deployment_destinations is not None:
        destinations = normalize_positions(deployment_destinations, env.N, env.Lx, env.Ly)
        reset_state = env.reset(
            seed=traj_seed,
            positions=positions,
            destinations=destinations,
            wind_seed=wind_seed,
            terrain_seed=terrain_seed,
            infra_seed=infra_seed,
        )
    else:
        apply_deployment(env, positions)
        reset_state = env.reset(
            seed=traj_seed,
            wind_seed=wind_seed,
            terrain_seed=terrain_seed,
            infra_seed=infra_seed,
        )
    state = reset_state if reset_state is not None else env._get_obs()

    # Keep the same stochastic action sampling behavior as original visualization.
    set_seed(traj_seed)

    grid_x = env.Lx
    grid_y = env.Ly
    visit_count = np.zeros((grid_x, grid_y), dtype=np.float64)
    visit_count_by_uav = np.zeros((env.N, grid_x, grid_y), dtype=np.float64)
    offload_heatmap = np.zeros((grid_x, grid_y), dtype=np.float64)
    offload_heatmaps_by_target = np.zeros((len(OFFLOAD_TARGETS), grid_x, grid_y), dtype=np.float64)
    trajectory_actions = []
    offloading_actions = []
    uav_lifetime_steps = np.full(env.N, np.nan, dtype=np.float64)
    step_count = 0

    done = False
    current_avg_uncertainty = float(np.mean(env.uncertainty_matrix))
    while not done:
        step_count += 1
        actions = [agent.take_action(state[n]) for n in range(env.N)]
        if return_actions:
            trajectory_actions.append(actions)
        next_state, _, done = env.step(actions)

        offload_data = env.get_obs_2()
        offload_actions = offload_agent.take_action(offload_data)
        if return_actions:
            offloading_actions.append(offload_actions)
        _, _, done = env.step_offload(offload_actions)
        state = next_state

        current_avg_uncertainty = float(np.mean(env.uncertainty_matrix))
        for i, uav in enumerate(env.uavs):
            if np.isnan(uav_lifetime_steps[i]) and bool(uav["done"]):
                uav_lifetime_steps[i] = float(step_count)

        for i, uav in enumerate(env.uavs):
            if uav["done"] and uav["position"] == uav["destination"]:
                continue

            x = int(uav["position"][0])
            y = int(uav["position"][1])
            visit_count[x, y] += 1.0
            visit_count_by_uav[i, x, y] += 1.0

            offload_action = offload_actions[i] if i < len(offload_actions) else 0
            offload_action = int(np.clip(offload_action, 0, len(OFFLOAD_TARGETS) - 1))
            offload_heatmap[x, y] += 1.0
            offload_heatmaps_by_target[offload_action, x, y] += 1.0

    if env.N > 0:
        uav_lifetime_steps = np.where(np.isnan(uav_lifetime_steps), float(step_count), uav_lifetime_steps)
        avg_uav_lifetime_steps = float(np.mean(uav_lifetime_steps))
    else:
        uav_lifetime_steps = np.array([], dtype=np.float64)
        avg_uav_lifetime_steps = float(step_count)
    avg_uav_lifetime_seconds = float(avg_uav_lifetime_steps * float(env.T))

    result = {
        "avg_uncertainty": current_avg_uncertainty,
        "offload_heatmap": offload_heatmap,
        "offload_heatmaps_by_target": offload_heatmaps_by_target,
        "offload_targets": OFFLOAD_TARGETS.copy(),
        "visit_count": visit_count,
        "visit_count_by_uav": visit_count_by_uav,
        "wind_seed": int(wind_seed),
        "terrain_seed": int(terrain_seed),
        "infra_seed": int(infra_seed),
        "traj_seed": int(traj_seed),
        "gbs_position": np.array(env.gbs_position, dtype=np.float64).copy(),
        "haps_position": np.array(env.haps_position, dtype=np.float64).copy(),
        "grid_cell_size_m": float(getattr(env, "grid_cell_size_m", env.X / env.Lx)),
        "step_count": int(step_count),
        "uav_lifetime_steps": uav_lifetime_steps.tolist(),
        "avg_uav_lifetime_steps": float(avg_uav_lifetime_steps),
        "avg_uav_lifetime_seconds": float(avg_uav_lifetime_seconds),
    }
    if return_actions:
        result["trajectory_actions"] = trajectory_actions
        result["offloading_actions"] = offloading_actions
    return result
