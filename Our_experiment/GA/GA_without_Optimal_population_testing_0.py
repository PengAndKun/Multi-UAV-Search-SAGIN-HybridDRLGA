import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(project_root)

from Our_experiment.HCSAC.UAVenv import UAVEnv as UAVenv
from Our_experiment.HCSAC.UAV_VIS_offloading import visualize_trajectory as vis
from Our_experiment.HCSAC import UAV_SAVE
from Our_experiment.GA.algorithm import SAC
import torch
import numpy as np
import PickleTA as pack

env = UAVenv(4)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device(
    "cpu")
hidden_dim = 128
gamma = 0.99
tau = 0.005  # 软更新参数
actor_lr = 3e-4
critic_lr = 3e-4
alpha_lr = 1e-4
state_dim = env.state_dim
action_dim = env.action_dim
offload_state_dim = env.offload_state_dim
offload_action_dim = env.offload_action_dim
target_entropy = -np.log(action_dim)
target_entropy_offload = -np.log(offload_action_dim)



agent = SAC.SAC(state_dim, hidden_dim, action_dim, actor_lr, critic_lr, alpha_lr,
                target_entropy, tau, gamma, device)
offload_agent = SAC.SAC(offload_state_dim, hidden_dim, offload_action_dim, actor_lr, critic_lr, alpha_lr, target_entropy_offload, tau, gamma, device, type ='GCN')
agent = UAV_SAVE.load_sac_agent(agent, path='../HCSAC/data/sac_model_fly',device=device)
offload_agent = UAV_SAVE.load_sac_agent(offload_agent, path='../HCSAC/data/sac_model_offload',device=device)

vis(agent, offload_agent, env,8876)