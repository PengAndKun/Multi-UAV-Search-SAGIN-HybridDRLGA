import torch
import pickle

def save_sac_agent(agent, path='sac_model'):
    """保存 SAC 智能体的所有必要组件"""
    # en:sure the directory exists
    torch.save({
        # 网络状态字典 en: Network state dictionaries
        'actor_state_dict': agent.actor.state_dict(),
        'critic_1_state_dict': agent.critic_1.state_dict(),
        'critic_2_state_dict': agent.critic_2.state_dict(),
        'target_critic_1_state_dict': agent.target_critic_1.state_dict(),
        'target_critic_2_state_dict': agent.target_critic_2.state_dict(),
        
        # 优化器状态字典 en: Optimizer state dictionaries
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_1_optimizer_state_dict': agent.critic_1_optimizer.state_dict(),
        'critic_2_optimizer_state_dict': agent.critic_2_optimizer.state_dict(),
        'log_alpha_optimizer_state_dict': agent.log_alpha_optimizer.state_dict(),
        
        # 其他参数 en: Other parameters
        'log_alpha': agent.log_alpha.item(),
        # 'tau': agent.tau,
        # 'gamma': agent.gamma,
        # 'target_entropy': agent.target_entropy
    }, f'{path}.pt')
    print(f"已保存 SAC 模型到 {path}.pt")

def load_sac_agent(agent, path='sac_model', device='cuda'):
    """加载 SAC 智能体的所有必要组件"""
    # en: Load the checkpoint
    checkpoint = torch.load(path + '.pt', map_location=device)
    
    # 加载网络参数 # en: Load network parameters
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
    agent.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
    agent.target_critic_1.load_state_dict(checkpoint['target_critic_1_state_dict'])
    agent.target_critic_2.load_state_dict(checkpoint['target_critic_2_state_dict'])
    
    # 加载优化器参数 # en: Load optimizer parameters
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    agent.critic_1_optimizer.load_state_dict(checkpoint['critic_1_optimizer_state_dict'])
    agent.critic_2_optimizer.load_state_dict(checkpoint['critic_2_optimizer_state_dict'])
    agent.log_alpha_optimizer.load_state_dict(checkpoint['log_alpha_optimizer_state_dict'])
    
    # 加载其他参数 # en: Load other parameters
    agent.log_alpha = torch.tensor(checkpoint['log_alpha'], requires_grad=True, device=device)
    
    print(f"已从 {path}.pt 加载 SAC 模型")
    return agent

# 同时也保存训练记录 # en: Also save training history
def save_training_history(return_list, return_list_2, path='training_history'):
    """保存训练过程中的回报列表"""
    # en: Save the training history
    history = {
        'return_list': return_list,
        'return_list_2': return_list_2
    }
    with open(f'{path}.pkl', 'wb') as f:
        pickle.dump(history, f)
    print(f"已保存训练历史到 {path}.pkl")

def load_training_history(path='training_history'):
    """加载训练过程中的回报列表"""
    # en: Load the training history
    with open(f'{path}.pkl', 'rb') as f:
        history = pickle.load(f)
    return history['return_list'], history['return_list_2']