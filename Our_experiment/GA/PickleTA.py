import pickle

def save_results(file_path, best_solution, best_fitness, trajectory_actions, offload_actions):
    """
    保存结果到文件
    """
    data = {
        "best_solution": best_solution,
        "best_fitness": best_fitness,
        "trajectory_actions": trajectory_actions,
        "offload_actions": offload_actions
    }
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Results saved to {file_path}")

def load_results(file_path):
    """
    从文件读取结果
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Results loaded from {file_path}")
    return data["best_solution"], data["best_fitness"], data["trajectory_actions"], data["offload_actions"]