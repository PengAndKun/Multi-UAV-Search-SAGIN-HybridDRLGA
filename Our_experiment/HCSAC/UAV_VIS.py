import pygame
import sys 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable

def visualize_trajectory(agent, offload_agent, env):
    # 初始化 Pygame  en: Initialize Pygame
    pygame.init()

    # 屏幕尺寸 en: Screen dimensions
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 800

    # 颜色 en: Define colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    PURPLE = (128, 0, 128)  # 添加紫色用于显示不确定度 en: Purple for uncertainty display

    # 初始化屏幕 en: Initialize the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("无人机轨迹") # en: Set window title

    # 网格设置 en: Grid settings
    GRID_SIZE = env.Lx
    CELL_SIZE = SCREEN_WIDTH // GRID_SIZE

    # 定义卸载目标的名称 en: Define offload targets
    OFFLOAD_TARGETS = ["L", "BS", "HAPS", "LEO", "CE"]
    # 卸载目标对应的颜色 en: Define colors for offload targets
    OFFLOAD_COLORS = [
        (0, 200, 0),     # 本地-绿色 en: Local - Green
        (0, 0, 200),     # 基站-蓝色 en: Base Station - Blue
        (200, 0, 200),   # HAPS-紫色 en: HAPS - Purple
        (165,42,42),   # LEO-黄色 en: LEO - Brown
        (0, 200, 200)    # 云端-青色 en: Cloud - Cyan
    ]

    def draw_grid():
        """在屏幕上绘制网格"""
        # en:Draw a grid on the screen
        for i in range(GRID_SIZE):
            # 绘制垂直线 # en: Draw vertical lines
            pygame.draw.line(screen, BLACK, (i * CELL_SIZE, 0), (i * CELL_SIZE, SCREEN_HEIGHT))
            # 绘制水平线 # en: Draw horizontal lines
            pygame.draw.line(screen, BLACK, (0, i * CELL_SIZE), (SCREEN_WIDTH, i * CELL_SIZE))

    def draw_uncertainty_matrix(uncertainty_matrix):
        """在屏幕上绘制不确定度矩阵"""
        # en: Draw the uncertainty matrix on the screen
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                uncertainty = uncertainty_matrix[i, j]
                # 根据不确定度大小设置颜色深浅
                # en: Set color intensity based on uncertainty
                color_intensity = int(255 * (1 - uncertainty))
                color = (255, color_intensity, color_intensity)
                pygame.draw.rect(screen, color, (i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def draw_wind_field(wind_u, wind_v):
        """绘制风场向量"""
        # en: Draw wind field vectors
        scale = 5.0  # 缩放因子，调整箭头长度 en: Scale factor to adjust arrow length
        for i in range(0, GRID_SIZE, 1):  # 每隔2个格子绘制一个箭头，避免过于密集 en: Draw arrows every 2 grid cells to avoid overcrowding
            for j in range(0, GRID_SIZE, 1):
                u = wind_u[j, i]
                v = wind_v[j, i]
                # 计算风向箭头的起点和终点 en: Calculate the start and end points of the wind direction arrow
                start_x = i * CELL_SIZE + CELL_SIZE//2
                start_y = j * CELL_SIZE + CELL_SIZE//2
                end_x = start_x + int(u * scale)
                end_y = start_y + int(v * scale)
                
                # 绘制箭头线 en: Draw the arrow line
                if abs(u) > 0.1 or abs(v) > 0.1:  # 只绘制足够大的风向 en: Only draw arrows for significant wind vectors
                    pygame.draw.line(screen, BLACK, (start_x, start_y), (end_x, end_y), 1)
                    # 计算箭头角度 en: Calculate the angle of the arrow
                    angle = np.arctan2(v, u)
                    # 绘制箭头头部 en: Draw the arrow head
                    head_len = 5
                    pygame.draw.line(screen, BLACK, (end_x, end_y), 
                                    (end_x - head_len * np.cos(angle + np.pi/6), 
                                     end_y - head_len * np.sin(angle + np.pi/6)), 1)
                    pygame.draw.line(screen, BLACK, (end_x, end_y), 
                                    (end_x - head_len * np.cos(angle - np.pi/6), 
                                     end_y - head_len * np.sin(angle - np.pi/6)), 1)

    def draw_visit_count(visit_count):
        """在屏幕上绘制访问次数"""
        # en: Draw visit count on the screen
        font = pygame.font.Font(None, 24)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                count = visit_count[i, j]
                if count > 0:
                    text = font.render(f"{count}", True, BLUE)
                    screen.blit(text, (i * CELL_SIZE + CELL_SIZE//2, j * CELL_SIZE + CELL_SIZE//2))

    def draw_unload_count(unload_count):
        """在屏幕上绘制卸载次数"""
        # en: Draw offload count on the screen
        font = pygame.font.Font(None, 18)
        for (x, y), counts in unload_count.items():
            text_y_offset = -30
            for i, count in enumerate(counts):
                if count > 0:
                    text = font.render(f"{OFFLOAD_TARGETS[i]}:{count}", True, OFFLOAD_COLORS[i])
                    screen.blit(text, (x - 30, y + text_y_offset))
                    text_y_offset += 15  # 向下移动以便显示下一个卸载信息 en: Move down for the next offload info

    def draw_avg_uncertainty(avg_uncertainty):
        """在右上角绘制当前平均不确定度信息"""
        # en: Draw current average uncertainty in the top right corner
        font = pygame.font.SysFont('Arial', 24, bold=True)
        # 创建一个半透明的背景矩形 en: Create a semi-transparent background rectangle
        bg_rect = pygame.Rect(SCREEN_WIDTH - 300, 10, 290, 50)
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
        bg_surface.set_alpha(180)
        bg_surface.fill(WHITE)
        screen.blit(bg_surface, bg_rect)
        
        # 绘制当前平均不确定度 en: Draw current average uncertainty
        text_current = font.render(f"average uncertainty: {avg_uncertainty:.4f}", True, PURPLE)
        screen.blit(text_current, (SCREEN_WIDTH - 290, 20))
        
        # 绘制边框 en: Draw border around the background rectangle
        pygame.draw.rect(screen, BLACK, bg_rect, 2)

    def draw_offload_ratio(unload_count):
        """在右下角绘制各卸载目标的比例"""
        # en: Draw the ratio of offload targets in the bottom right corner
        font = pygame.font.SysFont('Arial', 18, bold=True)
        # 计算各卸载目标的总数 en: Calculate total counts for each offload target
        totals = [0] * len(OFFLOAD_TARGETS)
        for counts in unload_count.values():
            for i, count in enumerate(counts):
                totals[i] += count
        
        total_offload = sum(totals)
        if total_offload == 0:
            ratios = [0] * len(OFFLOAD_TARGETS)
        else:
            ratios = [count / total_offload for count in totals]

        # 创建一个半透明的背景矩形 en: Create a semi-transparent background rectangle
        bg_rect = pygame.Rect(SCREEN_WIDTH - 300, SCREEN_HEIGHT - 140, 290, 130)
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
        bg_surface.set_alpha(180)
        bg_surface.fill(WHITE)
        screen.blit(bg_surface, bg_rect)

        # 绘制卸载比例 en: Draw offload ratios
        y_offset = SCREEN_HEIGHT - 130
        for i, (target, ratio) in enumerate(zip(OFFLOAD_TARGETS, ratios)):
            text = font.render(f"{target}: {ratio:.2%} ({totals[i]})", True, OFFLOAD_COLORS[i])
            screen.blit(text, (SCREEN_WIDTH - 290, y_offset))
            y_offset += 25

        # 绘制边框 en: Draw border around the background rectangle
        pygame.draw.rect(screen, BLACK, bg_rect, 2)

    def simulate_trajectory():
        """模拟并可视化无人机轨迹"""
        # en: Simulate and visualize UAV trajectories
        state = env._get_obs()  # 获取初始状态 en: Get initial state
        trajectories = [[] for _ in range(env.N)]  # 存储每个无人机的轨迹 en: Store trajectories for each UAV
        # 获取每个无人机的初始位置并添加到轨迹中 en: Get initial positions of each UAV and add to trajectories
        for i in range(env.N):
            pos = env.uavs[i]['position']
            screen_x = int(pos[0] * CELL_SIZE + CELL_SIZE//2)
            screen_y = int(pos[1] * CELL_SIZE + CELL_SIZE//2)
            trajectories[i].append((screen_x, screen_y))
        
        visit_count = np.zeros((GRID_SIZE, GRID_SIZE))  # 存储每个网格单元的访问次数 en: Store visit counts for each grid cell
        unload_count = {}  # 存储每个位置的卸载次数，格式: {(x, y): [本地, 基站, HAPS, LEO, 云端]} en: Store offload counts for each position, format: {(x, y): [Local, Base Station, HAPS, LEO, Cloud]}
        
        # 新增：存储两点之间路径的经过次数 en: New: Store the count of paths between two points
        path_count = {}  # 格式: {((x1, y1), (x2, y2)): count}
        
        done = False
        
        # 主模拟循环
        while not done:
            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # 从训练好的智能体获取动作
            actions = [agent.take_action(state[n]) for n in range(env.N)]
                    
            # 在环境中执行步骤
            next_state, reward, done = env.step(actions)
            
            # 额外的卸载动作决策
            offload_data = env.get_obs_2()  # 状态是一个图数据
            offload_actions = offload_agent.take_action(offload_data)
            next_offload_data, offload_reward, done = env.step_offload(offload_actions)
            
            
            state = next_state
            
            # 计算当前平均不确定度
            current_avg_uncertainty = np.mean(env.uncertainty_matrix)

            # 存储当前位置
            for i, uav in enumerate(env.uavs):
                # 只有未结束的无人机才会被计入访问统计
                if uav['done']:
                    continue
                    
                pos = uav['position']
                # 将网格位置转换为屏幕坐标
                screen_x = int(pos[0] * CELL_SIZE + CELL_SIZE//2)
                screen_y = int(pos[1] * CELL_SIZE + CELL_SIZE//2)
                current_pos = (screen_x, screen_y)
                
                # 如果移动到新位置，记录路径
                if len(trajectories[i]) > 0:
                    prev_pos = trajectories[i][-1]
                    if prev_pos != current_pos:
                        # 为了确保路径键的唯一性，无论方向如何，总是将较小坐标放在前面
                        path_key = tuple(sorted([prev_pos, current_pos]))
                        path_count[path_key] = path_count.get(path_key, 0) + 1
                        trajectories[i].append(current_pos)
                
                # 更新访问次数
                grid_x = int(pos[0])
                grid_y = int(pos[1])
                visit_count[grid_x, grid_y] += 1

                # 更新卸载次数
                if current_pos not in unload_count:
                    unload_count[current_pos] = [0, 0, 0, 0, 0]  # 5种卸载方式
                
                # 记录当前卸载行为
                offload_action = offload_actions[i] if i < len(offload_actions) else 0
                unload_count[current_pos][offload_action] += 1
            
            # 清屏并绘制
            screen.fill(WHITE)
            draw_uncertainty_matrix(env.uncertainty_matrix)
            draw_grid()
            draw_wind_field(env.wind_u, env.wind_v)  # 绘制风场
            # draw_visit_count(visit_count)  # 可选：不再显示访问次数，减少视觉干扰
            
            # 绘制轨迹
            colors = [RED, BLUE, GREEN, YELLOW]  # 不同无人机的不同颜色
            for i, traj in enumerate(trajectories):
                if len(traj) > 1:
                    for j in range(len(traj) - 1):
                        start_pos = traj[j]
                        end_pos = traj[j + 1]
                        
                        # 计算该路径的经过次数并据此调整宽度
                        path_key = tuple(sorted([start_pos, end_pos]))
                        path_frequency = path_count.get(path_key, 1)
                        width = min(1 + path_frequency, 5)  # 根据路径经过次数调整宽度
                        
                        pygame.draw.line(screen, colors[i % len(colors)], start_pos, end_pos, width)
            
            # draw_unload_count(unload_count)
            draw_avg_uncertainty(current_avg_uncertainty)
            draw_offload_ratio(unload_count)
            
            # 绘制当前无人机位置
            for i, uav in enumerate(env.uavs):
                if uav['done']:
                    continue
                    
                pos = uav['position']
                screen_x = int(pos[0] * CELL_SIZE + CELL_SIZE//2)
                screen_y = int(pos[1] * CELL_SIZE + CELL_SIZE//2)
                pygame.draw.circle(screen, colors[i % len(colors)], (screen_x, screen_y), 10)
            
            pygame.display.flip()
            pygame.time.wait(0)  # 添加延迟以便于可视化
        
        pygame.time.wait(3000)  # 显示最终结果3秒
        pygame.quit()

        return current_avg_uncertainty

    # 运行模拟
    return simulate_trajectory()