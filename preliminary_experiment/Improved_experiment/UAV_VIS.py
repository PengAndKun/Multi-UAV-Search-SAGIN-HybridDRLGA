import pygame
import sys 
import numpy as np

def visualize_trajectory(agent, env):
    # 初始化 Pygame en: Initialize pygame
    pygame.init()

    # 屏幕尺寸 en: Define screen dimensions
    SCREEN_WIDTH = 800
    SCREEN_HEIGHT = 800

    # 颜色 en: Define colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)
    GREEN = (0, 255, 0)
    YELLOW = (255, 255, 0)
    PURPLE = (128, 0, 128)  # 添加紫色用于显示不确定度  en: Define purple color for uncertainty display

    # 初始化屏幕 en: Initialize the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("无人机轨迹")

    # 网格设置 en: Grid settings
    GRID_SIZE = 10
    CELL_SIZE = SCREEN_WIDTH // GRID_SIZE

    def draw_grid():
        """在屏幕上绘制网格"""
        #en: Draw grid lines on the screen
        for i in range(GRID_SIZE):
            # 绘制垂直线 en: Draw vertical lines
            pygame.draw.line(screen, BLACK, (i * CELL_SIZE, 0), (i * CELL_SIZE, SCREEN_HEIGHT))
            # 绘制水平线 en: Draw horizontal lines
            pygame.draw.line(screen, BLACK, (0, i * CELL_SIZE), (SCREEN_WIDTH, i * CELL_SIZE))

    def draw_uncertainty_matrix(uncertainty_matrix):
        """在屏幕上绘制不确定度矩阵"""
        #en: Draw the uncertainty matrix on the screen
        font = pygame.font.Font(None, 24)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                uncertainty = uncertainty_matrix[i, j]
                text = font.render(f"{uncertainty:.2f}", True, BLACK)
                screen.blit(text, (i * CELL_SIZE + 5, j * CELL_SIZE + 5))

    def draw_visit_count(visit_count):
        """在屏幕上绘制访问次数"""
        #en: Draw visit count on the screen
        font = pygame.font.Font(None, 24)
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                count = visit_count[i, j]
                if count > 0:
                    text = font.render(f"{count}", True, BLUE)
                    screen.blit(text, (i * CELL_SIZE + CELL_SIZE//2, j * CELL_SIZE + CELL_SIZE//2))

    def draw_stay_count(stay_count):
        """在屏幕上绘制停留次数"""
        #en: Draw stay count on the screen
        font = pygame.font.Font(None, 24)
        for (x, y), count in stay_count.items():
            if count > 1:
                text = font.render(f"{count}", True, GREEN)
                screen.blit(text, (x - 10, y - 10))

    def draw_colored_grid(visit_count):
        """根据访问次数绘制带颜色的网格"""
        #en: Draw a colored grid based on visit count
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                count = visit_count[i, j]
                if count > 0:
                    color_intensity = min(255, count * 20)  # 根据需要调整乘数 en: Adjust the multiplier as needed
                    color = (255, 255 - color_intensity, 0)
                    pygame.draw.rect(screen, color, (i * CELL_SIZE, j * CELL_SIZE, CELL_SIZE, CELL_SIZE))

    def draw_unload_count(unload_count):
        """在屏幕上绘制卸载次数"""
        #en: Draw unload count on the screen
        font = pygame.font.Font(None, 24)
        for (x, y), counts in unload_count.items():
            local_count, base_count = counts
            text = font.render(f"L:{local_count} B:{base_count}", True, BLACK)
            screen.blit(text, (x - 20, y - 20))

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
        
        # 绘制当前平均不确定度 en: Render the current average uncertainty
        text_current = font.render(f"average uncertainty: {avg_uncertainty:.4f}", True, PURPLE)
        screen.blit(text_current, (SCREEN_WIDTH - 290, 20))
        
        # 绘制边框 en: Draw the border around the background rectangle
        pygame.draw.rect(screen, BLACK, bg_rect, 2)

    def draw_unload_ratio(unload_count):
        """在右下角绘制本地卸载和基地卸载的比例"""
        #en: Draw the local and base unload ratios in the bottom right corner
        font = pygame.font.SysFont('Arial', 24, bold=True)
        total_local_unload = sum(counts[0] for counts in unload_count.values())
        total_base_unload = sum(counts[1] for counts in unload_count.values())
        total_unload = total_local_unload + total_base_unload
        if total_unload > 0:
            local_ratio = total_local_unload / total_unload
            base_ratio = total_base_unload / total_unload
        else:
            local_ratio = 0
            base_ratio = 0

        # 创建一个半透明的背景矩形 en: Create a semi-transparent background rectangle
        bg_rect = pygame.Rect(SCREEN_WIDTH - 300, SCREEN_HEIGHT - 60, 290, 50)
        bg_surface = pygame.Surface((bg_rect.width, bg_rect.height))
        bg_surface.set_alpha(180)
        bg_surface.fill(WHITE)
        screen.blit(bg_surface, bg_rect)

        # 绘制本地卸载比例 en: Render the local unload ratio
        text_local = font.render(f"Local: {local_ratio:.2%} ", True, PURPLE)
        screen.blit(text_local, (SCREEN_WIDTH - 290, SCREEN_HEIGHT - 50))

        # 绘制基地卸载比例 en: Render the base unload ratio
        text_base = font.render(f"Base: {base_ratio:.2%}", True, PURPLE)
        screen.blit(text_base, (SCREEN_WIDTH - 150, SCREEN_HEIGHT - 50))

        # 绘制边框 en: Draw the border around the background rectangle
        pygame.draw.rect(screen, BLACK, bg_rect, 2)

    def simulate_trajectory():
        """模拟并可视化无人机轨迹"""
        #en: Simulate and visualize UAV trajectories
        state = env.reset()
        trajectories = [[] for _ in range(env.N)]  # 存储每个无人机的轨迹 en: Store trajectories for each UAV
        # 获取每个无人机的初始位置并添加到轨迹中
        # Initialize trajectories with the starting positions of each UAV
        for i in range(env.N):
            pos = env.uavs[i]['position']
            screen_x = int(pos[0] * CELL_SIZE + CELL_SIZE//2)
            screen_y = int(pos[1] * CELL_SIZE + CELL_SIZE//2)
            trajectories[i].append((screen_x, screen_y))
        
        stay_count = {}  # 存储每个位置的停留次数 en: Store stay count for each position
        visit_count = np.zeros((GRID_SIZE, GRID_SIZE))  # 存储每个网格单元的访问次数 en: Store visit count for each grid cell
        unload_count = {}  # 存储每个位置的卸载次数 en: Store unload count for each position
        
        # 新增：存储两点之间路径的经过次数 en: New: Store the count of paths between two points
        path_count = {}  # 格式: {((x1, y1), (x2, y2)): count}
        
        done = False
        
        # 主模拟循环 #en: Main simulation loop
        while not done:
            # 处理事件 #en: Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # 从训练好的智能体获取动作 #en: Get actions from the trained agent
            agent.epsilon = 0.1  # 设置epsilon为0以执行贪婪动作 #en: Set epsilon to 0 for greedy actions
            actions = [agent.take_action(state[n]) for n in range(env.N)]
                    
            # 在环境中执行步骤 #en: Step in the environment
            next_state, reward, done = env.step(actions)
            state = next_state
            
            # 计算当前平均不确定度 #en: Calculate the current average uncertainty
            current_avg_uncertainty = np.mean(env.uncertainty_matrix)

            # 存储当前位置    #en: Store current positions
            for i, uav in enumerate(env.uavs):
                # 只有未结束的无人机才会被计入停留和访问统计 #en: Only count stay and visit statistics for UAVs that are not done
                if uav['done']:
                    continue
                    
                pos = uav['position']
                # 将网格位置转换为屏幕坐标 #en: Convert grid position to screen coordinates
                screen_x = int(pos[0] * CELL_SIZE + CELL_SIZE//2)
                screen_y = int(pos[1] * CELL_SIZE + CELL_SIZE//2)
                current_pos = (screen_x, screen_y)
                
                # 检查无人机是否停留在相同位置 #en: Check if the UAV is staying at the same position
                if trajectories[i] and trajectories[i][-1] == current_pos:
                    stay_count[current_pos] = stay_count.get(current_pos, 0) + 1
                else:
                    # 如果移动到新位置，记录路径 并更新停留次数 #en: If moved to a new position, record the path and update stay count
                    if len(trajectories[i]) > 0:
                        prev_pos = trajectories[i][-1]
                        # 为了确保路径键的唯一性，无论方向如何，总是将较小坐标放在前面 #en: Ensure path keys are unique regardless of direction
                        path_key = tuple(sorted([prev_pos, current_pos]))
                        path_count[path_key] = path_count.get(path_key, 0) + 1
                    
                    trajectories[i].append(current_pos)
                    stay_count[current_pos] = 1
                
                # 更新访问次数 #en: Update visit count
                grid_x = int(pos[0])
                grid_y = int(pos[1])
                visit_count[grid_x, grid_y] += 1

                # 更新卸载次数 #en: Update unload count
                if current_pos not in unload_count:
                    unload_count[current_pos] = [0, 0]
                if uav['offload'] == 0:  # 本地卸载的条件  en: Local offloading condition
                    unload_count[current_pos][0] += 1
                elif uav['offload'] == 1:  # 基站卸载的条件 #en: Base offloading condition
                    unload_count[current_pos][1] += 1
            
            # 清屏并绘制 内容 #en: Clear the screen and draw the content
            screen.fill(WHITE)
            draw_colored_grid(visit_count)
            draw_grid()
            draw_uncertainty_matrix(env.uncertainty_matrix)
            draw_visit_count(visit_count)
            draw_stay_count(stay_count)
            draw_unload_count(unload_count)
            
            # 绘制轨迹 #en: Draw trajectories
            colors = [RED, BLUE, GREEN, YELLOW]  # 不同无人机的不同颜色 #en: Different colors for different UAVs
            for i, traj in enumerate(trajectories):
                if len(traj) > 1:
                    for j in range(len(traj) - 1):
                        start_pos = traj[j]
                        end_pos = traj[j + 1]
                        
                        # 计算该路径的经过次数并据此调整宽度 #en: Calculate the frequency of the path and adjust the width accordingly
                        path_key = tuple(sorted([start_pos, end_pos]))
                        path_frequency = path_count.get(path_key, 1)
                        width = min(1 + path_frequency, 10)  # 根据路径经过次数调整宽度 #en: Adjust width based on path frequency
                        
                        pygame.draw.line(screen, colors[i % len(colors)], start_pos, end_pos, width)
            
            # 在右上角绘制平均不确定度信息 #en: Draw average uncertainty information in the top right corner
            draw_avg_uncertainty(current_avg_uncertainty)
            draw_unload_ratio(unload_count)  # 添加这一行调用绘制卸载比例的函数     en: Add this line to draw unload ratio
            
            # 绘制当前无人机位置和停留圆圈 #en: Draw current UAV positions and stay circles
            for i, uav in enumerate(env.uavs):
                if uav['done']:
                    continue
                    
                pos = uav['position']
                screen_x = int(pos[0] * CELL_SIZE + CELL_SIZE//2)
                screen_y = int(pos[1] * CELL_SIZE + CELL_SIZE//2)
                pygame.draw.circle(screen, colors[i % len(colors)], (screen_x, screen_y), 10)
                
                # 如果无人机停留在相同位置，则绘制停留圆圈 #en: If the UAV is staying at the same position, draw a stay circle
                if (screen_x, screen_y) in stay_count and stay_count[(screen_x, screen_y)] > 1:
                    radius = 12 + stay_count[(screen_x, screen_y)]
                    pygame.draw.circle(screen, GREEN, (screen_x, screen_y), radius, width=2)
            
            pygame.display.flip()
            pygame.time.wait(50)  # 添加延迟以便于可视化 #en: Add a delay for visualization
        
        # print(f"第{u+1}次平均不确定度: {current_avg_uncertainty:.6f}")
        pygame.time.wait(500)
        pygame.quit()

        return current_avg_uncertainty

    # 运行模拟 #en: Run the simulation
    return simulate_trajectory()

