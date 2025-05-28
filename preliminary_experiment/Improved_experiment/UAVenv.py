import numpy as np

class UAVEnv:
    def __init__(self,N=2):
        # 环境参数 en:environment parameters
        self.H = 20  # UAV的高度（米） en: UAV height
        self.B = 0.1e6  # 每个通道的带宽（Hz） en: Bandwidth of each channel
        self.h0 = -50  # 信道功率增益（dB） en: Channel power gain in dB
        self.X = 200  # 搜索区域长度（米） en: Length of search area
        self.Y = 200  # 搜索区域宽度（米） en: Width of search area
        self.M = 1  # UAV的质量（kg） en: UAV mass
        self.L = 4  # 通道数量 en: Number of channels
        self.Lx, self.Ly = 10, 10  # 搜索区域的网格划分 en: Grid division of search area
        self.N = N  # UAV数量 en: Number of UAVs
        # self.N = 4  # UAV数量
        self.V = 5  # UAV的飞行速度（m/s）     en: UAV flight speed
        self.f_ln = 1e9  # UAV的处理能力（cycles/s） en: UAV processing capability
        self.Clx_ly = 2000  # 任务处理密度（cycles/bit） en: Task processing density
        self.kn = 1e-24  # 切换电容系数 (en: Switching capacitance coefficient)
        self.Pn = 20  # UAV的传输功率（dBm） en: UAV transmission power
        self.sigma2 = 1e-9  # 白高斯噪声功率 (en: White Gaussian noise power)
        self.initial_energy = 1e4  # UAV的初始能量（J） en: Initial energy of UAVs
        self.max_search_time = 360  # 最大搜索时间（秒） en: Maximum search time
        self.delta = 0.4  # 目标检测精度 en: Target detection accuracy
        
        # 8个可能的移动方向（x, y） en: 8 possible movement directions (x, y)
        self.directions = [
            (0, 1),    # 向上  en: Up
            (1, 1),    # 右上  en: Up-right
            (1, 0),    # 向右  en: Right
            (1, -1),    # 右下 en: Down-right
            (0, -1),   # 向下  en: Down
            (-1, -1),  # 左下 en: Down-left
            (-1, 0),   # 向左 en: Left
            (-1, 1),   # 左上 en: Up-left
        ]
        self.gird_position = [
            (0, 0), 
            (self.Lx-1, self.Ly-1),
            (self.Lx-1, 0), 
            (0, self.Ly-1)
        ]
        # UAVs状态 en: UAVs state
        self.uavs = [self._initialize_uav(u) for u in range(self.N)]

        # 初始化不确定度矩阵，10x10，初始全为1 en: Initialize uncertainty matrix, 10x10, all initialized to 1
        self.uncertainty_matrix = np.ones((self.Lx, self.Ly))
        
        # 剩余搜索时间 en: Remaining search time
        self.remaining_time = self.max_search_time
        
        #一次飞行的时间 en: Time for one flight
        self.T = np.sqrt(2) * (self.X/self.Lx) / self.V
        
        #一次飞行消耗的最大能量 en: Maximum energy consumed in one flight
        self.flight_energy = 0.5 * self.M * (self.V ** 2) * self.T
        
        #动作空间大小(8个方向，N个无人机) en: Action space size (8 directions, N UAVs)
        self.action_dim = 8
        
        #状态空间大小 en: State space size
        self.state_dim = len(self._get_obs()[0])
        
    def _initialize_uav(self,u):
        """初始化每个UAV的位置和状态，随机选择4个角中的一个作为初始位置"""
        #en: Initialize each UAV's position and state, randomly select one of the 4 corners as the initial position
        initial_position = self.gird_position[u]
            
        return {
            'position': initial_position,  # UAV初始位置在网格中 en: UAV initial position in the grid
            'energy': self.initial_energy,  # UAV初始能量   en: UAV initial energy
            'done': False,  #UAV初始done en: UAV initial done status
            'offload': 0 #UAV初始卸载默认为0   en: UAV initial offload status
        }

    def _update_uncertainty_matrix(self, uav_position):
        """根据UAV的当前位置直接更新不确定度矩阵"""
        #en: Update the uncertainty matrix based on the UAV's current position
        x, y = uav_position
        self.uncertainty_matrix[x, y] = self.uncertainty_matrix[x, y] * (1-self.delta)  # 每次访问减少该点的不确定度 en: Reduce the uncertainty of the point by delta each time it is visited
        
    def step(self, actions):
        """根据给定的动作更新环境, action_index为0到7的整数，表示8个方向中的一个"""
        #en: Update the environment based on the given actions, action_index is an integer from 0 to 7, representing one of the 8 directions
        # action = sum(actions[n] * (16 ** n) for n in range(self.N))  # 将 actions 转换为单个 action
        μ=20000 #任务期望大小
        #无人机卸载到GBS的数量 en: Number of UAVs offloading to GBS
        X_t = 0 
        # for u in range(self.N):
        #     if not self.uavs[u]['done']:  # 只计算未结束的无人机
        #         # X_t += (action//(16**u)) % 2
        #         X_t += actions[u] % 2
        value_old = sum(sum(self.uncertainty_matrix))  #记录初始不确定度 en: Record initial uncertainty
        reward = [0] * self.N
        for u, uav in enumerate(self.uavs):


            offload = 0
            action_index = actions[u] 
            
            uav['offload'] = offload
            ε = 0  #惩罚 en: Penalty
            if uav['done'] == True:
                continue
            
            # 根据action_index选择方向 en: Select direction based on action_index
            direction = np.array(self.directions[action_index])
            
            # 更新无人机在网格中的位置 en: Update UAV's position in the grid
            new_position = tuple(uav['position'] + direction)   

            # 确保UAV不越界（保持在网格内） en: Ensure UAV does not go out of bounds (stay within the grid)
            if new_position[0]>=self.Lx or new_position[0]<0 or new_position[1]>= self.Ly or new_position[1]<0:
                ε += 0.05
            else:
                # 检查是否是首次访问该位置（不确定度为1） en: Check if it is the first visit to this position (uncertainty is 1)
                x, y = new_position
                is_new_area = self.uncertainty_matrix[x, y] == 1
                # 检查新位置是否已经有其他无人机 en: Check if the new position already has other UAVs
                if any(new_position == other_uav['position'] for other_uav in self.uavs if other_uav != uav):
                    ε += 0.05  # 如果新位置已经有其他无人机，增加惩罚 en: If the new position already has other UAVs, increase penalty
                else:
                    # 如果是首次访问该区域，添加额外奖励 en: If it is the first visit to this area, add extra reward
                    if is_new_area:
                        ε -= 0.01  # 减少惩罚值相当于增加奖励 en: Reduce penalty value, equivalent to increasing reward
                    uav['position'] = new_position  # 更新位置 en: Update position
                
            # 判断飞行产生的能耗 en: Determine the energy consumed during flight
            uav['energy'] -= (1+action_index % 2) * self.flight_energy / 2
            
            #计算与基站的距离 en: Calculate the distance to the GBS
            d2 = (np.linalg.norm(uav['position']-np.array([4.5, 4.5]))*self.X/self.Lx)**2+self.H**2
            
            h_ngt=10**(self.h0/10)/d2 #信道功率增益 en: Channel power gain

            #计算卸载本地或者卸载到GBS产生的能耗 en: Calculate the energy consumed by local processing or offloading to GBS
            uav['energy'] -= (1-offload)*self.kn * μ * self.Clx_ly * self.f_ln**2 + offload*X_t*10**(self.Pn/10-3)*μ/(self.L*self.B* np.log2(1+10**(self.Pn/10-3)*h_ngt/self.sigma2))

            # 更新不确定度矩阵 en: Update the uncertainty matrix
            self._update_uncertainty_matrix(uav['position'])
            value_new = sum(sum(self.uncertainty_matrix))  #记录新的不确定度 en: Record new uncertainty
            reward[u] = (value_old - value_new)/100 - ε  # 奖励 = 平均不确定度减少-惩罚 en: Reward = Average uncertainty reduction - Penalty
            value_old = value_new
            
            #判断能否返航 en: Determine if the UAV can return
            if uav['energy'] - 0.5 * self.M * self.V * np.linalg.norm(uav['position']-np.array(self.gird_position[u]))*self.X/self.Lx <=0:
                uav['done'] = True
                uav['energy'] = 0
                uav['position'] = (0,0)
                

        
        # 更新剩余搜索时间 en: Update remaining search time
        self.remaining_time -= self.T
        # self.remaining_time -= 1
        
        # 检测目标或完成任务 en: Check if the target is detected or the task is completed
        done = self._check_done()

        # 返回状态：UAV位置、剩余电量、剩余搜索时间、不确定矩阵 en: Return state: UAV positions, remaining energy, remaining search time, uncertainty matrix
        state = self._get_obs()

        return state, reward, done
    
    def _check_done(self):
        """检查是否达到终止条件"""
        #en: Check if termination conditions are met
        # 当所有UAV能量耗尽或超出搜索时间时，任务结束 en: When all UAVs run out of energy or exceed the search time, the task ends
        if all(uav['done'] == True for uav in self.uavs) or self.remaining_time <= 0:
            return True
        return False
        
    def _get_obs(self):
        state = []
        # 为每个UAV添加周围5*5矩阵的不确定度信息 en: Add uncertainty information of the surrounding 5x5 matrix for each UAV
        for uav in self.uavs:
            x, y = uav['position']
            state_i = np.zeros((5, 5))  # 初始化5x5矩阵 en: Initialize 5x5 matrix
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    nx, ny = x + dx, y + dy
                    # 如果位置在网格范围内，添加该位置的不确定度 en: If the position is within the grid range, add the uncertainty of that position
                    if 0 <= nx < self.Lx and 0 <= ny < self.Ly:
                        # 检查该位置是否有其他无人机 en: Check if there are other UAVs at this position
                        if any((nx, ny) == other_uav['position'] for other_uav in self.uavs if other_uav != uav):
                            state_i[dx + 2, dy + 2] = 0  # 如果有其他无人机，设置为0 en: If there are other UAVs, set to 0
                        else:
                            state_i[dx + 2, dy + 2] = self.uncertainty_matrix[nx, ny]
                    else:
                        # 如果超出网格范围，添加一个特殊值表示不可用 en: If out of grid range, add a special value to indicate unavailable
                        state_i[dx + 2, dy + 2] = 0
            state.append(state_i)
        return np.array(state)

        return state
        
    def reset(self):
        """重置环境"""
        #en: Reset environment
        self.uavs = [self._initialize_uav(u) for u in range(self.N)] #初始化无人机 en: Initialize UAVs
        self.uncertainty_matrix = np.ones((self.Lx, self.Ly))  # 重置不确定度矩阵 en: Reset uncertainty matrix
        # 重置剩余搜索时间 en: Reset remaining search time
        self.remaining_time = self.max_search_time
        state = self._get_obs()
        return state

env = UAVEnv(4)
print(env.reset())
print(env.state_dim)
print(env.action_dim)