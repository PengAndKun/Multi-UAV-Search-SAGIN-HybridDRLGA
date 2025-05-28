import numpy as np

class UAVEnv:
    def __init__(self):
        # 环境参数 en: environment parameters
        self.H = 20  # UAV的高度（米） en: height of UAV
        self.B = 0.1e6  # 每个通道的带宽（Hz）en: bandwidth of each channel
        self.h0 = -50  # 信道功率增益（dB）en: channel power gain in dB
        self.X = 200  # 搜索区域长度（米）en: length of search area
        self.Y = 200  # 搜索区域宽度（米）en: width of search area
        self.M = 1  # UAV的质量（kg）en: mass of UAV
        self.L = 4  # 通道数量 (en: number of channels)
        self.Lx, self.Ly = 10, 10  # 搜索区域的网格划分 (en: grid division of search area)
        self.N = 2  # UAV数量 (en: number of UAVs)
        # self.N = 4  # UAV数量 (en: number of UAVs)
        self.V = 5  # UAV的飞行速度（m/s） en: speed of UAV
        self.f_ln = 1e9  # UAV的处理能力（cycles/s） en: processing capability of UAV
        self.Clx_ly = 2000  # 任务处理密度（cycles/bit） en: task processing density
        self.kn = 1e-24  # 切换电容系数 (en: switching capacitance coefficient)
        self.Pn = 20  # UAV的传输功率（dBm） en: transmission power of UAV
        self.sigma2 = 1e-9  # 白高斯噪声功率 (en: white Gaussian noise power)
        self.initial_energy = 1e4  # UAV的初始能量（J） en: initial energy of UAV
        self.max_search_time = 360  # 最大搜索时间（秒） en: maximum search time in seconds
        self.delta = 0.4  # 目标检测精度 (en: target detection accuracy)
        
        # 8个可能的移动方向（x, y） en: 8 possible movement directions (x, y)
        self.directions = [
            (0, 1),    # 向上  en: up
            (1, 1),    # 右上  en: up-right
            (1, 0),    # 向右  en: right
            (1, -1),    # 右下 en: down-right
            (0, -1),   # 向下 en: down
            (-1, -1),  # 左下 en: down-left
            (-1, 0),   # 向左 en: left
            (-1, 1),   # 左上 en: up-left
        ]
        self.gird_position = [
            (0, 0), 
            (self.Lx-1, self.Ly-1),
            (self.Lx-1, 0), 
            (0, self.Ly-1)
        ]
        # UAVs状态
        # en: UAVs state
        self.uavs = [self._initialize_uav(u) for u in range(self.N)]

        # 初始化不确定度矩阵，10x10，初始全为1
        # en: initialize uncertainty matrix, 10x10, all set to 1
        self.uncertainty_matrix = np.ones((self.Lx, self.Ly))
        
        # 剩余搜索时间
        # en: remaining search time
        self.remaining_time = self.max_search_time
        
        #一次飞行的时间
        # en: time for one flight
        self.T = np.sqrt(2) * (self.X/self.Lx) / self.V
        
        #一次飞行消耗的最大能量
        # en: maximum energy consumed in one flight
        self.flight_energy = 0.5 * self.M * (self.V ** 2) * self.T
        
        #动作空间大小(8个方向，2个选择，N个无人机)
        # en: action space size (8 directions, 2 choices, N UAVs)
        self.action_dim = 16 ** self.N
        
        #状态空间大小(不确定矩阵10*10，无人机位置电量，一个时间限制)
        self.state_dim = 10 * 10+ 1 + 3 * self.N
        
    def _initialize_uav(self,u):
        """初始化每个UAV的位置和状态，随机选择4个角中的一个作为初始位置"""
        # en: Initialize each UAV's position and state, randomly select one of the 4 corners as the initial position
        initial_position = self.gird_position[u]
            
        return {
            'position': initial_position,  # UAV初始位置在网格中 en: initial position of UAV in grid
            'energy': self.initial_energy,  # UAV初始能量 en: initial energy of UAV
            'done': False,  #UAV初始done en: UAV initial done status
            'offload': 0 #UAV初始卸载默认为0  en: UAV initial offload status is 0
        }

    def _update_uncertainty_matrix(self, uav_position):
        """根据UAV的当前位置直接更新不确定度矩阵"""
        #"""Update the uncertainty matrix based on the UAV's current position"""
        x, y = uav_position
        self.uncertainty_matrix[x, y] = self.uncertainty_matrix[x, y] * (1-self.delta)  # 每次访问减少该点的不确定度 en: reduce uncertainty at the point by delta each visit
        
    def step(self, action):
        """根据给定的动作更新环境, action_index为0到7的整数，表示8个方向中的一个"""
        # en: Update the environment based on the given action, action_index is an integer from 0 to 7, representing one of the 8 directions
        μ=20000 #任务期望大小 en: expected size of task
        #无人机卸载到GBS的数量
        # en: number of UAVs offloading to GBS
        X_t = 0 
        for u in range(self.N):
            if not self.uavs[u]['done']:  # 只计算未结束的无人机 en: only count UAVs that are not done
                X_t += (action//(16**u)) % 2
        value_old = sum(sum(self.uncertainty_matrix))  #记录初始不确定度 en: record initial uncertainty
        ε = 0  #总惩罚 en: total penalty
        for u, uav in enumerate(self.uavs):
            action_u = action % 16
            action = action//16
            offload = action_u % 2
            action_index = action_u // 2
            
            uav['offload'] = offload
            
            if uav['done'] == True:
                continue
            
            # 根据action_index选择方向
            # en: Select direction based on action_index
            direction = np.array(self.directions[action_index])
            
            # 更新无人机在网格中的位置
            # en: Update UAV's position in the grid
            new_position = uav['position'] + direction   

            # 确保UAV不越界（保持在网格内）
            # en: Ensure UAV does not go out of bounds (stay within grid)
            if new_position[0]>=self.Lx or new_position[0]<0 or new_position[1]>= self.Ly or new_position[1]<0:
                ε += 0.05
            else:
                uav['position'] = new_position  # 更新位置  en: Update position
                
            # 判断飞行产生的能耗
            # en: Check energy consumption due to flight
            uav['energy'] -= (1+action_index % 2) * self.flight_energy / 2
            
            #计算与基站的距离
            # en: Calculate distance to GBS
            d2 = (np.linalg.norm(uav['position']-np.array([4.5, 4.5]))*self.X/self.Lx)**2+self.H**2
            
            h_ngt=10**(self.h0/10)/d2 #信道功率增益 en: channel power gain

            #计算卸载本地或者卸载到GBS产生的能耗
            # en: Calculate energy consumption for local processing or offloading to GBS
            uav['energy'] -= (1-offload)*self.kn * μ * self.Clx_ly * self.f_ln**2 + offload*X_t*10**(self.Pn/10-3)*μ/(self.L*self.B* np.log2(1+10**(self.Pn/10-3)*h_ngt/self.sigma2))

            # 更新不确定度矩阵
            # en: Update uncertainty matrix
            self._update_uncertainty_matrix(uav['position'])
            
            #判断能否返航
            # en: Check if UAV can return
            if uav['energy'] - 0.5 * self.M * self.V * np.linalg.norm(uav['position']-np.array(self.gird_position[u]))*self.X/self.Lx <=0:
                uav['done'] = True
                uav['energy'] = 0
                uav['position'] = (0,0)

        
        # 更新剩余搜索时间
        # en: Update remaining search time
        self.remaining_time -= self.T
        # self.remaining_time -= 1
        
        # 检测目标或完成任务
        # en: Check for target detection or task completion
        done = self._check_done()
        value_new = sum(sum(self.uncertainty_matrix))  #记录新的不确定度 en: record new uncertainty
        reward = (value_old - value_new)/100 - ε  # 奖励 = 平均不确定度减少-惩罚 en: reward = average uncertainty reduction - penalty

        # 返回状态：UAV位置、剩余电量、剩余搜索时间、不确定矩阵
        # en: Return state: UAV positions, remaining energy, remaining search time, uncertainty matrix
        state = self._get_obs()

        return state, reward, done
    
    def _check_done(self):
        """检查是否达到终止条件"""
        # en: Check if termination conditions are met
        # 当所有UAV能量耗尽或超出搜索时间时，任务结束
        # en: When all UAVs run out of energy or exceed search time, the task ends
        if all(uav['done'] == True for uav in self.uavs) or self.remaining_time <= 0:
            return True
        return False
        
    def _get_obs(self):
        state = [((uav['position'][0]+1) / self.Lx, (uav['position'][1]+1) / self.Ly) for uav in self.uavs]
        state = np.append(state,[uav['energy']/self.initial_energy for uav in self.uavs])
        state = np.append(state,self.remaining_time/self.max_search_time)
        # state = [((uav['position'][0]+1), (uav['position'][1]+1)) for uav in self.uavs]
        # state = np.append(state,[uav['energy'] for uav in self.uavs])
        # state = np.append(state,self.remaining_time)
        state = np.append(state,np.ravel(self.uncertainty_matrix))
        return state
        
    def reset(self):
        """重置环境"""
        # en: Reset the environment
        self.uavs = [self._initialize_uav(u) for u in range(self.N)] #初始化无人机 en: reinitialize UAVs
        self.uncertainty_matrix = np.ones((self.Lx, self.Ly))  # 重置不确定度矩阵 en: reset uncertainty matrix
        # 重置剩余搜索时间
        # en: reset remaining search time
        self.remaining_time = self.max_search_time
        state = self._get_obs()
        return state
