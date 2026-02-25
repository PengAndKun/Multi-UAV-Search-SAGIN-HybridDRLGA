import numpy as np
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from Our_experiment.HCSAC.wind import extract_wind_subregion  #导入函数，方便后面不断更新新地图
import os.path
import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Batch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import collections
import random

class UAVEnv:
    def __init__(self,N=2):
        # 环境参数
        self.H = 200  # UAV的高度（米） en: UAV height
        self.B = 10e6  # 每个通道的带宽（Hz） en: Bandwidth per channel
        self.h0 = -30  # 信道功率增益（dB） en: Channel power gain
        self.X = 1e4  # 搜索区域长度（米） en: Length of search area
        self.Y = 1e4  # 搜索区域宽度（米） en: Width of search area
        self.M = 1  # UAV的质量（kg） en: UAV mass
        self.L = 64  # 通道数量 en: Number of channels
        self.Lx, self.Ly = 20, 20  # 搜索区域的网格划分 (en: Grid size of search area)
        self.grid_cell_size_m = self.X / self.Lx  # 单格边长（米） en: Grid cell size in meters (500m for 20x20 over 10km)
        self.N = N  # UAV数量 (en: Number of UAVs)
        self.V = 20  # UAV的飞行方向上的速度（m/s）,真实无人机的速度 en: UAV speed
        self.f_ln = 2e9  # UAV的处理能力（cycles/s） en: UAV processing capability
        self.f_B = 1e11  # 基站的处理能力（cycles/s） en: Base station processing capability
        self.f_h = 1e10  # HAPS的处理能力（cycles/s） en: HAPS processing capability
        self.f_l = 1e10  # LEO的处理能力（cycles/s） en: LEO processing capability
        self.f_c = 5e11  # 云服务器的处理能力（cycles/s） en: Cloud server processing capability
        self.Clx_ly = 1 #2000  # 任务处理密度（cycles/bit） # en: Task processing density
        self.μ = 8e9  # 任务大小（bit） # en: Task size
        self.kn = 1e-26  # 切换电容系数 (en: Switching capacitance coefficient)
        self.Pn = 23  # UAV的传输功率（dBm） # en: UAV transmission power
        self.sigma2 = 4e-14  # 白高斯噪声功率 （W） # en: White Gaussian noise power
        self.initial_energy = 3.6e5  # UAV的初始能量（J） # en: Initial energy of UAV
        self.max_search_time = 18000  # 最大搜索时间（秒）
        self.delta = 1  # 目标检测精度 en: Target detection accuracy
        
        # 首先尝试相对路径 en: Relative path
        if os.path.exists('OUR_ENV_WITH_WIND_JSON/wind.json'):
            self.file_path = 'OUR_ENV_WITH_WIND_JSON/wind.json'
        # 如果相对路径不存在，则尝试当前目录 en: Current directory
        elif os.path.exists('./wind.json'):
            self.file_path = './wind.json'
        else:
            raise FileNotFoundError("找不到风速数据文件，请确保 'wind.json' 或 'OUR_ENV_WITH_WIND_JSON/wind.json' 存在")
        self.wind_u, self.wind_v, _, _ = extract_wind_subregion(self.file_path,self.Lx)
        
        # 8个可能的移动方向（x, y） en: 8 possible movement directions (x, y)
        self.directions = [
            (0, 1),    # 向上 en: Up
            (1, 1),    # 右上 en: Up-Right
            (1, 0),    # 向右 en: Right
            (1, -1),    # 右下 en: Down-Right
            (0, -1),   # 向下   en: Down
            (-1, -1),  # 左下 en: Down-Left
            (-1, 0),   # 向左 en: Left
            (-1, 1),   # 左上 en: Up-Left
        ]
        self.gird_position = [
            (0, 0), 
            (self.Lx-1, self.Ly-1),
            (self.Lx-1, 0), 
            (0, self.Ly-1),
            (self.Lx // 2, 0),                # 上边中点 en: Top middle
            (self.Lx // 2, self.Ly - 1),      # 下边中点 en: Bottom middle
            (0, self.Ly // 2),                # 左边中点 en: Left middle
            (self.Lx - 1, self.Ly // 2)       # 右边中点 en: Right middle
        ]
        # 定义基站和HAPS在网格中的地面投影位置 en: Define GBS/HAPS ground projection positions in grid coordinates
        self.default_gbs_position = np.array([self.Lx // 2 - 0.5, self.Ly // 2 - 0.5], dtype=np.float64)
        self.default_haps_position = np.array([self.Lx // 2 - 0.5, self.Ly // 2 - 0.5], dtype=np.float64)
        self.gbs_position = self.default_gbs_position.copy()
        self.haps_position = self.default_haps_position.copy()
        self.infra_seed = None
        self.haps_height = 20000   # HAPS的高度 en: HAPS height
        self.leo_height = 200000   # LEO的高度 en: LEO height
        self.d_c = 1000000   # 云服务器的距离 en: Distance to cloud server

        # UAVs状态 en: UAVs state
        self.uavs = [self._initialize_uav(self.gird_position[u],self.gird_position[u]) for u in range(self.N)] #初始化无人机 en: Initialize UAVs

        # 初始化不确定度矩阵，10x10，初始全为1 en: Initialize uncertainty matrix, 20x20, all set to 1
        self.uncertainty_matrix = np.ones((self.Lx, self.Ly))

        # 初始化任务难度矩阵，20x20，取值范围1-4 (也可以看作是地形图) en: Initialize task difficulty matrix, 20x20, values range from 1 to 4 (can also be seen as terrain map)
        self.task_matrix = np.random.randint(1, 5, size=(self.Lx, self.Ly))
        
        # 剩余搜索时间 en: Remaining search time
        self.remaining_time = self.max_search_time
        
        #一次飞行的时间 en: Time for one flight
        self.T = np.sqrt(2) * (self.X/self.Lx) / self.V
        
        #动作空间大小(8个方向，N个无人机) en: Action space size (8 directions, N UAVs)
        self.action_dim = 8
        
        #状态空间大小 en: State space size
        self.state_dim = 15

        #卸载决策空间大小 en: Offloading action space size
        self.offload_action_dim = 5

        #卸载状态空间大小 en: Offloading state space size
        self.offload_state_dim = 6
        
        self.G = self.generate_sagin_graph()  # 初始化图结构 en: Initialize graph structure

    def _sample_infrastructure_positions(self, infra_seed=None):
        """根据基础设施种子设置GBS/HAPS地面投影位置"""
        # en: Set GBS/HAPS ground projection positions based on infrastructure seed.
        self.infra_seed = infra_seed
        if infra_seed is None:
            self.gbs_position = self.default_gbs_position.copy()
            self.haps_position = self.default_haps_position.copy()
            return

        rng = np.random.default_rng(infra_seed)
        low = np.array([0.5, 0.5], dtype=np.float64)
        high = np.array([self.Lx - 0.5, self.Ly - 0.5], dtype=np.float64)
        self.gbs_position = rng.uniform(low=low, high=high)
        self.haps_position = rng.uniform(low=low, high=high)

    def _horizontal_distance_m(self, pos_a, pos_b):
        """网格坐标转换到真实距离后的水平距离（米）"""
        # en: Horizontal distance in meters after converting from grid coordinates.
        delta = (np.array(pos_a, dtype=np.float64) - np.array(pos_b, dtype=np.float64)) * self.grid_cell_size_m
        return float(np.linalg.norm(delta))

    def _squared_link_distance_m(self, uav_pos, endpoint_pos, endpoint_height):
        """三维链路距离平方（米^2）"""
        # en: Squared 3D link distance in meters^2.
        horizontal_distance = self._horizontal_distance_m(uav_pos, endpoint_pos)
        return horizontal_distance ** 2 + float(endpoint_height) ** 2
        
    def _initialize_uav(self, initial_position, destination):
        """初始化无人机的状态"""
        # en: Initialize UAV state
        return {
            'position': initial_position,  # UAV初始位置在网格中 en: UAV initial position in the grid
            'destination': destination, # UAV初始位置在网格中目的地位置 en: UAV initial destination position in the grid
            'energy': self.initial_energy,  # UAV初始能量 en: UAV initial energy
            'done': False,  #UAV初始done en: UAV initial done status
            'offload': 0, #UAV初始卸载默认为0 en: UAV initial offload status
            'link': 0, #UAV初始连接基站默认为0 en: UAV initial link status
        }

    def _update_uncertainty_matrix(self, uav_position):
        """根据UAV的当前位置直接更新不确定度矩阵"""
        #en: Update the uncertainty matrix based on the UAV's current position
        x, y = uav_position
        self.uncertainty_matrix[x, y] = self.uncertainty_matrix[x, y] * (1-self.delta)  # 每次访问减少该点的不确定度 en: Reduce uncertainty at the point by delta each visit

    def _get_real_v(self, direction, wind_vector):
        V_u = np.linalg.norm(direction * (self.V/np.sqrt(2)) - wind_vector) # 无人机的速度向量（在风速的影响下） en: UAV speed vector (affected by wind speed)
        return V_u

    def step(self, actions):
        """根据给定的移动动作更新环境, action_index为0到7的整数，表示8个方向中的一个"""
        # en: Update the environment based on the given movement actions, action_index is an integer from 0 to 7, representing one of the 8 directions
        value_old = sum(sum(self.uncertainty_matrix))  #记录初始不确定度 en: Record initial uncertainty
        reward = [0] * self.N
        for u, uav in enumerate(self.uavs):
            action_index = actions[u] 
            
            ε = 0  #惩罚 en: Penalty ε
            if uav['done'] == True:
                continue
            
            # 根据action_index选择方向 en： Select direction based on action_index
            direction = np.array(self.directions[action_index])
            
            #获取对应位置的风速 en: Get the wind speed at the corresponding position
            x, y = uav['position']
            # 计算风速矢量 en： Calculate wind vector
            wind_vector = np.array([self.wind_u[x, y], self.wind_v[x, y]])

            V_u = self._get_real_v(direction,wind_vector)  # 无人机的速度向量（在风速的影响下）en: UAV speed vector (affected by wind speed)
            
            # 更新无人机在网格中的位置 en: Update UAV position in the grid
            new_position = tuple(uav['position'] + direction)   

            # 确保UAV不越界（保持在网格内） en: Ensure UAV does not go out of bounds (stay within the grid)
            if new_position[0]>=self.Lx or new_position[0]<0 or new_position[1]>= self.Ly or new_position[1]<0:
                ε += 0.05
            else:
                # 检查是否是首次访问该位置（不确定度为1） en: Check if it is the first visit to this position (uncertainty is 1)
                x, y = new_position
                is_new_area = self.uncertainty_matrix[x, y] == 1
                # 检查新位置是否已经有其他无人机 en: Check if the new position already has other UAVs
                if any(new_position == other_uav['position'] for other_uav in self.uavs if other_uav != uav and other_uav['done'] == False):
                    ε += 0.05  # 如果新位置已经有其他无人机，增加惩罚 en: If the new position already has other UAVs, increase penalty ε
                else:
                    # 如果是首次访问该区域，添加额外奖励 en: If it is the first visit to this area, add extra reward
                    if is_new_area:
                        ε -= 0.01  # 减少惩罚值相当于增加奖励 en: Reduce penalty value, equivalent to increasing reward
                    uav['position'] = new_position  # 更新位置 en: Update position
                
            # 判断飞行产生的能耗（在风速的影响下）en: Calculate energy consumption during flight (affected by wind speed)
            e = 0.5 * self.M * (V_u ** 2) * self.T
            uav['energy'] -= e

            # 更新不确定度矩阵 en: Update uncertainty matrix
            self._update_uncertainty_matrix(uav['position'])
            value_new = sum(sum(self.uncertainty_matrix))  #记录新的不确定度 en: Record new uncertainty
            reward[u] = (value_old - value_new)/100 - ε - e/5e5  # 奖励 = 平均不确定度减少-惩罚-电量惩罚 en: Reward = average uncertainty reduction - penalty - energy penalty
            value_old = value_new
            
            #判断能否返航 en: Check if the UAV can return home
            if uav['energy'] - self.return_energy_with_wind(uav) <=0:
                uav['done'] = True     #无人机任务完成 en: UAV task completed
                uav['energy'] = 0      #能量耗尽 en: Energy exhausted
                uav['position'] = uav['destination']    # 返航 e
        
        # 更新剩余搜索时间 en: Update remaining search time
        self.remaining_time -= self.T

        # 检查是否达到终止条件 en: Check if termination conditions are met
        done = self._check_done()

        # 返回状态：UAV的局部15*15的视野，经过检测效果最好 en: Return state: UAV's local 15*15 view, detection effect is best
        state = self._get_obs()

        return state, reward, done
    
    def step_offload(self, actions):
        """处理无人机的卸载决策, 0:本地处理, 1:卸载到GBS, 2:HAPS, 3:LEO, 4:云服务器"""
        # en: Process UAV offloading decisions, 0: local processing, 1: offload to GBS, 2: HAPS, 3: LEO, 4: cloud server
        reward = [0] * self.N

        X_t_B = sum(actions[u]==1 for u in range(self.N) if not self.uavs[u]['done'])  #卸载到GBS的数量 en: Number of UAVs offloading to GBS
        X_t_C = sum(actions[u]==4 for u in range(self.N) if not self.uavs[u]['done'])  #卸载到云服务器的数量 en: Number of UAVs offloading to cloud server
        #无人机卸载到GBS的数量（这里假设卸载到云服务器需要经过GBS） en: Number of UAVs offloading to GBS (assuming offloading to cloud server requires passing through GBS)
        X_t = X_t_B + X_t_C  
        #无人机卸载到HAPS的数量 en: Number of UAVs offloading to HAPS
        X_h = sum(actions[u]==2 for u in range(self.N) if not self.uavs[u]['done'])      
        #无人机卸载到LEO的数量 e
        X_l = sum(actions[u]==3 for u in range(self.N) if not self.uavs[u]['done'])
        
        for u, uav in enumerate(self.uavs):
            offload = actions[u]    
            uav['offload'] = offload
            
            if uav['done'] == True:
                continue

            μ = self.μ  #任务大小 en: Task size
            self.Clx_ly = self.task_matrix[uav['position'][0], uav['position'][1]]  #任务难度 e

            #兼容之前只有两种卸载方式的情况 en: Compatible with previous cases with only two offloading methods
            if offload == 0:
                #计算卸载本地产生的能耗 en: Calculate energy consumption for local processing
                T = μ * self.Clx_ly / self.f_ln
                E = self.kn * μ * self.Clx_ly * self.f_ln**2
                uav['energy'] -= E
                reward[u] = (self.T - T)/1e4 - E/1e4    #返回负的处理时间和能耗 en: Return negative processing time and energy consumption

            elif offload == 1:
                ε = 0
                #计算与基站的距离 en: Calculate distance to base station
                d2 = self._squared_link_distance_m(uav['position'], self.gbs_position, self.H)
                h_ngt=10**(self.h0/10)/d2 #信道功率增益 en: Channel power gain
                #计算卸载到GBS产生的能耗 en: Calculate energy consumption for offloading to GBS
                E = X_t*10**(self.Pn/10-3)*μ/(self.L*self.B* np.log2(1+10**(self.Pn/10-3)*h_ngt/self.sigma2))*(1+1/6)
                uav['energy'] -= E
                #计算处理时间 = 发送时间+处理时间+接收时间 en: Calculate processing time = transmission time + processing time + reception time
                T = X_t * μ/(self.L*self.B* np.log2(1+10**(self.Pn/10-3)*h_ngt/self.sigma2)) *7/6 + X_t_B * μ * self.Clx_ly / self.f_B
                if T > self.T or uav['link'] == 0:
                    # 如果处理时间超过最大时间，或者没有连接基站，增加惩罚，改为本地处理 en: If processing time exceeds maximum time, or not connected to base station, increase penalty and switch to local processing
                    ε = 0.01
                    uav['energy'] += E
                    #改用本地卸载 en: Switch to local processing
                    T = μ * self.Clx_ly / self.f_ln
                    E = self.kn * μ * self.Clx_ly * self.f_ln**2
                    uav['energy'] -= E

                reward[u] = (self.T - T)/1e4 - E/1e4 - ε   #返回负的处理时间和能耗 en: Return negative processing time and energy consumption

            #剩下的卸载方式后面补充 en: The remaining offloading methods will be added later
            elif offload == 2:
                ε = 0
                # 计算与HAPS的距离（水平距离+垂直高度） en: Calculate distance to HAPS (horizontal + vertical)
                d2 = self._squared_link_distance_m(uav['position'], self.haps_position, self.haps_height)
                # HAPS信道功率增益可能与GBS不同，一般空地链路衰减更小 en: HAPS channel power gain may differ from GBS, generally lower free-space path loss
                h_haps = 10**(self.h0/10)/d2  # 可能需要不同的信道模型 en: HAPS channel model may need to be different
                # 计算卸载到HAPS产生的能耗 en: Calculate energy consumption for offloading to HAPS
                E = X_h*10**(self.Pn/10-3)*μ/(self.L*self.B* np.log2(1+10**(self.Pn/10-3)*h_haps/self.sigma2))*(1+1/6)
                uav['energy'] -= E
                # 计算处理时间 en: Calculate processing time
                T = X_h * μ/(self.L*self.B* np.log2(1+10**(self.Pn/10-3)*h_haps/self.sigma2))*7/6 + X_h * μ * self.Clx_ly / self.f_h
                if T > self.T:
                    ε = 0.01
                    uav['energy'] += E
                    #改用本地卸载 en: Switch to local processing
                    T = μ * self.Clx_ly / self.f_ln
                    E = self.kn * μ * self.Clx_ly * self.f_ln**2
                    uav['energy'] -= E
                reward[u] = (self.T - T)/1e4 - E/1e4 - ε    #返回负的处理时间和能耗 en: Return negative processing time and energy consumption

            elif offload == 3:
                ε = 0
                # 计算与LEO的距离 - LEO轨道高度显著高于HAPS en: Calculate distance to LEO - LEO orbit height is significantly higher than HAPS
                d2 = self.leo_height**2
                # LEO卫星信道模型可能需要考虑更多的自由空间路径损耗 en: LEO satellite channel model may need to consider more free-space path loss
                h_leo = 10**(self.h0/10)/d2  # 可能需要更特殊的信道模型 en: LEO channel model may need to be more specialized
                # 计算卸载到LEO产生的能耗 en: Calculate energy consumption for offloading to LEO
                E = X_l*10**(self.Pn/10-3)*μ/(self.L*self.B* np.log2(1+10**(self.Pn/10-3)*h_leo/self.sigma2))*(1+1/6)
                uav['energy'] -= E
                # 计算处理时间 en: Calculate processing time
                T = X_l * μ/(self.L*self.B* np.log2(1+10**(self.Pn/10-3)*h_leo/self.sigma2))*7/6 + X_l * μ * self.Clx_ly / self.f_l
                if T > self.T:
                    ε = 0.01
                    uav['energy'] += E
                    #改用本地卸载 en: Switch to local processing
                    T = μ * self.Clx_ly / self.f_ln
                    E = self.kn * μ * self.Clx_ly * self.f_ln**2
                    uav['energy'] -= E
                reward[u] = (self.T - T)/1e4 - E/1e4 - ε   #返回负的处理时间和能耗 en: Return negative processing time and energy consumption

            elif offload == 4:
                ε = 0
                #计算与基站的距离 en: Calculate distance to base station
                d2 = self._squared_link_distance_m(uav['position'], self.gbs_position, self.H)
                h_ngt=10**(self.h0/10)/d2 #信道功率增益 en: Channel power gain
                #计算卸载到GBS产生的能耗 en: Calculate energy consumption for offloading to GBS
                E = X_t*10**(self.Pn/10-3)*μ/(self.L*self.B* np.log2(1+10**(self.Pn/10-3)*h_ngt/self.sigma2))*(1+1/6)
                uav['energy'] -= E
                #计算处理时间 en: Calculate processing time
                T = X_t * μ/(self.L*self.B* np.log2(1+10**(self.Pn/10-3)*h_ngt/self.sigma2))*7/6 + X_t_C * μ * self.Clx_ly / self.f_c + 2*self.d_c/3e8  #多一个来回传输时间
                if T > self.T or uav['link'] == 0:
                    # 如果处理时间超过最大时间，或者没有连接基站，增加惩罚，改为本地处理 en: If processing time exceeds maximum time, or not connected to base station, increase penalty and switch to local processing
                    ε = 0.01
                    uav['energy'] += E
                    #改用本地卸载 en: Switch to local processing
                    T = μ * self.Clx_ly / self.f_ln
                    E = self.kn * μ * self.Clx_ly * self.f_ln**2
                    uav['energy'] -= E
                reward[u] = (self.T - T)/1e4 - E/1e4 - ε   #返回负的处理时间和能耗 en: Return negative processing time and energy consumption

            #判断能否返航 en: Check if the UAV can return home
            if uav['energy'] - self.return_energy_with_wind(uav) <=0:

                uav['done'] = True     #无人机任务完成 en: UAV task completed
                uav['energy'] = 0      #能量耗尽 en: Energy exhausted
                uav['position'] = uav['destination']    # 返航 en: Return home

        # 卸载是在飞行中进行的，所以不需要计算时间        en: Offloading is done during flight, so no need to calculate time
        self.remaining_time -= 0 
        
        # 检查是否达到终止条件 en: Check if termination conditions are met
        done = self._check_done()

        # 返回状态：图结构 en: Return state: graph structure
        state = self.get_obs_2()
           
        return state, reward , done

    def return_energy_with_wind(self,uav):
        """计算无人机返航所需的能量"""

        # 如果无人机已经到达返航位置，需要悬停到达目的地 en: If the UAV has already reached the return position, it needs to hover to reach the destination
        x, y = uav['destination']
        wind_vector = np.array([self.wind_u[x, y], self.wind_v[x, y]])
        V_u = self._get_real_v(np.array([0, 0]), wind_vector)  # 考虑风速影响的速度 en: Calculate the speed considering wind speed effect
        # 计算悬停到达目的地的能量消耗
        hover_energy = 0.5 * self.M * (V_u ** 2) * self.T  # 悬停能量消耗 en: Hover energy consumption
   
        if uav['position'] == uav['destination']:
            return hover_energy # 如果已经到达目的地，返回悬停能量 en: If already at the destination, return hover energy
        
        current_pos = np.array(uav['position'])
        home_pos = np.array(uav['destination'])  # 返航位置 en: Return position
        
        # 计算返航路径上的每一步 en: Calculate each step on the return path
        path_length = np.linalg.norm(current_pos - home_pos)
        num_steps = int(path_length)  # 至少一步 en: At least one step
        
        # 如果只需要一步，直接计算一步的能耗 en: If only one step is needed, directly calculate the energy for one step
        if num_steps == 1:
            x, y = uav['position']
            wind_vector = np.array([self.wind_u[x, y], self.wind_v[x, y]])
            direction = home_pos - current_pos
            V_u = self._get_real_v(direction, wind_vector)  # 考虑风速影响的速度 en: Calculate the speed considering wind speed effect
            return 0.5 * self.M * (V_u ** 2) * self.T + hover_energy  #标准走一步需要的能量 en: Standard energy required for one step
        
        # 如果距离较远，沿路径取若干点计算平均能耗 en: If the distance is far, take several points along the path to calculate average energy consumption
        total_energy = 0
        for i in range(num_steps):
            # 计算路径上的点
            pos = current_pos + (home_pos - current_pos) * i / num_steps
            pos = np.clip(pos, [0, 0], [self.Lx-1, self.Ly-1]).astype(int)  # 确保在网格范围内 en: Ensure within grid bounds
            # 获取该点的风速 en: Get wind speed at that point
            wind_vector = np.array([self.wind_u[pos[0], pos[1]], self.wind_v[pos[0], pos[1]]])
            # 计算方向向量 en: Calculate direction vector
            direction = (home_pos - current_pos) / path_length  # 归一化方向向量 en: Normalize direction vector
            # 计算考虑风速后的实际速度 en: Calculate actual speed considering wind speed
            V_u = self._get_real_v(direction, wind_vector)
            # 计算单步能耗并累加 en: Calculate step energy consumption and accumulate
            step_energy = 0.5 * self.M * (V_u ** 2) * self.T  # 每一步的能耗 en: Energy consumption for each step
            total_energy += step_energy
        return total_energy + hover_energy  # 返回总能耗 en: Return total energy consumption

    def _check_done(self):
        """检查是否达到终止条件"""
        # en: Check if termination conditions are met
        # 当所有UAV能量耗尽或超出搜索时间时，任务结束 en: Check if termination conditions are met
        if all(uav['done'] == True for uav in self.uavs) or self.remaining_time <= 0:
            return True
        return False
        
    def _get_obs(self):
        """无人机飞行决策的状态"""
        # en: State for UAV flight decision
        state = []
        # 为每个UAV添加局部视野矩阵的不确定度信息 en: Add uncertainty information of local view matrix for each UAV
        for uav in self.uavs:
            x, y = uav['position']
            state_dim = 15
            state_i = np.zeros((state_dim, state_dim))  # 初始化局部视野矩阵 en: Initialize local view matrix
            state_i_wind_u = np.zeros((state_dim, state_dim))      # 风速u分量矩阵 en: Wind speed u component matrix
            state_i_wind_v = np.zeros((state_dim, state_dim))      # 风速v分量矩阵 en: Wind speed v component matrix
            for dx in range(-state_dim//2, state_dim//2+1):
                for dy in range(-state_dim//2, state_dim//2+1): 
                    nx, ny = x + dx, y + dy
                    # 如果位置在网格范围内，添加该位置的不确定度 en: If position is within grid bounds, add uncertainty at that position
                    if 0 <= nx < self.Lx and 0 <= ny < self.Ly:
                        # 检查该位置是否有其他无人机 en: Check if there are other UAVs at this position
                        if any((nx, ny) == other_uav['position'] for other_uav in self.uavs if other_uav != uav and other_uav['done'] == False):
                            state_i[dx + state_dim//2, dy + state_dim//2] = -1  # 如果有其他无人机，设置为0 en: If there are other UAVs, set to -1
                        else:
                            state_i[dx + state_dim//2, dy + state_dim//2] = self.uncertainty_matrix[nx, ny]

                        # 添加风速信息 en: Add wind speed information
                        state_i_wind_u[dx + state_dim//2, dy + state_dim//2] = self.wind_u[nx, ny]
                        state_i_wind_v[dx + state_dim//2, dy + state_dim//2] = self.wind_v[nx, ny]
                    else:
                        # 如果超出网格范围，添加一个特殊值表示不可用 en: If out of grid bounds, add a special value indicating unavailable
                        state_i[dx + state_dim//2, dy + state_dim//2] = -1
                        # 添加风速信息 en: Add wind speed information
                        state_i_wind_u[dx + state_dim//2, dy + state_dim//2] = 0  # 网格外风速设为0 en: Wind speed outside grid set to 0
                        state_i_wind_v[dx + state_dim//2, dy + state_dim//2] = 0  # 网格外风速设为0 en: Wind speed outside grid set to 0
            # 将三个矩阵叠加为一个三通道矩阵 en: Combine the three matrices into a three-channel matrix
            combined_state = np.stack([state_i, state_i_wind_u, state_i_wind_v], axis=0)
            state.append(combined_state)
        return np.array(state)

    def generate_sagin_graph(self):
        """
        生成无人机与卸载设备之间的图结构，并转换为 PyTorch Geometric 的 Data 对象。
        Generate the graph structure between UAVs and offloading devices, and convert it to a PyTorch Geometric Data object.
        """
        # 创建一个无向图 en: Create an undirected graph
        G = nx.Graph()
        # 添加无人机节点 en: Add UAV nodes
        for i, uav in enumerate(self.uavs):
            i_attr = self.task_matrix[uav['position'][0], uav['position'][1]]  # 任务难度 en: Task difficulty
            G.add_node(f"UAV_{i}", type="UAV",node_type=0, x = [i_attr,1.0,0,0,0,0])  #这里x是特征值，这个目前先这么写 en: Add UAV nodes with attributes
            
        # 添加卸载设备节点，先默认都是1个设备，后面可以改成多个设备 en: Add offloading device nodes, initially assuming only one device, can be changed to multiple devices later
        G.add_node("GBS", type="GBS",node_type=1, x = [0,0,1.0,0,0,0])  #这里后期改成多基站的情况 en: Add GBS node with attributes
        G.add_node("HAPS", type="HAPS",node_type=2, x = [0,0,0,1.0,0,0])
        G.add_node("LEO", type="LEO",node_type=3, x = [0,0,0,0,1.0,0])
        G.add_node("Cloud", type="Cloud",node_type=4, x = [0,0,0,0,0,1.0])   #云服务器应当只有一个 en: Add Cloud node with attributes

        # 云服务器只与基站相连 en: Cloud server is only connected to GBS
        G.add_edge("GBS", "Cloud", edge_weight =1)
        # HAPS和LEO都与基站相连 en: HAPS and LEO are both connected to GBS
        G.add_edge("GBS", "HAPS", edge_weight =1)
        G.add_edge("GBS", "LEO", edge_weight =1)

        # 无人机与非基站卸载设备之间默认相连 en: UAVs are connected to non-GBS offloading devices by default
        for i, uav in enumerate(self.uavs):
            # 与HAPS默认连接 en: UAVs are connected to HAPS by default
            G.add_edge(f"UAV_{i}", "HAPS", edge_weight =1)
            # 与LEO默认连接 en: UAVs are connected to LEO by default
            G.add_edge(f"UAV_{i}", "LEO", edge_weight =1)

        return G
    
    def get_obs_2(self):
        """获取卸载决策的状态"""
        G = self.G.copy()  # 复制图结构 en: Copy the graph structure
        max_grid_distance = np.sqrt((self.Lx - 1) ** 2 + (self.Ly - 1) ** 2)
        # 计算无人机与基站之间的距离并添加边 en: Calculate distance between UAVs and GBS and add edges
        for i, uav in enumerate(self.uavs):
            uav['link'] = 0  # 初始化连接状态 en: Initialize link status
            uav_pos = np.array(uav['position'])
            # 与基站的距离
            gbs_distance = np.linalg.norm(uav_pos - self.gbs_position)
            if gbs_distance <= self.Lx//2:  # 这里假设距离不大于Lx//2就可以连接 en: Assume connection to GBS is possible if distance is less than Lx//2
                G.add_edge(f"UAV_{i}", "GBS", edge_weight = 1-2*gbs_distance/self.Lx)
                uav['link'] = 1  # 连接基站 en: Connected to GBS
            # HAPS链路按水平距离衰减，保持全域可连 en: HAPS edge weight decays with distance while staying globally connected.
            if G.has_edge(f"UAV_{i}", "HAPS"):
                G.remove_edge(f"UAV_{i}", "HAPS")
            haps_distance = np.linalg.norm(uav_pos - self.haps_position)
            haps_weight = float(np.clip(1.0 - haps_distance / max_grid_distance, 0.05, 1.0))
            G.add_edge(f"UAV_{i}", "HAPS", edge_weight=haps_weight)

        # 转换为PyTorch Geometric数据格式 en: Convert to PyTorch Geometric data format
        data = from_networkx(G)
        return data

    def reset(self, seed=None, positions=None, destinations=None, wind_seed=None, terrain_seed=None, infra_seed=None):
        """重置环境"""
        # en: Reset the environment
        # Backward compatibility:
        # legacy `seed` controls wind/terrain/infra when dedicated seeds are not provided.
        if wind_seed is None and terrain_seed is None and infra_seed is None:
            wind_seed = seed
            terrain_seed = seed
            infra_seed = seed
        else:
            if wind_seed is None:
                wind_seed = seed
            if terrain_seed is None:
                terrain_seed = seed
            if infra_seed is None:
                infra_seed = seed

        if positions is not None and destinations is not None and len(positions) == self.N and len(destinations) == self.N:
            self.uavs = [self._initialize_uav(positions[u],destinations[u]) for u in range(self.N)]  # 初始化无人机 en: Initialize UAVs
        else:
            self.uavs = [self._initialize_uav(self.gird_position[u],self.gird_position[u]) for u in range(self.N)] #初始化无人机 en: Initialize UAVs
        self.uncertainty_matrix = np.ones((self.Lx, self.Ly))  # 重置不确定度矩阵 en: Reset uncertainty matrix
        if terrain_seed is not None:
            terrain_rng = np.random.default_rng(terrain_seed)
            self.task_matrix = terrain_rng.integers(1, 5, size=(self.Lx, self.Ly))
        else:
            self.task_matrix = np.random.randint(1, 5, size=(self.Lx, self.Ly)) # 重置任务难度矩阵 en: Reset task difficulty matrix
        
        # 起飞点的不确定度设为0，表示不需要处理任务 en: Set uncertainty at takeoff points to 0, indicating no task processing needed
        for uav in self.uavs:
            x, y = uav['position']
            self.uncertainty_matrix[x, y] = 0  # 起飞点不需要处理任务 en: Set uncertainty at takeoff points to 0, indicating no task processing needed
            self.uncertainty_matrix[x, y] = 0  # 起飞点不需要处理任务 en: Set uncertainty at takeoff points to 0, indicating no task processing needed
        # 基础设施位置（GBS/HAPS）重采样 en: Resample infrastructure positions (GBS/HAPS).
        self._sample_infrastructure_positions(infra_seed)
        #重置风的区域 en: Reset wind subregion
        if wind_seed is not None:
            random.seed(wind_seed)
        self.wind_u, self.wind_v, _, _ = extract_wind_subregion(self.file_path,self.Lx)
        # 重置剩余搜索时间 en: Reset remaining search time
        self.remaining_time = self.max_search_time
        state = self._get_obs()
        return state
    



class PolicyNet_CNN(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(32 * state_dim * state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # x = x.unsqueeze(1)  # 添加通道维度 (batch, 1, 5, 5)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x - x.max(dim=1, keepdim=True)[0]  # softmax 数值稳定 trick
        return F.softmax(x, dim=1)


class QValueNet_CNN(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(32 * state_dim * state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        # x = x.unsqueeze(1)  # 添加通道维度 (batch, 1, 5, 5)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PolicyNet_GCN(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet_GCN, self).__init__()
        self.conv1 = GCNConv(state_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, action_dim)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight.float()
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        x = x - x.max(dim=1, keepdim=True)[0]  # softmax 数值稳定 trick
        x = F.softmax(x, dim=1)
        # 只保留 UAV 的特征
        uav_mask = data.node_type == 0  # 0 代表 UAV
        uav_output = x[uav_mask]
        return uav_output


class QValueNet_GCN(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNet_GCN, self).__init__()
        self.conv1 = GCNConv(state_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, action_dim)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight.float()
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = self.conv2(x, edge_index, edge_weight)
        # 只保留 UAV 的特征
        uav_mask = data.node_type == 0
        uav_output = x[uav_mask]
        return uav_output


class SAC:
    ''' 处理离散动作的SAC算法 '''

    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 alpha_lr, target_entropy, tau, gamma, device, type='CNN'):
        self.type = type
        if type == 'CNN':
            self.actor = PolicyNet_CNN(state_dim, hidden_dim, action_dim).to(device)
            self.critic_1 = QValueNet_CNN(state_dim, hidden_dim, action_dim).to(device)
            self.critic_2 = QValueNet_CNN(state_dim, hidden_dim, action_dim).to(device)
            self.target_critic_1 = QValueNet_CNN(state_dim, hidden_dim, action_dim).to(device)
            self.target_critic_2 = QValueNet_CNN(state_dim, hidden_dim, action_dim).to(device)
        elif type == 'GCN':
            self.actor = PolicyNet_GCN(state_dim, hidden_dim, action_dim).to(device)
            self.critic_1 = QValueNet_GCN(state_dim, hidden_dim, action_dim).to(device)
            self.critic_2 = QValueNet_GCN(state_dim, hidden_dim, action_dim).to(device)
            self.target_critic_1 = QValueNet_GCN(state_dim, hidden_dim, action_dim).to(device)
            self.target_critic_2 = QValueNet_GCN(state_dim, hidden_dim, action_dim).to(device)
        else:
            # 策略网络
            self.actor = PolicyNet_GCN(state_dim, hidden_dim, action_dim).to(device)
            # 第一个Q网络
            self.critic_1 = QValueNet_GCN(state_dim, hidden_dim, action_dim).to(device)
            # 第二个Q网络
            self.critic_2 = QValueNet_GCN(state_dim, hidden_dim, action_dim).to(device)
            self.target_critic_1 = QValueNet_GCN(state_dim, hidden_dim,
                                                 action_dim).to(device)  # 第一个目标Q网络
            self.target_critic_2 = QValueNet_GCN(state_dim, hidden_dim,
                                                 action_dim).to(device)  # 第二个目标Q网络
        # 令目标Q网络的初始参数和Q网络一样
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)
        # 使用alpha的log值,可以使训练结果比较稳定
        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True  # 可以对alpha求梯度
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy  # 目标熵的大小
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state):
        ''' 根据当前状态选择动作 '''
        if self.type == 'GCN':
            data = state.to(self.device)
            probs = self.actor(data)
            action_dist = torch.distributions.Categorical(probs)
            actions = action_dist.sample()
            return actions.cpu().numpy().tolist()
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            probs = self.actor(state)
            action_dist = torch.distributions.Categorical(probs)
            action = action_dist.sample()
            return action.item()

    # 计算目标Q值,直接用策略网络的输出概率进行期望计算
    def calc_target(self, rewards, next_states, dones):
        next_probs = self.actor(next_states)
        next_log_probs = torch.log(next_probs + 1e-8)
        entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
        q1_value = self.target_critic_1(next_states)
        q2_value = self.target_critic_2(next_states)
        min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)
        next_value = min_qvalue + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        if self.type == 'GCN':
            # 处理图数据
            states = Batch.from_data_list(transition_dict['states']).to(self.device)
            actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
            rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
            next_states = Batch.from_data_list(transition_dict['next_states']).to(self.device)
            dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)
        else:
            states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
            actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)  # 动作不再是float类型
            rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
            next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
            dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # 更新两个Q网络
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_q_values = self.critic_1(states).gather(1, actions)
        critic_1_loss = torch.mean(
            F.mse_loss(critic_1_q_values, td_target.detach()))
        critic_2_q_values = self.critic_2(states).gather(1, actions)
        critic_2_loss = torch.mean(
            F.mse_loss(critic_2_q_values, td_target.detach()))
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新策略网络
        probs = self.actor(states)
        log_probs = torch.log(probs + 1e-8)
        # 直接根据概率计算熵
        entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
        q1_value = self.critic_1(states)
        q2_value = self.critic_2(states)
        min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                               dim=1,
                               keepdim=True)  # 直接根据概率计算期望
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新alpha值
        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)


class ReplayBuffer2:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, done):  # 将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return list(state), action, reward, list(next_state), done

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)
