import json
import random
import numpy as np
import os.path

def extract_wind_subregion(json_path, subregion_size=40):
    """
    从风速数据JSON文件中随机提取指定大小的子区域
    # en: Extract a subregion of specified size from wind speed data in a JSON file
    
    参数: en: Parameters:
        json_path (str): JSON文件路径 # en: Path to the JSON file containing wind speed data
        subregion_size (int): 要提取的子区域大小，默认为40x40 # en: Size of the subregion to extract, default is 20x20
        
    返回:
        tuple: (u子矩阵, v子矩阵, 起始x坐标, 起始y坐标) # en: Returns a tuple of (u submatrix, v submatrix, start x coordinate, start y coordinate)
    """
    # 读取JSON文件 # en: Read the JSON file
    with open(json_path, 'r') as file:
        data = json.load(file)

    # 从JSON中获取宽度和高度 # en: Get width and height from JSON data
    width = data['width']
    height = data['height']

    # 提取u和v的风速分量 # en: Extract u and v wind speed components
    u_data = data['u']['array']
    v_data = data['v']['array']

    # 将一维数据转换为二维矩阵 # en: Convert 1D data to 2D matrices
    u_matrix = np.array(u_data).reshape(height, width)
    v_matrix = np.array(v_data).reshape(height, width)

    # 确保能够提取指定大小的区域 # en: Ensure that the specified subregion can be extracted
    max_start_x = width - subregion_size
    max_start_y = height - subregion_size

    if max_start_x < 0 or max_start_y < 0:
        # en: The original data size is insufficient to extract a {subregion_size}×{subregion_size} region, current size: {width}×{height}
        raise ValueError(f"原始数据尺寸不足以提取{subregion_size}×{subregion_size}区域，当前尺寸: {width}×{height}")

    # 随机选择起始点 # en: Randomly select the starting point
    start_x = random.randint(0, max_start_x)
    start_y = random.randint(0, max_start_y)

    # 提取子区域 # en: Extract the subregion
    u_submatrix = u_matrix[start_y:start_y+subregion_size, start_x:start_x+subregion_size]
    v_submatrix = v_matrix[start_y:start_y+subregion_size, start_x:start_x+subregion_size]
    
    return u_submatrix, v_submatrix, start_x, start_y

def wind_direction_std(u_submatrix, v_submatrix):
    """
    计算风场u和v分量矩阵的风向标准差   en: Calculate the wind direction standard deviation from u and v component matrices
    
    参数: en: Parameters:
    u_submatrix : 风场东西向分量矩阵，正值表示东风   en: u component matrix, positive values indicate east wind
    v_submatrix : 风场南北向分量矩阵，正值表示北风   en: v component matrix, positive values indicate north wind
    
    返回:
    风向的循环标准差（单位：度） # en: Returns the circular standard deviation of wind direction (in degrees)
    """ 
    # 计算每个点的风速（非零风速才考虑在内） # en: Calculate wind speed for each point (only consider non-zero wind speeds)
    speeds = np.sqrt(u_submatrix**2 + v_submatrix**2)
    
    # 计算加权平均的风向矢量 # en: Calculate the weighted average wind direction vector
    mean_u = np.sum(u_submatrix)
    mean_v = np.sum(v_submatrix)
    
    # 计算风向一致性指数R (0-1) # en: Calculate the wind direction consistency index R (0-1)
    R = np.sqrt(mean_u**2 + mean_v**2) / np.sum(speeds)
 
    # 计算循环标准差 # en: Calculate the circular standard deviation
    circular_std_rad = np.sqrt(-2 * np.log(R))
    
    # 转换为角度 # en: Convert to degrees
    return np.degrees(circular_std_rad)


def wind_speed_std(u_submatrix, v_submatrix):   
    """
    计算子区域风速的标准差，体现风场强度差异 en: Calculate the standard deviation of wind speed in a subregion, reflecting differences in wind field intensity

    参数:
        u_submatrix (np.ndarray): u分量子矩阵 en: u component submatrix
        v_submatrix (np.ndarray): v分量子矩阵 en: v component submatrix

    返回:
        float: 风速标准差（m/s） en: Returns the standard deviation of wind speed (m/s)
    """
    # 计算风速 en: Calculate wind speed
    wind_speed = np.sqrt(u_submatrix**2 + v_submatrix**2)
    # 计算标准差 en: Calculate standard deviation
    std = np.std(wind_speed)
    return std

def wind_speed_mean(u_submatrix, v_submatrix):
    """
    计算子区域风速的均值，体现风场强度差异 en: Calculate the mean wind speed in a subregion, reflecting differences in wind field intensity

    参数:
        u_submatrix (np.ndarray): u分量子矩阵 en: u component submatrix
        v_submatrix (np.ndarray): v分量子矩阵 en: v component submatrix

    返回:
        float: 风速均值（m/s） en: Returns the mean wind speed (m/s)
    """
    # 计算风速 en: Calculate wind speed
    wind_speed = np.sqrt(u_submatrix**2 + v_submatrix**2)
    # 计算均值 en: Calculate mean
    mean = np.mean(wind_speed)
    return mean

# 使用示例
if __name__ == "__main__":
    try:
        # 首先尝试相对路径 'OUR_ENV_WITH_WIND_JSON/wind.json' # en: First try the relative path 'OUR_ENV_WITH_WIND_JSON/wind.json'
        if os.path.exists('OUR_ENV_WITH_WIND_JSON/wind.json'):
            file_path = 'OUR_ENV_WITH_WIND_JSON/wind.json'
        # 如果相对路径不存在，则尝试当前目录 'wild.json' # en: If the relative path does not exist, try the current directory 'wind.json'
        elif os.path.exists('wind.json'):
            file_path = 'wind.json'
        else:
            raise FileNotFoundError("找不到风速数据文件，请确保 'wind.json' 或 'OUR_ENV_WITH_WIND_JSON/wind.json' 存在")
        
        u_sub, v_sub, x, y = extract_wind_subregion(file_path)
        print(f"从位置 ({x}, {y}) 提取了40×40的子区域")
        print(f"U子矩阵形状: {u_sub.shape}")
        print(f"V子矩阵形状: {v_sub.shape}")
    except Exception as e:
        print(f"错误: {e}")