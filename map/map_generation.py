import pygame
import noise
import random
import sys
import heapq
from collections import deque
import math


class MapGenerator:
    def __init__(self, width=1000, height=1000, tile_size=1, water_level=0.3, scale=50.0,
                 octaves=5, persistence=0.5, lacunarity=2.0,
                 big_lake_threshold=300, min_mountain_size=100,
                 region_size=50, task_threshold=400, min_tasks=1,seed=None):
        # 地图基本参数  Basic parameters of the map
        self.WIDTH = width
        self.HEIGHT = height
        self.TILE_SIZE = tile_size
        self.ROWS = self.HEIGHT // self.TILE_SIZE
        self.COLS = self.WIDTH // self.TILE_SIZE

        self.water_level = water_level
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.lacunarity = lacunarity
        self.BIG_LAKE_THRESHOLD = big_lake_threshold
        self.min_mountain_size = min_mountain_size
        self.region_size = region_size
        self.task_threshold = task_threshold
        self.min_tasks = min_tasks
        self.seed = seed

        # 设置随机种子
        if self.seed is not None:
            random.seed(self.seed)

        self.COLORS = {
            'LAKE': (0, 105, 148),
            'RIVER': (30, 144, 255),
            'BEACH': (194, 178, 128),
            'GRASS': (34, 139, 34),
            'FOREST': (0, 100, 0),
            'MOUNTAIN': (139, 69, 19),
        }

        # 内部数据存储
        # Internal data storage
        self.height_map = None
        self.lake_mask = None
        self.river_mask = None
        self.terrain_map = None
        self.region_scores = None
        self.task_matrix = None

    def generate_height_map(self, base=None):
        if base is None:
            base = random.randint(0, 1000)
        height_map = []
        for y in range(self.ROWS):
            row_data = []
            for x in range(self.COLS):
                val = noise.pnoise2(
                    x / self.scale,
                    y / self.scale,
                    octaves=self.octaves,
                    persistence=self.persistence,
                    lacunarity=self.lacunarity,
                    repeatx=self.COLS,
                    repeaty=self.ROWS,
                    base=base
                )
                val = (val + 1) / 2  # 映射到0~1
                row_data.append(val)
            height_map.append(row_data)
        return height_map

    def identify_lakes(self, height_map):
        visited = [[False]*self.COLS for _ in range(self.ROWS)]
        lake_mask = [[False]*self.COLS for _ in range(self.ROWS)]
        lakes = []
        lake_id = 0

        for y in range(self.ROWS):
            for x in range(self.COLS):
                if height_map[y][x] < self.water_level and not visited[y][x]:
                    queue = deque([(x, y)])
                    visited[y][x] = True
                    lake_cells = []
                    while queue:
                        cx, cy = queue.popleft()
                        lake_cells.append((cx, cy))
                        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                            nx, ny = cx+dx, cy+dy
                            if 0 <= nx < self.COLS and 0 <= ny < self.ROWS:
                                if not visited[ny][nx] and height_map[ny][nx] < self.water_level:
                                    visited[ny][nx] = True
                                    queue.append((nx, ny))
                    for (lx, ly) in lake_cells:
                        lake_mask[ly][lx] = True
                    lakes.append((lake_id, lake_cells))
                    lake_id += 1

        return lake_mask, lakes

    def is_path_too_straight(self, path):
        if not path or len(path) < 10:
            return True
        turns = 0
        for i in range(1, len(path)-1):
            (x0, y0) = path[i-1]
            (x1, y1) = path[i]
            (x2, y2) = path[i+1]
            dx1, dy1 = x1 - x0, y1 - y0
            dx2, dy2 = x2 - x1, y2 - y1
            cross = dx1*dy2 - dy1*dx2
            if abs(cross) > 0:
                turns += 1
        return turns < (len(path) // 10)

    def convert_terrain_to_numbers(self):
        """
        将地形字符串矩阵转换为数字矩阵。
        ---
        Convert the terrain string matrix to a numeric matrix.

        返回:
        - 数字矩阵，每个地形类型映射为一个整数。
        """
        # 定义地形到数字的映射
        terrain_to_number = {
            'LAKE': 0,
            'RIVER': 1,
            'BEACH': 2,
            'GRASS': 3,
            'FOREST': 4,
            'MOUNTAIN': 5,
        }

        number_map = []
        for row in self.terrain_map:
            number_row = [terrain_to_number[terrain] for terrain in row]
            number_map.append(number_row)

        return number_map

    def find_long_river_path(self, height_map, lake_mask, has_big_lake, big_lake_cells):
        MOUNTAIN_HEIGHT = 0.8
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
        INF = float('inf')

        big_lake_set = set(big_lake_cells) if has_big_lake else set()

        dist = [[INF] * self.COLS for _ in range(self.ROWS)]
        prev = [[None] * self.COLS for _ in range(self.ROWS)]
        visited_lake_dist = [[[INF, INF] for _ in range(self.COLS)] for _ in range(self.ROWS)]

        pq = []
        # 初始化顶部行
        # Initialize top row
        for x in range(self.COLS):
            if height_map[0][x] > MOUNTAIN_HEIGHT:
                continue
            dist[0][x] = height_map[0][x]
            visited_lake_dist[0][x][0] = height_map[0][x]
            heapq.heappush(pq, (dist[0][x], (x, 0), False))

        while pq:
            cost, (cx, cy), visited_lake = heapq.heappop(pq)
            lake_idx = 1 if visited_lake else 0
            if cost > visited_lake_dist[cy][cx][lake_idx]:
                continue

            if cy == self.ROWS - 1:
                if not has_big_lake or visited_lake:
                    best_path = []
                    curx, cury = cx, cy
                    while (curx, cury) is not None:
                        best_path.append((curx, cury))
                        cur = prev[cury][curx]
                        if cur is None:
                            break
                        curx, cury = cur
                    best_path.reverse()
                    return best_path

            for dx, dy in directions:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.COLS and 0 <= ny < self.ROWS:
                    if height_map[ny][nx] > MOUNTAIN_HEIGHT:
                        continue

                    new_lake_visited = visited_lake
                    if (nx, ny) in big_lake_set:
                        new_lake_visited = True
                    new_lake_idx = 1 if new_lake_visited else 0

                    new_cost = cost + abs(height_map[ny][nx] - height_map[cy][cx]) + 0.001
                    if new_cost < visited_lake_dist[ny][nx][new_lake_idx]:
                        visited_lake_dist[ny][nx][new_lake_idx] = new_cost
                        prev[ny][nx] = (cx, cy)
                        heapq.heappush(pq, (new_cost, (nx, ny), new_lake_visited))

        return None

    def remove_small_mountain_areas(self, terrain_map):
        rows = len(terrain_map)
        cols = len(terrain_map[0]) if rows > 0 else 0
        visited = [[False] * cols for _ in range(rows)]
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        for y in range(rows):
            for x in range(cols):
                if terrain_map[y][x] == "MOUNTAIN" and not visited[y][x]:
                    stack = [(x, y)]
                    visited[y][x] = True
                    mountain_cells = []
                    while stack:
                        cx, cy = stack.pop()
                        mountain_cells.append((cx, cy))
                        for dx, dy in directions:
                            nx, ny = cx + dx, cy + dy
                            if 0 <= nx < cols and 0 <= ny < rows:
                                if terrain_map[ny][nx] == "MOUNTAIN" and not visited[ny][nx]:
                                    visited[ny][nx] = True
                                    stack.append((nx, ny))
                    if len(mountain_cells) < self.min_mountain_size:
                        for (mx, my) in mountain_cells:
                            terrain_map[my][mx] = "FOREST"
        return terrain_map

    def determine_terrain(self, height_map, lake_mask, river_mask):
        terrain_map = []
        for y in range(self.ROWS):
            row_data = []
            for x in range(self.COLS):
                h = height_map[y][x]
                if (x, y) in river_mask:
                    terrain = 'RIVER'
                elif lake_mask[y][x]:
                    terrain = 'LAKE'
                else:
                    if h < self.water_level + 0.02:
                        terrain = 'BEACH'
                    elif h < 0.5:
                        terrain = 'GRASS'
                    elif h < 0.6:
                        terrain = 'FOREST'
                    elif h < 0.8:
                        terrain = 'MOUNTAIN'
                    else:
                        terrain = 'MOUNTAIN' # 这里没有NO则默认MOUNTAIN  If there is no 'NO' here, it defaults to 'MOUNDAIN'
                row_data.append(terrain)
            terrain_map.append(row_data)
        return terrain_map

    def calculate_region_scores(self, terrain_map):
        """
        计算每个区域的探索难度评分。
        ---
        Calculate the exploration difficulty score for each region.

        参数:
        Parameters:
        - terrain_map: 地形矩阵。 Terrain matrix.
        - region_size: 每个区域的大小 (正方形的边长)。 The size of each region (the side length of the square).
        """
        scores = []
        regions_per_row = self.WIDTH // self.region_size
        regions_per_col = self.HEIGHT // self.region_size

        TERRAIN_SCORES = {
            'MOUNTAIN': 5,
            'FOREST': 3,
            'GRASS': 1,
            'BEACH': 1,
            'LAKE': 0,
            'RIVER': 0,
        }

        for region_y in range(regions_per_col):
            row_scores = []
            for region_x in range(regions_per_row):
                score = 0
                for y in range(region_y * self.region_size, (region_y + 1) * self.region_size):
                    for x in range(region_x * self.region_size, (region_x + 1) * self.region_size):
                        terrain_type = terrain_map[y][x]
                        score += TERRAIN_SCORES.get(terrain_type, 0)
                row_scores.append(score)
            scores.append(row_scores)

        return scores

    def generate_task_matrix_by_threshold(self, scores):
        task_matrix = []
        for row in scores:
            task_row = []
            for score in row:
                task_count = max(self.min_tasks, math.ceil(score / self.task_threshold))
                task_row.append(task_count)
            task_matrix.append(task_row)
        return task_matrix

    def create_map(self):
        max_attempts = 1000
        path = None

        for attempt in range(max_attempts):
            base_seed = self.seed if self.seed is not None else random.randint(1000, 1000000)
            print(base_seed)
            self.height_map = self.generate_height_map(base=base_seed)
            self.lake_mask, lakes = self.identify_lakes(self.height_map)
            big_lakes = [(lid, cells) for (lid, cells) in lakes if len(cells) > self.BIG_LAKE_THRESHOLD]
            has_big_lake = len(big_lakes) > 0
            big_lake_cells = []
            if has_big_lake:
                big_lakes.sort(key=lambda l: len(l[1]), reverse=True)
                big_lake_cells = big_lakes[0][1]

            path = self.find_long_river_path(self.height_map, self.lake_mask, has_big_lake, big_lake_cells)

            if path is None:
                continue

            if self.is_path_too_straight(path):
                continue

            # 找到合适的路径后退出循环
            # Exit the loop after finding the appropriate path
            self.river_mask = set(path)
            break

        if path is None or self.is_path_too_straight(path):
            print("Failed to obtain the desired map within the limited number of attempts.")
            return False

        self.terrain_map = self.determine_terrain(self.height_map, self.lake_mask, self.river_mask)
        self.terrain_map = self.remove_small_mountain_areas(self.terrain_map)
        self.region_scores = self.calculate_region_scores(self.terrain_map)
        self.task_matrix = self.generate_task_matrix_by_threshold(self.region_scores)
        return True

    def get_height_map(self):
        return self.height_map

    def get_terrain_map(self):
        return self.terrain_map

    def get_terrain_map_num(self):
        return self.convert_terrain_to_numbers()

    def get_region_scores(self):
        return self.region_scores

    def get_task_matrix(self):
        return self.task_matrix


class MapVisualizer:
    def __init__(self, map_gen: MapGenerator):
        pygame.init()
        self.map_gen = map_gen
        self.screen = pygame.display.set_mode((map_gen.WIDTH, map_gen.HEIGHT))
        pygame.display.set_caption("Map Generation")
        self.clock = pygame.time.Clock()

    def draw_map(self):
        for y in range(self.map_gen.ROWS):
            for x in range(self.map_gen.COLS):
                color = self.map_gen.COLORS[self.map_gen.terrain_map[y][x]]
                pygame.draw.rect(self.screen, color, (x * self.map_gen.TILE_SIZE, y * self.map_gen.TILE_SIZE, self.map_gen.TILE_SIZE, self.map_gen.TILE_SIZE))

    def draw_tasks(self):
        font = pygame.font.Font(None, 20)
        scores = self.map_gen.region_scores
        task_matrix = self.map_gen.task_matrix
        region_size = self.map_gen.region_size
        for region_y, row in enumerate(task_matrix):
            for region_x, task_count in enumerate(row):
                text = font.render(str(task_count), True, (255, 0, 0))
                x = region_x * region_size + region_size // 2
                y = region_y * region_size + region_size // 2
                self.screen.blit(text, (x, y))

    def draw_grid(self):
        grid_size = self.map_gen.region_size
        for x in range(0, self.map_gen.WIDTH, grid_size):
            pygame.draw.line(self.screen, (255, 255, 255), (x, 0), (x, self.map_gen.HEIGHT), 1)
        for y in range(0, self.map_gen.HEIGHT, grid_size):
            pygame.draw.line(self.screen, (255, 255, 255), (0, y), (self.map_gen.WIDTH, y), 1)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            self.screen.fill((0, 0, 0))
            self.draw_map()
            self.draw_grid()
            self.draw_tasks()
            pygame.display.flip()
            self.clock.tick(30)

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    map_gen = MapGenerator(seed=796020)
    if map_gen.create_map():
        # 如果仅需数据输出:
        # If only data output is required:
        height_map = map_gen.get_height_map()
        terrain_map = map_gen.get_terrain_map()
        terrain_map_num = map_gen.get_terrain_map_num()
        region_scores = map_gen.get_region_scores()
        task_matrix = map_gen.get_task_matrix()

        # 可在此进行数据的存储或进一步处理
        # Data can be stored or further processed here

        print("Height Map:", len(height_map))
        print("Terrain Map:", len(terrain_map),'  ',terrain_map_num[0][0])
        print("Region Scores:", len(region_scores))
        print("Task Matrix:", len(task_matrix))

        # 如果需要可视化： If visualization is required:
        visualizer = MapVisualizer(map_gen)
        visualizer.run()
    else:
        print("Map generation failed.")