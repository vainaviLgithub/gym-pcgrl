import numpy as np
import random
import json
from collections import deque

def random_walk_grid(width, height, obstacle_density=0.2):
    grid = [['0' for _ in range(width)] for _ in range(height)]
    num_walls = int(width * height * obstacle_density)

    for _ in range(num_walls):
        x, y = random.randint(0, height - 1), random.randint(0, width - 1)
        grid[x][y] = '#'

    return grid

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def place_game_elements(grid, max_distance=5):
    height, width = len(grid), len(grid[0])
    def get_open_cells():
        return [(i, j) for i in range(height) for j in range(width) if grid[i][j] == '0']
    open_cells = get_open_cells()
    random.shuffle(open_cells)

    for taxi in open_cells:
        for passenger in open_cells:
            if taxi == passenger or manhattan(taxi, passenger) > max_distance:
                continue
            for dropoff in open_cells:
                if dropoff in [taxi, passenger]:
                    continue
                if manhattan(passenger, dropoff) > max_distance:
                    continue
                return taxi, passenger, dropoff

    return random.sample(open_cells, 3)

def is_playable(grid, taxi_pos, passenger_pos, dropoff_pos):
    def bfs(start, goal):
        queue = deque([start])
        visited = set([start])
        directions = [(0,1), (0,-1), (1,0), (-1,0)]

        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] != '#' and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False

    return bfs(taxi_pos, passenger_pos) and bfs(passenger_pos, dropoff_pos)

def generate_taxi_level(width=5, height=5, difficulty="easy"):
    difficulty_settings = {
        "easy": 0.1,
        "medium": 0.25,
        "hard": 0.4
    }
    obstacle_density = difficulty_settings.get(difficulty, 0.2)

    while True:
        grid = random_walk_grid(width, height, obstacle_density)
        taxi_pos, passenger_pos, dropoff_pos = place_game_elements(grid)
        if is_playable(grid, taxi_pos, passenger_pos, dropoff_pos):
            return {
                'grid': grid,
                'taxi': taxi_pos,
                'passenger': passenger_pos,
                'dropoff': dropoff_pos,
                'difficulty': difficulty
            }


def save_levels(num_levels=30, width=10, height=10):
    levels = []
    for i in range(num_levels):
        if i < num_levels * 0.33:
            difficulty = "easy"
        elif i < num_levels * 0.66:
            difficulty = "medium"
        else:
            difficulty = "hard"

        level = generate_taxi_level(width, height, difficulty)
        levels.append(level)

    with open("game_levels.json", "w") as f:
        json.dump(levels, f, indent=4)

    print(f"Saved {num_levels} levels: "
          f"{int(num_levels*0.33)} easy, "
          f"{int(num_levels*0.33)} medium, "
          f"{num_levels - 2*(num_levels//3)} hard.")


# Run only if script is executed directly
if __name__ == "__main__":
    save_levels(num_levels=10, width=10, height=10)
    print("Levels saved successfully in game_levels.json!")
