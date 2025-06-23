import json
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class GridGameEnv(gym.Env):
    def __init__(self, config_file="game_levels.json", levels=None, use_curriculum=True):
        super(GridGameEnv, self).__init__()

        self.use_curriculum = use_curriculum

        if levels is not None:
            if not isinstance(levels, list):
                raise ValueError(f"levels must be a list, got {type(levels)}")
            self.all_levels = []
            for level in levels:
                if not isinstance(level, dict):
                    print(f"Warning: Skipping invalid level (not a dict): {level}")
                    continue
                if self._validate_level(level):
                    self.all_levels.append(level)
                else:
                    print(f"Warning: Skipping invalid level structure: {level}")
            if not self.all_levels:
                raise ValueError("No valid levels provided")
        else:
            try:
                with open(config_file, 'r') as f:
                    raw_levels = json.load(f)
                if not isinstance(raw_levels, list):
                    raise ValueError("config_file must contain a list of levels")
                self.all_levels = []
                for level in raw_levels:
                    if not isinstance(level, dict):
                        print(f"Warning: Skipping invalid level (not a dict): {level}")
                        continue
                    if self._validate_level(level):
                        self.all_levels.append(level)
                    else:
                        print(f"Warning: Skipping invalid level structure: {level}")
                if not self.all_levels:
                    raise ValueError("No valid levels loaded from config_file")
            except Exception as e:
                raise ValueError(f"Failed to load levels from {config_file}: {e}")

        print(f"Initialized GridGameEnv with {len(self.all_levels)} valid levels: {[level.get('difficulty', 'unknown') for level in self.all_levels]}")

        self.current_difficulty = "easy"
        self.difficulty_order = ["easy", "medium", "hard"]
        self.curriculum_levels = self._filter_levels_by_difficulty(self.current_difficulty)
        self.current_level_idx = 0
        self.completed_levels = 0
        self.performance_log = {}

        self.max_steps = 1000
        self.step_count = 0
        self.grid = None
        self.taxi_pos = None
        self.passenger_pos = None
        self.dropoff_pos = None
        self.passenger_in_taxi = False

        self.use_custom_level = False

        self.action_space = spaces.Discrete(4)
        self.reset()

        grid_size = self.grid.shape[0] * self.grid.shape[1]
        self.observation_space = spaces.Box(low=0, high=4, shape=(grid_size,), dtype=np.int32)

    def _validate_level(self, level):
        try:
            grid = level.get("grid", [])
            taxi = level.get("taxi", [])
            passenger = level.get("passenger", [])
            dropoff = level.get("dropoff", [])
            return (isinstance(grid, list) and len(grid) == 10 and all(isinstance(row, list) and len(row) == 10 for row in grid) and
                    all(all(cell in ['0', '#'] for cell in row) for row in grid) and
                    isinstance(taxi, list) and len(taxi) == 2 and all(isinstance(x, int) and 0 <= x < 10 for x in taxi) and
                    isinstance(passenger, list) and len(passenger) == 2 and all(isinstance(x, int) and 0 <= x < 10 for x in passenger) and
                    isinstance(dropoff, list) and len(dropoff) == 2 and all(isinstance(x, int) and 0 <= x < 10 for x in dropoff))
        except Exception as e:
            print(f"Level validation error: {e}")
            return False

    def _filter_levels_by_difficulty(self, difficulty):
        filtered = [level for level in self.all_levels if isinstance(level, dict) and level.get("difficulty", "easy") == difficulty]
        print(f"Filtered {len(filtered)} levels for difficulty: {difficulty}")
        return filtered

    def set_difficulty(self, difficulty):
        self.current_difficulty = difficulty
        self.curriculum_levels = self._filter_levels_by_difficulty(difficulty)
        self.current_level_idx = 0
        print(f"Set difficulty to {difficulty} with {len(self.curriculum_levels)} levels")

    def set_level(self, level_dict):
        if not isinstance(level_dict, dict):
            raise ValueError(f"level_dict must be a dictionary, got {type(level_dict)}")
        if not self._validate_level(level_dict):
            raise ValueError("Invalid level structure")
        grid = level_dict['grid']
        self.grid = np.array(grid)
        self.taxi_pos = tuple(level_dict['taxi'])
        self.passenger_pos = tuple(level_dict['passenger'])
        self.dropoff_pos = tuple(level_dict['dropoff'])
        self.passenger_in_taxi = False
        self.step_count = 0
        self.prev_taxi_pos = None
        self.current_difficulty = level_dict.get("difficulty", "easy")
        self.use_custom_level = True
        print(f"Set custom level with difficulty: {self.current_difficulty}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.passenger_in_taxi = False

        if self.use_custom_level:
            self.use_custom_level = False
            return self._get_observation(), {}
        else:
            if not self.use_curriculum:
                if not self.all_levels:
                    raise ValueError("No levels available for evaluation.")
                level = random.choice(self.all_levels)
            else:
                if not self.curriculum_levels:
                    raise ValueError(f"No levels available for difficulty {self.current_difficulty}")
                if self.current_level_idx >= len(self.curriculum_levels):
                    self._next_level()
                level = self.curriculum_levels[self.current_level_idx]

            if not isinstance(level, dict):
                raise ValueError(f"Selected level is not a dictionary: {level}")
            grid = level['grid']
            if len(grid) != 10 or any(len(row) != 10 for row in grid):
                raise ValueError(f"Level grid must be 10x10. Got {len(grid)}x{len(grid[0]) if grid else 0}")
            self.grid = np.array(grid)
            self.taxi_pos = tuple(level['taxi'])
            self.passenger_pos = tuple(level['passenger'])
            self.dropoff_pos = tuple(level['dropoff'])
            self.prev_taxi_pos = None
            self.current_difficulty = level.get("difficulty", "easy")
            print(f"Reset environment with level difficulty: {self.current_difficulty}")

        return self._get_observation(), {}

    def _get_observation(self):
        obs_grid = np.zeros_like(self.grid, dtype=np.int32)
        for i in range(obs_grid.shape[0]):
            for j in range(obs_grid.shape[1]):
                cell = self.grid[i][j]
                obs_grid[i][j] = 1 if cell == '#' else 0
        obs_grid[self.dropoff_pos] = 2
        if not self.passenger_in_taxi:
            obs_grid[self.passenger_pos] = 3
        obs_grid[self.taxi_pos] = 4
        return obs_grid.flatten()

    def step(self, action):
        new_pos = list(self.taxi_pos)
        if action == 0: new_pos[0] -= 1
        elif action == 1: new_pos[0] += 1
        elif action == 2: new_pos[1] -= 1
        elif action == 3: new_pos[1] += 1

        if self._is_valid_move(new_pos):
            self.taxi_pos = tuple(new_pos)

        reward = -0.1
        terminated = False
        truncated = False

        if self.taxi_pos == self.passenger_pos and not self.passenger_in_taxi:
            self.passenger_in_taxi = True
            reward += 5
        elif self.passenger_in_taxi and self.taxi_pos == self.dropoff_pos:
            reward += 15
            terminated = True
            self.completed_levels += 1
            self._next_level()

        self.step_count += 1
        truncated = self.step_count >= self.max_steps

        if self.step_count % 100 == 0:
            print(f"Step {self.step_count}: Taxi @ {self.taxi_pos}")

        if self.taxi_pos == self.prev_taxi_pos:
            reward -= 0.5

        self.prev_taxi_pos = self.taxi_pos

        return self._get_observation(), reward, terminated, truncated, {}

    def get_agent_position(self):
        return self.taxi_pos

    def _is_valid_move(self, new_pos):
        x, y = new_pos
        return 0 <= x < self.grid.shape[0] and 0 <= y < self.grid.shape[1] and self.grid[x][y] != '#'

    def _next_level(self):
        self.current_level_idx += 1

        if self.use_curriculum:
            if self.current_level_idx >= len(self.curriculum_levels):
                current_idx = self.difficulty_order.index(self.current_difficulty)
                if current_idx < len(self.difficulty_order) - 1:
                    self.current_difficulty = self.difficulty_order[current_idx + 1]
                    self.curriculum_levels = self._filter_levels_by_difficulty(self.current_difficulty)
                    self.current_level_idx = 0
                    print(f"Advanced to difficulty: {self.current_difficulty}")
                else:
                    print("Restarting curriculum from easy.")
                    self.current_difficulty = self.difficulty_order[0]
                    self.curriculum_levels = self._filter_levels_by_difficulty(self.current_difficulty)
                    self.current_level_idx = 0

    def render(self, mode="ansi"):
        display_grid = self.grid.copy().tolist()
        display_grid[self.taxi_pos[0]][self.taxi_pos[1]] = 'T'
        if not self.passenger_in_taxi:
            display_grid[self.passenger_pos[0]][self.passenger_pos[1]] = 'P'
        display_grid[self.dropoff_pos[0]][self.dropoff_pos[1]] = 'D'

        grid_str = "\n".join("".join(str(cell) for cell in row) for row in display_grid)
    
        if mode == "ansi":
            return grid_str
        else:
            print(grid_str)