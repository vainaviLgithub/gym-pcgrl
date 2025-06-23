import os
import numpy as np
from flask import Flask, render_template, jsonify, request, session
from flask_session import Session
import base64
import json
from gym_pcgrl.envs.pcgrl_env import GridGameEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

app = Flask(__name__)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Verify static/assets directory
assets_dir = os.path.join('static', 'assets')
if not os.path.exists(assets_dir):
    print(f"Creating assets directory: {assets_dir}")
    os.makedirs(assets_dir)

# Load and group levels
try:
    with open('./game_levels.json', 'r') as f:
        raw_levels = json.load(f)
    if not isinstance(raw_levels, list):
        raise ValueError("game_levels.json must contain a list of levels")
except FileNotFoundError:
    print("Error: game_levels.json not found. Creating a default level.")
    raw_levels = [
        {
            "difficulty": "easy",
            "grid": [["0" if i != 0 and i != 9 and j != 0 and j != 9 else "#" for j in range(10)] for i in range(10)],
            "taxi": [1, 1],
            "passenger": [3, 3],
            "dropoff": [8, 8]
        }
    ]
    with open('./game_levels.json', 'w') as f:
        json.dump(raw_levels, f, indent=2)
except (json.JSONDecodeError, ValueError) as e:
    print(f"Error loading game_levels.json: {e}. Using default level.")
    raw_levels = [
        {
            "difficulty": "easy",
            "grid": [["0" if i != 0 and i != 9 and j != 0 and j != 9 else "#" for j in range(10)] for i in range(10)],
            "taxi": [1, 1],
            "passenger": [3, 3],
            "dropoff": [8, 8]
        }
    ]

levels = {"easy": [], "medium": [], "hard": []}
for level in raw_levels:
    if not isinstance(level, dict):
        print(f"Warning: Skipping invalid level entry (not a dict): {level}")
        continue
    diff = level.get("difficulty", "easy").lower()
    grid = level.get("grid", [])
    taxi = level.get("taxi", [])
    passenger = level.get("passenger", [])
    dropoff = level.get("dropoff", [])
    # Relaxed validation to handle new levels
    if (isinstance(grid, list) and len(grid) == 10 and all(isinstance(row, list) and len(row) == 10 for row in grid) and
        all(all(cell in ['0', '#', 'T', 'P', 'D'] for cell in row) for row in grid) and
        isinstance(taxi, list) and len(taxi) == 2 and all(isinstance(x, (int, float)) and 0 <= x < 10 for x in taxi) and
        isinstance(passenger, list) and len(passenger) == 2 and all(isinstance(x, (int, float)) and 0 <= x < 10 for x in passenger) and
        isinstance(dropoff, list) and len(dropoff) == 2 and all(isinstance(x, (int, float)) and 0 <= x < 10 for x in dropoff)):
        if diff in levels:
            levels[diff].append(level)
        else:
            print(f"Warning: Invalid difficulty {diff}. Defaulting to easy.")
            levels["easy"].append(level)
    else:
        print(f"Warning: Skipping invalid level - grid: {len(grid)}x{len(grid[0]) if grid else 0}, taxi: {taxi}, passenger: {passenger}, dropoff: {dropoff}")

# Ensure at least one level
if not any(levels[diff] for diff in levels):
    default_level = {
        "difficulty": "easy",
        "grid": [["0" if i != 0 and i != 9 and j != 0 and j != 9 else "#" for j in range(10)] for i in range(10)],
        "taxi": [1, 1],
        "passenger": [3, 3],
        "dropoff": [8, 8]
    }
    levels["easy"].append(default_level)
    with open('./game_levels.json', 'w') as f:
        json.dump([default_level], f, indent=2)
    print("Created default level in game_levels.json")

# Log available levels
for diff in levels:
    print(f"Loaded {len(levels[diff])} levels for difficulty: {diff}")

# Load images
symbol_images = {
    'T': 'taxi.png',
    'P': 'player.png',
    'D': 'goal.png',
    '#': 'wall.png',
    '0': 'empty.png',
    'G': 'danger.png',
    'W': 'wall.png'
}
base64_images = {}
for symbol, filename in symbol_images.items():
    image_path = os.path.join('static', 'assets', filename)
    if os.path.exists(image_path):
        with open(image_path, 'rb') as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
            base64_images[symbol] = f"data:image/png;base64,{encoded}"
    else:
        print(f"Warning: Image file {image_path} not found")
        base64_images[symbol] = base64_images.get('0', "")

# Environment setup
env = None
human_env = None
model = None
try:
    flat_levels = []
    for diff in levels:
        flat_levels.extend(levels[diff])
    if not flat_levels:
        raise ValueError("No valid levels available for environment")
    env = DummyVecEnv([lambda: GridGameEnv(levels=flat_levels, use_curriculum=False)])
    if os.path.exists("./taxi_logs/vecnormalize.pkl"):
        env = VecNormalize.load("./taxi_logs/vecnormalize.pkl", env)
        env.training = False
        env.norm_reward = False
    model = PPO.load("./taxi_logs/final_model.zip", env=env)
    human_env = GridGameEnv(levels=flat_levels, use_curriculum=False)
    print("Environment initialized successfully")
except Exception as e:
    print(f"Error initializing environment: {e}")
    try:
        default_level = levels["easy"][0] if levels["easy"] else {
            "difficulty": "easy",
            "grid": [["0" if i != 0 and i != 9 and j != 0 and j != 9 else "#" for j in range(10)] for i in range(10)],
            "taxi": [1, 1],
            "passenger": [3, 3],
            "dropoff": [8, 8]
        }
        env = DummyVecEnv([lambda: GridGameEnv(levels=[default_level], use_curriculum=False)])
        model = PPO.load("./taxi_logs/final_model.zip", env=env)
        human_env = GridGameEnv(levels=[default_level], use_curriculum=False)
        print("Environment initialized with default level")
    except Exception as e2:
        print(f"Fallback environment initialization failed: {e2}")
        env = None
        model = None
        human_env = None

def get_base_env(env):
    if env is None:
        return None
    while hasattr(env, 'envs'):
        env = env.envs[0]
    return env

@app.before_request
def initialize_session():
    if 'session_state' not in session:
        session['session_state'] = {
            "page": "home",
            "agent_steps": 0,
            "agent_reward": None,
            "human_steps": 0,
            "human_reward": 0.0,
            "human_done": False,
            "human_env_state": None,
            "current_difficulty": "easy",
            "initial_level": None,
            "obs": None,
            "base64_images": base64_images,
            "agent_played_level": None,
            "human_last_grid": None
        }
        if levels["easy"] and env and human_env:
            try:
                session['session_state']['initial_level'] = np.random.choice(levels["easy"])
                base_env = get_base_env(env)
                base_env.set_level(session['session_state']['initial_level'])
                human_env.set_level(session['session_state']['initial_level'])
                reset_result = env.reset()
                obs = reset_result if not isinstance(reset_result, tuple) else reset_result[0]
                obs = np.array(obs).flatten() if obs is not None else np.zeros(100, dtype=np.float32)
                if obs.shape != (100,):
                    print(f"Warning: Reshaping obs from {obs.shape} to (100,) in initialize_session")
                    obs = obs.reshape(100)
                session['session_state']['obs'] = obs.tolist()
                session['session_state']['human_env_state'] = human_env.reset()[0]
                print("Session initialized with default level")
            except Exception as e:
                print(f"Error initializing session environment: {e}")
                session['session_state']['obs'] = None
                session['session_state']['initial_level'] = None
        else:
            print("Warning: No levels or environment available for session initialization")
        session.modified = True

@app.route('/')
def home():
    session['session_state']['page'] = "home"
    session.modified = True
    return render_template('splash.html')

@app.route('/game')
def game():
    session['session_state']['page'] = "game"
    if not session['session_state']['initial_level'] and levels["easy"] and env and human_env:
        try:
            session['session_state']['initial_level'] = np.random.choice(levels["easy"])
            base_env = get_base_env(env)
            base_env.set_level(session['session_state']['initial_level'])
            human_env.set_level(session['session_state']['initial_level'])
            reset_result = env.reset()
            obs = reset_result if not isinstance(reset_result, tuple) else reset_result[0]
            obs = np.array(obs).flatten() if obs is not None else np.zeros(100, dtype=np.float32)
            if obs.shape != (100,):
                print(f"Warning: Reshaping obs from {obs.shape} to (100,) in game route")
                obs = obs.reshape(100)
            session['session_state']['obs'] = obs.tolist()
            session['session_state']['human_env_state'] = human_env.reset()[0]
            print("Game route initialized new level")
        except Exception as e:
            print(f"Error setting initial level in game route: {e}")
            session['session_state']['obs'] = None
            session['session_state']['initial_level'] = None
    session.modified = True
    return render_template('game.html',
                          levels=levels,
                          initial_level=session['session_state'].get('initial_level'),
                          base64_images=session['session_state'].get('base64_images'),
                          agent_steps=session['session_state'].get('agent_steps'),
                          agent_reward=session['session_state'].get('agent_reward'),
                          human_steps=session['session_state'].get('human_steps'),
                          human_reward=session['session_state'].get('human_reward'),
                          human_done=session['session_state'].get('human_done'),
                          current_difficulty=session['session_state'].get('current_difficulty'))

@app.route('/api/get_images', methods=['GET'])
def get_images():
    return jsonify(base64_images)

@app.route('/api/reset_game', methods=['POST'])
def reset_game():
    print(f"Received /api/reset_game request with data: {request.get_json()}")
    data = request.get_json() or {}
    requested_difficulty = data.get('difficulty', 'easy')
    if requested_difficulty not in levels:
        requested_difficulty = "easy"
    session['session_state']['current_difficulty'] = requested_difficulty
    if not levels[requested_difficulty]:
        session['session_state']['current_difficulty'] = "easy"
        if not levels["easy"]:
            session['session_state']['initial_level'] = None
            session['session_state']['obs'] = None
            session['session_state']['agent_played_level'] = None
            session['session_state']['human_last_grid'] = None
            session.modified = True
            print(f"Error: No levels available for difficulty {requested_difficulty} or easy")
            return jsonify({
                "success": False,
                "error": f"No levels available for difficulty {requested_difficulty} or easy",
                "initial_level": None,
                "agent_steps": 0,
                "agent_reward": None,
                "human_steps": 0,
                "human_reward": 0.0,
                "human_done": False,
                "current_difficulty": session['session_state']['current_difficulty'],
                "obs": None
            })
    try:
        if not env or not human_env:
            raise ValueError("Environment not initialized. Check server logs for initialization errors.")
        session['session_state']['initial_level'] = np.random.choice(levels[session['session_state']['current_difficulty']])
        session['session_state']['agent_played_level'] = None
        session['session_state']['agent_steps'] = 0
        session['session_state']['agent_reward'] = None
        session['session_state']['human_steps'] = 0
        session['session_state']['human_reward'] = 0.0
        session['session_state']['human_done'] = False
        session['session_state']['human_env_state'] = None
        session['session_state']['human_last_grid'] = None
        base_env = get_base_env(env)
        if base_env is None:
            raise ValueError("Base environment not available")
        base_env.set_level(session['session_state']['initial_level'])
        human_env.set_level(session['session_state']['initial_level'])
        reset_result = env.reset()
        print(f"env.reset() returned: {reset_result}")
        obs = reset_result if not isinstance(reset_result, tuple) else reset_result[0]
        obs = np.array(obs).flatten() if obs is not None else np.zeros(100, dtype=np.float32)
        if obs.shape != (100,):
            print(f"Warning: Reshaping obs from {obs.shape} to (100,) in reset_game")
            obs = obs.reshape(100)
        session['session_state']['obs'] = obs.tolist()
        human_reset_result = human_env.reset()
        session['session_state']['human_env_state'] = human_reset_result[0] if isinstance(human_reset_result, tuple) else human_reset_result
        session.modified = True
        response = {
            "success": True,
            "initial_level": session['session_state']['initial_level'],
            "agent_steps": session['session_state']['agent_steps'],
            "agent_reward": session['session_state']['agent_reward'],
            "human_steps": session['session_state']['human_steps'],
            "human_reward": session['session_state']['human_reward'],
            "human_done": session['session_state']['human_done'],
            "current_difficulty": session['session_state']['current_difficulty'],
            "obs": session['session_state']['obs']
        }
        print(f"Reset game response: success={response['success']}, level_grid_size={len(response['initial_level']['grid'])}x{len(response['initial_level']['grid'][0])}, obs_length={len(response['obs'])}")
        return jsonify(response)
    except Exception as e:
        session['session_state']['initial_level'] = None
        session['session_state']['obs'] = None
        session['session_state']['agent_played_level'] = None
        session['session_state']['human_last_grid'] = None
        session.modified = True
        print(f"Error resetting game: {e}")
        response = {
            "success": False,
            "error": f"Failed to reset game: {str(e)}",
            "initial_level": None,
            "agent_steps": 0,
            "agent_reward": None,
            "human_steps": 0,
            "human_reward": 0.0,
            "human_done": False,
            "current_difficulty": session['session_state']['current_difficulty'],
            "obs": None
        }
        print(f"Reset game error response: {response}")
        return jsonify(response)

@app.route('/api/play_agent', methods=['POST'])
def play_agent():
    if env is None or session['session_state']['obs'] is None or session['session_state']['initial_level'] is None:
        print("Play agent failed: Environment or session state invalid")
        return jsonify({"success": False, "error": "No level loaded or environment not initialized"})
    obs = np.array(session['session_state']['obs'], dtype=np.float32)
    steps_data = []
    total_reward = 0.0
    done = False
    max_steps = 50
    base_env = get_base_env(env)
    if not base_env:
        print("Play agent failed: Base environment not initialized")
        return jsonify({"success": False, "error": "Environment not initialized"})
    try:
        # Reconstruct a valid level dictionary from human_last_grid if available
        level_to_use = session['session_state']['initial_level'].copy()
        if session['session_state']['human_last_grid']:
            level_to_use = {
                "grid": ansi_to_numeric_grid(session['session_state']['human_last_grid']),
                "taxi": level_to_use['taxi'],  # Retain original taxi position
                "passenger": level_to_use['passenger'],  # Retain original passenger position
                "dropoff": level_to_use['dropoff'],  # Retain original dropoff position
                "difficulty": session['session_state']['current_difficulty']
            }
        base_env.set_level(level_to_use)
        human_env.set_level(level_to_use)
        reset_result = env.reset()
        obs = reset_result if not isinstance(reset_result, tuple) else reset_result[0]
        obs = np.array(obs).flatten() if obs is not None else np.zeros(100, dtype=np.float32)
        if obs.shape != (100,):
            print(f"Warning: Reshaping obs from {obs.shape} to (100,) in play_agent with human grid")
            obs = obs.reshape(100)
        for step in range(max_steps):
            action, _ = model.predict(obs.reshape(1, -1), deterministic=True)
            step_result = env.step([action.item() if hasattr(action, 'item') else action[0]])
            # Handle 4-tuple (obs, reward, done, info) or 5-tuple (obs, reward, terminated, truncated, info)
            if len(step_result) == 4:
                obs, reward, done, info = step_result
                terminated = done
                truncated = False
            elif len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                raise ValueError(f"Unexpected step result length: {len(step_result)}")
            total_reward += reward[0]
            grid = ansi_to_symbol_grid(base_env.render(mode="ansi"))
            steps_data.append({"step": step + 1, "grid": grid, "reward": float(reward[0])})
            obs = obs[0] if len(obs.shape) > 1 else obs
            if obs is None or obs.size == 0:
                print(f"Warning: Invalid obs at step {step}, using zeros")
                obs = np.zeros(100, dtype=np.float32)
            elif obs.shape != (100,):
                print(f"Warning: Reshaping obs from {obs.shape} to (100,) at step {step}")
                obs = obs.reshape(100)
            session['session_state']['obs'] = obs.tolist()
            if done:
                break
        session['session_state']['agent_steps'] = step + 1
        session['session_state']['agent_reward'] = float(total_reward)
        session['session_state']['agent_played_level'] = level_to_use.copy()
        session['session_state']['agent_played_level']['grid'] = grid
        session.modified = True
        print(f"Agent play completed: steps={step + 1}, reward={total_reward}")
        return jsonify({"success": True, "steps": step + 1, "reward": float(total_reward), "steps_data": steps_data})
    except Exception as e:
        print(f"Error in play_agent: {e}")
        return jsonify({"success": False, "error": f"Agent play failed: {str(e)}"})

@app.route('/api/move_human', methods=['POST'])
def move_human():
    if human_env is None or session['session_state']['initial_level'] is None:
        print("Move human failed: Environment or session state invalid")
        return jsonify({"success": False, "error": "No level loaded or environment not initialized"})
    if session['session_state']['human_env_state'] is None:
        if session['session_state']['agent_played_level']:
            human_env.set_level(session['session_state']['agent_played_level'])
            session['session_state']['initial_level'] = session['session_state']['agent_played_level'].copy()
        else:
            human_env.set_level(session['session_state']['initial_level'])
        human_reset_result = human_env.reset()
        session['session_state']['human_env_state'] = human_reset_result[0] if isinstance(human_reset_result, tuple) else human_reset_result
    data = request.get_json()
    action = data.get('action')
    if action not in ['up', 'left', 'down', 'right']:
        print(f"Move human failed: Invalid action {action}")
        return jsonify({"success": False, "error": "Invalid action"})
    action_map = {'up': 0, 'left': 2, 'down': 1, 'right': 3}
    try:
        obs, reward, terminated, truncated, info = human_env.step(action_map[action])
        done = terminated or truncated
        session['session_state']['human_env_state'] = obs
        session['session_state']['human_steps'] += 1
        session['session_state']['human_reward'] += float(reward)
        session['session_state']['human_done'] = done or session['session_state']['human_steps'] >= 50
        session['session_state']['human_last_grid'] = ansi_to_symbol_grid(human_env.render(mode="ansi"))
        session.modified = True
        print(f"Human move: action={action}, steps={session['session_state']['human_steps']}, reward={session['session_state']['human_reward']}, done={done}")
        return jsonify({
            "success": True,
            "steps": session['session_state']['human_steps'],
            "reward": session['session_state']['human_reward'],
            "done": session['session_state']['human_done'],
            "grid": session['session_state']['human_last_grid']
        })
    except Exception as e:
        print(f"Error in move_human: {e}")
        return jsonify({"success": False, "error": f"Human move failed: {str(e)}"})

def ansi_to_symbol_grid(ansi_grid):
    symbol_grid = []
    for row in ansi_grid.strip().split('\n'):
        if row.strip():
            symbol_row = []
            for char in row:
                symbol = {
                    'T': 'T',
                    'P': 'P',
                    'D': 'D',
                    '#': '#',
                    '0': '0',
                    '1': '#',
                    'G': 'G',
                    'W': 'W'
                }.get(char, '0')
                symbol_row.append(symbol)
            symbol_grid.append(symbol_row)
    while len(symbol_grid) < 10:
        symbol_grid.append(['0'] * 10)
    symbol_grid = symbol_grid[:10]
    for row in symbol_grid:
        while len(row) < 10:
            row.append('0')
        row[:] = row[:10]
    return symbol_grid

def ansi_to_numeric_grid(symbol_grid):
    """Convert symbolic grid to numeric grid compatible with GridGameEnv."""
    numeric_grid = []
    for i in range(10):
        numeric_row = []
        for j in range(10):
            cell = symbol_grid[i][j]
            if cell == 'T':
                numeric_row.append(2)  # Taxi
            elif cell == 'P':
                numeric_row.append(3)  # Passenger
            elif cell == 'D':
                numeric_row.append(4)  # Dropoff
            elif cell in ['#', 'W']:
                numeric_row.append(1)  # Wall
            elif cell == '0':
                numeric_row.append(0)  # Empty
            else:
                numeric_row.append(0)  # Default to empty for unknown symbols
        numeric_grid.append(numeric_row)
    return numeric_grid

if __name__ == '__main__':
    app.run(debug=True)