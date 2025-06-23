from gym_pcgrl.envs.probs.problem import Problem

class TaxiProblem(Problem):
    """
    A Procedural Content Generation (PCG) version of the Taxi problem.
    Similar to OpenAI Gym's Taxi-v3 but integrated into the PCGRL framework.
    """

    def __init__(self):
        super().__init__()
        self._width = 5
        self._height = 5
        self._tile_size = 16
        self._passenger_locations = [(0, 0), (0, 4), (4, 0), (4, 4)]
        self._destination_locations = [(0, 0), (0, 4), (4, 0), (4, 4)]
        self._obstacles = []
        self.reset({})

    def get_tile_types(self):
        """Defines the different tile types in the environment."""
        return ["empty", "taxi", "passenger", "destination", "obstacle"]

    def reset(self, start_stats):
        """Resets the environment and places the taxi, passenger, and destination randomly."""
        import random
        self._start_stats = start_stats
        self.taxi_pos = (random.randint(0, 4), random.randint(0, 4))
        self.passenger_pos = random.choice(self._passenger_locations)
        self.destination_pos = random.choice(self._destination_locations)
        self._obstacles = [(random.randint(1, 3), random.randint(1, 3))]  # Random obstacle

    def get_stats(self, map):
        """Returns the current stats of the Taxi environment."""
        return {
            "taxi_pos": self.taxi_pos,
            "passenger_pos": self.passenger_pos,
            "destination_pos": self.destination_pos
        }

    def get_reward(self, new_stats, old_stats):
        """Calculates rewards based on actions taken."""
        if new_stats["taxi_pos"] == new_stats["passenger_pos"]:
            return 10  # Reward for picking up passenger
        elif new_stats["taxi_pos"] == new_stats["destination_pos"]:
            return 20  # Reward for successfully dropping off
        return -1  # Small penalty for movement

    def get_episode_over(self, new_stats, old_stats):
        """Checks if the episode should end (passenger dropped off)."""
        return new_stats["taxi_pos"] == new_stats["destination_pos"]

    def get_debug_info(self, new_stats, old_stats):
        """Provides debugging info."""
        return {
            "taxi_position": new_stats["taxi_pos"],
            "passenger_position": new_stats["passenger_pos"],
            "destination_position": new_stats["destination_pos"]
        }
