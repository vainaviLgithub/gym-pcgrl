from gymnasium.utils import seeding
from PIL import Image

"""
The base class for all the problems that can be handled by the interface
"""
class Problem:
    """
    Constructor for the problem that initializes all the basic parameters
    """
    def __init__(self):
        self._width = 9
        self._height = 9
        tiles = self.get_tile_types()
        self._prob = {tile: 1.0 / len(tiles) for tile in tiles}  # Use a dictionary for probabilities
        
        self._border_size = (1, 1)
        self._border_tile = tiles[0]
        self._tile_size = 16
        self._graphics = None

    """
    Seeding the used random variable to get the same result. If the seed is None,
    it will seed it with a random start.
    """
    def seed(self, seed=None):
        self._random, seed = seeding.np_random(seed)
        return seed

    """
    Resets the problem to the initial state and saves the start_stats from the starting map.
    """
    def reset(self, start_stats):
        self._start_stats = start_stats

    """
    Get a list of all the different tile names
    """
    def get_tile_types(self):
        raise NotImplementedError('get_tile_types is not implemented')

    """
    Adjust the parameters for the current problem
    """
    def adjust_param(self, **kwargs):
        self._width = kwargs.get('width', self._width)
        self._height = kwargs.get('height', self._height)
        prob = kwargs.get('probs')
        if prob is not None:
            for t in prob:
                if t in self._prob:
                    self._prob[t] = prob[t]

    """
    Get the current stats of the map
    """
    def get_stats(self, map):
        raise NotImplementedError('get_stats is not implemented')

    """
    Get the current game reward between two stats
    """
    def get_reward(self, new_stats, old_stats):
        raise NotImplementedError('get_reward is not implemented')

    """
    Uses the stats to check if the problem ended (episode_over)
    """
    def get_episode_over(self, new_stats, old_stats):
        raise NotImplementedError('get_episode_over is not implemented')

    """
    Get any debug information needed to be printed
    """
    def get_debug_info(self, new_stats, old_stats):
        raise NotImplementedError('get_debug_info is not implemented')

    """
    Get an image on how the map will look like for a specific map
    """
    def render(self, map):
        tiles = self.get_tile_types()
        if self._graphics is None:
            self._graphics = {}
            for i in range(len(tiles)):
                color = (int(i * 255 / len(tiles)), int(i * 255 / len(tiles)), int(i * 255 / len(tiles)), 255)
                self._graphics[tiles[i]] = Image.new("RGBA", (self._tile_size, self._tile_size), color)

        full_width = len(map[0]) + 2 * self._border_size[0]
        full_height = len(map) + 2 * self._border_size[1]
        lvl_image = Image.new("RGBA", (full_width * self._tile_size, full_height * self._tile_size), (0, 0, 0, 255))

        for y in range(full_height):
            for x in range(self._border_size[0]):
                lvl_image.paste(self._graphics[self._border_tile],
                                (x * self._tile_size, y * self._tile_size))
                lvl_image.paste(self._graphics[self._border_tile],
                                ((full_width - x - 1) * self._tile_size, y * self._tile_size))

        for x in range(full_width):
            for y in range(self._border_size[1]):
                lvl_image.paste(self._graphics[self._border_tile],
                                (x * self._tile_size, y * self._tile_size))
                lvl_image.paste(self._graphics[self._border_tile],
                                (x * self._tile_size, (full_height - y - 1) * self._tile_size))

        for y in range(len(map)):
            for x in range(len(map[y])):
                lvl_image.paste(self._graphics[map[y][x]],
                                ((x + self._border_size[0]) * self._tile_size, (y + self._border_size[1]) * self._tile_size))
        
        return lvl_image