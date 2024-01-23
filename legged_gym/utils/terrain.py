import numpy as np
from numpy.random import choice
from scipy import interpolate
import noise
import random
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
from legged_gym import LEGGED_GYM_ROOT_DIR

import os
import time # Ege
from tqdm import tqdm # Ege
import pickle # Ege

from isaacgym import terrain_utils
from configs.definitions import TerrainConfig

class Terrain:
    def __init__(self, cfg: TerrainConfig) -> None:

        self.cfg = cfg
        self.type = cfg.mesh_type
        # TODO: add these to config
        self.calc_stats = True
        self.log_stats = False
        # Ege - refer to 'resources/terrain' for all terrain artifacts (heightmaps, images, stats etc.)
        #       (make the folder if it doesn't exist yet)
        self.resources_dirname = os.path.join(LEGGED_GYM_ROOT_DIR, 'resources', 'terrain')
        if not os.path.isdir(self.resources_dirname):
            os.makedirs(self.resources_dirname, exist_ok=True)  # creates dirs for the path if needed

        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length
        self.env_width = cfg.terrain_width
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        # Ege
        self.stats = {}  # used to store extra info per env/tile (one cell of the grid)
        self.slope_normals = np.tile(np.array([0,0,1], dtype=float), (cfg.num_rows, cfg.num_cols, 1))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale)
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale)

        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)

    # ============= Terrain Multiplexing ============================
        #TODO: needs to be cleaned up and clearer
        tile_func = getattr(self, "_tile_" + cfg.terrain_type)
        self.from_tiles(tile_func=tile_func, tile_args=cfg.terrain_kwargs)
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_threshold)
        # Ege - add slope normals if a well-defined slope is available for each tile
        # TODO - super weird special case, make generic
        if 'slope' in self.stats:
            self.slope_normals = slope_normal(self.stats['slope'])
        # Ege - log stats with other experiment logs if needed
        if self.log_stats:
            with open("terrain_stats.pickle", "wb+") as f:
                pickle.dump(self.stats, f)

    # ============= Saving/Loading Helpers ============================
    def save_terrain(self, terrain_name, heightmaps, stats=None, inner_path=""):
        with open(os.path.join(self.resources_dirname, inner_path, f"{terrain_name}.pickle"), "wb+") as f:
            pickle.dump({"heightmaps": heightmaps, "stats": stats}, f)

    def load_terrain(self, terrain_name, inner_path=""):
        with open(os.path.join(self.resources_dirname, inner_path, f"{terrain_name}.pickle"), "rb") as f:
            data = pickle.load(f)
            if type(data) is not dict or "heightmaps" not in data:
                raise ValueError("Terrain pickle should be a dict containing a " +
                                 "heightmap grid with the key 'heightmaps'!")
            heightmaps = data["heightmaps"]
            grid_dims = (self.cfg.num_rows, self.cfg.num_cols)
            if heightmaps.shape[:2] != grid_dims:
                raise ValueError(f"Number of rows/cols in loaded terrain {heightmaps.shape[:2]} " +
                                 f"doesn't match the config {grid_dims}!")
            env_dims = (self.width_per_env_pixels, self.length_per_env_pixels)
            if heightmaps.shape[2:4] != env_dims:
                raise ValueError(f"Size per env in pixels in loaded terrain {heightmaps.shape[2:4]} " +
                                 f"doesn't match the config {env_dims}!")
            if 'stats' in data:
                self.stats = data['stats']
            def load_tile(terrain, i, j):
                terrain.height_field_raw[:,:] = heightmaps[i][j][:env_dims[0], :env_dims[1]]
            self.from_tiles(load_tile)
        
    # ============= Terrain Union/Duplication ==========================
    def from_tiles(self, tile_func, tile_args={}, save_terrain=False, name="terrain"):
        width, length = self.width_per_env_pixels, self.length_per_env_pixels
        if save_terrain:
            heightmaps = np.zeros((self.cfg.num_rows, self.cfg.num_cols, width, length))
        for k in range(self.num_sub_terrains):
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))
            terrain = terrain_utils.SubTerrain(f"{name}_row_{i}_col_{j}",
                              width=width,
                              length=length,
                              vertical_scale=self.cfg.vertical_scale,
                              horizontal_scale=self.cfg.horizontal_scale)
            tile_stats = tile_func(terrain, i, j, **tile_args)
            if save_terrain:
                heightmaps[i, j, :width, :width] = terrain.height_field_raw
            if self.calc_stats:
                for key, stat in tile_stats.items():
                    if key not in self.stats:
                        array_size = [self.cfg.num_rows, self.cfg.num_cols]
                        if type(stat) == np.ndarray and np.ndim(stat) > 0:
                            array_size.append(np.array(stat).shape)
                        self.stats[key] = np.full(array_size, np.nan)
                    self.stats[key][i][j] = tile_stats[key]
            self.add_terrain_to_map(terrain, i, j)
        if save_terrain:
            self.save_terrain(name, heightmaps, self.stats)

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # map coordinate system
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 1) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 1) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 1) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 1) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    # ============= Tile Helper Functions ============================
    def locomotion_tilemaker(self, terrain, choice, difficulty):
        # TODO: why does this exist if we also have the terrain multiplexing stuff at the top?
        # difficulty is between 0 and 1
        slope = difficulty * 0.4 # in radians (affects straight slope and noisy terrain)
        step_height = 0.05 + 0.18 * difficulty # 23 cm height (may want to decrease for blind locomotion, e.g., to 13 cm)
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < self.proportions[0]:
            if choice < self.proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
        elif choice < self.proportions[1]:
            terrain_utils.pyramid_sloped_terrain(terrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < self.proportions[3]:
            if choice<self.proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < self.proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(terrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < self.proportions[5]:
            terrain_utils.stepping_stones_terrain(terrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < self.proportions[6]:
            self.gap_tilemaker(terrain, gap_size=gap_size, platform_size=3.)
        elif choice < self.proportions[7]:
            self.pit_tilemaker(terrain, depth=pit_depth, platform_size=4.)
        elif choice < self.proportions[8]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-self.cfg.terrain_noise_magnitude,
                                                 max_height=self.cfg.terrain_noise_magnitude, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < self.proportions[9]:
            terrain_utils.random_uniform_terrain(terrain, min_height=-0.05, max_height=0.05,
                                                 step=self.cfg.terrain_smoothness, downsampled_scale=0.2)
            terrain.height_field_raw[0:terrain.length // 2, :] = 0
    
    def gap_tilemaker(self, terrain, gap_size, platform_size):
        gap_size = int(gap_size / terrain.horizontal_scale)
        platform_size = int(platform_size / terrain.horizontal_scale)

        center_x = terrain.length // 2
        center_y = terrain.width // 2
        x1 = (terrain.length - platform_size) // 2
        x2 = x1 + gap_size
        y1 = (terrain.width - platform_size) // 2
        y2 = y1 + gap_size

        terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
        terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

    def pit_tilemaker(self, terrain, gap_size, platform_size=1.):
        gap_size = int(gap_size / terrain.horizontal_scale)
        platform_size = int(platform_size / terrain.horizontal_scale)

        center_x = terrain.length // 2
        center_y = terrain.width // 2
        x1 = (terrain.length - platform_size) // 2
        x2 = x1 + gap_size
        y1 = (terrain.width - platform_size) // 2
        y2 = y1 + gap_size

        terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
        terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0
    
    # ============= Tile Implementations  ============================
    def _tile_locomotion_random(self, terrain, row, col):
        choice = np.random.uniform(0, 1)
        difficulty = np.random.choice([0.5, 0.75, 0.9])
        self.locomotion_tilemaker(terrain, choice, difficulty)

    def _tile_locomotion_curriculum(self, terrain, row, col):
        difficulty = row / self.cfg.num_rows
        choice = col / self.cfg.num_cols + 0.001
        self.locomotion_tilemaker(terrain, choice, difficulty)

    def _tile_valley(self, terrain, row, col):
        width = terrain.width
        gap_size = int(0 * width)
        slope_size = int(0.5 * width)
        x_left = slope_size
        x_right = x_left + gap_size
        slope = 0.1 * row
        slope *= terrain.horizontal_scale / terrain.vertical_scale
        #noise_level = np.ceil(j * j) * (1 + slope ** 2) / 5
        noise_level = col * np.sqrt(1 + slope ** 2) / 6
        noises = np.random.normal(0, noise_level, size=(2 * x_left, width)) #- (noise_level / 2)

        # flat part at the bottom
        terrain.height_field_raw[x_left : x_right, :] = -1. * slope * slope_size
        # banks of the valley
        for x in range(x_left):
            for y in range(0, width):
                terrain.height_field_raw[x, y] = (-1. * slope * x) + noises[x][y]
                terrain.height_field_raw[width - x - 1, y] = (-1. * slope * x) + noises[x + x_left][y]
        if self.calc_stats:
            roughness = residual_variance(terrain.height_field_raw[:x_left, :], terrain.vertical_scale, terrain.horizontal_scale, width*width) / 2 #terrain_roughness_index(terrain.height_field_raw[:x_left, :]) / 2 
            roughness += residual_variance(terrain.height_field_raw[x_right:, :], terrain.vertical_scale, terrain.horizontal_scale, width*width) / 2 # terrain_roughness_index(terrain.height_field_raw[x_right:, :]) / 2
            return {"roughness": roughness}
        return None

    def _tile_semivalley(self, terrain, row, col):
        width = terrain.width
        slope = 0.1 * row
        roughness_level = 0.2 * col
        heightmap_slope = slope * terrain.horizontal_scale / terrain.vertical_scale

        #noise_level = np.ceil(j * j) * (1 + slope ** 2) / 5
        noise_level = roughness_level * np.sqrt(2 * np.pi) * (1 + slope ** 2) #** (3/4)
        noises = np.random.normal(0, noise_level, size=(terrain.width, terrain.width)) #- (noise_level / 2)
        for x in range(terrain.width):
            for y in range(terrain.width):
                terrain.height_field_raw[x, y] = (-1. * heightmap_slope * x) + noises[x][y]
        if self.calc_stats:
            area = (width ** 2) * (terrain.horizontal_scale ** 2) * np.sqrt(1 + slope ** 2)
            roughness = residual_variance(terrain.height_field_raw, terrain.vertical_scale, terrain.horizontal_scale, area) #terrain_roughness_index(terrain.height_field_raw) 
            return {"slope": slope, "roughness": roughness}
        return None

    def _tile_perlin_noise(self, terrain, row, col):
        scale = 30.0
        height_multiplier = 50 * row
        octaves = 7
        persistence = 0.1 * col
        lacunarity = 2.0
        seed = np.random.randint(0,100)
        width = terrain.width
        world = np.zeros((width, width))

        for x in range(10, width-10):
            for y in range(10, width-10):
                world[x][y] = noise.pnoise2(x/scale,
                                            y/scale,
                                            octaves=octaves,
                                            persistence=persistence,
                                            lacunarity=lacunarity,
                                            repeatx=1024,
                                            repeaty=1024,
                                            base=seed) 
                world[x][y] *= height_multiplier
        terrain.height_field_raw = world.astype('int32')

    def _tile_slope(self, terrain, row, col, slope=1.0):
        slope *= terrain.horizontal_scale / terrain.vertical_scale
        for x in range(terrain.width):
            terrain.height_field_raw[x, :] = (row * terrain.width + x) * slope

    def _tile_flat(self, terrain, row, col):
        terrain.height_field_raw[:] = 0

# ======== Terrain Stats Helper Functions ========

def rotate_to_best_fit_plane(heightmap, vertical_scale, horizontal_scale):
    points = np.stack(([], [], []), axis=1)
    for index, height in np.ndenumerate(heightmap):
        points = np.vstack((points, np.array([index[0], index[1], height * vertical_scale / horizontal_scale])))
    pca = PCA(n_components=2)
    pca.fit(points)
    projected = pca.inverse_transform(pca.transform(points))
    distances = np.linalg.norm(points - projected, axis=1) * np.sign(points[:,2] - projected[:,2])
    rotated_heights = np.zeros_like(heightmap)
    for i in range(points.shape[0]):
        x, y, _ = points[i]
        rotated_heights[int(x), int(y)] = distances[i]
    return rotated_heights

def residual_variance(heightmap, vertical_scale, horizontal_scale, total_area):
    points = np.stack(([], [], []), axis=1)
    for index, height in np.ndenumerate(heightmap):
        points = np.vstack((points, np.array([index[0] * horizontal_scale, index[1] * horizontal_scale, height * vertical_scale])))
    pca = PCA(n_components=2)
    pca.fit(points)
    projected = pca.inverse_transform(pca.transform(points))
    # (0.4 / (0.5 * 0.3)) here is a factor based on the (robot height) / (robot area in the xy-plane) when standing upright
    return (0.4 / (0.5 * 0.3)) * np.sum(np.linalg.norm(points - projected, axis=1)) / total_area

# looks at the average difference between each pixel that has 8 neighbors
# with those neighbors, then takes the average of that as a measure of roughness
def terrain_roughness_index(heightmap):
    heightmap = rotate_to_best_fit_plane(heightmap, 1, 1)
    total_diff = 0.0
    for i in range(1, heightmap.shape[0] - 1):
        for j in range(1, heightmap.shape[1] - 1):
            neighbors = np.array([heightmap[i+1, j-1], heightmap[i+1, j], heightmap[i+1, j+1],
                                  heightmap[i-1, j-1], heightmap[i-1, j], heightmap[i-1, j+1],
                                  heightmap[i, j+1], heightmap[i, j-1]])
            total_diff += np.linalg.norm(heightmap[i][j] - neighbors)
    return total_diff / float((heightmap.shape[0] - 1) * (heightmap.shape[1] - 1))

def slope_normal(slopes):
    denoms = np.sqrt(1 + slopes ** 2)
    return np.stack((-slopes / denoms, np.zeros_like(slopes), 1 / denoms), axis=-1)
