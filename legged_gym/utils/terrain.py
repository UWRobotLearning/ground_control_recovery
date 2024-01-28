import numpy as np
import noise
from sklearn.decomposition import PCA
from legged_gym import LEGGED_GYM_ROOT_DIR

import os
from dataclasses import dataclass, field # Ege
from typing import Optional # Ege
import pickle # Ege

from isaacgym import terrain_utils
from configs.definitions import TerrainConfig

class Terrain:
    def __init__(self, cfg: TerrainConfig) -> None:
        self.cfg = cfg
        # skip terrain generation if mesh type already specifies terrain
        if self.cfg.mesh_type in ["none", 'plane']:
            return
        # Ege - refer to 'resources/terrain' for all terrain artifacts (heightmaps, images, stats etc.)
        #       (make the folder if it doesn't exist yet)
        self.resources_dirname = os.path.join(LEGGED_GYM_ROOT_DIR, 'resources', 'terrain')
        if not os.path.isdir(self.resources_dirname):
            os.makedirs(self.resources_dirname, exist_ok=True)  # creates dirs for the path if needed
        # calculated terrain dimensions
        self.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.tile_width_px = int(self.cfg.tile_width / cfg.horizontal_scale)
        self.tile_length_px = int(self.cfg.tile_length / cfg.horizontal_scale)
        self.border = int(cfg.border_size/self.cfg.horizontal_scale)
        self.tot_cols = int(cfg.num_cols * self.tile_width_px) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.tile_length_px) + 2 * self.border
        # terrain info data structures
        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)
        self.tile_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))
        self.vertices, self.triangles = None, None
        # Ege
        self.stats = {}  # used to store extra info per tile (one cell of the grid)
        # Ege - if a load path is given, load the terrain from there, otherwise create terrain based on tile choice
        if cfg.load_inner_path is not None:
            self.load_terrain()
        else:
            tile_func = getattr(self, "_tile_" + cfg.terrain_type)
            self.from_tiles(tile_func=tile_func, tile_args=cfg.terrain_kwargs)
        # Create heightsamples and populate trimesh if needed
        self.heightsamples = self.height_field_raw
        if self.cfg.mesh_type=="trimesh" and (self.vertices is None or self.triangles is None):
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_threshold)
        # Ege - save terrain as specified in config if it's not already loaded
        if cfg.save_terrain and cfg.load_inner_path is None:
            self.save_terrain(terrain_name=cfg.terrain_type)
        # Ege - log stats with other experiment logs if needed
        if self.cfg.log_stats:
            with open("terrain_stats.pickle", "wb+") as f:
                pickle.dump(self.stats, f)

    # ============= Saving/Loading Helpers ============================
                
    def get_dims(self):
        return SavedTerrain.Dimensions(
                num_rows=self.cfg.num_rows, 
                num_cols=self.cfg.num_cols,
                tile_width_px=self.tile_width_px,
                tile_length_px=self.tile_length_px,
                border_px=self.border,
                horizontal_scale=self.cfg.horizontal_scale,
                vertical_scale=self.cfg.vertical_scale
            )
    
    def save_terrain(self, terrain_name, save_trimesh=True, save_stats=True, inner_path=""):
        trimesh_vertices, trimesh_faces = (self.vertices, self.triangles) if save_trimesh else (None, None)
        terrain_to_save = SavedTerrain(
            name=terrain_name,
            heightmap=self.height_field_raw,
            tile_origins=self.tile_origins,
            dims=self.get_dims(),
            trimesh_vertices=trimesh_vertices,
            trimesh_faces=trimesh_faces,
            stats=self.stats if save_stats else dict()
        )
        with open(os.path.join(self.resources_dirname, inner_path, f"{terrain_name}.pickle"), "wb+") as f:
            pickle.dump(terrain_to_save, f)

    def load_terrain(self):
        with open(os.path.join(self.resources_dirname, self.cfg.load_inner_path), "rb") as f:
            loaded_terrain: SavedTerrain = pickle.load(f)
            if type(loaded_terrain) is not SavedTerrain:
                raise ValueError("Terrain pickle should be a SavedTerrain object!")
            if loaded_terrain.dims != self.get_dims():
                raise ValueError(f"Dimensions of loaded terrain ({loaded_terrain.dims}) " +
                                 f"doesn't match the config ({self.get_dims()})!")
            self.height_field_raw = loaded_terrain.heightmap
            self.tile_origins = loaded_terrain.tile_origins
            self.stats = loaded_terrain.stats
            self.vertices, self.triangles = loaded_terrain.trimesh_vertices, loaded_terrain.trimesh_faces
        
    # ============= Terrain Union/Duplication ==========================

    def from_tiles(self, tile_func, tile_args={}):
        for row, col in np.ndindex((self.cfg.num_rows, self.cfg.num_cols)):
            # Create subterrain object to hold tile heightmap to pass tile_func
            subterrain = terrain_utils.SubTerrain("terrain",
                width=self.tile_width_px,
                length=self.tile_length_px,
                vertical_scale=self.cfg.vertical_scale,
                horizontal_scale=self.cfg.horizontal_scale)
            # Populate tile heightmap and stats via passed in tile function
            tile_stats = tile_func(subterrain, row, col, **tile_args)
            # Place the tile into the main heightmap
            start_x = self.border + row * self.tile_length_px
            end_x = start_x + self.tile_length_px
            start_y = self.border + col * self.tile_width_px
            end_y = start_y + self.tile_length_px
            self.height_field_raw[start_x: end_x, start_y: end_y] = subterrain.height_field_raw
            # Set tile origin to be in the middle of the tile, at the same height as the highest point in tile
            tile_origin_x = (row + 0.5) * self.cfg.tile_length
            tile_origin_y = (col + 0.5) * self.cfg.tile_width
            tile_origin_z = np.max(subterrain.height_field_raw) * subterrain.vertical_scale
            self.tile_origins[row, col] = [tile_origin_x, tile_origin_y, tile_origin_z]
            # Record stats that have been calculated
            if self.cfg.calc_stats and tile_stats is not None:
                for key, stat in tile_stats.items():
                    # If a key doesn't exist, create numpy array with shape (num_rows, num_cols, <shape of stat>)
                    # with NaN values indicating no stat has been collected yet.
                    if key not in self.stats:
                        array_size = [self.cfg.num_rows, self.cfg.num_cols]
                        if type(stat) == np.ndarray and np.ndim(stat) > 0:
                            array_size.append(np.array(stat).shape)
                        self.stats[key] = np.full(array_size, np.nan)
                    # Populate the stats dictionary with the newly calculated stat
                    self.stats[key][row][col] = tile_stats[key]

    # ============= Tile Helper Functions ============================
                    
    def locomotion_tile_helper(self, subterrain, choice, difficulty, tile_type_proportions):
        # TODO: why does this exist if we also have the terrain multiplexing stuff at the top?
        proportions = [np.sum(tile_type_proportions[:i+1]) for i in range(len(tile_type_proportions))]
        # difficulty is between 0 and 1
        slope = difficulty * 0.4 # in radians (affects straight slope and noisy terrain)
        step_height = 0.05 + 0.18 * difficulty # 23 cm height (may want to decrease for blind locomotion, e.g., to 13 cm)
        discrete_obstacles_height = 0.05 + difficulty * 0.2
        stepping_stones_size = 1.5 * (1.05 - difficulty)
        stone_distance = 0.05 if difficulty==0 else 0.1
        gap_size = 1. * difficulty
        pit_depth = 1. * difficulty
        if choice < proportions[0]:
            if choice < proportions[0]/ 2:
                slope *= -1
            terrain_utils.pyramid_sloped_terrain(subterrain, slope=slope, platform_size=3.)
        elif choice < proportions[1]:
            terrain_utils.pyramid_sloped_terrain(subterrain, slope=slope, platform_size=3.)
            terrain_utils.random_uniform_terrain(subterrain, min_height=-0.05, max_height=0.05, step=0.005, downsampled_scale=0.2)
        elif choice < proportions[3]:
            if choice<proportions[2]:
                step_height *= -1
            terrain_utils.pyramid_stairs_terrain(subterrain, step_width=0.31, step_height=step_height, platform_size=3.)
        elif choice < proportions[4]:
            num_rectangles = 20
            rectangle_min_size = 1.
            rectangle_max_size = 2.
            terrain_utils.discrete_obstacles_terrain(subterrain, discrete_obstacles_height, rectangle_min_size, rectangle_max_size, num_rectangles, platform_size=3.)
        elif choice < proportions[5]:
            terrain_utils.stepping_stones_terrain(subterrain, stone_size=stepping_stones_size, stone_distance=stone_distance, max_height=0., platform_size=4.)
        elif choice < proportions[6]:
            self.gap_tile_helper(subterrain, gap_size=gap_size, platform_size=3.)
        elif choice < proportions[7]:
            self.pit_tile_helper(subterrain, depth=pit_depth, platform_size=4.)
        elif choice < proportions[8]:
            terrain_utils.random_uniform_terrain(subterrain, min_height=-self.cfg.terrain_noise_magnitude,
                                                 max_height=self.cfg.terrain_noise_magnitude, step=0.005,
                                                 downsampled_scale=0.2)
        elif choice < proportions[9]:
            terrain_utils.random_uniform_terrain(subterrain, min_height=-0.05, max_height=0.05,
                                                 step=self.cfg.terrain_smoothness, downsampled_scale=0.2)
            subterrain.height_field_raw[0:subterrain.length // 2, :] = 0
    
    def gap_tile_helper(self, subterrain, gap_size, platform_size):
        gap_size = int(gap_size / subterrain.horizontal_scale)
        platform_size = int(platform_size / subterrain.horizontal_scale)

        center_x = subterrain.length // 2
        center_y = subterrain.width // 2
        x1 = (subterrain.length - platform_size) // 2
        x2 = x1 + gap_size
        y1 = (subterrain.width - platform_size) // 2
        y2 = y1 + gap_size

        subterrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
        subterrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

    def pit_tile_helper(self, subterrain, gap_size, platform_size=1.):
        gap_size = int(gap_size / subterrain.horizontal_scale)
        platform_size = int(platform_size / subterrain.horizontal_scale)

        center_x = subterrain.length // 2
        center_y = subterrain.width // 2
        x1 = (subterrain.length - platform_size) // 2
        x2 = x1 + gap_size
        y1 = (subterrain.width - platform_size) // 2
        y2 = y1 + gap_size

        subterrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
        subterrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0
    
    # ============= Tile Implementations  ============================
        
    def _tile_locomotion_random(self, subterrain, row, col, tile_type_proportions):
        choice = np.random.uniform(0, 1)
        difficulty = np.random.choice([0.5, 0.75, 0.9])
        self.locomotion_tilemaker(subterrain, choice, difficulty, tile_type_proportions)
        

    def _tile_locomotion_curriculum(self, subterrain, row, col, tile_type_proportions):
        difficulty = row / self.cfg.num_rows
        choice = col / self.cfg.num_cols + 0.001
        self.locomotion_tilemaker(subterrain, choice, difficulty, tile_type_proportions)

    def _tile_valley(self, subterrain, row, col):
        width, length = subterrain.width, subterrain.length
        hf = subterrain.height_field_raw
        hor_scale, ver_scale = subterrain.horizontal_scale, subterrain.vertical_scale
        gap_size = 0
        slope_size = int(0.5 * length)
        x_left = slope_size
        x_right = x_left + gap_size
        slope = 0.1 * row
        slope *= hor_scale / ver_scale
        #noise_level = np.ceil(j * j) * (1 + slope ** 2) / 5
        noise_level = col * np.sqrt(1 + slope ** 2) / 6
        noises = np.random.normal(0, noise_level, size=(2 * x_left, width)) #- (noise_level / 2)

        # flat part at the bottom
        hf[x_left : x_right, :] = -1. * slope * slope_size
        # banks of the valley
        for x in range(x_left):
            for y in range(width):
                hf[x, y] = (-1. * slope * x) + noises[x][y]
                hf[length - x - 1, y] = (-1. * slope * x) + noises[x + x_left][y]
        if self.cfg.calc_stats:
            roughness = residual_variance(hf[:x_left, :], ver_scale, hor_scale, width*length) / 2 #terrain_roughness_index(terrain.height_field_raw[:x_left, :]) / 2 
            roughness += residual_variance(hf[x_right:, :], ver_scale, hor_scale, width*length) / 2 # terrain_roughness_index(terrain.height_field_raw[x_right:, :]) / 2
            return {"roughness": roughness}

    def _tile_semivalley(self, subterrain, row, col):
        width, length = subterrain.width, subterrain.length
        hf = subterrain.height_field_raw
        hor_scale, ver_scale = subterrain.horizontal_scale, subterrain.vertical_scale
        slope = 0.1 * row
        roughness_level = 0.2 * col
        heightmap_slope = slope * hor_scale / ver_scale

        #noise_level = np.ceil(j * j) * (1 + slope ** 2) / 5
        noise_level = roughness_level * np.sqrt(2 * np.pi) * (1 + slope ** 2) #** (3/4)
        noises = np.random.normal(0, noise_level, size=(width, length)) #- (noise_level / 2)
        for x in range(length):
            for y in range(width):
                hf[x, y] = (-1. * heightmap_slope * x) + noises[x][y]
        if self.cfg.calc_stats:
            area = (width * length) * (hor_scale ** 2) * np.sqrt(1 + slope ** 2)
            roughness = residual_variance(hf, ver_scale, hor_scale, area) #terrain_roughness_index(terrain.height_field_raw) 
            return {"slope": slope, "slope_normal": slope_normal(slope), "roughness": roughness}

    def _tile_perlin_noise(self, subterrain, row, col):
        scale = 30.0
        height_multiplier = 50 * row
        octaves = 7
        persistence = 0.1 * col
        lacunarity = 2.0
        seed = np.random.randint(0,100)
        width, length = subterrain.width, subterrain.length
        world = np.zeros((length, width))

        for x in range(10, length-10):
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
        subterrain.height_field_raw = world.astype('int32')

    def _tile_slope(self, subterrain, row, col, slope=1.0):
        slope *= subterrain.horizontal_scale / subterrain.vertical_scale
        for x in range(subterrain.width):
            subterrain.height_field_raw[x, :] = (row * subterrain.width + x) * slope

    def _tile_flat(self, subterrain, row, col):
        subterrain.height_field_raw[:] = 0

# ======== Terrain Saving Dataclass ==============
    
@dataclass
class SavedTerrain:
    @dataclass
    class Dimensions:
        num_rows: int
        num_cols: int
        tile_width_px: int
        tile_length_px: int
        borderpx: int
        horizontal_scale: float
        vertical_scale: float
    name: str
    heightmap: np.ndarray
    tile_origins: np.ndarray
    dims: Dimensions
    trimesh_vertices: Optional[np.ndarray] = None
    trimesh_faces: Optional[np.ndarray] = None
    stats: dict = field(default_factory=lambda: dict())

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
