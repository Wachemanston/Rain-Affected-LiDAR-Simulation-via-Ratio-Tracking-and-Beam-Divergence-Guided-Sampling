from enum import Enum
import json
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray


MAX_WORKERS = 24
PI = np.pi
RNG = np.random.default_rng()

__config_dir = Path.cwd() / 'config'

with open(__config_dir / 'config.json', 'r') as f:
    scene_cfg = json.load(f)

with open(__config_dir / 'scene_mediums.json', 'r') as f:
    scene_mediums_cfg = json.load(f)


with open(__config_dir / 'scene_trajectory.json', 'r', encoding='utf-8') as f:
    __raw_pos = json.load(f)

with open(__config_dir / 'scene_offset.json', 'r', encoding='utf-8') as f:
    __offset = json.load(f)


def _get_trajectory_ndarray(meta: Dict[str, Any]) -> NDArray:
    if 'reference_path' not in meta:
        print(f'[Constant] No reference trajectory of the scene {meta=}')
        return np.zeros((1, 3))

    reference_path = meta['reference_path']
    traj = __pos_meta_to_ndarray(__raw_pos[reference_path])
    offs = __pos_meta_to_ndarray(__offset[reference_path])
    return traj + offs


def __pos_meta_to_ndarray(dict_pos: Dict[str, float | List[float]], max_seq_len: int = 100) -> NDArray:
    x, y, z = dict_pos['x'], dict_pos['y'], dict_pos['z']
    xs = x if isinstance(x, list) else [x] * max_seq_len
    ys = y if isinstance(y, list) else [y] * max_seq_len
    zs = z if isinstance(z, list) else [z] * max_seq_len
    min_len = min(len(xs), len(ys), len(zs))
    return np.column_stack((xs[:min_len], ys[:min_len], zs[:min_len]))


class DatasetParams(NamedTuple):
    fields: Tuple = ('x', 'y', 'z', 'intensity', 'channel')
    intensity_multiplier: int = 1
    types: Tuple = (np.float32, np.float32, np.float32, np.float32, np.uint8)


AugmentedDataset = DatasetParams(fields=('x', 'y', 'z', 'intensity', 'channel', 'label'), types=(np.float32, np.float32, np.float32, np.uint8, np.float32, np.uint16, np.uint8, np.uint8))
DefaultDataset = DatasetParams()


class Weather(Enum):
    Clear = 'clear'
    Fog80 = 'fog80'
    Rain100 = 'rain100'
    Rain60 = 'rain60'


__weather_to_velocity: Dict[Weather, NDArray] = {  # m/s
    Weather.Clear: np.array([10, 0, 0]),
    Weather.Fog80: np.array([6, 0, 0]),
    Weather.Rain60: np.array([10, 0, 0]),
    Weather.Rain100: np.array([10, 0, 0]),
}


class SceneMeta(NamedTuple):
    name: str
    weather: Weather

    @property
    def trajectory(self) -> NDArray:
        return _get_trajectory_ndarray(scene_cfg[self.name])

    @property
    def velocity(self) -> NDArray:
        global __weather_to_velocity
        return __weather_to_velocity[self.weather]
