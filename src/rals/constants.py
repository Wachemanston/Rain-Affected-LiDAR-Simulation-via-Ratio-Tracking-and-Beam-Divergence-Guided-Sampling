from enum import Enum
import json
from pathlib import Path
from typing import Dict, NamedTuple, Tuple

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
    def velocity(self) -> NDArray:
        global __weather_to_velocity
        return __weather_to_velocity[self.weather]
