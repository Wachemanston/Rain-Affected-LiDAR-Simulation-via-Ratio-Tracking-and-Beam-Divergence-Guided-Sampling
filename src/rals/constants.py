import json
from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np


MAX_WORKERS = 24
PI = np.pi
RNG = np.random.default_rng()

__config_dir = Path.cwd() / 'config'

with open(__config_dir / 'scene_mediums.json', 'r') as f:
    scene_cfg = json.load(f)

with open(__config_dir / 'scene_mediums.json', 'r') as f:
    scene_mediums_cfg = json.load(f)


class DatasetParams(NamedTuple):
    fields: Tuple = ('x', 'y', 'z', 'intensity', 'channel')
    intensity_multiplier: int = 1
    types: Tuple = (np.float32, np.float32, np.float32, np.float32, np.uint8)


AugmentedDataset = DatasetParams(fields=('x', 'y', 'z', 'intensity', 'channel', 'label'), types=(np.float32, np.float32, np.float32, np.uint8, np.float32, np.uint16, np.uint8, np.uint8))
DefaultDataset = DatasetParams()