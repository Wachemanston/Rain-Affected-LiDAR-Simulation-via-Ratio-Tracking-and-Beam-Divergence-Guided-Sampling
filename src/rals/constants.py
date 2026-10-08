import json
from pathlib import Path

import numpy as np


MAX_WORKERS = 24
PI = np.pi
RNG = np.random.default_rng()

__config_dir = Path.cwd() / 'config'

with open(__config_dir / 'scene_mediums.json', 'r') as f:
    scene_cfg = json.load(f)

with open(__config_dir / 'scene_mediums.json', 'r') as f:
    scene_mediums_cfg = json.load(f)
