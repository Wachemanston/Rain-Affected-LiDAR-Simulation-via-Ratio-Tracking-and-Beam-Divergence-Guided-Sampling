from numpy.typing import NDArray
import numpy as np
from pypcd4 import PointCloud
from typing import List, Tuple, Union

from constants import DatasetParams, RNG


def calculate_execution_time(func):
    import time

    def warp(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        print(f'[Timer] Function "{func.__name__}": {round(time.time() - start, 2)} sec')
        return res

    return warp


class classproperty(property):
    def __get__(self, obj, cls):
        return self.fget(cls)


def extract_existed_pc_cols(pc: PointCloud, fields: Tuple[str]) -> List[Union[NDArray, None]]:
    res = [None] * len(fields)
    for i, field in enumerate(fields):
        if field in pc.fields:
            col = pc.numpy((field,))
            res[i] = col.reshape(-1, 1)
    return res


def pad_zero_for_non_existed_pc_cols(cols: List[Union[NDArray, None]], N: int) -> List[NDArray]:
    default_col = np.full((N, 1), 0)
    for i, field in enumerate(cols):
        if field is None:
            cols[i] = default_col
    return cols


def extract_pc_cols(pc: PointCloud, fields: Tuple[str]) -> NDArray:
    N = pc.points
    pc_fields = set(pc.fields)
    assert set(fields) & pc_fields, f'[Util] {fields} not in {pc_fields}'

    cols = extract_existed_pc_cols(pc, fields)
    cols = pad_zero_for_non_existed_pc_cols(cols, N)

    return np.hstack(cols)


def reform_pc(pc: PointCloud, dataset_params: DatasetParams) -> PointCloud:
    points = extract_pc_cols(pc, dataset_params.fields)

    intensity_idx = dataset_params.fields.find('intensity')
    if intensity_idx > 0:
        points[intensity_idx] *= dataset_params.intensity_multiplier

    return PointCloud.from_points(points, fields=dataset_params.fields, types=dataset_params.types)


def uniformly_sample_points_in_sphere(r: float, n: int) -> NDArray:
    phi = 2 * np.pi * RNG.uniform(0, 1, size=n)
    costheta = 2 * RNG.uniform(0, 1, size=n) - 1
    theta = np.arccos(costheta)

    radius = r * np.cbrt(RNG.uniform(0, 1, size=n))

    return radius * radian_angles_to_vector(theta, phi)


def radian_angles_to_vector(el: np.float64 | NDArray, az: np.float64 | NDArray) -> NDArray:

    if type(el) == np.float64 and type(az) == NDArray:
        el = np.ones(az.shape) * el
    if type(az) == np.float64 and type(el) == NDArray:
        az = np.ones(el.shape) * az

    assert az.shape == el.shape, f'az.shape = {az.shape} != {el.shape} = el.shape'

    x = np.sin(el) * np.cos(az)
    y = np.sin(el) * np.sin(az)
    z = np.cos(el)

    return np.stack([x, y, z], axis=-1)