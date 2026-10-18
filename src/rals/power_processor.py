from enum import Enum
from typing import NamedTuple, Tuple

import numpy as np
from numpy.typing import NDArray
from scipy.constants import speed_of_light as c


DEBUG = False


class ReturnMode(Enum):
    Final = 'final'
    Strongest = 'strongest'


class PowerProcessParams(NamedTuple):
    channel: int
    focal_offset: float
    focal_slope: float
    intensity_range: Tuple[float, float]


class _PowerProcessor:
    def __init__(self):
        self.mode: ReturnMode = ReturnMode.Strongest
        self.__mode_to_selector = {
            ReturnMode.Final: self.select_final,
            ReturnMode.Strongest: self.select_strongest
        }
        self.__intervals_per_meter = 0.
        self.__tau_h = 0.
        self.__threshold = 0.

    def select_final(self, powers: NDArray) -> int:
        res = np.argwhere(powers >= self.__threshold)
        return res[-1] if len(res) else -1

    def select_strongest(self, powers: NDArray, consider_thres: bool = False) -> int:
        i_max = np.argmax(powers)
        if consider_thres and powers[i_max] < self.__threshold:
            return -1
        return i_max

    # adapt from: https://github.com/SysCV/LiDAR_snow_sim/blob/main/tools/snowfall/simulation.py
    def process(self, j: int, d_orig: float, i_orig: float, i: NDArray, pc: NDArray, params: PowerProcessParams) -> None:
        channel, focal_offset, focal_slope, intensity_range = params
        min_intensity, max_intensity = intensity_range

        max_index = self.__mode_to_selector[self.mode](i)
        i_max = i[max_index]
        d_max = (max_index / self.__intervals_per_meter) - (c * self.__tau_h / 2)

        i_max += max_intensity * focal_slope * np.abs(focal_offset - (1 - d_max / 120) ** 2)
        i_max = np.clip(i_max, min_intensity, max_intensity)
        # np.random.random() < 0.01 and print(i[(0 < i) < 1e-4])
        # np.random.random() < 0.01 and print(i_max, i_orig, d_max, d_orig)#, '->', i_orig * np.exp(-2 * np.log(20) / 80 * d_orig))

        if abs(d_max - d_orig) < 2 * (1 / self.__intervals_per_meter):  # only alter intensity

            pc[j, 4] = 1

            new_i = int(i_max)
            # np.random.random() < 0.01 and print(i_max, i_orig)

            if new_i > (i_orig + 1) and DEBUG:
                print(f'\nnew intensity ({new_i}) in channel {channel} bigger than before ({i_orig}) '
                      f'=> clipping to {i_orig}')
                new_i = np.clip(new_i, min_intensity, i_orig)
                pc[j, 4] = 0

        else:  # replace point of hard target with snowflake

            pc[j, 4] = 2

            d_scaling_factor = d_max / d_orig

            pc[j, 0] = pc[j, 0] * d_scaling_factor
            pc[j, 1] = pc[j, 1] * d_scaling_factor
            pc[j, 2] = pc[j, 2] * d_scaling_factor

            new_i = int(i_max)

        assert new_i >= 0, f'new intensity is negative ({new_i})'

        clipped_i = np.clip(new_i, min_intensity, max_intensity)

        pc[j, 3] = clipped_i

    def set_params(self, **kwargs) -> None:
        self.__tau_h = kwargs['tau_h'] if 'tau_h' in kwargs else self.__tau_h
        self.__threshold = kwargs['threshold'] if 'threshold' in kwargs else self.__threshold
        self.__intervals_per_meter = kwargs['intervals_per_meter'] if 'intervals_per_meter' in kwargs else self.__intervals_per_meter


PowerProcessor = _PowerProcessor()

