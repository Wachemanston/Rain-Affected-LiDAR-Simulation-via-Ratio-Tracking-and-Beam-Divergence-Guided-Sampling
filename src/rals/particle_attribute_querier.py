from abc import ABC, abstractmethod
from typing import TypeAlias, Union

import numpy as np
from numpy.typing import NDArray

from medium import Dust, Fog, Medium, MediumType, Rain, UniformDust
from lidar import LiDARParams, Ouster0_128x1024
from constants import scene_mediums_cfg


class Geometry(ABC):
    @abstractmethod
    def contains(self, pos: NDArray) -> NDArray:
        pass


class Cuboid(Geometry):
    def __init__(self, pos: NDArray, size: NDArray):
        assert pos.shape == (3,), pos.shape
        assert size.shape == (3,), size.shape
        self.__bbox_min = pos - 0.5 * size
        self.__bbox_max = pos + 0.5 * size

    def contains(self, pos: NDArray) -> NDArray:
        mask = np.all((pos >= self.__bbox_min) & (pos <= self.__bbox_max), axis=1)
        return mask


class XInterval(Geometry):
    def __init__(self, left: float, right: float):
        self.__left = left
        self.__right = right

    def contains(self, pos: NDArray):
        return (self.__left <= pos[:, 0]) & (pos[:, 0] < self.__right)


GeometryType: TypeAlias = Union[Cuboid, XInterval]


__lidar_params = Ouster0_128x1024()
__metas = []


def set_params(scene_name: str, lidar_params: LiDARParams, use_expected: bool = True) -> None:
    global __lidar_params, __sample_func, __get_func, __metas
    __lidar_params = lidar_params
    __sample_func = __sample_expected_mie_coefficients if use_expected else __sample_mie_coefficients
    __get_func = __get_expected_mie_coefficients if use_expected else __get_mie_coefficients

    metas = scene_mediums_cfg[scene_name]
    __metas = [(Medium.create(meta), __generate_geometry(meta)) for meta in metas]


def __generate_geometry(meta: dict) -> GeometryType:
    geometry = meta['shape']

    if geometry == 'cuboid':
        return Cuboid(pos=np.array(meta['center']), size=np.array(meta['size']))
    elif geometry == 'x-interval':
        return XInterval(left=meta['left'], right=meta['right'])
    else:
        raise KeyError(f'[Medium Radius Querier] Unrecognized geometry: {geometry}')


def __generate_medium(meta: dict) -> MediumType:
    medium = meta['medium']

    if medium == 'dust':
        return Dust()
    elif medium == 'udust':
        return UniformDust()
    elif medium == 'rain':
        return Rain(rain_rate=meta['rain_rate'])
    elif medium == 'fog':
        return Fog(visibility=meta['visibility'])
    else:
        raise KeyError(f'[Medium Radius Querier] Unrecognized medium: {medium}')


def __get_expected_mie_coefficients(medium: MediumType, radii_mum: NDArray = None) -> tuple:
    return medium.expected_q_ext, medium.expected_q_back


def __sample_expected_mie_coefficients(medium: MediumType, N: int) -> tuple:
    return medium.expected_q_ext, medium.expected_q_back


def __get_mie_coefficients(medium: MediumType, radii_mum: NDArray) -> tuple:
    return medium.mie_efficiencies(radii_mum, __lidar_params.wavelength_nm)


def __sample_mie_coefficients(medium: MediumType, N: int) -> tuple:
    radii_mum = medium.sample_radii_mum(N)
    return __get_mie_coefficients(medium, radii_mum)


def __iter_mask_and_medium(pos: NDArray) -> NDArray:
    for medium, geometry in __metas:
        mask = geometry.contains(pos)
        if np.sum(mask) > 0:
            yield mask, medium


def query_coefficients(pos: NDArray) -> (NDArray, NDArray):
    assert pos.ndim == 2 and pos.shape[1] == 3, pos.shape

    sigma_ext, sigma_back = np.zeros(len(pos)), np.zeros(len(pos))

    for mask, medium in __iter_mask_and_medium(pos):
        sigma_ext[mask], sigma_back[mask] = __sample_func(medium, N=int(np.sum(mask)))

    return sigma_ext, sigma_back


def query_coefficients_with_radius(pos: NDArray, radii_mum: NDArray) -> (NDArray, NDArray):
    assert pos.ndim == 2 and pos.shape[1] == 3, pos.shape
    assert len(pos) == len(radii_mum), (len(pos), len(radii_mum))

    sigma_ext, sigma_back = np.zeros(len(pos)), np.zeros(len(pos))

    for mask, medium in __iter_mask_and_medium(pos):
        sigma_ext[mask], sigma_back[mask] = __get_func(medium, radii_mum=radii_mum[mask])

    return sigma_ext, sigma_back


def query_radii(pos: NDArray) -> NDArray:
    assert pos.ndim == 2 and pos.shape[1] == 3, pos.shape

    radii = np.zeros(len(pos))

    for mask, medium in __iter_mask_and_medium(pos):
        radii[mask] = medium.sample_radii_mum(N=int(np.sum(mask)))

    return radii


__sample_func = __sample_expected_mie_coefficients
__get_func = __get_expected_mie_coefficients
