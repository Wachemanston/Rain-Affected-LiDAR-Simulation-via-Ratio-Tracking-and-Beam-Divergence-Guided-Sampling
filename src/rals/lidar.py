from abc import ABC, abstractmethod
from pathlib import Path
import yaml

import numpy as np
from numpy.typing import NDArray

from utils import classproperty

__all__ = ['VLP16', 'VLP32C', 'HDL64E', 'LiDARParams']

CALIB_ROOT = Path.cwd() / 'calib'


def _load_calib(yaml_path: Path):
    with open(yaml_path, 'r') as f:
        lidar_info = yaml.safe_load(f)

    return lidar_info['lasers']


def _select_useful_info(info: dict) -> dict:
    max_intensity = info['max_intensity'] if 'max_intensity' in info else 255
    min_intensity = info['min_intensity'] if 'min_intensity' in info else 0

    return {
                'max_intensity': max_intensity,
                'min_intensity': min_intensity,
                'focal_slope': info['focal_slope'],
                'focal_distance': info['focal_distance'],
            }


class LiDARParams(ABC):
    _channel_infos = None
    _wavelength_nm = None

    @classmethod
    @abstractmethod
    def _init_channel_infos(cls) -> None:
        pass

    @classmethod
    @abstractmethod
    def _rpm_to_deg_resolution(cls) -> dict:
        pass

    @classmethod
    def azimuth_rad(cls, rpm: int = 600) -> NDArray:
        rpm_to_deg_resolution = cls._rpm_to_deg_resolution()

        if rpm not in rpm_to_deg_resolution:
            default_rpm = list(rpm_to_deg_resolution.keys())[0]
            print(f'[LiDAR Params] RPM {rpm} was not implemented; use RPM {default_rpm} instead.')
            rpm = default_rpm

        resolution = rpm_to_deg_resolution[rpm]
        return np.deg2rad(np.arange(0., 360., resolution))

    @classproperty
    def channel_infos(cls) -> [dict]:
        if not cls._channel_infos:
            cls._init_channel_infos()
        return cls._channel_infos

    @classproperty
    @abstractmethod
    def beam_divergence(cls) -> float:
        pass

    @classproperty
    @abstractmethod
    def elevation_rad(cls) -> NDArray:
        pass

    @classproperty
    @abstractmethod
    def max_range(cls) -> float:
        pass

    @classproperty
    @abstractmethod
    def num_lasers(cls) -> int:
        pass

    @classproperty
    @abstractmethod
    def tau_h(cls) -> float:
        pass

    @classproperty
    def wavelength_nm(cls) -> int:
        return cls._wavelength_nm

    @staticmethod
    def transform_elevation(elevation_rad: NDArray) -> NDArray:
        # 0 = xy-plane -> 0 = +z axis
        # [-PI/2, PI/2] -> [PI, 0]
        return 0.5 * np.pi - elevation_rad

    @staticmethod
    def get(name: str) -> 'LiDARParams':
        if name in {'VLP-32C', 'VLP_32C'}:
            return VLP32C
        elif name in {'HDL-64E', 'HDL_64E'}:
            return HDL64E
        elif name in {'VLP-16E', 'VLP_16E'}:
            return VLP16
        elif name in {'OS0-128x1024', 'OS0_128x1024'}:
            return Ouster0_128x1024()
        raise NotImplementedError(f'Unrecognized LiDAR: {name}')

    @classmethod
    def calibrate_intensity(cls, i_orig: NDArray, d_orig: NDArray, channels: NDArray) -> NDArray:
        return i_orig

    @classmethod
    def get_channel_max_intensity(cls, ch: int) -> NDArray:
        return 255


class VLP32C(LiDARParams):
    __num_lasers: int = 32
    _wavelength_nm = 903  # nm

    @classmethod
    def _init_channel_infos(cls) -> None:
        cls._channel_infos = [None] * cls.__num_lasers

        # https://github.com/ros-drivers/velodyne/blob/ros2/velodyne_pointcloud/params/VeloView-VLP-32C.yaml
        for info in _load_calib(CALIB_ROOT / 'VeloView-VLP-32C.yaml'):
            cls._channel_infos[int(info['laser_id'])] = _select_useful_info(info)

    @classmethod
    def _rpm_to_deg_resolution(cls) -> dict:
        # https://data.ouster.io/downloads/velodyne/user-manual/vlp-32c-revd-user-manual.pdf
        return {
            300: 0.1,
            600: 0.2,
            900: 0.3,
            1200: 0.4,
        }

    @classproperty
    def beam_divergence(cls) -> float:
        # https://data.ouster.io/downloads/velodyne/user-manual/vlp-32c-revd-user-manual.pdf
        return 0.00343

    @classproperty
    def elevation_rad(cls) -> NDArray:
        rad = np.loadtxt(CALIB_ROOT / 'VLP_32C.txt')
        return cls.transform_elevation(rad)

    @classproperty
    def max_range(cls) -> float:
        # https://www.mapix.com/wp-content/uploads/2018/07/63-9378_Rev-D_ULTRA-Puck_VLP-32C_Datasheet_Web.pdf
        return 200.

    @classproperty
    def num_lasers(cls) -> int:
        return cls.__num_lasers

    @classproperty
    def tau_h(cls) -> float:
        return 10.


class HDL64E(LiDARParams):
    __num_lasers = 64
    _wavelength_nm = 903  # nm

    @classmethod
    def _init_channel_infos(cls) -> None:
        cls._channel_infos = [None] * cls.__num_lasers

        # https://github.com/SysCV/LiDAR_snow_sim/blob/main/calib/20171102_64E_S3.yaml
        for info in _load_calib(CALIB_ROOT / '20171102_64E_S3.yaml'):
            cls._channel_infos[int(info['laser_id'])] = _select_useful_info(info)

    @classmethod
    def _rpm_to_deg_resolution(cls) -> dict:
        # https://www.manualslib.com/manual/1988532/Velodyne-Hdl-64e-S3.html
        return {
            300: 0.0864,
            600: 0.1728,
            900: 0.2592,
            1200: 0.3456,
        }

    @classproperty
    def beam_divergence(cls) -> float:
        # No info in the official S3 manual/datasheet; use HDL 64E S2's value instead
        # https://www.mdpi.com/2072-4292/2/6/1610
        return 0.002

    @classproperty
    def elevation_rad(cls) -> NDArray:
        rad = np.loadtxt(CALIB_ROOT / 'WaymoTopLidarInclination.txt')
        return cls.transform_elevation(rad)

    @classproperty
    def max_range(cls) -> float:
        # https://www.mapix.com/wp-content/uploads/2018/07/63-9194_Rev-J_HDL-64E_S3_Spec-Sheet-Web.pdf
        return 120.

    @classproperty
    def num_lasers(cls) -> int:
        return cls.__num_lasers

    @classproperty
    def tau_h(cls) -> float:
        return 10.

    @classmethod
    def calibrate_intensity(cls, i_orig: NDArray, d_orig: NDArray, channels: NDArray) -> NDArray:
        channel_infos = cls.channel_infos
        ch_info_min_intensity = np.array([ci.get('min_intensity', 0) for ci in channel_infos])
        ch_info_focal_distance = np.array([ci['focal_distance'] for ci in channel_infos]) * 100
        ch_info_focal_slope = np.array([['focal_slope'] for ci in channel_infos])
        ch_info_focal_offset = (1 - ch_info_focal_distance / 13100) ** 2  # from velodyne manual

        max_intensity = cls.get_channel_max_intensity(channels)

        i_adjusted = i_orig - 255 * ch_info_focal_slope * np.abs(ch_info_focal_offset - (1 - d_orig / cls.max_range) ** 2)
        i_adjusted = np.clip(i_adjusted, ch_info_min_intensity, max_intensity)
        return i_adjusted

    @classmethod
    def get_channel_max_intensity(cls, channels: NDArray) -> NDArray:
        return np.where((channels == 53) or (channels == 55) or (channels == 56) or (channels == 58), 230, 255)


class VLP16(VLP32C):
    __num_lasers: int = 16
    _wavelength_nm = 903  # nm

    @classmethod
    def _init_channel_infos(cls) -> None:
        cls._channel_infos = [None] * cls.__num_lasers

        # https://github.com/ros-drivers/velodyne/blob/ros2/velodyne_pointcloud/params/VLP16db.yaml
        for info in _load_calib(CALIB_ROOT / 'VLP16db.yaml'):
            cls._channel_infos[int(info['laser_id'])] = _select_useful_info(info)

    @classmethod
    def _rpm_to_deg_resolution(cls) -> dict:
        # https://data.ouster.io/downloads/velodyne/user-manual/vlp-16-user-manual-revf.pdf
        return {
            300: 0.1,
            600: 0.2,
            900: 0.3,
            1200: 0.4,
        }

    @classproperty
    def beam_divergence(cls) -> float:
        # https://data.ouster.io/downloads/velodyne/user-manual/vlp-16-user-manual-revf.pdf
        return 0.003

    @classproperty
    def num_lasers(cls) -> int:
        return cls.__num_lasers

    @classproperty
    def elevation_rad(cls) -> NDArray:
        return cls.transform_elevation(np.deg2rad(np.linspace(-15., 15., 16)))

    @classproperty
    def max_range(cls) -> float:
        # Although the official spec claims 200m, this value is different based on your setting. You may try 200m if needed.
        # https://github.com/ros-drivers/velodyne/blob/ros2/velodyne_pointcloud/config/VLP16-velodyne_transform_node-params.yaml
        return 130.

    @classproperty
    def tau_h(cls) -> float:
        return 10.


class Ouster0_128x1024(LiDARParams):
    __num_lasers = 128
    __horizontal_resolution = 1024
    _wavelength_nm = 865  # nm

    @classmethod
    def _init_channel_infos(cls) -> None:
        cls._channel_infos = [{
                'max_intensity': 255,
                'min_intensity': 0,
                'focal_slope': 0,
                'focal_distance': 0,
            }] * cls.__num_lasers

    @classmethod
    def _rpm_to_deg_resolution(cls) -> dict:
        # https://data.ouster.io/downloads/datasheets/datasheet-rev7-v3p1-os0.pdf
        # configurable: assume as 1024
        return {
            600: 360 / cls.__horizontal_resolution,
            1200: 360 / cls.__horizontal_resolution,
        }

    @classproperty
    def beam_divergence(cls) -> float:
        # https://data.ouster.io/downloads/datasheets/datasheet-rev7-v3p1-os0.pdf
        return np.deg2rad(0.35)

    @classproperty
    def num_lasers(cls) -> int:
        return cls.__num_lasers

    @classproperty
    def elevation_rad(cls) -> NDArray:
        # https://data.ouster.io/downloads/datasheets/datasheet-rev7-v3p1-os0.pdf
        return cls.transform_elevation(np.deg2rad(np.linspace(-45.4, 45.4, cls.__num_lasers)))

    @classproperty
    def max_range(cls) -> float:
        # https://data.ouster.io/downloads/datasheets/datasheet-rev7-v3p1-os0.pdf
        return 75.

    @classproperty
    def tau_h(cls) -> float:
        return 10.

