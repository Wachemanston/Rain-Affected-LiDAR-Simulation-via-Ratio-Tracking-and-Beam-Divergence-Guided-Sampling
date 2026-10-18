from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
from numpy.typing import NDArray
from pypcd4 import PointCloud, Encoding
from scipy.constants import speed_of_light as c

from constants import AugmentedDataset, DefaultDataset, MAX_WORKERS, PI
from density_estimator import PCMQuerier
import particle_attribute_querier as PAQ
from shared_memory import PerceptResShm
from lidar import LiDARParams
import ray_info as ray_info_instance
from utils import reform_pc


density_estimator = PCMQuerier()  # can replace PCMQuerier with NoOpKDE for tests
ray_info_instance.set_export_func(export_if=lambda ray_idx, _: ray_idx <= 100)


class ProcessParams(NamedTuple):
    beta_0: float
    intervals_per_meter: int  # => 1/intervals_per_meter m discretization (i.e. measurement resolution)
    lidar_params: LiDARParams
    scene_name: str
    sigma_max: int

    @property
    def M_extended(self) -> int:
        M = self.lidar_params.max_range * self.intervals_per_meter
        return int(np.ceil(M + c * self.lidar_params.tau_h * self.intervals_per_meter))

    @property
    def lidar_range_extended(self) -> float:
        return self.lidar_params.max_range + c * self.lidar_params.tau_h


def list_sorted_pcds(pcd_dir: Path) -> List[Path]:
    return list(sorted(pcd_dir.glob('*.pcd')))


def preset(scene_name: str) -> ProcessParams:
    lidar_params = LiDARParams.get(scene_name)
    PAQ.set_params(scene_name, lidar_params, use_expected=False)
    ray_info_instance.set_stem('ray')
    params = ProcessParams(lidar_params=lidar_params, beta_0=1 * 10 ** -6 / PI, intervals_per_meter=10, sigma_max=50)
    return params


def process_pcds(pcd_dir: Path, save_dir: Path, particle_dir: Path, params: ProcessParams) -> None:
    save_dir.mkdir(exist_ok=True, parents=True)

    for i, pcd_path in enumerate(list_sorted_pcds(pcd_dir)):
        percept_res_shm = PerceptResShm(params.scene_name, scan_idx=i, stem='pulse')
        particles = np.load(str(particle_dir / f'particle_{i:04d}.npy')).astype(np.float32)
        assert particles.shape[1] == 3 and particles.ndim == 2, particles.shape
        density_estimator.set_kernel_with_points(particles)

        augment_pc = process_pcd(pcd_path, percept_res_shm, params)
        assert augment_pc.fields == AugmentedDataset.fields, augment_pc
        augment_pc.save(save_dir / f'augment_{i:04d}.pcd', encoding=Encoding.BINARY)


def process_pcd(pcd_path: Path, percept_res_shm: PerceptResShm, params: ProcessParams) -> PointCloud:
    lidar_params = params.lidar_params
    pc = PointCloud.from_path(pcd_path)
    pc = reform_pc(pc, DefaultDataset)
    points = pc.numpy(DefaultDataset.fields)
    mask = get_interesting_points(points)
    points = points[mask]
    ray_idx = np.arange(pc.points)[mask]

    x, y, z, intensity = points[:, 0], points[:, 1], points[:, 2], points[:, 3]
    channels = points[:, 4].astype(np.uint8)
    distance = np.linalg.norm([x, y, z], axis=0)
    unit_dir = points[0:3] / distance.reshape(-1, 1)
    i_adjusted = lidar_params.calibrate_intensity(i_orig=intensity, d_orig=distance, channels=channels)
    ch_max_intensity = lidar_params.get_channel_max_intensity(channels)

    percept_res_shm.reset_pulses_shm(ray_cnt=len(points), pulse_length=params.M_extended)
    percept_res_shm.reset_pc_shm(points=pc[:, :4])

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tmp = {executor.submit(process_single_beam, rd, d, i, c, ud, _i): _i for _i, (rd, d, i, c, ud) in enumerate(zip(ray_idx, distance, i_adjusted, channels, unit_dir))}


def process_single_beam():
    pass


def get_interesting_points(points: NDArray) -> NDArray:
    """ filter points before ray-casting for a smaller test. """
    """ ex: return points[: 0] > 0 """
    return np.ones(len(points), dtype=bool)


def main():
    params = preset('')
    process_pcds('', '', '', params)


if __name__ == '__main__':
    main()
