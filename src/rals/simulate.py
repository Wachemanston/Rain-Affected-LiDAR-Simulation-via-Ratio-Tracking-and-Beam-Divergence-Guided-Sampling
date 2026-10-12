from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List

import numpy as np
from numpy.typing import NDArray
from pypcd4 import PointCloud, Encoding

from constants import AugmentedDataset, DefaultDataset, MAX_WORKERS
from density_estimator import PCMQuerier
import particle_attribute_querier as PAQ
from lidar import LiDARParams
from utils import reform_pc


density_estimator = PCMQuerier()  # can replace PCMQuerier with NoOpKDE for tests


def list_sorted_pcds(pcd_dir: Path) -> List[Path]:
    return list(sorted(pcd_dir.glob('*.pcd')))


def preset(scene_name: str) -> None:
    lidar_params = LiDARParams.get(scene_name)
    PAQ.set_params(scene_name, lidar_params, use_expected=False)


def process_pcds(pcd_dir: Path, save_dir: Path, particle_dir: Path) -> None:
    save_dir.mkdir(exist_ok=True, parents=True)

    for i, pcd_path in enumerate(list_sorted_pcds(pcd_dir)):
        particles = np.load(str(particle_dir / f'particle_{i:04d}.npy')).astype(np.float32)
        assert particles.shape[1] == 3 and particles.ndim == 2, particles.shape
        density_estimator.set_kernel_with_points(particles)

        augment_pc = process_pcd(pcd_path)
        assert augment_pc.fields == AugmentedDataset.fields, augment_pc
        augment_pc.save(save_dir / f'augment_{i:04d}.pcd', encoding=Encoding.BINARY)


def process_pcd(pcd_path: Path) -> PointCloud:
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

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        tmp = {executor.submit(process_single_beam, rd, d, i, c, ud, _i): _i for _i, (rd, d, i, c, ud) in enumerate(zip(ray_idx, distance, i_adjusted, channels, unit_dir))}


def process_single_beam():
    pass


def get_interesting_points(points: NDArray) -> NDArray:
    """ filter points before ray-casting for a smaller test. """
    """ ex: return points[: 0] > 0 """
    return np.ones(len(points), dtype=bool)


if __name__ == '__main__':
    # preset()
    # process_pcds()
    pass
