from pathlib import Path
from typing import List

import numpy as np
from pypcd4 import PointCloud, Encoding

from density_estimator import PCMQuerier
import particle_attribute_querier as PAQ
from lidar import LiDARParams


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
        assert len(augment_pc.fields) == 6, augment_pc  # expected: x, y, z, intensity, channel, label
        augment_pc.save(save_dir / f'augment_{i:04d}.pcd', encoding=Encoding.BINARY)


def process_pcd(pcd_path: Path) -> PointCloud:
    pass


if __name__ == '__main__':
    # preset()
    # process_pcds()
    pass
