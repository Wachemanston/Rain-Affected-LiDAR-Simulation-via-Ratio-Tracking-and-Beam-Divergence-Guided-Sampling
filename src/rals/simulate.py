from pathlib import Path
from typing import List

from pypcd4 import PointCloud, Encoding

import particle_attribute_querier as PAQ
from lidar import LiDARParams


def list_sorted_pcds(pcd_dir: Path) -> List[Path]:
    return list(sorted(pcd_dir.glob('*.pcd')))


def preset(scene_name: str) -> None:
    lidar_params = LiDARParams.get(scene_name)
    PAQ.set_params(scene_name, lidar_params, use_expected=False)


def process_pcds(pcd_dir: Path, save_dir: Path) -> None:
    save_dir.mkdir(exist_ok=True, parents=True)

    for i, pcd_path in enumerate(list_sorted_pcds(pcd_dir)):
        augment_pc = process_pcd(pcd_path)
        assert len(augment_pc.fields) == 6, augment_pc  # expected: x, y, z, intensity, channel, label
        augment_pc.save(save_dir / f'augment_{i:04d}.pcd', encoding=Encoding.BINARY)


def process_pcd(pcd_path: Path) -> PointCloud:
    pass


if __name__ == '__main__':
    # preset()
    # process_pcds()
    pass
