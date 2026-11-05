from collections import defaultdict
import json
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
from numpy.typing import NDArray
from pypcd4 import PointCloud, Encoding

from constants import AugmentedDataset, DefaultDataset, objs_cfg, PI, SceneMeta, Transform, Weather
from lidar import LiDARParams
from raycast_mesh import get_percept_geometry
from utils import radian_angles_to_vector, reform_pc, transform_nx3_array


def clear_weather_simulation(scene_meta: SceneMeta, obj_scene_name: str, output_root: Path) -> None:
    ideal_pcd_path = Path.cwd() / 'Ouster.pcd'
    scene_name, trajectory, weather = scene_meta.name, scene_meta.trajectory, scene_meta.weather
    assert trajectory.shape[1] == 3 and trajectory.ndim == 2, trajectory.shape

    ideal_pc = PointCloud.from_path(ideal_pcd_path)
    ideal_pc = reform_pc(ideal_pc, AugmentedDataset)
    assert 'intensity' in ideal_pc.fields, ideal_pc.fields
    points = ideal_pc.numpy()

    assert obj_scene_name in objs_cfg, f'[Root] Unrecognized scene: {obj_scene_name}'
    obj_metas = objs_cfg[obj_scene_name]
    obj_labels = get_labels(obj_metas)
    export_scene_obj_label(scene_name, obj_labels, obj_metas)

    output_dir = output_root / scene_name
    output_dir.mkdir(parents=True, exist_ok=True)

    idxes = range(len(trajectory))
    for scan_idx in idxes:
        percept_points = percept_meshes(scene_meta.velocity, trajectory[scan_idx], points.copy(), obj_metas, obj_labels)
        mesh_pc = PointCloud.from_points(percept_points, AugmentedDataset.fields, AugmentedDataset.types)
        mesh_pc.save(output_dir / f'sim_{scan_idx:04d}.pcd', encoding=Encoding.BINARY)


def percept_meshes(lidar_velocity: NDArray, lidar_translate: NDArray, points_in_lidar_space: NDArray, objs: List[dict], obj_labels: List[np.uint8]) -> NDArray:
    pos, intensities = points_in_lidar_space[:, 0:3], points_in_lidar_space[:, 3]
    dists = np.linalg.norm(pos, axis=1)
    labels = np.zeros(len(dists), dtype=np.uint8)

    unit_ray_directions = pos / dists.reshape(-1, 1)
    ray_origins = rolling_shutter_effect(lidar_velocity, unit_ray_directions)
    CA_P0 = 1  # no matter what value this is, it is combined into the given obj_meta[reflectivity] & obj_meta[factor]

    for label, obj_meta in zip(obj_labels, objs):
        geometry = get_percept_geometry(obj_meta)
        is_hit, hit_dists, incident_rads = geometry.percept(unit_ray_directions, lidar_translate, ray_origins)

        if is_hit:
            factor = obj_meta['factor'] if 'factor' in obj_meta else 0
            mask = hit_dists < dists
            points_in_lidar_space[mask, 0:3] *= (hit_dists[mask] / dists[mask]).reshape(-1, 1)
            points_in_lidar_space[mask, 3] = CA_P0 * np.cos(incident_rads[mask]) * obj_meta['reflectivity'] / (hit_dists[mask]) ** factor

            dists[mask] = hit_dists[mask]
            labels[mask] = label

    points = np.hstack((points_in_lidar_space, labels.reshape(-1, 1)))
    return points


def rolling_shutter_effect(lidar_velocity, unit_ray_directions, sampling_rate=10):
    yaw, pitch, roll = -0.0032020896, 0.83862830012, 0
    transform = Transform(translate=np.zeros(3), rotate=np.array([-roll, -pitch, -yaw]), scale=np.ones(3))
    rotation_reverted_unit_dir = transform_nx3_array(unit_ray_directions, transform)
    gamma = PI - cartesian_to_spherical(rotation_reverted_unit_dir)[:, 1]
    return lidar_velocity / sampling_rate * gamma.reshape(-1, 1) / (2 * PI)


def cartesian_to_spherical(points: np.ndarray) -> np.ndarray:
    assert points.ndim == 2 and points.shape[1] == 3, points.shape
    r = np.linalg.norm(points, axis=1)
    theta = np.arctan2(points[:, 1], points[:, 0])
    phi = np.zeros(len(r))
    phi[r != 0] = np.arccos(points[:, 2] / r)
    return np.vstack((r, theta, phi)).T


def get_labels(objs: List[dict]) -> List[np.uint8]:
    assert len(objs) <= 255, 'default data type of label is `np.uint8`'
    return np.arange(1, len(objs)+1).astype(np.uint8).tolist()  # 0 is for empty


def export_scene_obj_label(scene_name: str, labels: List[np.uint8], objs: List[dict]) -> None:
    obj_id_to_label = {}
    name_counter = defaultdict(int)

    for label, obj in zip(labels, objs):
        name = obj['name'] if 'name' in obj else 'unknown'
        obj_id = f'{name}_{name_counter[name]:02d}'
        obj_id_to_label[obj_id] = label
        name_counter[name] += 1

    with open(Path.cwd() / 'config' / f'{scene_name}_obj_label.json', 'w', encoding='utf-8') as f:
        json.dump(obj_id_to_label, f, indent=4)


class IdealLiDARParams(NamedTuple):
    position: np.ndarray
    rotate: np.ndarray
    rpm: int


def create_ideal_point_cloud(save_to: str | Path, model: LiDARParams, params: IdealLiDARParams) -> None:
    position, rotate, rpm = params
    azimuth_angles, elevation_angles, max_range = model.azimuth_rad(rpm), model.elevation_rad, model.max_range
    assert position.shape == (3,)

    direction = calculate_lidar_percept_directions(model, rpm)
    direction = transform_nx3_array(direction, Transform(translate=np.zeros(3), rotate=rotate, scale=np.ones(3)))
    xyz = max_range * direction

    intensities = np.zeros(len(xyz))
    channels = np.hstack(tuple([channel] * len(azimuth_angles) for channel in range(len(elevation_angles))))
    points = np.column_stack((xyz, intensities, channels))
    print('total points', points.shape)

    pc = PointCloud.from_points(points, fields=DefaultDataset.fields, types=DefaultDataset.types)
    pc.save(save_to, encoding=Encoding.BINARY)


def calculate_lidar_percept_directions(params: LiDARParams, rpm: int = 600) -> np.ndarray:
    azimuth, elevation = params.azimuth_rad(rpm), params.elevation_rad
    direction = spherical_to_cartesian_by_mesh(1, elevation, azimuth)
    direction = direction.reshape(-1, 3)
    return direction


def spherical_to_cartesian_by_mesh(r: float, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    assert theta.ndim == 1 and phi.ndim == 1

    m_theta, m_phi = np.meshgrid(theta, phi, indexing='ij')

    return r * radian_angles_to_vector(m_theta, m_phi).reshape((len(theta), len(phi), 3))
