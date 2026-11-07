import numpy as np
from pathlib import Path

from create_clear_sequence import create_ideal_point_cloud, clear_weather_simulation, IdealLiDARParams
from simulate import preset, process_pcds
from constants import SceneMeta, Weather
from lidar import LiDARParams


example_scene_name = 'scene_rain100'
example_ideal_pcd_path = './example/Ouster.pcd'
example_obj_scene_name = 'artc_north'
example_seq_output_dir = './example/pcd_dir'
example_velocity = 10  # m/s
example_particle_dir = './example/particle_dir'
example_final_output_dir = './example'

print('[1] Create pure LiDAR perception')
for p in [example_particle_dir, example_seq_output_dir]:
    Path(p).mkdir(exist_ok=True, parents=True)

scene_meta = SceneMeta(example_scene_name, Weather.Clear)

yaw, pitch, roll = -0.0032020896, 0.83862830012, 0
params = IdealLiDARParams(position=np.array([0, 0, 3]), rotate=np.array([roll, pitch, yaw]), rpm=600)

create_ideal_point_cloud(save_to=example_ideal_pcd_path, model=LiDARParams.get(example_scene_name), params=params)

print('[2] Create solid-object-only LiDAR perception')
clear_weather_simulation(scene_meta,
                         ideal_pcd_path=Path(example_ideal_pcd_path),
                         obj_scene_name=example_obj_scene_name,
                         output_root=Path(example_seq_output_dir),
                         lidar_velocity=np.array([example_velocity, 0, 0]))

print('[3] Apply medium effect')
params = preset(example_scene_name)
process_pcds(Path(example_seq_output_dir), Path(example_final_output_dir), Path(example_particle_dir), params)
