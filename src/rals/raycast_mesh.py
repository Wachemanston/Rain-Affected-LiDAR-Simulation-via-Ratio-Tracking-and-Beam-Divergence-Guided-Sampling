from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeAlias, Union

import numpy as np
from numpy.typing import NDArray
import trimesh

from constants import Transform
from utils import transform_nx3_array, create_trs_matrix


class PerceptGeometry(ABC):

    @abstractmethod
    def __init__(self, meta: dict):
        self._is_static = ('static' in meta) and (meta['static'] is True)

    @abstractmethod
    def _get_transformed_mesh(self, lidar_translation: NDArray) -> trimesh.Trimesh:
        pass

    def percept(self, unit_ray_directions: NDArray, lidar_translation: NDArray, ray_origins: NDArray) -> (bool, NDArray, NDArray):
        transformed_mesh = self._get_transformed_mesh(lidar_translation)
        locations, index_ray, index_tri = transformed_mesh.ray.intersects_location(
            ray_origins=ray_origins,
            ray_directions=unit_ray_directions,
            multiple_hits=False
        )
        is_hit = len(index_ray) > 0
        if not is_hit:
            return False, None, None

        hit_dist = np.full(len(unit_ray_directions), np.inf)
        hit_origins = ray_origins[index_ray]
        vectors = locations - hit_origins
        hit_dist[index_ray] = np.linalg.norm(vectors, axis=1)

        incident_rad = np.full(len(unit_ray_directions), np.inf)
        hit_ray_dirs = unit_ray_directions[index_ray]
        hit_normals = transformed_mesh.face_normals[index_tri]
        cos_theta = np.sum(np.abs((-hit_ray_dirs) * hit_normals), axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        incident_rad[index_ray] = np.arccos(cos_theta)
        return True, hit_dist, incident_rad


sx, sy, sz = 0.5, 0.5, 0.5
cuboid_local_vertices = np.array([
    [-sx, -sy, -sz],
    [sx, -sy, -sz],
    [sx, sy, -sz],
    [-sx, sy, -sz],
    [-sx, -sy, sz],
    [sx, -sy, sz],
    [sx, sy, sz],
    [-sx, sy, sz]
])

cuboid_triangle_indices = np.array([
    # (z-)
    [0, 3, 2], [0, 2, 1],
    # (z+)
    [5, 6, 7], [5, 7, 4],
    # (x-)
    [4, 7, 3], [4, 3, 0],
    # (x+)
    [1, 2, 6], [1, 6, 5],
    # (y+)
    [3, 7, 6], [3, 6, 2],
    # (y-)
    [4, 0, 1], [4, 1, 5]
])

plane_local_vertices = np.array([
    [-sx, -sy, 0],
    [sx, -sy, 0],
    [sx, sy, 0],
    [-sx, sy, 0],
])

plane_triangle_indices = np.array([
    [0, 3, 2], [0, 2, 1],
    [0, 2, 3], [0, 1, 2]
])


class Cuboid(PerceptGeometry):
    def __init__(self, meta: dict):
        super().__init__(meta)
        self.__transform = Transform(meta['center'], np.deg2rad(meta['rotate']), meta['size'])

    def _get_transformed_mesh(self, lidar_translation: NDArray) -> trimesh.Trimesh:
        _t, r, s = self.__transform
        t = _t if self._is_static else (_t - lidar_translation)
        world_vertices = transform_nx3_array(cuboid_local_vertices, Transform(t, r, s))
        transformed_triangles = world_vertices[cuboid_triangle_indices]

        vertices = transformed_triangles.reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.merge_vertices()

        return mesh


class Mesh(PerceptGeometry):

    @staticmethod
    def shape_to_mesh_path(shape: str) -> Path:
        if shape == 'vehicle':
            return Path.cwd() / 'asset' / 'RAV4_low_poly.stl'
        if shape == 'pedestrian':
            return Path.cwd() / 'asset' / 'SM_People_07.stl'
        else:
            raise KeyError(f'[PG Mesh] Unrecognized shape {shape}')

    def __init__(self, meta: dict):
        super().__init__(meta)
        self.__transform = Transform(meta['center'], np.deg2rad(meta['rotate']), meta['size'])
        self.__mesh = trimesh.load(Mesh.shape_to_mesh_path(meta['shape']), force='mesh')

    def _get_transformed_mesh(self, lidar_translation: NDArray) -> trimesh.Trimesh:
        _t, r, s = self.__transform
        t = _t - lidar_translation
        trs_matrix = create_trs_matrix(Transform(t, r, s))
        self.__mesh.apply_transform(trs_matrix)

        return self.__mesh


class Plane(PerceptGeometry):
    def __init__(self, meta: dict):
        super().__init__(meta)
        self.__transform = Transform(meta['center'], np.deg2rad(meta['rotate']), meta['size'])

    def _get_transformed_mesh(self, lidar_translation: NDArray) -> trimesh.Trimesh:
        _t, r, s = self.__transform
        t = _t if self._is_static else (_t - lidar_translation)
        world_vertices = transform_nx3_array(plane_local_vertices, Transform(t, r, s))
        transformed_triangles = world_vertices[plane_triangle_indices]

        vertices = transformed_triangles.reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.merge_vertices()

        return mesh


class Pillars(PerceptGeometry):
    __four_faces_indices = cuboid_triangle_indices[4:12]

    def __init__(self, meta: dict):
        super().__init__(meta)
        self.__size = meta['size']
        self.__centers = meta['centers']

    def _get_transformed_mesh(self, lidar_translation: NDArray) -> trimesh.Trimesh:
        transformed_triangles = []
        for center in self.__centers:
            transform = Transform(center - lidar_translation, np.zeros(3), self.__size)
            world_vertices = transform_nx3_array(cuboid_local_vertices, transform)
            transformed_triangles.append(world_vertices[Pillars.__four_faces_indices])
        res = np.vstack(transformed_triangles)

        vertices = res.reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.merge_vertices()

        return mesh


class Sidewalks(PerceptGeometry):
    __four_faces_indices = cuboid_triangle_indices[[0, 1, 2, 3, 8, 9, 10, 11]]

    def __init__(self, meta: dict):
        super().__init__(meta)
        self.__size = meta['size']
        self.__centers = meta['centers']

    def _get_transformed_mesh(self, lidar_translation: NDArray) -> trimesh.Trimesh:
        transformed_triangles = []
        for center in self.__centers:
            transform = Transform(center - lidar_translation, np.zeros(3), self.__size)
            world_vertices = transform_nx3_array(cuboid_local_vertices, transform)
            transformed_triangles.append(world_vertices[Sidewalks.__four_faces_indices])
        res = np.vstack(transformed_triangles)

        vertices = res.reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.merge_vertices()

        return mesh


class MultiPlane(PerceptGeometry):
    def __init__(self, meta: dict):
        super().__init__(meta)
        self.__centers = meta['centers']
        self.__rotate = np.deg2rad(meta['rotate'])
        self.__size = meta['size']

    def _get_transformed_mesh(self, lidar_translation: NDArray) -> trimesh.Trimesh:
        transformed_triangles = []
        for center in self.__centers:
            t = center if self._is_static else (center - lidar_translation)
            transform = Transform(t, self.__rotate, self.__size)
            world_vertices = transform_nx3_array(plane_local_vertices, transform)
            transformed_triangles.append(world_vertices[plane_triangle_indices])
        res = np.vstack(transformed_triangles)

        vertices = res.reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.merge_vertices()

        return mesh


class Cone(PerceptGeometry):
    def __init__(self, meta: dict):
        super().__init__(meta)
        self.__transform = Transform(meta['center'], np.deg2rad(meta['rotate']), meta['size'])
        self.__local_vertices = Cone.__get_local_vertices(np.zeros(3), meta['height'], meta['radius'], meta['num_face'])

    def _get_transformed_mesh(self, lidar_translation: NDArray) -> trimesh.Trimesh:
        _t, r, s = self.__transform
        t = _t if self._is_static else (_t - lidar_translation)
        apex, base_vertices = self.__local_vertices
        world_apex = transform_nx3_array(apex, Transform(t, r, s))[0]
        world_base_vertices = transform_nx3_array(base_vertices, Transform(t, r, s))
        transformed_triangles = Cone.__vertices_to_triangles(world_apex, world_base_vertices)

        vertices = transformed_triangles.reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.merge_vertices()

        return mesh

    @staticmethod
    def __vertices_to_triangles(apex: NDArray, base_vertices: NDArray) -> NDArray:
        n = len(base_vertices)
        triangles = np.array([[apex, base_vertices[i], base_vertices[(i+1) % n]] for i in range(n)])
        return triangles

    @staticmethod
    def __get_local_vertices(center: NDArray, height: float, r: float, n: int) -> (NDArray, NDArray):
        assert center.shape == (3,)
        cx, cy, cz = center

        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

        x_base = cx + r * np.cos(angles)
        y_base = cy + r * np.sin(angles)
        z_base = np.full(n, cz)

        base_vertices = np.column_stack((x_base, y_base, z_base))
        apex = np.array([[cx, cy, cz + height]])

        return apex, base_vertices


class Triangle(PerceptGeometry):
    __triangle_local_vertices = np.array([[1, 0, 0],
                                        [-np.sin(np.pi / 6), np.cos(np.pi / 6), 0],
                                        [-np.sin(np.pi / 6), -np.cos(np.pi / 6), 0]])
    __triangle_triangle_indices = np.array([[0, 1, 2], [0, 2, 1]])

    def __init__(self, meta: dict):
        super().__init__(meta)
        self.__transform = Transform(meta['center'], np.deg2rad(meta['rotate']), meta['size'])

    def _get_transformed_mesh(self, lidar_translation: NDArray) -> trimesh.Trimesh:
        _t, r, s = self.__transform
        t = _t if self._is_static else (_t - lidar_translation)
        world_vertices = transform_nx3_array(Triangle.__triangle_local_vertices, Transform(t, r, s))
        transformed_triangles = world_vertices[Triangle.__triangle_triangle_indices]

        vertices = transformed_triangles.reshape(-1, 3)
        faces = np.arange(len(vertices)).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        mesh.merge_vertices()

        return mesh


__shape_to_geometry = {
    'cuboid': Cuboid,
    'vehicle': Mesh,
    'plane': Plane,
    'pillars': Pillars,
    'sidewalks': Sidewalks,
    'planes': MultiPlane,
    'pedestrian': Mesh,
    'cone': Cone,
    'triangle': Triangle
}

PerceptGeometryType: TypeAlias = Union[Cuboid, Mesh, Plane, Pillars, Sidewalks, MultiPlane, Cone, Triangle]


def get_percept_geometry(meta: dict) -> PerceptGeometryType:
    assert 'shape' in meta
    shape = meta['shape']

    geometry = __shape_to_geometry[shape]
    return geometry(meta)