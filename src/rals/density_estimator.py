import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Union

from pypcd4 import PointCloud
from scipy.spatial import cKDTree


class PCMQuerier:
    def __init__(self):
        self.__refer_points = None
        self.__tree = None

    def kernel_count(self) -> int:
        return len(self.__refer_points)

    def set_kernel_with_pc(self, refer_pc: PointCloud) -> None:
        self.set_kernel_with_points(refer_pc.numpy(('x', 'y', 'z')))  # (P, 3))

    def set_kernel_with_points(self, points: np.ndarray) -> None:
        self.__refer_points = points
        self.__tree = cKDTree(self.__refer_points[:, 0:3])

    def query(self, points: NDArray, h: float | NDArray = 1.) -> Union[NDArray, Tuple[NDArray, List]]:
        assert points.ndim == 2 and points.shape[-1] == 3
        q = points.copy()
        q[:, 0] = q[:, 0] % 15
        points_cnt = self.__tree.query_ball_point(q.tolist(), r=h, return_length=True)
        vol = 4/3 * np.pi * h * h * h
        return points_cnt / vol

    def query_points(self, points: NDArray, h: float | NDArray = 1.) -> List[NDArray]:
        assert points.ndim == 2 and points.shape[-1] == 3
        q = points.copy()
        q[:, 0] = q[:, 0] % 15
        indices = self.__tree.query_ball_point(q.tolist(), r=h)
        return [self.__refer_points[idxes] for idxes in indices]


class NoOpKDE:
    def __init__(self):
        pass

    def kernel_count(self) -> int:
        return 1

    def set_kernel_with_pc(self, refer_pc: PointCloud) -> None:
        pass

    def set_kernel_with_points(self, points: np.ndarray) -> None:
        pass

    def query(self, points: NDArray, h: float | NDArray = 1.) -> Union[NDArray, Tuple[NDArray, List]]:
        return np.ones(len(points))

    def query_points(self, points: NDArray, h: float | NDArray = 1.) -> List[NDArray]:
        # return np.empty((len(points)), 0))
        return [np.zeros((1, 3))] * len(points)
