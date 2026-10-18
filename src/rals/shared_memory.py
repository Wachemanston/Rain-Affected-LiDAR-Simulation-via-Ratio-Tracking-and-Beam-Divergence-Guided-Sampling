from multiprocessing import shared_memory
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class SharedMemoryMeta(NamedTuple):
    dshape: tuple
    dtype: np.dtype
    shm_name: str


class SharedMemoryAccessor:
    def __init__(self, meta: SharedMemoryMeta):
        self.__dshape = meta.dshape
        self.__dtype = meta.dtype
        self.__shm = shared_memory.SharedMemory(name=meta.shm_name)

    @property
    def shared_ndarray(self):
        return np.ndarray(self.__dshape, dtype=self.__dtype, buffer=self.__shm.buf)

    def close(self):
        self.__shm.close()
        self.__shm = None
        self.__dtype = None
        self.__dshape = None


class SharedMemoryCreator(object):
    def __init__(self, data: NDArray):
        self.__dshape = data.shape
        self.__dtype = data.dtype
        self.__shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        self.__shm_array = np.ndarray(self.__dshape, dtype=self.__dtype, buffer=self.__shm.buf)
        np.copyto(self.__shm_array, data)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def meta(self) -> SharedMemoryMeta:
        return SharedMemoryMeta(self.__dshape, self.__dtype, self.__shm.name)

    @property
    def shared_ndarray(self) -> NDArray:
        return self.__shm_array

    def close(self):
        self.__shm.unlink()
        self.__shm.close()
        self.__shm = None
        self.__shm_array = None
        self.__dtype = None
        self.__dshape = None


class PerceptResShm:
    __col_to_idx = {'x': 0, 'y': 1, 'z': 2, 'intensity': 3, 'type': 4}
    __percept_cols = __col_to_idx['x'], __col_to_idx['y'], __col_to_idx['z'], __col_to_idx['intensity']

    def __init__(self, scene_name: str, scan_idx: int, max_elements: int = int(1e9), stem: str = 'pulse'):
        self.__max_elements = max_elements
        self.__stem = stem
        pulse_folder = Path.cwd() / scene_name / self.__stem
        pulse_folder.mkdir(parents=True, exist_ok=True)
        self.__pulse_path = pulse_folder / f'{self.__stem}_{scan_idx:04d}.bin'
        self.__pc_shm: SharedMemoryCreator | None = None
        self.__pulses_shm: SharedMemoryCreator | None = None

    @staticmethod
    def __load_pc_to_shm(shm: SharedMemoryCreator, points: NDArray) -> None:
        pc = shm.shared_ndarray
        pc[:, PerceptResShm.__percept_cols] = points[:len(pc), :4]

    @staticmethod
    def metas_to_accessors(metas: tuple) -> (SharedMemoryAccessor, SharedMemoryAccessor):
        pc_shm_meta, pulse_shm_meta = metas
        return SharedMemoryAccessor(pc_shm_meta), SharedMemoryAccessor(pulse_shm_meta)

    @staticmethod
    def read_pulses(pulse_path: str | Path) -> NDArray:
        data = np.fromfile(pulse_path, dtype=np.float32)
        shape = data[0:2].astype(int)
        pulses = data[2:].reshape(shape)
        return pulses

    @property
    def metas(self) -> (SharedMemoryMeta, SharedMemoryMeta):
        pc_meta = self.__pc_shm.meta if self.__pc_shm else None
        pulses_meta = self.__pulses_shm.meta if self.__pulses_shm else None
        return pc_meta, pulses_meta

    @property
    def percept_copy(self) -> NDArray | None:
        if self.__pc_shm:
            return np.copy(self.__pc_shm.shared_ndarray[:, PerceptResShm.__percept_cols])
        return None

    @property
    def type_copy(self) -> NDArray | None:
        if self.__pc_shm:
            return np.copy(self.__pc_shm.shared_ndarray[:, PerceptResShm.__col_to_idx['type']])
        return None

    def __create_shm_if_row_not_to_many(self, ray_cnt: int, pulse_length: int) -> SharedMemoryCreator:
        num_of_rows = min(ray_cnt, self.__max_elements // pulse_length)

        if num_of_rows < ray_cnt:
            raise ValueError(f'[PerceptResShm] expected rows ({ray_cnt}) > maximum rows ({num_of_rows})')

        return SharedMemoryCreator(data=np.zeros((num_of_rows, pulse_length)))

    def __close_pc_shm(self) -> None:
        self.__pc_shm and self.__pc_shm.close()

    def __close_pulses_shm(self) -> None:
        self.__pulses_shm and self.__pulses_shm.close()

    def close_shms(self) -> None:
        self.__close_pulses_shm()
        self.__close_pc_shm()

    def reset_pulses_shm(self, ray_cnt: int, pulse_length: int) -> None:
        self.__close_pulses_shm()
        self.__pulses_shm = self.__create_shm_if_row_not_to_many(ray_cnt, pulse_length)

    def reset_pc_shm(self, points: NDArray) -> None:
        self.__close_pc_shm()
        self.__pc_shm = SharedMemoryCreator(data=np.zeros((len(points), len(PerceptResShm.__col_to_idx))))
        PerceptResShm.__load_pc_to_shm(self.__pc_shm, points)

    def export_pulses(self) -> None:
        if not self.__pulses_shm:
            raise ValueError('[PerceptResShm] No pulses can be exported.')

        data = np.copy(self.__pulses_shm.shared_ndarray)
        shape = data.shape
        data = np.hstack((shape, data.flatten()))
        data.astype(np.float32).tofile(self.__pulse_path)
