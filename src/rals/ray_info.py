import numpy as np
from numpy.typing import NDArray
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple


__info_cols = 'alphas', 'betas', 'sample_pos', 'transmittance'
__optional_cols = 'weights', 'unit_dir'
__stem = 'ray'
__scene_name = 'default'
__scan_idx = 0
__export_if = lambda *_: False


def set_export_func(export_if: Callable[[int, NDArray], bool]):
    global __export_if
    __export_if = export_if


def __create_empty_row(row_length: int) -> NDArray:
    return np.zeros(row_length)


def __check_data_length_and_dim(kwargs: Dict[str, NDArray]) -> None:
    array_lengths = np.array([len(v) for v in kwargs.values()])
    alpha_len, beta_len, transmittance_len = len(kwargs['alphas']), len(kwargs['betas']), len(kwargs['transmittance'])
    assert alpha_len == beta_len == transmittance_len, f'[RayInfo] Inconsistent length: {array_lengths}'
    array_dims = np.array([kwargs[col].ndim for col in __info_cols])
    assert np.all(array_dims == [1, 1, 2, 1]), f'[RayInfo] dim {array_dims} != [1, 1, 2, 1]'


def __transform_kwargs_value_type_to_ndarray(kwargs: Dict[str, NDArray | List | Tuple]) -> Dict[str, NDArray]:
    for key, value in kwargs.items():
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, list) or isinstance(value, tuple):
            kwargs[key] = np.array(value)
    return kwargs


def __filter_invalid_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    for key in kwargs.keys():
        if key not in __info_cols and key not in __optional_cols:
            del kwargs[key]
    return kwargs


def read(file_path: str | Path) -> Dict[str, NDArray]:
    data = np.load(str(file_path))
    __check_data_length_and_dim(data)
    data_length = len(list(data.values())[0])
    res = {col: data[col] if col in data else __create_empty_row(data_length) for col in __info_cols}

    for col in __optional_cols:
        if col in data:
            res[col] = data[col]

    return res


def get_store_folder() -> Path:
    folder = Path.cwd() / __scene_name / __stem / f'{__scan_idx:04d}'
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def export(ray_idx: int, unit_dir: NDArray, **kwargs) -> None:
    if __export_if(ray_idx, unit_dir):
        kwargs = __filter_invalid_kwargs(kwargs)
        kwargs = __transform_kwargs_value_type_to_ndarray(kwargs)
        __check_data_length_and_dim(kwargs)
        np.savez(str(get_store_folder() / f'{__stem}_{ray_idx:06d}.npz'), unit_dir=unit_dir, **kwargs)


def set_scene_name(scene_name: str) -> None:
    global __scene_name
    __scene_name = scene_name


def set_scan_idx(scan_idx: int) -> None:
    global __scan_idx
    __scan_idx = scan_idx


def set_stem(stem: str) -> None:
    global __stem
    __stem = stem
