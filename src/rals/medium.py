from abc import ABC, abstractmethod
from multiprocessing import Pool, cpu_count
import os
from typing import TypeAlias, Union

import numpy as np
os.environ['MIEPYTHON_USE_JIT'] = '1'
import miepython as mie
from scipy.integrate import simpson

from utils import classproperty
from constants import PI, RNG


def _calculate_mie_scattering_task(diameter_nm: np.ndarray, sphere_refractive_idx: float, medium_refractive_idx: float,
                             wavelength_nm: float) -> (np.ndarray, np.ndarray):
    assert len(diameter_nm), f'[Medium] `diameter_nm` is empty'

    m, n_env, lambda0 = sphere_refractive_idx, medium_refractive_idx, wavelength_nm
    q_exti, q_scat, q_back, q_anti = mie.efficiencies(m, diameter_nm, lambda0, n_env)
    return q_exti, q_back


class Medium(ABC):
    @property
    @abstractmethod
    def expected_q_back(self) -> float:
        pass

    @property
    @abstractmethod
    def expected_q_ext(self) -> float:
        pass

    @staticmethod
    def create(meta: dict) -> 'MediumType':
        assert 'medium' in meta, '[Medium] Invalid meta: no specified medium.'
        if meta['medium'] == 'dust':
            return Dust()
        elif meta['medium'] == 'udust':
            return UniformDust()
        elif meta['medium'] == 'rain':
            assert 'rain_rate' in meta, '[Medium] Medium Rain has no `rain_rate`.'
            return Rain(rain_rate=meta['rain_rate'])
        elif meta['medium'] == 'fog':
            assert 'visibility' in meta, '[Medium] Medium Rain has no `visibility`.'
            return Fog(visibility=meta['visibility'])
        raise NotImplementedError(f'[Medium] Unrecognized medium: {meta}')


class ParticleBasedMedium(Medium):
    @classproperty
    @abstractmethod
    def refractive_idx(cls) -> float:
        pass

    @staticmethod
    def calculate_mie_scattering(diameter_nm: np.ndarray, sphere_refractive_idx: float, medium_refractive_idx: float,
                                 wavelength_nm: float) -> (np.ndarray, np.ndarray):
        N = len(diameter_nm)
        if N < 1e5:
            return _calculate_mie_scattering_task(diameter_nm, sphere_refractive_idx, medium_refractive_idx, wavelength_nm)

        n_workers = cpu_count()
        chunk = int(N / n_workers) + 1
        with Pool(n_workers) as pool:
            res = []
            for start in range(0, N, chunk):
                end = min(start + chunk, N)
                res.append(pool.apply_async(_calculate_mie_scattering_task, (diameter_nm[start:end], sphere_refractive_idx, medium_refractive_idx, wavelength_nm)))
            res = [t.get() for t in res]
            return np.hstack([rs[0] for rs in res]), np.hstack([rs[1] for rs in res])

    @classmethod
    def mie_efficiencies(cls, radii_mum: np.ndarray, wavelength_nm: float) -> (np.ndarray, np.ndarray):
        air_refractive_idx = 1.
        diameter_nm = radii_mum * 2 * 1e3
        return ParticleBasedMedium.calculate_mie_scattering(diameter_nm, cls.refractive_idx, air_refractive_idx, wavelength_nm)

    @classmethod
    @abstractmethod
    def get_radius_linear_space(cls) -> np.ndarray:
        pass

    @abstractmethod
    def size_distribution(self, radii_mum: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_integral_variable(self) -> np.ndarray:
        pass

    def calculate_expected_mie_coefficients(self, wavelength_nm: float) -> (float, float):
        r_mum = self.get_radius_linear_space()
        q_exti, q_back = self.mie_efficiencies(r_mum, wavelength_nm)

        u = self.get_integral_variable()  # u could be radius(i.e. = r) or diameter (µm)
        p_u = self.size_distribution(r_mum)

        valid_size_ratio = simpson(p_u, u)  # \int_0 p(u) du
        assert 0 <= (1. - valid_size_ratio) < 1e-6, f'Invalid size distribution: {valid_size_ratio}'

        r_m = r_mum * 1e-6
        expected_extinction_coefficient = PI * simpson(r_m ** 2 * q_exti * p_u, u)  # \int_0 pi * Q_ext(r) p(u) du
        expected_backscattering_coefficient = PI * simpson(r_m ** 2 * q_back * p_u, u)  # \int_0 pi * Q_back(r) * p(u) du
        return expected_extinction_coefficient, expected_backscattering_coefficient  # m^2


class Dust(ParticleBasedMedium):

    @classproperty
    def refractive_idx(cls) -> float:
        return 1.57

    @classmethod
    def get_radius_linear_space(cls) -> np.ndarray:
        factor = 5
        ln_r_mean, ln_std = cls.__get_mean_and_std()
        r_min, r_max = np.exp(ln_r_mean - factor * ln_std), np.exp(ln_r_mean + factor * ln_std)
        N = int((r_max - r_min) * 10)  # every 0.1 µm
        return np.linspace(r_min, r_max, N)

    @classmethod
    def __get_mean_and_std(cls):
        return np.log(11.25), np.log(1.7)

    @classmethod
    def sample_radii_mum(cls, N: int) -> np.ndarray:
        ln_r_mean, ln_std = cls.__get_mean_and_std()
        sample_radii = RNG.lognormal(mean=ln_r_mean, sigma=ln_std, size=N)
        return sample_radii

    def __init__(self, density: float = -1.):
        self.density: float = density  # TODO: remove this

    @property
    def expected_q_back(self) -> float:
        return 2.501240284417909e-09

    @property
    def expected_q_ext(self) -> float:
        return 1.4516836097076037e-09

    def size_distribution(self, radii_mum: np.ndarray) -> np.ndarray:
        assert np.all(radii_mum > 0), 'Exist invalid radius'
        ln_r_mean, ln_std = self.__get_mean_and_std()
        pr = (1 / (radii_mum * ln_std * np.sqrt(2 * np.pi))) * np.exp(-1 * (np.log(radii_mum) - ln_r_mean) ** 2 / (2 * ln_std ** 2))
        return pr

    def get_integral_variable(self) -> np.ndarray:
        return self.get_radius_linear_space()


class UniformDust(Dust):
    __radius = 50.  # mu-m

    @classproperty
    def radius_m(cls) -> float:
        return cls.__radius * 1e-6

    @classmethod
    def get_radius_linear_space(cls) -> np.ndarray:
        return np.array([cls.__radius])

    @classmethod
    def sample_radii_mum(cls, N: int, seed=None) -> np.ndarray:
        return cls.__radius * np.ones(N)

    def __init__(self, density: float = -1.):
        super().__init__(density)

    @property
    def expected_q_back(self) -> float:
        return 0.25 * UniformDust.radius_m * UniformDust.radius_m

    @property
    def expected_q_ext(self) -> float:
        return np.pi * UniformDust.radius_m * UniformDust.radius_m

    def size_distribution(self, radii_mum: np.ndarray) -> np.ndarray:
        assert np.all(radii_mum > 0), 'Exist invalid radius'
        res = np.zeros(len(radii_mum))
        res[radii_mum == self.__radius] = 1
        return res


class Rain(ParticleBasedMedium):

    @classproperty
    def refractive_idx(cls) -> float:
        return 1.33

    @staticmethod
    def __compute_occupancy(Rr: float) -> float:
        terminal_v = 4
        return Rr / (3.6 * 1e6 * terminal_v)

    def get_radius_linear_space(self):
        lam = self.get_lambda()
        left = 2e-13  #mu-m
        right = -np.log(1e-7) / lam * 1e3
        assert left < right, (left, right)
        return 0.5 * np.linspace(left, right, int((right - left)))

    def __init__(self, rain_rate: float):
        self.rain_rate: float = rain_rate

    @property
    def expected_q_back(self) -> float:
        rain_rate_to_q_back = {
            5.: 6.462740656108741e-07,
            20.: 1.1485902137628399e-06,
            25.: 1.6446417459196993e-06,
            50.: 2.1475749978155096e-06,
            60: 1.7869852093977044e-06,
            100.: 2.147574997815509e-06,
        }
        return rain_rate_to_q_back[self.rain_rate]

    @property
    def expected_q_ext(self) -> float:
        rain_rate_to_q_ext = {
            5.: 3.692863929750647e-07,
            20.: 6.604470196806439e-07,
            25.: 7.252453930722991e-07,
            50.: 9.699579402704046e-07,
            60: 1.0470548984728339e-06,
            100.: 1.2972914585171672e-06,
        }
        return rain_rate_to_q_ext[self.rain_rate]

    def get_lambda(self):
        return 4.1 * (self.rain_rate ** -0.21)

    def sample_radii_mum(self, N: int = -1) -> np.ndarray:
        N_D = int(8000 / self.get_lambda()) if N <= 0 else N
        diameters = RNG.exponential(scale=1 / self.get_lambda(), size=N_D)
        return diameters * 0.5 * 1e3

    def size_distribution(self, radii_mum: np.ndarray) -> np.ndarray:
        D = radii_mum * 2 * 1e-3
        lam = self.get_lambda()
        return lam * np.exp(-lam * D)

    def get_integral_variable(self) -> np.ndarray:
        radii_mm = 1e-3 * self.get_radius_linear_space()
        return 2 * radii_mm


class Fog(Medium):
    __const = np.log(1 / 0.05)

    def __init__(self, visibility: float = 100.):
        self.__mor: float = visibility

    @property
    def expected_q_back(self) -> float:
        # https://github.com/MartinHahner/LiDAR_fog_sim/blob/main/fog_simulation.py
        # Ctrl+F self.mor
        return 0.046 / self.__mor

    @property
    def expected_q_ext(self) -> float:
        # Koschmieder's Law
        return Fog.__const / self.__mor


MediumType: TypeAlias = Union[Dust, Fog, Rain, UniformDust]
