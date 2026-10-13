import csv
import json
import math
import os
import random
from collections import defaultdict
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from scipy.signal.windows import tukey
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_BASE_DIR = Path(
    os.environ.get(
        "BATHY_OUTPUT_DIR",
        str(SCRIPT_DIR / "outputs" / "main_comparison_repeated5"),
    )
)
DATA_DIR = Path(os.environ.get("BATHY_DATA_DIR", str(SCRIPT_DIR.parent / "data")))
GEBCO_VAR = os.environ.get("BATHY_GEBCO_VAR", "elevation")

# ---------------------------------------------------------------------------
# 区域定义: (lat_up, lat_down, lon_left, lon_right)
# --------------------------------------------------------------------------- #
REGIONS = [
    (41, 38, 169, 172),
    (22, 19, 176, 179),
    (18, 15, 155, 158),
]
# 数据包中的文件命名用到的区域存档名(与 data/ 目录文件名一一对应)
REGION_ARCHIVES = {
    "41_38_169_172": "Hawaiian_Emperor_Seamount_Chain",
    "22_19_176_179": "Central_Pacific_Seamounts",
    "18_15_155_158": "Northwest_Pacific_Ocean",
}

DEFAULT_REGIONS_TO_RUN = ("41_38_169_172", "22_19_176_179", "18_15_155_158")
REPEAT_SEEDS = (11, 22, 33, 44, 55)
TARGET_SIZE = None            # 运行时自动取 faa 网格形状
# 论文口径为 720 网格上的 25/30 像元块;换用其他分辨率数据时可按
# 比例缩放(如 180 网格对应约 6/8),或用环境变量显式覆盖。
OUTER_BLOCK_SIZE = int(os.environ.get("BATHY_OUTER_BLOCK", 25))
OUTER_TEST_FRACTION = 0.10
INNER_BLOCK_SIZE = int(os.environ.get("BATHY_INNER_BLOCK", 30))
INNER_VALIDATION_FRACTION = 0.20
SAVE_FULL_PREDICTION = True

# 本次重跑的物理方法(带网格搜索);两个神经网络方法沿用既有 CSV 结果。
METHODS = ("ggm", "eggm", "vparker", "sas")
NN_METHODS = ("pdnn", "sandwelldnn")
ALL_METHODS = METHODS + NN_METHODS
METHOD_LABELS = {
    "ggm": "GGM",
    "eggm": "EGGM",
    "vparker": "VParker",
    "sas": "SAS",
    "pdnn": "PDNN",
    "sandwelldnn": "SandwellDNN",
}
METHOD_ALIASES = {"V.Parker": "VParker"}

# EGGM 的地形改正核以 numba 加速;缺失时退回等价的纯 numpy 实现。
try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False


# =========================================================================== #
# 数据读取(替代内部数据服务器;全部来自 data/ 目录的公开数据文件)
# =========================================================================== #
def _resample_to(grid, target_shape):
    """最近邻重采样到目标形状(行=纬度自北向南,列=经度自西向东)。"""
    grid = np.asarray(grid, dtype=float)
    if grid.shape == tuple(target_shape):
        return grid
    rows = np.linspace(0, grid.shape[0] - 1, target_shape[0]).round().astype(int)
    cols = np.linspace(0, grid.shape[1] - 1, target_shape[1]).round().astype(int)
    return grid[np.ix_(rows, cols)]


def _load_region_array(subdir, stem, region, target_shape):
    code = "_".join(str(v) for v in region)
    name = REGION_ARCHIVES[code]
    for candidate in (DATA_DIR / subdir).glob(f"{stem}_*.npy"):
        if candidate.stem.endswith(code) or candidate.stem.endswith(name):
            return _resample_to(np.load(candidate), target_shape)
    raise FileNotFoundError(
        f"missing array {subdir}/{stem}_*_{code}.npy under {DATA_DIR}"
    )


def _load_gebco_nc(region, target_shape):
    """读取 GEBCO NetCDF(变量 elevation,北→南、西→东)并重采样。

    按文件魔数自动判别 NetCDF-3(CDF\x01,用 scipy 纯 Python 读取器,
    可正确处理含非 ASCII 字符的路径)与 NetCDF-4/HDF5(xarray/netCDF4/h5py)。
    """
    lat_up, lat_down, lon_left, lon_right = region
    pattern = (f"gebco_*_n{float(lat_up)}_s{float(lat_down)}"
               f"_w{float(lon_left)}_e{float(lon_right)}.nc")
    matches = sorted((DATA_DIR / "gebco").glob(pattern))
    if not matches:
        matches = sorted((DATA_DIR / "gebco").glob("*.nc"))
    if not matches:
        raise FileNotFoundError(f"no GEBCO NetCDF found under {DATA_DIR}/gebco")
    path = matches[0]
    with open(path, "rb") as handle:
        magic = handle.read(4)
    elevation = None
    if magic.startswith(b"CDF"):  # NetCDF-3 classic
        from scipy.io import netcdf_file

        with netcdf_file(str(path), "r", mmap=False) as dataset:
            elevation = np.asarray(
                dataset.variables[GEBCO_VAR][:].copy(), dtype=float
            )
    else:  # NetCDF-4 / HDF5
        try:
            import xarray as xr

            dataset = xr.open_dataset(path)
            elevation = np.asarray(dataset[GEBCO_VAR].values, dtype=float)
            dataset.close()
        except Exception:
            try:
                from netCDF4 import Dataset

                with Dataset(str(path)) as dataset:
                    elevation = np.asarray(
                        dataset.variables[GEBCO_VAR][:], dtype=float
                    )
            except Exception:
                import h5py

                with h5py.File(path, "r") as dataset:
                    elevation = np.asarray(dataset[GEBCO_VAR][()], dtype=float)
    if elevation.ndim != 2:
        raise ValueError(f"unexpected GEBCO shape {elevation.shape}: {path}")
    # GEBCO 官方产品纬度恒为北→南排列,直接重采样到目标网格即可
    return _resample_to(elevation, target_shape)


def determine_target_size(region):
    code = "_".join(str(v) for v in region)
    name = REGION_ARCHIVES[code]
    matches = sorted((DATA_DIR / "swot_data").glob(f"faa_*_{code}.npy"))
    if not matches:
        matches = sorted((DATA_DIR / "swot_data").glob(f"faa_{name}_*.npy"))
    if not matches:
        raise FileNotFoundError(
            f"missing faa grid for {code} under {DATA_DIR}/swot_data"
        )
    return np.load(matches[0]).shape


def load_gravity(region, kind, target_shape):
    return _load_region_array("swot_data", kind, region, target_shape)


def load_ship_raw(region, target_shape):
    return _load_region_array("ship", "ship", region, target_shape)


def load_gebco(region, target_shape):
    return _load_gebco_nc(region, target_shape)


# =========================================================================== #
# 地球空间工具(内联自内部 es 模块,接口保持一致)
# =========================================================================== #
def geocentric_radius(lat_deg):
    """WGS84 椭球地心半径(km)。"""
    a = 6378137.0
    f = 1.0 / 298.257223563
    b = a * (1 - f)
    phi = math.radians(lat_deg)
    cos_phi, sin_phi = math.cos(phi), math.sin(phi)
    numerator = math.sqrt((a * a * cos_phi) ** 2 + (b * b * sin_phi) ** 2)
    denominator = math.sqrt((a * cos_phi) ** 2 + (b * sin_phi) ** 2)
    return numerator / denominator * 1e-3


def points_distance(point1, point2):
    """Haversine 球面距离(km),point = (lat, lon)。"""
    lat1, lon1 = point1
    lat2, lon2 = point2
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = geocentric_radius(math.degrees((lat1 + lat2) / 2))
    return c * r


def ship_clean_by_gebco(ship_grid, gebco, confidence=5):
    """以 GEBCO 为参考的 5σ 残差清洗:剔除残差超过 confidence·RMS 的船测点。"""
    mask_ship = ship_grid != 0
    residual = ship_grid[mask_ship] - gebco[mask_ship]
    deviation = confidence * np.sqrt(np.mean(np.square(residual)))
    outlier = np.abs(residual) > deviation
    mask_well = mask_ship.copy()
    mask_well[mask_well] = ~outlier
    ship_grid_well = ship_grid.copy()
    ship_grid_well[~mask_well] = 0
    print(f"{outlier.sum()} outlier points have been removed!")
    return ship_grid_well


# =========================================================================== #
# 长波场模块(内联自内部 longwave 实现)
# =========================================================================== #
class CloughLongwave:
    """高斯归一化加权 + 逐级扩σ补洞的稀疏长波场。"""

    def __init__(self, sigma):
        self.sigma = sigma
        self.filtered_matrix = None

    def fit(self, origin_matrix):
        self.filtered_matrix = gaussian_filter(
            origin_matrix, sigma=self.sigma, mode="constant", truncate=10, cval=0.0
        )

    def fit_transform(self, origin_matrix):
        self.fit(origin_matrix)
        return self.filtered_matrix

    def fit_sparse(self, origin_matrix, sparse_value=0.0):
        valid_mask = origin_matrix != sparse_value
        target_size = origin_matrix.shape
        x = np.arange(0, target_size[0], 1)
        y = np.arange(0, target_size[1], 1)
        lon_grid, lat_grid = np.meshgrid(x, y)
        pts = np.column_stack((lon_grid[valid_mask], lat_grid[valid_mask]))
        vals = origin_matrix[valid_mask]
        grid_z = griddata(pts, vals, (lon_grid, lat_grid), method="linear",
                          fill_value=sparse_value)
        valid_mask2 = grid_z != 0
        weights = valid_mask2.astype(float)
        smoothed_values = gaussian_filter(grid_z, sigma=self.sigma, mode="constant",
                                          truncate=10, cval=0.0)
        smoothed_weights = gaussian_filter(weights, sigma=self.sigma, mode="constant",
                                           truncate=10, cval=0.0)
        mask_ok = smoothed_weights != 0
        self.filtered_matrix = np.zeros_like(grid_z)
        self.filtered_matrix[mask_ok] = smoothed_values[mask_ok] / smoothed_weights[mask_ok]

    def fit_sparse_safety(self, origin_matrix, sparse_value=0.0):
        """fit_sparse 后对仍未覆盖的像元逐级倍增 σ 补全。"""
        self.fit_sparse(origin_matrix, sparse_value)
        origin_sigma = self.sigma
        mask_invalidate = ~(self.filtered_matrix != 0)
        t = 1
        while np.any(mask_invalidate):
            t += 1
            origin_sigma_local = origin_sigma * (2 ** (t - 1))
            weights = (~mask_invalidate).astype(float)
            values = self.filtered_matrix
            sv = gaussian_filter(values, sigma=origin_sigma_local, mode="constant",
                                 truncate=10, cval=0.0)
            sw = gaussian_filter(weights, sigma=origin_sigma_local, mode="constant",
                                 truncate=10, cval=0.0)
            mask_ok = sw != 0
            fill = mask_ok & mask_invalidate
            self.filtered_matrix[fill] = sv[fill] / sw[fill]
            mask_invalidate = ~mask_ok
        return self.filtered_matrix

    def fit_transform_sparse(self, origin_matrix, sparse_value=0):
        self.fit_sparse(origin_matrix, sparse_value)
        return self.filtered_matrix

    def fit_transform_sparse_safety(self, origin_matrix, sparse_value=0.0):
        self.fit_sparse_safety(origin_matrix, sparse_value)
        return self.filtered_matrix

    def predict(self, target_mask):
        return self.filtered_matrix[target_mask]

    def predict_matrix(self):
        return self.filtered_matrix


class GaussianLongwave:
    def __init__(self, sigma):
        self.sigma = sigma
        self.filtered_matrix = None

    def fit(self, origin_matrix):
        self.filtered_matrix = gaussian_filter(origin_matrix, sigma=self.sigma,
                                               mode="constant", cval=0.0)

    def fit_transform(self, origin_matrix):
        self.fit(origin_matrix)
        return self.filtered_matrix


class SandwellLongwave:
    """频域高斯低通长波场(Sandwell 式),稀疏输入先高斯加权填充。"""

    def __init__(self, constant_long=30):
        self.constant_long = constant_long
        self.filtered_matrix = None

    @staticmethod
    def _frequency(origin_matrix, longrkm, longckm):
        nrow, ncol = origin_matrix.shape
        ii = np.where(np.arange(nrow) <= nrow / 2, np.arange(nrow),
                      np.arange(nrow) - nrow)
        jj = np.where(np.arange(ncol) <= ncol / 2, np.arange(ncol),
                      np.arange(ncol) - ncol)
        return np.sqrt((ii[:, None] / longrkm) ** 2
                       + (jj[None, :] / longckm) ** 2)

    def fit(self, origin_matrix, longrkm, longckm):
        frequency = self._frequency(origin_matrix, longrkm, longckm)
        lowpass_filter = np.exp(-2 * (np.pi * frequency * self.constant_long) ** 2)
        self.filtered_matrix = ifft2(lowpass_filter * fft2(origin_matrix)).real

    def fit_sparse_safety(self, origin_matrix, longrkm, longckm, sparse_value=0.0):
        valid_mask = origin_matrix != sparse_value
        weights = valid_mask.astype(float)
        values = np.where(valid_mask, origin_matrix, 0.0)
        sigma = 1
        sv = gaussian_filter(values, sigma=sigma, mode="constant", truncate=4, cval=0.0)
        sw = gaussian_filter(weights, sigma=sigma, mode="constant", truncate=4, cval=0.0)
        mask_ok = sw != 0.0
        filled = np.zeros_like(origin_matrix, dtype=float)
        filled[mask_ok] = sv[mask_ok] / sw[mask_ok]
        while np.any(filled == 0.0):
            sigma *= 2
            sv = gaussian_filter(values, sigma=sigma, mode="constant", truncate=4, cval=0.0)
            sw = gaussian_filter(weights, sigma=sigma, mode="constant", truncate=4, cval=0.0)
            current = np.zeros_like(origin_matrix, dtype=float)
            ok = sw != 0.0
            current[ok] = sv[ok] / sw[ok]
            filled[filled == 0.0] = current[filled == 0.0]
        self.fit(filled, longrkm, longckm)

    def fit_transform_sparse_safety(self, origin_matrix, longrkm, longckm,
                                    sparse_value=0.0):
        self.fit_sparse_safety(origin_matrix, longrkm, longckm, sparse_value)
        return self.filtered_matrix

    def predict(self, target_mask):
        return self.filtered_matrix[target_mask]

    def predict_matrix(self):
        return self.filtered_matrix


# =========================================================================== #
# GGM:重力地质法(Kim et al., 2011 思路;Bouguer 板状体 + 长波区域场)
# =========================================================================== #
class GGMModel:
    def __init__(self, delta_rho, sigma):
        self.delta_rho = delta_rho
        self.sigma = sigma
        self.G = 6.67e-3
        self.ga_long_model = CloughLongwave(sigma)
        self.ga_grid = None
        self.reference_depth = None

    def fit(self, ga_grid, ship_fit):
        self.ga_grid = ga_grid
        self.reference_depth = ship_fit.min()
        mask_ship = ship_fit != 0
        ga_short = np.zeros_like(ga_grid)
        ga_short[mask_ship] = 2 * np.pi * self.G * self.delta_rho * (
            ship_fit[mask_ship] - self.reference_depth)
        ga_long = np.zeros_like(ga_grid)
        ga_long[mask_ship] = ga_grid[mask_ship] - ga_short[mask_ship]
        self.ga_long_model.fit_sparse_safety(ga_long)

    def predict(self, mask_target):
        ga_long_target = self.ga_long_model.predict(mask_target)
        ga_short_target = self.ga_grid[mask_target] - ga_long_target
        return ga_short_target / 2 / np.pi / self.delta_rho / self.G + self.reference_depth

    def predict_matrix(self):
        ga_long_matrix = self.ga_long_model.predict_matrix()
        ga_short_matrix = self.ga_grid - ga_long_matrix
        return ga_short_matrix / 2 / np.pi / self.delta_rho / self.G + self.reference_depth


def _safe_rmse(pred, true):
    """有限值上的 RMSE;发散组合(出现 inf/NaN 或溢出量级)给出大而有限的值,
    使网格搜索仍能稳定选参,而不是被 inf 污染。"""
    diff = np.asarray(pred, dtype=np.float64) - np.asarray(true, dtype=np.float64)
    diff = diff[np.isfinite(diff)]
    if diff.size == 0:
        return float("inf")
    return float(np.sqrt(np.mean(np.clip(diff, -1e9, 1e9) ** 2)))


def _ggm_train_and_evaluate(params, ga_grid, ship_fit, ship_validate, mask_validate):
    i, j, dr, sg = params
    model = GGMModel(dr, sg)
    model.fit(ga_grid, ship_fit)
    pred_depth = model.predict(mask_validate)
    true_depth = ship_validate[mask_validate]
    return i, j, _safe_rmse(pred_depth, true_depth)


class GGMPlus:
    def __init__(self, delta_rho_values=(1.3, 1.6, 1.9), sigma_values=(20, 24, 36)):
        self.delta_rho_values = delta_rho_values
        self.sigma_values = sigma_values
        self.ggm_model = None

    def fit(self, ga_grid, ship_fit, ship_validate):
        mask_validate = ship_validate != 0
        params_list = [
            (i, j, dr, sg)
            for i, dr in enumerate(self.delta_rho_values)
            for j, sg in enumerate(self.sigma_values)
        ]
        worker = partial(_ggm_train_and_evaluate, ga_grid=ga_grid,
                         ship_fit=ship_fit, ship_validate=ship_validate,
                         mask_validate=mask_validate)
        with Pool(processes=None) as pool:
            results = pool.map(worker, params_list)
        rmse_matrix = np.zeros((len(self.delta_rho_values), len(self.sigma_values)))
        for i, j, rmse in results:
            rmse_matrix[i, j] = rmse
        di, si = np.unravel_index(np.argmin(rmse_matrix), rmse_matrix.shape)
        exploration = {
            "delta_rhos": self.delta_rho_values,
            "sigmas": self.sigma_values,
            "rmse_matrix": rmse_matrix,
            "best_delta_rho": self.delta_rho_values[di],
            "best_sigmas": self.sigma_values[si],
        }
        self.ggm_model = GGMModel(self.delta_rho_values[di], self.sigma_values[si])
        self.ggm_model.fit(ga_grid, ship_fit + ship_validate)
        return exploration

    def predict(self, mask_target):
        return self.ggm_model.predict(mask_target)

    def predict_matrix(self):
        return self.ggm_model.predict_matrix()


# =========================================================================== #
# EGGM:扩展重力地质法(GGM 之上叠加先验地形的“补偿质量”改正)
# =========================================================================== #
def _tc_gravity_numpy(padded_grid, mask_ship, width, step_rm, step_cm):
    rows, cols = np.nonzero(mask_ship)
    d = np.hypot(
        np.linspace(-width, width, 2 * width + 1)[:, None] * step_rm,
        np.linspace(-width, width, 2 * width + 1)[None, :] * step_cm,
    )
    output = np.empty(rows.shape[0])
    for n in range(rows.shape[0]):
        r, c = rows[n] + width, cols[n] + width
        window = padded_grid[r - width:r + width + 1, c - width:c + width + 1] ** 2
        idepth = padded_grid[r, c] ** 2
        output[n] = -np.sum(1 / np.sqrt(window + d) - 1 / np.sqrt(idepth + d))
    return output * step_rm * step_cm


if _HAS_NUMBA:

    @njit(parallel=True, fastmath=True)
    def _calculate_tc_gravity_(padded_grid, mask_ship, width, step_rm, step_cm):
        rows, cols = np.nonzero(mask_ship)
        local_x = np.square(np.linspace(-width, width, 2 * width + 1) * step_rm)
        local_y = np.square(np.linspace(-width, width, 2 * width + 1) * step_cm)
        local_distance = np.zeros((2 * width + 1, 2 * width + 1))
        for i in prange(2 * width + 1):
            for j in prange(2 * width + 1):
                local_distance[i, j] = local_x[i] + local_y[j]
        output = np.zeros(rows.shape[0])
        for index in prange(rows.shape[0]):
            irow, icol = rows[index] + width, cols[index] + width
            window = padded_grid[irow - width:irow + width + 1,
                                 icol - width:icol + width + 1] ** 2
            idepth = padded_grid[irow, icol] ** 2
            output[index] = -np.sum(
                1 / np.sqrt(window + local_distance)
                - 1 / np.sqrt(idepth + local_distance)
            )
        return output * step_rm * step_cm

else:
    _calculate_tc_gravity_ = _tc_gravity_numpy


class EGGMModel:
    def __init__(self, delta_rho2, width, delta_rho1, sigma):
        self.delta_rho1 = delta_rho1
        self.delta_rho2 = delta_rho2
        self.width = int(width)
        self.sigma = sigma
        self.G = 6.67 * 1e-3
        self.reference_depth = None
        self.ggm_model = GGMModel(delta_rho1, sigma)
        self.ga_long_model = CloughLongwave(sigma)
        self.padded_prior_grid = None
        self.ga_grid = None
        self.longrm, self.longcm = None, None

    def fit(self, ga_grid, ship_fit, longrm, longcm):
        self.ga_grid = ga_grid
        self.longrm, self.longcm = longrm, longcm
        mask_ship = ship_fit != 0
        self.ggm_model.fit(ga_grid, ship_fit)
        self.reference_depth = ship_fit[mask_ship].mean()
        prior_grid = self.ggm_model.predict_matrix()
        ga_short_sequence = (2 * np.pi * self.G * self.delta_rho1
                             * (ship_fit[mask_ship] - self.reference_depth))
        nrow, ncol = ga_grid.shape
        step_rm, step_cm = longrm / nrow, longcm / ncol
        padded_grid = np.pad(prior_grid, pad_width=self.width, mode="reflect")
        self.padded_prior_grid = padded_grid
        ga_tc = _calculate_tc_gravity_(padded_grid, mask_ship, self.width,
                                       step_rm, step_cm) * self.G * self.delta_rho2
        ga_long_grid = np.zeros_like(ga_grid)
        ga_long_grid[mask_ship] = ga_grid[mask_ship] - ga_short_sequence - ga_tc
        self.ga_long_model.fit_sparse_safety(ga_long_grid)

    def predict(self, mask_target):
        nrow, ncol = mask_target.shape
        step_rm, step_cm = self.longrm / nrow, self.longcm / ncol
        ga_tc = _calculate_tc_gravity_(self.padded_prior_grid, mask_target,
                                       self.width, step_rm, step_cm
                                       ) * self.G * self.delta_rho2
        ga_long = self.ga_long_model.predict(mask_target)
        ga_short = self.ga_grid[mask_target] - ga_tc - ga_long
        return ga_short / 2 / np.pi / self.G / self.delta_rho1 + self.reference_depth

    def predict_matrix(self):
        nrow, ncol = self.ga_grid.shape
        step_rm, step_cm = self.longrm / nrow, self.longcm / ncol
        mask_target = np.ones_like(self.ga_grid, dtype=bool)
        ga_tc_grid = _calculate_tc_gravity_(self.padded_prior_grid, mask_target,
                                            self.width, step_rm, step_cm
                                            ) * self.G * self.delta_rho2
        ga_long_grid = self.ga_long_model.predict_matrix()
        ga_short_grid = self.ga_grid - ga_tc_grid.reshape(nrow, ncol) - ga_long_grid
        return ga_short_grid / 2 / np.pi / self.G / self.delta_rho1 + self.reference_depth


def _eggm_train_and_evaluate(params, ga_grid, ship_fit, ship_validate,
                             longrm, longcm, mask_validate):
    i, j, dr2, width, dr1, sigma = params
    model = EGGMModel(delta_rho1=dr1, delta_rho2=dr2, width=width, sigma=sigma)
    model.fit(ga_grid, ship_fit, longrm, longcm)
    pred_depth = model.predict(mask_validate)
    true_depth = ship_validate[mask_validate]
    return i, j, _safe_rmse(pred_depth, true_depth)


class EGGMPlus:
    def __init__(self, delta_rho2_values=(1.3, 1.6, 1.9),
                 width_values=(5, 10, 15), delta_rho1=1.1, sigma=4):
        self.delta_rho1 = delta_rho1
        self.delta_rho2_values = delta_rho2_values
        self.width_values = width_values
        self.sigma = sigma
        self.eggm_model = None

    def fit(self, ga_grid, ship_fit, ship_validate, longrm, longcm):
        mask_validate = ship_validate != 0
        params_list = [
            (i, j, dr2, w, self.delta_rho1, self.sigma)
            for i, dr2 in enumerate(self.delta_rho2_values)
            for j, w in enumerate(self.width_values)
        ]
        worker = partial(_eggm_train_and_evaluate, ga_grid=ga_grid,
                         ship_fit=ship_fit, ship_validate=ship_validate,
                         longrm=longrm, longcm=longcm, mask_validate=mask_validate)
        with Pool(processes=None) as pool:
            results = pool.map(worker, params_list)
        rmse_matrix = np.zeros((len(self.delta_rho2_values), len(self.width_values)))
        for i, j, rmse in results:
            rmse_matrix[i, j] = rmse
        bi, bw = np.unravel_index(np.argmin(rmse_matrix), rmse_matrix.shape)
        exploration = {
            "delta_rho2s": self.delta_rho2_values,
            "widths": self.width_values,
            "rmse_matrix": rmse_matrix,
            "best_delta_rho2": self.delta_rho2_values[bi],
            "best_width": self.width_values[bw],
        }
        self.eggm_model = EGGMModel(
            delta_rho1=self.delta_rho1,
            delta_rho2=self.delta_rho2_values[bi],
            width=self.width_values[bw],
            sigma=self.sigma,
        )
        self.eggm_model.fit(ga_grid, ship_fit + ship_validate, longrm, longcm)
        return exploration

    def predict(self, mask_target):
        return self.eggm_model.predict(mask_target)

    def predict_matrix(self):
        return self.eggm_model.predict_matrix()


# =========================================================================== #
# SAS:Smith & Sandwell 式频域反演(带 Wiener 下延低通与长波参考场)
# =========================================================================== #
class SAS:
    def __init__(self, delta_rho=1.0, long_constant=30):
        self.delta_rho = delta_rho
        self.long_constant = long_constant
        self.G = 6.67
        self.frequency = None
        self.reference_depth = None
        self.sandwell_longwave_model = SandwellLongwave(long_constant)
        self.band_topography = None

    @staticmethod
    def _frequency(ga_grid, longrkm, longckm):
        nrow, ncol = ga_grid.shape
        ii = np.where(np.arange(nrow) <= nrow / 2, np.arange(nrow),
                      np.arange(nrow) - nrow)
        jj = np.where(np.arange(ncol) <= ncol / 2, np.arange(ncol),
                      np.arange(ncol) - ncol)
        return np.sqrt((ii[:, None] / longrkm) ** 2
                       + (jj[None, :] / longckm) ** 2)

    def highpass_filter(self, frequency):
        return 1 - np.exp(-2 * (np.pi * frequency * self.long_constant) ** 2)

    def lowpass_filter(self, frequency):
        return 1 / (1 + 9500 * frequency ** 4
                    * np.exp(4 * np.pi * frequency * self.reference_depth))

    def fit(self, ga_grid, ship_fit, longrkm, longckm):
        self.frequency = self._frequency(ga_grid, longrkm, longckm)
        mask_ship = ship_fit != 0
        self.reference_depth = np.abs(ship_fit[mask_ship].mean() * 1e-3)
        highpass_filter = self.highpass_filter(self.frequency)
        lowpass_filter = self.lowpass_filter(self.frequency)
        self.sandwell_longwave_model.fit_transform_sparse_safety(
            ship_fit, longrkm, longckm, sparse_value=0.0)
        gravity_fft2 = (fft2(ga_grid) * highpass_filter * lowpass_filter
                        * np.exp(2 * np.pi * self.frequency * self.reference_depth))
        self.band_topography = (ifft2(gravity_fft2 / 2 / np.pi / self.G).real
                                * 1e3 / self.delta_rho)

    def predict(self, mask_target):
        topography = self.sandwell_longwave_model.predict_matrix() + self.band_topography
        return topography[mask_target]

    def predict_matrix(self):
        return self.sandwell_longwave_model.predict_matrix() + self.band_topography


def _sas_train_and_evaluate(params, ga_grid, ship_fit, ship_validate,
                            mask_validate, longrkm, longckm):
    i, j, delta_rho, long_constant = params
    model = SAS(delta_rho, long_constant)
    model.fit(ga_grid, ship_fit, longrkm, longckm)
    pred_depth = model.predict(mask_validate)
    true_depth = ship_validate[mask_validate]
    return i, j, _safe_rmse(pred_depth, true_depth)


class SASPlus:
    def __init__(self, delta_rho_values=(0.5, 1.0, 1.5),
                 long_constant_values=(5, 10, 15),
                 wiener_const_values=None):
        # wiener_const_values 保留参数位(历史签名),当前 SAS 网格不使用
        self.delta_rho_values = delta_rho_values
        self.long_constant_values = long_constant_values
        self.wiener_const_values = wiener_const_values
        self.sas_model = None

    def fit(self, ga_grid, ship_fit, ship_validate, longrkm, longckm):
        mask_validate = ship_validate != 0
        parameters_list = [
            (i, j, dr, lc)
            for i, dr in enumerate(self.delta_rho_values)
            for j, lc in enumerate(self.long_constant_values)
        ]
        worker = partial(_sas_train_and_evaluate, ga_grid=ga_grid,
                         ship_fit=ship_fit, ship_validate=ship_validate,
                         mask_validate=mask_validate, longrkm=longrkm,
                         longckm=longckm)
        with Pool(processes=None) as pool:
            results = pool.map(worker, parameters_list)
        rmse_matrix = np.zeros((len(self.delta_rho_values),
                                len(self.long_constant_values)))
        for i, j, rmse in results:
            rmse_matrix[i, j] = rmse
        bi, bj = np.unravel_index(np.argmin(rmse_matrix), rmse_matrix.shape)
        exploration = {
            "delta_rhos": self.delta_rho_values,
            "long_constants": self.long_constant_values,
            "rmse_matrix": rmse_matrix,
            "best_delta_rho": self.delta_rho_values[bi],
            "best_long_constant": self.long_constant_values[bj],
        }
        self.sas_model = SAS(self.delta_rho_values[bi],
                             self.long_constant_values[bj])
        self.sas_model.fit(ga_grid, ship_fit + ship_validate, longrkm, longckm)
        return exploration

    def predict(self, mask_target):
        return self.sas_model.predict(mask_target)

    def predict_matrix(self):
        return self.sas_model.predict_matrix()


# =========================================================================== #
# VParker:迭代 Parker 谱反演(Tukey 截断 + 对数低通 + 高斯长波参考)
# =========================================================================== #
class VParker:
    def __init__(self, delta_rho, wh, long_width, mu, alpha, ga_width, t=5):
        self.mu = mu
        self.delta_rho = delta_rho
        self.wh = wh
        self.alpha = alpha
        self.ga_width = ga_width
        self.long_width = long_width
        self.t = t
        self.G = 6.67
        self.reference_depth = None
        self.long_topography = None
        self.short_topography = None

    @staticmethod
    def apply_truncation_window(matrix, edge=0.02):
        tky = np.array([
            row_tky * col_tky
            for row_tky in tukey(matrix.shape[0], edge)
            for col_tky in tukey(matrix.shape[1], edge)
        ]).reshape(matrix.shape)
        return tky * matrix

    @staticmethod
    def _omega(ga_grid, longrkm, longckm):
        nrow, ncol = ga_grid.shape
        ii = np.where(np.arange(nrow) <= nrow / 2, np.arange(nrow),
                      np.arange(nrow) - nrow)
        jj = np.where(np.arange(ncol) <= ncol / 2, np.arange(ncol),
                      np.arange(ncol) - ncol)
        return 2 * np.pi * np.sqrt((ii[:, None] / longrkm) ** 2
                                   + (jj[None, :] / longckm) ** 2)

    def _build_lowpass_filter(self, frequency):
        ratio = np.where(frequency > self.wh, frequency / self.wh, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.where(frequency > self.wh, np.log(ratio), 0.0)
        return np.where(
            frequency > self.wh,
            ratio ** (1 - self.alpha)
            - (1 - self.alpha) * log_ratio * ratio ** (1 - self.alpha),
            1.0,
        )

    def fit(self, ga_grid, ship_fit, longrkm, longckm, ga_grid_short=None):
        if ga_grid_short is None:
            ga_grid_short = ga_grid - GaussianLongwave(
                sigma=self.ga_width).fit_transform(ga_grid)
        frequency = self._omega(ga_grid_short, longrkm, longckm)
        lowpass_filter = self._build_lowpass_filter(frequency)
        mask_ship = ship_fit != 0
        self.reference_depth = np.abs(ship_fit[mask_ship]).mean() * 1e-3
        tuk_ga_grid_short = self.apply_truncation_window(ga_grid_short)
        tuk_fft = fft2(tuk_ga_grid_short)
        transformed_ga_fft = (tuk_fft * np.exp(self.reference_depth * frequency)
                              / 2 / np.pi / self.G / self.delta_rho)
        transformed_ga_fft[0, 0] = 0j
        topo_initial = ifft2(transformed_ga_fft * lowpass_filter).real
        # 个别参数组合的迭代级数会发散(幂次溢出);数值上交给
        # _safe_rmse 处理,这里只抑制告告警刷屏。
        with np.errstate(over="ignore", invalid="ignore"):
            for i in range(2, self.t + 1):
                topo_i_fft = 0 + 0j
                for j in range(2, i + 1):
                    coef = (frequency - self.mu) ** (j - 1) / math.factorial(j)
                    topo_i_fft = topo_i_fft + coef * fft2(topo_initial ** j)
                topo_i_fft_first = transformed_ga_fft - topo_i_fft
                topo_i_fft_first[0, 0] = 0j
                topo_initial = ifft2(topo_i_fft_first * lowpass_filter).real
        self.short_topography = topo_initial - CloughLongwave(
            self.long_width).fit_transform_sparse(topo_initial)
        self.long_topography = CloughLongwave(
            self.long_width).fit_transform_sparse_safety(ship_fit)

    def predict(self, mask_target):
        sequence = (self.long_topography[mask_target]
                    + self.short_topography[mask_target] * 1e3)
        return sequence

    def predict_matrix(self):
        return self.long_topography + self.short_topography * 1e3


def _vparker_train_and_evaluate(params, ga_grid, ship_fit, ship_validate,
                                mask_validate, longrkm, longckm, ga_grid_short):
    i, j, k, dr, wh, lw, mu, ap, gw, tt = params
    model = VParker(dr, wh, lw, mu, ap, gw, tt)
    model.fit(ga_grid, ship_fit, longrkm, longckm, ga_grid_short)
    pred_depth = model.predict(mask_validate)
    true_depth = ship_validate[mask_validate]
    rmse = _safe_rmse(pred_depth, true_depth)
    print(f"delta_rho={dr}; wh={wh}; long_width={lw}; rmse={rmse}")
    return i, j, k, rmse


class VParkerPlus:
    def __init__(self, delta_rho_values=(1, 1.5, 2),
                 noise_const_values=(0.15, 0.2, 0.25),
                 long_width_values=(2, 3, 4), mu=0.01, alpha=32,
                 ga_width=120, t=5):
        self.delta_rho_values = delta_rho_values
        self.noise_const_values = noise_const_values
        self.long_width_values = long_width_values
        self.mu = mu
        self.alpha = alpha
        self.ga_width = ga_width
        self.t = t
        self.vparker_model = None

    def fit(self, ga_grid, ship_fit, ship_validate, longrkm, longckm):
        mask_validate = ship_validate != 0
        parameter_list = [
            (i, j, k, dr, wh, lw, self.mu, self.alpha, self.ga_width, self.t)
            for i, dr in enumerate(self.delta_rho_values)
            for j, wh in enumerate(self.noise_const_values)
            for k, lw in enumerate(self.long_width_values)
        ]
        ga_grid_short = ga_grid - GaussianLongwave(
            sigma=self.ga_width).fit_transform(ga_grid)
        worker = partial(_vparker_train_and_evaluate, ga_grid=ga_grid,
                         ship_fit=ship_fit, ship_validate=ship_validate,
                         mask_validate=mask_validate, longrkm=longrkm,
                         longckm=longckm, ga_grid_short=ga_grid_short)
        with Pool(processes=None) as pool:
            results = pool.map(worker, parameter_list)
        rmse_matrix = np.zeros((len(self.delta_rho_values),
                                len(self.noise_const_values),
                                len(self.long_width_values)))
        for i, j, k, rmse in results:
            rmse_matrix[i, j, k] = rmse
        bi, bj, bk = np.unravel_index(np.argmin(rmse_matrix), rmse_matrix.shape)
        exploration = {
            "delta_rhos": self.delta_rho_values,
            "whs": self.noise_const_values,
            "long_widths": self.long_width_values,
            "rmse_matrix": rmse_matrix,
            "best_delta_rho": self.delta_rho_values[bi],
            "best_noise_const": self.noise_const_values[bj],
            "best_long_width": self.long_width_values[bk],
        }
        self.vparker_model = VParker(
            mu=self.mu, delta_rho=self.delta_rho_values[bi],
            wh=self.noise_const_values[bj], alpha=self.alpha,
            ga_width=self.ga_width, long_width=self.long_width_values[bk],
            t=self.t,
        )
        self.vparker_model.fit(ga_grid, ship_fit + ship_validate, longrkm,
                               longckm, ga_grid_short)
        return exploration

    def predict(self, mask_target):
        return self.vparker_model.predict(mask_target)

    def predict_matrix(self):
        return self.vparker_model.predict_matrix()


# =========================================================================== #
# 实验流程(划分、指标、运行、汇总)
# =========================================================================== #
def canonical_method(name):
    return METHOD_ALIASES.get(name, name)


def region_id(region):
    return "_".join(str(value) for value in region)


def regions_to_run():
    raw = os.environ.get("BATHY_REGIONS_TO_RUN", "").strip()
    if not raw:
        wanted = list(DEFAULT_REGIONS_TO_RUN)
    elif raw.lower() == "all":
        return list(REGIONS)
    else:
        wanted = [item.strip() for item in raw.split(",") if item.strip()]
    known = {region_id(region): region for region in REGIONS}
    unknown = [code for code in wanted if code not in known]
    if unknown:
        raise ValueError(
            f"BATHY_REGIONS_TO_RUN contains unknown region codes {unknown}; "
            f"available region codes: {sorted(known)}"
        )
    wanted_set = set(wanted)
    return [region for region in REGIONS if region_id(region) in wanted_set]


NUMERIC_ROW_FIELDS = {
    "repeat": int, "seed": int, "n_samples": int,
    "train_samples": int, "test_samples": int,
    "fit_samples": int, "valid_samples": int,
    "RMSE": float, "MAE": float, "Bias": float, "R2": float,
    "MAPE": float, "Max_Error": float, "Min_Error": float,
}


def coerce_row(row):
    output = dict(row)
    for key, caster in NUMERIC_ROW_FIELDS.items():
        value = output.get(key)
        if value in (None, ""):
            continue
        try:
            output[key] = caster(float(value))
        except (TypeError, ValueError):
            pass
    return output


def load_previous_rows(path, drop_regions, drop_methods=None):
    """读取既有结果 CSV,丢掉指定区域的行(drop_methods 给定时再按方法过滤)。"""
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = [coerce_row(row) for row in csv.DictReader(handle)]
    for row in rows:
        if row.get("method") is not None:
            row["method"] = canonical_method(row["method"])
    valid_ids = {region_id(region) for region in REGIONS}
    stale = sorted({row["region"] for row in rows if row.get("region") not in valid_ids})
    drop_method_labels = None if drop_methods is None else set(drop_methods)
    kept = []
    for row in rows:
        region = row.get("region")
        if region not in valid_ids:
            continue
        if region in drop_regions:
            if drop_method_labels is None or row.get("method") in drop_method_labels:
                continue
        kept.append(row)
    return kept, stale


def record_sort_key(row):
    try:
        repeat = int(row.get("repeat", 0))
    except (TypeError, ValueError):
        repeat = 0
    return (str(row.get("region", "")), repeat, str(row.get("method", "")))


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def metrics(prediction, truth):
    """在 truth 非零且有限的位置计算测试指标(MAPE 为标准逐点定义)。"""
    prediction = np.asarray(prediction)
    truth = np.asarray(truth)
    if prediction.shape != truth.shape:
        raise ValueError(
            f"prediction/truth shape mismatch: {prediction.shape} vs {truth.shape}"
        )
    mask = (truth != 0) & np.isfinite(truth) & np.isfinite(prediction)
    p, t = prediction[mask].astype(float), truth[mask].astype(float)
    if p.size == 0:
        return {field: np.nan for field in
                ("RMSE", "MAE", "Bias", "R2", "MAPE", "Max_Error", "Min_Error")} | {
            "n_samples": 0}
    residual = p - t
    # 病态参数组合可能产生天文数字残差;裁剪后再平方,防止 float64 溢出
    # 把整行指标污染成 inf(正常量级 <1e4 m,不受影响)。
    residual = np.clip(residual, -1e12, 1e12)
    nonzero_t = np.abs(t) > 1e-12
    r2 = float(r2_score(t, p)) if np.unique(t).size >= 2 else np.nan
    return {
        "RMSE": float(np.sqrt(np.mean(residual ** 2))),
        "MAE": float(np.mean(np.abs(residual))),
        "Bias": float(np.mean(residual)),
        "R2": r2,
        "MAPE": float(np.mean(np.abs(residual[nonzero_t] / t[nonzero_t])) * 100)
        if np.any(nonzero_t) else np.nan,
        "Max_Error": float(np.max(residual)),
        "Min_Error": float(np.min(residual)),
        "n_samples": int(p.size),
    }


def prepare_region(region):
    """读取一个区域的重力、船测与 GEBCO,执行 5σ 清洗,只执行一次。"""
    target_shape = determine_target_size(region)
    lat_up, lat_down, lon_left, lon_right = region
    ga = load_gravity(region, "faa", target_shape)
    gebco = load_gebco(region, target_shape)
    raw_ship = load_ship_raw(region, target_shape)
    ship = ship_clean_by_gebco(raw_ship, gebco)
    longrkm = points_distance(
        (lat_up, (lon_left + lon_right) / 2), (lat_down, (lon_left + lon_right) / 2),
    )
    longckm = points_distance(
        ((lat_up + lat_down) / 2, lon_left), ((lat_up + lat_down) / 2, lon_right),
    )
    return {
        "region": region,
        "ga": ga,
        "ship": ship,
        "longrkm": longrkm,
        "longckm": longckm,
        "target_shape": target_shape,
    }


def split_ship_by_blocks_cover_all(ship_grid, block_size, test_fraction, random_state):
    """按空间块划分,并覆盖网格边缘的不足一个 block 的区域。"""
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    nrows, ncols = ship_grid.shape
    blocks = [
        (row_start, min(row_start + block_size, nrows),
         col_start, min(col_start + block_size, ncols))
        for row_start in range(0, nrows, block_size)
        for col_start in range(0, ncols, block_size)
    ]
    mask_ship = ship_grid != 0
    eligible_indices = np.asarray(
        [index for index, (r0, r1, c0, c1) in enumerate(blocks)
         if np.any(mask_ship[r0:r1, c0:c1])],
        dtype=int,
    )
    if eligible_indices.size < 2:
        raise ValueError(
            f"Only {eligible_indices.size} non-empty spatial blocks are available; "
            "at least two are required for a train/test split."
        )
    _, test_indices = train_test_split(
        eligible_indices, test_size=test_fraction, random_state=random_state,
    )
    test_index_set = set(int(index) for index in test_indices)
    mask_test_blocks = np.zeros((nrows, ncols), dtype=bool)
    for index, (r0, r1, c0, c1) in enumerate(blocks):
        if index in test_index_set:
            mask_test_blocks[r0:r1, c0:c1] = True
    mask_test = mask_ship & mask_test_blocks
    mask_train = mask_ship & ~mask_test_blocks
    ship_train = np.where(mask_train, ship_grid, 0.0)
    ship_test = np.where(mask_test, ship_grid, 0.0)
    return ship_train, ship_test


def make_shared_split(data, seed):
    """生成一组可审计的外层 train/test 和内层 fit/validation 分块。"""
    ship = data["ship"]
    train, test = split_ship_by_blocks_cover_all(
        ship, block_size=OUTER_BLOCK_SIZE,
        test_fraction=OUTER_TEST_FRACTION, random_state=seed,
    )
    fit, valid = split_ship_by_blocks_cover_all(
        train, block_size=INNER_BLOCK_SIZE,
        test_fraction=INNER_VALIDATION_FRACTION, random_state=seed + 10000,
    )
    ship_mask = ship != 0
    train_mask, test_mask = train != 0, test != 0
    fit_mask, valid_mask = fit != 0, valid != 0
    if np.any(train_mask & test_mask):
        raise AssertionError("Outer train and test blocks overlap")
    if np.any(fit_mask & valid_mask):
        raise AssertionError("Inner fit and validation blocks overlap")
    if not np.array_equal(train_mask | test_mask, ship_mask):
        raise AssertionError("Outer train/test split does not cover the ship mask")
    if not np.array_equal(fit_mask | valid_mask, train_mask):
        raise AssertionError("Inner fit/validation split does not cover outer train")
    return {"train": train, "test": test, "fit": fit, "valid": valid}


def save_split(data, split, repeat, seed):
    out = OUTPUT_BASE_DIR / "shared_splits" / f"repeat_{repeat:02d}" / region_id(
        data["region"]
    )
    out.mkdir(parents=True, exist_ok=True)
    if SAVE_FULL_PREDICTION:
        for name in ("train", "test", "fit", "valid"):
            np.save(out / f"{name}.npy", split[name])
    return {
        "repeat": repeat,
        "seed": seed,
        "region": region_id(data["region"]),
        "train_samples": int(np.count_nonzero(split["train"])),
        "test_samples": int(np.count_nonzero(split["test"])),
        "fit_samples": int(np.count_nonzero(split["fit"])),
        "valid_samples": int(np.count_nonzero(split["valid"])),
    }


def save_parameter_grid(output_base, region, run_tag, method, experiment, axes):
    if run_tag is None or "rmse_matrix" not in experiment:
        return None
    rmse_matrix = np.asarray(experiment["rmse_matrix"], dtype=float)
    expected_shape = tuple(len(values) for _, values in axes)
    if rmse_matrix.shape != expected_shape:
        raise ValueError(
            f"{method}: RMSE grid shape {rmse_matrix.shape} does not match "
            f"parameter axes {expected_shape}"
        )
    output_dir = (
        Path(output_base) / "parameter_grids" / str(run_tag) / region_id(region)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"rmse_matrix": rmse_matrix}
    for index, (name, values) in enumerate(axes):
        payload[f"axis_{index}_name"] = np.asarray(str(name))
        payload[f"axis_{index}_values"] = np.asarray(values, dtype=float)
    output_path = output_dir / f"{method}.npz"
    np.savez_compressed(output_path, **payload)
    return str(output_path)


def add_parameter_grid_record(params, output_base, region, run_tag, method,
                              experiment, axes):
    path = save_parameter_grid(output_base, region, run_tag, method, experiment, axes)
    if path is not None:
        params["parameter_grid_file"] = path
    return params


def run_method(method, data, split, seed, run_tag=None):
    """在同一组分块和同一测试集上运行一个方法。"""
    ga = data["ga"]
    train, test = split["train"], split["test"]
    fit, valid = split["fit"], split["valid"]
    longrkm, longckm = data["longrkm"], data["longckm"]
    longrm, longcm = longrkm * 1e3, longckm * 1e3
    params = {}
    region = data["region"]
    set_global_seed(seed)
    if method == "ggm":
        # 论文口径搜索区间(与 720 网格标定一致;粗网格下按网格边长裁剪上限)
        delta_rho = np.linspace(0.7, 2.5, 19)
        sigma = np.arange(2, 13, 1)
        sigma = sigma[sigma <= max(ga.shape[0], ga.shape[1]) // 6]
        if sigma.size == 0:
            sigma = np.array([2])
        model = GGMPlus(delta_rho, sigma)
        exp = model.fit(ga, fit, valid)
        prediction = model.predict_matrix()
        params = {
            "best_delta_rho": float(exp["best_delta_rho"]),
            "best_sigmas": float(exp["best_sigmas"]),
            "best_validation_rmse": float(np.min(exp["rmse_matrix"])),
        }
        params = add_parameter_grid_record(
            params, OUTPUT_BASE_DIR, region, run_tag, method, exp,
            [("delta_rho", delta_rho), ("sigma", sigma)],
        )
    elif method == "eggm":
        ggm_delta_rho = np.linspace(0.7, 2.5, 19)
        ggm_sigma = np.arange(2, 13, 1)
        ggm_sigma = ggm_sigma[
            ggm_sigma <= max(ga.shape[0], ga.shape[1]) // 6
        ]
        if ggm_sigma.size == 0:
            ggm_sigma = np.array([2])
        ggm = GGMPlus(ggm_delta_rho, ggm_sigma)
        g_exp = ggm.fit(ga, fit, valid)
        delta_rho2 = np.linspace(1, 20, 12)
        # 改正窗口必须 ≥1 像元且为整数(int 截断会得到 0 = 无改正,
        # 粗网格上还会诱发数值病态),按网格边长裁剪上限。
        max_width = max(1, min(10, max(ga.shape[0], ga.shape[1]) // 24))
        width = np.unique(
            np.clip(np.linspace(1, max_width, 10).round().astype(int), 1, None)
        )
        model = EGGMPlus(delta_rho2, width, g_exp["best_delta_rho"],
                         g_exp["best_sigmas"])
        exp = model.fit(ga, fit, valid, longrm, longcm)
        prediction = model.predict_matrix()
        params = {
            "best_delta_rho2": float(exp["best_delta_rho2"]),
            "best_width": float(exp["best_width"]),
            "best_validation_rmse": float(np.min(exp["rmse_matrix"])),
        }
        params = add_parameter_grid_record(
            params, OUTPUT_BASE_DIR, region, run_tag, method, exp,
            [("delta_rho2", delta_rho2), ("width", width)],
        )
        base_grid_file = save_parameter_grid(
            OUTPUT_BASE_DIR, region, run_tag, "eggm_base_ggm", g_exp,
            [("delta_rho", ggm_delta_rho), ("sigma", ggm_sigma)],
        )
        if base_grid_file is not None:
            params["base_ggm_parameter_grid_file"] = base_grid_file
    elif method == "sas":
        # 论文标定网格:Δρ 1–10(10 点)× C_long 0.5–14(步长 0.5,28 点);
        # 下延拓低通的噪信比固定 9500(SAS.lowpass_filter 内置)。
        delta_rho = np.linspace(1, 10, 10)
        long_constant = np.linspace(0.5, 14, 28)
        model = SASPlus(delta_rho, long_constant)
        exp = model.fit(ga, fit, valid, longrkm, longckm)
        prediction = model.predict_matrix()
        params = {
            "best_delta_rho": float(exp["best_delta_rho"]),
            "best_long_constant": float(exp["best_long_constant"]),
            "best_wiener_const": 9500.0,
            "best_validation_rmse": float(np.min(exp["rmse_matrix"])),
        }
        params = add_parameter_grid_record(
            params, OUTPUT_BASE_DIR, region, run_tag, method, exp,
            [("delta_rho", delta_rho), ("long_constant", long_constant)],
        )
    elif method == "vparker":
        # 论文标定网格:Δρ 1–3(步长 0.1,21 点)× 长波宽度 (5…14)。
        # 噪声常数(低通截止)跨量级搜索:720 原生网格上验证最优落在 9500;
        # 小值候选保证粗网格/含混叠噪声的输入上 Parker 迭代仍有稳定解。
        delta_rho = np.linspace(1, 3, 21)
        noise_const = (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 20.0, 100.0,
                       500.0, 2000.0, 9500.0, 50000.0)
        long_width = (5, 6, 7, 8, 10, 12, 14)
        ga_width = max(5, min(120, max(ga.shape[0], ga.shape[1]) // 6))
        model = VParkerPlus(delta_rho, noise_const, long_width,
                            mu=0.01, alpha=32, ga_width=ga_width, t=5)
        exp = model.fit(ga, fit, valid, longrkm, longckm)
        prediction = model.predict_matrix()
        params = {
            "best_delta_rho": float(exp["best_delta_rho"]),
            "best_noise_const": float(exp["best_noise_const"]),
            "best_long_width": float(exp["best_long_width"]),
            "best_validation_rmse": float(np.min(exp["rmse_matrix"])),
        }
        params = add_parameter_grid_record(
            params, OUTPUT_BASE_DIR, region, run_tag, method, exp,
            [("delta_rho", delta_rho), ("noise_const", noise_const),
             ("long_width", long_width)],
        )
    elif method in NN_METHODS:
        raise ValueError(
            f"{METHOD_LABELS[method]} 不属于本开源包(依赖内部训练框架);"
            "其结果行会直接从既有 CSV 沿用参与汇总。"
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    prediction = np.asarray(prediction, dtype=float)
    result = metrics(prediction, test)
    result.update(
        {
            "repeat": int((REPEAT_SEEDS.index(seed) + 1)
                          if seed in REPEAT_SEEDS else -1),
            "seed": int(seed),
            "region": region_id(data["region"]),
            "method": METHOD_LABELS[method],
        }
    )
    return result, prediction, params


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_prediction(prediction, split, data, repeat, method):
    if not SAVE_FULL_PREDICTION:
        return
    out = (
        OUTPUT_BASE_DIR / "predictions" / f"repeat_{repeat:02d}"
        / region_id(data["region"])
    )
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{method}.npy", prediction)
    np.save(out / "test_truth.npy", split["test"])


def summarize(records):
    """输出区域-方法均值/标准差,以及跨区域宏平均和样本加权 RMSE。"""
    metrics_fields = ("RMSE", "MAE", "Bias", "R2", "MAPE",
                      "Max_Error", "Min_Error")
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["region"], row["method"])].append(row)
    region_rows = []
    for (region, method), rows in sorted(grouped.items()):
        output = {"region": region, "method": method, "n_repeats": len(rows)}
        for field in metrics_fields:
            values = np.asarray([row[field] for row in rows], dtype=float)
            output[f"{field}_mean"] = float(np.nanmean(values))
            output[f"{field}_std"] = (
                float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan
            )
        output["n_samples_mean"] = float(np.mean([row["n_samples"] for row in rows]))
        region_rows.append(output)
    overall_rows = []
    for method in (METHOD_LABELS[name] for name in ALL_METHODS):
        rows = [row for row in records if row["method"] == method]
        if not rows:
            continue
        rmse = np.asarray([row["RMSE"] for row in rows], dtype=float)
        n = np.asarray([row["n_samples"] for row in rows], dtype=float)
        output = {
            "method": method,
            "n_region_repeat": len(rows),
            "RMSE_macro_mean": float(np.nanmean(rmse)),
            "RMSE_macro_std": float(np.nanstd(rmse, ddof=1)),
            "RMSE_sample_weighted": float(
                np.sqrt(np.nansum(rmse ** 2 * n) / np.nansum(n))),
            "MAE_macro_mean": float(np.nanmean([row["MAE"] for row in rows])),
            "Bias_macro_mean": float(np.nanmean([row["Bias"] for row in rows])),
            "R2_macro_mean": float(np.nanmean([row["R2"] for row in rows])),
            "MAPE_macro_mean": float(np.nanmean([row["MAPE"] for row in rows])),
        }
        ranked = []
        best_count = 0
        for key in sorted({(row["repeat"], row["region"]) for row in rows}):
            group = [row for row in records
                     if row["repeat"] == key[0] and row["region"] == key[1]]
            values = np.asarray([row["RMSE"] for row in group], dtype=float)
            this_value = next(row["RMSE"] for row in group if row["method"] == method)
            ranked.append(1 + int(np.sum(values < this_value)))
            if np.isclose(this_value, np.nanmin(values), rtol=0, atol=1e-12):
                best_count += 1
        output["mean_rank"] = float(np.mean(ranked))
        output["best_count"] = int(best_count)
        overall_rows.append(output)
    return region_rows, overall_rows


def main():
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    run_regions = regions_to_run()
    rerun_ids = {region_id(region) for region in run_regions}
    reused_ids = [region_id(region) for region in REGIONS
                  if region_id(region) not in rerun_ids]
    print("=" * 72)
    print(f"Regions to run : {sorted(rerun_ids)}")
    print(f"Regions reused : {reused_ids if reused_ids else 'none'}")
    print(f"Data directory : {DATA_DIR}")
    print("=" * 72)
    config = {
        "regions": [list(region) for region in REGIONS],
        "regions_run": [region_id(region) for region in run_regions],
        "methods": [METHOD_LABELS[name] for name in ALL_METHODS],
        "methods_rerun": [METHOD_LABELS[name] for name in METHODS],
        "methods_reused": [METHOD_LABELS[name] for name in NN_METHODS],
        "repeat_seeds": REPEAT_SEEDS,
        "outer_block_size": OUTER_BLOCK_SIZE,
        "outer_test_fraction": OUTER_TEST_FRACTION,
        "inner_block_size": INNER_BLOCK_SIZE,
        "inner_validation_fraction": INNER_VALIDATION_FRACTION,
        "save_full_prediction": SAVE_FULL_PREDICTION,
        "data_dir": str(DATA_DIR),
        "note": ("Physics methods tune on inner fit/validation; neural methods "
                 "are not part of this open-source build and reuse previous CSV rows."),
    }
    (OUTPUT_BASE_DIR / "experiment_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    all_records = []
    split_records = []
    parameter_records = []
    for region_index, region in enumerate(run_regions, start=1):
        print(f"\n[{region_index}/{len(run_regions)}] Preparing region {region}")
        data = prepare_region(region)
        print(f"  grid {data['target_shape']}, long {data['longrkm']:.0f} x "
              f"{data['longckm']:.0f} km, ship points "
              f"{int(np.count_nonzero(data['ship']))}")
        for repeat, seed in enumerate(REPEAT_SEEDS, start=1):
            print(f"  Repeat {repeat}/5, seed={seed}")
            split = make_shared_split(data, seed)
            split_records.append(save_split(data, split, repeat, seed))
            for method in METHODS:
                print(f"    Running {METHOD_LABELS[method]}")
                run_tag = f"repeat_{repeat:02d}_seed_{seed}"
                result, prediction, params = run_method(
                    method, data, split, seed, run_tag=run_tag
                )
                result["repeat"] = repeat
                all_records.append(result)
                parameter_records.append(
                    {
                        "repeat": repeat,
                        "seed": seed,
                        "region": region_id(region),
                        "method": METHOD_LABELS[method],
                        "parameters": json.dumps(params, ensure_ascii=False),
                    }
                )
                save_prediction(prediction, split, data, repeat, method)
    expected_per_region = len(ALL_METHODS) * len(REPEAT_SEEDS)
    rerun_method_labels = {METHOD_LABELS[name] for name in METHODS}
    previous_metrics, stale_metrics = load_previous_rows(
        OUTPUT_BASE_DIR / "metrics_all_repeats.csv", rerun_ids, rerun_method_labels
    )
    previous_splits, _ = load_previous_rows(
        OUTPUT_BASE_DIR / "split_summary.csv", rerun_ids
    )
    previous_parameters, _ = load_previous_rows(
        OUTPUT_BASE_DIR / "parameters_all_repeats.csv", rerun_ids, rerun_method_labels
    )
    if stale_metrics:
        print(f"Dropped regions no longer in REGIONS: {stale_metrics}")
    if previous_metrics:
        per_region = defaultdict(int)
        for row in previous_metrics:
            per_region[row["region"]] += 1
        reused_expected = len(NN_METHODS) * len(REPEAT_SEEDS)
        for region_code in sorted(rerun_ids):
            count = per_region.get(region_code, 0)
            if count != reused_expected:
                print(
                    f"Warning: region {region_code} is being rerun and should reuse "
                    f"{reused_expected} previous rows but found {count}."
                )
        print(
            f"Reused {len(previous_metrics)} rows from previous runs "
            f"(regions: {sorted(per_region)})."
        )
    else:
        print("No previous results found; summaries contain only rerun regions.")
        if NN_METHODS:
            print(
                "Warning: " + ", ".join(METHOD_LABELS[n] for n in NN_METHODS)
                + " are not rerun, so their rows will be missing entirely."
            )
    all_records = previous_metrics + all_records
    split_records = previous_splits + split_records
    parameter_records = previous_parameters + parameter_records
    all_records.sort(key=record_sort_key)
    split_records.sort(key=record_sort_key)
    parameter_records.sort(key=record_sort_key)
    detail_fields = [
        "repeat", "seed", "region", "method",
        "RMSE", "MAE", "Bias", "R2", "MAPE", "Max_Error", "Min_Error",
        "n_samples",
    ]
    write_csv(OUTPUT_BASE_DIR / "metrics_all_repeats.csv", all_records, detail_fields)
    write_csv(
        OUTPUT_BASE_DIR / "split_summary.csv",
        split_records,
        ["repeat", "seed", "region", "train_samples", "test_samples",
         "fit_samples", "valid_samples"],
    )
    write_csv(
        OUTPUT_BASE_DIR / "parameters_all_repeats.csv",
        parameter_records,
        ["repeat", "seed", "region", "method", "parameters"],
    )
    region_summary, overall_summary = summarize(all_records)
    region_fields = ["region", "method", "n_repeats", "n_samples_mean"]
    for field in ("RMSE", "MAE", "Bias", "R2", "MAPE", "Max_Error", "Min_Error"):
        region_fields.extend([f"{field}_mean", f"{field}_std"])
    write_csv(OUTPUT_BASE_DIR / "summary_by_region_method.csv",
              region_summary, region_fields)
    write_csv(
        OUTPUT_BASE_DIR / "summary_overall_by_method.csv",
        overall_summary,
        ["method", "n_region_repeat", "RMSE_macro_mean", "RMSE_macro_std",
         "RMSE_sample_weighted", "MAE_macro_mean", "Bias_macro_mean",
         "R2_macro_mean", "MAPE_macro_mean", "mean_rank", "best_count"],
    )
    print("\nFinished.")
    print(f"Regions rerun  : {sorted(rerun_ids)}")
    print(f"Methods rerun  : {[METHOD_LABELS[n] for n in METHODS]}")
    print(f"Result rows    : {len(all_records)}")
    print(f"Results written to: {OUTPUT_BASE_DIR}")


if __name__ == "__main__":
    main()
