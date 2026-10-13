from .clough_longwave import CloughLongwave
from .ggm import GGMModel
import numpy as np
from multiprocessing import Pool
from numba import njit, prange
from functools import partial


@njit(parallel=True, fastmath=True)
def _calculate_tc_gravity_(padded_grid, mask_ship, width, step_rm, step_cm):
    rows, cols = np.nonzero(mask_ship)
    local_x_distance = np.square(np.linspace(-width, width, 2*width+1) * step_rm)
    local_y_distance = np.square(np.linspace(-width, width, 2*width+1) * step_cm)
    local_distance = np.zeros((2*width+1, 2*width+1))
    for local_i in prange(2*width+1):
        for local_j in prange(2*width+1):
            local_distance[local_i, local_j] = local_x_distance[local_i] + local_y_distance[local_j]
    # pad
    output = np.zeros(rows.shape[0])
    for index in prange(rows.shape[0]):
        irow, icol = rows[index] + width, cols[index] + width
        window = (padded_grid[irow-width:irow+width+1, icol-width:icol+width+1])**2
        idepth = (padded_grid[irow, icol])**2
        refine_value = -np.sum(1 / np.sqrt(window + local_distance) - 1 / np.sqrt(idepth + local_distance))
        output[index] = refine_value
    result = output * step_rm * step_cm
    return result


class EGGMModel:
    def __init__(self, delta_rho2, width, delta_rho1, sigma):
        self.delta_rho1 = delta_rho1
        self.delta_rho2 = delta_rho2
        self.width = int(width)
        self.sigma = sigma
        # initial
        self.G = 6.67*1e-3
        self.reference_depth = None
        self.ggm_model = GGMModel(delta_rho1, sigma)
        # gravity long model
        self.ga_long_model = CloughLongwave(sigma)
        # padded prior background topography
        self.padded_prior_grid = None
        self.ga_grid = None
        self.longrm, self.longcm = None, None

    def fit(self, ga_grid, ship_fit, longrm, longcm):
        self.ga_grid = ga_grid
        self.longrm, self.longcm = longrm, longcm
        mask_ship = ship_fit != 0
        self.ggm_model.fit(ga_grid, ship_fit)
        # calculate reference_depth
        self.reference_depth = ship_fit[mask_ship].mean()
        # prior matrix
        prior_grid = self.ggm_model.predict_matrix()
        # calculate short gravity anomalies from ship fit
        ga_short_sequence = 2 * np.pi * self.G * self.delta_rho1 * (ship_fit[mask_ship] - self.reference_depth)
        # calculate complement gravity anomalies
        nrow, ncol = ga_grid.shape
        step_rm, step_cm = longrm / nrow, longcm / ncol
        padded_grid = np.pad(prior_grid, pad_width=self.width, mode='reflect')
        self.padded_prior_grid = padded_grid
        ga_tc_squence = _calculate_tc_gravity_(padded_grid, mask_ship, self.width, step_rm,
                                               step_cm) * self.G * self.delta_rho2
        # calculate long gravity anomalies
        ga_long_grid = np.zeros_like(ga_grid)
        ga_long_grid[mask_ship] = ga_grid[mask_ship] - ga_short_sequence - ga_tc_squence
        # long gravity anomalies model
        self.ga_long_model.fit_sparse_safety(ga_long_grid)

    def predict(self, mask_target):
        nrow, ncol = mask_target.shape
        step_rm, step_cm = self.longrm / nrow, self.longcm / ncol
        ga_tc_target_sequence = _calculate_tc_gravity_(self.padded_prior_grid, mask_target, self.width, step_rm,
                                                       step_cm) * self.G * self.delta_rho2
        ga_long_sequence = self.ga_long_model.predict(mask_target)
        ga_short = self.ga_grid[mask_target] - ga_tc_target_sequence - ga_long_sequence
        return ga_short / 2 / np.pi / self.G / self.delta_rho1 + self.reference_depth

    def predict_matrix(self):
        nrow, ncol = self.ga_grid.shape
        step_rm, step_cm = self.longrm / nrow, self.longcm / ncol
        mask_target = np.ones_like(self.ga_grid).astype(bool)
        ga_tc_grid = _calculate_tc_gravity_(self.padded_prior_grid, mask_target, self.width, step_rm,
                                            step_cm) * self.G * self.delta_rho2
        ga_long_grid = self.ga_long_model.predict_matrix()
        ga_short_grid = self.ga_grid - ga_tc_grid.reshape(nrow, ncol) - ga_long_grid
        return ga_short_grid / 2 / np.pi / self.G / self.delta_rho1 + self.reference_depth


def _train_and_evaluate(params, ga_grid, ship_fit, ship_validate, longrm, longcm, mask_validate):
    i, j, dr2, width, dr1, sigma = params
    # 创建模型并训练
    model = EGGMModel(delta_rho1=dr1, delta_rho2=dr2, width=width, sigma=sigma)
    model.fit(ga_grid, ship_fit, longrm, longcm)
    # 预测验证集
    pred_depth = model.predict(mask_validate)
    # 真实深度
    true_depth = ship_validate[mask_validate]
    # 计算 RMSE
    rmse = np.sqrt(np.mean((pred_depth - true_depth) ** 2))
    return i, j, rmse


class EGGMPlus:
    def __init__(self, delta_rho2_values=(1.3, 1.6, 1.9),
                 width_values=(5, 10, 15),
                 delta_rho1 = 1.1,
                 sigma=4):
        """
        :param delta_rho1_values: density contrast for ground topography => short component
        :param delta_rho2_values: density contrast for compliment => corrected short component
        :param width_values: width for correction window => corrected short component
        :param sigma_values: window size for long gravity anomaly calculation => long gravity anomaly
        """
        self.delta_rho1 = delta_rho1
        self.delta_rho2_values = delta_rho2_values
        self.width_values = width_values
        self.sigma = sigma
        self.eggm_model = None

    def fit(self, ga_grid, ship_fit, ship_validate, longrm, longcm):
        mask_validate = ship_validate != 0
        n_delta_rho2 = len(self.delta_rho2_values)
        n_width = len(self.width_values)
        # formulate the parameter list
        params_list = []
        for i, dr2 in enumerate(self.delta_rho2_values):
            for j, w in enumerate(self.width_values):
                        params_list.append((i, j, dr2, w, self.delta_rho1, self.sigma))
        # initialize the worker
        worker = partial(
            _train_and_evaluate,
            ga_grid=ga_grid,
            ship_fit=ship_fit,
            ship_validate=ship_validate,
            longrm=longrm,
            longcm=longcm,
            mask_validate=mask_validate
        )
        with Pool(processes=None) as pool:
            results = pool.map(worker, params_list)
        rmse_matrix = np.zeros((n_delta_rho2, n_width))
        for i, j, rmse in results:
            rmse_matrix[i, j] = rmse
        besti, bestj = np.unravel_index(np.argmin(rmse_matrix), rmse_matrix.shape)
        parameter_exploration = {
            "delta_rho2s": self.delta_rho2_values,
            "widths": self.width_values,
            "rmse_matrix": rmse_matrix,
            "best_delta_rho2": self.delta_rho2_values[besti],
            "best_width": self.width_values[bestj]
        }
        self.eggm_model = EGGMModel(delta_rho1=self.delta_rho1,
                                    delta_rho2=self.delta_rho2_values[besti],
                                    width=self.width_values[bestj],
                                    sigma=self.sigma)
        self.eggm_model.fit(ga_grid, ship_fit+ship_validate, longrm, longcm)
        return parameter_exploration

    def predict(self, mask_target):
        return self.eggm_model.predict(mask_target)

    def predict_matrix(self):
        return self.eggm_model.predict_matrix()




























