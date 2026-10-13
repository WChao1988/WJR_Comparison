import numpy as np
from .clough_longwave import CloughLongwave
from multiprocessing import Pool
from functools import partial


class GGMModel:
    def __init__(self, delta_rho, sigma):
        """
        :param delta_rho: density contrast
        :param sigma: the window size for long wave gravity calculation
        """
        self.delta_rho = delta_rho
        self.sigma = sigma
        self.G = 6.67e-3
        # initialize the long ga model
        self.ga_long_model = CloughLongwave(sigma)
        # gravity anomalies
        self.ga_grid = None
        self.reference_depth = None

    def fit(self, ga_grid, ship_fit):
        """
        :param ga_grid: gravity anomalies matrix
        :param ship_fit: ship bathymetry matrix in m
        :return:
        """
        self.ga_grid = ga_grid
        self.reference_depth = ship_fit.min()
        # calculate the short gavity as ship fit
        mask_ship = ship_fit != 0
        ga_short = np.zeros_like(ga_grid)
        ga_short[mask_ship] = 2 * np.pi * self.G * self.delta_rho * (ship_fit[mask_ship] - self.reference_depth)
        # calculate the long wave gravity
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


def _train_and_evaluate(params, ga_grid, ship_fit, ship_validate, mask_validate):
    i, j, dr, sg = params
    model = GGMModel(dr, sg)
    model.fit(ga_grid, ship_fit)
    pred_depth = model.predict(mask_validate)
    true_depth = ship_validate[mask_validate]
    rmse = np.sqrt(np.mean(np.square(pred_depth - true_depth)))
    return i, j, rmse


class GGMPlus:
    def __init__(self, delta_rho_values=(1.3, 1.6, 1.9), sigma_values=(20, 24, 36)):
        self.delta_rho_values = delta_rho_values
        self.sigma_values = sigma_values
        self.ggm_model = None

    def fit(self, ga_grid, ship_fit, ship_validate):
        mask_validate = ship_validate != 0
        n_delta_rho, n_sigma = len(self.delta_rho_values), len(self.sigma_values)
        params_list = []
        for i, dr in enumerate(self.delta_rho_values):
            for j, sg in enumerate(self.sigma_values):
                params_list.append([i, j, dr, sg])
        # initialize the worker
        worker = partial(
            _train_and_evaluate,
            ga_grid=ga_grid,
            ship_fit=ship_fit,
            ship_validate=ship_validate,
            mask_validate=mask_validate
        )
        with Pool(processes=None) as pool:
            results = pool.map(worker, params_list)
        rmse_matrix = np.zeros((n_delta_rho, n_sigma))
        for i, j, rmse in results:
            rmse_matrix[i, j] = rmse
        delta_rho_index, sigma_index = np.unravel_index(np.argmin(rmse_matrix), rmse_matrix.shape)
        parameter_exploration = {"delta_rhos": self.delta_rho_values,
                                 "sigmas": self.sigma_values,
                                 "rmse_matrix": rmse_matrix,
                                 "best_delta_rho": self.delta_rho_values[delta_rho_index],
                                 "best_sigmas": self.sigma_values[sigma_index],
                                 }
        self.ggm_model = GGMModel(self.delta_rho_values[delta_rho_index], self.sigma_values[sigma_index])
        self.ggm_model.fit(ga_grid, ship_fit+ship_validate)
        return parameter_exploration

    def predict(self, mask_target):
        return self.ggm_model.predict(mask_target)

    def predict_matrix(self):
        return self.ggm_model.predict_matrix()
