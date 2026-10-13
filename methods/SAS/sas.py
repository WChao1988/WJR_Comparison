from scipy.fftpack import fft2, ifft2
import numpy as np
from .sandwell_longwave import SandwellLongwave
from functools import partial
from multiprocessing import Pool


class SAS:
    def __init__(self, delta_rho=1.0, long_constant=30):
        self.delta_rho = delta_rho
        self.long_constant = long_constant
        self.G = 6.67
        self.frequency = None
        self.reference_depth = None
        self.sandwell_longwave_model = SandwellLongwave(long_constant)
        self.band_topography = None

    @classmethod
    def __frequency__(cls, ga_grid, longrkm, longckm):
        nrow, ncol = ga_grid.shape
        frequency = np.zeros((nrow, ncol))
        for i in range(nrow):
            for j in range(ncol):
                ii = i if i <= nrow / 2 else i - nrow
                jj = j if j <= ncol / 2 else j - ncol
                frequency[i, j] = np.sqrt((ii / longrkm) ** 2 + (jj / longckm) ** 2)
        return frequency

    def highpass_filter(self, frequency):
        return 1 - np.exp(-2 * (np.pi * frequency * self.long_constant) ** 2)

    def lowpass_filter(self, frequency):
        return 1 / (1 + 9500 * frequency ** 4 * np.exp(4 * np.pi * frequency * self.reference_depth))

    def fit(self, ga_grid, ship_fit, longrkm, longckm):
        # calculate the frequency
        self.frequency = self.__frequency__(ga_grid, longrkm, longckm)
        mask_ship = ship_fit != 0
        self.reference_depth = np.abs(ship_fit[mask_ship].mean() * 1e-3)
        # calculate the filters
        highpass_filter = self.highpass_filter(self.frequency)
        lowpass_filter = self.lowpass_filter(self.frequency)
        # form long topography model
        self.sandwell_longwave_model.fit_transform_sparse_safety(ship_fit,longrkm, longckm, sparse_value=0.0)
        # calculate short topograph grid
        gravity_fft2 = fft2(ga_grid) * highpass_filter * lowpass_filter * np.exp(
            2 * np.pi * self.frequency * self.reference_depth)
        self.band_topography = ifft2(gravity_fft2 / 2 / np.pi / self.G).real * 1e3 / self.delta_rho

    def predict(self, mask_target):
        topography = self.sandwell_longwave_model.predict_matrix() + self.band_topography
        return topography[mask_target]

    def predict_matrix(self):
        topography = self.sandwell_longwave_model.predict_matrix() + self.band_topography
        return topography


def _train_and_evaluate(params, ga_grid, ship_fit, ship_validate, mask_validate, longrkm, longckm):
    i, j, delta_rho, long_constant = params
    model = SAS(delta_rho, long_constant)
    model.fit(ga_grid, ship_fit, longrkm, longckm)
    pred_depth = model.predict(mask_validate)
    true_depth = ship_validate[mask_validate]
    rmse = np.sqrt(np.mean((true_depth - pred_depth) ** 2))
    return i, j, rmse


class SASPlus:
    def __init__(self, delta_rho_values=(0.5, 1.0, 1.5), long_constant_values=(5, 10, 15)):
        self.delta_rho_values = delta_rho_values
        self.long_constant_values = long_constant_values
        self.sas_model = None

    def fit(self, ga_grid, ship_fit, ship_validate, longrkm, longckm):
        parameters_list = []
        for i, delta_rho in enumerate(self.delta_rho_values):
            for j, long_constant in enumerate(self.long_constant_values):
                parameters_list.append((i, j, delta_rho, long_constant))
        mask_validate = ship_validate != 0
        worker = partial(
            _train_and_evaluate,
            ga_grid=ga_grid,
            ship_fit=ship_fit,
            ship_validate=ship_validate,
            mask_validate=mask_validate,
            longrkm=longrkm,
            longckm=longckm,
        )
        with Pool(processes=None) as pool:
            results = pool.map(worker, parameters_list)
        rmse_matrix = np.zeros((len(self.delta_rho_values), len(self.long_constant_values)))
        for i, j, rmse in results:
            rmse_matrix[i, j] = rmse
        besti, bestj = np.unravel_index(np.argmin(rmse_matrix, axis=None), rmse_matrix.shape)
        parameter_exploration = {
            "delta_rhos": self.delta_rho_values,
            "long_constants": self.long_constant_values,
            "rmse_matrix": rmse_matrix,
            "best_delta_rho": self.delta_rho_values[besti],
            "best_long_constant": self.long_constant_values[bestj]
        }
        self.sas_model = SAS(self.delta_rho_values[besti], self.long_constant_values[bestj])
        self.sas_model.fit(ga_grid, ship_fit+ship_validate, longrkm, longckm)
        return parameter_exploration

    def predict(self, mask_target):
        return self.sas_model.predict(mask_target)

    def predict_matrix(self):
        return self.sas_model.predict_matrix()




