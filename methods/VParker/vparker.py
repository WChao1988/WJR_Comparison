from scipy.fftpack import fft2, ifft2
import numpy as np
from scipy.signal.windows import tukey
from .gaussian_longwave import GaussianLongwave
from .clough_longwave import CloughLongwave
from functools import partial
from math import factorial
from multiprocessing import Pool


class VParker:
    def __init__(self, delta_rho, wh, long_width, mu,  alpha, ga_width, t=5):
        """
        :param mu: 0~0.1 density gradient
        :param delta_rho: 0.5~2.5 density contrast
        :param wh: filter cutoff for noise gravity anomalies
        :param alpha: filter compress content for noise gravity anomalies
        :param ga_width: filter width for gravity anomalies useful
        :param long_width: long wave topography width
        """
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

    @classmethod
    def apply_truncation_window(cls, matrix, edge=0.02):
        tky = np.array([row_tky * col_tky for row_tky in tukey(matrix.shape[0], edge)
                        for col_tky in tukey(matrix.shape[1], edge)]).reshape(matrix.shape)
        return tky * matrix

    @classmethod
    def __cul_omega__(cls, ga_grid, longrkm, longckm):
        frequency = np.zeros(ga_grid.shape)
        for i in range(ga_grid.shape[0]):
            for j in range(ga_grid.shape[1]):
                ii = i if i <= ga_grid.shape[0] / 2 else i - ga_grid.shape[0]
                jj = j if j <= ga_grid.shape[1] / 2 else j - ga_grid.shape[1]
                frequency[i, j] = 2 * np.pi * np.sqrt((ii / longrkm) ** 2 + (jj / longckm) ** 2)
        return frequency

    def __build_lowpass_filter__(self, frequency):
        _filtor = np.ones(frequency.shape)
        for i in range(frequency.shape[0]):
            for j in range(frequency.shape[1]):
                if frequency[i, j] > self.wh:
                    ratio = frequency[i, j] / self.wh
                    _filtor[i, j] = ratio ** (1 - self.alpha) - (1 - self.alpha) * np.log(ratio) * ratio ** (
                            1 - self.alpha)
        return _filtor

    def fit(self, ga_grid, ship_fit, longrkm, longckm, ga_grid_short=None):
        if ga_grid_short is None:
            ga_grid_short = ga_grid - GaussianLongwave(sigma=self.ga_width).fit_transform(ga_grid)
        # calculate the frequency and lowpass filter
        frequency = self.__cul_omega__(ga_grid_short, longrkm, longckm)
        lowpass_filter = self.__build_lowpass_filter__(frequency)
        # calculate reference depth
        mask_ship = ship_fit != 0
        self.reference_depth = np.abs(ship_fit[mask_ship]).mean() * 1e-3
        # calculate gravity frequency component
        tuk_ga_grid_short = self.apply_truncation_window(ga_grid_short)
        tuk_ga_grid_short_fft = fft2(tuk_ga_grid_short)
        transformed_ga_fft = tuk_ga_grid_short_fft * np.exp(
            self.reference_depth * frequency) / 2 / np.pi / self.G / self.delta_rho
        transformed_ga_fft[0, 0] = 0j
        topo_initial = ifft2(transformed_ga_fft * lowpass_filter).real
        for i in range(2, self.t + 1):
            topo_i_fft = 0 + 0j
            for j in range(2, i + 1):
                coef = (frequency - self.mu) ** (j-1) / factorial(j)
                topo_i_fft += coef * fft2(topo_initial ** j)
            topo_i_fft_first = transformed_ga_fft - topo_i_fft
            topo_i_fft_first[0, 0] = 0j
            topo_initial = ifft2(topo_i_fft_first * lowpass_filter).real
        # calculate the compliment short and long topography
        self.short_topography = topo_initial - CloughLongwave(self.long_width).fit_transform_sparse(topo_initial)
        self.long_topography = CloughLongwave(self.long_width).fit_transform_sparse_safety(ship_fit)

    def predict(self, mask_target):
        sequence = self.long_topography[mask_target] + self.short_topography[mask_target] * 1e3
        return sequence

    def predict_matrix(self):
        return self.long_topography + self.short_topography * 1e3


def _train_and_evaluate(params, ga_grid, ship_fit, ship_validate, mask_validate, longrkm, longckm, ga_grid_short):
    i, j, k, dr, wh, lw, mu, ap, gw, tt = params
    model = VParker(dr, wh, lw, mu, ap, gw, tt)
    model.fit(ga_grid, ship_fit, longrkm, longckm, ga_grid_short)
    pred_depth = model.predict(mask_validate)
    true_depth = ship_validate[mask_validate]
    rmse = np.sqrt(np.mean(np.square(pred_depth - true_depth)))
    print("delta_rho={}; wh={}; long_width={}; rmse={}".format( dr, wh, lw, rmse))
    return i, j, k, rmse


class VParkerPlus:
    def __init__(self,
                 delta_rho_values=(1, 1.5, 2),
                 wh_values=(0.15, 0.2, 0.25),
                 long_width_values=(2, 3, 4),
                 mu=0,
                 alpha=32,
                 ga_width=120,
                 t=5):

        self.delta_rho_values = delta_rho_values
        self.wh_values = wh_values
        self.long_width_values = long_width_values
        self.mu = mu
        self.alpha = alpha
        self.ga_width = ga_width
        self.t = t
        self.vparker_model =None

    def fit(self, ga_grid, ship_fit, ship_validate, longrkm, longckm):
        mask_validate = ship_validate != 0
        nni, nnj, nnk = len(self.delta_rho_values), len(self.wh_values), len(self.long_width_values)
        parameter_list = []
        for i in range(nni):
            for j in range(nnj):
                for k in range(nnk):
                    param = (i, j, k, self.delta_rho_values[i], self.wh_values[j], self.long_width_values[k],
                             self.mu, self.alpha, self.ga_width, self.t)
                    parameter_list.append(param)
        ga_grid_short = ga_grid - GaussianLongwave(sigma=self.ga_width).fit_transform(ga_grid)
        worker = partial(_train_and_evaluate,
                         ga_grid=ga_grid,
                         ship_fit=ship_fit,
                         ship_validate=ship_validate,
                         mask_validate=mask_validate,
                         longrkm=longrkm,
                         longckm=longckm,
                         ga_grid_short=ga_grid_short)
        with Pool(processes=16) as pool:
            results = pool.map(worker, parameter_list)
        rmse_matrix = np.zeros((nni, nnj, nnk))
        for i, j, k, rmse in results:
            rmse_matrix[i, j, k] = rmse
        bti, btj, btk = np.unravel_index(np.argmin(rmse_matrix), rmse_matrix.shape)
        parameter_exploration = {
            "delta_rhos": self.delta_rho_values,
            "whs": self.wh_values,
            "long_widths": self.long_width_values,
            "rmse_matrix": rmse_matrix,
            "best_delta_rho": self.delta_rho_values[bti],
            "best_wh": self.wh_values[btj],
            "best_long_width": self.long_width_values[btk],
        }
        self.vparker_model = VParker(mu=self.mu,
                                     delta_rho=self.delta_rho_values[bti],
                                     wh=self.wh_values[btj],
                                     alpha=self.alpha,
                                     ga_width=self.ga_width,
                                     long_width=self.long_width_values[btk],
                                     t=self.t)
        self.vparker_model.fit(ga_grid, ship_fit+ship_validate, longrkm, longckm, ga_grid_short)
        return parameter_exploration

    def predict(self, mask_target):
        return self.vparker_model.predict(mask_target)

    def predict_matrix(self):
        return self.vparker_model.predict_matrix()





