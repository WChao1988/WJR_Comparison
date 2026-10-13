from scipy.fftpack import fft2, ifft2
from scipy.ndimage import gaussian_filter
import numpy as np


class SandwellLongwave:
    def __init__(self, constant_long=30):
        self.constant_long = constant_long
        self.filtered_matrix = None

    def fit(self, origin_matrix, longrkm, longckm):
        spectrum = fft2(origin_matrix)
        nrow, ncol = origin_matrix.shape
        frequency = np.zeros_like(origin_matrix)
        for i in range(nrow):
            for j in range(ncol):
                ii = i if i <= nrow / 2 else i - nrow
                jj = j if j <= ncol / 2 else j - ncol
                frequency[i, j] = np.sqrt((ii / longrkm) ** 2 + (jj / longckm) ** 2)
        lowpass_filter = np.exp(-2*(np.pi*frequency * self.constant_long)**2)
        filtered_spectrum = lowpass_filter * spectrum
        self.filtered_matrix = ifft2(filtered_spectrum).real

    def fit_sparse(self, origin_matrix, longrkm, longckm, sparse_value=0.0):
        valid_mask = origin_matrix != sparse_value
        weights = valid_mask.astype(float)
        values = np.where(valid_mask, origin_matrix, 0.0)
        smoothed_values = gaussian_filter(values, sigma=3, mode="constant", truncate=30, cval=0.0)
        smoothed_weights = gaussian_filter(weights, sigma=3, mode="constant", truncate=30, cval=0.0)
        filled_matrix = smoothed_values / smoothed_weights
        self.fit(filled_matrix, longrkm, longckm)

    def fit_sparse_safety(self, origin_matrix, longrkm, longckm, sparse_value=0.0):
        valid_mask = origin_matrix != sparse_value
        weights = valid_mask.astype(float)
        values = np.where(valid_mask, origin_matrix, 0.0)
        sigma = 1
        smoothed_values = gaussian_filter(values, sigma=sigma, mode="constant", truncate=4, cval=0.0)
        smoothed_weights = gaussian_filter(weights, sigma=sigma, mode="constant", truncate=4, cval=0.0)
        mask_invalidate_matrix = smoothed_weights != 0.0
        filled_matrix = np.zeros_like(origin_matrix)
        filled_matrix[mask_invalidate_matrix] = smoothed_values[mask_invalidate_matrix] / smoothed_weights[
            mask_invalidate_matrix]
        while np.any(~mask_invalidate_matrix):
            sigma = 2 * sigma
            smoothed_values_current = gaussian_filter(values, sigma=sigma, mode="constant", truncate=4, cval=0.0)
            smoothed_weights_current = gaussian_filter(weights, sigma=sigma, mode="constant", truncate=4, cval=0.0)
            mask_invalidate_matrix_current = smoothed_weights_current != 0.0
            filled_matrix_current = np.zeros_like(origin_matrix)
            filled_matrix_current[mask_invalidate_matrix_current] = smoothed_values_current[
                                                                        mask_invalidate_matrix_current] / \
                                                                    smoothed_weights_current[
                                                                        mask_invalidate_matrix_current]
            filled_matrix[~mask_invalidate_matrix] = filled_matrix_current[~mask_invalidate_matrix]
            mask_invalidate_matrix = filled_matrix != 0.0
        self.fit(filled_matrix, longrkm, longckm)

    def fit_transform(self, origin_matrix, longrkm, longckm):
        self.fit(origin_matrix, longrkm, longckm)
        return self.filtered_matrix

    def fit_transform_sparse(self, origin_matrix, longrkm, longckm, sparse_value=0.0):
        self.fit_sparse(origin_matrix, longrkm, longckm, sparse_value)
        return self.filtered_matrix

    def fit_transform_sparse_safety(self, origin_matrix, longrkm, longckm, sparse_value=0.0):
        self.fit_sparse_safety(origin_matrix, longrkm, longckm, sparse_value)
        return self.filtered_matrix

    def predict(self, target_mask):
        return self.filtered_matrix[target_mask]

    def predict_matrix(self):
        return self.filtered_matrix



