import numpy as np
from scipy.ndimage import gaussian_filter


class GaussianLongwave:
    def __init__(self, sigma):
        self.sigma = sigma
        self.filtered_matrix = None

    def fit(self, origin_matrix):
        self.filtered_matrix = gaussian_filter(origin_matrix, sigma=self.sigma, mode='constant', truncate=10, cval=0.0)

    def fit_transform(self, origin_matrix):
        self.fit(origin_matrix)
        return self.filtered_matrix

    def fit_sparse(self, origin_matrix, sparse_value=0.0):
        valid_mask = origin_matrix != sparse_value
        weights = valid_mask.astype(float)
        values = np.where(valid_mask, origin_matrix, 0.0)
        smoothed_values = gaussian_filter(values, sigma=self.sigma, mode="constant", truncate=30, cval=0.0)
        smoothed_weights = gaussian_filter(weights, sigma=self.sigma, mode="constant", truncate=30, cval=0.0)
        mask_validate_matrix = smoothed_weights != 0
        self.filtered_matrix = np.zeros_like(origin_matrix)
        self.filtered_matrix[mask_validate_matrix] = smoothed_values[mask_validate_matrix] / smoothed_weights[
            mask_validate_matrix]

    def fit_sparse_safety(self, origin_matrix, sparse_value=0.0):
        origin_sigma = self.sigma
        self.fit_sparse(origin_matrix)
        filtered_matrix = self.filtered_matrix
        mask_invalidate_matrix = filtered_matrix == 0.0
        while np.any(mask_invalidate_matrix):
            self.sigma = 2 * self.sigma
            self.fit_sparse(origin_matrix, sparse_value)
            filter_matrix_current = self.filtered_matrix
            filtered_matrix[mask_invalidate_matrix] = filter_matrix_current[mask_invalidate_matrix]
            mask_invalidate_matrix = filtered_matrix == 0.0
        self.filtered_matrix = filtered_matrix
        self.sigma = origin_sigma

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