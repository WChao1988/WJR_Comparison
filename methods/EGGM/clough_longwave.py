import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.interpolate import griddata


class CloughLongwave:
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
        target_size = origin_matrix.shape
        x = np.arange(0, target_size[0], 1)
        y = np.arange(0, target_size[1], 1)
        lon_grid, lat_grid = np.meshgrid(x, y)
        # fetch the x, y, z
        pts = np.column_stack((lon_grid[valid_mask], lat_grid[valid_mask]))
        vals = origin_matrix[valid_mask]
        # interpolated
        grid_z_cubic = griddata(pts, vals, (lon_grid, lat_grid), method='linear', fill_value=sparse_value)
        valid_mask2 = grid_z_cubic != 0
        weights = valid_mask2.astype(float)
        values = grid_z_cubic
        smoothed_values = gaussian_filter(values, sigma=self.sigma, mode="constant", truncate=10, cval=0.0)
        smoothed_weights = gaussian_filter(weights, sigma=self.sigma, mode="constant", truncate=10, cval=0.0)
        mask_validate_matrix = smoothed_weights != 0
        self.filtered_matrix = np.zeros_like(grid_z_cubic)
        self.filtered_matrix[mask_validate_matrix] = smoothed_values[mask_validate_matrix] / smoothed_weights[
            mask_validate_matrix]

    def fit_sparse_safety(self, origin_matrix, sparse_value=0.0):
        valid_mask = origin_matrix != sparse_value
        target_size = origin_matrix.shape
        x = np.arange(0, target_size[0], 1)
        y = np.arange(0, target_size[1], 1)
        lon_grid, lat_grid = np.meshgrid(x, y)
        # fetch the x, y, z
        pts = np.column_stack((lon_grid[valid_mask], lat_grid[valid_mask]))
        vals = origin_matrix[valid_mask]
        # interpolated
        grid_z_cubic = griddata(pts, vals, (lon_grid, lat_grid), method='linear', fill_value=sparse_value)
        valid_mask2 = grid_z_cubic != 0
        weights = valid_mask2.astype(float)
        values = grid_z_cubic
        smoothed_values = gaussian_filter(values, sigma=self.sigma, mode="constant", truncate=10, cval=0.0)
        smoothed_weights = gaussian_filter(weights, sigma=self.sigma, mode="constant", truncate=10, cval=0.0)
        mask_validate_matrix = smoothed_weights != 0
        self.filtered_matrix = np.zeros_like(grid_z_cubic)
        self.filtered_matrix[mask_validate_matrix] = smoothed_values[mask_validate_matrix] / smoothed_weights[
            mask_validate_matrix]
        origin_sigma = self.sigma
        mask_invalidate_matrix = ~mask_validate_matrix
        t = 1
        while np.any(mask_invalidate_matrix):
            print(t)
            t = t + 1
            origin_sigma = 2 * origin_sigma
            mask_validate = ~mask_invalidate_matrix
            weights = mask_validate.astype(float)
            values = self.filtered_matrix
            smoothed_values = gaussian_filter(values, sigma=origin_sigma, mode="constant", truncate=10, cval=0.0)
            smoothed_weights = gaussian_filter(weights, sigma=origin_sigma, mode="constant", truncate=10, cval=0.0)
            mask_validate_matrix = smoothed_weights != 0
            mask = mask_validate_matrix * mask_invalidate_matrix
            self.filtered_matrix[mask] = smoothed_values[mask] / smoothed_weights[mask]
            mask_invalidate_matrix = ~mask_validate_matrix

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
