from Ocean.dataserver import GravitySwot, ShipMulti, ProductFast
from Ocean.methods.GGMs import GGMModel
import time
import numpy as np
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    start = time.time()
    lat_up, lat_down, lon_left, lon_right = 18, 15, 155, 158
    gv = GravitySwot(lat_up, lat_down, lon_left, lon_right)
    vgg_matrix = gv.gravity(gravity='faa')

    target_size = vgg_matrix.shape

    ship = ShipMulti(lat_up, lat_down, lon_left, lon_right)
    ship_grid = ship.grid_ship(target_size, method='nearest')
    mask = ship_grid != 0
    rows, cols = np.nonzero(ship_grid)
    frac = 0.8
    rows_train, rows_test, cols_train, cols_test = \
        train_test_split(rows, cols, train_size=int(len(rows) * frac), random_state=1234)
    mask_train, mask_test = np.full_like(mask, fill_value=False), np.full_like(mask, fill_value=False)
    mask_train[rows_train, cols_train], mask_test[rows_test, cols_test] = True, True
    gebco = ProductFast('gebco')
    gebco_grid = gebco.format_bathymetry(lat_up, lat_down, lon_left, lon_right, target_size)

    gm = GGMModel(1.63, lat_up, lat_down, lon_left, lon_right, radius=3, sigma=0.0001)

    reference_depth = ship_grid.min()

    gm.fit(vgg_matrix, ship_grid, reference_depth)

    prediction = gm.prediction_matrix()
    print(prediction)
    np.savetxt(f'ggm_{lat_up}_{lat_down}_{lon_left}_{lon_right}.npy',prediction)
    
    # 计算真实值和预测值
    true_values = ship_grid[mask_test]
    predicted_values = prediction[mask_test]
    
    # 过滤掉误差超过 500 米的点
    errors = np.abs(predicted_values - true_values)
    mask_valid = errors <= 500
    
    # 只保存误差小于等于 500 米的点
    ggm = np.column_stack((true_values[mask_valid], predicted_values[mask_valid]))
    np.savetxt(f'ture_pre_ggm_{lat_up}_{lat_down}_{lon_left}_{lon_right}.npy',ggm)
    
    print(f'Total points: {len(true_values)}')
    print(f'Valid points (error <= 500m): {np.sum(mask_valid)}')
    print(f'Invalid points (error > 500m): {len(true_values) - np.sum(mask_valid)}')
    




