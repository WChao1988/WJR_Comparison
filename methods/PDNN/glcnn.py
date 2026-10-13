from .cnn import CNN
from .mcnn import MCNN
from .gaussian_longwave import GaussianLongwave
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GLCNNPlus:
    def __init__(self, in_channel=5,
                 kernel_size=3,
                 kernel_sizes=(1, 3, 5, 7),
                 out_channel=128,
                 out_channels=(5, 9, 64, 64),
                 hidden_sizes=(128, 128, 64, 64), long_width=30, multi=False, longwave=False):
        self.long_width = long_width
        self.multi = multi
        self.longwave = longwave
        self.topography = None
        if multi:
            self.model = MCNN(in_channel, kernel_sizes, out_channels, hidden_sizes).to(device)
        else:
            self.model = CNN(in_channel, out_channel, kernel_size, hidden_sizes).to(device)

    def fit(self, ga_grid, vg_grid, et_grid, nh_grid, ship_train, gis=None, nround=2000, lr=1e-3):
        # calculate long topography if multi
        if self.longwave:
            long_topography = GaussianLongwave(self.long_width).fit_transform_sparse_safety(ship_train)
            scaled_long_topography = StandardScaler().fit_transform(long_topography.reshape(-1, 1)).reshape(
                ship_train.shape)
            inputs = np.stack([ga_grid, vg_grid, et_grid, nh_grid, scaled_long_topography], axis=0)
            inputs_tensor = torch.from_numpy(inputs).float().to(device)
        else:
            inputs = np.stack([ga_grid, vg_grid, et_grid, nh_grid, gis[0], gis[1]], axis=0)
            inputs_tensor = torch.from_numpy(inputs).float().to(device)
        # standardize topography
        mask_train = ship_train != 0
        ship_scalor = StandardScaler()
        scaled_topography = np.zeros_like(ship_train)
        scaled_topography[mask_train] = ship_scalor.fit_transform(ship_train[mask_train].reshape(-1, 1))[:, 0]

        targets = torch.from_numpy(scaled_topography).float().to(device)
        mask_targets = targets != 0
        # training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        for epoch in range(nround):
            optimizer.zero_grad()
            prediction = self.model(inputs_tensor)[mask_targets]
            loss = criterion(prediction, targets[mask_targets])
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                print(f'epoch = {epoch} / {nround}, loss = {loss.item():.4f}')

        pred_topography = self.model(inputs_tensor).to('cpu').detach().numpy()
        self.topography = ship_scalor.inverse_transform(pred_topography)

    def predict(self, mask_target):
        return self.topography[mask_target]

    def predict_matrix(self):
        return self.topography




