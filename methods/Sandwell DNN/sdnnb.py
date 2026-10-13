import torch
import numpy as np
from .sandwell_longwave import SandwellLongwave
from sklearn.preprocessing import StandardScaler


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SDNNB(torch.nn.Module):
    def __init__(self, in_channel, hidden_sizes=(128, 128)):
        super(SDNNB,self).__init__()
        self.in_channel = in_channel
        self.hidden_sizes = hidden_sizes
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                input_size = in_channel
            else:
                input_size = hidden_sizes[i - 1]
            self.__setattr__(f'f{i}', torch.nn.Linear(input_size, hidden_size))
        self.fo = torch.nn.Linear(hidden_sizes[-1], 1)

    def forward(self, input_tensor):
        """
        input_tensor: channel × height × width
        """
        featured_tensor = input_tensor.permute(1, 2, 0)
        for i in range(len(self.hidden_sizes)):
            fx = self.__getattr__(f'f{i}')(featured_tensor)
            featured_tensor = torch.relu(fx)
        return self.fo(featured_tensor).squeeze()


class SDNNBPlus:
    def __init__(self, in_channel, hidden_sizes=(128, 128, 64, 64), constant_long=30):
        self.model = SDNNB(in_channel, hidden_sizes).to(device)
        self.constant_long = constant_long
        self.long_topography = None
        self.short_topography = None

    def fit(self, ga_grid, vg_grid, et_grid, nh_grid, gis, ship_train, longrkm, longckm, nround=2000, lr=1e-3):
        # calculate the long topography
        self.long_topography = SandwellLongwave(self.constant_long).fit_transform_sparse_safety(ship_train, longrkm, longckm)
        # short topography
        short_topography = np.zeros_like(ship_train)
        mask_train = ship_train != 0
        short_topography[mask_train] = ship_train[mask_train] - self.long_topography[mask_train]
        # stand the short topography
        short_scalor = StandardScaler()
        scaled_short_topography = np.zeros_like(ship_train)
        scaled_short_topography[mask_train] = short_scalor.fit_transform(short_topography[mask_train].reshape(-1, 1))[:, 0]

        # stack all inputs
        inputs = np.stack([ga_grid, vg_grid, et_grid, nh_grid, gis[0], gis[1]], axis=0)
        inputs_tensor = torch.from_numpy(inputs).float().to(device)

        targets = torch.from_numpy(scaled_short_topography).float().to(device)
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
        pred_short_topography = self.model(inputs_tensor).to('cpu').detach().numpy()
        self.short_topography = short_scalor.inverse_transform(pred_short_topography)

    def predict(self, mask_target):
        return (self.long_topography + self.short_topography)[mask_target]

    def predict_matrix(self):
        return self.long_topography + self.short_topography



