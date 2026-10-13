import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self, in_channel, out_channel=128, kernel_size=3, hidden_sizes=(128, 128, 64, 64)):
        super(CNN, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.hidden_sizes = hidden_sizes

        self.conv2d = nn.Conv2d(in_channel, out_channel, kernel_size=self.kernel_size, padding=kernel_size//2)

        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                input_size = out_channel
            else:
                input_size = hidden_sizes[i - 1]
            self.__setattr__(f'f{i}', torch.nn.Linear(input_size, hidden_size))
        self.fo = torch.nn.Linear(hidden_sizes[-1], 1)

    def forward(self, inputs_tensor):
        featured_tensor = inputs_tensor.unsqueeze(dim=0)
        featured_input1 = self.conv2d(featured_tensor)
        featured_input2 = featured_input1.squeeze(dim=0).permute(1, 2, 0)
        for i, output_size in enumerate(self.hidden_sizes):
            xo = self.__getattr__(f'f{i}')(featured_input2)
            featured_input2 = torch.relu(xo)
        output = self.fo(featured_input2)
        return output.squeeze()