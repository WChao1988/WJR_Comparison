import torch
from torch import nn
from math import fsum


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MultiScaleConvEfficient(nn.Module):
    """
    高效多尺度卷积模块
    使用深度可分离卷积减少参数量
    """

    def __init__(self, in_channel, out_channels=(5, 9, 64, 64),
                 kernel_sizes=(1, 3, 5, 7)):
        super().__init__()
        self.num_branches = len(kernel_sizes)
        self.branches = nn.ModuleList()

        for j, k in enumerate(kernel_sizes):
            padding = (k - 1) // 2 if k % 2 == 1 else k // 2

            # 深度可分离卷积：深度卷积 + 逐点卷积
            branch = nn.Sequential(
                # 深度卷积
                nn.Conv2d(in_channel, in_channel*k, kernel_size=k,
                          padding=padding, groups=in_channel, bias=False),
                nn.BatchNorm2d(in_channel*k),
                nn.ReLU(inplace=True),

                # 逐点卷积
                nn.Conv2d(in_channel*k, out_channels[j],
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels[j]),
                nn.ReLU(inplace=True)
            )
            self.branches.append(branch)

        self.out_channels = int(fsum(out_channels))

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        return torch.cat(outputs, dim=1)


class MCNN(torch.nn.Module):
    def __init__(self, in_channel=5, kernel_sizes=(1, 3, 5, 7), out_channels=(5, 9, 64, 64),
                 hidden_sizes=(32, 32, 32, 16, 16, 16)):
        super(MCNN, self).__init__()
        self.in_channel, self.out_channels = in_channel, kernel_sizes
        self.kernel_sizes = kernel_sizes
        self.hidden_sizes = hidden_sizes
        self.multi_conv1 = MultiScaleConvEfficient(in_channel, out_channels, kernel_sizes)
        input_size2 = int(fsum(out_channels))
        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                input_size = input_size2
            else:
                input_size = hidden_sizes[i - 1]
            self.__setattr__(f'f{i}', torch.nn.Linear(input_size, hidden_size))
        self.fo = torch.nn.Linear(hidden_sizes[-1], 1)

    def forward(self, inputs_tensor):
        """
        input_tensor: channel × height × width
        """
        featured_tensor = inputs_tensor.unsqueeze(dim=0)
        featured_input1 = self.multi_conv1(featured_tensor)
        featured_hidden = featured_input1.squeeze(dim=0).permute(1, 2, 0)
        for i, output_size in enumerate(self.hidden_sizes):
            xo = self.__getattr__(f'f{i}')(featured_hidden)
            featured_hidden = torch.relu(xo)
        output = self.fo(featured_hidden)
        return output.squeeze()