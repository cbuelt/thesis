from torch.nn import Conv2d, MaxPool2d, Linear, ModuleList, Flatten, Dropout, Module
import torch.nn.functional as F
import torch


class CNN(Module):
    def __init__(self, channels=1):
        super().__init__()
        self.conv_1 = Conv2d(
            in_channels=channels, out_channels=128, kernel_size=(3, 3), stride = 2, padding = 1
        )
        self.conv_2 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=(3, 3), stride = 2, padding = 1
        )
        self.conv_3 = Conv2d(
            in_channels=128, out_channels=16, kernel_size=(3, 3), stride = 2, padding = 1
        )
        self.flatten = Flatten()
        self.linear_1 = Linear(in_features = 256, out_features = 4)
        self.linear_2 = Linear(in_features = 4, out_features = 8)
        self.linear_3 = Linear(in_features = 8, out_features = 16)
        self.output = Linear(in_features = 16, out_features=2)
    

    def forward(self, x):
        # First convolutions
        x = F.relu(self.conv_1(x))
        x = F.relu(self.conv_2(x))
        x = F.relu(self.conv_3(x))

        # Linear layers
        x = self.flatten(x)
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = F.relu(self.linear_3(x))
        output = self.output(x)
        return output

if __name__ == '__main__':
    net = CNN()
    test = torch.rand(size=(1, 1, 25, 25))
    res = net(test)
    print(res.shape)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(pytorch_total_params)

