from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, Dropout, Module, ModuleList
import torch.nn.functional as F
import torch

  
class CNN_pool(Module):
    def __init__(self, dropout = 0, channels=1):
        super().__init__()
        self.conv_input = Conv2d(
            in_channels=channels, out_channels=32, kernel_size=(3, 3), padding = "same"
        )
        self.conv_64 = Conv2d(
            in_channels=32, out_channels=32, kernel_size=(3, 3), padding = "same"
        )
        self.conv_128 = Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding = "same"
        )
        self.conv_128_2 = Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), padding = "same"
        )
        self.conv_256 = Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding = "same"
        )
        self.conv_256_2 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=(3, 3), padding = "same"
        )
        self.pool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.flatten = Flatten()
        self.linear_1 = Linear(in_features = 1152, out_features = 16)
        self.linear_2 = Linear(in_features = 16, out_features = 32)
        self.output = Linear(in_features = 32, out_features = 2)
        self.output_1 = Linear(in_features = 32, out_features=1)
        self.output_2 = Linear(in_features = 32, out_features=1)
        self.dropout = Dropout(p = dropout)


    def forward(self, x):
        # First convolutions
        x = F.relu(self.conv_input(x))
        x = F.relu(self.conv_64(x))
        x = self.pool(x)
        x = F.relu(self.conv_128(x))
        x = F.relu(self.conv_128_2(x))
        x = self.pool(x)
        x = F.relu(self.conv_256(x))
        x = F.relu(self.conv_256_2(x))    
        x = self.pool(x)

        # Linear layers
        x = self.flatten(x)
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        x = F.relu(self.linear_2(x))
        x = self.dropout(x)
        output_1 = self.output_1(x)
        output_2 = F.sigmoid(self.output_2(x))
        output = torch.cat([output_1, output_2], dim = 1)
        return output
    

class CNN_test(Module):
    def __init__(self, dropout = 0, channels=1):
        super().__init__()
        self.conv_input = Conv2d(
            in_channels=channels, out_channels=64, kernel_size=(3, 3), padding = "same"
        )
        self.conv_64 = Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), padding = "same"
        )
        self.conv_128 = Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding = "same"
        )
        self.conv_128_2 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=(3, 3),   padding = "same"
        )
        self.conv_256 = Conv2d(
            in_channels=128, out_channels=256, kernel_size=(3, 3), padding = "same"
        )
        self.conv_256_2 = Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), padding = "same"
        )
        self.pool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.flatten = Flatten()
        self.linear_1 = Linear(in_features = 2304, out_features = 128)
        self.linear_2 = Linear(in_features = 128, out_features = 64)
        self.output = Linear(in_features = 64, out_features = 2)
        self.output_1 = Linear(in_features = 64, out_features=1)
        self.output_2 = Linear(in_features = 64, out_features=1)
        self.dropout = Dropout(p = dropout)


    def forward(self, x):
        # First convolutions
        x = F.relu(self.conv_input(x))
        x = F.relu(self.conv_64(x))
        x = self.pool(x)
        x = F.relu(self.conv_128(x))
        x = F.relu(self.conv_128_2(x))
        x = self.pool(x)
        x = F.relu(self.conv_256(x))
        x = F.relu(self.conv_256_2(x))    
        x = self.pool(x)

        # Linear layers
        x = self.flatten(x)
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        x = F.relu(self.linear_2(x))
        x = self.dropout(x)
        output_1 = self.output_1(x)
        output_2 = F.sigmoid(self.output_2(x))
        output = torch.cat([output_1, output_2], dim = 1)
        return output

    

class CNN_var(Module):
    def __init__(self, channels=1):
        super().__init__()
        self.conv_input = Conv2d(
            in_channels=channels, out_channels=64, kernel_size=(3, 3), padding = "same"
        )
        self.conv_64 = Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), padding = "same"
        )
        self.conv_128 = Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding = "same"
        )
        self.conv_128_2 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=(3, 3), padding = "same"
        )
        self.conv_256 = Conv2d(
            in_channels=128, out_channels=256, kernel_size=(3, 3), padding = "same"
        )
        self.conv_256_2 = Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), padding = "same"
        )
        self.conv_1 = Conv2d(
            in_channels = 256, out_channels = 1024, kernel_size=(1,1), padding = "same"
        )
        self.pool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.flatten = Flatten()
        self.linear = Linear(in_features = 1024, out_features = 256)
        self.output_1 = Linear(in_features = 256, out_features=1)
        self.output_2 = Linear(in_features = 256, out_features=1)


    
    def forward(self, x):
        # First convolutions
        x = F.relu(self.conv_input(x))
        x = F.relu(self.conv_64(x))
        x = self.pool(x)
        x = F.relu(self.conv_128(x))
        x = F.relu(self.conv_128_2(x))
        x = self.pool(x)
        x = F.relu(self.conv_256(x))
        x = F.relu(self.conv_256_2(x))     
        x = self.pool(x)
        x = F.relu(self.conv_1(x))
        x = torch.amax(x, dim = (2,3))
        x = self.flatten(x)
        x = F.relu(self.linear(x))
        output_1 = self.output_1(x)
        output_2 = F.sigmoid(self.output_2(x))
        output = torch.cat([output_1, output_2], dim = 1)

        return output






if __name__ == '__main__':
    net = CNN_test(channels=5)
    test = torch.rand(size=(1, 5, 25, 25))
    res = net(test)
    print(res.shape)

    pytorch_total_params = sum(p.numel() for p in net.parameters())
    print(pytorch_total_params)

