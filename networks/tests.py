from torch.nn import Conv2d, MaxPool2d, Linear, Flatten, Dropout, Module, ModuleList
import torch.nn.functional as F
import torch
from torch import optim
import scipy as sc
import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from utils.dataloader import get_data_loader, get_train_val_loader
from utils.network import Scheduler

def retransform(params):
    result = torch.zeros(size=params.shape)
    result[:, 0] = torch.exp(params[:, 0])
    result[:, 1] = params[:, 1] * 2
    return result

class CNN_test(Module):
    def __init__(self, dropout=0, channels=1):
        super().__init__()
        self.name = "cnn_interval"
        self.conv_input = Conv2d(
            in_channels=channels, out_channels=32, kernel_size=(3, 3), padding="same"
        )
        self.conv_64 = Conv2d(
            in_channels=32, out_channels=32, kernel_size=(3, 3), padding="same"
        )
        self.conv_128 = Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same"
        )
        self.conv_128_2 = Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same"
        )
        self.conv_256 = Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same"
        )        
        self.conv_256_2 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"
        )
        self.pool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.flatten = Flatten()
        #self.linear_1 = Linear(in_features=1152, out_features=16)
        self.linear_1 = Linear(in_features = 128, out_features = 16)
        self.linear_2 = Linear(in_features=16, out_features=32)
        self.output = Linear(in_features=32, out_features=2)
        self.output_r_1 = Linear(in_features=32, out_features=1)
        self.output_r_2 = Linear(in_features=32, out_features=1)
        self.output_s_1 = Linear(in_features=32, out_features=1)
        self.output_s_2 = Linear(in_features=32, out_features=1)
        self.output_m_1 = Linear(in_features=32, out_features=1)
        self.output_m_2 = Linear(in_features=32, out_features=1)
        self.output_mu_1 = Linear(in_features=32, out_features=1)
        self.output_mu_2 = Linear(in_features=32, out_features=1)
        self.output_sigma_1 = Linear(in_features=32, out_features=1)
        self.output_sigma_2 = Linear(in_features=32, out_features=1)
        self.dropout = Dropout(p=dropout)

    def forward(self, x):
        # First convolutions
        x = F.relu(self.conv_input(x))
        x = F.relu(self.conv_64(x))
        x = self.pool(x)
        x_res_in = F.relu(self.conv_128(x))
        x = x_res_in+F.relu(self.conv_128_2(x_res_in))
        x = self.pool(x)
        x_res_in = F.relu(self.conv_256(x))
        x = F.relu(self.conv_256_2(x_res_in))
        x = self.pool(x)

        # Linear layers
        x = self.flatten(x)
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        x = F.relu(self.linear_2(x))
        x = self.dropout(x)
        #Interval output
        output_r_1 = self.output_r_1(x)
        output_r_2 = output_r_1 + F.relu(self.output_r_2(x))
        output_s_1 = F.sigmoid(self.output_s_1(x))
        output_s_2 = output_s_1 + F.sigmoid(self.output_s_2(x)) * (1-output_s_1)
        output_lower = torch.cat([output_r_1, output_s_1], dim = 1)
        output_upper = torch.cat([output_r_2, output_s_2], dim = 1)

        #Mean output
        output_m_1 = self.output_m_1(x)
        output_m_2 = F.sigmoid(self.output_m_2(x))
        output_mean = torch.cat([output_m_1, output_m_2], dim = 1)

        #CRPS normal output
        output_mu_1 = F.softplus(self.output_mu_1(x))
        output_mu_2 = F.sigmoid(self.output_mu_2(x))
        output_sigma_1 = F.softplus(self.output_sigma_1(x)) + 0.001
        output_sigma_2 = F.softplus(self.output_sigma_2(x)) + 0.001
        return output_lower, output_upper

class CNN_var(Module):
    def __init__(self, channels=1):
        super().__init__()
        self.conv_input = Conv2d(
            in_channels=channels, out_channels=64, kernel_size=(3, 3), padding="same"
        )
        self.conv_64 = Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same"
        )
        self.conv_128 = Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same"
        )
        self.conv_128_2 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"
        )
        self.conv_256 = Conv2d(
            in_channels=128, out_channels=256, kernel_size=(3, 3), padding="same"
        )
        self.conv_256_2 = Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), padding="same"
        )
        self.conv_1 = Conv2d(
            in_channels=256, out_channels=1024, kernel_size=(1, 1), padding="same"
        )
        self.pool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.flatten = Flatten()
        self.linear = Linear(in_features=1024, out_features=256)
        self.output_1 = Linear(in_features=256, out_features=1)
        self.output_2 = Linear(in_features=256, out_features=1)

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
        x = torch.amax(x, dim=(2, 3))
        x = self.flatten(x)
        x = F.relu(self.linear(x))
        output_1 = self.output_1(x)
        output_2 = F.sigmoid(self.output_2(x))
        output = torch.cat([output_1, output_2], dim=1)

        return output

class CNN_bernstein(Module):
    def __init__(self, device, dropout=0, channels=1, d = 12):
        super().__init__()
        #Define degree
        self.d = d
        self.device = device
        #Define grid for evaluation
        self.h = torch.linspace(0.0001,1,100)
        self.conv_input = Conv2d(
            in_channels=channels, out_channels=32, kernel_size=(3, 3), padding="same"
        )
        self.conv_64 = Conv2d(
            in_channels=32, out_channels=32, kernel_size=(3, 3), padding="same"
        )
        self.conv_128 = Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding="same"
        )
        self.conv_128_2 = Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), padding="same"
        )
        self.conv_256 = Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding="same"
        )        
        self.conv_256_2 = Conv2d(
            in_channels=128, out_channels=128, kernel_size=(3, 3), padding="same"
        )
        self.pool = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.flatten = Flatten()
        self.linear_1 = Linear(in_features=1152, out_features=32)
        self.linear_2 = Linear(in_features=32, out_features=64)
        self.output = Linear(in_features=64, out_features=d+1)
        self.dropout = Dropout(p=dropout)

    def forward(self, x):
        # First convolutions
        x = F.relu(self.conv_input(x))
        x = F.relu(self.conv_64(x))
        x = self.pool(x)
        x_res_in = F.relu(self.conv_128(x))
        x = x_res_in+F.relu(self.conv_128_2(x_res_in))
        x = self.pool(x)
        x_res_in = F.relu(self.conv_256(x))
        x = F.relu(self.conv_256_2(x_res_in))
        x = self.pool(x)

        # Linear layers
        x = self.flatten(x)
        x = F.relu(self.linear_1(x))
        x = self.dropout(x)
        x = F.relu(self.linear_2(x))
        x = self.dropout(x)
        output = F.relu(self.output(x))
        return output

    def bernstein_polynomial(self, h, i, d):
        res = sc.special.comb(d,i)*torch.pow(h,i)*torch.pow((1-h), (d-i))
        return res

    def transform_output(self, coef):
        res = torch.zeros(size = (self.d+1,len(self.h))).to(device)
        for i in range(self.d+1):
            res[i,:] = self.bernstein_polynomial(self.h, i, self.d)
        return torch.matmul(coef, res)
    
    def corr_func(self, h, method, r, s):
        if method=="brown":
            res = np.power((h/r),s)
        elif method=="powexp":
            res = np.exp(-np.power((h/r),s))        
        elif method == "whitmat":
            res = np.power(2, (1-s))/sc.special.gamma(s)*\
                np.power((h/r),2)*sc.special.kv(s, (h/r))
        return res    
    
    def extremal_coefficient(self, h_original, param, method):
        h = h_original*40
        r = param[0]
        s = param[1]
        if method=="brown":
            res = 2*sc.stats.norm.cdf(np.sqrt(self.corr_func(h, method, r, s))/2,loc = 0, scale = 1)    
        else:
            res = 1+np.sqrt(1-self.corr_func(h, method, r, s)/2)        
        return res    
    
    def increments(self, pred):
        res = torch.zeros(size = pred.shape).to(self.device)
        res[:,0] = pred[:,0]
        for i in range(1,self.d+1):
            res[:,i] = pred[:,i] + res[:,(i-1)]
        return res

    

if __name__ == "__main__":
    # Set model
    model = "whitmat"
    exp = "exp_4"
    epochs = 100
    batch_size = 32

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    learning_rate = 0.0007
    n_val = 500
    # Set path
    path = f"data/{exp}/data/"
    # Get dataloaders
    train_dataloader, val_dataloader, _, _ = get_train_val_loader(
        data_path=path, model=model, batch_size=batch_size, batch_size_val=n_val
    )
    # Define model
    net = CNN_bernstein(device, channels=1)
    net.to(device)

    # Specify parameters and functions
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Initialize Scheduler
    scheduler = Scheduler(
        path=f"",
        name=f"test",
        patience=5,
        min_delta=0,
    )

    # Run experiment
    for epoch in range(epochs):
        running_loss = 0
        for sample in train_dataloader:
            img, param = sample
            img = img.to(device)
            param = retransform(param).to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            net.train()

            # forward + backward + optimize
            outputs = net(img)
            # Implement basis functions
            #outputs = net.increments(outputs)
            transformed = net.transform_output(outputs)
            coeff = torch.Tensor(np.apply_along_axis(lambda x: net.extremal_coefficient(net.h.cpu(), x, model), arr = param.cpu().numpy(), axis = 1)).to(device)
            loss = criterion(transformed, coeff)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Calculate val loss
        for sample in val_dataloader:
            img, param = sample
            break
        img = img.to(device)
        param = retransform(param).to(device)
        net.eval()
        outputs = net(img)
        #outputs = net.increments(outputs)
        transformed = net.transform_output(outputs)
        coeff = torch.Tensor(np.apply_along_axis(lambda x: net.extremal_coefficient(net.h.cpu(), x, model), arr = param.cpu().numpy(), axis = 1)).to(device)
        loss = criterion(transformed, coeff)
        val_loss = loss.item()
        rmse_train = running_loss / len(train_dataloader)

        print(
            f"Epoch: {epoch} \t Training loss: {rmse_train:.4f} \t Validation loss: {val_loss:.4f}"
        )

        stop = scheduler(np.mean(val_loss), epoch, net)
        if stop:
            break



