import numpy as np
import torch
from torch import optim
import os
import sys

sys.path.append(os.getcwd())
from utils.dataloader import get_data_loader, train_val_loader
from networks.cnn import CNN, CNN_pool, CNN_var
from networks.tests import VisionTransformer


def retransform(params):
    result = np.zeros(shape=params.shape)
    result[:, 0] = np.exp(params[:, 0])
    result[:, 1] = (2 * np.exp(params[:, 1])) / (1 + np.exp(params[:, 1]))
    return result


def retransform2(params):
    result = np.zeros(shape=params.shape)
    #result[:, 0] = -np.log(params[:,0])
    result[:, 0] = np.exp(params[:, 0])
    result[:, 1] = params[:, 1] * 2
    return result


def run_experiment(
    n: int,
    model: str,
    exp: str,
    epochs: int,
    batch_size: int,
    device,
    transform,
    learning_rate: float = 0.001,
    n_val: int = 1000,
):
    # Set path
    path = f"data/{exp}/data/"
    # Get dataloaders
    train_dataloader, _ = get_data_loader(
        data_path=path, model=model, batch_size=batch_size, var="train"
    )
    val_dataloader, _ = get_data_loader(
        data_path=path, model=model, batch_size=n_val, var="val"
    )    
    # Define model
    net = CNN_pool()
    net.to(device)

    # Specify parameters and functions
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Create output array
    mse_results = np.zeros(shape=(n, 2))
    mae_results = np.zeros(shape=(n, 2))

    # Run experiment
    for i in range(n):
        for epoch in range(epochs):
            for sample in train_dataloader:
                img, param = sample
                img = img.to(device)
                param = param.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()
                net.train()

                # forward + backward + optimize
                outputs = net(img)
                loss = criterion(outputs, param)
                loss.backward()
                optimizer.step()

        # Calculate val loss
        for sample in val_dataloader:
            img, param = sample
            break
        img = img.to(device)
        param = param.to(device)
        net.eval()
        outputs = net(img)
        output_re = transform(outputs.cpu().detach().numpy())
        param_re = transform(param.cpu().detach().numpy())
        res = output_re - param_re
        mse_results[i] = np.sqrt(np.mean(np.square(res), axis=0))
        mae_results[i] = np.mean(np.abs(res), axis=0)

    # Calculate mean and Sd
    rmse_mean = np.mean(mse_results, axis=0)
    rmse_std = np.std(mse_results, axis=0)
    mae_mean = np.mean(mae_results, axis=0)
    mae_std = np.std(mae_results, axis=0)
    print(
        f"MSE for {model} \t Range : {rmse_mean[0]:.4f} ({rmse_std[0]:.4f}) \t Smoothness: {rmse_mean[1]:.4f} ({rmse_std[1]:.4f})"
    )
    print(
        f"MAE for {model} \t Range : {mae_mean[0]:.4f} ({mae_std[0]:.4f}) \t Smoothness: {mae_mean[1]:.4f} ({mae_std[1]:.4f})"
    )
    return(net)


def run_model(
    exp: str,
    epochs: int,
    batch_size: int,
    device,
    transform,
    learning_rate: float = 0.001,
    n_val: int = 400,
):
    # Set path
    path = f"data/{exp}/data/"
    # Get dataloaders
    train_dataloader, val_dataloader, _, _ = train_val_loader(data_path=path, batch_size=batch_size,
                                                              batch_size_val=n_val)
    # Define model
    net = CNN_pool()
    net.to(device)

    # Specify parameters and functions
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)


    # Run experiment
    for epoch in range(epochs):
        running_loss = 0
        for sample in train_dataloader:
            img, param,_ = sample
            img = img.to(device)
            param = param.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            net.train()

            # forward + backward + optimize
            outputs = net(img)
            loss = criterion(outputs, param)
            loss.backward()
            optimizer.step()
            running_loss += np.sqrt(loss.item())

        # Calculate val loss
        for sample in val_dataloader:
            img, param,_ = sample
            break
        img = img.to(device)
        param = param.to(device)
        net.eval()
        outputs = net(img)
        output_re = transform(outputs.cpu().detach().numpy())
        param_re = transform(param.cpu().detach().numpy())
        res = output_re - param_re        
        rmse_val = np.sqrt(np.mean(np.square(res), axis = 0))
        rmse_train = running_loss / len(train_dataloader)

        print(
            f"Epoch: {epoch} \t Training loss: {rmse_train:.4f} \t Validation loss: {rmse_val[0]:.4f} - {rmse_val[1]:.4f}"
        )
    return(net)


if __name__ == "__main__":
    # Set model
    models = ["brown", "schlather"]
    exp = "exp_3_1"
    epochs = 40
    batch_size = 32
    n = 10

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = "schlather"
    trained_net = run_model(exp, epochs, batch_size, device, retransform2, n_val = 1000)
    torch.save(trained_net.state_dict(), f"data/{exp}/checkpoints/cnn.pt")


