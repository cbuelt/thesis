import numpy as np
import torch
from torch import optim
import os
import sys

sys.path.append(os.getcwd())
from utils.dataloader import get_data_loader, train_val_loader
from networks.cnn import CNN_pool, CNN_var, CNN_test
from networks.tests import VisionTransformer
from utils.network import Scheduler


def retransform(params):
    result = np.zeros(shape=params.shape)
    result[:, 0] = np.exp(params[:, 0])
    result[:, 1] = params[:, 1] * 2
    return result


def run_model(
    exp: str,
    model: str,
    epochs: int,
    batch_size: int,
    device,
    transform,
    learning_rate: float = 0.0007,
    n_val: int = 400,
):
    # Set path
    path = f"data/{exp}/data/"
    # Get dataloaders
    train_dataloader, val_dataloader, _, _ = train_val_loader(
        data_path=path, model=model, batch_size=batch_size, batch_size_val=n_val
    )
    # Define model
    net = CNN_pool(channels=1)
    net.to(device)

    # Specify parameters and functions
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Initialize Scheduler
    scheduler = Scheduler(
        path=f"data/{exp}/checkpoints/",
        name=f"{model}_cnn_pool",
        patience=5,
        min_delta=0,
    )
    # Run experiment
    for epoch in range(epochs):
        running_loss = 0
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
            running_loss += np.sqrt(loss.item())

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
        rmse_val = np.sqrt(np.mean(np.square(res), axis=0))
        rmse_train = running_loss / len(train_dataloader)

        print(
            f"Epoch: {epoch} \t Training loss: {rmse_train:.4f} \t Validation loss: {rmse_val[0]:.4f} - {rmse_val[1]:.4f}"
        )

        stop = scheduler(np.mean(rmse_val), epoch, net)
        if stop:
            break


if __name__ == "__main__":
    # Set model
    models = ["brown", "powexp"]
    exp = "exp_4"
    epochs = 100
    batch_size = 32

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for model in models:
        run_model(exp, model, epochs, batch_size, device, retransform, n_val=500)
