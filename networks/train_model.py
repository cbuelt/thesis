import numpy as np
import torch
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader
import os
import sys
sys.path.append(os.getcwd())
from utils.dataloader import  get_train_val_loader, get_test_loader
from networks.cnn import CNN, CNN_ES
from utils.network import Scheduler
from utils.utils import retransform_parameters
from utils.losses import EnergyScore



def train_model(
    exp: str,
    model: str,
    epochs: int,
    batch_size: int,
    device,
    type: str = "normal",
    learning_rate: float = 0.0007, 
    n_val: int = 500,
    dropout: float = 0.0,
    sample_dim: float = 100,
):
    # Set path
    path = f"data/{exp}/data/"
    # Get dataloaders
    train_dataloader, val_dataloader, _, _ = get_train_val_loader(
        data_path=path, model=model, batch_size=batch_size, batch_size_val=n_val
    )
    # Specify model
    if type == "normal":
        net = CNN(dropout = dropout)
        loss_function = torch.nn.MSELoss()
    elif type == "energy":
        net = CNN_ES(sample_dim = sample_dim)
        loss_function = EnergyScore()


    net.to(device)
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate)

    # Initialize Scheduler
    scheduler = Scheduler(
        path=f"data/{exp}/checkpoints/",
        name=f"{model}_{net.name}",
        patience=5,
        min_delta=0,
    )
    # Run experiment
    for epoch in range(1,epochs):
        train_loss = 0
        for sample in train_dataloader:
            img, param = sample
            img = img.to(device)
            param = param.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            net.train()

            # forward + backward + optimize
            outputs = net(img)
            param = torch.unsqueeze(param, -1)
            loss = loss_function(param, outputs)
            total_loss = loss
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item()/len(train_dataloader)

        # Calculate val loss
        val_loss = 0
        for sample in val_dataloader:
            img, param = sample
            img = img.to(device)
            param = param.to(device)
            net.eval()
            outputs = net(img)
            param = torch.unsqueeze(param, -1)
            loss = loss_function(param, outputs)
            total_loss = loss
            val_loss += total_loss.item()/len(val_dataloader)
        print(
            f"Epoch: {epoch} \t Training loss: {train_loss:.4f} \t Validation loss: {val_loss:.4f}"
        )

        stop = scheduler(np.mean(val_loss), epoch, net)
        if stop:
            break
    return net

def predict_test_data(
        exp : str,
        model: str,
        test_size: int,
        type: str = "normal",
        batch_size_test: int = 500,
        sample_dim: int = 500,
):
    # Set path
    path = f"data/{exp}/data/"
    test_loader, _ = get_test_loader(data_path = path, model = model, batch_size = batch_size_test)

    # Load model
    if type == "normal":
        net = CNN()
    elif type == "energy":
        net = CNN_ES(sample_dim = sample_dim)

    net.load_state_dict(torch.load(f"data/{exp}/checkpoints/{model}_{net.name}.pt"))

    #Send model to device
    net.to(device)

    # Prepare arrays
    if type == "normal":
        test_results = np.zeros(shape = (test_size, 2))
    elif type == "energy":
        test_results = np.zeros(shape = (test_size, 2, sample_dim))
 
    #Calculate test samples
    for i, sample in enumerate(test_loader):
        img, param = sample
        img = img.to(device)
        param = param.to(device)
        net.eval()
        outputs = net(img)
        # Retransform
        outputs = retransform_parameters(outputs.detach().cpu().numpy())
        if type == "normal":
            test_results[:, (i*batch_size_test):((i+1)*batch_size_test)] = np.squeeze(outputs)
        elif type == "energy":
            test_results[:, (i*batch_size_test):((i+1)*batch_size_test),:] = outputs

    np.save(file = f"data/{exp}/results/{model}_{net.name}.npy", arr = test_results)
    print(f"Saved results for model {model} and network {net.name}")


if __name__ == "__main__":
    # Set model
    models = ["brown"]
    exp = "application"
    types = ["normal", "energy"]
    epochs = 100
    batch_size = 100

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for model in models:
        for type in types:
            #trained_net = train_model(exp, model, epochs, batch_size, device, type = type)
            predict_test_data(exp, model, test_size = 6, type = type)
