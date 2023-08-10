import numpy as np
import torch
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader
import os
import sys
sys.path.append(os.getcwd())
from utils.dataloader import  get_train_val_loader, get_test_loader
from networks.cnn import CNN_pool, CNN_var, CNN_test
from utils.network import Scheduler
from utils.utils import retransform_parameters
from utils.losses import IntervalScore, QuantileScore, NormalCRPS, TruncatedNormalCRPS



def train_model(
    exp: str,
    model: str,
    epochs: int,
    batch_size: int,
    device,
    learning_rate: float = 0.0007, 
    n_val: int = 500,
):
    # Set path
    path = f"data/{exp}/data/"
    # Get dataloaders
    train_dataloader, val_dataloader, _, _ = get_train_val_loader(
        data_path=path, model=model, batch_size=batch_size, batch_size_val=n_val
    )
    # Define model
    net = CNN_test()
    net.to(device)

    # Specify parameters and functions
    #criterion2 = torch.nn.MSELoss()
    criterion = IntervalScore(alpha = 0.1)
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
            img, param, _ = sample
            img = img.to(device)
            param = param.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            net.train()

            # forward + backward + optimize
            outputs = net(img)
            interval_loss = criterion(param, outputs[0], outputs[1])
            #mse_loss = criterion2(param[:,1:2], outputs[2], outputs[3])
            total_loss = interval_loss#sum([interval_loss, mse_loss])
            total_loss.backward()
            optimizer.step()
            train_loss += np.sqrt(total_loss.item())/len(train_dataloader)

        # Calculate val loss
        val_loss = 0
        for sample in val_dataloader:
            img, param, _ = sample
            img = img.to(device)
            param = param.to(device)
            net.eval()
            outputs = net(img)
            interval_loss = criterion(param, outputs[0], outputs[1])
            #mse_loss = criterion2(param[:,1:2], outputs[2], outputs[3])
            total_loss = interval_loss#sum([interval_loss, mse_loss])
            val_loss += np.sqrt(total_loss.item())/len(val_dataloader)
        print(
            f"Epoch: {epoch} \t Training loss: {train_loss:.4f} \t Validation loss: {val_loss:.4f}"
        )

        stop = scheduler(np.mean(val_loss), epoch, net)
        if stop:
            break
    return net

def predict(
        exp : str,
        model: str,
        net,
        save_train : bool = True,
        batch_size : int = 1000,
        batch_size_val: int = 1000,
        batch_size_test: int = 500,
        train_size: int = 5000,
        test_size: int = 500
):
    # Set path
    path = f"data/{exp}/data/"
    # Get dataloaders
    _ , _, train_dataset, val_dataset = get_train_val_loader(
        data_path=path, model=model, batch_size=batch_size, batch_size_val=batch_size_val
    )
    test_loader, _ = get_test_loader(data_path = path, model = model, batch_size = batch_size_test)
    train_loader = DataLoader(ConcatDataset([train_dataset, val_dataset]), batch_size = batch_size, shuffle = False)

    #Send model to device
    net.to(device)

    # Prepare arrays
    train_results = np.zeros(shape = (3, train_size, 2))
    test_results = np.zeros(shape = (3, test_size, 2))

    
    #Calculate training samples
    if save_train:
        for i, sample in enumerate(train_loader):
            img, param = sample
            img = img.to(device)
            param = param.to(device)
            net.eval()
            outputs = net(img)
            #output_re = retransform_parameters(outputs.cpu().detach().numpy())
            #param_re = retransform_parameters(param.cpu().detach().numpy())
            lower = retransform_parameters(outputs[0].cpu().detach().numpy())
            upper = retransform_parameters(outputs[1].cpu().detach().numpy())
            train_results[0, (i*batch_size):((i+1)*batch_size),:] = lower
            train_results[1, (i*batch_size):((i+1)*batch_size),:] = upper

        # Save results
        np.save(file = f"data/{exp}/results/{net.name}_{model}_train.npy", arr = train_results)
 
    #Calculate test samples
    for i, sample in enumerate(test_loader):
        img, param, m = sample
        img = img.to(device)
        param = param.to(device)
        net.eval()
        outputs = net(img)
        #output_re = retransform_parameters(outputs.cpu().detach().numpy())
        #param_re = retransform_parameters(param.cpu().detach().numpy())
        lower = retransform_parameters(outputs[0].cpu().detach().numpy())
        upper = retransform_parameters(outputs[1].cpu().detach().numpy())
        #mean = retransform_parameters(outputs[2].cpu().detach().numpy())
        test_results[0, (i*batch_size_test):((i+1)*batch_size_test),:] = lower
        test_results[1, (i*batch_size_test):((i+1)*batch_size_test),:] = upper
        test_results[2, (i*batch_size_test):((i+1)*batch_size_test),:] = torch.unsqueeze(m, dim = 1).cpu().detach().numpy()
    np.save(file = f"data/{exp}/results/{net.name}_{model}_test.npy", arr = test_results)
    print(f"Saved results for model {model} and network {net.name}")


if __name__ == "__main__":
    # Set model
    models = ["all"]#["brown", "powexp", "whitmat"]
    exp = "exp_6"
    epochs = 100
    batch_size = 64

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for model in models:
        trained_net = train_model(exp, model, epochs, batch_size, device, learning_rate = 0.0007)
        predict(exp, model, trained_net, save_train = False, test_size = 1500)
