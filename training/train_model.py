import numpy as np
import torch
from torch import optim
import os
import sys
sys.path.append(os.getcwd())
from utils.dataloader import get_data_loader
from networks.cnn import CNN, CNN_pool

def retransform(params):
    result = np.zeros(shape = params.shape)
    result[:,0] = np.exp(params[:,0])
    result[:,1] = (2*np.exp(params[:,1]))/(1+np.exp(params[:,1]))
    return result   

def retransform2(params):
    result = np.zeros(shape = params.shape)
    result[:,0] = params[:,0]*(3-0.1)+0.1
    result[:,1] = params[:,1]*2
    return result   


if __name__ == '__main__':
    #Set model
    model = "schlather"
    exp = "exp_2"
    #Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #Define model
    net = CNN(dropout = 0.1)
    net.to(device)
    #Specify parameters and functions
    epochs = 32
    batch_size = 40
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)


    #Get dataloaders
    path = f"data/{exp}/data/"
    train_dataloader, train_dataset = get_data_loader(data_path = path, model = model, batch_size=batch_size, var = "train")
    val_dataloader, val_dataset = get_data_loader(data_path = path, model = model, batch_size=400, var = "val")

    for epoch in range(epochs):
        running_loss = 0.0
        for i, sample in enumerate(train_dataloader):
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

            # Print epoch loss
            running_loss += loss.item()     
        
        #Calculate val loss
        for i, sample in enumerate(val_dataloader):
            img, param = sample
            img = img.to(device)
            param = param.to(device)
            net.eval()
            outputs = net(img)
            output_re = retransform2(outputs.cpu().detach().numpy())
            param_re = retransform2(param.cpu().detach().numpy())
            param_re = torch.tensor(param_re)
            output_re = torch.tensor(output_re)
            val_loss = criterion(output_re, param_re)

        print(f'Epoch: {epoch + 1} loss: {running_loss / len(train_dataloader):.3f} \t Val-loss: {val_loss.item()}')
        running_loss = 0.0

    print('Finished Training')

#Checkout prediction


    
# Save model
checkpoint_path = f"data/{exp}/checkpoints/cnn_{model}.pt"
torch.save(net.state_dict(), checkpoint_path)
print("Model saved")
        


