import numpy as np
import torch
from torch import optim
import os
import sys
sys.path.append(os.getcwd())
from utils.dataloader import get_data_loader
from networks.cnn import CNN, CNN_pool




if __name__ == '__main__':
    #Set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    #Define model
    net = CNN()
    net.to(device)
    #Specify parameters and functions
    epochs = 80
    batch_size = 64
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.005)


    #Get dataloaders
    dataloader, dataset = get_data_loader(data_path = "data/exp_1/data/", batch_size=batch_size)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, sample in enumerate(dataloader,):
            img, param = sample
            img = img.to(device)
            param = param.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(img)
            loss = criterion(outputs, param)
            loss.backward()
            optimizer.step()

            # Print epoch loss
            running_loss += loss.item()
        print(f'Epoch: {epoch + 1} loss: {running_loss / len(dataloader):.3f}')
        running_loss = 0.0

    print('Finished Training')
    
# Save model
torch.save(net.state_dict(), "data/exp_1/checkpoints/cnn_test.pt")
print("Model saved")
        


