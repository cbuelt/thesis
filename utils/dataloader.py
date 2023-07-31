import numpy as np
from torch.utils.data import Dataset, DataLoader
#from utils.utils import load_data, load_params
import torchvision.transforms as T
from torchvision.transforms.functional import rotate
import random
import torch


def get_data_loader(data_path, model, batch_size = 64, var = "train"):
    """Dataloader for SpatialField

    Args:
        data_path (_type_): _description_
        model (_type_): _description_
        batch_size (int, optional): _description_. Defaults to 64.
        var (str, optional): _description_. Defaults to "train".

    Returns:
        _type_: _description_
    """
    shuffle = False if var == "test" else True
    dataset = SpatialField(
        data_path=data_path,
        model = model,
        var = var
    )    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloader, dataset


def train_val_loader(data_path, model, batch_size = 64, batch_size_val = 64):
    train_dataset = SpatialField(
        data_path=data_path,
        model = model,
        var = "train"
        
    )  
    val_dataset = SpatialField(
        data_path=data_path,
        model = model,
        var = "val"        
    ) 
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size_val, shuffle = False)
    return train_loader, val_loader, train_dataset, val_dataset

def test_loader(data_path, model, batch_size = 750):
    test_dataset = SpatialField(
        data_path = data_path,
        model = model,
        var = "test"
        
    )
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle=False)
    return test_loader, test_dataset


class CombinedSpatialField(Dataset):
    def __init__(
        self,
        data_path : str,
        var: str,
    ):
        self.data_path = data_path
        self.var = var
        self.img_data = np.load(self.data_path+self.var+"_data.npy")
        self.param_data = np.load(self.data_path+self.var+"_params.npy")
        self.sample_size = len(self.param_data)

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        img = self.img_data[idx,:,:]
        param = self.param_data[idx,0:2].astype("float32")
        model = self.param_data[idx,2]

        #Transform
        img = np.log(img)            
        param[0] = np.log(param[0])
        param[1] = param[1]/2

        #Expand dimension of image
        img = np.expand_dims(img, axis = 0).astype("float32")
        return img, param, model



class SpatialField(Dataset):
    """Dataset for train and test data split into two files.

    Args:
        Dataset (_type_): _description_
    """
    def __init__(
        self,
        data_path : str,
        model : str,
        var: str,
    ):
        self.data_path = data_path
        self.var = var
        self.model = model
        self.img_data = np.load(self.data_path+self.model+"_"+self.var+"_data.npy")
        self.param_data = np.load(self.data_path+self.model+"_"+self.var+"_params.npy")
        self.sample_size = self.param_data.shape[0]

    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        img = self.img_data[idx,:,:]
        param = self.param_data[idx].astype("float32")

        #Transform   
        img_mean = img.mean()
        img_std = img.std()
        img = (img - img_mean)/img_std            
        param[0] = np.log(param[0])
        param[1] = param[1]/2

        #Expand dimension of image
        img = np.expand_dims(img, axis = 0).astype("float32")
        #img = img.astype("float32")

        if self.var == "train":
            #Rotation of image
            img = torch.from_numpy(np.swapaxes(img, 0, 2))
            angle = random.choice([0, 180])
            img = rotate(torch.swapaxes(img, 0, 2) ,angle = angle)
            # Vertical and horizontal flip
            hflipper = T.RandomHorizontalFlip(p=0.3)
            vflipper = T.RandomVerticalFlip(p=0.3)
            img = hflipper(img)
            img = vflipper(img)
        return img, param


if __name__ == '__main__':
    exp = "exp_4"
    data_path = f"data/{exp}/data/"
    train_loader, val_loader, train_dataset, val_dataset = train_val_loader(data_path=data_path, model = "brown")
    for sample in val_loader:
        img, param = sample
        break
    print(img.shape)
    print(val_dataset.__len__())





