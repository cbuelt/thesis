import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.utils import load_data, load_params

def get_data_loader(data_path, model, batch_size = 64, var = "train"):
    shuffle = False if var == "test" else True
    dataset = SpatialField(
        data_path=data_path,
        model = model,
        var = var
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloader, dataset


class SpatialField(Dataset):
    def __init__(
        self,
        data_path : str,
        model : str,
        var: str,        
    ):
        self.data_path = data_path
        self.model = model
        self.var = var
        self.img_data = load_data(data_path, model, var = var)
        self.param_data = load_params(data_path, model, var = var)

        self.sample_size = len(self.param_data)


    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        img = self.img_data[idx,:,:]
        param = self.param_data[idx].astype("float32")

        #Expand dimension of image
        img = np.expand_dims(img, axis = 0).astype("float32")

        return img, param


if __name__ == '__main__':
    exp = "exp_2"
    model = "brown"
    path = f"data/{exp}/data/"
    dataloader, dataset = get_data_loader(data_path = path, model = model, batch_size=64)


