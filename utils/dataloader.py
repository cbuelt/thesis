import numpy as np
from torch.utils.data import Dataset, DataLoader



def get_data_loader(data_path, batch_size = 64, train = True):
    shuffle = True if train == True else False
    dataset = SpatialField(
        data_path=data_path,
        train = train
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataloader, dataset


class SpatialField(Dataset):
    def __init__(
        self,
        data_path : str,
        #model,
        train : bool = True,        
    ):
        self.data_path = data_path
        self.train = train
        if train:
            self.img_data = np.load(data_path+"train_data.npy")
            self.param_data = np.load(data_path+"train_params.npy")
        else:
            self.img_data = np.load(data_path+"test_data.npy")
            self.param_data = np.load(data_path+"test_params.npy")
        self.sample_size = len(self.param_data)


    def __len__(self):
        return self.sample_size

    def __getitem__(self, idx):
        img = self.img_data[:,:,idx]
        param = self.param_data[idx].astype("float32")

        #Transform

        #Expand dimension of image
        img = np.expand_dims(img, axis = 0).astype("float32")

        return img, param


if __name__ == '__main__':
    dataloader, dataset = get_data_loader(data_path = "data/data_test/")
    for i in dataloader:
        break
    img, param = i
    print(img.shape)
    print(param.shape)


