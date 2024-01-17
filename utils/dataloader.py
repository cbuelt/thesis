#
# This file includes the dataloaders for the neural network.
#

import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.utils import transform_parameters
import torchvision.transforms as T
from torchvision.transforms.functional import rotate
import random
import torch


def get_train_val_loader(
    data_path: str, model: str, batch_size: int = 64, batch_size_val: int = 64
):
    """Returns the training and validation dataloader.

    Args:
        data_path (str): Path to data folder.
        model (str): Max-stable model.
        batch_size (int, optional): Training batch size. Defaults to 64.
        batch_size_val (int, optional): Validation batch size. Defaults to 64.

    Returns:
        _type_: Training and validation dataloader and datasets.
    """
    train_dataset = SpatialField(data_path=data_path, model=model, var="train")
    val_dataset = SpatialField(data_path=data_path, model=model, var="val")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)
    return train_loader, val_loader, train_dataset, val_dataset


def get_test_loader(data_path: str, model: str, batch_size=750):
    """Returns the test data dataloader.

    Args:
        data_path (str): Path to data folder.
        model (str): Max-stable model.
        batch_size (int, optional): Testing batch size. Defaults to 750.

    Returns:
        _type_: Test data dataloader and dataset.
    """
    test_dataset = SpatialField(data_path=data_path, model=model, var="test")
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader, test_dataset


class SpatialField(Dataset):
    """Dataset class for max-stable field with parameters"""

    def __init__(
        self,
        data_path: str,
        model: str,
        var: str,
    ):
        """Initialize the dataset.

        Args:
            data_path (str): Path to data folder.
            model (str): Max-stable model.
            var (str): Variant of dataset, either training or validation/testing.
        """
        self.data_path = data_path
        self.var = var
        self.model = model
        self.img_data = np.load(
            self.data_path + self.model + "_" + self.var + "_data.npy"
        )
        self.sample_size = self.img_data.shape[0]
        try:
            self.param_data = np.load(
                self.data_path + self.model + "_" + self.var + "_params.npy"
            )
        except:
            self.param_data = np.ones(shape=(self.sample_size, 2))

    def __len__(self) -> int:
        """Get length of the dataset.

        Returns:
            int: Length.
        """
        return self.sample_size

    def __getitem__(self, idx: int):
        """Get item from the dataset.

        Args:
            idx (int): Index of the data.

        Returns:
            _type_: Image and parameters of the sample.
        """
        img = self.img_data[idx, :, :]
        param = self.param_data[idx].astype("float32")

        # Transform
        img_mean = img.mean()
        img_std = img.std()
        img = (img - img_mean) / img_std
        param = transform_parameters(param)

        # Expand dimension of image
        img = np.expand_dims(img, axis=0).astype("float32")
        # img = img.astype("float32")

        if self.var == "train":
            # Rotation of image
            img = torch.from_numpy(np.swapaxes(img, 0, 2))
            angle = random.choice([0, 0, 180])
            img = rotate(torch.swapaxes(img, 0, 2), angle=angle)
            # Vertical and horizontal flip
            hflipper = T.RandomHorizontalFlip(p=0.2)
            vflipper = T.RandomVerticalFlip(p=0.2)
            img = hflipper(img)
            img = vflipper(img)
        return img, param


if __name__ == "__main__":
    exp = "exp_4"
    data_path = f"data/{exp}/data/"
    train_loader, val_loader, train_dataset, val_dataset = get_train_val_loader(
        data_path=data_path, model="brown"
    )
    for sample in val_loader:
        img, param = sample
        break
    print(img.shape)
    print(val_dataset.__len__())
