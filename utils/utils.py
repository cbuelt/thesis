import numpy as np
import pyreadr

def load_data(path, model, var = "train"):
    data = pyreadr.read_r(path+model+"_"+var+"_data.RData")
    name = list(data.keys())[0]
    data = data[name].to_numpy()
    data = np.reshape(np.swapaxes(data, 0,1), newshape = (-1, 25, 25))
    return(data)

def load_params(path, model, var = "train"):
    data = pyreadr.read_r(path+model+"_"+var+"_params.RData")
    name = list(data.keys())[0]
    data = data[name].to_numpy()
    return(data)
