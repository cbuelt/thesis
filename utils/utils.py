import numpy as np
import pyreadr

def load_data(path, model, var = "train", transform = True):
    data = pyreadr.read_r(path+model+"_"+var+"_data.RData")
    name = list(data.keys())[0]
    data = data[name].to_numpy()
    data = np.reshape(np.swapaxes(data, 0,1), newshape = (-1, 25, 25))
    if transform:
        data = np.log(data)
    return(data)

def load_params(path, model, var = "train", transform = True):
    data = pyreadr.read_r(path+model+"_"+var+"_params.RData")
    name = list(data.keys())[0]
    data = data[name].to_numpy()
    if transform:
        data[:,0] = np.log(data[:,0])
        data[:,1] = np.log(data[:,1]/(2-data[:,1]))
    return(data)
