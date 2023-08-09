import numpy as np
import pyreadr

def transform_parameters(params):
    """
    Transform parameters.
    First parameter is log transformed.
    Second parameter is transformed to [0,1] range.

    Args:
        params : Input parameters

    Returns:
        result: Transformed parameters
    """
    result = np.zeros(shape = params.shape).astype("float32")
    result[0] = np.log(params[0])
    result[0] = params[0]
    result[1] = params[1]/2
    return result

def retransform_parameters(params):
    """
    Retransform parameters.
    First parameter is transformed with exponential.
    Second parameter is transformed to [0,2] range.       

    Args:
        params : Input parameters

    Returns:
        result: Transformed parameters
    """
    result = np.zeros(shape=params.shape)
    result[:, 0] = np.exp(params[:, 0])
    result[:,0] = params[:,0]
    result[:, 1] = params[:, 1] * 2
    return result

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
