#
# This file includes general utility functions.
#

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
    result[:, 1] = params[:, 1] * 2
    return result
