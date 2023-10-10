import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
import pyreadr
import itertools
import sys
import xarray as xr
import multiprocessing as mp
from typing import Mapping

def get_mse(a: float, b:float, sd: bool = False) -> float:
    """Returns MSE across first axis of two arrays

    Args:
        a (float): Array one
        b (float): Array two
        sd (bool): Indicator whether to return standard deviation

    Returns:
        float: MSE
    """
    if sd == False:
        return np.mean(np.power(a-b,2), axis = 0)
    else:
        return np.mean(np.power(a-b,2), axis = 0), np.std(np.power(a-b,2), axis = 0)


def corr_func(h: float, model: str, r: float, s: float) -> float:
    """Calculates the correlation function (powexp, whitmat) or the variogram (brown) depending on the model parameters
    and distance h.

    Args:
        h (float): Distance or array of distances
        model (str): String describing the underlying model
        r (float): Range parameter
        s (float): _Smoothness parameter
    Returns:
        float: Returns value of correlation function / variogram
    """
    if model == "brown":
        res = np.power((h / r), s)
    elif model == "powexp":
        res = np.exp(-np.power((h / r), s))
    elif model == "whitmat":
        res = (
            np.power(2, (1 - s))
            / sc.special.gamma(s)
            * np.power((h / r), 2)
            * sc.special.kv(s, (h / r))
        )
    return res


def extremal_coefficient(h: float, model: str, r: float, s: float) -> float:
    """Calculates the extremal coefficient depending on the model, model parametersand distance h.

    Args:
        h (float): Distance or array of distances
        model (str): String describing the underlying model
        r (float): Range parameter
        s (float): _Smoothness parameter
    Returns:
        float: Returns value of the extremal coefficient
    """
    if model == "brown":
        res = 2 * sc.special.ndtr(
            np.sqrt(corr_func(h, model, r, s)) / 2
        )
    else:
        res = 1 + np.sqrt(1 - corr_func(h, model, r, s) / 2)
    return res

def sample_extremal_coefficient(h: float, model: str, r: float, s: float, mean: bool = True) -> float:
    sample_size = r.shape[0]
    if mean:
        return np.mean(np.array([extremal_coefficient(h, model, r[i], s[i]) for i in range(sample_size)]), axis = 0)
    else:
         return np.array([extremal_coefficient(h, model, r[i], s[i]) for i in range(sample_size)])


def error_function(
    h: float,
    model: str,
    true: Mapping[float, float],
    estimate: Mapping[float, float],
    method: str = "single",
) -> float:
    """Generates the squared error between two extremal coefficient functions evaluated at the distance h.

    Args:
        h (float): Distance or array of distances
        model (str): String describing the underlying model
        true (Mapping[float, float]): Mapping with the two parameters r,s of the true model
        estimate (Mapping[float, float]): Mapping with the parameters r,s of the estimated model
        method (str): Whether to calculate the error based on a single estimation or multiple samples

    Returns:
        float: Squared error evaluated at h
    """
    r_true = true[0]
    s_true = true[1]
    r_est = estimate[0]
    s_est = estimate[1]
    if method == "single":
        error = np.power(
            extremal_coefficient(h, model, r_true, s_true)
            - extremal_coefficient(h, model, r_est, s_est),
            2,
        )
    elif method == "sample":
        error = np.power(
            extremal_coefficient(h, model, r_true, s_true)
            - np.mean(extremal_coefficient(h, model, r_est, s_est)),
            2,
        )
    return error


def get_integrated_error(
    model: str,
    true: float,
    estimate: float,
    method: str = "single",
    max_length: float = 30,
    sd: bool = False,
) -> float:
    """_summary_

    Args:
        model (str): String describing the underlying model
        true (float): Array with the two parameters r,s of the true model
        estimate (float Array with the parameters r,s of the estimated model
        method (str): Whether to calculate the error based on a single estimation or multiple samples
        max_length (float, optional): Used to compute the max interval of integration. Defaults to 10.
        sd (bool): Indicator whether to return standard deviation

    Returns:
        float: Integrated error
    """
    n_samples = true.shape[0]
    integrate = lambda true, estimate: quad(error_function, 0, np.sqrt(np.power(max_length, 2)),args=(model, true, estimate, method))[0]
    error = np.array([integrate(true[i,:], estimate[i,:]) for i in range(n_samples)])
    if sd == False:
        return np.mean(error)
    else:
        return np.mean(error), np.std(error)


def get_interval_score(
    observations,
    alpha,
    q_left=None,
    q_right=None,
    full: bool = False,
    sd: bool = False
):
    sharpness = q_right - q_left
    calibration = (
        (
            np.clip(q_left - observations, a_min=0, a_max=None)
            + np.clip(observations - q_right, a_min=0, a_max=None)
        )
        * 2
        / alpha
    )
    total = sharpness + calibration
    if full:
        return total
    if sd == False:
        return np.mean(total, axis = 0)
    else:
        return np.mean(total, axis = 0), np.std(total, axis = 0)
    

def get_pointwise_is(h, model, r_true, s_true, r_est, s_est, alpha):
    #Get true function
    true = extremal_coefficient(h, model, r_true, s_true)
    #Get quantile functions
    quantiles = np.quantile(sample_extremal_coefficient(h, model, r_est, s_est, mean = False), q = [alpha/2, 1 - (alpha/2)])
    interval_score = get_interval_score(true, alpha = alpha, q_left = quantiles[0], q_right = quantiles[1], full = True)
    return interval_score


def is_wrapper(model, r_true, s_true, r_est, s_est, max_length, alpha):
    h = np.linspace(0, np.sqrt(np.power(max_length, 2)), 1000)
    res = get_pointwise_is(h, model, r_true, s_true, r_est, s_est, alpha)
    return res.mean()

def get_integrated_is(
    model: str,
    true: Mapping[float, float],
    estimate: Mapping[float, float],
    max_length: float = 30,
    alpha: float = 0.05,
    sd: bool = False,
    par: bool = True) -> float:

    # Number of samples
    n_samples = true.shape[0]

    if par:
        pool = mp.Pool(mp.cpu_count()-2)
        results = [pool.apply_async(is_wrapper, args = (model, true[i,0], true[i,1], estimate[i,0], estimate[i,1], max_length, alpha)) for i in range(n_samples)]
        pool.close()
        pool.join()
        error = np.array([r.get() for r in results])
    else:
        error = np.array([is_wrapper(model, true[i,0], true[i,1], estimate[i,0], estimate[i,1], max_length, alpha) for i in range(n_samples)])
    if sd == False:
        return np.mean(error)
    else:
        return np.mean(error), np.std(error)


def get_energy_score(y_true, y_pred, sd: bool = False):
    """
    Compute mean energy score from samples of the predictive distribution.

    Parameters
    ----------
    y_true : array, shape (n_examples, n_dim)
        True values.
    y_pred : array, shape (n_examples, n_dim, n_samples)
        Samples from predictive distribution.
    sd (bool): Indicator whether to return standard deviation

    Returns
    -------
    float or tuple of (float, array)
        Mean energy score. If return_single_scores is True also returns scores for single examples.

    """
    N = y_true.shape[0]
    M = y_pred.shape[2]

    es_12 = np.zeros(y_true.shape[0])
    es_22 = np.zeros(y_true.shape[0])

    for i in range(N):
        es_12[i] = np.sum(np.sqrt(np.sum(np.square((y_true[[i],:].T - y_pred[i,:,:])), axis=0)))
        es_22[i] = np.sum(np.sqrt(np.sum(np.square(np.expand_dims(y_pred[i,:,:], axis=2) - np.expand_dims(y_pred[i,:,:], axis=1)), axis=0)))
    
    scores = es_12/M - 0.5* 1/(M*M) * es_22
    if sd == False:
        return np.mean(scores)
    else:
        return np.mean(scores), np.std(scores)
