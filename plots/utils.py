import numpy as np
from evaluation.metrics import bivariate_density

def sample_ll(t, param_grid, model, data, distmat, cutoff = 5):
    r = param_grid[t,0]
    s = param_grid[t,1]    
    ll = np.log(bivariate_density(data, np.transpose(data), model, distmat, r, s))
    res = np.tril(ll, -1)
    res[distmat>=cutoff] = 0

    return np.sum(res), t    

def sample_ll_multiple(t, param_grid, model, data, distmat, cutoff = 5):
    r = param_grid[t,0]
    s = param_grid[t,1]    
    ll = np.log(bivariate_density(data, np.transpose(data), model,  np.expand_dims(distmat, axis = 1), r, s))
    res = np.tril(np.mean(ll, axis = 1), -1)
    res[distmat>=cutoff] = 0
    return np.sum(res), t  
